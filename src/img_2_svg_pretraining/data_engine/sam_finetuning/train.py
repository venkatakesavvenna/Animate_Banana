"""Step D: fine-tune SAM3's mask decoder on human-accepted masks.

Freezes `vision_encoder` (454M params) entirely; fully fine-tunes
`prompt_encoder` + `mask_decoder` + `shared_image_embedding` (~4.2M
trainable params, confirmed via direct inspection) -- LoRA/peft is not used
since a 4.2M-param module is already cheap to fully train (LoRA exists to
cheaply adapt large modules, which doesn't apply here), and `peft` isn't
installed.

Single-GPU training (dataset/trainable-param count is small enough that DDP
adds complexity without benefit -- see plan doc). Run with:

    CUDA_VISIBLE_DEVICES=5 python -m img_2_svg_pretraining.data_engine.sam_finetuning.train \
        --run-name my_run

Saves the final checkpoint as a full HF-style `Sam3TrackerModel` (frozen
vision-encoder weights included, so it's self-contained) via
`save_pretrained`, reloadable exactly like `Sam3Runner`/
`Sam3InteractiveBackend` already do -- no new loading code needed downstream.

Logs to Weights & Biases (project `sam3-mask-finetuning` by default) in
addition to the local `log.csv` -- the wandb run URL is printed at startup
and also written to `checkpoints/<run_name>/wandb_url.txt`. Requires
`WANDB_API_KEY` in the environment (this repo's `.env` at the repo root
carries it; source it before running, or pass `--no-wandb` to skip).
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from img_2_svg_pretraining.data_engine.sam_finetuning.dataset import (
    MASKS_DIR, Sam3MaskDataset, load_or_make_split, make_collate_fn,
)

logger = logging.getLogger("sam_finetuning.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_HF_REPO = "facebook/sam3"
DEFAULT_WANDB_PROJECT = "sam3-mask-finetuning"


def dice_loss(pred_logits: torch.Tensor, gt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """pred_logits, gt: [B, H, W]. gt is {0,1} float."""
    pred = torch.sigmoid(pred_logits)
    pred = pred.flatten(1)
    gt = gt.flatten(1)
    intersection = (pred * gt).sum(-1)
    union = pred.sum(-1) + gt.sum(-1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def mask_loss(pred_logits: torch.Tensor, gt: torch.Tensor,
              iou_pred: torch.Tensor | None = None,
              iou_loss_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt)
    dice = dice_loss(pred_logits, gt)
    total = bce + dice

    logs = {"bce": bce.item(), "dice": dice.item()}
    if iou_loss_weight > 0 and iou_pred is not None:
        with torch.no_grad():
            pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
            inter = (pred_mask * gt).flatten(1).sum(-1)
            union = ((pred_mask + gt) > 0).float().flatten(1).sum(-1)
            true_iou = inter / union.clamp(min=1.0)
        iou_loss = F.mse_loss(iou_pred, true_iou)
        total = total + iou_loss_weight * iou_loss
        logs["iou_loss"] = iou_loss.item()
    logs["total"] = total.item()
    return total, logs


def freeze_vision_encoder(model: Sam3TrackerModel) -> None:
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder"):
            param.requires_grad_(False)


def trainable_params(model: Sam3TrackerModel):
    return [p for p in model.parameters() if p.requires_grad]


def run_epoch(model, loader, device, optimizer=None, iou_loss_weight: float = 0.0,
              wandb_run=None, split: str = "train", global_step: int = 0):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        gt_masks = batch["gt_masks"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(**inputs, multimask_output=False)
            # pred_masks: [B, num_objects=1, num_masks_per_object=1, H, W]
            pred_logits = outputs.pred_masks[:, 0, 0]
            iou_pred = (outputs.iou_scores[:, 0, 0]
                        if hasattr(outputs, "iou_scores") else None)
            loss, logs = mask_loss(pred_logits, gt_masks, iou_pred, iou_loss_weight)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if wandb_run is not None:
            wandb_run.log(
                {f"{split}/{k}_step": v for k, v in logs.items()},
                step=global_step,
            )
            global_step += 1

        total_loss += logs["total"]
        n_batches += 1
    return total_loss / max(n_batches, 1), global_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--manifest", type=Path, default=MASKS_DIR / "manifest.jsonl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-images", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--iou-loss-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip W&B logging (local log.csv is always written).")
    args = parser.parse_args()

    out_dir = CHECKPOINTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.run_name,
            config=vars(args),
            dir=str(out_dir),
        )
        logger.info("W&B run: %s", wandb_run.url)
        (out_dir / "wandb_url.txt").write_text(wandb_run.url + "\n")

    split = load_or_make_split(args.manifest, args.val_images, args.split_seed)
    logger.info("train images=%d val images=%d", len(split["train"]), len(split["val"]))

    train_ds = Sam3MaskDataset(args.manifest, split["train"])
    val_ds = Sam3MaskDataset(args.manifest, split["val"])
    logger.info("train instances=%d val instances=%d", len(train_ds), len(val_ds))

    processor = Sam3TrackerProcessor.from_pretrained(args.hf_repo)
    collate_fn = make_collate_fn(processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    model = Sam3TrackerModel.from_pretrained(args.hf_repo).to(args.device)
    freeze_vision_encoder(model)
    n_trainable = sum(p.numel() for p in trainable_params(model))
    logger.info("trainable params: %.2fM", n_trainable / 1e6)

    optimizer = torch.optim.AdamW(trainable_params(model), lr=args.lr)

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "seconds"])

        best_val = float("inf")
        global_step = 0
        for epoch in range(args.epochs):
            t0 = time.perf_counter()
            train_loss, global_step = run_epoch(
                model, train_loader, args.device, optimizer, args.iou_loss_weight,
                wandb_run, "train", global_step)
            val_loss, global_step = run_epoch(
                model, val_loader, args.device, None, args.iou_loss_weight,
                wandb_run, "val", global_step)
            dt = time.perf_counter() - t0
            logger.info("epoch=%d train_loss=%.4f val_loss=%.4f (%.1fs)",
                       epoch, train_loss, val_loss, dt)
            writer.writerow([epoch, train_loss, val_loss, dt])
            f.flush()

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch, "train/loss_epoch": train_loss,
                    "val/loss_epoch": val_loss, "epoch_seconds": dt,
                }, step=global_step)

            if val_loss < best_val:
                best_val = val_loss
                model.save_pretrained(out_dir / "best")
                processor.save_pretrained(out_dir / "best")

    model.save_pretrained(out_dir / "final")
    processor.save_pretrained(out_dir / "final")
    logger.info("Saved final checkpoint to %s", out_dir / "final")

    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.finish()


if __name__ == "__main__":
    main()
