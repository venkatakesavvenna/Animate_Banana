"""Validation: run a fine-tuned checkpoint over the held-out val split,
compute per-instance IoU against the filled GT mask, and render a visual
grid (raw crop / GT mask / predicted mask) for inspection.

Run:
    python -m img_2_svg_pretraining.data_engine.sam_finetuning.validate \
        --checkpoint checkpoints/full_run_v1/best --device cuda:5
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from img_2_svg_pretraining.annotation_tool.compositor import _overlay_mask
from img_2_svg_pretraining.data_engine.sam_finetuning.dataset import (
    MASKS_DIR, load_or_make_split, load_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
QA_DIR = Path(__file__).resolve().parent / "qa"

_GT_COLOR = (220, 40, 40, 110)      # red: ground-truth (filled) mask
_PRED_COLOR = (66, 133, 244, 110)   # blue: model prediction
_PRED_EDGE = (25, 80, 200, 255)

_PANEL_SIZE = 220
_PADDING = 8


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 1.0


@torch.no_grad()
def predict_mask(model, processor, image: Image.Image, point: tuple[float, float],
                 device: str) -> tuple[np.ndarray, float]:
    input_points = [[[[point[0], point[1]]]]]
    input_labels = [[[1]]]
    inputs = processor(images=image, input_points=input_points,
                       input_labels=input_labels, return_tensors="pt").to(device)
    outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
    )[0]
    mask = masks[0, 0].numpy().astype(bool)
    score = (float(outputs.iou_scores[0, 0, 0].cpu())
             if hasattr(outputs, "iou_scores") else 1.0)
    return mask, score


def _crop_around(mask: np.ndarray, pad: int = 40) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad)
    return x0, y0, x1, y1


def _panel(image: Image.Image, mask: np.ndarray | None, color=None, edge=None) -> Image.Image:
    base = image.convert("RGBA")
    if mask is not None:
        _overlay_mask(base, mask, color, edge=edge)
    return base.convert("RGB").resize((_PANEL_SIZE, _PANEL_SIZE), Image.LANCZOS)


def _row(image_path: Path, gt_mask: np.ndarray, pred_mask: np.ndarray,
         label: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    crop_box = _crop_around(gt_mask | pred_mask)
    x0, y0, x1, y1 = crop_box
    crop = image.crop((x0, y0, x1, y1))
    gt_crop = gt_mask[y0:y1, x0:x1]
    pred_crop = pred_mask[y0:y1, x0:x1]

    panels = [
        _panel(crop, None),
        _panel(crop, gt_crop, _GT_COLOR),
        _panel(crop, pred_crop, _PRED_COLOR, edge=_PRED_EDGE),
    ]
    label_h = 20
    row_img = Image.new("RGB", (_PANEL_SIZE * 3 + _PADDING * 4,
                                 _PANEL_SIZE + _PADDING * 2 + label_h),
                         (30, 30, 30))
    for i, p in enumerate(panels):
        row_img.paste(p, (_PADDING + i * (_PANEL_SIZE + _PADDING), _PADDING))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(row_img)
    draw.text((_PADDING, _PANEL_SIZE + _PADDING + 2), label, fill=(230, 230, 230))
    return row_img


def run_validation(
    checkpoint: Path,
    manifest_path: Path = MASKS_DIR / "manifest.jsonl",
    val_images: int = 30,
    split_seed: int = 0,
    device: str = "cuda",
    out_path: Path | None = None,
    n_visualize: int = 20,
    seed: int = 0,
) -> dict:
    model = Sam3TrackerModel.from_pretrained(str(checkpoint)).to(device).eval()
    processor = Sam3TrackerProcessor.from_pretrained(str(checkpoint))

    split = load_or_make_split(manifest_path, val_images, split_seed)
    val_ids = set(split["val"])
    rows = [r for r in load_manifest(manifest_path) if r["image_id"] in val_ids]

    results = []
    pred_masks_by_row: list[np.ndarray] = []
    for row in rows:
        image_path = _REPO_ROOT / row["file_path"]
        image = Image.open(image_path).convert("RGB")
        gt_mask = np.array(Image.open(
            Path(__file__).resolve().parent / "masks_processed" / row["mask_path"]
        )) > 0
        point = (row["points"][0]["x"], row["points"][0]["y"])
        pred_mask, score = predict_mask(model, processor, image, point, device)
        pred_masks_by_row.append(pred_mask)
        results.append({
            "instance_id": row["instance_id"],
            "image_id": row["image_id"],
            "iou": iou(pred_mask, gt_mask),
            "model_score": score,
        })

    ious = np.array([r["iou"] for r in results])
    summary = {
        "checkpoint": str(checkpoint),
        "n_instances": len(results),
        "n_images": len(val_ids),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "iou_ge_0.5": float((ious >= 0.5).mean()),
        "iou_ge_0.7": float((ious >= 0.7).mean()),
        "iou_ge_0.9": float((ious >= 0.9).mean()),
    }
    print(json.dumps(summary, indent=2))

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(rows), size=min(n_visualize, len(rows)), replace=False)

    row_imgs = []
    for i in sample_idx:
        row = rows[i]
        r = results[i]
        image_path = _REPO_ROOT / row["file_path"]
        gt_mask = np.array(Image.open(
            Path(__file__).resolve().parent / "masks_processed" / row["mask_path"]
        )) > 0
        pred_mask = pred_masks_by_row[i]
        label = f"{row['image_id']} / {row['instance_id']}  IoU={r['iou']:.2f}  score={r['model_score']:.2f}"
        row_imgs.append(_row(image_path, gt_mask, pred_mask, label))

    row_h = row_imgs[0].height
    row_w = row_imgs[0].width
    sheet = Image.new("RGB", (row_w, row_h * len(row_imgs)), (30, 30, 30))
    for i, img in enumerate(row_imgs):
        sheet.paste(img, (0, i * row_h))

    if out_path is None:
        QA_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = QA_DIR / f"validation_{ts}.png"
    sheet.save(out_path)
    print(f"Wrote validation grid: {out_path}")

    summary["visualization"] = str(out_path)
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps({"summary": summary, "per_instance": results}, indent=2))
    print(f"Wrote per-instance results: {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MASKS_DIR / "manifest.jsonl")
    parser.add_argument("--val-images", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-visualize", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_validation(
        checkpoint=args.checkpoint, manifest_path=args.manifest,
        val_images=args.val_images, split_seed=args.split_seed,
        device=args.device, out_path=args.out, n_visualize=args.n_visualize,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
