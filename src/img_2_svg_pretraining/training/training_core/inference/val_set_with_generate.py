"""
Validation script: run inference on the held-out validation split using
model.generate() (no teacher forcing), compute task metrics in parallel, and
write a JSON report.

Key design points
-----------------
* Uses the **same** seed and split_ratio as training so the val split is
  identical to what was held out during training.
* Loads VLMSam or VLMOnly automatically (detected from sam_version config).
* Metric computation (CPU-bound) runs in a ``ProcessPoolExecutor`` in
  parallel with the next GPU inference batch — no sequential wait.
* Outputs are stored as per-shard ``.pkl`` files written by a background
  thread, keeping GPU inference and I/O overlapped.
* Metrics are task-agnostic via ``MetricAccumulator.from_annotation_spec()``:
  layout → mAP, captioning → BLEU-4 + exact-match, unknown → no-op.
* A ``metrics_report.json`` and a human-readable table are written at the end.

Usage
-----
    python -m img_2_svg_pretraining.training.training_core.inference.val_set_with_generate \\
        --cfg /code/src/img_2_svg_pretraining/training/configs/encoder_swap/mn_molmo_7b_veclip_ft.yaml \\
        --devices cuda:0 cuda:1 ... \\
        [--num_metric_workers 4] \\
        [--save_visualizations]
"""
import json
import os
import copy
import pickle
import queue
import argparse
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

# Import datasets for auto-registration
from img_2_svg_pretraining.training.training_core.datasets.layout import doclaynet, indicdlp, m6doc, d4la, omni_docs, doc_bank, annopage, prima, comp_hr, promptable_v1  # noqa: F401
from img_2_svg_pretraining.training.training_core.datasets import pixmo as _pixmo  # noqa: F401
from img_2_svg_pretraining.training.training_core.datasets import public as _public  # noqa: F401

from img_2_svg_pretraining.training.training_core.builders.dataset_builder import build_single_dataset
from img_2_svg_pretraining.training.training_core.inference.factory import load_inference_model
from img_2_svg_pretraining.training.training_core.inference.utils import (
    ShardedDataset,
    decode_supervised_targets,
    move_to_device,
    prepare_generate_batch,
)
from img_2_svg_pretraining.training.training_core.validation.metric_accumulator import (
    MetricAccumulator,
    NoOpMetric,
    worker_update,
)

# SAM mask unpacking (only used when pred_masks are returned)
try:
    from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import unpack_mask_batch
    _HAS_SAM_UNPACK = True
except ImportError:
    _HAS_SAM_UNPACK = False


# ---------------------------------------------------------------------------
# Serialisation helpers (GPU tensors → picklable numpy)
# ---------------------------------------------------------------------------

def _masks_to_numpy(masks) -> Optional[List[np.ndarray]]:
    """Convert a list/tensor of masks to a list of (N, H, W) uint8 ndarrays."""
    if masks is None:
        return None
    result = []
    for m in masks:
        if isinstance(m, torch.Tensor):
            arr = m.detach().cpu().float().numpy()
            arr = (arr > 0).astype(np.uint8)
        elif isinstance(m, np.ndarray):
            arr = (m > 0).astype(np.uint8)
        else:
            arr = np.asarray(m)
        result.append(arr)
    return result


# ---------------------------------------------------------------------------
# Background PKL writer (non-blocking)
# ---------------------------------------------------------------------------

class _PklWriter:
    """Write batch results as .pkl files in a background thread."""

    def __init__(self, output_dir: str, shard_id: int):
        self._dir = output_dir
        self._shard = shard_id
        self._q: queue.Queue = queue.Queue()
        self._batch_count = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def enqueue(self, batch_result: dict):
        self._q.put(batch_result)
        self._batch_count += 1

    def stop_and_join(self):
        self._q.put(None)  # sentinel
        self._thread.join()

    def _loop(self):
        idx = 0
        while True:
            item = self._q.get()
            if item is None:
                break
            os.makedirs(self._dir, exist_ok=True)
            path = os.path.join(self._dir, f"shard{self._shard}_batch{idx:06d}.pkl")
            with open(path, "wb") as f:
                pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
            idx += 1


# ---------------------------------------------------------------------------
# Main inference + metric loop (per GPU worker)
# ---------------------------------------------------------------------------

def run_inference_and_metrics(
    model,
    data_module,
    data_args,
    val_dataset,
    dataset_name: str,
    rank: int,
    output_dir: str,
    num_metric_workers: int = 4,
    save_visualizations: bool = False,
) -> dict:
    """Infer on val_dataset and compute metrics concurrently.

    Args:
        model:              Loaded VLMSam or VLMOnly model (already on device).
        data_module:        DataModule for this dataset.
        data_args:          DataArguments (contains annotation_spec).
        val_dataset:        Validation dataset (already sharded for this rank).
        dataset_name:       Name for logging / output paths.
        rank:               GPU rank (for logging + output subdirs).
        output_dir:         Root directory for pkl outputs + report.
        num_metric_workers: Number of CPU worker processes for metric computation.
        save_visualizations: If True, also save per-image PNG visualisations.

    Returns:
        Final metric dict for this shard.
    """
    device = model.device
    tokenizer = data_module.processor.tokenizer
    collator = data_module.Collator
    annotation_spec = data_args.annotation_spec

    # Auto-select the right metric accumulator based on annotation spec
    accumulator_template: MetricAccumulator = MetricAccumulator.from_annotation_spec(annotation_spec)
    futures = []

    # Background pkl writer
    pkl_dir = os.path.join(output_dir, dataset_name, f"gpu_{rank}", "pkl")
    writer = _PklWriter(pkl_dir, shard_id=rank)

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )

    print(f"[GPU {rank}] {dataset_name}: {len(val_dataset)} samples | "
          f"metric: {type(accumulator_template).__name__}")

    with ProcessPoolExecutor(max_workers=num_metric_workers) as pool:
        for batch_idx, batch in enumerate(tqdm(
            val_loader,
            desc=f"[GPU {rank}] {dataset_name}",
            position=rank,
        )):
            labels = decode_supervised_targets(batch["labels"], tokenizer)
            batch = move_to_device(batch, device)
            generate_batch = prepare_generate_batch(batch, tokenizer=tokenizer)

            try:
                with torch.no_grad():
                    preds, pred_masks_raw = model.generate(**generate_batch)

                # Unpack GT masks (only possible if SAM batch keys exist)
                gt_masks_raw = None
                if _HAS_SAM_UNPACK and batch.get("masks") is not None:
                    gt_masks_raw = unpack_mask_batch(
                        batch["masks"],
                        mask_counts=batch.get("mask_counts"),
                        image_size_list=batch["orig_image_size_list"],
                    )

                # Build stable image IDs for AP computation
                image_ids = [
                    f"{dataset_name}_gpu{rank}_b{batch_idx}_s{si}"
                    for si in range(len(preds))
                ]

                # Convert to CPU-picklable format before submitting to pool
                batch_result = {
                    "preds": list(preds),
                    "labels": list(labels),
                    "pred_masks": _masks_to_numpy(pred_masks_raw),
                    "gt_masks": _masks_to_numpy(gt_masks_raw),
                    "image_ids": image_ids,
                }

                # Submit metric computation to worker pool (non-blocking).
                # copy.copy() gives a fresh empty accumulator of the right type
                # (same label_classes / parse_fn config but empty state).
                future = pool.submit(
                    worker_update,
                    copy.copy(accumulator_template),
                    batch_result,
                )
                futures.append(future)

                # Enqueue pkl write (background thread, non-blocking)
                writer.enqueue(batch_result)

            except Exception as exc:
                print(f"[WARN] GPU {rank} | {dataset_name} | batch {batch_idx}: {exc}")

        # Collect all metric futures and merge into one final accumulator
        final_accumulator: MetricAccumulator = copy.copy(accumulator_template)
        failed_futures = 0
        for future in as_completed(futures):
            try:
                partial = future.result()
                final_accumulator.merge(partial)
            except Exception as exc:
                failed_futures += 1
                print(f"[WARN] Metric worker failed: {exc}")

    writer.stop_and_join()

    if failed_futures:
        print(f"[WARN] {failed_futures} metric batches failed on GPU {rank}")

    return final_accumulator.compute()


# ---------------------------------------------------------------------------
# Per-GPU worker entry point
# ---------------------------------------------------------------------------

def worker(rank: int, cfg: DictConfig, devices: List[str], args):
    """Worker process for multi-GPU evaluation."""
    device = devices[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    torch.cuda.set_device(0)

    print(f"[Worker {rank}] Using device {device}")

    model_name_or_path = cfg.base_model
    output_dir = cfg.val_output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for dataset_cfg in cfg.datasets.val:
        dataset_name = dataset_cfg.name
        print(f"[Worker {rank}] Building dataset: {dataset_name}")

        # Build dataset using the same logic as training (identical split)
        bundle = build_single_dataset(
            dataset_spec=dataset_cfg,
            model_name_or_path=model_name_or_path,
            cfg=cfg,
            train=False,
        )
        data_args = bundle["data_args"]
        data_module = bundle["data_module"]
        val_dataset = bundle["val_dataset"]

        # Load model (VLMSam or VLMOnly depending on sam_version)
        model, _ = load_inference_model(
            checkpoint_path=cfg.val_checkpoint,
            model_name_or_path=model_name_or_path,
            sam_checkpoint=cfg.get("sam_checkpoint"),
            device="cuda:0",
            vlm_family=cfg.vlm_family,
            model_family_name=cfg.model_family_name,
            sam_version=cfg.sam_version,
            attn_implementation=cfg.get("attn_implementation"),
            data_module=data_module,
        )

        # Shard val set across GPUs
        val_dataset = ShardedDataset(val_dataset, shard_id=rank, num_shards=len(devices))

        metrics = run_inference_and_metrics(
            model=model,
            data_module=data_module,
            data_args=data_args,
            val_dataset=val_dataset,
            dataset_name=dataset_name,
            rank=rank,
            output_dir=output_dir,
            num_metric_workers=getattr(args, "num_metric_workers", 4),
            save_visualizations=getattr(args, "save_visualizations", False),
        )

        all_results[dataset_name] = metrics
        print(f"\n[GPU {rank}] {dataset_name} results: {metrics}")

    # Write per-shard report
    report_path = os.path.join(output_dir, f"metrics_report_gpu{rank}.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[GPU {rank}] Report saved to {report_path}")

    # Print summary table
    _print_summary_table(all_results, rank)


def _print_summary_table(results: dict, rank: int):
    """Print a compact metrics summary table to stdout."""
    print(f"\n{'='*60}")
    print(f"  GPU {rank} — Metric Summary")
    print(f"{'='*60}")
    for dataset, metrics in results.items():
        print(f"\n  Dataset: {dataset}")
        if not metrics:
            print("    (no metrics computed — task not yet instrumented)")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:20s}: {v:.4f}")
            else:
                print(f"    {k:20s}: {v}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run val-set inference with model.generate() + parallel metric computation"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="/code/src/img_2_svg_pretraining/training/configs/archive/run_pramana.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
        help="List of CUDA devices to use",
    )
    parser.add_argument(
        "--num_metric_workers",
        type=int,
        default=4,
        help="Number of CPU worker processes for parallel metric computation",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        default=False,
        help="Save per-image PNG visualizations (slower, more disk space)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    if len(args.devices) > 1:
        mp.spawn(
            worker,
            args=(cfg, args.devices, args),
            nprocs=len(args.devices),
            join=True,
        )
    else:
        worker(0, cfg, args.devices, args)


if __name__ == "__main__":
    main()
