"""Torch Dataset over prepare_masks.py's manifest, plus an image-level
train/val split.

Split is at the image level (not instance level) so masks from the same
image never leak across train/val -- diagrams often have several
near-duplicate node shapes, and splitting by instance would let the model
memorize a shape it saw a sibling of during training.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Sam3TrackerProcessor

_REPO_ROOT = Path(__file__).resolve().parents[4]
MASKS_DIR = Path(__file__).resolve().parent / "masks_processed"
SPLITS_PATH = Path(__file__).resolve().parent / "splits.json"

DEFAULT_VAL_IMAGES = 30
DEFAULT_SEED = 0


def load_manifest(manifest_path: Path) -> list[dict]:
    return [json.loads(line) for line in open(manifest_path)]


def make_split(
    manifest_path: Path = MASKS_DIR / "manifest.jsonl",
    val_images: int = DEFAULT_VAL_IMAGES,
    seed: int = DEFAULT_SEED,
    out_path: Path = SPLITS_PATH,
) -> dict[str, list[str]]:
    """Deterministic image-level train/val split, written to disk for
    reproducibility. Re-running with the same manifest/seed reproduces the
    same split."""
    rows = load_manifest(manifest_path)
    image_ids = sorted({r["image_id"] for r in rows})
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)

    val_ids = sorted(shuffled[:val_images])
    train_ids = sorted(shuffled[val_images:])
    split = {"train": train_ids, "val": val_ids}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    return split


def load_or_make_split(
    manifest_path: Path = MASKS_DIR / "manifest.jsonl",
    val_images: int = DEFAULT_VAL_IMAGES,
    seed: int = DEFAULT_SEED,
    split_path: Path = SPLITS_PATH,
) -> dict[str, list[str]]:
    if split_path.exists():
        return json.load(open(split_path))
    return make_split(manifest_path, val_images, seed, split_path)


class Sam3MaskDataset(Dataset):
    """One sample = one (image, point prompt, GT mask) instance.

    `__getitem__` returns raw PIL image + prompt/mask arrays; batching is
    left to `collate_fn` since `Sam3TrackerProcessor` expects one call per
    batch (needs the full list of images/points), not per-sample.
    """

    def __init__(self, manifest_path: Path, image_ids: set[str] | list[str],
                 masks_dir: Path = MASKS_DIR):
        self.masks_dir = masks_dir
        image_ids = set(image_ids)
        rows = load_manifest(manifest_path)
        self.rows = [r for r in rows if r["image_id"] in image_ids]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(_REPO_ROOT / row["file_path"]).convert("RGB")
        mask = np.array(Image.open(self.masks_dir / row["mask_path"])) > 0
        # One positive point per instance in this dataset (see datamodel.py /
        # annotation tool workflow) -- take the first point as the prompt.
        point = row["points"][0]
        return {
            "image": image,
            "point": (point["x"], point["y"]),
            "mask": mask.astype(np.float32),
            "instance_id": row["instance_id"],
        }


def make_collate_fn(processor: Sam3TrackerProcessor):
    """Runs the SAM3 processor over a whole batch at once (its own image
    preprocessing + point-coordinate rescaling), and stacks GT masks
    resized to the model's native mask-decoder output resolution (288x288,
    confirmed via a direct forward-pass probe) so the loss compares
    like-for-like without upsampling predictions."""
    PRED_MASK_SIZE = 288

    def collate_fn(batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        input_points = [[[list(b["point"])]] for b in batch]
        input_labels = [[[1]] for b in batch]

        inputs = processor(
            images=images, input_points=input_points, input_labels=input_labels,
            return_tensors="pt",
        )

        gt_masks = []
        for b in batch:
            m = Image.fromarray((b["mask"] * 255).astype(np.uint8))
            m = m.resize((PRED_MASK_SIZE, PRED_MASK_SIZE), Image.NEAREST)
            gt_masks.append(np.array(m) > 0)
        gt_masks = torch.from_numpy(np.stack(gt_masks).astype(np.float32))

        return {
            "inputs": inputs,
            "gt_masks": gt_masks,
            "instance_ids": [b["instance_id"] for b in batch],
        }

    return collate_fn
