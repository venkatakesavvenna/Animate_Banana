"""Step A: morphological fill over accepted annotation masks.

Reads every `accepted` instance with a non-null mask (see
`annotation_tool.store`/`datamodel`), applies a closing (fills small gaps
from partial annotation) followed by a *conditional* opening (strips small
noise specks, but only kept if a reasonably large mask survives -- otherwise
the closed-only mask is kept, per the user's instruction not to let opening
erase a legitimately small mask). Writes one PNG per instance plus a single
manifest.jsonl consumed by qa_contact_sheet.py and dataset.py.

Run: python -m img_2_svg_pretraining.data_engine.sam_finetuning.prepare_masks
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from img_2_svg_pretraining.annotation_tool import store
from img_2_svg_pretraining.annotation_tool.datamodel import Instance, ImageRecord, rle_to_mask

logger = logging.getLogger("sam_finetuning.prepare_masks")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

OUT_DIR = Path(__file__).resolve().parent / "masks_processed"


@dataclass
class FillResult:
    mask: np.ndarray
    opening_applied: bool


def iter_accepted_instances() -> Iterator[tuple[ImageRecord, Instance]]:
    """Yield (image_record, instance) for every accepted, masked instance."""
    for image_id in store.list_annotated_image_ids():
        loaded = store.load_image_record(image_id)
        if loaded is None:
            continue
        record, instances = loaded
        for iid in record.instances:
            inst = instances.get(iid)
            if inst is not None and inst.state == "accepted" and inst.mask_rle is not None:
                yield record, inst


def fill_mask(
    mask: np.ndarray,
    close_kernel: int = 5,
    open_kernel: int = 3,
    min_area_ratio: float = 0.5,
) -> FillResult:
    """Close (dilate->erode) to fill small gaps/holes from partial
    annotation, then conditionally open (erode->dilate) to strip small noise
    specks -- but only keep the opened result if it retains at least
    `min_area_ratio` of the closed mask's area, so opening never erases a
    legitimately small mask down to (near) nothing."""
    m = mask.astype(np.uint8)
    close_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_elem)

    open_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_elem)

    closed_area = int(closed.sum())
    opened_area = int(opened.sum())
    if closed_area > 0 and opened_area >= min_area_ratio * closed_area:
        return FillResult(mask=opened.astype(bool), opening_applied=True)
    return FillResult(mask=closed.astype(bool), opening_applied=False)


def process_all(
    close_kernel: int = 5,
    open_kernel: int = 3,
    min_area_ratio: float = 0.5,
    out_dir: Path = OUT_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    n_written = 0
    n_opened = 0
    with open(manifest_path, "w") as manifest_f:
        for record, inst in iter_accepted_instances():
            mask = rle_to_mask(inst.mask_rle)
            result = fill_mask(mask, close_kernel, open_kernel, min_area_ratio)

            image_out_dir = out_dir / record.id
            image_out_dir.mkdir(parents=True, exist_ok=True)
            mask_path = image_out_dir / f"{inst.id}.png"
            Image.fromarray((result.mask * 255).astype(np.uint8)).save(mask_path)

            row = {
                "instance_id": inst.id,
                "image_id": record.id,
                "file_path": record.file_path,
                "mask_path": str(mask_path.relative_to(out_dir)),
                "width": record.width,
                "height": record.height,
                "points": [
                    {"x": p.x, "y": p.y, "label": p.label} for p in inst.points
                ],
                "mask_source": inst.mask_source,
                "opening_applied": result.opening_applied,
            }
            manifest_f.write(json.dumps(row) + "\n")
            n_written += 1
            n_opened += int(result.opening_applied)

    logger.info(
        "Wrote %d masks (%d with opening applied) to %s",
        n_written, n_opened, manifest_path,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--min-area-ratio", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    process_all(
        close_kernel=args.close_kernel,
        open_kernel=args.open_kernel,
        min_area_ratio=args.min_area_ratio,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
