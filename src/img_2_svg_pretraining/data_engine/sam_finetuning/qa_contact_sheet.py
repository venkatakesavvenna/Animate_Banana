"""Step B: random-20 visual QA contact sheet.

Samples 20 rows at random from prepare_masks.py's manifest and renders a
grid PNG: image crop / original SAM3 mask overlay / closed(+maybe opened)
mask overlay, side by side per row. Reuses `compositor._overlay_mask` for
the alpha-blended overlay + boundary drawing rather than reimplementing it.

Run: python -m img_2_svg_pretraining.data_engine.sam_finetuning.qa_contact_sheet
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from img_2_svg_pretraining.annotation_tool.compositor import _overlay_mask
from img_2_svg_pretraining.annotation_tool.datamodel import rle_to_mask
from img_2_svg_pretraining.annotation_tool import store

_REPO_ROOT = Path(__file__).resolve().parents[4]
MASKS_DIR = Path(__file__).resolve().parent / "masks_processed"
QA_DIR = Path(__file__).resolve().parent / "qa"

_ORIG_MASK_COLOR = (220, 40, 40, 110)     # red: original SAM3-accepted mask
_FILLED_MASK_COLOR = (66, 133, 244, 110)  # blue: closed(+opened) mask
_FILLED_MASK_EDGE = (25, 80, 200, 255)

_PANEL_SIZE = 220
_PADDING = 8
_LABEL_H = 18


def _load_manifest(manifest_path: Path) -> list[dict]:
    return [json.loads(line) for line in open(manifest_path)]


def _crop_around_mask(mask: np.ndarray, pad: int = 40) -> tuple[int, int, int, int]:
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


def _row_for_instance(row: dict) -> Image.Image:
    image_path = _REPO_ROOT / row["file_path"]
    image = Image.open(image_path).convert("RGB")

    # Original accepted mask is not in the manifest directly -- reload it
    # from the annotation JSON so the QA sheet shows what changed.
    loaded = store.load_image_record(row["image_id"])
    orig_mask = None
    if loaded is not None:
        _, instances = loaded
        inst = instances.get(row["instance_id"])
        if inst is not None and inst.mask_rle is not None:
            orig_mask = rle_to_mask(inst.mask_rle)

    filled_mask = np.array(Image.open(_REPO_ROOT
        / "src/img_2_svg_pretraining/data_engine/sam_finetuning"
        / "masks_processed" / row["mask_path"])) > 0

    crop_mask = filled_mask if orig_mask is None else (orig_mask | filled_mask)
    x0, y0, x1, y1 = _crop_around_mask(crop_mask)
    crop = image.crop((x0, y0, x1, y1))
    orig_crop = orig_mask[y0:y1, x0:x1] if orig_mask is not None else None
    filled_crop = filled_mask[y0:y1, x0:x1]

    panels = [
        _panel(crop, None),
        _panel(crop, orig_crop, _ORIG_MASK_COLOR),
        _panel(crop, filled_crop, _FILLED_MASK_COLOR, edge=_FILLED_MASK_EDGE),
    ]
    row_img = Image.new("RGB", (_PANEL_SIZE * 3 + _PADDING * 4,
                                 _PANEL_SIZE + _PADDING * 2 + _LABEL_H),
                         (30, 30, 30))
    for i, p in enumerate(panels):
        row_img.paste(p, (_PADDING + i * (_PANEL_SIZE + _PADDING), _PADDING))
    return row_img


def sample_and_render(manifest_path: Path, n: int = 20, seed: int | None = None,
                       out_path: Path | None = None) -> Path:
    rows = _load_manifest(manifest_path)
    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))

    row_imgs = [_row_for_instance(r) for r in sampled]
    row_h = row_imgs[0].height
    row_w = row_imgs[0].width
    sheet = Image.new("RGB", (row_w, row_h * len(row_imgs)), (30, 30, 30))
    for i, img in enumerate(row_imgs):
        sheet.paste(img, (0, i * row_h))

    if out_path is None:
        QA_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = QA_DIR / f"contact_sheet_{ts}.png"
    sheet.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=MASKS_DIR / "manifest.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = sample_and_render(args.manifest, args.n, args.seed, args.out)
    print(f"Wrote contact sheet: {out_path}")


if __name__ == "__main__":
    main()
