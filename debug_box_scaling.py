#!/usr/bin/env python3
"""Re-draws already-cached Gemma4 raster boxes (data_engine/cache/boxes/) under
several coordinate-scaling hypotheses, to diagnose whether the model's "box"
values are raw pixel coordinates (as the prompt asks for and the parser
currently assumes) or actually normalized/rescaled coordinates that need
converting before they're usable.

Does NOT call the model -- purely reinterprets the cached JSON against the
source image at several candidate scales, side by side, so the right
interpretation can be picked by eye without spending more GPU time.

Run anywhere (just needs PIL, no torch/transformers):
    python debug_box_scaling.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
BOXES_DIR = ROOT / "src/img_2_svg_pretraining/data_engine/cache/boxes/molmo2-8b__gemma-4-31b"
IMAGES_DIR = ROOT / "data/partial_test_set/images"
OUT_DIR = ROOT / "src/img_2_svg_pretraining/data_engine/debug/box_scaling"

SAMPLE_IDS = [
    "arch_arxiv_ai_2025_000000116"
]

# Each hypothesis: (label, fn(x, y, w, h) -> (x', y') in pixel coords).
# raw_pixels: what the parser currently assumes (prompt asked for pixel
# coords directly). div1000_*: many VLMs emit 0-1000 normalized coords
# regardless of what's asked (Molmo's pointing format does this too --
# see pointing/molmo2.py's _extract_points). square_*_then_scale: some
# VLMs (Gemma's image processor resizes to a fixed square internally --
# confirmed image_seq_length=280/patch_size=16 -> ~16x16 patch grid, i.e.
# a 256px-equivalent square before further processing) may reason in that
# resized square's coordinate frame rather than the original image's.
HYPOTHESES = [
    ("raw_pixels", lambda x, y, w, h: (x, y)),
    ("div1000_x_w_h", lambda x, y, w, h: (x / 1000 * w, y / 1000 * h)),
]


def _draw_boxes(image: Image.Image, boxes: list[dict], scale_fn, w: int, h: int) -> Image.Image:
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    for entry in boxes:
        y0, x0, y1, x1 = entry["box"]
        sx0, sy0 = scale_fn(x0, y0, w, h)
        sx1, sy1 = scale_fn(x1, y1, w, h)
        draw.rectangle((sx0, sy0, sx1, sy1), outline="red", width=2)
        draw.text((sx0 + 2, sy0 + 2), f"r{entry['object_id']}", fill="red", font=font)
    return marked


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_id in SAMPLE_IDS:
        box_path = BOXES_DIR / f"{sample_id}.json"
        image_path = IMAGES_DIR / f"{sample_id}.png"
        if not box_path.exists():
            print(f"{sample_id}: SKIP (no cached boxes)")
            continue
        if not image_path.exists():
            print(f"{sample_id}: SKIP (no image)")
            continue

        data = json.loads(box_path.read_text())
        boxes = data.get("rasters", [])
        if not boxes:
            print(f"{sample_id}: SKIP (0 cached boxes)")
            continue

        image = Image.open(image_path)
        w, h = image.size
        print(f"{sample_id}: {w}x{h}, {len(boxes)} boxes")

        import pdb; pdb.set_trace()

        panels = []
        for label, scale_fn in HYPOTHESES:
            panel = _draw_boxes(image, boxes, scale_fn, w, h)
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, w, 18), fill="black")
            draw.text((3, 2), label, fill="white")
            panels.append(panel)

        # stack panels vertically with a thin separator, one combined PNG per sample
        sep = 4
        total_h = sum(p.height for p in panels) + sep * (len(panels) - 1)
        combined = Image.new("RGB", (w, total_h), "white")
        y_off = 0
        for panel in panels:
            combined.paste(panel, (0, y_off))
            y_off += panel.height + sep

        out_path = OUT_DIR / f"{sample_id}_scaling_compare.png"
        combined.save(out_path)
        print(f"  -> {out_path}")

    print(f"\nDone. Output at {OUT_DIR}")


if __name__ == "__main__":
    main()
