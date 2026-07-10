#!/usr/bin/env python3
"""Renders only the raster hint points (from cached pointing JSON) onto
their source images -- no re-inference, just re-reading
data_engine/cache/points/<pointing-model>/*.json and drawing the "rasters"
array over the corresponding image, same marker style as
run_data_engine.py::_overlay_points' raster group (deepskyblue circles,
"r<id>" labels).

Run anywhere (just needs PIL, no torch/transformers):
    python render_raster_points.py
    python render_raster_points.py --pointing-model molmo2-8b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pointing-model", default="molmo2-8b")
    parser.add_argument("--data-root", type=str, default="/code/data/train")
    parser.add_argument("--image-subdir", type=str, default="original_images")
    parser.add_argument("--dataset", type=str, default=None,
                        help="cache namespace under cache/<dataset>/; defaults to basename of --data-root")
    return parser.parse_args()


def _overlay_rasters(image, raster_points):
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    for p in raster_points:
        r = 8
        x, y = p["x"], p["y"]
        draw.ellipse((x - r, y - r, x + r, y + r), fill="deepskyblue", outline="black", width=2)
        draw.text((x + r + 2, y - r), f"r{p['object_id']}", fill="deepskyblue", font=font)
    return marked


def main():
    args = _parse_args()

    dataset = args.dataset or Path(args.data_root).name
    images_dir = Path(args.data_root) / args.image_subdir
    cache_root = ROOT / "src/img_2_svg_pretraining/data_engine/cache" / dataset
    points_dir = cache_root / "points" / args.pointing_model
    out_dir = cache_root / "raster_points" / args.pointing_model
    out_dir.mkdir(parents=True, exist_ok=True)

    point_files = sorted(points_dir.glob("*.json"))
    print(f"Found {len(point_files)} cached point files in {points_dir}")

    n_rendered, n_empty, n_skipped = 0, 0, 0
    for pf in point_files:
        sample_id = pf.stem
        image_path = images_dir / f"{sample_id}.png"
        if not image_path.exists():
            print(f"{sample_id}: SKIP (no image)")
            n_skipped += 1
            continue

        data = json.loads(pf.read_text())
        raster_points = data.get("rasters", [])
        if not raster_points:
            n_empty += 1
            continue

        image = Image.open(image_path)
        overlay = _overlay_rasters(image, raster_points)
        overlay.save(out_dir / f"{sample_id}.png")
        n_rendered += 1

    print(f"Done: {n_rendered} rendered, {n_empty} had 0 raster points, {n_skipped} skipped (no image).")
    print(f"Output at {out_dir}")


if __name__ == "__main__":
    main()
