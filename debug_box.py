#!/usr/bin/env python3
"""Standalone debug runner for the Gemma4 raster-boxing stage
(data_engine/boxing/gemma4.py): loads a sample's cached raster points, runs
Gemma4BoxRunner.box_rasters on it, and draws the resulting boxes over the
source image so results are visible without digging through JSON.

MUST run from the dedicated `/environments/gemma4` venv, not the main
`img_2_svg_pretraining` venv -- the main venv's `transformers` drifted down
to 5.3.0 (via an unrelated streamlit install pulling a different resolved
version) and 5.3.0 predates `gemma4` architecture support entirely
(`AutoConfig.from_pretrained` raises `KeyError: 'gemma4'`). The dedicated
venv pins `transformers==5.13.0` (confirmed to include gemma4 support) on
top of `--system-site-packages` torch, same pattern as the `molmo_point`
venv used for the pointing stage.

    source /environments/gemma4/bin/activate && cd /code && \\
        python debug_box.py CVPR_2025_arch00033
    python debug_box.py CVPR_2025_arch00033 --gpu 3 --box-model gemma-4-31b

Requires the sample to already have cached raster points from a prior
`run_data_engine.py point` run (data_engine/cache/points/<pointing-model>/<id>.json).

Output (defaults to data_engine/debug/box/<box-model>/):
    <sample_id>.json   raw [{"object_id": ..., "box": [...]}] list
    <sample_id>.png     source image with boxes overlaid
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample_id", help="e.g. CVPR_2025_arch00033 (must have cached points already)")
    parser.add_argument("--data-root", type=str, default="/code/data/train")
    parser.add_argument("--image-subdir", type=str, default="original_images")
    parser.add_argument("--dataset", type=str, default=None,
                         help="cache namespace under cache/<dataset>/; defaults to basename of --data-root")
    parser.add_argument("--pointing-model", default="molmo2-8b",
                         help="which cache/<dataset>/points/<id>.json to read")
    parser.add_argument("--box-model", default="gemma-4-31b", help="benchmark.models registry key")
    parser.add_argument("--gpu", type=str, default="3", help="single physical GPU id to run on")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="defaults to data_engine/debug/box/<box-model>/")
    return parser.parse_args()


def _overlay_boxes(image, box_results):
    from PIL import ImageDraw, ImageFont

    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    for b in box_results:
        draw.rectangle(b.box, outline="deepskyblue", width=2)
        draw.text((b.box[0] + 2, b.box[1] + 2), f"r{b.object_id}", fill="deepskyblue", font=font)
    return marked


def main():
    args = _parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Deferred: CUDA_VISIBLE_DEVICES must be set before any torch/transformers import.
    from PIL import Image

    from img_2_svg_pretraining.benchmark.models import get_model
    from img_2_svg_pretraining.data_engine.boxing.gemma4 import Gemma4BoxRunner
    from img_2_svg_pretraining.data_engine.pointing.common import PointResult

    dataset = args.dataset or Path(args.data_root).name
    module_dir = Path(__file__).parent / "src" / "img_2_svg_pretraining" / "data_engine"
    output_dir = Path(args.output_dir) if args.output_dir else module_dir / "debug" / "box" / dataset / args.box_model
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = module_dir / "cache" / dataset / "points" / args.pointing_model / f"{args.sample_id}.json"
    if not points_path.exists():
        raise FileNotFoundError(
            f"No cached points for {args.sample_id} at {points_path} -- run "
            f"`run_data_engine.py point --pointing-model {args.pointing_model}` first."
        )
    data = json.loads(points_path.read_text())
    raster_points = [PointResult(**p) for p in data["rasters"]]
    print(f"{len(raster_points)} raster points loaded for {args.sample_id}")
    if not raster_points:
        print("No raster points for this sample -- nothing to box. Pick a different sample_id.")
        return

    image_path = Path(args.data_root) / args.image_subdir / f"{args.sample_id}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Loading box-grounding model ({args.box_model})...")
    t0 = time.time()
    runner = Gemma4BoxRunner(get_model(args.box_model))
    print(f"Model loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    boxes = runner.box_rasters(image_path, raster_points)
    print(f"{len(raster_points)} raster hints -> {len(boxes)} boxes ({time.time()-t0:.1f}s)")

    (output_dir / f"{args.sample_id}.json").write_text(json.dumps(
        [{"object_id": b.object_id, "box": list(b.box)} for b in boxes], indent=2,
    ))

    image = Image.open(image_path)
    overlay = _overlay_boxes(image, boxes)
    overlay.save(output_dir / f"{args.sample_id}.png")

    print(f"Done. Output at {output_dir / args.sample_id}.{{json,png}}")


if __name__ == "__main__":
    main()
