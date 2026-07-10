#!/usr/bin/env python3
"""CLI for the classical-CV node segmentation pipeline (point-seeded
flood-fill bounded by dark box borders, cross-checked by contour detection).
See src/img_2_svg_pretraining/data_engine/segmentation/classical_cv.py for
the pipeline itself (including its current status vs. the SAM2-AMG
alternative in segmentation/sam2_amg.py) and README.md for the parameter
table.

    python segment_nodes.py --image clean.png --points points.json --out out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2

from img_2_svg_pretraining.data_engine.segmentation.classical_cv import (
    MAX_FRAC,
    MIN_AREA,
    RECT_EXTENT_MIN,
    SEED_STEP,
    SEED_WIN,
    binarize,
    load_normalize,
    load_points,
    reconcile,
    write_outputs,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="path to the diagram image")
    parser.add_argument("--points", required=True, help="path to points.json: [{id, x, y}, ...]")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--min-area", type=int, default=MIN_AREA)
    parser.add_argument("--seed-win", type=int, default=SEED_WIN)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    parser.add_argument("--max-frac", type=float, default=MAX_FRAC)
    parser.add_argument("--rect-extent-min", type=float, default=RECT_EXTENT_MIN)
    return parser.parse_args()


def main():
    args = _parse_args()

    gray = load_normalize(args.image)
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    points = load_points(args.points)

    for p in points:
        h, w = gray.shape
        if not (0 <= p.x < w and 0 <= p.y < h):
            raise ValueError(f"Point '{p.id}' at ({p.x}, {p.y}) is outside the image ({w}x{h}).")

    bw, walls, otsu_t = binarize(gray)

    result = reconcile(
        points, walls, bw, gray.shape,
        min_area=args.min_area, seed_win=args.seed_win, seed_step=args.seed_step,
        max_frac=args.max_frac, rect_extent_min=args.rect_extent_min,
    )
    result.otsu_threshold = otsu_t

    write_outputs(result, image_bgr, args.out)

    n_review = len(result.needs_review())
    print(f"{len(points)} points in / {len(result.nodes) - n_review} nodes out / {n_review} needs_review")
    for node in result.needs_review():
        print(f"  needs_review: {node.id}")


if __name__ == "__main__":
    main()
