"""Test for the classical-CV node segmentation pipeline
(segmentation/classical_cv.py / segment_nodes.py), against a synthetic
figure built to match the characteristics the pipeline was specified
against: white background, a white-outline box style, a gray-filled box
style, text labels (including letters with enclosed counters -- "o", "e",
"D" -- the documented main failure mode), two boxes joined by a connector
line, and a node nested inside a container panel.

Run inside the docker container (needs opencv-python, numpy):
    python -m pytest src/img_2_svg_pretraining/data_engine/tests/test_cv_segmentation.py -v
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from img_2_svg_pretraining.data_engine.segmentation.classical_cv import (
    Point,
    binarize,
    load_normalize,
    reconcile,
    render_overlay,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# cv2.rectangle draws a 1px stroke centered on the nominal coordinate, plus
# binarize()'s 1px wall dilation eats one more pixel inward -- so the
# flood-filled interior is inset from the *drawn* (nominal) box by 2px on
# each side. Ground truth in this test is defined on those nominal drawing
# coordinates, so this constant converts to what flood-fill actually finds.
STROKE_INSET = 2


def _draw_box(img, x0, y0, w, h, style, label):
    x1, y1 = x0 + w, y0 + h
    if style == "outline":
        cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), 1)
    elif style == "gray":
        cv2.rectangle(img, (x0, y0), (x1, y1), (245, 245, 245), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), 1)
    cv2.putText(img, label, (x0 + 6, y0 + h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (20, 20, 20), 1, cv2.LINE_AA)


def _build_synthetic_figure() -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]], list[Point]]:
    """Builds a 500x400 3-channel BGR figure with:
    - n1: white-outline box, label with an 'o' and 'e' (letter counters)
    - n2: gray-filled box, label 'RoBERTa' (letter counters in 'o'/'e')
    - n3: white-outline box connected to n2 by a straight line (adjacent
      boxes joined by a connector -- must not merge)
    - n4: a node nested inside a larger outline container panel (tests the
      smallest-containing-box rule and multi-point-in-container handling)
    """
    h, w = 400, 500
    img = np.full((h, w, 3), 255, np.uint8)

    boxes = {}

    # n1: outline box, label has 'o'/'e' letter counters
    boxes["n1"] = (30, 30, 113, 43)
    _draw_box(img, *boxes["n1"], style="outline", label="BoolQ")

    # n2: gray-filled box, label 'RoBERTa' has letter counters
    boxes["n2"] = (30, 120, 162, 75)
    _draw_box(img, *boxes["n2"], style="gray", label="RoBERTa")

    # n3: outline box, connected to n2 by a line
    boxes["n3"] = (250, 130, 137, 42)
    _draw_box(img, *boxes["n3"], style="outline", label="ReCoRD")
    cv2.line(img, (30 + 162, 120 + 37), (250, 130 + 21), (30, 30, 30), 2)

    # container panel with a node inside it
    container = (300, 220, 170, 130)
    cv2.rectangle(img, (container[0], container[1]),
                  (container[0] + container[2], container[1] + container[3]), (40, 40, 40), 2)
    boxes["n4"] = (320, 260, 120, 50)
    _draw_box(img, *boxes["n4"], style="outline", label="Config")

    points = [
        Point(id="n1", x=boxes["n1"][0] + boxes["n1"][2] // 2, y=boxes["n1"][1] + boxes["n1"][3] // 2),
        # n2's point deliberately lands inside the 'o' of "RoBERTa" (a
        # letter counter) to exercise the documented main failure mode.
        Point(id="n2", x=boxes["n2"][0] + 12, y=boxes["n2"][1] + boxes["n2"][3] // 2),
        Point(id="n3", x=boxes["n3"][0] + boxes["n3"][2] // 2, y=boxes["n3"][1] + boxes["n3"][3] // 2),
        Point(id="n4", x=boxes["n4"][0] + boxes["n4"][2] // 2, y=boxes["n4"][1] + boxes["n4"][3] // 2),
    ]
    return img, boxes, points


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@pytest.fixture(scope="module")
def synthetic_figure():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    img, boxes, points = _build_synthetic_figure()
    path = FIXTURE_DIR / "synthetic_figure.png"
    cv2.imwrite(str(path), img)
    return path, img, boxes, points


def test_acceptance_criteria(synthetic_figure, tmp_path):
    path, img_bgr, gt_boxes, points = synthetic_figure

    t0 = time.time()
    gray = load_normalize(path)
    bw, walls, otsu_t = binarize(gray)
    result = reconcile(points, walls, bw, gray.shape)
    elapsed = time.time() - t0

    review = result.needs_review()
    assert len(review) == 0, f"expected zero needs_review, got {[n.id for n in review]}"

    for node in result.nodes:
        assert node.box is not None
        x, y, w, h = node.box
        assert w * h <= 0.25 * gray.shape[0] * gray.shape[1], f"{node.id} leaked (box too large)"
        # Ground truth here is the box as *drawn* (outer edge of a 2px
        # stroke); flood-fill correctly recovers the *interior* (inset by
        # ~stroke-width/2 + the 1px wall dilation), so allow that small,
        # constant inset rather than requiring pixel-exact IoU against the
        # nominal drawing coordinates -- a hand-drawn ground truth for a
        # real dataset would itself be annotated on the interior, not the
        # outer stroke pixel.
        gx, gy, gw, gh = gt_boxes[node.id]
        inset = STROKE_INSET
        shrunk_gt = (gx + inset, gy + inset, gw - 2 * inset, gh - 2 * inset)
        iou = _iou(node.box, shrunk_gt)
        assert iou >= 0.9, f"{node.id}: IoU {iou:.3f} < 0.9 vs ground truth {shrunk_gt}, got {node.box}"

    assert elapsed < 1.0, f"runtime {elapsed:.3f}s exceeds 1s/image budget"

    overlay = render_overlay(img_bgr, result)
    overlay_path = tmp_path / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    assert overlay_path.exists()
    assert overlay.shape == img_bgr.shape


def test_letter_counter_does_not_produce_garbage_region(synthetic_figure):
    """n2's seed point is placed inside the 'o' of 'RoBERTa' -- verifies the
    multi-seed-take-largest fix recovers the full node box, not a ~tens-of-px
    letter-counter region."""
    path, img_bgr, gt_boxes, points = synthetic_figure
    gray = load_normalize(path)
    bw, walls, _ = binarize(gray)
    result = reconcile(points, walls, bw, gray.shape)

    n2 = next(n for n in result.nodes if n.id == "n2")
    assert n2.source == "flood"
    _, _, w, h = n2.box
    assert w * h > 1000, "n2 resolved to a garbage region instead of the node interior"


def test_needs_review_when_point_outside_any_box(synthetic_figure):
    path, img_bgr, gt_boxes, points = synthetic_figure
    gray = load_normalize(path)
    bw, walls, _ = binarize(gray)

    stray_point = Point(id="stray", x=450, y=380)  # empty background area
    result = reconcile([stray_point], walls, bw, gray.shape)

    assert result.nodes[0].source == "needs_review"
