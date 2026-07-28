"""
Canonical coordinate normalization for img_2_svg_pretraining training datasets.

All datasets normalize their annotations to [0, 1] float space before
returning from get_source.  VLM-family data modules convert from this
canonical form to their own token representation.

Bounding boxes: (x1, y1, x2, y2) normalized to [0, 1] relative to image size.
Points: (x, y) normalized to [0, 1] relative to image size.
"""

from typing import Tuple


# ---------------------------------------------------------------------------
# Pixel → [0, 1] canonical space
# ---------------------------------------------------------------------------

def pixel_bbox_to_01(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    return (x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)


def pixel_point_to_01(x: float, y: float, img_w: int, img_h: int) -> Tuple[float, float]:
    return (x / img_w, y / img_h)


# ---------------------------------------------------------------------------
# Bbox format conversions (all return xyxy)
# ---------------------------------------------------------------------------

def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------------------------------------------------------------------------
# [0, 1] → VLM-family serialization (used by get_source for autoregressive mode)
# ---------------------------------------------------------------------------

def bbox_01_to_generic_tokens(x1: float, y1: float, x2: float, y2: float) -> str:
    """Generic text representation. <box>x1,y1,x2,y2</box> with 4-decimal precision."""
    return f"<box>{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}</box>"


def bbox_01_to_qwen_tokens(x1: float, y1: float, x2: float, y2: float) -> str:
    """Qwen2.5-VL bounding box token format."""
    # Qwen uses integer thousandths in [0, 1000]
    def _q(v: float) -> int:
        return min(max(int(round(v * 1000)), 0), 1000)
    return f"<|box_start|>({_q(x1)},{_q(y1)}),({_q(x2)},{_q(y2)})<|box_end|>"


def point_01_to_generic_tokens(x: float, y: float) -> str:
    """Generic point token format."""
    return f"<point>{x:.4f},{y:.4f}</point>"


def point_01_to_molmo_space(x: float, y: float) -> Tuple[float, float]:
    """Convert [0, 1] point to Molmo's [0, 100] coordinate space."""
    return (x * 100.0, y * 100.0)


def molmo_space_to_01(x: float, y: float) -> Tuple[float, float]:
    """Convert Molmo's [0, 100] point back to [0, 1] canonical space."""
    return (x / 100.0, y / 100.0)


# ---------------------------------------------------------------------------
# Clamp helpers
# ---------------------------------------------------------------------------

def clamp_bbox_01(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    return (
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    )


def clamp_point_01(x: float, y: float) -> Tuple[float, float]:
    return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
