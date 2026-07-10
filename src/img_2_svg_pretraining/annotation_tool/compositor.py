"""Server-side canvas rendering (spec sections 3 and 8).

Streamlit has no live DOM canvas: every rerun we composite the base diagram
with the current mask + point layers into one PIL image and display it via
`streamlit-image-coordinates`. Points and masks are ALWAYS drawn together on
the one canvas -- never separate tabs/modes.

Zoom is crop-based (no freeform pan): when an instance is active the
composite is cropped to a fixed-padding region around its points, then
scaled by the zoom factor. `ViewTransform` maps a click on the displayed
image back to full-image pixel coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .datamodel import Instance, rle_to_mask

CROP_PADDING = 150          # px around the active instance's points
POINT_RADIUS = 6            # px, at 100% zoom

# Colors (RGBA)
_ACTIVE_MASK = (66, 133, 244, 110)      # blue fill for the active instance's mask
_ACTIVE_MASK_EDGE = (25, 80, 200, 255)
_ACCEPTED_MASK = (52, 168, 83, 60)      # faint green for already-accepted masks
_CONFIRMED_PT = (30, 160, 60, 255)      # green: human-added/-moved point
_FLAGGED_PT = (220, 40, 40, 255)        # red (dashed ring): unreviewed molmo point
_NEGATIVE_PT = (150, 40, 200, 255)      # purple: negative point
_SELECTED_RING = (255, 200, 0, 255)     # thick amber outline on selected point


@dataclass
class ViewTransform:
    """Displayed-image coords -> full-image coords."""
    crop_x0: int = 0
    crop_y0: int = 0
    scale: float = 1.0

    def to_image(self, disp_x: float, disp_y: float) -> tuple[float, float]:
        return self.crop_x0 + disp_x / self.scale, self.crop_y0 + disp_y / self.scale


def compute_crop(instance: Instance | None, img_w: int, img_h: int,
                 padding: int = CROP_PADDING) -> tuple[int, int, int, int]:
    """Crop box (x0, y0, x1, y1) centered on the active instance's points,
    clamped to image bounds. Full image when nothing to center on."""
    if instance is None or not instance.points:
        return 0, 0, img_w, img_h
    xs = [p.x for p in instance.points]
    ys = [p.y for p in instance.points]
    x0 = max(0, int(min(xs)) - padding)
    y0 = max(0, int(min(ys)) - padding)
    x1 = min(img_w, int(max(xs)) + padding)
    y1 = min(img_h, int(max(ys)) + padding)
    return x0, y0, x1, y1


def _draw_dashed_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                        r: float, color, width: int = 2, dashes: int = 12):
    for i in range(dashes):
        a0 = i * (360 / dashes)
        a1 = a0 + (360 / dashes) * 0.6
        draw.arc([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=color, width=width)


def _overlay_mask(base: Image.Image, mask: np.ndarray, fill, edge=None):
    if mask.shape != (base.height, base.width):
        # A mask that doesn't match the image (corrupt/foreign annotation
        # file) must not crash the whole review app -- skip drawing it.
        return
    color = Image.new("RGBA", base.size, fill)
    alpha = Image.fromarray((mask * fill[3]).astype(np.uint8), mode="L")
    color.putalpha(alpha)
    base.alpha_composite(color)
    if edge is not None:
        # 1px-ish boundary: mask XOR its erosion, drawn as edge color
        m = mask.astype(np.uint8)
        er = np.zeros_like(m)
        er[1:-1, 1:-1] = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                          & m[1:-1, :-2] & m[1:-1, 2:])
        boundary = (m & ~er).astype(bool)
        edge_img = Image.new("RGBA", base.size, edge)
        edge_img.putalpha(Image.fromarray((boundary * 255).astype(np.uint8), "L"))
        base.alpha_composite(edge_img)


def compose(
    image: Image.Image,
    instances: dict[str, Instance],
    active_instance_id: str | None,
    selected_point_id: str | None,
    zoom: float = 1.0,
    crop_to_active: bool = True,
) -> tuple[Image.Image, ViewTransform]:
    """Base diagram + mask layers + point layers -> one displayed image.

    Draws: accepted instances' masks (faint green), the active instance's
    mask (blue w/ edge), and the ACTIVE instance's points -- confirmed
    (green), unreviewed molmo (red dashed), negative (purple), selected
    (amber ring). Other instances' points are hidden to keep the canvas
    readable; the instance list panel is how you reach them.
    """
    base = image.convert("RGBA")
    active = instances.get(active_instance_id) if active_instance_id else None

    for inst in instances.values():
        if inst.state == "accepted" and inst.mask_rle and inst.id != active_instance_id:
            _overlay_mask(base, rle_to_mask(inst.mask_rle), _ACCEPTED_MASK)
    if active is not None and active.mask_rle:
        _overlay_mask(base, rle_to_mask(active.mask_rle), _ACTIVE_MASK,
                      edge=_ACTIVE_MASK_EDGE)

    draw = ImageDraw.Draw(base)
    if active is not None:
        for p in active.points:
            r = POINT_RADIUS
            if p.label == 0:
                color = _NEGATIVE_PT
            elif p.source == "molmo":
                color = _FLAGGED_PT
            else:
                color = _CONFIRMED_PT
            if p.id == selected_point_id:
                draw.ellipse([p.x - r - 4, p.y - r - 4, p.x + r + 4, p.y + r + 4],
                             outline=_SELECTED_RING, width=4)
            if p.source == "molmo" and p.label == 1:
                _draw_dashed_circle(draw, p.x, p.y, r, color)
                draw.ellipse([p.x - 2, p.y - 2, p.x + 2, p.y + 2], fill=color)
            else:
                draw.ellipse([p.x - r, p.y - r, p.x + r, p.y + r],
                             fill=color, outline=(255, 255, 255, 255), width=2)
            if p.label == 0:  # minus bar on negative points
                draw.line([p.x - r + 2, p.y, p.x + r - 2, p.y],
                          fill=(255, 255, 255, 255), width=2)

    # Crop/zoom
    tf = ViewTransform()
    if crop_to_active and active is not None and active.points:
        x0, y0, x1, y1 = compute_crop(active, image.width, image.height)
        base = base.crop((x0, y0, x1, y1))
        tf.crop_x0, tf.crop_y0 = x0, y0
    if zoom != 1.0:
        base = base.resize((int(base.width * zoom), int(base.height * zoom)),
                           Image.LANCZOS)
        tf.scale = zoom
    return base.convert("RGB"), tf


def instance_thumbnail(image: Image.Image, instance: Instance,
                       size: int = 72) -> Image.Image:
    """Small crop around the instance for the list panel rows."""
    x0, y0, x1, y1 = compute_crop(instance, image.width, image.height, padding=25)
    if x1 <= x0 or y1 <= y0:
        crop = image.convert("RGB")
    else:
        crop = image.convert("RGB").crop((x0, y0, x1, y1))
    crop.thumbnail((size, size))
    return crop
