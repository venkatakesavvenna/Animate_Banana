"""Locate raster regions with the chat VLM's own bounding-box output.

This replaces the Molmo2 -> verify -> SAM3 -> map chain with a single call to
the model that is already in the pipeline. The saving is not just latency: the
old route needed two extra checkpoints resident on GPU under a second venv, so
a run could never have only one model loaded at a time. Here Stage 1b uses the
same backend as every other agent.

The collapse works because Stage 1a already tells us what to look for. It emits
placeholders carrying an `xml id` and the label text of what belongs there, so
the detector can be asked to find *those specific things* rather than to
propose regions blind and have a second pass decide which placeholder each one
belongs to. Detection and mapping become the same answer, which is also why the
mapping step disappears: the model returns the `xml id` it matched.

Coordinates come back in Gemini's convention -- `[ymin, xmin, ymax, xmax]`,
each value an integer on a 0-1000 grid, normalized against the image's own
dimensions regardless of aspect ratio. `to_pixels` is the only place that
convention is decoded; everything downstream sees ordinary pixel boxes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..backends import ChatBackend, Message
from ..extract import extract_json

# Gemini normalizes box coordinates onto this grid, not onto the image's real
# pixel dimensions.
GRID = 1000.0

DETECT_PROMPT = """You are locating embedded raster graphics in a scientific figure.

A vector reconstruction of this figure was generated, and it drew empty placeholder boxes
wherever content appeared that cannot be redrawn with lines and shapes -- photographs,
rendered 3D objects, screenshots, heatmaps, plots, natural images, icons and logos.

Your job is to find, in the ORIGINAL figure shown to you, the region that belongs in each
placeholder.

The placeholders are:
{placeholders}

For each placeholder you can confidently locate, return its bounding box in the original
image.

Rules:
- Return the box around the actual imagery only -- not its caption, axis labels, or the
  surrounding whitespace.
- If a placeholder's content is drawn line art (a plain box, an arrow, a text label, a simple
  geometric shape) rather than genuine raster imagery, OMIT it. A placeholder correctly left
  empty is a better result than one filled with the wrong thing.
- If a placeholder shows a composite graphic -- a grid of panels, a figure with sub-plots --
  return the box around the WHOLE composite, not one panel of it.
- Never return the same region for two different placeholders.
- Omit any placeholder you cannot locate. Do not guess.

Respond with ONLY a JSON object, no markdown fences and no commentary. Use the key
"box_2d" with coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000:

{"regions": [{"xml_id": "fig_encoder", "box_2d": [120, 340, 410, 660], "label": "encoder heatmap"}],
 "omitted": {"fig_arrow": "plain arrow, line art"}}"""


@dataclass
class DetectedRegion:
    """One located raster region, already mapped to its placeholder."""

    xml_id: str
    bbox: tuple[float, float, float, float]   # pixels, (x0, y0, x1, y1)
    box_2d: list[int] = field(default_factory=list)   # as returned, for debugging
    label: str = ""


def to_pixels(box_2d, image_size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    """Decode Gemini's `[ymin, xmin, ymax, xmax]`/1000 box into a pixel box.

    Returns `(x0, y0, x1, y1)` in the image's own pixel space, or None when the
    box is malformed or degenerate. Out-of-range values are clamped rather than
    rejected: a model that overshoots the grid edge by a few units has still
    located the region correctly.
    """
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box_2d)
    except (TypeError, ValueError):
        return None

    width, height = image_size
    x0 = max(0.0, min(1.0, xmin / GRID)) * width
    y0 = max(0.0, min(1.0, ymin / GRID)) * height
    x1 = max(0.0, min(1.0, xmax / GRID)) * width
    y1 = max(0.0, min(1.0, ymax / GRID)) * height

    # Some models emit the pairs the other way round. Ordering is recoverable
    # and a swap is unambiguous, so normalize instead of discarding the answer.
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def detect_regions(backend: ChatBackend, image_path: Path, placeholders: list,
                   params: dict | None = None) -> tuple[list[DetectedRegion], dict]:
    """Locate the region belonging in each placeholder, in one VLM call.

    `placeholders` are `RasterPlaceholder` records from Stage 1a. Returns the
    regions the model located and a report dict recording what it omitted and
    why, which is written next to the crops so a run can be audited.

    A failed or unparseable call returns no regions. That is the safe default:
    splicing the wrong crop corrupts the diagram, while splicing nothing leaves
    the placeholder exactly as Stage 1a drew it.
    """
    from PIL import Image

    if not placeholders:
        return [], {"error": "no placeholders"}

    listing = "\n".join(
        f"  - xml id={p.xml_id}" + (f'  (the reconstruction labels it "{p.text[:80]}")'
                                    if p.text else "")
        for p in placeholders
    )
    prompt = DETECT_PROMPT.replace("{placeholders}", listing)

    result = backend.generate([Message.user(prompt, images=[image_path])], **(params or {}))
    if not result.ok:
        return [], {"error": result.error or "detection call failed"}

    data = extract_json(result.text)
    if not isinstance(data, dict):
        return [], {"error": "detection response was not a JSON object"}

    image_size = Image.open(image_path).size
    known = {p.xml_id for p in placeholders}

    regions: list[DetectedRegion] = []
    seen: set[str] = set()
    rejected: dict[str, str] = {}

    for entry in data.get("regions") or []:
        if not isinstance(entry, dict):
            continue
        xml_id = str(entry.get("xml_id", "")).strip()
        # A hallucinated id has nothing to splice into, and a repeated one
        # would overwrite an earlier good match.
        if xml_id not in known:
            rejected[xml_id or "<missing>"] = "not a placeholder in this document"
            continue
        if xml_id in seen:
            rejected[xml_id] = "duplicate entry"
            continue
        bbox = to_pixels(entry.get("box_2d"), image_size)
        if bbox is None:
            rejected[xml_id] = f"unusable box_2d {entry.get('box_2d')!r}"
            continue
        regions.append(DetectedRegion(
            xml_id=xml_id, bbox=bbox,
            box_2d=list(entry.get("box_2d") or []),
            label=str(entry.get("label", "")),
        ))
        seen.add(xml_id)

    omitted = {str(k): str(v) for k, v in (data.get("omitted") or {}).items()
               if isinstance(data.get("omitted"), dict)}
    report = {
        "detected": len(regions),
        "placeholders": len(placeholders),
        "omitted": omitted,
        "rejected": rejected,
        "image_size": list(image_size),
    }
    return regions, report


def write_overlay(image_path: Path, regions: list[DetectedRegion], out_path: Path) -> Path | None:
    """Draw the detected boxes on the figure, for eyeballing a run afterwards."""
    if not regions:
        return None
    from .som import draw_marks

    items = [{"mark": r.xml_id, "bbox": r.bbox} for r in regions]
    return draw_marks(image_path, items, out_path)


def dump(regions: list[DetectedRegion]) -> str:
    return json.dumps([{"xml_id": r.xml_id, "bbox": list(r.bbox),
                        "box_2d": r.box_2d, "label": r.label} for r in regions], indent=2)
