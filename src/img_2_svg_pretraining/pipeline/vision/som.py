"""Set-of-Mark overlays and the Gemini verification layers built on them.

Two distinct SoM passes, both grounded the same way: numbered markers are
drawn on the figure and the model only ever sees integers, so it never has to
transcribe coordinates or long identifiers, and parsing maps numbers back to
objects afterwards. Same trick as `data_engine/edges.py::draw_marks`.

Pass 1 -- point verification. Molmo2 proposes candidate raster regions, but
it over-proposes: it points at drawn boxes, text labels and arrowheads as
well as genuine embedded imagery. A VLM sees the marked figure and says which
marks are real photographic/rendered content. This is a filter, and a filter
is exactly where a strong general VLM beats a pointing specialist -- Molmo2
is good at *locating* things, less good at judging what counts as a raster.

Pass 2 -- crop-to-placeholder mapping. The transmuter emits placeholder nodes
with `xml class=raster_node` at known coordinates. Geometric IoU matches most
of them to detected regions, but it fails when the generated layout is
displaced relative to the source figure -- which happens often, since the
model reconstructs positions by eye. So the same VLM is shown the detections
marked on the source figure and the placeholders marked on the render, and
asked to pair them up by *content* rather than position.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..backends import ChatBackend, Message
from ..extract import extract_json

MARK_COLORS = ["#ff3b30", "#007aff", "#34c759", "#ff9500", "#af52de",
               "#00c7be", "#ff2d55", "#5856d6"]


# -- overlay drawing -----------------------------------------------------

def draw_marks(image_path: Path, items: list[dict], out_path: Path,
               box_key: str = "bbox", label_key: str = "label") -> Path:
    """Draw a numbered marker per item, and its box when one is known.

    `items` entries carry `mark` (the integer shown), optional `bbox`
    (x0, y0, x1, y1) and optional `point` (x, y). Boxes are drawn in rotating
    colours so adjacent regions stay distinguishable.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1.0, min(image.width, image.height) / 900)
    try:
        font = ImageFont.load_default(size=int(20 * scale))
    except TypeError:  # older Pillow
        font = ImageFont.load_default()

    for i, item in enumerate(items):
        color = MARK_COLORS[i % len(MARK_COLORS)]
        mark = str(item.get("mark", i + 1))
        bbox = item.get(box_key)

        if bbox:
            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, int(3 * scale)))
            cx, cy = x0 + 14 * scale, y0 + 14 * scale
        else:
            cx, cy = item.get("point", (0, 0))

        radius = 15 * scale
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=color, outline="white", width=max(1, int(2 * scale)))
        draw.text((cx, cy), mark, fill="white", font=font, anchor="mm")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


# -- pass 1: verify Molmo2's point proposals -----------------------------

VERIFY_PROMPT = """You are checking a raster-region detector's output on a scientific figure.

The image has numbered markers on it. Each marker sits on a region the detector believes is an
embedded graphic -- a photograph, a rendered 3D object, a screenshot, a plot, a natural image,
an icon or a logo.

Your job is to say which markers are correct.

A marker is CORRECT when it sits on genuinely photographic or rendered raster content that
could not be redrawn with simple vector shapes.

A marker is WRONG when it sits on:
- a plain box, rectangle or arrow that is part of the diagram's line art
- a text label, caption, equation or axis label
- empty background or whitespace
- a simple geometric shape (circle, triangle, plain coloured block) with no image content
- the same region as another marker (a duplicate -- keep the better one, reject the rest)

There are {count} markers, numbered 1 to {count}.

Respond with ONLY a JSON object, no markdown fences and no commentary:

{"accept": [1, 3, 4], "reject": [2, 5], "notes": {"2": "plain arrow", "5": "duplicate of 4"}}

Every marker number must appear in exactly one of "accept" or "reject"."""


@dataclass
class VerifiedDetection:
    mark: int
    point_id: int
    x: float
    y: float
    bbox: tuple[float, float, float, float] | None
    score: float
    accepted: bool
    note: str = ""
    # False when the VLM call failed or returned unparseable output and the
    # detection was accepted by fallback rather than by judgment. Without
    # this an unverified run is indistinguishable from a clean one -- it just
    # looks like a filter that accepted everything.
    verified: bool = True


def verify_points(backend: ChatBackend, image_path: Path, detections: list,
                  work_dir: Path, params: dict | None = None) -> list[VerifiedDetection]:
    """Ask a VLM which detections are genuine raster regions.

    On an unparseable reply everything is accepted rather than dropped: a
    failed verification should degrade to the unfiltered detector, not
    silently delete every region.
    """
    if not detections:
        return []

    items = [{"mark": i + 1, "bbox": d.bbox, "point": (d.x, d.y)}
             for i, d in enumerate(detections)]
    marked = draw_marks(image_path, items, work_dir / "verify_points.png")

    prompt = VERIFY_PROMPT.replace("{count}", str(len(detections)))
    result = backend.generate([Message.user(prompt, images=[marked])], **(params or {}))

    accepted_marks = set(range(1, len(detections) + 1))
    notes: dict[str, str] = {}
    verified = False
    reason = result.error or "response contained no usable accept list"

    if result.ok:
        data = extract_json(result.text)
        if isinstance(data, dict) and "accept" in data:
            try:
                accepted_marks = {int(m) for m in data.get("accept", [])}
                notes = {str(k): str(v) for k, v in (data.get("notes") or {}).items()}
                verified = True
            except (TypeError, ValueError):
                pass

    if not verified:
        print(f"    verification unavailable ({reason[:90]}); "
              f"accepting all {len(detections)} proposals unfiltered")

    return [
        VerifiedDetection(
            mark=i + 1, point_id=d.point_id, x=d.x, y=d.y, bbox=d.bbox, score=d.score,
            accepted=(i + 1) in accepted_marks, note=notes.get(str(i + 1), ""),
            verified=verified,
        )
        for i, d in enumerate(detections)
    ]


# -- pass 2: map detections to placeholders ------------------------------

MAP_PROMPT = """You are matching detected image regions to placeholder boxes.

IMAGE A is the original scientific figure. Numbered markers show regions where real embedded
graphics were detected -- photographs, rendered objects, screenshots, plots, icons.

IMAGE B is a vector reconstruction of the same figure. Lettered markers show placeholder boxes
that were drawn where an embedded graphic should go, but which currently contain no imagery.

Match each lettered placeholder in IMAGE B to the numbered region in IMAGE A that belongs in it.

Match by CONTENT and by position in the diagram's structure -- which pipeline stage it belongs
to, what the surrounding labels say, what the region depicts. Do NOT match purely on pixel
coordinates: the reconstruction's layout is often shifted or rescaled relative to the original.

Placeholders in IMAGE B:
{placeholders}

Detected regions in IMAGE A: numbered 1 to {count}.

Leave a placeholder out entirely if no detected region genuinely belongs in it. Never use one
numbered region for two placeholders.

Respond with ONLY a JSON object, no markdown fences and no commentary:

{"matches": {"A": 2, "B": 1, "C": 4}, "unmatched": ["D"]}"""


@dataclass
class Placeholder:
    xml_id: str
    label: str                       # A, B, C ... shown to the model
    bbox: tuple[float, float, float, float] | None = None
    text: str = ""


def map_to_placeholders(backend: ChatBackend, source_image: Path, render_image: Path | None,
                        detections: list[VerifiedDetection], placeholders: list[Placeholder],
                        work_dir: Path, params: dict | None = None) -> dict[str, int]:
    """Return {xml_id: index into `detections`} for the pairs the VLM accepts.

    An unparseable reply yields no matches, which leaves the geometric
    fallback in charge -- the opposite default to `verify_points`, because a
    wrong mapping splices the wrong picture into the diagram, while a missing
    mapping merely leaves the placeholder as it was.
    """
    if not detections or not placeholders:
        return {}

    source_items = [{"mark": i + 1, "bbox": d.bbox, "point": (d.x, d.y)}
                    for i, d in enumerate(detections)]
    marked_source = draw_marks(source_image, source_items, work_dir / "map_source.png")

    images = [marked_source]
    if render_image and Path(render_image).exists():
        render_items = [{"mark": p.label, "bbox": p.bbox} for p in placeholders if p.bbox]
        if render_items:
            images.append(draw_marks(render_image, render_items, work_dir / "map_render.png"))

    listing = "\n".join(
        f"  {p.label}. xml id={p.xml_id}" + (f"  (label: {p.text})" if p.text else "")
        for p in placeholders
    )
    prompt = (MAP_PROMPT
              .replace("{placeholders}", listing)
              .replace("{count}", str(len(detections))))

    result = backend.generate([Message.user(prompt, images=images)], **(params or {}))
    if not result.ok:
        return {}

    data = extract_json(result.text)
    if not isinstance(data, dict):
        return {}

    by_label = {p.label: p for p in placeholders}
    mapping: dict[str, int] = {}
    used: set[int] = set()
    for label, mark in (data.get("matches") or {}).items():
        placeholder = by_label.get(str(label).strip().upper())
        try:
            index = int(mark) - 1
        except (TypeError, ValueError):
            continue
        # Reject out-of-range marks and reuse of one region for two
        # placeholders -- both would splice the wrong crop.
        if placeholder is None or not (0 <= index < len(detections)) or index in used:
            continue
        mapping[placeholder.xml_id] = index
        used.add(index)
    return mapping
