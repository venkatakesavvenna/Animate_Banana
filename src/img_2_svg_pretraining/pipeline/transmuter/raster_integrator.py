"""Stage 1b -- Raster Integrator.

Fills the transmuter's raster placeholders with the real imagery from the
source figure, in three steps:

  1. DETECT  One call to the same VLM the rest of the pipeline uses. It is
             shown the source figure and the list of placeholders Stage 1a
             emitted, and returns a box per placeholder it can locate, in
             Gemini's `[ymin, xmin, ymax, xmax]`/1000 convention.
  2. CROP    Each box is cut from the source figure and written next to the
             .tex.
  3. SPLICE  Placeholder bodies are rewritten to \\includegraphics, preserving
             every `xml id`.

This used to be a five-step chain: Molmo2 proposed points, a Set-of-Mark pass
filtered them, a fine-tuned SAM3 checkpoint segmented the survivors, and a
second SoM pass mapped crops back to placeholders. Detection now happens inside
the model that was already doing the filtering, which is what lets the pipeline
hold only one model at a time -- the old route needed two vision checkpoints on
GPU under a separate venv, because Molmo2 requires transformers==4.57.1.

Mapping disappears rather than moving: because Stage 1a's placeholders are part
of the detection prompt, the model returns the `xml id` it matched, so locating
a region and deciding where it belongs are one answer instead of two passes
that could disagree.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..cache import write_text
from ..runner import AgentContext, SampleOutcome, StageReport
from ..samples import PaperSample
from ..vision.gemini_boxes import detect_regions, dump, write_overlay
from .rasters import find_placeholders, splice
from .tikz_rasters import RasterPlaceholder

AGENT = "raster_integrator"


def _expand_to_placeholder(bbox, placeholder_box, image_size) -> tuple:
    """Grow a detected box toward the placeholder's aspect ratio when it under-covers.

    A detector asked for "the region that belongs here" can still return one
    panel of a composite graphic -- a 2x2 grid of heatmaps, a figure with
    sub-plots -- rather than the whole thing. Observed on `CVPR_2025_pipe00002`:
    the four-panel Phi(x,delta) graphic came back as its bottom row only, half
    the height of its neighbours.

    The placeholder's declared box says how much space the transmuter expected
    the graphic to occupy, so it is a usable prior for how much was missed.
    Only ever grows, never shrinks, and stays inside the image.
    """
    if not bbox or not placeholder_box:
        return bbox

    x0, y0, x1, y1 = bbox
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    px0, py0, px1, py1 = placeholder_box
    pw, ph = max(1.0, px1 - px0), max(1.0, py1 - py0)

    target = pw / ph
    current = w / h
    # Only act on a clear mismatch; small differences are just crop padding
    # and differing margins. The observed sub-panel failure came in at 1.55x
    # (a 2.00 box against a 1.29 placeholder), so the band has to be tighter
    # than that to catch it.
    if 0.75 < current / target < 1.35:
        return bbox

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if current < target:      # too tall/narrow -> widen
        w = h * target
    else:                     # too wide/short -> heighten
        h = w / target
    iw, ih = image_size
    return (max(0.0, cx - w / 2), max(0.0, cy - h / 2),
            min(float(iw), cx + w / 2), min(float(ih), cy + h / 2))


def _crop(image_path: Path, bbox, out_path: Path, pad: float = 2.0) -> Path | None:
    """Write a crop of the source figure, padded slightly and clamped."""
    from PIL import Image

    if not bbox:
        return None
    image = Image.open(image_path).convert("RGB")
    x0, y0, x1, y1 = bbox
    box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)),
           min(image.width, int(x1 + pad)), min(image.height, int(y1 + pad)))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None  # degenerate box; not worth splicing
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(out_path)
    return out_path


def _integrate(ctx: AgentContext, sample: PaperSample, code: str,
               placeholders: list[RasterPlaceholder], params: dict) -> SampleOutcome:
    """Detect, crop and splice one sample."""
    from PIL import Image

    work_dir = ctx.paths.rasters(sample.id)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.paths.code_final(sample.id)

    regions, report = detect_regions(ctx.backend, sample.image_path, placeholders, params)

    # No usable detection: pass the placeholder version through unchanged so
    # downstream stages can always read code_final regardless of whether 1b
    # found anything.
    if not regions:
        write_text(out, code)
        write_text(work_dir / "detections.json", json.dumps({
            **report, "replaced": [], "provenance": ctx.provenance(),
        }, indent=2, default=str))
        detail = report.get("error") or f"0/{len(placeholders)} placeholders located"
        # A detection call that failed outright is not the same as one that
        # ran and correctly found nothing worth splicing.
        status = "unresolved" if report.get("error") else "ok"
        return SampleOutcome(sample.id, status, out, detail)

    write_overlay(sample.image_path, regions, work_dir / "detected.png")
    write_text(work_dir / "regions.json", dump(regions))

    by_id = {p.xml_id: p for p in placeholders}
    image_size = Image.open(sample.image_path).size

    replacements: dict[str, str] = {}
    for region in regions:
        bbox = _expand_to_placeholder(
            region.bbox, by_id[region.xml_id].bbox(), image_size)
        crop_path = _crop(sample.image_path, bbox, work_dir / f"crop_{region.xml_id}.png")
        if crop_path:
            # Absolute path. A relative one would be tempting for portability,
            # but every consumer compiles the source from somewhere other than
            # its own directory -- `viewer/compile.py` writes it to a temp work
            # dir, and the exporter to the export dir -- so a relative path
            # resolves against the wrong root and the document fails with
            # `File not found`. Paths stay valid across host/container because
            # /code is the same tree on both.
            replacements[region.xml_id] = crop_path.resolve().as_posix()

    new_code, replaced = splice(code, replacements, ctx.cfg.target)
    write_text(out, new_code)

    write_text(work_dir / "detections.json", json.dumps({
        **report,
        "regions": [{"xml_id": r.xml_id, "box_2d": r.box_2d,
                     "bbox": list(r.bbox), "label": r.label} for r in regions],
        "replaced": replaced,
        "provenance": ctx.provenance(),
    }, indent=2, default=str))

    detail = (f"{len(placeholders)} placeholder(s) -> {len(regions)} located "
              f"-> {len(replaced)} filled")
    return SampleOutcome(sample.id, "ok", out, detail)


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    report = StageReport(agent=AGENT)
    agent = ctx.agent

    if agent.option("enabled", False) is False:
        print(f"[{AGENT}] disabled in config; skipping")
        return report

    # -- which samples need work ----------------------------------------
    pending: list[tuple[PaperSample, str, list[RasterPlaceholder]]] = []
    for sample in samples:
        out = ctx.paths.code_final(sample.id)
        if out.exists() and not force:
            report.outcomes.append(SampleOutcome(sample.id, "skipped", out, "cached"))
            continue
        code_path = ctx.paths.code(sample.id)
        if not code_path.exists():
            report.outcomes.append(SampleOutcome(
                sample.id, "failed", None, "no diagram code; run convert-code first"))
            continue
        code = code_path.read_text(encoding="utf-8")
        placeholders = find_placeholders(code, ctx.cfg.target)
        if not placeholders:
            # Nothing to integrate: copy through so downstream stages can
            # always read code_final without caring whether 1b applied.
            write_text(out, code)
            report.outcomes.append(SampleOutcome(
                sample.id, "ok", out, "no raster placeholders"))
            continue
        pending.append((sample, code, placeholders))

    if not pending:
        return report

    params = ctx.params()
    print(f"[{AGENT}] {len(pending)} sample(s): locating raster regions via "
          f"{agent.backend} ({cfg.backend_model(agent.backend)})")

    for sample, code, placeholders in pending:
        try:
            report.outcomes.append(_integrate(ctx, sample, code, placeholders, params))
        except Exception as e:
            # One sample failing must not abandon the rest of the batch.
            report.outcomes.append(SampleOutcome(
                sample.id, "failed", None, f"{type(e).__name__}: {e}"))

    ctx.unload()
    return report
