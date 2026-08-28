"""Stage-3 metrics: animation compilation, code quality, integration footprint.

AIF (§2.3.3) measures how cleanly animation was layered onto the diagram code:
a low ratio means the animator added frames without disturbing the drawing, a
high one means it rewrote the diagram to animate it. It is computed from a
real line diff between the stage-1 code and the animation source, so it needs
no model.

Animation CSR is measured twice on purpose. `anim_csr` runs the same source
transforms the exporter runs (which repair a couple of recurring model slips),
so it reports what actually ships; `raw_compiles` skips them, so the value of
that repair layer stays visible instead of being quietly folded into the score.
"""
from __future__ import annotations

import difflib
from pathlib import Path

from img_2_svg_pretraining.pipeline.export.render import RenderError, compile_pdf
from img_2_svg_pretraining.pipeline.export.tikz_source import to_multipage_pdf_source

from .stage1_code import code_quality


def _try_compile(source: str, out_pdf: Path) -> tuple[bool, str]:
    try:
        compile_pdf(source, out_pdf)
        return True, ""
    except RenderError as e:
        return False, str(e)[-1500:]
    except Exception as e:  # latexmk absent, timeout, etc.
        return False, f"{type(e).__name__}: {e}"


def compile_animation(anim_code: str, work_dir: Path,
                      target: str = "tikz") -> dict:
    """Does the animation code actually build?

    `target` matters: an SVG animation is CSS `@keyframes` inside an XML
    document, and running pdflatex over it reports `anim_csr: 0.0` for an
    animation that exported 23 frames perfectly well. That is a property of
    the harness, not of the animation, and it is exactly the kind of zero
    that looks like a finding.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if target == "svg":
        # There is no repair step to compare against, so raw and repaired are
        # the same document and `repair_rescued` is meaningless rather than
        # False-because-it-failed.
        ok, log = _svg_builds(anim_code)
        return {
            "anim_csr": 1.0 if ok else 0.0,
            "anim_compiles": ok,
            "raw_compiles": ok,
            "repair_rescued": False,
            "anim_compile_log": "" if ok else log,
        }

    ok, log = _try_compile(to_multipage_pdf_source(anim_code), work_dir / "repaired.pdf")
    raw_ok, _ = _try_compile(anim_code, work_dir / "raw.pdf")

    return {
        "anim_csr": 1.0 if ok else 0.0,
        "anim_compiles": ok,
        "raw_compiles": raw_ok,
        "repair_rescued": bool(ok and not raw_ok),
        "anim_compile_log": "" if ok else log,
    }


def _svg_builds(anim_code: str) -> tuple[bool, str]:
    """Well-formed XML carrying an <svg> root, and rasterisable.

    Parsing alone is too weak -- an XML document with no <svg> element parses
    fine and animates nothing. Rendering is the SVG counterpart of "pdflatex
    exited 0": `render_svg` reports the same CompileResult contract that
    `compile_tikz` does, deliberately (see svg_render.py).
    """
    import tempfile
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(anim_code)
    except ET.ParseError as e:
        return False, f"SVG is not well-formed XML: {e}"
    if not root.tag.endswith("svg"):
        return False, f"document root is <{root.tag}>, not <svg>"

    from img_2_svg_pretraining.pipeline.svg_render import render_svg
    with tempfile.TemporaryDirectory() as tmp:
        result = render_svg(anim_code, Path(tmp))
    return bool(result.ok), "" if result.ok else (result.log or "SVG did not render")


def integration_footprint(diagram_code: str, anim_code: str) -> dict:
    """AIF = diagram lines touched / animation lines added.

    `difflib` opcodes give exactly the two quantities the doc asks for:
    'replace' and 'delete' consume pre-existing diagram lines (touched),
    while 'insert' and the replacement side of 'replace' are animation lines
    added. A purely additive animation scores 0.
    """
    before = diagram_code.splitlines()
    after = anim_code.splitlines()
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)

    touched = 0     # diagram lines modified or deleted
    added = 0       # animation-related lines added
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            touched += i2 - i1
        elif tag == "replace":
            touched += i2 - i1
            added += j2 - j1
        elif tag == "insert":
            added += j2 - j1

    return {
        "aif": (touched / added) if added else None,
        "aif_lines_touched": touched,
        "aif_lines_added": added,
        "aif_note": ("no lines added; animation is identical to the diagram"
                     if not added else ""),
    }


def run(judge, diagram_code: str, anim_code: str, work_dir: Path,
        include_quality: bool = True, target: str = "tikz") -> dict:
    record = {"suite": "stage3", "target": target}
    record.update(compile_animation(anim_code, work_dir, target=target))
    record.update(integration_footprint(diagram_code, anim_code))
    if judge is not None and include_quality:
        record.update(code_quality(judge, anim_code, "animation"))
    return record
