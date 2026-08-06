"""Stage-1 metrics: diagram compilation, component classes, render fidelity.

Compilation is mechanical. The other two are judged, and both are deliberately
run against the **code** rather than the induced XML: the semantic class tags
originate at the diagram-to-code step, so an error there propagates into the
XML and scoring the XML would attribute a D2C mistake to the parser.

Rendering fidelity is CSR-gated (design doc §4.2): code that does not compile
scores 0 rather than None, because "no render" is a real failure of the stage,
not missing data.
"""
from __future__ import annotations

import re
from pathlib import Path

from img_2_svg_pretraining.pipeline.prompts import load_and_render

from ..judge import JudgeError, PROMPTS_ROOT

# `xml class=raster_node` / `xml class="raster_node"` inside a node's options.
_CLASS_RE = re.compile(r"xml\s+class\s*=\s*\{?\"?([a-z_]+)\"?\}?")
_ID_RE = re.compile(r"xml\s+id\s*=\s*\{?\"?([\w:.-]+)\"?\}?")

SEMANTIC_CLASSES = ("block", "standalone_node", "child_node", "raster_node",
                    "text_node", "composite_whole", "composite_part", "edge")


def compile_diagram(code: str, cache_dir: Path) -> dict:
    """Compilation Success Rate input, plus the render used by fidelity."""
    from img_2_svg_pretraining.viewer.compile import compile_tikz

    result = compile_tikz(code, cache_dir)
    return {
        "csr": 1.0 if result.ok else 0.0,
        "compiles": bool(result.ok),
        "render_path": str(result.png_path) if result.ok and result.png_path else None,
        "compile_log": "" if result.ok else result.log[-2000:],
    }


def declared_classes(code: str) -> list[dict]:
    """Every (id, class) the diagram code tags, in source order."""
    out: list[dict] = []
    for match in _CLASS_RE.finditer(code):
        # The id normally precedes the class within the same option list;
        # take the nearest preceding one.
        head = code[:match.start()]
        ids = _ID_RE.findall(head)
        out.append({"id": ids[-1] if ids else f"<unnamed@{match.start()}>",
                    "declared_class": match.group(1)})
    return out


def component_score(judge, code: str, image_path: Path, tagged: list[dict]) -> dict:
    """Diagram Component Score: is each element's semantic class right? (§2.1.3)

    The judge sees the figure and the code together and rules on each tagged
    element. Its verdicts are matched back by id and anything it invents is
    discarded, so the accuracy denominator is always the code's own tag count.
    """
    if not tagged:
        return {"component_accuracy": None,
                "component_detail": "code declares no xml class tags"}

    listing = "\n".join(f"  {t['id']}: {t['declared_class']}" for t in tagged)
    prompt = load_and_render("component_score.yaml#prompt", {
        "diagram_code": code,
        "tagged_elements": listing,
    }, root=PROMPTS_ROOT)

    try:
        data = judge.ask_json(prompt, images=[image_path], tag="component_score")
    except JudgeError as e:
        return {"component_accuracy": None, "component_error": str(e)}

    known = {t["id"]: t["declared_class"] for t in tagged}
    verdicts: list[dict] = []
    ignored: list[str] = []
    for entry in data.get("elements") or []:
        if not isinstance(entry, dict):
            continue
        el_id = str(entry.get("id", ""))
        if el_id not in known:
            ignored.append(el_id)
            continue
        correct = bool(entry.get("correct"))
        verdicts.append({
            "id": el_id,
            "declared": known[el_id],
            "correct": correct,
            "should_be": str(entry.get("should_be") or "") if not correct else "",
            "reason": str(entry.get("reason") or "")[:300],
        })

    judged = {v["id"] for v in verdicts}
    unjudged = sorted(set(known) - judged)
    correct_n = sum(v["correct"] for v in verdicts)

    per_class: dict[str, dict] = {}
    for v in verdicts:
        bucket = per_class.setdefault(v["declared"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += v["correct"]

    # The two confusion modes the design doc calls out (§2.1.3).
    text_vs_child = [v for v in verdicts if not v["correct"]
                     and {v["declared"], v["should_be"]} == {"text_node", "child_node"}]
    raster_over = [v for v in verdicts if not v["correct"]
                   and v["declared"] == "raster_node"]

    return {
        "component_accuracy": (correct_n / len(verdicts)) if verdicts else None,
        "component_judged": len(verdicts),
        "component_tagged": len(known),
        "component_unjudged": unjudged,
        "component_ignored_ids": ignored,
        "component_per_class": per_class,
        "confusion_text_vs_child": len(text_vs_child),
        "confusion_raster_over_trigger": len(raster_over),
        "component_verdicts": verdicts,
    }


def rendering_fidelity(judge, source_image: Path, render_path: Path | None) -> dict:
    """Rendering Fidelity (§2.3.1) on the SFS rubric (§2.2.1)."""
    if render_path is None:
        # CSR-gated: a non-compiling diagram has zero fidelity, not unknown.
        return {"rendering_fidelity": 0.0,
                "rendering_fidelity_note": "diagram did not compile"}

    prompt = load_and_render("rendering_fidelity.yaml#prompt", {}, root=PROMPTS_ROOT)
    try:
        data = judge.ask_json(prompt, images=[source_image, Path(render_path)],
                              tag="rendering_fidelity")
    except JudgeError as e:
        return {"rendering_fidelity": None, "rendering_fidelity_error": str(e)}

    score = data.get("score")
    return {
        "rendering_fidelity": float(score) if isinstance(score, (int, float)) else None,
        "rendering_fidelity_breakdown": data.get("breakdown"),
        "rendering_fidelity_notes": str(data.get("notes") or "")[:1000],
    }


def code_quality(judge, code: str, variant: str = "diagram") -> dict:
    """Code Quality (§2.3.3). Optional -- costs a judge call per sample."""
    prompt = load_and_render(f"code_quality.yaml#{variant}", {"code": code},
                             root=PROMPTS_ROOT)
    try:
        data = judge.ask_json(prompt, tag=f"{variant}_code_quality")
    except JudgeError as e:
        return {f"{'code_quality' if variant == 'diagram' else 'anim_code_quality'}": None,
                "code_quality_error": str(e)}

    key = "code_quality" if variant == "diagram" else "anim_code_quality"
    score = data.get("score")
    return {
        key: float(score) if isinstance(score, (int, float)) else None,
        f"{key}_breakdown": data.get("breakdown"),
        f"{key}_findings": data.get("findings"),
    }


def run(judge, code: str, image_path: Path, cache_dir: Path,
        include_quality: bool = False) -> dict:
    record = {"suite": "stage1"}
    compiled = compile_diagram(code, cache_dir)
    record.update(compiled)

    tagged = declared_classes(code)
    if judge is not None:
        record.update(component_score(judge, code, image_path, tagged))
        record.update(rendering_fidelity(judge, image_path, compiled["render_path"]))
        if include_quality:
            record.update(code_quality(judge, code, "diagram"))
    else:
        record["judge_skipped"] = "no judge configured"
    return record
