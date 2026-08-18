"""Side-by-side comparison viewer for an AnimateBench run.

Shows, for one sample and one animation style, four things at once:

  1. the source figure (ground truth input),
  2. the benchmark's own reference animation, shipped in the sample bundle,
  3-5. our pipeline's animation for each configured model.

The point is qualitative: play them together and see which pipeline tells the
better story. Numeric metrics come later.

Each model is a separate pipeline config, so its artifacts live under its own
cache lineage. This resolves them by loading every config and asking its
`CachePaths` where things are, rather than guessing at directory names -- when
a lineage component changes, the paths move, and only the config knows where
to.

Run inside the container:
    python -m img_2_svg_pretraining.pipeline.inspector.compare --port 7861

Port 7861 keeps this separate from the single-run inspector on 7860, so both
can be open at once. Both ports must be published at container creation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from flask import Flask, abort, jsonify, send_file

from img_2_svg_pretraining.pipeline.cache import CachePaths
from img_2_svg_pretraining.pipeline.config import load_config
from img_2_svg_pretraining.pipeline.samples import discover_samples
from img_2_svg_pretraining.pipeline.schema import AnimationSequence
from img_2_svg_pretraining.pipeline.styles import STYLES

app = Flask(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "configs"

# The three comparison configs, in display order. Label -> config file.
DEFAULT_CONFIGS = {
    "Gemini 3.6 Flash": "bench_gemini.yaml",
    "Gemma 4 31B": "bench_gemma4.yaml",
    "Qwen3.6 27B": "bench_qwen.yaml",
}

# Styles the benchmark ships, in the order its own outputs list them.
BENCH_STYLES = ["progressive_reveal", "colour_pop", "alpha_masking",
                "hopping_bounding_box", "sliding_bounding_box"]

STATE: dict = {"models": {}, "samples": [], "by_id": {}, "styles": BENCH_STYLES}


def _init(config_files: dict[str, str], dataset_root: str | None) -> None:
    models = {}
    samples = None
    for label, filename in config_files.items():
        path = CONFIG_DIR / filename
        if not path.exists():
            print(f"  ! {label}: no config at {path}, skipping")
            continue
        cfg = load_config(path)
        models[label] = {"cfg": cfg, "file": filename,
                         "model": cfg.backend_model(cfg.agent("designer").backend)}
        if samples is None:
            root = dataset_root or cfg.dataset_root
            samples = discover_samples(root)

    if not models:
        raise SystemExit("no comparison configs could be loaded")

    STATE.update(models=models, samples=samples or [],
                 by_id={s.id: s for s in (samples or [])})


def _paths_for(label: str, style: str) -> CachePaths:
    """CachePaths for one model at one style.

    The style is part of the sequence lineage, so it must be set on the config
    before paths are resolved -- otherwise every style would resolve to the
    config's default and the viewer would show the same video five times.
    """
    cfg = STATE["models"][label]["cfg"]
    cfg.style = style
    cfg.raw["animation_style"] = style
    return CachePaths.from_config(cfg)


def _reference(sample_id: str) -> dict:
    """The benchmark's own outputs for this sample, as imported."""
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        return {}
    ref = Path(sample.directory) / "reference"
    index = ref / "index.json"
    if not index.exists():
        return {}
    try:
        return json.loads(index.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _metrics(cfg, config_file: str, style: str, sample_id: str) -> dict:
    """AnimateBench scores for one panel, with each metric's description.

    Reads the eval records the metrics CLI writes; a panel that has not been
    scored yet simply reports nothing rather than blocking the videos.
    """
    from img_2_svg_pretraining.animatebench import descriptions, results

    root = _evals_root(cfg)
    config_name = Path(config_file).stem

    suites: list[dict] = []
    for suite in ("stage1", "xml", "sequence", "stage3", "animation"):
        record = results.read_record(
            results.suite_path(root, config_name, style, sample_id, suite))
        if record is None:
            continue
        entries = []
        for key in descriptions.ordered(suite):
            if key not in record or record[key] is None:
                continue
            entries.append({"key": key, "value": record[key],
                            **descriptions.describe(key)})
        if entries or record.get("error"):
            suites.append({"suite": suite, "metrics": entries,
                           "error": record.get("error")})
    return {"available": bool(suites), "suites": suites}


def _sequence_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        seq = AnimationSequence.load(path)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {
        "steps": len(seq.nodes),
        "traversal_style": seq.traversal_style,
        "narrated": sum(1 for n in seq.nodes if n.narrative),
        "elements": len(seq.focus_ids()),
        "bench_schema": any(n.element_classes for n in seq.nodes),
    }


# Suite -> column group label, matching the four pipeline stages.
_TABLE_SUITES = [("stage1", "Code"), ("xml", "XML"),
                  ("sequence", "Sequence"), ("stage3", "Animation"),
                  ("animation", "Anim. quality")]

# Metric -> the record keys holding the evidence behind its score. These are
# the per-element detail lists every metric writes alongside its number and
# that the panel/scoreboard views drop; the explain view is the one place
# they surface, so a reviewer can see *which* elements produced the score.
_EVIDENCE: dict[str, list[str]] = {
    "csr": ["compile_log"],
    "component_accuracy": ["component_per_class", "component_verdicts",
                           "component_unjudged", "component_ignored_ids",
                           "confusion_text_vs_child",
                           "confusion_raster_over_trigger"],
    "rendering_fidelity": ["rendering_fidelity_breakdown",
                           "rendering_fidelity_notes",
                           "rendering_fidelity_note"],
    "paa": ["parent_detail"],
    "matched_coverage": ["matched", "scorable_gt_elements"],
    "edge_f1": ["edge_precision", "edge_recall", "gt_edges",
                "pred_edges_comparable", "matched_edges", "missed_gt_edges",
                "spurious_pred_edges", "pred_edges_uncontractable"],
    "depth_violation_rate": ["depth_violations", "elements_checked"],
    "vfs": ["vfs_raw", "vfs_frame_scores", "vfs_frames", "vfs_errors",
            "vfs_frame_policy", "vfs_frames_judged", "vfs_frames_available"],
    "ascs_pass": ["ascs_frames_judged", "ascs_frames_discarded",
                  "ascs_discarded_frames", "ascs_unenforced_rules",
                  "ascs_frame_detail", "ascs_errors"],
    "omission_rate": ["element_omission_count", "omission_checklist_size",
                      "omission_elements_remaining", "omission_frame_detail",
                      "omission_errors"],
    "arrow_omission_count": ["omission_arrows_remaining",
                             "omission_scores_edges"],
    "sss": ["sss_band_mean", "sss_steps_scored", "sss_steps_total",
            "sss_steps_invalid", "sss_step_detail", "sss_errors"],
    "gps": ["gps_band_mean", "gps_steps_scored", "gps_steps_total",
            "gps_steps_invalid", "gps_step_detail", "gps_errors"],
    "repetition_rate": ["repetition_status", "repetition_note",
                        "unnecessary_repetition_count_element",
                        "unnecessary_repetition_count_arrow",
                        "repetition_unjustified_elements",
                        "repetition_revisited", "repetition_frame_detail",
                        "repetition_errors"],
    "coverage_recall": ["missed_groups", "gt_animated_unmatchable",
                        "gt_animated", "gt_animated_matchable"],
    "coverage_precision": ["covered_groups", "pred_animated_unmatchable",
                           "pred_animated", "pred_animated_matchable"],
    "tof": ["tof_detail"],
    "sscr_pass": ["sscr_violations"],
    "dovr": ["dovr_violations", "edges_tested", "edges_untestable"],
    "anim_csr": ["anim_compiles", "raw_compiles", "repair_rescued",
                 "anim_compile_log"],
    "aif": ["aif_lines_touched", "aif_lines_added", "aif_note"],
}


def _evals_root(cfg) -> Path:
    """Where eval records live.

    Normally beside the artifacts, but the pipeline cache is root-owned
    wherever a root container wrote it first, so `run_eval --evals-root` can
    put records somewhere an ordinary user can write. `ANIMATEBENCH_EVALS_ROOT`
    points this viewer at the same place, and must match whatever that run
    used -- otherwise the tab silently shows no scores for a sample that has
    them.
    """
    from img_2_svg_pretraining.animatebench import results

    override = os.environ.get("ANIMATEBENCH_EVALS_ROOT")
    if override:
        return Path(override)
    return results.evals_root(cfg.cache_root, cfg.dataset_root.name)


def _local_candidates(stored: str | None, cache_root: Path) -> list[Path]:
    """Where a recorded artifact path might actually live on this machine.

    Eval records written inside the container hold `/code/...` paths, and
    `/code` does not exist outside it. Rather than guess how many segments to
    strip, re-anchor on the local cache root: the tail of the stored path from
    `cache/` onward is stable across both layouts.
    """
    if not stored:
        return []
    candidates = [Path(stored)]
    parts = Path(stored).parts
    if "cache" in parts:
        tail = parts[parts.index("cache") + 1:]
        candidates.append(Path(cache_root).joinpath(*tail))
    return candidates


def _evidence_for(key: str, record: dict) -> dict:
    """The stored evidence behind one metric, empties dropped."""
    out = {}
    for name in _EVIDENCE.get(key, []):
        value = record.get(name)
        if value is None or value == "" or value == [] or value == {}:
            continue
        out[name] = value
    return out


@app.get("/api/table/<label>/<style>")
def api_table(label, style):
    """Paper-style scoreboard: rows = samples, columns = stage -> metrics.

    One table per (model, style), since a style change is a different
    pipeline run end to end. Reuses the same eval records _metrics() reads
    for the per-panel view, just pivoted into one row per sample.
    """
    from img_2_svg_pretraining.animatebench import descriptions

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    cfg = STATE["models"][label]["cfg"]
    config_file = STATE["models"][label]["file"]

    groups = []
    for suite, suite_label in _TABLE_SUITES:
        keys = descriptions.ordered(suite)
        groups.append({
            "suite": suite, "label": suite_label,
            "columns": [{"key": k, **descriptions.describe(k)} for k in keys],
        })

    rows = []
    for sample in STATE["samples"]:
        m = _metrics(cfg, config_file, style, sample.id)
        by_suite = {s["suite"]: s for s in m["suites"]}
        cells = {}
        for suite, _ in _TABLE_SUITES:
            entry = by_suite.get(suite)
            if entry is None:
                cells[suite] = {"error": None, "values": {}}
                continue
            cells[suite] = {
                "error": entry.get("error"),
                "values": {e["key"]: e["value"] for e in entry["metrics"]},
            }
        rows.append({"id": sample.id, "title": sample.title, "cells": cells})

    return jsonify({"style": style, "label": label, "groups": groups, "rows": rows})


@app.get("/api/explain/<label>/<sample_id>/<style>")
def api_explain(label, sample_id, style):
    """One sample, one style: what it looked like, what it scored, and why.

    Every number is joined to the formula that produced it, instantiated on
    this record's own terms, and to the evidence the metric stored alongside
    it. Nothing here is recomputed -- this reads what the eval wrote, so the
    view cannot disagree with the score.
    """
    from img_2_svg_pretraining.animatebench import descriptions, results

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404, f"unknown sample '{sample_id}'")

    cfg = STATE["models"][label]["cfg"]
    config_name = Path(STATE["models"][label]["file"]).stem
    paths = _paths_for(label, style)
    root = _evals_root(cfg)

    suites, provenance = [], None
    for suite, suite_label in _TABLE_SUITES:
        record = results.read_record(
            results.suite_path(root, config_name, style, sample_id, suite))
        if record is None:
            continue
        provenance = provenance or record.get("provenance")
        metrics = []
        for key in descriptions.ordered(suite):
            if key not in record or record[key] is None:
                continue
            metrics.append({
                "key": key, "value": record[key],
                **descriptions.describe(key),
                "instantiated": descriptions.instantiate(key, record),
                "evidence": _evidence_for(key, record),
            })
        if metrics or record.get("error"):
            suites.append({
                "suite": suite, "label": suite_label, "metrics": metrics,
                "error": record.get("error"),
                "skipped": record.get("coverage_skipped"),
                **descriptions.suite_note(suite),
            })

    alignment = results.read_record(
        results.alignment_path(root, config_name, sample_id)) or {}

    exports = paths.exports(sample_id)
    frames_dir = exports / "frames"
    reference = _reference(sample_id)
    ref_videos = reference.get("videos", {})
    ref_key = f"tikz|{style}|full"

    return jsonify({
        "id": sample_id, "title": sample.title, "style": style,
        "label": label, "model": STATE["models"][label]["model"],
        "scored": bool(suites),
        "figure": f"/api/figure/{sample_id}",
        "render": f"/api/render/{label}/{sample_id}/{style}",
        "reference_video": (f"/api/reference/{sample_id}/{ref_videos[ref_key]}"
                            if ref_key in ref_videos else None),
        "pipeline_video": (f"/api/video/{label}/{sample_id}/{style}"
                           if (exports / "animation.mp4").is_file() else None),
        "frames": len(list(frames_dir.glob("*.png"))) if frames_dir.is_dir() else 0,
        "suites": suites,
        "alignment": {
            "groups": alignment.get("groups", {}),
            "gt_unmatched": alignment.get("gt_unmatched", []),
            "pred_unmatched": alignment.get("pred_unmatched", []),
            "notes": alignment.get("notes", []),
        },
        "provenance": provenance,
    })


# Which record key holds the per-call trace for each node of the animation
# tree, and how that node's prompt is assembled. The prompt is rendered here
# through the metric module's own `_adapter` + `load_and_render`, never a copy
# pasted from the YAML: a reviewer tuning a prompt has to be looking at the
# text the judge actually received, or the tuning is guesswork.
_ANIM_NODES = [
    ("vfs", "Elimination 1 · Visual Fidelity", "animation_fidelity.yaml",
     "frames", "adapter_frames", "vfs_frames"),
    ("ascs", "Elimination 2 · Style Compliance", "animation_style.yaml",
     "frames", "adapter", "ascs_frame_detail"),
    ("omission", "Elimination 3 · Omitted Elements", "animation_omission.yaml",
     "frames", "adapter", "omission_frame_detail"),
    ("sss", "Contributor 1 · Selection Sensibility", "animation_selection.yaml",
     "user", "adapter", "sss_step_detail"),
    ("gps", "Contributor 2 · Granularity & Pacing", "animation_pacing.yaml",
     "user", "adapter", "gps_step_detail"),
    ("repetition", "Contributor 3 · Unnecessary Repetition",
     "animation_repetition.yaml", "frames", "adapter", "repetition_frame_detail"),
]


def _anim_prompt(prompt_file: str, body_key: str, adapter_prefix: str,
                 style: str) -> dict:
    """Render one node's prompt exactly as the metric would send it."""
    from img_2_svg_pretraining.animatebench.judge import PROMPTS_ROOT
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq
    from img_2_svg_pretraining.pipeline.prompts import load_and_render

    try:
        values = {"style_adapter": aq._adapter(prompt_file, style, adapter_prefix)}
        for extra, key in (("output_schema", "schema"),):
            try:
                values[extra] = aq._adapter(prompt_file, style, key)
            except Exception:
                pass
        text = load_and_render(f"{prompt_file}#{body_key}", values,
                               root=PROMPTS_ROOT)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "file": prompt_file}
    return {"file": prompt_file, "key": body_key, "text": text,
            "adapter": values.get("style_adapter", ""),
            "chars": len(text)}


def _checklist_prompt() -> dict:
    """The shared first call. It takes no style adapter -- one prompt builds the
    checklist that both omission and repetition consume."""
    from img_2_svg_pretraining.animatebench.judge import PROMPTS_ROOT
    from img_2_svg_pretraining.pipeline.prompts import load_and_render
    try:
        text = load_and_render("animation_checklist.yaml#prompt",
                               {"diagram_code": "<diagram code appended at call time>",
                                "sequence_view": "<bucketed sequence appended at call time>"},
                               root=PROMPTS_ROOT)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"file": "animation_checklist.yaml", "key": "prompt",
            "text": text, "chars": len(text)}


@app.get("/api/animation/<label>/<sample_id>/<style>")
def api_animation(label, sample_id, style):
    """The animation tree for one cell: prompts, call counts, per-call trace.

    Exists so the judged half can be audited the way the programmatic half
    already can. A score from a model is only reviewable if you can see the
    instruction it was given and the answer it gave for each frame.
    """
    from img_2_svg_pretraining.animatebench import results

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    cfg = STATE["models"][label]["cfg"]
    _paths_for(label, style)
    root = _evals_root(cfg)
    config_name = Path(STATE["models"][label]["file"]).stem
    record = results.read_record(results.suite_path(
        root, config_name, style, sample_id, "animation"))
    if record is None:
        return jsonify({"scored": False, "nodes": []})

    nodes = []
    for key, title, prompt_file, body_key, adapter_prefix, detail_key in _ANIM_NODES:
        detail = record.get(detail_key) or []
        errors = record.get(f"{key}_errors") or []
        manifest = {k: record.get(f"{key}_{k}") for k in
                    ("frame_policy", "frames_available", "frames_judged",
                     "frame_long_edge")
                    if record.get(f"{key}_{k}") is not None}
        nodes.append({
            "key": key, "title": title,
            "ran": key in (record.get("stages_run") or []),
            "skipped": (record.get("stages_skipped") or {}).get(key),
            "status": record.get("repetition_status") if key == "repetition" else None,
            "note": record.get("repetition_note") if key == "repetition" else None,
            "calls": len(detail), "errors": errors,
            "manifest": manifest,
            "prompt": ({"unspecified": True}
                       if key == "repetition"
                       and record.get("repetition_status") == "not_specified"
                       else _anim_prompt(prompt_file, body_key, adapter_prefix, style)),
            "trace": detail,
        })

    checklist = None
    cl_path = root / "checklist" / config_name / style / f"{sample_id}.json"
    if cl_path.exists():
        try:
            checklist = json.loads(cl_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checklist = None

    total = sum(n["calls"] for n in nodes) + (1 if checklist else 0)
    return jsonify({
        "scored": True, "style": style, "sample": sample_id,
        "total_calls": total,
        "would_eliminate": record.get("would_eliminate") or [],
        "unenforced_rules": record.get("ascs_unenforced_rules") or [],
        "checklist": checklist,
        "checklist_prompt": _checklist_prompt() if checklist else None,
        "nodes": nodes,
    })


@app.get("/api/render/<label>/<sample_id>/<style>")
def api_render(label, sample_id, style):
    """The compiled Stage-1 diagram, as rendering fidelity saw it.

    `stage1.json` records the render it scored, but as a container path that
    does not exist outside the container. Try it, then the same path with the
    `/code` prefix stripped, and only then recompile -- so the image shown is
    the one that was actually judged whenever it is still on disk.
    """
    from img_2_svg_pretraining.animatebench import results

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    cfg = STATE["models"][label]["cfg"]
    config_name = Path(STATE["models"][label]["file"]).stem
    paths = _paths_for(label, style)
    root = _evals_root(cfg)

    record = results.read_record(
        results.suite_path(root, config_name, style, sample_id, "stage1")) or {}
    stored = record.get("render_path")
    for candidate in _local_candidates(stored, Path(cfg.cache_root)):
        if candidate.is_file():
            return send_file(candidate, mimetype="image/png")

    try:
        code_path = paths.resolve_code(sample_id)
    except FileNotFoundError:
        abort(404, "no diagram code to render")
    from img_2_svg_pretraining.viewer.compile import compile_tikz
    result = compile_tikz(code_path.read_text(encoding="utf-8"),
                          paths.compile_cache())
    if not result.ok or not result.png_path:
        abort(404, "diagram does not compile")
    return send_file(result.png_path, mimetype="image/png")


@app.get("/api/frame/<label>/<sample_id>/<style>/<int:index>")
def api_frame(label, sample_id, style, index):
    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    frames = sorted((_paths_for(label, style).exports(sample_id)
                     / "frames").glob("*.png"))
    if not 0 <= index < len(frames):
        abort(404)
    return send_file(frames[index], mimetype="image/png")


@app.get("/api/index")
def api_index():
    return jsonify({
        "models": [
            {"label": label, "model": m["model"], "config": m["file"]}
            for label, m in STATE["models"].items()
        ],
        "styles": STATE["styles"],
        "samples": [
            {"id": s.id, "title": s.title,
             "has_reference": bool(_reference(s.id).get("videos"))}
            for s in STATE["samples"]
        ],
    })


@app.get("/api/compare/<sample_id>/<style>")
def api_compare(sample_id, style):
    if style not in STYLES:
        abort(404, f"unknown style '{style}'")
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404, f"unknown sample '{sample_id}'")

    reference = _reference(sample_id)
    ref_videos = reference.get("videos", {})

    panels = []
    for label in STATE["models"]:
        paths = _paths_for(label, style)
        exports = paths.exports(sample_id)
        mp4 = exports / "animation.mp4"
        frames_dir = exports / "frames"
        panels.append({
            "label": label,
            "model": STATE["models"][label]["model"],
            "video": f"/api/video/{label}/{sample_id}/{style}" if mp4.exists() else None,
            "frames": len(list(frames_dir.glob("*.png"))) if frames_dir.is_dir() else 0,
            "sequence": _sequence_summary(paths.sequence_narrated(sample_id))
                        or _sequence_summary(paths.sequence(sample_id)),
            "code": paths.animation(sample_id).exists(),
            "dir": str(exports),
            "metrics": _metrics(STATE["models"][label]["cfg"],
                                STATE["models"][label]["file"], style, sample_id),
        })

    # The bundle ships one reference video per (target, style, tier); the
    # full-context TikZ one is the closest match to how we run.
    ref_key = f"tikz|{style}|full"
    talk = reference.get("original_presentation")
    talk_dir = Path(sample.directory) / "reference" / "original_presentation"
    return jsonify({
        "id": sample_id,
        "title": sample.title,
        "style": style,
        "figure": f"/api/figure/{sample_id}",
        "reference": {
            "video": (f"/api/reference/{sample_id}/{ref_videos[ref_key]}"
                      if ref_key in ref_videos else None),
            "available_keys": sorted(ref_videos),
            "reviews": reference.get("reviews", []),
        },
        "presentation": ({
            "title": talk.get("title"),
            "uploader": talk.get("uploader"),
            "url": talk.get("webpage_url"),
            "duration": talk.get("duration"),
            "video": (f"/api/presentation/{sample_id}/video"
                      if (talk_dir / "video.mp4").is_file() else None),
            "transcript": ((talk_dir / "transcript.txt").read_text(encoding="utf-8")
                           if (talk_dir / "transcript.txt").is_file() else None),
        } if talk else None),
        "panels": panels,
    })


@app.get("/api/figure/<sample_id>")
def api_figure(sample_id):
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404)
    return send_file(sample.image_path)


@app.get("/api/reference/<sample_id>/<name>")
def api_reference(sample_id, name):
    sample = STATE["by_id"].get(sample_id)
    if sample is None or ".." in name or "/" in name:
        abort(404)
    path = Path(sample.directory) / "reference" / "videos" / name
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.get("/api/presentation/<sample_id>/video")
def api_presentation_video(sample_id):
    """The author's real conference talk, shipped as reference/original_presentation."""
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404)
    path = Path(sample.directory) / "reference" / "original_presentation" / "video.mp4"
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.get("/api/video/<label>/<sample_id>/<style>")
def api_video(label, sample_id, style):
    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    path = _paths_for(label, style).exports(sample_id) / "animation.mp4"
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.get("/api/sequence/<label>/<sample_id>/<style>")
def api_sequence(label, sample_id, style):
    """Full narrated sequence, in the benchmark's own dialect for comparison."""
    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    paths = _paths_for(label, style)
    path = paths.sequence_narrated(sample_id)
    if not path.exists():
        path = paths.sequence(sample_id)
    if not path.exists():
        abort(404, "no sequence")
    seq = AnimationSequence.load(path)
    return jsonify({"native": seq.to_dict(), "bench": seq.to_bench_dict()})


@app.get("/api/reference-sequence/<sample_id>/<style>")
def api_reference_sequence(sample_id, style):
    """The benchmark's own narrated sequence for this style."""
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404)
    directory = Path(sample.directory) / "reference" / "narration"
    # The two bundle generations name these differently. v2 ships
    # `<style>_<id>_<target>.json` (the convention gt.py already uses); v1
    # ships `<target>_<style>_img_and_context_<id>.json`. Try v2 first, then
    # fall back, so one viewer serves both datasets.
    matches = sorted(directory.glob(f"{style}_{sample_id}_tikz.json")) \
        or sorted(directory.glob(f"tikz_{style}_img_and_context_*.json"))
    if not matches:
        abort(404, "no reference narration")
    try:
        return jsonify(json.loads(matches[0].read_text()))
    except json.JSONDecodeError as e:
        # One file in the shipped bundle is malformed (an element missing its
        # "id" key). Report it rather than 500ing.
        return jsonify({"error": f"reference file is not valid JSON: {e}"}), 200


@app.get("/")
def index():
    return (Path(__file__).parent / "compare.html").read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dataset", default=None,
                        help="override the dataset root from the configs")
    parser.add_argument("--configs", default=None,
                        help="comma-separated label=config_file.yaml pairs, "
                             "replacing DEFAULT_CONFIGS (e.g. for a different bench)")
    args = parser.parse_args()

    config_files = DEFAULT_CONFIGS
    if args.configs:
        config_files = {}
        for pair in args.configs.split(","):
            label, _, filename = pair.partition("=")
            if not filename:
                raise SystemExit(f"--configs: bad pair '{pair}', expected label=file.yaml")
            config_files[label.strip()] = filename.strip()

    _init(config_files, args.dataset)
    print(f"{len(STATE['models'])} model(s), {len(STATE['samples'])} sample(s)")
    print(f"open http://<host>:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
