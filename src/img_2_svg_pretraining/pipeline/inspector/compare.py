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

    root = results.evals_root(cfg.cache_root, cfg.dataset_root.name)
    config_name = Path(config_file).stem

    suites: list[dict] = []
    for suite in ("stage1", "xml", "sequence", "stage3"):
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
                  ("sequence", "Sequence"), ("stage3", "Animation")]


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
    matches = sorted(directory.glob(f"tikz_{style}_img_and_context_*.json"))
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
