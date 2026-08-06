"""Stage-1 critic A/B viewer: the same code, before and after review.

Shows what the diagram critic actually changed, per model and per sample:
the source figure, the render of the pre-critic code (`code_final`), the
render of the post-critic code (`code_reviewed`), the fidelity scores the
critic assigned to each, and the findings it diagnosed in between.

Both renders are produced on demand from the stored code by the same
`compile_tikz` the pipeline uses, so nothing here can show a stale or
specially-prepared image -- what you see is what that code compiles to today.

Run inside the container:
    python -m img_2_svg_pretraining.pipeline.inspector.critic_ab --port 8602

Port 8602 keeps it clear of the single-run inspector (7860) and the model
comparison viewer (8601); all three are published on the container.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, abort, jsonify, send_file

from img_2_svg_pretraining.pipeline.cache import CachePaths
from img_2_svg_pretraining.pipeline.config import load_config
from img_2_svg_pretraining.pipeline.samples import discover_samples

app = Flask(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "configs"

DEFAULT_CONFIGS = {
    "Gemini 3.6 Flash": "bench_gemini.yaml",
    "Gemma 4 31B": "bench_gemma4.yaml",
    "Qwen3.6 27B": "bench_qwen.yaml",
}

STATE: dict = {"models": {}, "samples": [], "by_id": {}}


def _init(config_files: dict[str, str], dataset_root: str | None) -> None:
    models, samples = {}, None
    seen_lineage: dict[str, str] = {}

    for label, filename in config_files.items():
        path = CONFIG_DIR / filename
        if not path.exists():
            continue
        cfg = load_config(path)
        paths = CachePaths.from_config(cfg)

        # Configs using `code_from` share another config's stage-1 artifacts,
        # so their critic results are literally the same files. Listing them
        # as separate panels would show one result three times and imply three
        # independent confirmations. Fold them into the owner instead.
        lineage = paths.code_lineage
        if lineage in seen_lineage:
            models[seen_lineage[lineage]]["shared_with"].append(label)
            continue
        seen_lineage[lineage] = label

        models[label] = {
            "cfg": cfg,
            "paths": paths,
            "model": cfg.backend_model(cfg.agent("code_converter").backend),
            "shared_with": [],
        }
        if samples is None:
            samples = discover_samples(dataset_root or cfg.dataset_root)

    if not models:
        raise SystemExit("no configs could be loaded")
    STATE.update(models=models, samples=samples or [],
                 by_id={s.id: s for s in (samples or [])})


def _report(paths: CachePaths, sample_id: str) -> dict | None:
    """The critic's own record of what it did for this sample."""
    path = paths.code_reviewed(sample_id).parent / f"{sample_id}.critic.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _render(paths: CachePaths, code_path: Path):
    """Compile one version on demand; None when it does not compile."""
    from img_2_svg_pretraining.viewer.compile import compile_tikz

    if not code_path.exists():
        return None
    result = compile_tikz(code_path.read_text(encoding="utf-8"), paths.compile_cache())
    return result.png_path if result.ok else None


@app.get("/api/index")
def api_index():
    out = []
    for label, m in STATE["models"].items():
        paths: CachePaths = m["paths"]
        ready = [s.id for s in STATE["samples"]
                 if paths.code_reviewed(s.id).exists()]
        out.append({"label": label, "model": m["model"], "reviewed": ready})
    return jsonify({
        "models": out,
        "samples": [{"id": s.id, "title": s.title} for s in STATE["samples"]],
    })


@app.get("/api/compare/<sample_id>")
def api_compare(sample_id):
    if sample_id not in STATE["by_id"]:
        abort(404, f"unknown sample '{sample_id}'")

    panels = []
    for label, m in STATE["models"].items():
        paths: CachePaths = m["paths"]
        report = _report(paths, sample_id)
        before_path = paths.code_final(sample_id)
        if not before_path.exists():
            before_path = paths.code(sample_id)
        after_path = paths.code_reviewed(sample_id)

        rounds = (report or {}).get("rounds") or []
        panels.append({
            "label": label,
            "model": m["model"],
            "shared_with": m.get("shared_with", []),
            "has_before": before_path.exists(),
            "has_after": after_path.exists(),
            "baseline_score": (report or {}).get("baseline_score"),
            "final_score": (report or {}).get("final_score"),
            "rounds_run": (report or {}).get("rounds_run"),
            "improved": (report or {}).get("improved"),
            "skipped": (report or {}).get("skipped"),
            # Findings from the first round that produced any: what the critic
            # said was wrong with the original.
            "findings": next((r["findings"] for r in rounds if r.get("findings")), []),
            "round_notes": [
                {"round": r.get("round"), "score": r.get("score"),
                 "notes": r.get("notes", "")} for r in rounds],
        })

    return jsonify({
        "id": sample_id,
        "title": STATE["by_id"][sample_id].title,
        "figure": f"/api/figure/{sample_id}",
        "panels": panels,
    })


@app.get("/api/figure/<sample_id>")
def api_figure(sample_id):
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404)
    return send_file(sample.image_path)


@app.get("/api/render/<label>/<sample_id>/<variant>")
def api_render(label, sample_id, variant):
    """Render `before` (pre-critic) or `after` (post-critic) on demand."""
    model = STATE["models"].get(label)
    if model is None or variant not in ("before", "after"):
        abort(404)
    paths: CachePaths = model["paths"]

    if variant == "after":
        code_path = paths.code_reviewed(sample_id)
    else:
        code_path = paths.code_final(sample_id)
        if not code_path.exists():
            code_path = paths.code(sample_id)

    png = _render(paths, code_path)
    if png is None:
        abort(404, "does not compile" if code_path.exists() else "no code")
    return send_file(png)


@app.get("/api/code/<label>/<sample_id>/<variant>")
def api_code(label, sample_id, variant):
    model = STATE["models"].get(label)
    if model is None or variant not in ("before", "after"):
        abort(404)
    paths: CachePaths = model["paths"]
    code_path = (paths.code_reviewed(sample_id) if variant == "after"
                 else paths.code_final(sample_id))
    if not code_path.exists():
        abort(404)
    return jsonify({"code": code_path.read_text(encoding="utf-8"),
                    "path": str(code_path)})


@app.get("/")
def index():
    return (Path(__file__).parent / "critic_ab.html").read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8602)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    _init(DEFAULT_CONFIGS, args.dataset)
    print(f"{len(STATE['models'])} model(s), {len(STATE['samples'])} sample(s)")
    print(f"open http://<host>:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
