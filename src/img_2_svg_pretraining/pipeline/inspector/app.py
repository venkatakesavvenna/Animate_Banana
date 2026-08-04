"""Pipeline inspector -- browse every intermediate artifact of a run.

Each pipeline agent writes its output to disk, which makes the whole run
inspectable after the fact. This serves those artifacts side by side so a
sample can be followed from source figure through code, structure XML,
strategy, sequence, animation, and finished exports.

Run inside the project docker container:
    python -m img_2_svg_pretraining.pipeline.inspector.app

Then open http://<host>:7860.

Defaults to 7860 because that is the only port the container currently
publishes that isn't taken by the annotation tool's Streamlit ports
(8600-8602). The annotation viewer also uses 7860, so run only one at a
time. To use a different port it must be published at container *creation*
-- `docker run -p` has no effect on an existing container, so switching
means `docker rm -f` plus a re-run of docker/init.sh (safe: code, data and
venvs all live on mounts).
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

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"

STATE: dict = {"cfg": None, "paths": None, "samples": [], "by_id": {}}


def _init(config_path: str) -> None:
    cfg = load_config(config_path)
    samples = discover_samples(cfg.dataset_root)
    STATE.update(
        cfg=cfg,
        paths=CachePaths.from_config(cfg),
        samples=samples,
        by_id={s.id: s for s in samples},
    )


def _sample(sample_id: str):
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404, f"unknown sample '{sample_id}'")
    return sample


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except OSError:
        return None


def _stage_rows(sample_id: str) -> list[dict]:
    """One row per artifact, in pipeline order, with presence flags.

    Absent artifacts are still listed so a gap in the run is visible rather
    than silently missing from the UI.
    """
    p: CachePaths = STATE["paths"]
    rows = [
        ("1a", "Diagram code", p.code(sample_id), "tex"),
        ("1b", "Code + rasters", p.code_final(sample_id), "tex"),
        ("2a", "Strategy", p.strategy(sample_id), "json"),
        ("2b", "Structure XML", p.xml(sample_id), "xml"),
        ("2c", "Sequence", p.sequence(sample_id), "json"),
        ("2d", "Sequence (final)", p.sequence_final(sample_id), "json"),
        ("3a", "Animation code", p.animation(sample_id), "tex"),
        ("3c", "Animation (final)", p.animation_final(sample_id), "tex"),
    ]
    return [
        {"stage": stage, "label": label, "kind": kind,
         "path": str(path), "exists": path.exists(),
         "size": path.stat().st_size if path.exists() else 0}
        for stage, label, path, kind in rows
    ]


def _export_files(sample_id: str) -> dict:
    directory = STATE["paths"].exports(sample_id)
    if not directory.is_dir():
        return {"dir": str(directory), "files": [], "frames": 0}
    files = [
        {"name": f.name, "size": f.stat().st_size}
        for f in sorted(directory.iterdir()) if f.is_file()
    ]
    frames_dir = directory / "frames"
    frames = len(list(frames_dir.glob("*.png"))) if frames_dir.is_dir() else 0
    return {"dir": str(directory), "files": files, "frames": frames}


# -- API -----------------------------------------------------------------

@app.get("/api/samples")
def api_samples():
    out = []
    for sample in STATE["samples"]:
        rows = _stage_rows(sample.id)
        out.append({
            "id": sample.id,
            "tiers": sample.available_tiers(),
            "title": sample.title,
            "done": sum(1 for r in rows if r["exists"]),
            "total": len(rows),
            "frames": _export_files(sample.id)["frames"],
        })
    cfg = STATE["cfg"]
    return jsonify({
        "samples": out,
        "config": {
            "path": str(cfg.path), "style": cfg.style, "target": cfg.target,
            "dataset_root": str(cfg.dataset_root),
            "fingerprint": cfg.fingerprint(),
            "agents": {name: {"backend": a.backend,
                              "model": cfg.backend_model(a.backend) if a.backend else None}
                       for name, a in cfg.agents.items()},
        },
    })


@app.get("/api/sample/<sample_id>")
def api_sample(sample_id):
    sample = _sample(sample_id)
    rows = _stage_rows(sample_id)
    for row in rows:
        row["content"] = _read(Path(row["path"])) if row["exists"] else None

    sequence = None
    final = STATE["paths"].sequence_final(sample_id)
    raw = _read(final) or _read(STATE["paths"].sequence(sample_id))
    if raw:
        try:
            sequence = json.loads(raw)
        except json.JSONDecodeError:
            pass

    return jsonify({
        "id": sample_id,
        "title": sample.title,
        "caption": sample.caption,
        "abstract": (sample.abstract or "")[:4000],
        "tiers": sample.available_tiers(),
        "stages": rows,
        "sequence": sequence,
        "exports": _export_files(sample_id),
    })


@app.get("/api/figure/<sample_id>")
def api_figure(sample_id):
    return send_file(_sample(sample_id).image_path)


@app.get("/api/render/<sample_id>")
def api_render(sample_id):
    """Compiled render of the static diagram code, built on demand."""
    from img_2_svg_pretraining.viewer.compile import compile_tikz

    code = _read(STATE["paths"].code_final(sample_id)) or _read(STATE["paths"].code(sample_id))
    if not code:
        abort(404, "no diagram code")
    result = compile_tikz(code, STATE["paths"].compile_cache())
    if not result.ok or result.png_path is None:
        abort(422, "diagram code does not compile")
    return send_file(result.png_path)


def _frames(sample_id: str) -> list[Path]:
    frames_dir = STATE["paths"].exports(sample_id) / "frames"
    return sorted(frames_dir.glob("*.png"),
                  key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))


@app.get("/api/frame/<sample_id>/<int:index>")
def api_frame(sample_id, index):
    frames = _frames(sample_id)
    if not frames:
        abort(404, "no frames")
    return send_file(frames[max(0, min(index, len(frames) - 1))])


@app.get("/api/thumb/<sample_id>/<int:index>")
def api_thumb(sample_id, index):
    """Downscaled frame for the contact sheet.

    Frames render at 300 dpi and run to tens per sample, so serving them
    full-size would mean tens of megabytes to show one grid. Thumbnails are
    cached next to the frames since they only change when the frames do.
    """
    import io

    from PIL import Image

    frames = _frames(sample_id)
    if not frames:
        abort(404, "no frames")
    source = frames[max(0, min(index, len(frames) - 1))]

    cache_dir = source.parent / "thumbs"
    cached = cache_dir / f"{source.stem}.jpg"
    if not cached.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(source).convert("RGB")
        image.thumbnail((420, 420), Image.LANCZOS)
        image.save(cached, "JPEG", quality=80)
    return send_file(cached)


@app.get("/api/frames/<sample_id>")
def api_frames(sample_id):
    frames = _frames(sample_id)
    return jsonify({
        "count": len(frames),
        "dir": str(STATE["paths"].exports(sample_id) / "frames"),
        "names": [f.name for f in frames],
    })


@app.get("/api/export/<sample_id>/<name>")
def api_export(sample_id, name):
    path = STATE["paths"].exports(sample_id) / name
    # Keep traversal inside the export dir.
    if ".." in name or "/" in name or not path.is_file():
        abort(404)
    return send_file(path)


@app.get("/api/rasters/<sample_id>")
def api_rasters(sample_id):
    """Stage-1b results: what was proposed, what verification made of it,
    and which crop landed in which placeholder."""
    directory = STATE["paths"].rasters(sample_id)
    detections = directory / "detections.json"
    if not detections.exists():
        return jsonify({"available": False})

    try:
        data = json.loads(detections.read_text())
    except (OSError, json.JSONDecodeError):
        return jsonify({"available": False})

    return jsonify({
        "available": True,
        "dir": str(directory),
        "proposals": data.get("proposals", 0),
        "accepted": data.get("accepted", 0),
        "placeholders": data.get("placeholders", []),
        "replaced": data.get("replaced", []),
        "verified": data.get("verified", []),
        "overlays": [p.name for p in sorted(directory.glob("*.png"))
                     if not p.name.startswith("crop_")],
        "crops": [p.name for p in sorted(directory.glob("crop_*.png"))],
    })


@app.get("/api/raster-asset/<sample_id>/<name>")
def api_raster_asset(sample_id, name):
    path = STATE["paths"].rasters(sample_id) / name
    if ".." in name or "/" in name or not path.is_file():
        abort(404)
    return send_file(path)


@app.get("/api/marked/<sample_id>")
def api_marked(sample_id):
    """Set-of-Mark overlay the sequence critic saw, if one was produced."""
    directory = STATE["paths"].root / "marked" / STATE["paths"].sequence_lineage
    marks = sorted(directory.glob(f"{sample_id}.round*.png"))
    if not marks:
        abort(404, "no marked figure")
    return send_file(marks[-1])


@app.get("/")
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    _init(args.config)
    print(f"{len(STATE['samples'])} samples | config {args.config}")
    print(f"open http://<host>:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
