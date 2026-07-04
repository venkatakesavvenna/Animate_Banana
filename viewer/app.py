"""Rudimentary viewer for scrutinizing/correcting TikZ code against ground-truth
architecture diagram images (Stage 1: image -> TikZ).

Run inside the project docker container:
    python viewer/app.py --data-root examples --port 7860

Then open http://<host>:7860 in a browser.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from samples import discover_samples
from compile import compile_tikz

app = Flask(__name__)

VIEWER_DIR = Path(__file__).parent
CACHE_DIR = VIEWER_DIR / "cache"

STATE = {
    "data_root": None,
    "samples": [],
}


def _reload_samples():
    STATE["samples"] = discover_samples(STATE["data_root"])


@app.route("/")
def index():
    return send_file(VIEWER_DIR / "templates" / "index.html")


@app.route("/api/samples")
def api_samples():
    return jsonify([s.to_dict() for s in STATE["samples"]])


@app.route("/api/samples/<path:sample_id>/tex")
def api_sample_tex(sample_id: str):
    sample = next((s for s in STATE["samples"] if s.id == sample_id), None)
    if sample is None:
        return jsonify({"error": "sample not found"}), 404
    return jsonify({"tex": sample.tex_path.read_text(encoding="utf-8")})


@app.route("/api/samples/<path:sample_id>/image")
def api_sample_image(sample_id: str):
    sample = next((s for s in STATE["samples"] if s.id == sample_id), None)
    if sample is None or sample.image_path is None:
        return jsonify({"error": "no ground-truth image"}), 404
    return send_file(sample.image_path)

@app.route("/api/compile", methods=["POST"])
def api_compile():
    payload = request.get_json(force=True)
    tex_source = payload.get("tex", "")
    result = compile_tikz(tex_source, CACHE_DIR)
    if not result.ok:
        return jsonify({"ok": False, "log": result.log})
    return jsonify({"ok": True, "png_url": f"/api/render/{result.png_path.name}"})


@app.route("/api/render/<path:filename>")
def api_render(filename: str):
    png_path = CACHE_DIR / filename
    if not png_path.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(png_path)


@app.route("/api/samples/<path:sample_id>/save", methods=["POST"])
def api_save(sample_id: str):
    sample = next((s for s in STATE["samples"] if s.id == sample_id), None)
    if sample is None:
        return jsonify({"error": "sample not found"}), 404
    payload = request.get_json(force=True)
    tex_source = payload.get("tex", "")
    sample.tex_path.write_text(tex_source, encoding="utf-8")
    return jsonify({"ok": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="../examples",
                         help="Directory to scan recursively for .tex files and paired images")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    STATE["data_root"] = Path(args.data_root).resolve()
    _reload_samples()
    print(f"Discovered {len(STATE['samples'])} samples under {STATE['data_root']}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
