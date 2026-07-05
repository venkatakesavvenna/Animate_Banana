"""Annotation tool for correcting TikZ code against ground-truth architecture
diagram images (Stage 1: image -> TikZ), so it matches the ground truth image.

Run inside the project docker container:
    python viewer/app.py --data-root /code/data/Set-2 --port 7860

Then open http://<host>:7860 in a browser, pick a user (user_1/2/3), and
annotate. Edits are saved to a SQLite DB (viewer/annotations.db by default),
not to the original .tex files, so the pristine dataset is never overwritten.
Export finished annotations with:
    curl http://<host>:7860/api/export -o ground_truth_export.zip
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from samples import discover_samples
from compile import compile_tikz
from db import AnnotationDB, USERS

app = Flask(__name__)

VIEWER_DIR = Path(__file__).parent
CACHE_DIR = VIEWER_DIR / "cache"

STATE = {
    "data_root": None,
    "samples": [],
    "by_id": {},
    "db": None,
}


def _reload_samples():
    STATE["samples"] = discover_samples(STATE["data_root"])
    STATE["by_id"] = {s.id: s for s in STATE["samples"]}


def _get_sample(sample_id: str):
    return STATE["by_id"].get(sample_id)


def _current_user() -> str | None:
    user = request.headers.get("X-Annotator-User") or request.args.get("user")
    return user if user in USERS else None


@app.route("/")
def index():
    return send_file(VIEWER_DIR / "templates" / "index.html")


@app.route("/api/users")
def api_users():
    return jsonify(USERS)


@app.route("/api/samples")
def api_samples():
    statuses = STATE["db"].all_statuses()
    out = []
    for s in STATE["samples"]:
        d = s.to_dict()
        ann = statuses.get(s.id)
        d["annotation_status"] = ann.status if ann else "pending"
        d["annotated_by"] = ann.annotated_by if ann else None
        out.append(d)
    return jsonify(out)


@app.route("/api/progress")
def api_progress():
    progress = STATE["db"].progress()
    progress["total"] = len(STATE["samples"])
    return jsonify(progress)


@app.route("/api/samples/<path:sample_id>/tex")
def api_sample_tex(sample_id: str):
    sample = _get_sample(sample_id)
    if sample is None:
        return jsonify({"error": "sample not found"}), 404
    ann = STATE["db"].get(sample_id)
    if ann is not None:
        return jsonify({
            "tex": ann.tex,
            "original_tex": sample.tex_path.read_text(encoding="utf-8"),
            "status": ann.status,
            "annotated_by": ann.annotated_by,
        })
    original = sample.tex_path.read_text(encoding="utf-8")
    return jsonify({"tex": original, "original_tex": original, "status": "pending", "annotated_by": None})


@app.route("/api/samples/<path:sample_id>/image")
def api_sample_image(sample_id: str):
    sample = _get_sample(sample_id)
    if sample is None or sample.image_path is None:
        return jsonify({"error": "no ground-truth image"}), 404
    return send_file(sample.image_path)


@app.route("/api/samples/<path:sample_id>/rendered")
def api_sample_rendered(sample_id: str):
    """Serve the dataset's precomputed render of the ground-truth TikZ, if any.

    Avoids recompiling all 8k+ ground-truth samples just to browse them; the
    on-demand /api/compile endpoint is for edits made in the viewer.
    """
    sample = _get_sample(sample_id)
    if sample is None or sample.rendered_image_path is None:
        return jsonify({"error": "no precomputed render"}), 404
    return send_file(sample.rendered_image_path)


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
    """Save a draft edit without changing done/pending status."""
    sample = _get_sample(sample_id)
    if sample is None:
        return jsonify({"error": "sample not found"}), 404
    user = _current_user()
    if user is None:
        return jsonify({"error": "missing or invalid user"}), 400
    payload = request.get_json(force=True)
    tex_source = payload.get("tex", "")
    ann = STATE["db"].save_draft(sample_id, tex_source, user)
    return jsonify({"ok": True, **ann.to_dict()})


@app.route("/api/samples/<path:sample_id>/status", methods=["POST"])
def api_set_status(sample_id: str):
    """Mark a sample done (matches ground truth) or reopen it back to pending."""
    sample = _get_sample(sample_id)
    if sample is None:
        return jsonify({"error": "sample not found"}), 404
    user = _current_user()
    if user is None:
        return jsonify({"error": "missing or invalid user"}), 400
    payload = request.get_json(force=True)
    tex_source = payload.get("tex", "")
    status = payload.get("status")
    if status not in ("done", "pending"):
        return jsonify({"error": "status must be 'done' or 'pending'"}), 400
    ann = STATE["db"].set_status(sample_id, tex_source, user, status)
    return jsonify({"ok": True, **ann.to_dict()})


@app.route("/api/export")
def api_export():
    """Bundle every sample marked 'done' as a zip of <id>.tex files plus a
    manifest.json (annotated_by / updated_at per sample) for provenance."""
    statuses = STATE["db"].all_statuses()
    done = {sid: ann for sid, ann in statuses.items() if ann.status == "done"}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {}
        for sample_id, ann in sorted(done.items()):
            zf.writestr(f"tex_files/{sample_id}.tex", ann.tex)
            manifest[sample_id] = {"annotated_by": ann.annotated_by, "updated_at": ann.updated_at}
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="ground_truth_export.zip",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="../examples",
                         help="Directory to scan for samples (auto-detects the Set-2 "
                              "tex_files/original_images/tex_images layout, or falls back "
                              "to a generic recursive .tex scan)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--db-path", type=str, default=str(VIEWER_DIR / "annotations.db"),
                         help="SQLite DB path for annotations (default: viewer/annotations.db)")
    args = parser.parse_args()

    STATE["data_root"] = Path(args.data_root).resolve()
    STATE["db"] = AnnotationDB(Path(args.db_path).resolve())
    _reload_samples()
    print(f"Discovered {len(STATE['samples'])} samples under {STATE['data_root']}")
    print(f"Annotation DB: {Path(args.db_path).resolve()}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
