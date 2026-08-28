"""Study server.

Deliberately thin: routes validate, delegate, and serialise. Assignment lives in
`scheduler`, persistence in `db`, question text in `questions/*.yaml`. Nothing
here imports the pipeline -- stimuli arrive through a frozen bundle.

    python -m img_2_svg_pretraining.study.app \
        --bundle data/study_bundles/pilot-v1 \
        --config src/img_2_svg_pretraining/pipeline/configs/study.yaml \
        --db data/study_bundles/pilot-v1/study.db --port 8604

Follows the annotation tool's conventions: HTML served as raw text, JS served
per route with an explicit mimetype, no Jinja, no static folder, no build step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from img_2_svg_pretraining.study import calibration
from img_2_svg_pretraining.study.admin import admin as admin_bp, STATE as ADMIN_STATE
from img_2_svg_pretraining.study.config import SCREEN, StudyConfig
from img_2_svg_pretraining.study.db import StudyDB
from img_2_svg_pretraining.study.questions import question_set, required_ids
from img_2_svg_pretraining.study.scheduler import (
    build_pool, coverage, next_trial, submit_trial)

app = Flask(__name__)

STATE: dict = {"db": None, "cfg": None, "bundle": None, "manifest": None,
               "config_version": None, "styles": {}}

HERE = Path(__file__).parent

# Shown to the participant so "does it follow the stated style?" is answerable.
# Copied here rather than imported from `pipeline.styles` to keep the runtime
# free of that dependency; the text is frozen into the bundle-facing config.
STYLE_TEXT = {
    "progressive_reveal":
        "Elements appear one at a time, building the figure up piece by piece. "
        "Nothing that has appeared is removed.",
    "colour_pop":
        "The whole figure is visible but muted; the part being discussed is "
        "picked out in colour.",
    "alpha_masking":
        "The whole figure is visible but faded; the part being discussed is "
        "brought to full opacity.",
    "hopping_bounding_box":
        "A box jumps from element to element, framing whichever part is being "
        "discussed.",
    "sliding_bounding_box":
        "A box glides smoothly from element to element, framing whichever part "
        "is being discussed.",
}


def _init(bundle: str, config_path: str, db_path: str) -> None:
    # Absolute, deliberately: Flask resolves a relative path passed to
    # send_file against the app's root_path (this package directory), not the
    # process CWD -- so a relative bundle passes Path.exists() here and then
    # 500s inside send_file, looking under study/data/... instead.
    bundle_dir = Path(bundle).resolve()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = StudyConfig.load(config_path)
    db = StudyDB(Path(db_path))
    db.import_bundle(manifest)

    active = db.active_config()
    # A configuration change starts a new version rather than editing the old
    # one, so a trial already collected keeps pointing at the rules it ran under.
    if active is None or active["params"] != cfg.to_dict():
        version = db.put_config(cfg.to_dict(), cfg.state, manifest["bundle_id"],
                                note="loaded from %s" % config_path)
    else:
        version = active["config_version"]

    STATE.update({"db": db, "cfg": cfg, "bundle": bundle_dir,
                  "manifest": manifest, "config_version": version})
    ADMIN_STATE.update(STATE)


def _db() -> StudyDB:
    return STATE["db"]


def _participant_or_401():
    pid = request.headers.get("X-Participant") or request.args.get("pid")
    if not pid:
        abort(401, "no participant")
    if _db().participant(pid) is None:
        abort(401, "unknown participant")
    return pid


# ------------------------------------------------------------------ pages --

@app.get("/")
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/study")
def study_page():
    return (HERE / "trial.html").read_text(encoding="utf-8")


@app.get("/prepare")
def prepare_page():
    return (HERE / "prepare.html").read_text(encoding="utf-8")


@app.get("/study.js")
def study_js():
    return send_file(HERE / "study.js", mimetype="application/javascript")


@app.get("/player.js")
def player_js():
    return send_file(HERE / "player.js", mimetype="application/javascript")


@app.get("/forms.js")
def forms_js():
    return send_file(HERE / "forms.js", mimetype="application/javascript")


# ------------------------------------------------------------------ media --

@app.get("/media/<media_id>")
def media(media_id: str):
    """Content-addressed. The id names bytes, never a condition.

    Looked up rather than joined onto a path: a request cannot reach outside
    the bundle whatever it puts in the URL.
    """
    if not media_id.isalnum() or len(media_id) > 32:
        abort(404)
    path = STATE["bundle"] / "media" / f"{media_id}.webp"
    if not path.exists():
        abort(404)
    response = send_file(path, mimetype="image/webp", conditional=True)
    # The URL changes when the bytes change, so caching is safe and wanted --
    # a participant replaying a 60s animation must not refetch every frame.
    response.headers["Cache-Control"] = "private, max-age=86400, immutable"
    return response


# ------------------------------------------------------------------- api --

@app.post("/api/register")
def api_register():
    data = request.get_json(force=True, silent=True) or {}
    education = (data.get("education_level") or "").strip()
    if education not in ("student", "faculty", "researcher", "other"):
        return jsonify({"error": "education_level required"}), 400

    pid = _db().create_participant(
        education_level=education,
        config_version=STATE["config_version"],
        bundle_id=STATE["manifest"]["bundle_id"],
        display_name=(data.get("display_name") or "").strip(),
        roll_no=(data.get("roll_no") or "").strip(),
        is_expert=bool(data.get("is_expert")),
        background={"area": data.get("area", ""),
                    "reads_papers": data.get("reads_papers", "")})
    _db().set_stage(pid, "consented" if data.get("consent") else "registered")
    return jsonify({"participant_id": pid})


@app.get("/api/state")
def api_state():
    pid = _participant_or_401()
    p = _db().participant(pid)
    cfg = STATE["cfg"]
    bundle_id = STATE["manifest"]["bundle_id"]
    with _db()._connect() as conn:
        done = {r["experiment"]: r["c"] for r in conn.execute(
            "SELECT experiment, COUNT(*) c FROM trial WHERE participant_id=?"
            " AND status='submitted' GROUP BY experiment", (pid,)).fetchall()}

        # Count only what this participant can actually be served. An arm with
        # no stimuli is skipped by the scheduler, so including it in the target
        # would park the progress bar at 40% and read as a broken session
        # rather than a finished one. Retirement is bounded the same way: a
        # pool that has run dry cannot supply the full per-experiment quota.
        total_target = 0
        live = []
        for experiment in cfg.enabled_experiments():
            pool = build_pool(conn, bundle_id, experiment)
            if not pool:
                continue
            live.append(experiment)
            # Bound by DISTINCT DIAGRAMS, not by cells. A participant sees each
            # diagram at most once per experiment, so an arm with 10 cells over
            # 5 figures (e.g. +K and -K narratives of the same diagram) can
            # only ever serve 5 -- counting cells parks the progress bar at 67%
            # on a session that is actually finished.
            reachable = len({c["diagram_id"] for c in pool})
            total_target += min(cfg.target_for(experiment), reachable)
    return jsonify({
        "participant_id": pid,
        "stage": p["stage"],
        "calibration_passed": bool(p["calibration_passed"]),
        "completed": sum(done.values()),
        "target": total_target,
        "live_experiments": live,
        "per_experiment": done,
    })


@app.get("/api/prepare")
def api_prepare():
    """Worked examples plus the calibration items, if a key exists yet."""
    pid = _participant_or_401()
    bundle_id = STATE["manifest"]["bundle_id"]
    ready = calibration.available(_db(), bundle_id)
    items = (calibration.build_items(_db(), bundle_id, STATE["cfg"], STYLE_TEXT)
             if ready else [])

    # The expected answers are the marking key. Sending them to the browser
    # would let a participant read the answers out of devtools, so only the
    # stimulus and the questions go over the wire.
    public = []
    for item in items:
        payload = _slot_payload(item)
        payload["questions"] = item["questions"]
        payload["cell_key"] = item["cell_key"]
        public.append(payload)

    return jsonify({
        "calibration_available": ready,
        "threshold": STATE["cfg"].calibration_pass_threshold,
        "items": public,
        "passed": bool(_db().participant(pid)["calibration_passed"]),
    })


def _slot_payload(item: dict) -> dict:
    """Stimulus for one calibration item, in the shape the player expects."""
    ids = [item["presentation_a_id"]] + (
        [item["presentation_b_id"]] if item["presentation_b_id"] else [])
    with _db()._connect() as conn:
        marks = ",".join("?" * len(ids))
        rows = {r["narrative_id"]: json.loads(r["payload_json"]) for r in conn.execute(
            f"SELECT narrative_id, payload_json FROM narrative"
            f" WHERE narrative_id IN ({marks})", ids).fetchall()}
        fig = conn.execute(
            "SELECT figure_media_id, figure_w, figure_h, title FROM diagram"
            " WHERE bundle_id=? AND diagram_id=?",
            (STATE["manifest"]["bundle_id"], item["diagram_id"])).fetchone()

    slots = []
    for letter, nid in zip(("A", "B"), ids):
        n = rows[nid]
        slots.append({"slot": letter, "frames": n["frames"],
                      "n_frames": len(n["frames"]),
                      "frame_w": n.get("frame_w"), "frame_h": n.get("frame_h"),
                      "duration": n["timeline"]["duration"],
                      "holds": n["timeline"]["holds"],
                      "cues": n["timeline"]["cues"] if item["show_captions"] else []})
    style = item["animation_style"]
    return {"experiment": item["experiment"], "screen": SCREEN.get(item["experiment"], "absolute"),
            "show_captions": item["show_captions"], "slots": slots,
            "style_name": style.replace("_", " "),
            "style_description": STYLE_TEXT.get(style, ""),
            "figure": {"media_id": fig["figure_media_id"], "w": fig["figure_w"],
                       "h": fig["figure_h"], "title": fig["title"]} if fig else None}


@app.post("/api/calibration/submit")
def api_calibration_submit():
    pid = _participant_or_401()
    data = request.get_json(force=True, silent=True) or {}
    bundle_id = STATE["manifest"]["bundle_id"]
    cfg = STATE["cfg"]

    if not calibration.available(_db(), bundle_id):
        return jsonify({"error": "calibration unavailable"}), 409

    items = calibration.build_items(_db(), bundle_id, cfg, STYLE_TEXT)
    result = calibration.score(items, data.get("answers", {}))
    passed = result["score"] >= cfg.calibration_pass_threshold

    with _db()._connect(immediate=True) as conn:
        attempt = conn.execute(
            "SELECT COUNT(*) c FROM calibration_attempt WHERE participant_id=?",
            (pid,)).fetchone()["c"]
        conn.execute("""INSERT INTO calibration_attempt
            (participant_id, attempt_index, answers_json, score, threshold, passed)
            VALUES (?,?,?,?,?,?)""",
            (pid, attempt, json.dumps(data.get("answers", {})), result["score"],
             cfg.calibration_pass_threshold, int(passed)))
        conn.execute("UPDATE participant SET calibration_attempts=?,"
                     " calibration_passed=? WHERE participant_id=?",
                     (attempt + 1, int(passed), pid))
    if passed:
        _db().set_stage(pid, "in_study")
    return jsonify({"passed": passed, "score": result["score"],
                    "threshold": cfg.calibration_pass_threshold,
                    "scored": result["scored"], "agreed": result["agreed"],
                    "attempt": attempt + 1})


@app.get("/api/trial/current")
def api_trial_current():
    """Assign or resume. The trial is persisted before this returns."""
    pid = _participant_or_401()
    cfg = STATE["cfg"]

    # Calibration gates the main study -- but only when it can actually score
    # anyone. With no expert session recorded there is no key, and refusing
    # entry would block the whole study on missing data.
    # Experts are exempt. They are recording the marking key, so gating them on
    # it means the moment one of them submits a trial they are locked out by
    # their own answers.
    participant = _db().participant(pid)
    if (not participant["is_expert"]
            and not participant["calibration_passed"]
            and calibration.available(_db(), STATE["manifest"]["bundle_id"])):
        return jsonify({"needs_calibration": True})
    trial = next_trial(_db(), pid, cfg)
    if trial is None:
        _db().set_stage(pid, "complete")
        return jsonify({"done": True})

    # The style is deliberately shown -- Experiment 1 asks whether the
    # animation follows it, and both sides of a pairwise trial always share it,
    # so it discloses nothing about condition. What does go is the raw slug:
    # it is a lineage path component, and keeping cache-shaped tokens out of
    # the client is what stops a future "just add it for debugging" edit from
    # reintroducing a real leak.
    style = trial.pop("animation_style")
    trial["style_name"] = style.replace("_", " ")
    trial["style_description"] = STYLE_TEXT.get(style, "")
    trial["questions"] = question_set(
        trial["experiment"], style_name=style,
        style_description=STYLE_TEXT.get(style, ""))

    # Familiarity is a property of the DIAGRAM's topic, not of the trial. A
    # participant meets the same figure again in later sections, and asking
    # every time is both repetitive and meaningless -- the answer cannot change.
    # Carry the earlier answer forward and drop the question.
    with _db()._connect() as conn:
        # Look the diagram up from the trial row rather than carrying it on the
        # public payload: an internal field threaded through the payload and
        # deleted afterwards leaks from every other caller of next_trial().
        prior = conn.execute("""
            SELECT r.value FROM response r
            JOIN trial t ON t.trial_id = r.trial_id
            WHERE r.participant_id = ? AND r.question_id = 'familiarity'
              AND t.diagram_id = (SELECT diagram_id FROM trial WHERE trial_id = ?)
            ORDER BY r.response_id DESC LIMIT 1""",
            (pid, trial["trial_id"])).fetchone()
    if prior:
        trial["questions"]["familiarity"] = None
        trial["familiarity_carried"] = json.loads(prior["value"])
        _db().add_response(trial["trial_id"], pid, "familiarity",
                           json.loads(prior["value"]))

    _db().add_event(trial["trial_id"], pid, "trial_open")
    return jsonify(trial)


@app.post("/api/trial/<trial_id>/answer")
def api_answer(trial_id: str):
    """Append one answer. A revision is a new row, never an overwrite."""
    pid = _participant_or_401()
    data = request.get_json(force=True, silent=True) or {}
    question_id = data.get("question_id")
    if not question_id or "value" not in data:
        return jsonify({"error": "question_id and value required"}), 400

    with _db()._connect() as conn:
        row = conn.execute("SELECT participant_id, status FROM trial WHERE trial_id=?",
                           (trial_id,)).fetchone()
    if row is None or row["participant_id"] != pid:
        abort(404)
    if row["status"] != "open":
        return jsonify({"error": "trial already submitted"}), 409

    _db().add_response(trial_id, pid, question_id, data["value"],
                       data.get("ms_since_open"))
    return jsonify({"ok": True, "answers": _db().latest_answers(trial_id)})


@app.post("/api/trial/<trial_id>/submit")
def api_submit(trial_id: str):
    pid = _participant_or_401()
    with _db()._connect() as conn:
        row = conn.execute("SELECT * FROM trial WHERE trial_id=?", (trial_id,)).fetchone()
    if row is None or row["participant_id"] != pid:
        abort(404)
    if row["status"] != "open":
        return jsonify({"ok": True, "already": True})

    answered = set(_db().latest_answers(trial_id))
    missing = [q for q in required_ids(row["experiment"])
               if q not in answered] + (["familiarity"] if "familiarity" not in answered
                                        else [])
    if missing:
        # The client disables submit, but the client is a suggestion; this is
        # the rule.
        return jsonify({"error": "incomplete", "missing": missing}), 409

    _db().add_event(trial_id, pid, "trial_submit")
    submit_trial(_db(), trial_id)
    return jsonify({"ok": True})


@app.post("/api/events")
def api_events():
    pid = _participant_or_401()
    data = request.get_json(force=True, silent=True) or {}
    for e in data.get("events", [])[:200]:
        if not e.get("trial_id") or not e.get("type"):
            continue
        _db().add_event(e["trial_id"], pid, e["type"], slot=e.get("slot"),
                        client_seq=e.get("client_seq"), t_video=e.get("t_video"),
                        t_wall_client=e.get("t_wall_client"),
                        payload=e.get("payload"))
    return jsonify({"ok": True})


@app.get("/api/coverage")
def api_coverage():
    return jsonify(coverage(_db(), STATE["cfg"], STATE["manifest"]["bundle_id"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--admin-token", default=None,
                    help="enables /admin. Without it the admin surface refuses "
                         "every request rather than defaulting to open.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8604)
    args = ap.parse_args()

    _init(args.bundle, args.config, args.db)
    ADMIN_STATE["token"] = args.admin_token
    app.register_blueprint(admin_bp)
    manifest = STATE["manifest"]
    print(f"bundle {manifest['bundle_id']}: {len(manifest['diagrams'])} diagrams, "
          f"{len(manifest['narratives'])} narratives")
    print(f"config version {STATE['config_version']}, "
          f"experiments {STATE['cfg'].enabled_experiments()}")
    print(f"http://localhost:{args.port}/")
    print(f"admin: {'http://localhost:%d/admin' % args.port if args.admin_token else 'disabled (no --admin-token)'}")
    # threaded: a participant preloading frames must not block another's
    # assignment.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
