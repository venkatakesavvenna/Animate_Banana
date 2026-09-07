"""Admin surface: monitoring, coverage, and export.

A separate blueprint behind a token, deliberately -- not a query flag on the
participant routes. Admin views are *unblinded* (that is their job: inspecting
which condition a trial actually showed), so the code paths that reveal
condition must not be reachable from a participant session by flipping a
parameter. Keeping them on distinct routes with a distinct auth check means a
leak needs a deliberate mistake rather than a typo.

Two views over the same tables, as the team asked for:
  * sample     -- per diagram: stratification, judgments, retired/active
  * experiment -- across diagrams: per-question means, medians, spread
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file

from img_2_svg_pretraining.study.questions import metric_map, question_set
from img_2_svg_pretraining.study.scheduler import build_pool, coverage

admin = Blueprint("admin", __name__)

STATE: dict = {}          # injected by app._init


def _auth():
    token = STATE.get("token")
    if not token:
        abort(503, "admin token not configured")
    given = request.headers.get("X-Admin-Token") or request.args.get("token")
    if given != token:
        abort(401, "bad admin token")


def _db():
    return STATE["db"]


@admin.get("/admin")
def admin_page():
    return (Path(__file__).parent / "admin.html").read_text(encoding="utf-8")


@admin.get("/admin/api/overview")
def overview():
    _auth()
    cfg, bundle_id = STATE["cfg"], STATE["manifest"]["bundle_id"]
    with _db()._connect() as conn:
        row = lambda q, *a: conn.execute(q, a).fetchone()
        participants = row("SELECT COUNT(*) c FROM participant")["c"]
        experts = row("SELECT COUNT(*) c FROM participant WHERE is_expert=1")["c"]
        started = row("SELECT COUNT(DISTINCT participant_id) c FROM trial")["c"]
        complete = row("SELECT COUNT(*) c FROM participant WHERE stage='complete'")["c"]
        submitted = row("SELECT COUNT(*) c FROM trial WHERE status='submitted'")["c"]
        open_trials = row("SELECT COUNT(*) c FROM trial WHERE status='open'")["c"]
        responses = row("SELECT COUNT(*) c FROM response")["c"]
        revisions = row("SELECT COUNT(*) c FROM response WHERE revision>0")["c"]
        events = row("SELECT COUNT(*) c FROM event")["c"]
        by_exp = {r["experiment"]: r["c"] for r in conn.execute(
            "SELECT experiment, COUNT(*) c FROM trial WHERE status='submitted'"
            " GROUP BY experiment").fetchall()}
        by_style = {r["animation_style"]: r["c"] for r in conn.execute(
            "SELECT animation_style, COUNT(*) c FROM trial WHERE status='submitted'"
            " GROUP BY animation_style").fetchall()}
        cal = {"attempts": row("SELECT COUNT(*) c FROM calibration_attempt")["c"],
               "passed": row("SELECT COUNT(*) c FROM participant WHERE calibration_passed=1")["c"]}
        durations = [r["ms"] for r in conn.execute(
            "SELECT MAX(server_ms_since_open) ms FROM response"
            " WHERE server_ms_since_open IS NOT NULL GROUP BY trial_id").fetchall()]

        pools = {}
        for experiment in cfg.enabled_experiments():
            pools[experiment] = len(build_pool(conn, bundle_id, experiment, cfg.study_day))

    cells = coverage(_db(), cfg, bundle_id)
    retired = sum(1 for c in cells if c["status"] == "retired")
    return jsonify({
        "bundle_id": bundle_id,
        "study_version": cfg.study_version,
        "state": cfg.state,
        "config_version": STATE["config_version"],
        "participants": participants, "experts": experts, "started": started,
        "complete": complete, "trials_submitted": submitted,
        "trials_open": open_trials, "responses": responses,
        "revisions": revisions, "events": events,
        "by_experiment": by_exp, "by_style": by_style,
        "calibration": cal,
        "pool_sizes": pools,
        "cells_total": len(cells), "cells_retired": retired,
        "quota": cfg.judgments_per_sample,
        "median_trial_seconds": round(statistics.median(durations) / 1000, 1)
                                if durations else None,
        # Capacity is set by the pool, not by recruitment: once every cell has
        # its quota there is nothing left to serve, however many people arrive.
        "capacity_participants": {
            e: (n * cfg.judgments_per_sample) // max(cfg.target_for(e), 1)
            for e, n in pools.items() if n},
    })


@admin.get("/admin/api/samples")
def samples():
    """Per-diagram view: stratification plus judgments per experiment."""
    _auth()
    cfg, bundle_id = STATE["cfg"], STATE["manifest"]["bundle_id"]
    with _db()._connect() as conn:
        diagrams = [dict(r) for r in conn.execute(
            "SELECT * FROM diagram WHERE bundle_id=? ORDER BY diagram_id",
            (bundle_id,)).fetchall()]
        counts = defaultdict(lambda: defaultdict(int))
        for r in conn.execute(
                "SELECT diagram_id, experiment, COUNT(*) c FROM trial"
                " WHERE status='submitted' GROUP BY diagram_id, experiment").fetchall():
            counts[r["diagram_id"]][r["experiment"]] = r["c"]
        narratives = defaultdict(list)
        for r in conn.execute(
                "SELECT diagram_id, animation_style, method, context_condition,"
                " verification_state, n_frames, n_steps, duration,"
                " spoken_step_fraction, timing_source FROM narrative WHERE bundle_id=?",
                (bundle_id,)).fetchall():
            narratives[r["diagram_id"]].append(dict(r))

    cells = defaultdict(list)
    for c in coverage(_db(), cfg, bundle_id):
        cells[c["diagram_id"]].append(c)

    out = []
    for d in diagrams:
        d["judgments"] = dict(counts[d["diagram_id"]])
        d["narratives"] = narratives[d["diagram_id"]]
        d["cells"] = cells[d["diagram_id"]]
        d["retired"] = sum(1 for c in cells[d["diagram_id"]] if c["status"] == "retired")
        out.append(d)
    return jsonify(out)


@admin.get("/admin/api/experiments")
def experiments():
    """Across-sample view: per-question distributions for each experiment."""
    _auth()
    cfg = STATE["cfg"]
    with _db()._connect() as conn:
        rows = conn.execute("""
            SELECT t.experiment, t.animation_style, t.diagram_id,
                   t.participant_id, r.question_id, r.value, r.revision, r.trial_id
            FROM response r JOIN trial t ON t.trial_id = r.trial_id
            WHERE t.status='submitted'
            ORDER BY r.revision
        """).fetchall()

    # Latest revision per (trial, question): a revision supersedes, it does not
    # add a second observation.
    latest = {}
    for r in rows:
        latest[(r["trial_id"], r["question_id"])] = r
    grouped = defaultdict(lambda: defaultdict(list))
    seen_participants = defaultdict(set)
    seen_diagrams = defaultdict(set)
    for r in latest.values():
        grouped[r["experiment"]][r["question_id"]].append(json.loads(r["value"]))
        seen_participants[r["experiment"]].add(r["participant_id"])
        seen_diagrams[r["experiment"]].add(r["diagram_id"])

    out = []
    for experiment in cfg.enabled_experiments():
        try:
            qset = question_set(experiment)
            metrics = metric_map(experiment)
        except FileNotFoundError:
            continue
        stats = []
        for q in qset["questions"]:
            values = grouped[experiment].get(q["id"], [])
            entry = {"question_id": q["id"], "prompt": q["prompt"],
                     "type": q["type"], "metric": metrics.get(q["id"]),
                     "n": len(values)}
            numeric = [v for v in values if isinstance(v, (int, float))
                       and not isinstance(v, bool)]
            if q["type"] == "likert5" and numeric:
                entry.update({
                    "mean": round(statistics.mean(numeric), 3),
                    "median": statistics.median(numeric),
                    "sd": round(statistics.stdev(numeric), 3) if len(numeric) > 1 else 0.0,
                    "dist": {str(i): numeric.count(i) for i in range(1, 6)}})
            elif values:
                counts = defaultdict(int)
                for v in values:
                    counts[json.dumps(v)] += 1
                entry["counts"] = dict(counts)
            stats.append(entry)
        out.append({"experiment": experiment,
                    "participants": len(seen_participants[experiment]),
                    "diagrams": len(seen_diagrams[experiment]),
                    "questions": stats})
    return jsonify(out)


@admin.get("/admin/api/participants")
def participants():
    """Sessions with identity attached -- the one place the two tables meet."""
    _auth()
    with _db()._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, i.display_name, i.roll_no,
                   (SELECT COUNT(*) FROM trial t WHERE t.participant_id=p.participant_id
                      AND t.status='submitted') AS submitted,
                   (SELECT COUNT(*) FROM qc_flag q WHERE q.participant_id=p.participant_id) AS flags,
                   (SELECT COUNT(*) FROM participant_annotation a
                      WHERE a.participant_id=p.participant_id AND a.kind='exclude') AS excluded
            FROM participant p LEFT JOIN participant_pii i
              ON i.participant_id = p.participant_id
            ORDER BY p.created_at DESC""").fetchall()]
    for r in rows:
        r["background"] = json.loads(r.pop("background_json") or "{}")
    return jsonify(rows)


@admin.post("/admin/api/participants/<pid>/annotate")
def annotate(pid: str):
    """Exclusion is an annotation. It never touches a response row, so every
    result can be recomputed with and without it."""
    _auth()
    data = request.get_json(force=True, silent=True) or {}
    kind = data.get("kind", "note")
    if kind not in ("exclude", "include", "note"):
        return jsonify({"error": "bad kind"}), 400
    if not (data.get("reason") or "").strip():
        return jsonify({"error": "a reason is required"}), 400
    _db().annotate_participant(pid, kind, data["reason"].strip(),
                               data.get("author", "admin"))
    return jsonify({"ok": True})


@admin.get("/admin/api/export")
def export():
    """Versioned raw export. Identity is deliberately absent."""
    _auth()
    bundle_id = STATE["manifest"]["bundle_id"]
    tables = ["participant", "diagram", "narrative", "trial", "response",
              "event", "calibration_attempt", "qc_flag",
              "participant_annotation", "study_config"]
    out = {"bundle_id": bundle_id,
           "study_version": STATE["cfg"].study_version,
           "config_version": STATE["config_version"],
           "tables": {}}
    with _db()._connect() as conn:
        for table in tables:
            out["tables"][table] = [dict(r) for r in
                                    conn.execute("SELECT * FROM %s" % table).fetchall()]
        out["row_counts"] = {t: len(v) for t, v in out["tables"].items()}
    # participant_pii is never exported: analysis has no use for a name, and an
    # export that carries one will eventually be mailed to someone.
    return jsonify(out)
