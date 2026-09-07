"""Trial assignment.

Experiment-major, as the team decided: a participant finishes all their samples
of Experiment 1 before seeing Experiment 2. The alternative -- all experiments
on one diagram before moving on -- was rejected because familiarity with a
figure biases every later judgment about it.

Diagrams may repeat ACROSS experiments (Exp2 rates narration on the same videos
Exp1 rated visually, which is what makes the two correlatable). `trial_index`
and experiment order are recorded so carry-over can be checked in analysis
rather than assumed away.

Replication works by retirement rather than by a coverage deficit: once a
sample has `judgments_per_sample` independent judgments it stops being offered
and a replacement is drawn from the same stratification class. That maximises
the number of distinct samples the study reports on, which is the point -- the
pool grows from 15 to ~100 and the participant count does not.

Every assignment runs inside one BEGIN IMMEDIATE transaction and is persisted
before it is returned, so a reload resumes the identical trial rather than
minting a new one.
"""
from __future__ import annotations

import hashlib
import json

from img_2_svg_pretraining.study.config import (
    ABSOLUTE_EXPERIMENTS, FIXED_SIDES, PAIR_AXIS, SCREEN, SHOWS_CAPTIONS,
    TOURNAMENT, StudyConfig)
from img_2_svg_pretraining.study.db import StudyDB, new_id


def _hash_int(*parts: str) -> int:
    raw = "|".join(parts).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def cell_key(experiment: str, narrative_ids: list[str]) -> str:
    """Canonical, position-independent identity for a stimulus-condition cell."""
    if len(narrative_ids) == 1:
        return f"{experiment}:{narrative_ids[0]}"
    return f"{experiment}:" + "|".join(sorted(narrative_ids))


# ----------------------------------------------------------------- pools --

def build_pool(conn, bundle_id: str, experiment: str,
               study_day: int | None = None) -> list[dict]:
    """Candidate cells for one experiment.

    With `study_day`, only diagrams the bundle stamped with that day are
    offered -- the main study runs ten fresh figures per day.

    Absolute experiments offer one narrative each. Pairwise experiments pair
    two narratives of the same (diagram, style) that differ on exactly the axis
    that experiment tests -- so an arm switches on when its data lands, with no
    code change here.
    """
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM narrative WHERE bundle_id=? AND is_attention_check=0",
        (bundle_id,)).fetchall()]
    if study_day is not None:
        todays = {r["diagram_id"] for r in conn.execute(
            "SELECT diagram_id FROM diagram WHERE bundle_id=? AND study_day=?",
            (bundle_id, study_day)).fetchall()}
        rows = [n for n in rows if n["diagram_id"] in todays]

    if experiment in ABSOLUTE_EXPERIMENTS:
        # Every generating METHOD is rated in isolation -- that is the point
        # of the main study's Experiment 1 (AnimateBanana vs each baseline on
        # the same figures). What is excluded: the human-verified bench
        # reference (it is the other side of the bench experiment, not a
        # system) and the original talk (a human presentation, not an
        # animation of the figure).
        return [{"cell_key": cell_key(experiment, [n["narrative_id"]]),
                 "diagram_id": n["diagram_id"],
                 "animation_style": n["animation_style"],
                 "narratives": [n], "conditions": [n["method"]]}
                for n in rows
                if n["verification_state"] != "verified"
                and n["method"] != "talk"
                and n["context_condition"] != "without_context"]

    if experiment in TOURNAMENT:
        # One cell per diagram holding all three contenders, in the design's
        # fixed order. A diagram missing any contender is not offered: a
        # two-way "tournament" would rank on different evidence.
        order = TOURNAMENT[experiment]
        by_diagram: dict[str, dict] = {}
        for n in rows:
            if n["method"] in order and n["context_condition"] != "without_context" \
                    and n["verification_state"] != "verified":
                by_diagram.setdefault(n["diagram_id"], {})[n["method"]] = n
        pool = []
        for diagram_id, variants in by_diagram.items():
            if all(m in variants for m in order):
                ns = [variants[m] for m in order]
                pool.append({
                    "cell_key": cell_key(experiment, [n["narrative_id"] for n in ns]),
                    "diagram_id": diagram_id,
                    "animation_style": ns[0]["animation_style"],
                    "narratives": ns, "conditions": list(order)})
        return pool

    field, (cond_a, cond_b) = PAIR_AXIS[experiment]
    by_group: dict[tuple, dict] = {}
    for n in rows:
        key = (n["diagram_id"], n["animation_style"])
        by_group.setdefault(key, {}).setdefault(n[field], n)

    pool = []
    for (diagram_id, style), variants in by_group.items():
        if cond_a in variants and cond_b in variants:
            a, b = variants[cond_a], variants[cond_b]
            pool.append({
                "cell_key": cell_key(experiment, [a["narrative_id"], b["narrative_id"]]),
                "diagram_id": diagram_id, "animation_style": style,
                "narratives": [a, b], "conditions": [cond_a, cond_b]})
    return pool


def reap_abandoned(conn, ttl_seconds: int) -> int:
    """Release trials whose participant walked away.

    In-flight trials must count toward a cell's quota, otherwise two people
    served the same cell at once both fill the last slot. But a participant who
    closes the tab leaves that trial open forever, and the cell then holds a
    reservation nobody will ever redeem -- quietly shrinking the study by one
    judgment per abandonment. Anything older than the TTL is marked abandoned
    and its slot returns to the pool.

    The row is kept, not deleted: it is evidence of a dropout, which the
    completion statistics need.
    """
    cur = conn.execute(
        "UPDATE trial SET status='abandoned'"
        " WHERE status='open'"
        "   AND created_at < datetime('now', ?)", ("-%d seconds" % ttl_seconds,))
    return cur.rowcount


def _judgment_counts(conn, experiment: str) -> dict:
    rows = conn.execute(
        "SELECT cell_key, COUNT(*) c FROM trial"
        " WHERE experiment=? AND status IN ('open','submitted') GROUP BY cell_key",
        (experiment,)).fetchall()
    return {r["cell_key"]: r["c"] for r in rows}


def _stratum(diagram: dict, style: str, fields: tuple) -> tuple:
    return tuple(diagram.get(f) for f in fields) + (style,)


# ------------------------------------------------------------- selection --

def _select_cell(pool, counts, seen_cells, diagrams, cfg, participant_id,
                 experiment, seen_diagrams=frozenset()) -> dict | None:
    """Least-judged eligible cell, breaking ties toward under-served strata.

    Retirement and stratum-matched replacement fall out of this ordering rather
    than needing a separate step: a retired cell is simply one whose count has
    reached the quota, and a fresh sample of an under-served stratum sorts
    ahead of an equally-unjudged one from an over-served stratum.
    """
    quota = cfg.judgments_per_sample
    candidates = [c for c in pool if c["cell_key"] not in seen_cells]
    if not candidates:
        return None

    # A retired cell is finished, not merely deprioritised. Returning None here
    # sends the caller on to the next experiment; over-filling instead would
    # spend a participant's attention gathering judgments we have already
    # declared unnecessary, and would quietly make `judgments_per_sample` a
    # suggestion rather than a quota.
    pool_to_use = [c for c in candidates if counts.get(c["cell_key"], 0) < quota]
    if not pool_to_use:
        return None

    served: dict[tuple, int] = {}
    for c in pool_to_use:
        st = _stratum(diagrams.get(c["diagram_id"], {}), c["animation_style"],
                      cfg.stratum_fields)
        served[st] = served.get(st, 0) + counts.get(c["cell_key"], 0)

    def sort_key(c):
        st = _stratum(diagrams.get(c["diagram_id"], {}), c["animation_style"],
                      cfg.stratum_fields)
        return (c["diagram_id"] in seen_diagrams,  # fresh figures first
                counts.get(c["cell_key"], 0),      # then fill evenly
                served.get(st, 0),                 # then under-served strata
                _hash_int(participant_id, experiment, c["cell_key"]))

    return min(pool_to_use, key=sort_key)


def _public_payload(row: dict, narratives: dict, figure: dict | None = None) -> dict:
    """What the client is allowed to see.

    Never the narrative_id, method, context condition, verification state,
    lineage or any filesystem path. Media is addressed by trial and slot, so a
    payload cannot betray which side is which.
    """
    ids = json.loads(row["presentation_ids"]) if row.get("presentation_ids") else \
          [row["presentation_a_id"]] + ([row["presentation_b_id"]] if row["presentation_b_id"] else [])
    slots = [{"slot": letter} for letter, _ in zip("ABC", ids)]
    for slot, nid in zip(slots, ids):
        payload = narratives[nid]
        slot.update({"duration": payload["timeline"]["duration"],
                     # Bare content hashes. They name bytes, not conditions --
                     # which is the whole point of addressing media by hash.
                     "frames": payload["frames"],
                     "n_frames": len(payload["frames"]),
                     "frame_w": payload.get("frame_w"),
                     "frame_h": payload.get("frame_h"),
                     "cues": payload["timeline"]["cues"] if row["show_captions"] else [],
                     "holds": payload["timeline"]["holds"]})
    return {
        "trial_id": row["trial_id"],
        "experiment": row["experiment"],
        "screen": SCREEN[row["experiment"]] if row["experiment"] in SCREEN else "absolute",
        "trial_index": row["trial_index"],
        "experiment_index": row["experiment_index"],
        "show_captions": bool(row["show_captions"]),
        "animation_style": row["animation_style"],
        "figure": figure,
        "slots": slots,
    }


def _figure(conn, row: dict) -> dict | None:
    """The source figure panel. Its media id is a content hash like any other,
    so it names bytes rather than the diagram it depicts."""
    r = conn.execute(
        "SELECT figure_media_id, figure_w, figure_h, title FROM diagram"
        " WHERE bundle_id=? AND diagram_id=?",
        (row["bundle_id"], row["diagram_id"])).fetchone()
    if not r:
        return None
    return {"media_id": r["figure_media_id"], "w": r["figure_w"],
            "h": r["figure_h"], "title": r["title"]}


def _narrative_payloads(conn, ids: list[str]) -> dict:
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT narrative_id, payload_json FROM narrative WHERE narrative_id IN ({marks})",
        ids).fetchall()
    return {r["narrative_id"]: json.loads(r["payload_json"]) for r in rows}


def next_trial(db: StudyDB, participant_id: str, cfg: StudyConfig) -> dict | None:
    """Assign (or resume) this participant's current trial.

    Returns the public payload, or None when the participant is finished.
    """
    with db._connect(immediate=True) as conn:
        participant = conn.execute(
            "SELECT * FROM participant WHERE participant_id=?",
            (participant_id,)).fetchone()
        if participant is None:
            raise KeyError(f"unknown participant {participant_id}")
        participant = dict(participant)
        bundle_id = participant["bundle_id"]
        reap_abandoned(conn, cfg.open_trial_ttl_seconds)

        # 1. Resume. A reload must never mint a new assignment.
        open_row = conn.execute(
            "SELECT * FROM trial WHERE participant_id=? AND status='open'",
            (participant_id,)).fetchone()
        if open_row:
            row = dict(open_row)
            ids = json.loads(row["presentation_ids"])
            return _public_payload(row, _narrative_payloads(conn, ids),
                                   _figure(conn, row))

        # 2. Which experiment is this participant on?
        done = {r["experiment"]: r["c"] for r in conn.execute(
            "SELECT experiment, COUNT(*) c FROM trial"
            " WHERE participant_id=? AND status='submitted' AND counts_toward_target=1"
            " GROUP BY experiment", (participant_id,)).fetchall()}

        diagrams = {r["diagram_id"]: dict(r) for r in conn.execute(
            "SELECT * FROM diagram WHERE bundle_id=?", (bundle_id,)).fetchall()}

        # 3. Walk the experiments in order and take the first that can actually
        #    serve a cell. Advancing on exhaustion -- rather than returning
        #    None -- matters: a pool can run dry before the target is met
        #    (fewer samples than `samples_per_experiment`, or retirement
        #    shrinking it), and ending the whole session there would silently
        #    truncate every later experiment.
        experiment = cell = pool = None
        for candidate in cfg.enabled_experiments():
            if done.get(candidate, 0) >= cfg.target_for(candidate):
                continue           # this participant is done with it
            candidate_pool = build_pool(conn, bundle_id, candidate, cfg.study_day)
            if not candidate_pool:
                continue           # arm has no stimuli yet; it ships dark

            prior = conn.execute(
                "SELECT cell_key, diagram_id FROM trial"
                " WHERE participant_id=? AND experiment=?",
                (participant_id, candidate)).fetchall()
            # A diagram is offered at most once per pairwise/tournament
            # experiment. In the absolute experiment the design wants the SAME
            # figure rated under every method by the same person (that is what
            # makes the per-method scores comparable), so there a diagram may
            # come back under another method -- but only after every other
            # figure has been seen once, so the two ratings are never back to
            # back. Across experiments repeats are allowed, and intended.
            seen_cells = {r["cell_key"] for r in prior}
            seen_diagrams = {r["diagram_id"] for r in prior}
            if candidate in ABSOLUTE_EXPERIMENTS:
                eligible = candidate_pool
            else:
                eligible = [c for c in candidate_pool
                            if c["diagram_id"] not in seen_diagrams]

            counts = _judgment_counts(conn, candidate)
            picked = _select_cell(eligible, counts, seen_cells, diagrams, cfg,
                                  participant_id, candidate, seen_diagrams)
            if picked is not None:
                experiment, cell, pool = candidate, picked, candidate_pool
                break

        if cell is None:
            return None            # nothing left anywhere for this participant

        # 4. Randomise A/B from a recorded seed, so the layout is reproducible
        #    and the analysis can still recover condition from position.
        trial_id = new_id()
        narratives, conditions = cell["narratives"], cell["conditions"]
        seed = str(_hash_int(trial_id, participant_id, cell["cell_key"]))
        # Pairwise sides are randomised so preference can be recovered from
        # condition rather than position -- EXCEPT where the design is
        # deliberately unblinded (bench: original left, corrected right). A
        # tournament keeps the design's order: the first two meet first.
        if len(narratives) == 2 and experiment not in FIXED_SIDES \
                and int(seed) % 2 == 1:
            narratives = [narratives[1], narratives[0]]
            conditions = [conditions[1], conditions[0]]

        trial_index = conn.execute(
            "SELECT COUNT(*) c FROM trial WHERE participant_id=?",
            (participant_id,)).fetchone()["c"]

        conn.execute("""
            INSERT INTO trial
            (trial_id, participant_id, trial_index, experiment, experiment_index,
             cell_key, diagram_id, animation_style, bundle_id,
             presentation_a_id, presentation_a_condition,
             presentation_b_id, presentation_b_condition,
             presentation_ids, presentation_conditions,
             position_seed, show_captions, assignment_reason,
             is_attention_check, counts_toward_target, config_version,
             status, served_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',datetime('now'))""", (
            trial_id, participant_id, trial_index, experiment,
            done.get(experiment, 0), cell["cell_key"], cell["diagram_id"],
            cell["animation_style"], bundle_id,
            narratives[0]["narrative_id"], conditions[0],
            narratives[1]["narrative_id"] if len(narratives) >= 2 else None,
            conditions[1] if len(conditions) >= 2 else None,
            json.dumps([n["narrative_id"] for n in narratives]),
            json.dumps(list(conditions)),
            seed, int(SHOWS_CAPTIONS.get(experiment, True)),
            json.dumps({"judgments_before": counts.get(cell["cell_key"], 0),
                        "quota": cfg.judgments_per_sample,
                        "pool_size": len(pool)}),
            0, 1, participant["config_version"]))

        # 5. Build the payload by reading the row back, so a resumed trial and
        #    a freshly served one take literally the same path.
        row = dict(conn.execute("SELECT * FROM trial WHERE trial_id=?",
                                (trial_id,)).fetchone())
        ids = json.loads(row["presentation_ids"])
        return _public_payload(row, _narrative_payloads(conn, ids),
                               _figure(conn, row))


def submit_trial(db: StudyDB, trial_id: str) -> None:
    with db._connect() as conn:
        conn.execute("UPDATE trial SET status='submitted', submitted_at=datetime('now')"
                     " WHERE trial_id=? AND status='open'", (trial_id,))


def coverage(db: StudyDB, cfg: StudyConfig, bundle_id: str) -> list[dict]:
    """Per-cell judgment counts, for the admin sample view."""
    out = []
    with db._connect() as conn:
        reap_abandoned(conn, cfg.open_trial_ttl_seconds)
        for experiment in cfg.enabled_experiments():
            counts = _judgment_counts(conn, experiment)
            for cell in build_pool(conn, bundle_id, experiment, cfg.study_day):
                n = counts.get(cell["cell_key"], 0)
                out.append({
                    "experiment": experiment, "cell_key": cell["cell_key"],
                    "diagram_id": cell["diagram_id"],
                    "animation_style": cell["animation_style"],
                    "judgments": n,
                    "quota": cfg.judgments_per_sample,
                    "status": "retired" if n >= cfg.judgments_per_sample else
                              ("active" if n else "untouched")})
    return out
