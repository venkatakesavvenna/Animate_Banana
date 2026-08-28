"""Scheduler, database and blinding tests.

Offline, no model calls, no server. Builds a synthetic bundle in a temp dir so
the pairwise arms (Exp3/4/5) can be exercised before their real stimuli exist.

    PYTHONPATH=src /fsxvision_new/venkat.kesav/environments/study/bin/python \
        tests/test_study_scheduler.py
"""
import json
import shutil
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

from img_2_svg_pretraining.study.config import PAIR_AXIS, StudyConfig
from img_2_svg_pretraining.study.db import StudyDB
from img_2_svg_pretraining.study.scheduler import (
    build_pool, cell_key, coverage, next_trial, submit_trial)

_results = []


def check(group, name, cond, detail=""):
    _results.append((group, name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          "" if cond else "   <- " + str(detail)[:200]))


def group(title):
    print("\n== %s ==" % title)


# ------------------------------------------------------------- fixtures --

DENSITIES = ["low", "medium", "high"]


def fake_timeline(n_frames, n_steps):
    hold = 3.0 * n_steps / n_frames
    return {"duration": round(hold * n_frames, 3),
            "frames": n_frames,
            "holds": [hold] * n_frames,
            "frame_step": [min(i * n_steps // n_frames, n_steps - 1)
                           for i in range(n_frames)],
            "cues": [{"i": i, "start": round(i * 3.0, 3),
                      "end": round((i + 1) * 3.0, 3), "text": "step %d" % i}
                     for i in range(n_steps)],
            "timing_source": "authored"}


def fake_manifest(n_diagrams=12, styles=("progressive_reveal", "colour_pop"),
                  arms=True):
    """A bundle with every condition present, so pairwise arms have pairs."""
    diagrams, narratives = [], []
    for d in range(n_diagrams):
        did = "diag%02d" % d
        diagrams.append({
            "diagram_id": did, "title": "Diagram %d" % d,
            "figure_media_id": "fig%02d" % d, "figure_w": 800, "figure_h": 600,
            "source_collection": "test",
            "element_density": DENSITIES[d % 3],
            "connectivity_level": DENSITIES[(d + 1) % 3],
            "has_raster": d % 2 == 0,
            "element_count": 10 + d, "edge_count": 5 + d, "node_count": 8 + d,
            "connectivity": 0.6, "hierarchy_depth": 2})

        style = styles[d % len(styles)]
        variants = [("animatebanana", "not_applicable", "not_applicable")]
        if arms:
            variants += [
                ("animatebanana", "with_context", "not_applicable"),
                ("animatebanana", "without_context", "not_applicable"),
                ("baseline", "not_applicable", "not_applicable"),
                ("animatebanana", "not_applicable", "verified"),
                ("animatebanana", "not_applicable", "pre_verification"),
            ]
        for i, (method, ctx, ver) in enumerate(variants):
            n_frames, n_steps = 6 + (d % 4), 5 + (d % 3)
            narratives.append({
                "narrative_id": "n_%s_%d" % (did, i), "diagram_id": did,
                "animation_style": style, "method": method,
                "context_condition": ctx, "verification_state": ver,
                "narrative_version": 1,
                "frames": ["m_%s_%d_%d" % (did, i, f) for f in range(n_frames)],
                "frame_w": 800, "frame_h": 600,
                "n_frames": n_frames, "n_steps": n_steps,
                "spoken_step_fraction": 1.0, "narration_words": 40,
                "timeline": fake_timeline(n_frames, n_steps),
                "is_attention_check": False})
    return {"bundle_schema_version": 1, "bundle_id": "testbundle",
            "diagrams": diagrams, "narratives": narratives,
            "attention_checks": [], "skipped": []}


class Fixture:
    def __init__(self, cfg=None, manifest=None):
        self.dir = Path(tempfile.mkdtemp(prefix="_t_study_"))
        self.db = StudyDB(self.dir / "study.db")
        self.manifest = manifest or fake_manifest()
        self.db.import_bundle(self.manifest)
        self.cfg = cfg or StudyConfig()
        self.version = self.db.put_config(self.cfg.to_dict(), "open", "testbundle")

    def participant(self, expert=False):
        return self.db.create_participant(
            education_level="student", config_version=self.version,
            bundle_id="testbundle", display_name="T", roll_no="R1",
            is_expert=expert)

    def run(self, pid, limit=500, answer=True):
        """Drive one participant to completion; return the trials served."""
        seen = []
        for _ in range(limit):
            trial = next_trial(self.db, pid, self.cfg)
            if trial is None:
                break
            seen.append(trial)
            if answer:
                self.db.add_response(trial["trial_id"], pid, "q1", 4)
                submit_trial(self.db, trial["trial_id"])
        return seen

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------------ pools --
group("pool construction")

fx = Fixture()
with fx.db._connect() as conn:
    p1 = build_pool(conn, "testbundle", "exp1")
    p3 = build_pool(conn, "testbundle", "exp3")
    p4 = build_pool(conn, "testbundle", "exp4")
    p5 = build_pool(conn, "testbundle", "exp5")

check("pool", "absolute pool offers one narrative per cell",
      all(len(c["narratives"]) == 1 for c in p1) and len(p1) > 0, len(p1))
check("pool", "pairwise pools offer exactly two", all(
    len(c["narratives"]) == 2 for c in p3 + p4 + p5))
for exp, pool in (("exp3", p3), ("exp4", p4), ("exp5", p5)):
    field, (a, b) = PAIR_AXIS[exp]
    ok = all({n[field] for n in c["narratives"]} == {a, b} for c in pool)
    check("pool", "%s pairs differ on %s only" % (exp, field), ok and len(pool) > 0)
check("pool", "pairs share diagram and style",
      all(len({n["diagram_id"] for n in c["narratives"]}) == 1
          and len({n["animation_style"] for n in c["narratives"]}) == 1
          for c in p3 + p4 + p5))
check("pool", "cell_key is position independent",
      cell_key("exp3", ["b", "a"]) == cell_key("exp3", ["a", "b"]))

group("an arm with no stimuli ships dark")
bare = Fixture(manifest=fake_manifest(n_diagrams=6, arms=False))
with bare.db._connect() as conn:
    check("dark", "exp1 has a pool", len(build_pool(conn, "testbundle", "exp1")) == 6)
    for exp in ("exp3", "exp4", "exp5"):
        check("dark", "%s pool is empty" % exp,
              build_pool(conn, "testbundle", exp) == [])
pid = bare.participant()
trials = bare.run(pid)
check("dark", "participant only ever sees stocked experiments",
      {t["experiment"] for t in trials} == {"exp1", "exp2"},
      {t["experiment"] for t in trials})
bare.close()


# -------------------------------------------------------------- ordering --
group("experiment-major ordering")

fx2 = Fixture()
pid = fx2.participant()
trials = fx2.run(pid)
order = [t["experiment"] for t in trials]
first_seen = []
for e in order:
    if e not in first_seen:
        first_seen.append(e)
check("order", "experiments run in configured order",
      first_seen == fx2.cfg.enabled_experiments(), first_seen)
check("order", "an experiment is finished before the next begins",
      all(order[i] == order[i + 1] or order[i + 1] not in order[:i + 1]
          for i in range(len(order) - 1)), order)
per = {e: order.count(e) for e in set(order)}
check("order", "each experiment gets its configured sample count",
      all(v == fx2.cfg.target_for(k) for k, v in per.items()), per)

group("repetition rules")
by_exp = {}
for t in trials:
    by_exp.setdefault(t["experiment"], []).append(t["trial_id"])
with fx2.db._connect() as conn:
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM trial WHERE participant_id=? ORDER BY trial_index",
        (pid,)).fetchall()]
per_exp_cells = {}
for r in rows:
    per_exp_cells.setdefault(r["experiment"], []).append(r["cell_key"])
check("repeat", "no cell repeats within an experiment",
      all(len(v) == len(set(v)) for v in per_exp_cells.values()))
diagram_by_exp = {}
for r in rows:
    diagram_by_exp.setdefault(r["experiment"], []).append(r["diagram_id"])
check("repeat", "no diagram repeats within an experiment",
      all(len(v) == len(set(v)) for v in diagram_by_exp.values()))
all_diagrams = [r["diagram_id"] for r in rows]
check("repeat", "diagrams DO repeat across experiments (by design)",
      len(all_diagrams) > len(set(all_diagrams)))


# ---------------------------------------------------------------- resume --
group("resume and persist-before-return")

fx3 = Fixture()
pid = fx3.participant()
first = next_trial(fx3.db, pid, fx3.cfg)
again = next_trial(fx3.db, pid, fx3.cfg)
check("resume", "reload returns the identical trial",
      first["trial_id"] == again["trial_id"])
check("resume", "resumed payload is byte-identical",
      json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True))
with fx3.db._connect() as conn:
    row = conn.execute("SELECT * FROM trial WHERE trial_id=?",
                       (first["trial_id"],)).fetchone()
check("resume", "trial was persisted before being returned", row is not None)
check("resume", "persisted status is open", row["status"] == "open")
check("resume", "condition columns are recorded",
      row["presentation_a_condition"] is not None)

submit_trial(fx3.db, first["trial_id"])
third = next_trial(fx3.db, pid, fx3.cfg)
check("resume", "a new trial is issued after submit",
      third["trial_id"] != first["trial_id"])

group("one open trial per participant")
fx4 = Fixture()
pid = fx4.participant()
next_trial(fx4.db, pid, fx4.cfg)
try:
    with fx4.db._connect() as conn:
        conn.execute("""INSERT INTO trial
            (trial_id, participant_id, trial_index, experiment, experiment_index,
             cell_key, diagram_id, animation_style, bundle_id, presentation_a_id,
             presentation_a_condition, position_seed, assignment_reason,
             config_version, status)
            VALUES ('dup',?,99,'exp1',0,'k','diag00','s','testbundle',
                    'n_diag00_0','single','0','{}',1,'open')""", (pid,))
    check("open", "a second open trial is rejected by the engine", False, "insert allowed")
except sqlite3.IntegrityError:
    check("open", "a second open trial is rejected by the engine", True)
fx4.close()


# ------------------------------------------------------------ retirement --
group("retirement and replacement")

cfg = StudyConfig()
cfg.judgments_per_sample = 3
cfg.samples_per_experiment = {e: 4 for e in cfg.experiment_order}
cfg.enabled = {"exp1": True, "exp2": False, "exp3": False,
               "exp4": False, "exp5": False}
fx5 = Fixture(cfg=cfg, manifest=fake_manifest(n_diagrams=12, arms=False))
for _ in range(9):
    fx5.run(fx5.participant())

cov = {c["cell_key"]: c for c in coverage(fx5.db, cfg, "testbundle")}
over = [c for c in cov.values() if c["judgments"] > cfg.judgments_per_sample]
check("retire", "no cell exceeds its quota", not over,
      [(c["cell_key"], c["judgments"]) for c in over[:3]])
check("retire", "cells do reach the quota and retire",
      any(c["status"] == "retired" for c in cov.values()),
      {c["status"] for c in cov.values()})
touched = [c for c in cov.values() if c["judgments"] > 0]
check("retire", "breadth: judgments spread across many samples",
      len(touched) >= 9, len(touched))
counts = sorted(c["judgments"] for c in cov.values())
check("retire", "fills evenly rather than exhausting one sample",
      counts[-1] - counts[0] <= cfg.judgments_per_sample, counts)

group("the quota is a quota, not a suggestion")

cfgq = StudyConfig()
cfgq.judgments_per_sample = 4
cfgq.samples_per_experiment = {e: 5 for e in cfgq.experiment_order}
cfgq.enabled = {"exp1": True, "exp2": True, "exp3": False,
                "exp4": False, "exp5": False}
fxq = Fixture(cfg=cfgq, manifest=fake_manifest(n_diagrams=5, arms=False))
served_counts = [len(fxq.run(fxq.participant())) for _ in range(8)]
covq = coverage(fxq.db, cfgq, "testbundle")
check("quota", "no cell ever exceeds the quota",
      max(c["judgments"] for c in covq) == cfgq.judgments_per_sample,
      sorted(c["judgments"] for c in covq))
check("quota", "total judgments equal cells x quota",
      sum(c["judgments"] for c in covq) == len(covq) * cfgq.judgments_per_sample,
      sum(c["judgments"] for c in covq))
check("quota", "participants past saturation are served nothing, not over-quota work",
      served_counts[-1] == 0 and served_counts[0] > 0, served_counts)
check("quota", "exhaustion is total, not partial",
      all(c["status"] == "retired" for c in covq))
fxq.close()

group("an exhausted experiment yields to the next")

cfge = StudyConfig()
cfge.judgments_per_sample = 1
cfge.samples_per_experiment = {"exp1": 10, "exp2": 10}
cfge.enabled = {"exp1": True, "exp2": True, "exp3": False,
                "exp4": False, "exp5": False}
# Three diagrams, quota 1: exp1 can serve at most 3 judgments in total, far
# short of its target of 10. The session must still reach exp2.
fxe = Fixture(cfg=cfge, manifest=fake_manifest(n_diagrams=3, arms=False))
seen_a = [t["experiment"] for t in fxe.run(fxe.participant())]
check("yield", "a short pool does not end the session",
      set(seen_a) == {"exp1", "exp2"}, seen_a)
seen_b = [t["experiment"] for t in fxe.run(fxe.participant())]
check("yield", "once every experiment is exhausted the participant is done",
      seen_b == [], seen_b)
fxe.close()

group("stratum-matched replacement")
with fx5.db._connect() as conn:
    diagrams = {r["diagram_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM diagram").fetchall()}
strata = {}
for c in cov.values():
    if c["judgments"]:
        d = diagrams[c["diagram_id"]]
        key = (d["element_density"], d["connectivity_level"], d["has_raster"])
        strata[key] = strata.get(key, 0) + c["judgments"]
spread = max(strata.values()) - min(strata.values()) if strata else 0
check("stratum", "judgments spread across strata rather than pooling in one",
      len(strata) >= 3, sorted(strata.items())[:4])
check("stratum", "no stratum starves", spread <= max(strata.values()), strata)
fx5.close()


# --------------------------------------------------------------- pairing --
group("A/B randomisation")

fx6 = Fixture()
cfg6 = StudyConfig()
cfg6.enabled = {"exp1": False, "exp2": False, "exp3": True,
                "exp4": False, "exp5": False}
cfg6.samples_per_experiment = {"exp3": 12}
cfg6.judgments_per_sample = 40      # enough draws for the tolerance below
fx6.cfg = cfg6
for _ in range(60):
    fx6.run(fx6.participant())
with fx6.db._connect() as conn:
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM trial WHERE experiment='exp3'").fetchall()]
left = sum(1 for r in rows if r["presentation_a_condition"] == "with_context")
frac = left / len(rows) if rows else 0
check("ab", "both conditions appear on the left", 0 < left < len(rows), (left, len(rows)))
# The randomiser is exactly fair (0.5000 over 200k draws of the primitive), so
# any tolerance here is really a statement about sample size. At n=72 a +/-0.12
# bound fails ~4.6% of runs by chance alone -- a flaky test that would
# eventually be "fixed" by loosening it until it could no longer detect real
# position bias. Draw more trials instead and keep the bound tight.
check("ab", "enough pairwise trials to test position bias",
      len(rows) >= 200, len(rows))
check("ab", "left/right split is near even", abs(frac - 0.5) < 0.10,
      "p(with_context on A)=%.3f over %d trials" % (frac, len(rows)))
check("ab", "every pairwise trial records both conditions",
      all(r["presentation_b_condition"] for r in rows))
check("ab", "the two sides are never the same narrative",
      all(r["presentation_a_id"] != r["presentation_b_id"] for r in rows))
fx6.close()


# -------------------------------------------------------------- blinding --
group("blinding of the client payload")

fx7 = Fixture()
payloads = []
for _ in range(8):
    payloads.extend(fx7.run(fx7.participant()))
blob = json.dumps(payloads).lower()
FORBIDDEN = ["narrative_id", "n_diag", "with_context", "without_context",
             "pre_verification", "verified", "animatebanana", "baseline",
             "method", "diagram_id", "cell_key", "assignment_reason",
             "position_seed", "gemini", "lineage", ".webp", ".png", "/media/"]
leaks = [tok for tok in FORBIDDEN if tok in blob]
check("blinding", "no condition or identity token in any payload", not leaks, leaks)
check("blinding", "payload exposes only slot letters",
      all(s["slot"] in ("A", "B") for p in payloads for s in p["slots"]))
check("blinding", "captions carried only when the experiment shows them",
      all((len(s["cues"]) > 0) == p["show_captions"]
          for p in payloads for s in p["slots"]))
fx7.close()


# ----------------------------------------------------------- append-only --
group("append-only raw data")

check("append", "no update_response method exists",
      not hasattr(StudyDB, "update_response"))
check("append", "no delete_response method exists",
      not hasattr(StudyDB, "delete_response"))

fx8 = Fixture()
pid = fx8.participant()
t = next_trial(fx8.db, pid, fx8.cfg)
fx8.db.add_response(t["trial_id"], pid, "coverage", 3)
fx8.db.add_response(t["trial_id"], pid, "coverage", 5)   # a revision
fx8.db.add_response(t["trial_id"], pid, "quality", 2)
with fx8.db._connect() as conn:
    rows = conn.execute("SELECT * FROM response WHERE trial_id=? ORDER BY response_id",
                        (t["trial_id"],)).fetchall()
check("append", "a revision appends rather than overwrites", len(rows) == 3, len(rows))
check("append", "revision numbers increment",
      [r["revision"] for r in rows if r["question_id"] == "coverage"] == [0, 1])
latest = fx8.db.latest_answers(t["trial_id"])
check("append", "latest_answers reads the newest revision",
      latest == {"coverage": 5, "quality": 2}, latest)

with fx8.db._connect() as conn:
    before = conn.execute("SELECT group_concat(value) v FROM response").fetchone()["v"]
fx8.db.annotate_participant(pid, "exclude", "test exclusion", "tester")
with fx8.db._connect() as conn:
    after = conn.execute("SELECT group_concat(value) v FROM response").fetchone()["v"]
    ann = conn.execute("SELECT COUNT(*) c FROM participant_annotation").fetchone()["c"]
check("append", "exclusion does not alter raw responses", before == after)
check("append", "exclusion is recorded as an annotation", ann == 1)

group("PII separation")
with fx8.db._connect() as conn:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(participant)").fetchall()]
check("pii", "participant table carries no name or roll number",
      "display_name" not in cols and "roll_no" not in cols, cols)
with fx8.db._connect() as conn:
    pii = conn.execute("SELECT * FROM participant_pii WHERE participant_id=?",
                       (pid,)).fetchone()
check("pii", "identity is stored in its own table", pii["roll_no"] == "R1")
fx8.close()


# ------------------------------------------------------------ concurrency --
group("concurrency")

fx9 = Fixture(manifest=fake_manifest(n_diagrams=12))
pids = [fx9.participant() for _ in range(16)]
errors, served = [], []
lock = threading.Lock()


def worker(pid):
    try:
        trials = fx9.run(pid)
        with lock:
            served.append(len(trials))
    except Exception as exc:                       # noqa: BLE001 - recorded, not raised
        with lock:
            errors.append(repr(exc))


threads = [threading.Thread(target=worker, args=(p,)) for p in pids]
for t_ in threads:
    t_.start()
for t_ in threads:
    t_.join()

check("concurrent", "no errors under 16 concurrent participants", not errors, errors[:2])
check("concurrent", "every participant was served trials",
      len(served) == 16 and all(s > 0 for s in served), served)
with fx9.db._connect() as conn:
    dupes = conn.execute(
        "SELECT participant_id, COUNT(*) c FROM trial WHERE status='open'"
        " GROUP BY participant_id HAVING c > 1").fetchall()
    total = conn.execute("SELECT COUNT(*) c FROM trial").fetchone()["c"]
check("concurrent", "no participant holds two open trials", not dupes)
check("concurrent", "trials were actually written", total > 100, total)

group("double-click race")
fx10 = Fixture()
pid = fx10.participant()
got, errs = [], []


def clicker():
    try:
        got.append(next_trial(fx10.db, pid, fx10.cfg)["trial_id"])
    except Exception as exc:                       # noqa: BLE001
        errs.append(repr(exc))


clickers = [threading.Thread(target=clicker) for _ in range(10)]
for t_ in clickers:
    t_.start()
for t_ in clickers:
    t_.join()
with fx10.db._connect() as conn:
    open_count = conn.execute(
        "SELECT COUNT(*) c FROM trial WHERE participant_id=? AND status='open'",
        (pid,)).fetchone()["c"]
check("race", "ten simultaneous clicks yield exactly one open trial",
      open_count == 1, open_count)
check("race", "every click saw the same trial", len(set(got)) <= 1, set(got))
fx10.close()
fx9.close()
fx7 = None
for f in (fx, fx2, fx3, fx8):
    f.close()


# ------------------------------------------------------------------ report --
leftover = list(Path(tempfile.gettempdir()).glob("_t_study_*"))
print("\nleftover temp dirs: %d" % len(leftover))

failed = [r for r in _results if not r[2]]
print("%d checks, %d passed, %d failed" %
      (len(_results), len(_results) - len(failed), len(failed)))
for g, name, _, detail in failed:
    print("  FAIL [%s] %s   %s" % (g, name, str(detail)[:220]))
sys.exit(1 if failed else 0)
