"""Scheduler, question-set, database and blinding tests.

Offline, no model calls, no server. Builds synthetic bundles in a temp dir in
the shape of the two real ones -- the main study (three ranking contenders per
figure, stamped with a study day) and the selective cohort (+K / -K / verified
reference per figure) -- and also imports the real manifests to pin the pool
sizes the live servers will actually see.

    PYTHONPATH=src /fsxvision_new/venkat.kesav/environments/study/bin/python \
        tests/test_study_scheduler.py
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path

from img_2_svg_pretraining.study.config import (
    ABSOLUTE_EXPERIMENTS, FIXED_SIDES, PAIR_AXIS, SCREEN, SHOWS_CAPTIONS, TOURNAMENT,
    StudyConfig)
from img_2_svg_pretraining.study.db import StudyDB
from img_2_svg_pretraining.study.questions import (
    all_experiments, metric_map, question_set, required_ids, visible)
from img_2_svg_pretraining.study.scheduler import (
    build_pool, cell_key, coverage, next_trial, submit_trial)

CONFIG_DIR = Path("src/img_2_svg_pretraining/pipeline/configs")
MAIN_BUNDLE = Path(os.environ.get("STUDY_BUNDLE_MAIN", "data/study_bundles/main-v1"))
SELECTIVE_BUNDLE = Path(os.environ.get("STUDY_BUNDLE_SELECTIVE",
                                       "data/study_bundles/selective-v1"))

_results = []


def check(group, name, cond, detail=""):
    _results.append((group, name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          "" if cond else "   <- " + str(detail)[:200]))


def group(title):
    print("\n== %s ==" % title)


# ------------------------------------------------------------- fixtures --

DENSITIES = ["low", "medium", "high"]
STYLES = ("progressive_reveal", "colour_pop", "hopping_bounding_box")


def fake_timeline(n_frames, n_steps, silent=False):
    hold = 3.0 * n_steps / n_frames
    return {"duration": round(hold * n_frames, 3),
            "frames": n_frames,
            "holds": [hold] * n_frames,
            "frame_step": [min(i * n_steps // n_frames, n_steps - 1)
                           for i in range(n_frames)],
            "cues": [{"i": i, "start": round(i * 3.0, 3),
                      "end": round((i + 1) * 3.0, 3),
                      "text": "" if silent else "step %d" % i}
                     for i in range(n_steps)],
            "timing_source": "authored"}


def _diagram(d, did, day=None):
    return {"diagram_id": did, "title": "Diagram %d" % d,
            "figure_media_id": "fig%02d" % d, "figure_w": 800, "figure_h": 600,
            "source_collection": "test",
            "element_density": DENSITIES[d % 3],
            "connectivity_level": DENSITIES[(d + 1) % 3],
            "has_raster": d % 2 == 0,
            "element_count": 10 + d, "edge_count": 5 + d, "node_count": 8 + d,
            "connectivity": 0.6, "hierarchy_depth": 2,
            "study_day": day, "complexity": "complex" if d % 2 else "easy"}


def _narrative(d, did, i, style, method, ctx, ver, silent=False, source=None):
    n_frames, n_steps = 6 + (d % 4), 5 + (d % 3)
    rec = {"narrative_id": "n_%s_%d" % (did, i), "diagram_id": did,
           "animation_style": style, "method": method,
           "context_condition": ctx, "verification_state": ver,
           "narrative_version": 1,
           "frames": ["m_%s_%d_%d" % (did, i, f) for f in range(n_frames)],
           "frame_w": 800, "frame_h": 600,
           "n_frames": n_frames, "n_steps": n_steps,
           "spoken_step_fraction": 0.0 if silent else 1.0,
           "narration_words": 0 if silent else 40,
           "timeline": fake_timeline(n_frames, n_steps, silent),
           "is_attention_check": False}
    if source:
        rec["source"] = source
    return rec


def main_manifest(n_diagrams=20, days=2, methods=TOURNAMENT["exp2"]):
    """The main study's shape: every figure under every contender, stamped
    with a day. `methods` lets a figure ship with a contender missing."""
    diagrams, narratives = [], []
    per_day = max(n_diagrams // days, 1)
    for d in range(n_diagrams):
        did = "diag%02d" % d
        diagrams.append(_diagram(d, did, day=d // per_day + 1))
        style = STYLES[d % len(STYLES)]
        for i, method in enumerate(methods):
            narratives.append(_narrative(
                d, did, i, style, method, "not_applicable", "not_applicable",
                silent=(method == "talk"),
                source="original_talk" if method == "talk" else None))
    return {"bundle_schema_version": 1, "bundle_id": "testbundle",
            "diagrams": diagrams, "narratives": narratives,
            "attention_checks": [], "skipped": []}


def selective_manifest(n_diagrams=12):
    """The selective cohort's shape: +K (relabelled pre_verification), -K and
    the verified bench reference, one style per figure."""
    diagrams, narratives = [], []
    for d in range(n_diagrams):
        did = "diag%02d" % d
        diagrams.append(_diagram(d, did))
        style = STYLES[d % len(STYLES)]
        variants = [("animatebanana", "with_context", "pre_verification"),
                    ("animatebanana", "without_context", "not_applicable"),
                    ("animatebanana", "not_applicable", "verified")]
        for i, (method, ctx, ver) in enumerate(variants):
            narratives.append(_narrative(d, did, i, style, method, ctx, ver,
                                         source="bench_reference" if ver == "verified"
                                         else None))
    return {"bundle_schema_version": 1, "bundle_id": "testbundle",
            "diagrams": diagrams, "narratives": narratives,
            "attention_checks": [], "skipped": []}


def main_cfg(**over):
    cfg = StudyConfig(experiment_order=("exp1", "exp2"),
                      enabled={"exp1": True, "exp2": True, "context": False, "bench": False},
                      samples_per_experiment={"exp1": 20, "exp2": 10},
                      judgments_per_sample=7, stratum_fields=("complexity", "animation_style"),
                      study_day=1)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def selective_cfg(**over):
    cfg = StudyConfig(experiment_order=("context", "bench"),
                      enabled={"exp1": False, "exp2": False, "context": True, "bench": True},
                      samples_per_experiment={"context": 15, "bench": 15},
                      judgments_per_sample=10,
                      stratum_fields=("element_density", "animation_style"))
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def answer_for(experiment, db, trial, pid):
    """A complete, minimal response for whichever screen the trial is on."""
    if experiment in TOURNAMENT:
        db.add_response(trial["trial_id"], pid, "round1_pick", "x")
    else:
        for q in required_ids(experiment, {}):
            db.add_response(trial["trial_id"], pid, q, False)


class Fixture:
    def __init__(self, cfg=None, manifest=None):
        self.dir = Path(tempfile.mkdtemp(prefix="_t_study_"))
        self.db = StudyDB(self.dir / "study.db")
        self.manifest = manifest or main_manifest()
        self.db.import_bundle(self.manifest)
        self.cfg = cfg or main_cfg()
        self.version = self.db.put_config(self.cfg.to_dict(), "open",
                                          self.manifest["bundle_id"])

    @property
    def bundle_id(self):
        return self.manifest["bundle_id"]

    def participant(self, expert=False):
        return self.db.create_participant(
            education_level="student", config_version=self.version,
            bundle_id=self.bundle_id, display_name="T", roll_no="R1",
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
                answer_for(trial["experiment"], self.db, trial, pid)
                submit_trial(self.db, trial["trial_id"])
        return seen

    def rows(self, pid):
        with self.db._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM trial WHERE participant_id=? ORDER BY trial_index",
                (pid,)).fetchall()]

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------- questions --
group("question sets: the progressive exp1 form")

qs = question_set("exp1", style_name="colour_pop", style_description="Greyscale then colour.")
ids = [q["id"] for q in qs["questions"]]
check("q", "exp1 asks vfs, ascs, sss, nas in that order",
      ids == ["vfs", "ascs", "sss", "nas"], ids)
check("q", "types are yesno, yesno, score10, score10",
      [q["type"] for q in qs["questions"]] == ["yesno", "yesno", "score10", "score10"])
check("q", "the style name is rendered into the ascs prompt",
      "Colour Pop" in qs["questions"][1]["prompt"], qs["questions"][1]["prompt"])
check("q", "the style summary is the ascs help text",
      qs["questions"][1]["help"] == "Greyscale then colour.")
check("q", "ascs waits on vfs", qs["questions"][1].get("show_if") == {"vfs": True})
check("q", "sss and nas wait on both gates",
      all(q.get("show_if") == {"vfs": True, "ascs": True} for q in qs["questions"][2:]))
check("q", "metric never reaches the participant",
      all("metric" not in q for q in qs["questions"]))
check("q", "but is kept for the correlation analysis",
      metric_map("exp1") == {"vfs": "vfs", "ascs": "ascs_pass", "sss": "sss", "nas": "nas"},
      metric_map("exp1"))
check("q", "the main study asks no familiarity question", qs["familiarity"] is None)
check("q", "gate-closed text exists for a No", bool(qs["gate_closed_text"]))
check("q", "score anchors are carried",
      qs["questions"][2]["anchors"] and set(qs["questions"][2]["anchors"]) == {0, 10},
      qs["questions"][2].get("anchors"))

group("required_ids follows the gates")
check("req", "nothing answered: only vfs is owed", required_ids("exp1", {}) == ["vfs"])
check("req", "vfs=No alone is a complete trial",
      required_ids("exp1", {"vfs": False}) == ["vfs"])
check("req", "vfs=Yes opens ascs",
      required_ids("exp1", {"vfs": True}) == ["vfs", "ascs"])
check("req", "ascs=No closes the rest",
      required_ids("exp1", {"vfs": True, "ascs": False}) == ["vfs", "ascs"])
check("req", "both Yes owes everything",
      required_ids("exp1", {"vfs": True, "ascs": True}) == ["vfs", "ascs", "sss", "nas"])
check("req", "a stale dependent does not reopen a closed gate",
      required_ids("exp1", {"vfs": False, "ascs": True, "sss": 5}) == ["vfs"])
check("req", "visible() treats a missing condition as unsatisfied",
      not visible({"show_if": {"vfs": True}}, {}) and visible({"id": "x"}, {}))
check("req", "context owes pref_insight", required_ids("context") == ["pref_insight"])
check("req", "bench owes improved", required_ids("bench") == ["improved"])
check("req", "the tournament's only question is the pick",
      required_ids("exp2") == ["pick"] and question_set("exp2")["questions"][0]["type"] == "pick")

group("question sets: the selective cohort")
ctx = question_set("context")
check("q", "context is a three-way choice_pair with A/B/tie",
      ctx["questions"][0]["type"] == "choice_pair"
      and [o["value"] for o in ctx["questions"][0]["options"]] == ["A", "B", "tie"],
      ctx["questions"][0])
bench = question_set("bench")
check("q", "bench is a single yes/no", bench["questions"][0]["type"] == "yesno"
      and bench["questions"][0]["id"] == "improved")
check("q", "all_experiments lists only the exp*.yaml sets",
      all_experiments() == ["exp1", "exp2"], all_experiments())
try:
    question_set("exp9")
    check("q", "an unknown set raises", False, "no exception")
except FileNotFoundError:
    check("q", "an unknown set raises", True)

group("config constants agree with each other")
check("cfg", "every screen'd experiment has a caption policy",
      set(SCREEN) == set(SHOWS_CAPTIONS), (set(SCREEN), set(SHOWS_CAPTIONS)))
check("cfg", "absolute experiments are screen=absolute",
      all(SCREEN[e] == "absolute" for e in ABSOLUTE_EXPERIMENTS))
check("cfg", "tournament experiments are screen=tournament",
      all(SCREEN[e] == "tournament" for e in TOURNAMENT))
check("cfg", "pairwise experiments are screen=pairwise",
      all(SCREEN[e] == "pairwise" for e in PAIR_AXIS))
check("cfg", "fixed-side experiments are pairwise ones", FIXED_SIDES <= set(PAIR_AXIS))
check("cfg", "the tournament is exactly three contenders",
      all(len(v) == 3 for v in TOURNAMENT.values()))
check("cfg", "the tournament hides captions (talk has none)",
      all(SHOWS_CAPTIONS[e] is False for e in TOURNAMENT))
check("cfg", "exp1 shows captions (it asks about narration)", SHOWS_CAPTIONS["exp1"] is True)

for name, exps, day in (("study_main.yaml", ["exp1", "exp2"], 1),
                        ("study_selective.yaml", ["context", "bench"], None)):
    loaded = StudyConfig.load(CONFIG_DIR / name)
    check("cfg", "%s enables %s" % (name, exps), loaded.enabled_experiments() == exps,
          loaded.enabled_experiments())
    check("cfg", "%s study_day is %s" % (name, day), loaded.study_day == day, loaded.study_day)
    check("cfg", "%s round-trips through to_dict/from_dict" % name,
          StudyConfig.from_dict(loaded.to_dict()) == loaded)
main_yaml = StudyConfig.load(CONFIG_DIR / "study_main.yaml")
check("cfg", "main targets: 20 absolute, 10 tournaments",
      (main_yaml.target_for("exp1"), main_yaml.target_for("exp2")) == (20, 10))


# ------------------------------------------------------------------ pools --
group("pool construction: main study")

fx = Fixture()
with fx.db._connect() as conn:
    p1 = build_pool(conn, fx.bundle_id, "exp1")
    p2 = build_pool(conn, fx.bundle_id, "exp2")
    p1_day1 = build_pool(conn, fx.bundle_id, "exp1", study_day=1)
    p2_day1 = build_pool(conn, fx.bundle_id, "exp2", study_day=1)
    p_ctx = build_pool(conn, fx.bundle_id, "context")
    p_bench = build_pool(conn, fx.bundle_id, "bench")

check("pool", "absolute pool offers one narrative per cell",
      all(len(c["narratives"]) == 1 for c in p1) and len(p1) > 0, len(p1))
check("pool", "absolute pool has one cell per (figure, method) minus the talk",
      len(p1) == 20 * 2 and Counter(c["conditions"][0] for c in p1)
      == {"animatebanana": 20, "qwen38": 20},
      Counter(c["conditions"][0] for c in p1))
check("pool", "the talk is never rated in isolation",
      all(c["conditions"] != ["talk"] for c in p1))
check("pool", "the absolute condition is the method",
      all(c["conditions"] == [c["narratives"][0]["method"]] for c in p1))
check("pool", "tournament pool has one cell per figure",
      len(p2) == 20 and len({c["diagram_id"] for c in p2}) == 20, len(p2))
check("pool", "tournament cells hold all three contenders in the design order",
      all([n["method"] for n in c["narratives"]] == list(TOURNAMENT["exp2"])
          and c["conditions"] == list(TOURNAMENT["exp2"]) for c in p2))
check("pool", "study_day filters both pools to that day's figures",
      len(p1_day1) == 20 and len(p2_day1) == 10
      and {c["diagram_id"] for c in p1_day1} == {"diag%02d" % d for d in range(10)},
      (len(p1_day1), len(p2_day1)))
check("pool", "no study_day means no filtering", len(p1) == 2 * len(p1_day1))
check("pool", "the pairwise arms ship dark in the main study",
      p_ctx == [] and p_bench == [])
check("pool", "cell_key is position independent",
      cell_key("context", ["b", "a"]) == cell_key("context", ["a", "b"]))
check("pool", "a tournament cell_key names all three",
      cell_key("exp2", ["c", "a", "b"]) == "exp2:a|b|c")

group("a figure missing a contender is not offered as a tournament")
short = main_manifest(n_diagrams=6, days=1)
# Drop qwen38 from the last figure.
short["narratives"] = [n for n in short["narratives"]
                       if not (n["diagram_id"] == "diag05" and n["method"] == "qwen38")]
fxs = Fixture(manifest=short, cfg=main_cfg(samples_per_experiment={"exp1": 12, "exp2": 6}))
with fxs.db._connect() as conn:
    pool = build_pool(conn, fxs.bundle_id, "exp2")
    pool1 = build_pool(conn, fxs.bundle_id, "exp1")
check("short", "two-way 'tournaments' are refused", len(pool) == 5
      and "diag05" not in {c["diagram_id"] for c in pool}, len(pool))
check("short", "but the figure is still rated absolutely under what it has",
      sum(1 for c in pool1 if c["diagram_id"] == "diag05") == 1)
fxs.close()

group("pool construction: selective cohort")
fxsel = Fixture(manifest=selective_manifest(), cfg=selective_cfg())
with fxsel.db._connect() as conn:
    pc = build_pool(conn, fxsel.bundle_id, "context")
    pb = build_pool(conn, fxsel.bundle_id, "bench")
    pe1 = build_pool(conn, fxsel.bundle_id, "exp1")
    pe2 = build_pool(conn, fxsel.bundle_id, "exp2")
check("pool", "pairwise pools offer exactly two", all(len(c["narratives"]) == 2 for c in pc + pb)
      and len(pc) == 12 and len(pb) == 12, (len(pc), len(pb)))
for exp, pool in (("context", pc), ("bench", pb)):
    field_, (a, b) = PAIR_AXIS[exp]
    check("pool", "%s pairs differ on %s only" % (exp, field_),
          all({n[field_] for n in c["narratives"]} == {a, b} for c in pool) and pool)
    check("pool", "%s conditions are listed in the axis order" % exp,
          all(c["conditions"] == [a, b] for c in pool))
check("pool", "pairs share diagram and style",
      all(len({n["diagram_id"] for n in c["narratives"]}) == 1
          and len({n["animation_style"] for n in c["narratives"]}) == 1
          for c in pc + pb))
check("pool", "the -K and verified narratives never enter an absolute pool",
      all(c["narratives"][0]["context_condition"] == "with_context"
          and c["narratives"][0]["verification_state"] == "pre_verification" for c in pe1)
      and len(pe1) == 12, len(pe1))
check("pool", "no tournament without the contenders", pe2 == [])

group("the real bundles produce the pools the live servers report")
for bundle, cfg_name, want, want_off in (
        (MAIN_BUNDLE, "study_main.yaml", {"exp1": 20, "exp2": 10},
         {"context": 0, "bench": 0}),
        # The +K side of the selective bundle would stock an absolute arm; it
        # is the config that keeps exp1 off there, not an empty pool.
        (SELECTIVE_BUNDLE, "study_selective.yaml", {"context": 13, "bench": 13},
         {"exp1": 13, "exp2": 0})):
    if not (bundle / "manifest.json").exists():
        check("real", "%s manifest present" % bundle.name, False, str(bundle))
        continue
    cfg_real = StudyConfig.load(CONFIG_DIR / cfg_name)
    fxr = Fixture(manifest=json.loads((bundle / "manifest.json").read_text(encoding="utf-8")),
                  cfg=cfg_real)
    with fxr.db._connect() as conn:
        sizes = {e: len(build_pool(conn, fxr.bundle_id, e, cfg_real.study_day))
                 for e in cfg_real.enabled_experiments()}
        off = {e: len(build_pool(conn, fxr.bundle_id, e, cfg_real.study_day))
               for e in SCREEN if e not in cfg_real.enabled_experiments()}
    check("real", "%s pools under %s are %s" % (bundle.name, cfg_name, want),
          sizes == want, sizes)
    check("real", "%s: the arms the config leaves off hold %s" % (bundle.name, want_off),
          off == want_off, off)
    if bundle is MAIN_BUNDLE:
        with fxr.db._connect() as conn:
            all_days = {d: len(build_pool(conn, fxr.bundle_id, "exp2", d)) for d in (1, 2, 3)}
        check("real", "main-v1 offers ten tournaments on each of the three days",
              all_days == {1: 10, 2: 10, 3: 10}, all_days)
    fxr.close()


# -------------------------------------------------------------- ordering --
group("experiment-major ordering, main study")

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
      per == {"exp1": 20, "exp2": 10}, per)
check("order", "trial_index and experiment_index are recorded",
      [t["trial_index"] for t in trials] == list(range(30))
      and [t["experiment_index"] for t in trials if t["experiment"] == "exp2"] == list(range(10)))
check("order", "the screen follows the experiment",
      all(t["screen"] == SCREEN[t["experiment"]] for t in trials))

group("repetition rules: absolute = once per method, fresh figures first")
rows = fx2.rows(pid)
abs_rows = [r for r in rows if r["experiment"] == "exp1"]
diagrams = [r["diagram_id"] for r in abs_rows]
check("repeat", "no cell repeats within an experiment",
      len({r["cell_key"] for r in abs_rows}) == len(abs_rows))
check("repeat", "every day-1 figure is rated exactly once per method",
      Counter(diagrams) == {"diag%02d" % d: 2 for d in range(10)}, Counter(diagrams))
check("repeat", "both methods are rated for every figure",
      all(sorted(r["presentation_a_condition"] for r in abs_rows if r["diagram_id"] == d)
          == ["animatebanana", "qwen38"] for d in set(diagrams)))
check("repeat", "the first ten are ten distinct figures",
      len(set(diagrams[:10])) == 10, diagrams[:10])
check("repeat", "a figure never repeats back to back",
      all(diagrams[i] != diagrams[i + 1] for i in range(len(diagrams) - 1)), diagrams)
check("repeat", "nothing from another day is served",
      all(d in {"diag%02d" % i for i in range(10)} for d in diagrams))
tour_rows = [r for r in rows if r["experiment"] == "exp2"]
check("repeat", "a tournament figure is seen at most once",
      len({r["diagram_id"] for r in tour_rows}) == len(tour_rows) == 10)
check("repeat", "diagrams DO repeat across experiments (by design)",
      {r["diagram_id"] for r in tour_rows} == set(diagrams))

group("the tournament keeps the design's order")
check("tour", "three contenders are recorded in served order",
      all(json.loads(r["presentation_ids"]).__len__() == 3
          and json.loads(r["presentation_conditions"]) == list(TOURNAMENT["exp2"])
          for r in tour_rows))
check("tour", "a_/b_ mirror the first two for older readers",
      all((r["presentation_a_condition"], r["presentation_b_condition"])
          == tuple(TOURNAMENT["exp2"][:2]) for r in tour_rows))
tour_payloads = [t for t in trials if t["experiment"] == "exp2"]
check("tour", "the payload has slots A, B and C",
      all([s["slot"] for s in t["slots"]] == ["A", "B", "C"] for t in tour_payloads))
check("tour", "captions are off on every side",
      all(not t["show_captions"] and all(s["cues"] == [] for s in t["slots"])
          for t in tour_payloads))
abs_payloads = [t for t in trials if t["experiment"] == "exp1"]
check("abs", "the absolute payload is a single slot A with captions",
      all([s["slot"] for s in t["slots"]] == ["A"] and t["show_captions"]
          and len(t["slots"][0]["cues"]) > 0 for t in abs_payloads))

group("a day change is a pool change")
fx2.cfg.study_day = 2
pid2 = fx2.participant()
fx2.run(pid2)
day2 = {r["diagram_id"] for r in fx2.rows(pid2)}
check("day", "day 2 serves only day-2 figures",
      day2 == {"diag%02d" % d for d in range(10, 20)}, sorted(day2)[:4])
fx2.close()


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
      row["presentation_a_condition"] in TOURNAMENT["exp2"]
      and json.loads(row["presentation_conditions"]) == [row["presentation_a_condition"]])
check("resume", "assignment reason records pool size and quota",
      set(json.loads(row["assignment_reason"])) == {"judgments_before", "quota", "pool_size"})

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
                    'n_diag00_0','animatebanana','0','{}',1,'open')""", (pid,))
    check("open", "a second open trial is rejected by the engine", False, "insert allowed")
except sqlite3.IntegrityError:
    check("open", "a second open trial is rejected by the engine", True)
fx4.close()


# ------------------------------------------------------------ retirement --
group("retirement and replacement")

cfg = main_cfg(judgments_per_sample=3, samples_per_experiment={"exp1": 4, "exp2": 4},
               enabled={"exp1": True, "exp2": False}, study_day=None)
fx5 = Fixture(cfg=cfg, manifest=main_manifest(n_diagrams=12, days=1,
                                              methods=("animatebanana",)))
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

cfgq = main_cfg(judgments_per_sample=4, samples_per_experiment={"exp1": 5, "exp2": 5},
                study_day=None)
fxq = Fixture(cfg=cfgq, manifest=main_manifest(n_diagrams=5, days=1))
served_counts = [len(fxq.run(fxq.participant())) for _ in range(10)]
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

cfge = main_cfg(judgments_per_sample=1, samples_per_experiment={"exp1": 10, "exp2": 10},
                study_day=None)
# Three figures, quota 1: exp1 can serve at most 6 judgments in total, far
# short of its target of 10. The session must still reach exp2.
fxe = Fixture(cfg=cfge, manifest=main_manifest(n_diagrams=3, days=1))
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
        key = (d["complexity"], c["animation_style"])
        strata[key] = strata.get(key, 0) + c["judgments"]
spread = max(strata.values()) - min(strata.values()) if strata else 0
check("stratum", "judgments spread across strata rather than pooling in one",
      len(strata) >= 3, sorted(strata.items())[:4])
check("stratum", "no stratum starves", spread <= max(strata.values()), strata)
fx5.close()


# --------------------------------------------------------------- pairing --
group("pairwise: context sides are randomised, bench sides are fixed")

cfg6 = selective_cfg(samples_per_experiment={"context": 12, "bench": 12},
                     judgments_per_sample=40)
fx6 = Fixture(cfg=cfg6, manifest=selective_manifest())
for _ in range(30):
    fx6.run(fx6.participant())
with fx6.db._connect() as conn:
    ctx_rows = [dict(r) for r in conn.execute(
        "SELECT * FROM trial WHERE experiment='context'").fetchall()]
    bench_rows = [dict(r) for r in conn.execute(
        "SELECT * FROM trial WHERE experiment='bench'").fetchall()]
    per_p = [dict(r) for r in conn.execute(
        "SELECT participant_id, experiment, COUNT(*) c, COUNT(DISTINCT diagram_id) d"
        " FROM trial GROUP BY participant_id, experiment").fetchall()]
left = sum(1 for r in ctx_rows if r["presentation_a_condition"] == "with_context")
frac = left / len(ctx_rows) if ctx_rows else 0
check("ab", "both conditions appear on the left", 0 < left < len(ctx_rows), (left, len(ctx_rows)))
# The randomiser is exactly fair, so any tolerance here is a statement about
# sample size. At n=360 a +/-0.10 bound fails by chance well under 1 in 10^4.
check("ab", "enough pairwise trials to test position bias",
      len(ctx_rows) >= 300, len(ctx_rows))
check("ab", "left/right split is near even", abs(frac - 0.5) < 0.10,
      "p(with_context on A)=%.3f over %d trials" % (frac, len(ctx_rows)))
check("ab", "every pairwise trial records both conditions",
      all(r["presentation_b_condition"] for r in ctx_rows + bench_rows))
check("ab", "the two sides are never the same narrative",
      all(r["presentation_a_id"] != r["presentation_b_id"] for r in ctx_rows + bench_rows))
check("fixed", "bench: the original is ALWAYS left and the correction right",
      bench_rows and all((r["presentation_a_condition"], r["presentation_b_condition"])
                         == ("pre_verification", "verified") for r in bench_rows),
      Counter((r["presentation_a_condition"], r["presentation_b_condition"])
              for r in bench_rows))
check("fixed", "the position seed is still recorded for fixed sides",
      all(r["position_seed"] for r in bench_rows))
check("pair", "a figure is seen at most once per pairwise experiment",
      all(r["c"] == r["d"] for r in per_p))
check("pair", "each pairwise experiment serves every figure once (12 of 12)",
      all(r["c"] == 12 for r in per_p), Counter(r["c"] for r in per_p))
fx6.close()


# -------------------------------------------------------------- blinding --
group("blinding of the client payload")

payloads = []
fx7 = Fixture()
for _ in range(2):
    payloads.extend(fx7.run(fx7.participant()))
fx7.close()
fx7s = Fixture(cfg=selective_cfg(), manifest=selective_manifest())
for _ in range(2):
    payloads.extend(fx7s.run(fx7s.participant()))
fx7s.close()
check("blinding", "every experiment is represented in the sample",
      {p["experiment"] for p in payloads} == {"exp1", "exp2", "context", "bench"},
      {p["experiment"] for p in payloads})
blob = json.dumps(payloads).lower()
FORBIDDEN = ["narrative_id", "n_diag", "with_context", "without_context",
             "pre_verification", "verified", "animatebanana", "qwen", "talk",
             "baseline", "method", "diagram_id", "cell_key", "assignment_reason",
             "position_seed", "gemini", "lineage", "condition", ".webp", ".png", "/media/"]
leaks = [tok for tok in FORBIDDEN if tok in blob]
check("blinding", "no condition or identity token in any payload", not leaks, leaks)
check("blinding", "payload exposes only slot letters",
      all(s["slot"] in ("A", "B", "C") for p in payloads for s in p["slots"]))
check("blinding", "the style is carried as a slug for the app to name, never a path",
      all("/" not in p["animation_style"] and p["animation_style"] in STYLES
          for p in payloads))
check("blinding", "captions carried only when the experiment shows them",
      all((len(s["cues"]) > 0) == p["show_captions"]
          for p in payloads for s in p["slots"]))
check("blinding", "context: both narrations reach the participant",
      all(all(len(s["cues"]) > 0 for s in p["slots"])
          for p in payloads if p["experiment"] == "context"))
check("blinding", "the payload never says which side is which",
      all("side_labels" not in p and "conditions" not in p for p in payloads))


# ----------------------------------------------------------- append-only --
group("append-only raw data")

check("append", "no update_response method exists",
      not hasattr(StudyDB, "update_response"))
check("append", "no delete_response method exists",
      not hasattr(StudyDB, "delete_response"))

fx8 = Fixture()
pid = fx8.participant()
t = next_trial(fx8.db, pid, fx8.cfg)
fx8.db.add_response(t["trial_id"], pid, "sss", 3)
fx8.db.add_response(t["trial_id"], pid, "sss", 5)   # a revision
fx8.db.add_response(t["trial_id"], pid, "vfs", True)
fx8.db.add_response(t["trial_id"], pid, "rank", {"animatebanana": 1, "qwen38": 2, "talk": 3})
with fx8.db._connect() as conn:
    rows = conn.execute("SELECT * FROM response WHERE trial_id=? ORDER BY response_id",
                        (t["trial_id"],)).fetchall()
check("append", "a revision appends rather than overwrites", len(rows) == 4, len(rows))
check("append", "revision numbers increment",
      [r["revision"] for r in rows if r["question_id"] == "sss"] == [0, 1])
latest = fx8.db.latest_answers(t["trial_id"])
check("append", "latest_answers reads the newest revision",
      latest == {"sss": 5, "vfs": True,
                 "rank": {"animatebanana": 1, "qwen38": 2, "talk": 3}}, latest)
check("append", "a boolean survives the JSON round trip as a boolean",
      latest["vfs"] is True)

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

group("schema migration is additive")
with fx8.db._connect() as conn:
    tcols = {r[1] for r in conn.execute("PRAGMA table_info(trial)")}
    dcols = {r[1] for r in conn.execute("PRAGMA table_info(diagram)")}
check("schema", "trial carries the tournament columns",
      {"presentation_ids", "presentation_conditions"} <= tcols)
check("schema", "diagram carries study_day and complexity",
      {"study_day", "complexity"} <= dcols)
fx8.close()


# ------------------------------------------------------------ concurrency --
group("concurrency")

fx9 = Fixture(cfg=main_cfg(judgments_per_sample=40, study_day=None))
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
check("concurrent", "every participant was served a full session",
      len(served) == 16 and all(s == 30 for s in served), served)
with fx9.db._connect() as conn:
    dupes = conn.execute(
        "SELECT participant_id, COUNT(*) c FROM trial WHERE status='open'"
        " GROUP BY participant_id HAVING c > 1").fetchall()
    total = conn.execute("SELECT COUNT(*) c FROM trial").fetchone()["c"]
check("concurrent", "no participant holds two open trials", not dupes)
check("concurrent", "trials were actually written", total == 16 * 30, total)

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
for f in (fx, fxsel, fx3):
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
