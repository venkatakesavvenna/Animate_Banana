"""Study bundle + timeline tests.

Offline, no model calls, no bundle build. Run from the repo root:

    PYTHONPATH=src /fsxvision_new/venkat.kesav/environments/study/bin/python \
        tests/test_study_bundle.py

Two real bundles are inspected in place and never rebuilt (a build decodes
talks and reference videos, and takes minutes):

    data/study_bundles/main-v1       30 figures x {animatebanana, qwen38, talk}
    data/study_bundles/selective-v1  13 figures x {+K, -K, verified reference}

The timeline invariants are still the point of this file. A caption that
drifts against the frame it describes makes "narration alignment" unratable,
and the failure is invisible by inspection -- it looks like a working player.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from img_2_svg_pretraining.study.timeline import (
    READING_FLOOR_SECONDS, build_timeline, reading_seconds, step_durations)
from img_2_svg_pretraining.study.build import build_bundle as bb
from img_2_svg_pretraining.study.build.build_bundle import (
    MIN_BASELINE_FRAMES, TALK_MAX_FRAMES, BuildLog, MediaStore, _frame_sort_key,
    _loads_lenient, _sorted_frames, bucket, build_baseline_narrative,
    build_reference_narrative, build_talk_narrative, narrative_id, stratification)
from img_2_svg_pretraining.study.config import PAIR_AXIS, TOURNAMENT

MAIN = Path(os.environ.get("STUDY_BUNDLE_MAIN", "data/study_bundles/main-v1"))
SELECTIVE = Path(os.environ.get("STUDY_BUNDLE_SELECTIVE",
                                "data/study_bundles/selective-v1"))
SELECTION = Path(os.environ.get("STUDY_SELECTION", "data/study_runs/main_selection.json"))

_results = []


def check(group, name, cond, detail=""):
    _results.append((group, name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          "" if cond else "   <- " + str(detail)[:160]))


def group(title):
    print("\n== %s ==" % title)


def nodes_for(n_steps, duration=3.0, words=8, silent=()):
    return [{"narrative": "" if i in silent else " ".join(["word"] * words),
             "duration": duration} for i in range(n_steps)]


# ---------------------------------------------------------------- timeline --
group("timeline invariants across frame/step ratios")

RATIOS = [(17, 19), (41, 7), (24, 24), (1, 5), (10, 1), (7, 30), (23, 22), (2, 2),
          (240, 240)]      # a four-minute talk at 1 fps
for n_frames, n_steps in RATIOS:
    t = build_timeline(n_frames, nodes_for(n_steps))
    tag = "f=%d s=%d" % (n_frames, n_steps)
    check("timeline", tag + " every frame held", len(t.holds) == n_frames)
    check("timeline", tag + " every frame mapped to a step",
          len(t.frame_step) == n_frames and set(t.frame_step) <= set(range(n_steps)))
    check("timeline", tag + " one cue per step", len(t.cues) == n_steps)
    check("timeline", tag + " holds sum to duration",
          abs(sum(t.holds) - t.duration) < 1e-6)
    check("timeline", tag + " starts at zero", abs(t.cues[0].start) < 1e-6)
    check("timeline", tag + " ends at duration",
          abs(t.cues[-1].end - t.duration) < 1e-6)
    check("timeline", tag + " cues never overlap",
          all(t.cues[i].end <= t.cues[i + 1].start + 1e-6
              for i in range(len(t.cues) - 1)),
          [(c.start, c.end) for c in t.cues[:6]])
    check("timeline", tag + " frame_step non-decreasing",
          all(t.frame_step[i] <= t.frame_step[i + 1]
              for i in range(len(t.frame_step) - 1)))

group("timeline pacing")

t = build_timeline(10, nodes_for(10, duration=0.1, words=40))
check("pacing", "authored duration stretched to reading time",
      all(h >= READING_FLOOR_SECONDS - 1e-6 for h in t.holds),
      "a 40-word caption cannot sit behind a 0.1s hold")

t = build_timeline(5, [{"narrative": "a b c", "duration": None} for _ in range(5)])
check("pacing", "missing durations fall back to reading time",
      t.timing_source == "estimated" and t.duration > 0, t.timing_source)

mixed = [{"narrative": "a b c", "duration": 3.0},
         {"narrative": "d e f", "duration": None}]
_, source = step_durations(mixed)
check("pacing", "mixed timing is reported as mixed", source == "mixed", source)

# A silent, authored step (a talk frame, a baseline frame) keeps its authored
# hold rather than being stretched to the silent-step floor.
silent = [{"narrative": "", "duration": 0.4} for _ in range(4)]
d, source = step_durations(silent)
check("pacing", "a silent authored step keeps its authored duration",
      d == [0.4] * 4 and source == "authored", (d, source))

check("pacing", "silent step still occupies time", reading_seconds("") > 0)
check("pacing", "reading time grows with word count",
      reading_seconds(" ".join(["w"] * 200)) > reading_seconds("w w w"))

group("timeline rejects impossible input")
for bad, label in [((0, nodes_for(3)), "zero frames"), ((5, []), "zero steps")]:
    try:
        build_timeline(*bad)
        check("guards", label + " raises", False, "no exception")
    except ValueError:
        check("guards", label + " raises", True)


# ------------------------------------------------------------------ build --
group("frame ordering")

names = ["frame-%d" % i for i in (1, 2, 3, 9, 10, 11, 20, 21)]
paths = [Path("/x/%s.png" % n) for n in names]
check("order", "numeric sort keeps 9 before 10",
      [p.stem for p in sorted(paths, key=_frame_sort_key)] == names,
      [p.stem for p in sorted(paths, key=_frame_sort_key)])
check("order", "lexical sort would NOT (guard is load-bearing)",
      [p.stem for p in sorted(paths)] != names)

group("derived fields")
check("derived", "bucket low/medium/high",
      (bucket(1, 5, 10), bucket(7, 5, 10), bucket(50, 5, 10)) == ("low", "medium", "high"))
check("derived", "narrative_id is stable",
      narrative_id("d", "s", "m", "c", "v", 1) == narrative_id("d", "s", "m", "c", "v", 1))
check("derived", "narrative_id separates conditions",
      narrative_id("d", "s", "m", "with_context", "v", 1)
      != narrative_id("d", "s", "m", "without_context", "v", 1))
check("derived", "narrative_id separates methods",
      narrative_id("d", "s", "animatebanana", "c", "v", 1)
      != narrative_id("d", "s", "qwen38", "c", "v", 1))
check("derived", "stratification of a missing xml is empty, not a crash",
      stratification(Path("/does/not/exist.xml")) == {})

group("lenient JSON for fenced reference narrations")
obj = {"sequence": [{"narrative": "a"}]}
check("lenient", "plain JSON parses", _loads_lenient(json.dumps(obj)) == obj)
check("lenient", "a trailing ``` fence is tolerated",
      _loads_lenient(json.dumps(obj) + "\n```") == obj)
check("lenient", "a full ```json block is tolerated",
      _loads_lenient("```json\n" + json.dumps(obj) + "\n```\n") == obj)
try:
    _loads_lenient("no braces here")
    check("lenient", "text without an object raises ValueError", False, "no exception")
except ValueError:
    check("lenient", "text without an object raises ValueError", True)


# ----------------------------------------------------- builders, offline --
group("baseline builder: the near-static guard")
work = Path(tempfile.mkdtemp(prefix="_t_bundle_"))
try:
    media = MediaStore(work / "media")
    sample = SimpleNamespace(id="diagX", directory=work / "diagX")

    def render(cell, n):
        frames = work / "base" / cell / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            im = Image.new("RGB", (64, 48), (i * 9 % 255, 20, 200))
            im.save(frames / ("frame-%d.png" % (i + 1)))

    render("diagX__colour_pop", MIN_BASELINE_FRAMES - 1)
    log = BuildLog()
    rec = build_baseline_narrative(sample, "colour_pop", work / "base", media, log)
    check("baseline", "fewer than MIN_BASELINE_FRAMES frames is excluded",
          rec is None and len(log.skipped) == 1, (rec, log.skipped))
    check("baseline", "the exclusion is recorded as a generation failure",
          log.skipped and "near-static" in log.skipped[0]["reason"], log.skipped)

    render("diagX__alpha_masking", MIN_BASELINE_FRAMES + 6)
    log = BuildLog()
    rec = build_baseline_narrative(sample, "alpha_masking", work / "base", media, log,
                                   target_seconds=48.0)
    check("baseline", "enough frames builds a narrative", rec is not None, log.skipped)
    if rec:
        check("baseline", "paced to the target wall-clock",
              abs(rec["timeline"]["duration"] - 48.0) < 0.05, rec["timeline"]["duration"])
        check("baseline", "labelled as the baseline method",
              rec["method"] == "baseline" and rec["source"] == "baseline_sonnet")
        check("baseline", "silent throughout",
              rec["spoken_step_fraction"] == 0.0 and rec["narration_words"] == 0)
        check("baseline", "frames are content hashes on disk",
              all((work / "media" / (m + ".webp")).exists() for m in rec["frames"])
              and all(len(m) == 16 for m in rec["frames"]))

    group("talk and reference builders skip cleanly when input is missing")
    empty_zip = work / "talks.zip"
    with zipfile.ZipFile(empty_zip, "w"):
        pass
    log = BuildLog()
    rec = build_talk_narrative(sample, "colour_pop", empty_zip, media, log, work)
    check("talk", "a zip without this sample's talk is a recorded skip",
          rec is None and log.skipped and "no original talk" in log.skipped[0]["reason"],
          log.skipped)
    bad_zip = work / "bad.zip"
    bad_zip.write_bytes(b"not a zip")
    log = BuildLog()
    rec = build_talk_narrative(sample, "colour_pop", bad_zip, media, log, work)
    check("talk", "an unreadable zip is a recorded skip, not a crash",
          rec is None and log.skipped and "unreadable" in log.skipped[0]["reason"],
          log.skipped)
    log = BuildLog()
    rec = build_reference_narrative(sample, "colour_pop", "svg", media, log, work)
    check("reference", "a sample without a reference video is a recorded skip",
          rec is None and log.skipped
          and "no style-matched reference video" in log.skipped[0]["reason"],
          log.skipped)
finally:
    shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------- CLI parsing --
group("--config spec parsing and --selection stamping")
# `main()` parses and calls `build`; swapping `build` out keeps this offline.
captured = {}


def fake_build(configs, out, name, labels_path, method, context_arm, verification,
               reference=False, only=None, baseline_root=None, talks_zip=None,
               labels_json=None):
    captured.update(dict(configs=configs, out=out, name=name, method=method,
                         arm=context_arm, reference=reference, only=only,
                         talks_zip=talks_zip, labels_json=labels_json))
    return {"diagrams": [], "narratives": [], "skipped": []}


real_build, real_argv = bb.build, sys.argv
try:
    bb.build = fake_build
    sys.argv = ["build_bundle", "--out", "/tmp/_t_never_written",
                "--config", "a.yaml=method:qwen38",
                "--config", "b.yaml=arm:without_context",
                "--config", "c.yaml=with_context",
                "--config", "d.yaml=method:talk,arm:with_context",
                "--config", "e.yaml",
                "--style", "progressive_reveal", "--style", "colour_pop",
                "--talks-zip", "talks.zip", "--reference"]
    bb.main()
    got = {(c, st): (arm, m) for c, st, arm, m in captured["configs"]}
    check("cli", "every config is enumerated per requested style",
          len(captured["configs"]) == 10 and
          {st for _, st, _, _ in captured["configs"]} == {"progressive_reveal", "colour_pop"},
          captured["configs"])
    check("cli", "=method:X labels that config's runs",
          got[("a.yaml", "colour_pop")] == ("not_applicable", "qwen38"), got)
    check("cli", "=arm:Y labels the context arm",
          got[("b.yaml", "colour_pop")] == ("without_context", "animatebanana"), got)
    check("cli", "a bare =with_context still means the arm",
          got[("c.yaml", "colour_pop")] == ("with_context", "animatebanana"), got)
    check("cli", "method and arm combine",
          got[("d.yaml", "progressive_reveal")] == ("with_context", "talk"), got)
    check("cli", "an unsuffixed config takes the global defaults",
          got[("e.yaml", "progressive_reveal")] == ("not_applicable", "animatebanana"), got)
    check("cli", "--talks-zip and --reference reach the builder",
          captured["talks_zip"] == "talks.zip" and captured["reference"] is True)
    check("cli", "without --selection nothing is stamped",
          captured["labels_json"] is None and captured["only"] is None)

    if SELECTION.exists():
        sel = json.loads(SELECTION.read_text(encoding="utf-8"))
        captured.clear()
        sys.argv = ["build_bundle", "--out", "/tmp/_t_never_written",
                    "--config", "a.yaml", "--selection", str(SELECTION)]
        bb.main()
        expected_ids = [sid for day in sel["days"] for sid in day]
        check("selection", "--selection restricts --only to its samples",
              sorted(captured["only"] or []) == sorted(expected_ids),
              (len(captured["only"] or []), len(expected_ids)))
        stamped = captured["labels_json"] or {}
        check("selection", "every selected sample is stamped with a day",
              all(stamped.get(s, {}).get("study_day") == d + 1
                  for d, ids in enumerate(sel["days"]) for s in ids))
        check("selection", "complexity is stamped from the selection",
              all(stamped[s]["complexity"] ==
                  ("complex" if sel["samples"][s]["complex"] else "easy")
                  for s in expected_ids))
        check("selection", "the element-count median rides along for the record",
              all(stamped[s].get("element_count_median") == sel["median_elements"]
                  for s in expected_ids))
    else:
        check("selection", "selection file present", False, str(SELECTION))
finally:
    bb.build, sys.argv = real_build, real_argv


# ----------------------------------------------------------- built bundles --

def bundle_checks(bundle: Path, tag: str):
    if not (bundle / "manifest.json").exists():
        group("built bundle %s" % tag)
        check(tag, "manifest exists", False, "build it first: %s" % bundle)
        return None
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    narratives = manifest["narratives"]
    media_dir = bundle / "media"

    group("built bundle %s: integrity" % tag)
    check(tag, "bundle_id matches its directory", manifest["bundle_id"] == bundle.name,
          manifest["bundle_id"])
    check(tag, "has narratives", len(narratives) > 0, len(narratives))
    check(tag, "every narrative_id is unique",
          len({n["narrative_id"] for n in narratives}) == len(narratives))
    check(tag, "every narrative_id is reproducible from its labels",
          all(n["narrative_id"] == narrative_id(
              n["diagram_id"], n["animation_style"], n["method"],
              n["context_condition"], n["verification_state"],
              n.get("narrative_version", 1)) for n in narratives))
    check(tag, "every diagram referenced is declared",
          {n["diagram_id"] for n in narratives}
          <= {d["diagram_id"] for d in manifest["diagrams"]})
    check(tag, "every declared diagram has stimuli",
          {d["diagram_id"] for d in manifest["diagrams"]}
          <= {n["diagram_id"] for n in narratives})
    check(tag, "configs record the method each run was labelled with",
          all("method" in c and "context_arm" in c for c in manifest["configs"]))

    missing = [m for n in narratives for m in n["frames"]
               if not (media_dir / ("%s.webp" % m)).exists()]
    check(tag, "every referenced frame exists on disk", not missing, missing[:3])
    figures = [d["figure_media_id"] for d in manifest["diagrams"]]
    check(tag, "every figure exists on disk",
          all((media_dir / ("%s.webp" % f)).exists() for f in figures))
    check(tag, "frame dimensions are recorded",
          all(n.get("frame_w") and n.get("frame_h") for n in narratives))

    group("built bundle %s: timeline consistency" % tag)
    bad = defaultdict(list)
    for n in narratives:
        tl, nid = n["timeline"], n["narrative_id"]
        if not (len(n["frames"]) == n["n_frames"] == len(tl["holds"])):
            bad["frames"].append(nid)
        if len(tl["cues"]) != n["n_steps"]:
            bad["cues"].append(nid)
        # Holds are stored rounded to 3dp, so a long deck accumulates a little
        # drift against the total. Scale the tolerance with frame count.
        tol = 0.001 * len(tl["holds"]) + 0.01
        if abs(sum(tl["holds"]) - tl["duration"]) >= tol:
            bad["duration"].append(nid)
        if not (tl["cues"][0]["start"] == 0
                and abs(tl["cues"][-1]["end"] - tl["duration"]) < tol):
            bad["closed"].append(nid)
        if any(tl["cues"][i]["end"] > tl["cues"][i + 1]["start"] + 1e-6
               for i in range(len(tl["cues"]) - 1)):
            bad["overlap"].append(nid)
    check(tag, "frame count agrees with media list everywhere", not bad["frames"], bad["frames"][:3])
    check(tag, "cue count agrees with step count everywhere", not bad["cues"], bad["cues"][:3])
    check(tag, "duration matches holds everywhere", not bad["duration"], bad["duration"][:3])
    check(tag, "cues ordered and closed everywhere", not bad["closed"], bad["closed"][:3])
    check(tag, "cues never overlap", not bad["overlap"], bad["overlap"][:3])

    group("built bundle %s: blinding" % tag)
    leaky = [p.name for p in media_dir.iterdir()
             if not p.stem.isalnum() or len(p.stem) != 16]
    check(tag, "media filenames are bare 16-char hashes", not leaky, leaky[:3])
    tokens = ["progressive_reveal", "colour_pop", "alpha_masking", "gemini", "qwen",
              "talk", "animatebanana", "with_context", "verified", "svg", "tikz"]
    names_blob = " ".join(p.name for p in media_dir.iterdir()).lower()
    check(tag, "no condition token appears in any filename",
          not any(tok in names_blob for tok in tokens))

    group("built bundle %s: honest labelling" % tag)
    check(tag, "every context_condition is a declared arm",
          {n["context_condition"] for n in narratives}
          <= {"with_context", "without_context", "not_applicable"},
          {n["context_condition"] for n in narratives})
    check(tag, "every verification_state is a declared side",
          {n["verification_state"] for n in narratives}
          <= {"pre_verification", "verified", "not_applicable"})
    pipeline_narratives = [n for n in narratives if "source" not in n]
    check(tag, "effective context tier is recorded for pipeline output",
          all("effective_context_tier" in n for n in pipeline_narratives),
          [n["narrative_id"] for n in pipeline_narratives
           if "effective_context_tier" not in n][:3])
    check(tag, "spoken_step_fraction present for the silent-step guard",
          all("spoken_step_fraction" in n for n in narratives))
    check(tag, "skipped list is present (may be empty)",
          isinstance(manifest.get("skipped"), list))
    return manifest


main = bundle_checks(MAIN, "main-v1")
if main:
    group("main-v1: the ranking design is complete")
    by_diag = defaultdict(dict)
    for n in main["narratives"]:
        by_diag[n["diagram_id"]][n["method"]] = n
    contenders = set(TOURNAMENT["exp2"])
    check("main", "methods are exactly the three tournament contenders",
          {n["method"] for n in main["narratives"]} == contenders,
          Counter(n["method"] for n in main["narratives"]))
    check("main", "every figure has all three contenders",
          all(set(v) == contenders for v in by_diag.values()),
          [d for d, v in by_diag.items() if set(v) != contenders][:3])
    check("main", "each contender appears once per figure",
          Counter(n["method"] for n in main["narratives"])
          == {m: len(by_diag) for m in contenders})
    check("main", "the three contenders of a figure share its style",
          all(len({n["animation_style"] for n in v.values()}) == 1
              for v in by_diag.values()))
    check("main", "our animation and the baseline are different renders",
          all(v["animatebanana"]["frames"] != v["qwen38"]["frames"]
              for v in by_diag.values()))
    talks = [n for n in main["narratives"] if n["method"] == "talk"]
    check("main", "talks are decoded from the original lecture",
          all(n.get("source") == "original_talk" and n.get("talk_seconds", 0) > 0
              for n in talks))
    check("main", "talks carry no narration track",
          all(n["spoken_step_fraction"] == 0 and n["narration_words"] == 0
              and all(not c["text"] for c in n["timeline"]["cues"]) for n in talks))
    check("main", "talks are sampled to at most TALK_MAX_FRAMES",
          all(2 <= n["n_frames"] <= TALK_MAX_FRAMES for n in talks),
          sorted(n["n_frames"] for n in talks)[-3:])
    check("main", "talk pacing covers the talk's real duration",
          all(abs(n["timeline"]["duration"] - n["talk_seconds"]) < 1.0 for n in talks),
          [(n["timeline"]["duration"], n["talk_seconds"]) for n in talks[:3]])
    check("main", "no context or verification arms in the main study",
          all(n["context_condition"] == "not_applicable"
              and n["verification_state"] == "not_applicable"
              for n in main["narratives"]))

    group("main-v1: day and complexity stamping")
    days = Counter(d.get("study_day") for d in main["diagrams"])
    check("main", "every figure is stamped with a study day",
          None not in days and set(days) == {1, 2, 3}, days)
    check("main", "ten fresh figures per day", all(v == 10 for v in days.values()), days)
    check("main", "complexity is complex|easy on every figure",
          {d.get("complexity") for d in main["diagrams"]} == {"complex", "easy"},
          Counter(d.get("complexity") for d in main["diagrams"]))
    if SELECTION.exists():
        sel = json.loads(SELECTION.read_text(encoding="utf-8"))
        want = {sid: day for day, ids in enumerate(sel["days"], start=1) for sid in ids}
        check("main", "the stamped days agree with main_selection.json",
              all(want.get(d["diagram_id"]) == d["study_day"] for d in main["diagrams"])
              and set(want) == {d["diagram_id"] for d in main["diagrams"]})

selective = bundle_checks(SELECTIVE, "selective-v1")
if selective:
    group("selective-v1: pairs exist for both cohort experiments")
    by_cell = defaultdict(dict)
    for n in selective["narratives"]:
        by_cell[(n["diagram_id"], n["animation_style"])][
            (n["context_condition"], n["verification_state"])] = n
    check("sel", "one style per figure", len(by_cell) == len(selective["diagrams"]),
          (len(by_cell), len(selective["diagrams"])))
    for exp, (field_, (a, b)) in PAIR_AXIS.items():
        paired = [cell for cell, v in by_cell.items()
                  if {n[field_] for n in v.values()} >= {a, b}]
        check("sel", "%s pair (%s vs %s) exists for every figure" % (exp, a, b),
              len(paired) == len(by_cell), (len(paired), len(by_cell)))
    ours = [v[("with_context", "pre_verification")] for v in by_cell.values()
            if ("with_context", "pre_verification") in v]
    nok = [v[("without_context", "not_applicable")] for v in by_cell.values()
           if ("without_context", "not_applicable") in v]
    refs = [v[("not_applicable", "verified")] for v in by_cell.values()
            if ("not_applicable", "verified") in v]
    check("sel", "the +K run is relabelled pre_verification once a reference exists",
          len(ours) == len(by_cell), len(ours))
    check("sel", "the context pair is the SAME animation with different narration",
          len(nok) == len(ours) and all(
              v[("with_context", "pre_verification")]["frames"]
              == v[("without_context", "not_applicable")]["frames"]
              for v in by_cell.values()))
    check("sel", "the two narrations actually differ",
          all(v[("with_context", "pre_verification")]["timeline"]["cues"]
              != v[("without_context", "not_applicable")]["timeline"]["cues"]
              for v in by_cell.values()))
    check("sel", "references are decoded bench videos, paced by reading time",
          len(refs) == len(by_cell) and all(
              n.get("source") == "bench_reference"
              and n["timeline"]["timing_source"] == "estimated" for n in refs))
    check("sel", "our side keeps its authored timing (difference is recorded, not hidden)",
          all(n["timeline"]["timing_source"] == "authored" for n in ours))
    check("sel", "the reference is a different render from ours",
          all(v[("not_applicable", "verified")]["frames"]
              != v[("with_context", "pre_verification")]["frames"]
              for v in by_cell.values()))
    check("sel", "no ranking contenders in the selective cohort",
          {n["method"] for n in selective["narratives"]} == {"animatebanana"})


# ------------------------------------------------------------------ report --
failed = [r for r in _results if not r[2]]
print("\n%d checks, %d passed, %d failed" %
      (len(_results), len(_results) - len(failed), len(failed)))
for g, name, _, detail in failed:
    print("  FAIL [%s] %s   %s" % (g, name, str(detail)[:200]))
sys.exit(1 if failed else 0)
