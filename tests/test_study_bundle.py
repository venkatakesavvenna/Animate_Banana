"""Study bundle + timeline tests.

Offline, no model calls. Run from the repo root:

    PYTHONPATH=src /fsxvision_new/venkat.kesav/environments/study/bin/python \
        tests/test_study_bundle.py

The timeline invariants are the point of this file. A caption that drifts
against the frame it describes makes Experiment 2's "narration alignment"
unratable, and the failure is invisible by inspection -- it looks like a
working player.
"""
import json
import os
import sys
from pathlib import Path

from img_2_svg_pretraining.study.timeline import (
    READING_FLOOR_SECONDS, build_timeline, reading_seconds, step_durations)
from img_2_svg_pretraining.study.build.build_bundle import (
    _frame_sort_key, _sorted_frames, bucket, narrative_id, stratification)

BUNDLE = Path(os.environ.get("STUDY_BUNDLE", "data/study_bundles/pilot-v2"))

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

RATIOS = [(17, 19), (41, 7), (24, 24), (1, 5), (10, 1), (7, 30), (23, 22), (2, 2)]
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
check("derived", "stratification of a missing xml is empty, not a crash",
      stratification(Path("/does/not/exist.xml")) == {})


# ----------------------------------------------------------- built bundle --
if not (BUNDLE / "manifest.json").exists():
    group("built bundle")
    check("bundle", "manifest exists", False, "build it first: %s" % BUNDLE)
else:
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    narratives = manifest["narratives"]
    media_dir = BUNDLE / "media"

    group("built bundle: integrity")
    check("bundle", "has narratives", len(narratives) > 0, len(narratives))
    check("bundle", "every narrative_id is unique",
          len({n["narrative_id"] for n in narratives}) == len(narratives))
    check("bundle", "every diagram referenced is declared",
          {n["diagram_id"] for n in narratives}
          <= {d["diagram_id"] for d in manifest["diagrams"]})

    missing = [m for n in narratives for m in n["frames"]
               if not (media_dir / ("%s.webp" % m)).exists()]
    check("bundle", "every referenced frame exists on disk", not missing, missing[:3])

    figures = [d["figure_media_id"] for d in manifest["diagrams"]]
    check("bundle", "every figure exists on disk",
          all((media_dir / ("%s.webp" % f)).exists() for f in figures))

    group("built bundle: timeline consistency")
    for n in narratives:
        tl, tag = n["timeline"], n["diagram_id"]
        check("bundle-tl", tag + " frame count agrees with media list",
              len(n["frames"]) == n["n_frames"] == len(tl["holds"]))
        check("bundle-tl", tag + " cue count agrees with step count",
              len(tl["cues"]) == n["n_steps"])
        # Holds are stored rounded to 3dp, so a long deck accumulates a little
        # drift against the total: 81 frames can differ by ~0.04s. Scale the
        # tolerance with frame count rather than asserting exact equality on
        # rounded values -- the invariant is "no frame is unaccounted for",
        # not "the rounding is lossless".
        check("bundle-tl", tag + " duration matches holds",
              abs(sum(tl["holds"]) - tl["duration"]) < 0.001 * len(tl["holds"]) + 0.01,
              (sum(tl["holds"]), tl["duration"], len(tl["holds"])))
        check("bundle-tl", tag + " cues ordered and closed",
              tl["cues"][0]["start"] == 0
              and abs(tl["cues"][-1]["end"] - tl["duration"])
                  < 0.001 * len(tl["holds"]) + 0.01)

    group("built bundle: blinding")
    # The media filename is the content hash, so a directory listing must not
    # betray style, method, condition or lineage.
    leaky = [p.name for p in media_dir.iterdir()
             if not p.stem.isalnum() or len(p.stem) != 16]
    check("blinding", "media filenames are bare 16-char hashes", not leaky, leaky[:3])

    tokens = ["progressive_reveal", "colour_pop", "alpha_masking", "gemini",
              "animatebanana", "with_context", "svg", "tikz"]
    names_blob = " ".join(p.name for p in media_dir.iterdir()).lower()
    check("blinding", "no condition token appears in any filename",
          not any(tok in names_blob for tok in tokens))

    group("built bundle: honest labelling")
    # A multi-arm bundle carries several context conditions by design; what
    # must hold is that each is a declared arm rather than derived per sample.
    check("labels", "every context_condition is a declared arm",
          {n["context_condition"] for n in narratives}
          <= {"with_context", "without_context", "not_applicable"},
          {n["context_condition"] for n in narratives})
    # Only pipeline narratives have a context tier -- the bench reference and
    # the end-to-end baseline were not produced by the narrative writer.
    pipeline_narratives = [n for n in narratives if "source" not in n]
    check("labels", "effective context tier is recorded for pipeline output",
          all("effective_context_tier" in n for n in pipeline_narratives),
          [n["narrative_id"] for n in pipeline_narratives
           if "effective_context_tier" not in n][:3])
    check("labels", "spoken_step_fraction present for the silent-step guard",
          all("spoken_step_fraction" in n for n in narratives))
    check("labels", "skipped list is present (may be empty)",
          isinstance(manifest.get("skipped"), list))


# ------------------------------------------------------------------ report --
failed = [r for r in _results if not r[2]]
print("\n%d checks, %d passed, %d failed" %
      (len(_results), len(_results) - len(failed), len(failed)))
for g, name, _, detail in failed:
    print("  FAIL [%s] %s   %s" % (g, name, str(detail)[:200]))
sys.exit(1 if failed else 0)
