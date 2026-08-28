# AnimateBanana user study — operator guide

Web tool for the human evaluation: five experiments over SVG-target animated
explainers, run on colleagues. Server-side trial assignment, blinded pairwise
comparisons, append-only responses.

Built as `src/img_2_svg_pretraining/study/`. It is a **separate app from the
pipeline** and never imports it at runtime — stimuli arrive through a frozen,
content-addressed bundle built offline.

---

## Running it

```bash
# start / restart (FRESH=1 wipes the scratch DB)
PORT=8607 BUNDLE=data/study_bundles/pilot-today DB=/tmp/study_pilot.db \
  bash scripts/run_study_server.sh

# all four suites
bash scripts/run_study_tests.sh

# load / stress
python scripts/simulate_participants.py --n 40 --threads 16 --stress
```

Everything runs on the **study venv**, not the project one:
`/fsxvision_new/venkat.kesav/environments/study/bin/python`.

| Instance | Port | Bundle | Purpose |
|---|---|---|---|
| Pilot | 8607 | `pilot-today` | 5 samples × 5 sections = 25 trials |
| Demo | 8609 | `pilot-demo1` | 1 sample × 5 sections = 5 trials |

Admin at `/admin`, token via `--admin-token` (`devtoken` from the runner).

---

## The five experiments

An experiment is **a question set plus a caption flag** — nothing in the code
branches on which one is running. Question text lives in
`study/questions/exp{1..5}.yaml` and can be rewritten without touching Python.

| Exp | Screen | Compares | Captions |
|---|---|---|---|
| 1 | absolute | our animation | off |
| 2 | absolute | the same animation | on |
| 3 | pairwise | (+K) vs (−K) narration | on |
| 4 | pairwise | ours vs Sonnet-5 end-to-end | **off, both sides** |
| 5 | pairwise | ours vs bench human-verified | on |

Exp1 and Exp2 show **pixel-identical visuals** — same media, caption layer
switched off — so their scores are directly comparable. Our animations carry no
burned-in narration (verified against the animation source), so nothing is
cropped.

Exp4 hides captions on **both** sides. The baseline produces no narration, so
showing them would leave one panel with a caption bar and the other empty,
un-blinding the comparison before anyone watches.

---

## Stimulus provenance

| Source | Produced by | Used as |
|---|---|---|
| `pipeline` | AnimateBanana stage 1→3 | Exp1/2 subject; Exp3 both arms; Exp5 pre-verification |
| `bench_reference` | AnimateBench human-checked mp4, decoded to frames | Exp5 verified side |
| `baseline_sonnet` | one Sonnet-5 call, figure → animated SVG | Exp4 baseline side |

**Only pipeline output is rated in isolation.** `build_pool` excludes
`verification_state == "verified"` and `method != "animatebanana"` from the
absolute arms — otherwise participants would score a human-corrected or
competitor animation as though it were ours, and the RQ1/RQ2 means would
silently absorb it.

`pilot-today`: 5 diagrams, 20 narratives (10 pipeline, 5 baseline, 5
reference), 508 media files, 15.5 MB.

| Sample | Style | Density | Raster |
|---|---|---|---|
| `Paper2Fig_pipe_diff_000000673` | progressive_reveal | medium | yes |
| `arch_arxiv_db_ir_2025_000000587` | progressive_reveal | medium | no |
| `arch_arxiv_db_ir_2025_000000589` | hopping_bounding_box | high | yes |
| `arch_arxiv_db_ir_2025_000000618` | colour_pop | high | yes |
| `arch_arxiv_db_ir_2026_000000715` | sliding_bounding_box | medium | no |

---

## Building a bundle

```bash
python -m img_2_svg_pretraining.study.build.build_bundle \
  --config <cfg>=with_context --config <cfg_nok>=without_context \
  --style progressive_reveal --style colour_pop --style alpha_masking \
  --style hopping_bounding_box --style sliding_bounding_box \
  --reference --baseline-root data/baseline_sonnet \
  --only <id1>,<id2> --out data/study_bundles/<name>
```

A bundle is **immutable**. New stimuli make a new bundle version; an existing
one is never patched, because collected trials point at media by content hash.

`--config <path>=<arm>` labels that config's runs as an Experiment 3 arm. The
`--reference` and `--baseline-root` sides belong to the *diagram*, so they are
emitted once per (diagram, style) regardless of how many configs are in play.

### Why the runtime cannot import the pipeline

`pipeline/inspector/compare.py::_paths_for` mutates a shared config to resolve
a lineage and never restores it. In a read-only viewer that races into showing
the wrong video; in a study it serves the **wrong condition** to a participant
and the response is unrecoverable — nothing on disk records what they actually
saw. Resolving a lineage inside a request handler is the failure this package
is arranged to prevent. Enforced by a test that imports the whole runtime with
`img_2_svg_pretraining.pipeline` blocked at the import hook.

---

## Playback

The exported mp4 is **never served**. At `fps: 2` a 15-frame animation plays in
7.5s against ~40s of authored narration — unreadable, and "narration alignment"
would be unratable. The player drives the frame deck from step durations
computed at build time.

- **Slider is the default.** Reaching the final frame marks it viewed; the old
  clock-based rule could strand someone whose tab was backgrounded, and could
  not work for a slider at all.
- **Questions stay hidden** until every player has been taken to the end.
- **Arrow keys** step frames and light their indicator.
- **Video tab** plays through automatically at 0.5× by default.
- A **section interstitial** appears once per section, keyed in `localStorage`
  so it survives a reload.

`timeline.py` mirrors `pipeline/export/narration.py::align_frames_to_clips`
rather than importing it. The duplication is deliberate: if the two disagree, a
caption drifts from the frame it describes and nobody notices until a
participant rates a lie.

Baseline animations are paced to **our** duration for the same figure, spread
over their frames. A flat 0.5s/frame made Exp4 run 13s against 57s — "which is
better paced" would then measure a timing constant, not either system.

---

## Scheduling

Experiment-major: all of a participant's Exp1 samples, then Exp2, and so on.
Chosen over diagram-major because familiarity with a figure biases every later
judgment about it.

- A **diagram appears at most once per experiment**; repeats **across**
  experiments are intended (Exp2 rates narration on the same videos Exp1 rated
  visually, which is what makes them correlatable).
- **Retirement, not deficit ranking.** A cell retires at
  `judgments_per_sample` and stops being offered; replacements come from the
  same stratification class. The quota is hard — an exhausted arm yields to the
  next rather than over-filling.
- Assignment runs in one `BEGIN IMMEDIATE` transaction and is **persisted
  before it returns**, so a reload resumes the identical trial. A partial
  unique index makes a double-click impossible to turn into two open trials.
- Abandoned trials are reaped after `open_trial_ttl_seconds` so a closed tab
  does not hold a quota slot forever; the row is kept as dropout evidence.

**Capacity is set by the pool, not by recruitment:**
`pool × quota ÷ samples_per_experiment` participants. The progress target is
bounded by **distinct diagrams**, not cells — an arm with 10 cells over 5
figures can only serve 5.

---

## Data model

sqlite, WAL. `trial`, `response`, `event`, `qc_flag`, `calibration_attempt`,
`participant_annotation` are **append-only** — there is deliberately no
`update_response` or `delete_response`, and a test asserts their absence.

- **A revision is an append.** Participants are expected to revise as they grow
  familiar with a figure; analysis reads the latest per (trial, question) and
  the log keeps the trail.
- **Exclusion is an annotation**, never a mutation. Raw responses are
  byte-identical before and after, so any result can be recomputed with and
  without a QC rule.
- **PII is a separate table.** `participant_pii` holds name and roll number and
  is **never exported**; `participant_id` is opaque everywhere else.
- Config is versioned; every trial records the version it ran under.

---

## Blinding

- Media is **content-addressed** (`media/<sha256[:16]>.webp`), so even a
  directory listing carries no style, lineage, method or condition.
- The trial payload carries only slot letters, never `narrative_id`, `method`,
  `context_condition`, `verification_state`, lineage or any path. The raw style
  slug is stripped too — `style_name` carries the meaning without a
  cache-shaped token.
- A/B position is randomised per trial from a recorded seed; both conditions
  are stored so analysis recovers preference from **condition**, not position.
- Admin is a separate blueprint behind its own token. Admin views are
  unblinded, so they must not be reachable from a participant session by
  flipping a parameter.

---

## Calibration

There are no authored examples. Researchers annotate through the same tool with
`is_expert`, and those sessions become both the worked examples and the marking
key. Likert scoring tolerates ±1 (a 4-vs-5 disagreement is not a
misunderstanding of the task); categorical answers are exact. The key never
reaches the browser.

**With no expert session recorded, calibration reports itself unavailable and
lets participants through.** Blocking the study on missing data would be worse;
a quiz that silently passes everyone while looking like a gate would be worse
still. **Experts are exempt** — otherwise the first expert to submit a trial is
locked out by their own answers.

---

## Tests

344 checks across four suites, repo convention (plain assertion scripts, run
from the root, no model calls).

| Suite | Checks | Covers |
|---|---|---|
| `test_study_bundle.py` | 168 | timeline invariants, frame ordering, manifest integrity |
| `test_study_scheduler.py` | 60 | ordering, retirement, quota, blinding, append-only, concurrency |
| `test_study_app.py` | 72 | HTTP lifecycle, media, submit gate, admin, calibration |
| `test_study_ui.py` | 44 | browser: layout, player, question flow, navigation |

The UI suite is load-bearing, not decorative: the offline and HTTP suites both
passed while the trial screen was unusable — a fixed `100vh` shell with
`overflow:hidden` that trapped the questions out of reach. Nothing short of
rendering it catches that.

Headless Chromium needs its shared libraries, which are not installed
system-wide (no root). They were fetched as `.deb`s and extracted under the
study venv; `run_study_tests.sh` sets `LD_LIBRARY_PATH` automatically.

---

## Known limits

- **Exp3 is a narration ablation.** Only `narrative_writer` receives paper
  context — `code_converter`, `parser`, `sequencer` and `designer` never do
  (`strategizer` has `context_tier` but is dropped from stage 2). Both arms
  therefore show **byte-identical frames** and differ only in narration. Say so
  in the paper.
- **Exp4 is visuals-only.** The baseline has no narration stage, so "which
  would you use to explain this figure" is answered on animation alone.
- **Exp5's two sides are paced differently by necessity.** Our side uses
  authored durations; the bench reference has `timestamp` but no `duration`, so
  it falls back to reading time. Recorded as `timing_source`.
- **14 of 47 animations open on a near-blank canvas** (all `progressive_reveal`)
  and hold it several seconds while the narration already describes content.
  That is a genuine narration/visual misalignment the study should measure —
  deliberately not "fixed".
- **Five of 15 source figures are under 800 px wide** (smallest 388×413). The
  rendered animation is sharper than the source figure; zoom cannot reveal
  detail the thumbnail lacks.
- **Attention checks and the analysis pipeline are not built.** Schema and
  config support them; `attention_check_fraction` is 0.0 and nothing writes to
  `qc_flag` yet. The export endpoint is version-stamped but no code computes RQ
  tables, CIs or `main_paper_numbers.json`.

---

## Generating more stimuli

**Exp3 (−K):** `bench_v3_or_svg_nok.yaml` differs from the main config in one
line (`narrative_writer.context_tier: image_only`). Only stage 2e re-runs;
frames are shared because `narrative_lineage` is not part of
`animation_lineage`, and the tier is folded into the lineage so the arms never
collide.

```bash
python -m img_2_svg_pretraining.pipeline.run_pipeline narrate \
  --config .../bench_v3_or_svg_nok.yaml --style <style> --only <id>
```

**Exp4 (baseline):** `study/build/baseline_sonnet.py`. One Sonnet-5 call per
figure, `max_tokens: 64000`.

```bash
python -m img_2_svg_pretraining.study.build.baseline_sonnet \
  --only "<id>:<style>,<id>:<style>" --out data/baseline_sonnet --render
```

Claude Sonnet 5 is in the **high-resolution vision tier**: 2576 px on the long
edge, and **coordinates map 1:1 to image pixels — no scale-factor math** (this
changed at Opus 4.7; Sonnet 4.6 and earlier capped at 1568 px). Every figure in
`animatebench_v3` is under 2576 px (max long edge 2069), so each is sent at
native resolution and the model's coordinates are directly SVG user units. The
prompt supplies the true pixel dimensions and requires a matching `viewBox`;
**do not add rescaling** — that is the class of change that silently
misaligns every element.

Two failure modes are recorded rather than hidden, and belong in RQ4's
generation-success statistics:

- **Truncation.** A response cut off before `</svg>` is a failure, not
  something to salvage — a half-written SVG would put a mangled figure in front
  of a participant and score the baseline on our repair job.
- **Near-static output.** Under `MIN_BASELINE_FRAMES` (6) the cell is excluded
  from pairing: a still image against a real animation measures the wrong
  thing.
