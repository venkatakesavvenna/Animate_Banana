# AnimateBanana user study — operator guide

Flask + vanilla JS + sqlite under `src/img_2_svg_pretraining/study/`. Two
studies run on the same code, selected by bundle + config:

| study | bundle | config | experiments | who |
|---|---|---|---|---|
| **Main** | `data/study_bundles/main-v2` | `pipeline/configs/study_main.yaml` | `exp1` quality, `exp2` ranking | ~7 users/day, 3 days |
| **Selective cohort** | `data/study_bundles/selective-v1` | `pipeline/configs/study_selective.yaml` | `context`, `bench` | ~10 hand-picked users |

Python: `/fsxvision_new/venkat.kesav/environments/study/bin/python` (the
project venv declares 3.12, which this host does not have).

## Running it

```bash
# main study, day 1, on 8620 (defaults)          FRESH=1 wipes the DB
bash scripts/run_study_server.sh
# selective cohort on 8621
PORT=8621 BUNDLE=data/study_bundles/selective-v1 \
  CONFIG=src/img_2_svg_pretraining/pipeline/configs/study_selective.yaml \
  DB=data/study_runs/selective.db bash scripts/run_study_server.sh
# public link (Cloudflare quick tunnel) with restart-on-death
ADMIN_TOKEN=$(cat data/study_runs/admin_token_main.txt) PORT=8620 \
  DB=data/study_runs/main_day1.db bash scripts/study_watchdog.sh
#   -> data/study_runs/tunnel_url_8620.txt
# tests (need the dev servers on 8610/8611)
bash scripts/run_study_tests.sh
```

Databases and logs live in `data/study_runs/` — never `/tmp`, which is swept.
The admin token for the public instance is `data/study_runs/admin_token_main.txt`
(600, outside git); `devtoken` is only for local dev servers.

### Day switching (main study)

`study_main.yaml` carries `study_day: 1`. Each figure in `main-v1` is stamped
with the day it belongs to (`data/study_runs/main_selection.json`: 10 per day,
one of each style, 6 complex / 4 easy by structural element count, median 75).
Each morning bump `study_day` and restart; the scheduler only offers that day's
figures, so day-2 participants never meet day-1 stimuli. A config change is a
new config version in the DB — that is deliberate.

## The experiments

**exp1 — Animation quality** (absolute screen). Figure left, one animation with
its narration right, the **animation style in bold** above the player with a
one-line summary and a collapsed "What this style requires" (verbatim from the
ASCS style adapters, `study/styles.yaml`). All questions on one page, revealed
progressively (`show_if` in `questions/exp1.yaml`):

1. `vfs` — shows all elements correctly? Yes/No. **No ends the trial.**
2. `ascs` — follows the named style? Yes/No. **No ends the trial.**
3. `sss` — next-element selection sensible? 0–10 dropdown.
4. `nas` — narration aligned? 0–10 dropdown.

Three methods: AnimateBanana (gemini-3.7-flash), Qwen 3.8 27B, and Gemini 3.1
Pro (the closed-weight baseline, delivered as a folder of SMIL-animated SVGs
and added with `--dropin`; it covers 19 of the 30 figures). **Capped at 10
trials per participant for now** (`samples_per_experiment`), so each person
rates 10 figures under a mix of methods, least-judged cells first. Raise the
cap to `10 × methods` to have everyone rate every figure under every method:
the scheduler already serves fresh figures first and brings a figure back
under another method only after all others have been seen.

**exp2 — Which would you use?** (tournament screen, captions off; capped at
10 per participant). Two
animations side by side; the participant picks one; the loser fades out and
the third takes its place; pick again. Ranks: final winner 1, the one beaten in
round 2 gets 2, the first loser 3. Arms: AnimateBanana, Qwen 3.8, and the
**original conference talk** (decoded from `data/Original_Presentations.zip` at
1 fps, capped at 240 frames). Stored as `round1_pick`, `round2_pick`, `rank`.
Reported as Personal Utility Win Rate (PUWR).

**context — Narration with vs without context** (pairwise, blind, sides
randomised). Same AnimateBanana animation, narration from the +K and −K
pipelines. Question: which narration is more insightful — First / Second /
Both are equally useful (`choice_pair`, values A/B/tie).

**bench — Do the corrections help?** (pairwise, **not blind**). Original
uncorrected animation always LEFT, human-verified reference always RIGHT, with
the labels on screen. Yes/No.

`familiarity` is only asked where a question set says `familiarity: true`; none
of the four does now.

## Bundles

A bundle is a frozen, content-addressed stimulus set: `manifest.json` +
`media/<sha16>.webp` + per-narrative frame decks and `timeline.json`. The
runtime never imports the pipeline; only `study/build/` may.

```bash
V=/fsxvision_new/venkat.kesav/environments/study/bin/python
CFG=src/img_2_svg_pretraining/pipeline/configs
# main: two methods + the talks, restricted to the selection
PYTHONPATH=src $V -m img_2_svg_pretraining.study.build.build_bundle \
  --config "$CFG/bench_zs_or_gemini37flash.yaml=method:animatebanana" \
  --config "$CFG/bench_zs_svg_qwen38_27b.yaml=method:qwen38" \
  --style progressive_reveal --style colour_pop --style alpha_masking \
  --style hopping_bounding_box --style sliding_bounding_box \
  --talks-zip data/Original_Presentations.zip \
  --selection data/study_runs/main_selection.json \
  --name main-v1 --out data/study_bundles/main-v1
# selective cohort: +K / -K arms plus the verified reference, v3 set
PYTHONPATH=src $V -m img_2_svg_pretraining.study.build.build_bundle \
  --config "$CFG/bench_v3_or_svg.yaml=arm:with_context" \
  --config "$CFG/bench_v3_or_svg_nok.yaml=arm:without_context" \
  --style ... --reference --only <caption-bearing ids> \
  --name selective-v1 --out data/study_bundles/selective-v1
```

`--config path=method:X` labels every narrative from that config with method
X; `=arm:Y` sets the context condition. An arm generated outside the pipeline
comes in through `--dropin "<root>=method:gemini31pro"` (`study/build/dropin.py`):
a folder of `SVGs/<StyleDir>/<id>.html` frame-group SVGs animated with SMIL,
`Narration_json/<id>.json`, and optional pre-rendered `Video_Frames`. Frames are
taken by pausing the SVG timeline and seeking each step's midpoint, which
reproduces their own renders pixel for pixel; setting `opacity` by hand does
not work because the SMIL animation owns that attribute.

Facts about the current bundles:

- `main-v1`: 30 figures × {animatebanana, qwen38, talk} = 90 narratives, no
  skips. Talks: 35 in the zip, 32 shared with both models on the 193 set, 30
  used (dropped `WACV_2022_set5_000025`, `_000028`). Style comes from
  `data/zs_style_map.json`, not the zip's folder names (they disagree on 15).
  Colour pop has one talk-bearing figure in the whole set.
- `selective-v1`: 13 figures (the v3 samples with a caption AND a sequence in
  both lineages; `arch_arxiv_db_ir_2025_000000611` has no −K sequence). 13 is
  below the 15 the design asks for — the pool is what v3 has.
- 13 of the v3 reference narrations were saved with a trailing ``` fence;
  `_loads_lenient` strips it rather than dropping a fifth of the reference set.
- A dataset "title" containing TeX (`\LaTeX\ Guidelines for Author Response`)
  is template junk and is dropped.

## Scheduling

Experiment-major, in `experiment_order`. Cells retire at
`judgments_per_sample`; a retired cell is finished, never over-filled. A
participant is served each figure once per experiment — except in absolute
experiments, where a figure comes back under each further method after every
other figure has been seen. Assignment is `BEGIN IMMEDIATE`, persisted before
return, one open trial per participant (partial unique index), abandoned trials
reaped after `open_trial_ttl_seconds` (900).

Pairwise sides are randomised from a recorded seed except `FIXED_SIDES`
(`bench`). A tournament keeps the design order: the first two meet first.

## Blinding

Media is addressed by content hash; the trial payload carries only slot
letters. A test scans payloads for `qwen`, `talk`, `animatebanana`, `gemini`,
`without_context`, `lineage`. The one deliberate exception is `bench`, whose
`side_labels` say "Original (uncorrected)" / "Verified and corrected" on
purpose.

## Data model

sqlite WAL. `participant`, `participant_pii` (separate, never exported),
`diagram` (with `study_day`, `complexity`), `narrative`, `trial`
(`presentation_ids` / `presentation_conditions` JSON lists, so a slot count of
three is nothing special), `response` (append-only; revisions append),
`event`, `calibration_attempt`, `qc_flag`, `participant_annotation`,
`study_config`. Analysis reads the last response per question; the log keeps
every version.

## Judge correlation — what exists

`data/judge_exports/zero_shot_193_judge/<model>/{GPS,NAS,SSS}/<id>.json` holds
Kimi-judge scores for the four locally served models. There are **no VFS or
ASCS judge scores**, and **nothing for gemini-3.7-flash** (the AnimateBanana
arm). Human–LLM correlation on SSS/NAS is possible for Qwen 3.8 today; for
AnimateBanana the judge run has to happen first.

## Tests

Four suites, no model calls, run from the repo root against the dev servers
(`8610` main, `8611` selective — `scripts/run_study_tests.sh` starts what is
missing). They register their own participants and restore what they touch.

## Known limits

- Gemini 3.1 Pro covers 19 of the 30 main-study figures; the other 11 have
  two methods in exp1. exp2 is unchanged (AnimateBanana, Qwen 3.8, talk).
- Calibration has no expert key yet; the prep page skips straight to the study.
- `selective-v1` has 13 pairs, not 15.
- "AANS" in the design document does not exist as a metric in the codebase;
  the export has SSS and NAS separately.
