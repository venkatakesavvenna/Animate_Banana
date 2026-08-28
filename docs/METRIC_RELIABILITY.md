# Are the animation metrics any good?

The animation tree has enough coverage to stop asking what the metrics say and
start asking whether to believe them. Two doubts were raised, and both turned
out to have answers that are the opposite of the obvious guess.

Everything here is reproducible:

```bash
python3 scripts/ascs_reliability.py            # Stage 0, no model calls
python3 scripts/vfs_video_sweep.py --all       # Stage 3, ~64 calls, $0
python3 scripts/vfs_video_correlation.py       # the comparison
python3 scripts/extract_gt_keyframes.py        # Stage 5, no model calls
```

---

## The corpus

**64 cells** carry a scored animation record — 45 TikZ, 19 SVG — spread over
five styles and 32 samples. That is smaller than the pipeline's coverage
(48 TikZ / 47 SVG generated); the gap is cells generated but not yet scored.

A 45-record `_stale_prompts_2026-08-19/` directory sits beside them and is an
archive, not a result. Every script here excludes it by name. Counting it
doubles the TikZ corpus and silently mixes two prompt generations.

---

## Doubt 1 — "ASCS is empirically unreliable"

`ascs_pass` is a strict AND over every judged frame, so one DISCARD in a
twenty-frame deck fails the animation. The suspicion was that the metric is
noisy. The data says something more specific and more fixable.

### The per-frame judgements are internally consistent

Split-half over each cell's accept vector (odd frames vs even frames, then
Spearman–Brown corrected):

| statistic | value |
|---|---|
| split-half Spearman ρ | **+0.680** |
| Spearman–Brown corrected | **+0.809** |

0.809 is high. A cell's frames agree with each other about whether the
animation complies with its style. **The judge is not the noisy part.**

Two supporting checks:

- **Derivability 100.0%** (1048/1048 frames). The stored `verdict` is exactly
  the fold over the per-rule `followed` flags, with zero exceptions — so
  leave-one-out is a valid operation, and the verdict is not a second opinion
  drifting from the rules.
- **No position artefact.** Frame 0 gets a structurally different prompt (no
  previous-frame line, temporal rules force-marked followed), which could have
  been a confound. Its discard rate is 3.1% against 8.1% overall, and the
  decile profile is flat (4.7%–10.1%). Nothing to correct for.

### The aggregation is the whole problem

| rule | pass | rate | cells flipped vs strict |
|---|---|---|---|
| allow 0 discards **(current)** | 33/64 | **51.6%** | — |
| allow 1 discard | 44/64 | 68.8% | 11 |
| allow 2 discards | 52/64 | 81.2% | 19 |
| allow 3 discards | 54/64 | 84.4% | 21 |
| ≤10% of frames discarded | 44/64 | 68.8% | 11 |
| ≤20% of frames discarded | 52/64 | 81.2% | 19 |

Of the 31 cells that fail strictly, **11 (35%) fail on exactly one frame** and
19 (61%) on two or fewer. The metric's headline number is decided by a single
frame in a third of its failures.

Continuous `pass_fraction`: mean 0.907, median 1.000, min 0.300. Half the
corpus is perfect and the rest degrades gracefully — a usable signal that the
binary gate throws away.

### One rule dominates, but is not the whole metric

| rule dropped | discards | pass rate | Δ |
|---|---|---|---|
| `cumulative_addition` | 65 | 62% | **+10.9%** |
| `cumulative_effect` | 12 | 58% | +6.2% |
| `box_transparency` | 7 | 58% | +6.2% |
| `no_masking_or_hiding` | 18 | 52% | +0.0% |
| `absolute_persistence` | 14 | 52% | +0.0% |

`cumulative_addition` causes more discards than everything else combined, but
removing it recovers only 11 points — ASCS is not that one rule wearing a
metric's clothing. Note the two rules with many discards and **zero** effect:
their discards always co-occur with another broken rule in the same cell, so
they never change a verdict on their own.

### Recommendation

Keep ASCS. Report `ascs_pass_fraction` as the primary number and `allow_1` as
the gate. Do **not** redefine `ascs_pass` in place — the published analysis
cites the strict value and it must stay reproducible.

---

## Doubt 2 — "club ASCS into the Visual Fidelity Score"

This required building the video judge that did not exist (below), then asking
whether a video-level fidelity score already contains the style verdict.

### The frame judge barely discriminates

**86% of cells sit at exactly VFS 1.0.** For four of the five styles
`VFS_POLICY` is `"last"`, so VFS is one judged call about one frame, and it
returns 10.0 almost always. A Pearson correlation against a variable that is
constant on 86% of its support is not a weak result; it is an undefined
question.

### The video judge discriminates, and disagrees

Over the cells scored so far, the video judge:

- splits **9 out of 9** of the cells the frame judge tied at 1.0, introducing
  spread from 0.62 to 0.99
- scores lower on **94%** of cells, mean difference **−2.45 points of 10**
- names a specific, timestamped defect the frame judge has no field for

Examples, verbatim:

> *"Between 00:03 and 00:05, the rightmost skip connection line extends past
> the upper container boundary into empty space before the final top adder is
> revealed."*

> *"From 00:07 to 00:22, the generative model symbol 'G' is replaced with 'A'
> in the network block and the legend."*

The first is independently corroborated: Qwen's per-frame scores for that same
cell are `[10, 10, 2.0, 2.5, 10, 10, 10]` — frames 3 and 4, which at 2 fps are
seconds 3 to 4. Two judges, two modalities, same defect.

### But it does not encode the style verdict

| | n | mean video VFS |
|---|---|---|
| ASCS pass | 11 | 0.667 |
| ASCS FAIL | 7 | 0.663 |

AUC of video VFS for predicting an ASCS failure: **0.435** — essentially
chance.

### Recommendation

**Do not merge.** The two metrics are measuring different things: a video
fidelity score cannot tell you whether the declared animation style was
implemented. Merging would drop that signal silently while looking like a
simplification.

The free redundancy check on stored data agrees by not disagreeing: for
`alpha_masking` both nodes used policy `"all"`, so the same frames were judged
twice — but only **1 of 78** paired frames was ever discarded, so the
point-biserial there is underpowered, not null. It is reported as such.

---

## What had to be built

### Video reaches a backend for the first time

`Part` was `TextPart | ImagePart`. Now it includes `VideoPart`, and:

- **`_fingerprint` hashes video bytes and raises on an unknown part.** Without
  this, two requests differing only in their mp4 collide on one cache entry.
  Not hypothetical: the TikZ and SVG configs share a dataset root, so the same
  (sample, style) cell sends an identical prompt and identical source image in
  both. Verified on real data — the two `alpha_masking/CVPR_2025_arch01541`
  twins produced two cache entries and two different scores (0.58 / 0.54), and
  9 calls produced 9 distinct entries.
- **Other backends raise rather than drop.** A backend that ignored the video
  would still return a confident score computed from the source image alone.

### The judged video is rebuilt, not reused

Two facts, both measured against the live API:

1. **Gemini samples inline video at exactly 1 fps.** Token cost is perfectly
   linear in duration (60 tokens/frame at 2, 5, 10, 20 and 40 frames). The
   exporter writes at `fps: 2`, so sending `exports/animation.mp4` would have
   shown the judge **half** the animation, sampler's choice, silently.
   `animatebench/video.py` re-times the deck to 1 fps.
2. **Video frames are tokenized coarsely.** 60 tokens/frame by default against
   ~700 for the same diagram as a still. Criterion 6 of the video prompt asks
   about text legibility, which the model cannot assess at that resolution.
   `media_resolution: MEDIA_RESOLUTION_HIGH` restores ~250 tokens/frame.

`--video-source export` keeps the alternative available as a robustness check.

### Cost

$0. The whole study runs on free-tier Google keys (75 keys × 20/day against
~64 calls). On the paid OpenRouter route it would be about **$0.19**, against
the $2 budget.

---

## Keyframe extraction from video

The 64 GT reference animations ship as video with no frame decks, so no
per-frame node can score them. `animatebench/keyframes.py` holds the three
detectors, dispatched by style, writing `frame-NN.png` decks that
`frames.list_frames()` reads unchanged plus a `keyframes.json` provenance
sidecar.

Two defects had to be fixed before they worked on this corpus:

**A fixed threshold is a threshold on canvas size.** The detectors call motion
`activity > 0.01` — one percent of the frame must change. A hopping bounding
box on a 4236×4236 figure peaks at **0.0027**, so the animation reads as
static end to end and 32 frames extract to one keyframe.
`adaptive_activity_threshold` scales it to each video's own 95th percentile.

**`last_saved_centroid` is never initialised** in the sliding-box extractor. It
starts as `None` and is only assigned inside a save branch, so on a video that
never settles — precisely the case the `min_travel` fallback exists for — the
fallback's own `is not None` guard can never become true. Measured: a 57-frame
slide with per-frame velocities of 400–3600px extracted one keyframe. Seeding
it from the first detected centroid gives 25.

### The finding: the two targets ship different artifacts

Across all 64 reference videos, the split is almost perfectly clean:

| target | n | source frames | keyframes | reduction | methods |
|---|---|---|---|---|---|
| **SVG** | 32 | 3225 | 502 | **84%** | 13 coverage_ssim, 12 ssim_settle, 5 motion_centroid, 2 fallback |
| **TikZ** | 32 | 759 | 687 | 9% | **29 passthrough**, 2 motion_centroid, 1 fallback |

TikZ reference videos are rendered at 2 fps and are *already* keyframe decks —
running a settle detector on them would discard everything but the opening
frame. A dwell guard detects this and passes them through intact, and an
outcome guard catches detectors that clear the dwell check yet still return
nothing usable.

`scikit-image` is not installed and should not be: it pulls scipy/networkx and
can move numpy off the 1.26.4 pin that cv2 and torch are built against, and the
container's `PIP_CONSTRAINT` has broken `google-genai` before. Only
`structural_similarity` was needed, so it is implemented on cv2 primitives —
verified equal to skimage's to 1e-6 against a real install placed off to one
side (see `test_ssim_fallback_matches_skimage`, which skips by default and
carries the command to run it for real).
