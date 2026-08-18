# AnimateBench — evaluation suite

Scores a pipeline run's artifacts against the benchmark's reference bundle. Implements the
metric subset selected from the AnimateBench design document (`evaluation.pdf`), with
**Gemini 3.6 Flash** as judge.

```bash
# programmatic metrics only — no API key needed
python -m img_2_svg_pretraining.animatebench.run_eval all \
  --config src/img_2_svg_pretraining/pipeline/configs/bench_gemini.yaml \
  --style progressive_reveal --only CVPR_2025_pipe00041 --no-judge

# with the judge (needs a Google row in api_keys.csv)
python -m img_2_svg_pretraining.animatebench.run_eval all --config ... --style ...

# aggregate everything written so far
python -m img_2_svg_pretraining.animatebench.run_eval report --config ...
```

Suites: `stage1` · `xml` · `sequence` · `stage3` · `all` · `report`.
`--force` recomputes, `--include-quality` adds the optional code-quality judgements.

---

## 1. What is measured

| Suite | Metric | Doc § | Kind | Needs GT |
|---|---|---|---|---|
| `stage1` | Compilation Success Rate | 2.3.1 | programmatic | no |
| | Diagram Component Score | 2.1.3 | judge | no |
| | Rendering Fidelity | 2.3.1 | judge | no |
| | Diagram Code Quality *(optional)* | 2.3.3 | judge | no |
| `xml` | Parent-Assignment Accuracy | 2.1.4 | programmatic | **yes** |
| | Edge Precision / Recall / F1 | 2.1.4 | programmatic | **yes** |
| | Depth Consistency | 2.1.4 | programmatic | no |
| `sequence` | Coverage of animatable elements | 2.1.8 | programmatic | **yes** |
| | Traversal-Order Fidelity | 2.1.8 | programmatic | no |
| | Style-Schema Compliance | 2.1.9 | programmatic | no |
| | Dependence-Order Violation Rate | 2.1.9 | programmatic | no |
| `stage3` | Animation Compilation Success Rate | 2.3.1 | programmatic | no |
| | Animation Code Quality | 2.3.3 | judge | no |
| | Animation Integration Footprint | 2.3.3 | programmatic | no |
| `animation` | Visual Fidelity Score | VFS | judge | no |
| | Ani-Style Compliance | ASCS | judge | no |
| | Omitted Elements / Arrows | — | judge | no |
| | Selection Sensibility | SSS | judge | no |
| | Granularity & Pacing | GPS | judge | no |
| | Unnecessary Repetition | — | judge | no |

Everything in the "no GT / programmatic" rows runs with **no API key**, which is most of the
suite. Judge metrics degrade to `null` with a recorded error rather than failing the run.

---

## 2. The central problem: GT ↔ prediction identity

Element ids are chosen independently by each system, so nothing matches textually. Worse, the
correspondence is **not one-to-one**. All three of these occur in `CVPR_2025_pipe00041` alone:

| Phenomenon | Reference | Our pipeline |
|---|---|---|
| Rename | `input_block` | `block_input` |
| Many GT → one pred | `rgb1`, `rgb2`, `rgb3` | `img_rgb_images` |
| One GT → many pred | `hist` (one raster node) | `chart_histogram` + 7 parts |
| No counterpart | `gear_icon` (+8 teeth), `mt1..mt5` | — |
| No counterpart | — | `txt_3d_cues` |

### Solution: one judged, validated, cached alignment

`alignment.py` asks the judge **once per (config, sample)** — the structure XML doesn't vary
with animation style, so five styles share one call — for a grouping:

```json
{"groups": [
  {"gt": ["rgb1","rgb2","rgb3"], "pred": ["img_rgb_images"], "label": "RGB thumbnails"}
]}
```

Groups are many-to-many by construction, so both granularity directions are expressible.
Omission is an explicitly valid answer: the prompt states that a forced match is worse than
none, because it corrupts every downstream number.

The judge is **not trusted**. `validate()` drops ids absent from their XML, rejects an id
placed in two groups, and discards groups left one-sided — the same enforcement pattern as
`planner/narrative_writer._graft`. Judge misbehaviour therefore lowers *match coverage*, which
is reported, and can never silently inflate a score.

All GT-dependent metrics consume this one artifact, so PAA, edge P/R and sequence coverage
cannot disagree about what "the same element" means.

### Comparing through the alignment

Predicted structure is **contracted** onto GT groups before comparison:

- **PAA** — a GT element is correct when its matched prediction's parent contracts to the same
  group as the GT parent. Reported beside `matched_coverage`, because PAA 1.0 over 1 of 6
  elements is not the result PAA 1.0 over 6 of 6 is.
- **Edges** — endpoints are contracted, then edge sets are compared. Edges with an
  uncontractable endpoint are listed separately: an edge to a hallucinated element is a
  different defect from a wrong connection between real ones.
- **Coverage** — element sets from both sequences are mapped to groups. Unmatchable ids are
  excluded from both ratios and reported; they are structure-XML gaps already scored by
  `matched_coverage`, not sequencing mistakes.

---

## 3. Two decisions that keep the numbers honest

**Depth consistency ignores root numbering.** The two XML conventions disagree about whether
`<Diagram>` counts as a level — and the reference bundle *mixes both inside one file*: its
top-level blocks declare `depth=2` while its top-level edges declare `depth=1`. A rule fixing a
single base depth reported the reference's own hand-authored XML as 19% invalid. Only the
parent relation (`depth == parent.depth + 1`) is checked, which holds under either convention.
Result on pipe00041: GT 0.148, prediction 0.175 — the same phenomenon on both sides (composite
parts declared at their parent's depth), now comparable.

**The animation tree's gates are recorded, not enforced.** Its six nodes come
from six design documents that specify every prompt, input and output schema —
and not one threshold, normalisation, or combination rule. So `run_eval
animation` scores every node it can and writes `would_eliminate` rather than
gating. Only style compliance can answer today, because its verdict is
categorical. A gate closed on an invented cutoff would zero every metric
beneath it, and nothing downstream could tell that from a genuine failure; the
thresholds are better chosen against the distribution the first sweep produces.

**Its frames are never batched.** Each gate walks the animation forward one
judged call per frame, carrying the state the previous call produced. Counts
are then computed from those per-frame reports rather than asked of the model:
a judge asked to total thirty frames of its own work is doing arithmetic nobody
can check, while a judge asked "what appeared in this frame" is being asked
what it can see. It also means a hallucinated name cannot shrink the
outstanding list — only names actually still outstanding are allowed to pop.

**SSCR scores the bucket contract, not our action vocabulary.** The bench dialect has no
`action` field, so `AnimationSequence` defaults every step to `"reveal"`. Running the native
style check unfiltered failed *every* bench-format bounding-box sequence for using an action
the source never specified. Action checks now apply only to sequences that genuinely declare
actions; bench-format sequences are judged on what the benchmark defines — which buckets may be
non-empty, and one target per timestamp for the box styles.

---

## 4. Layout

```
animatebench/
  run_eval.py        CLI; orchestrates suites, caches records
  judge.py           strict-JSON judge over a pipeline ChatBackend
  alignment.py       GT<->pred grouping: judge, validate, cache
  gt.py              reference-bundle access + XML element table
  results.py         record io + aggregate report
  metrics/
    stage1_code.py       CSR, component score, rendering fidelity, code quality
    stage2_xml.py        PAA, edge P/R/F1, depth consistency
    stage2_sequence.py   coverage, TOF, SSCR, DOVR
    stage3_anim.py       anim CSR, code quality, AIF
  prompts/           alignment · component_score · rendering_fidelity · code_quality
```

Results land beside the artifacts they score:

```
pipeline/cache/<dataset>/evals/
  alignment/<config>/<sample>.json          style-independent, one judge call
  <config>/<style>/<sample>/{stage1,xml,sequence,stage3}.json
  raw/<config>/NNN_<tag>.txt                every judge transcript
  report.json  report.md
```

Every record keeps **per-element evidence**, not just the headline: `parent_detail` names the
mis-parented elements, `missed_gt_edges` names the missing connections, `dovr_violations` names
the arrows drawn too early. A score you cannot audit is not worth reporting.

---

## 5. Reuse

Nothing here re-implements what the pipeline already has:

| Need | Reused from |
|---|---|
| Judge calls, key rotation, response cache | `pipeline.backends.make_backend` |
| JSON extraction, truncation detection | `pipeline.extract` |
| Diagram compile + render | `viewer.compile.compile_tikz` |
| Animation compile (with export repairs) | `pipeline.export.{tikz_source,render}` |
| Both sequence dialects | `pipeline.schema.AnimationSequence` |
| Style constraint checks | `pipeline.styles.check_style` |
| Artifact paths per config/style | `pipeline.cache.CachePaths` |

The backend's content-hash response cache means re-running an eval never re-pays for unchanged
inputs; `--force` recomputes records, and judge responses stay cached beneath it.

---

## 6. Verification

`pipeline/tests/test_animatebench_metrics.py` — 24 tests, no API calls. Fixtures are small
enough to verify by hand and cover the cases that motivated the design: many-to-one alignment,
disagreeing depth conventions, spurious edges, early-arrow DOVR, bbox-style SSCR, and
hallucinated ids in the judge's alignment output.

Observed on `CVPR_2025_pipe00041` / `bench_gemini`:

| Metric | progressive_reveal | hopping_bounding_box |
|---|---|---|
| Diagram CSR | 1.000 | 1.000 |
| Depth violation rate | 0.175 | 0.175 |
| TOF | 0.798 | 1.000 |
| DOVR | 0.000 (16 edges) | — |
| SSCR | pass | pass |
| Animation CSR | 0.000 | 1.000 |
| AIF | 1.376 | 0.025 |

The AIF spread is the metric doing its job: the bounding-box animation was layered on almost
purely additively, while progressive_reveal rewrote the diagram body to gate opacity per frame.
The `anim_csr=0` is the `extra }` compile failure independently diagnosed for this sample.

With the judge enabled, the same sample scores `component_accuracy=1.000`,
`rendering_fidelity=0.300`, `code_quality=0.650`, `paa=0.842`, `edge_f1=0.938`,
`coverage_precision/recall=1.000`.

### What the judge caught that no programmatic check could

`rendering_fidelity=0.300` against `csr=1.000` looked like a broken metric. It was not — it is
a real Stage-1a defect the compile check is structurally blind to.

The generated TikZ declares each `fit` block **after** the nodes it contains, and gives it an
opaque `fill`. TikZ paints in source order, so every container paints over its own children:
`block_step1` alone covers ten elements. The document compiles with zero warnings, all 11
raster crops are embedded in the PDF (confirmed with `pdfimages -list`), and the render is
still almost empty — containers, titles and arrows only.

This is exactly the gap the design document's split between CSR and Rendering Fidelity exists
to expose: *"a diagram that compiles cleanly can still fail"*. A compile check answers "did
LaTeX exit 0"; only looking at the output answers "is the figure there".

Verified by hand before accepting the score: rendered the PDF independently, confirmed the
images are embedded at zero effective width, and reproduced a correct render of the same node
in isolation to rule out the coordinate override and `\includegraphics` as causes.
