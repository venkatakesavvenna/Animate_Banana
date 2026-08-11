# AnimateBanana Pipeline

Static scientific figure -> animation-aware code -> animation plan -> rendered animation.

Three stages, each a set of agents. Every agent runs over the whole sample set as its own
CLI subcommand and hands off to the next purely through on-disk artifacts, so only the model
a given agent needs is ever loaded, and any agent can be re-run alone after a prompt change.

```
STAGE 1  DIAGRAM TRANSMUTER
  1a convert-code        image -> animation-aware diagram code (TikZ/SVG)
  1b integrate-rasters   VLM locates each placeholder's region -> crop -> splice
  1c critique-diagram    render, compare with the source, diagnose, repair

STAGE 2  ANIMATION PLANNER
  2a strategize          traversal strategy: OVERVIEW_FIRST | DETAIL_FIRST   ┐ independent
  2b parse               diagram structure/hierarchy XML                     ┘
  2c sequence            animation sequence, conditioned on the style
  2d critique-sequence   review + repair, Set-of-Mark grounded
  2e narrate             spoken narration written into the sequence

STAGE 3  DIAGRAM ANIMATOR
  3a design              animation code from the sequence + style
  3c critique-animation  compile-driven error detection + repair
  3d export              PDF / MP4 / GIF / PPTX
```

Stages 2a and 2b are independent — neither consumes the other — so their cache paths omit each
other's model and re-running one leaves the other's artifacts valid. The two lineages merge at
the sequencer.

Around the pipeline sit three things worth knowing about up front:

- **[animatebench/](../animatebench/README.md)** scores a run against the reference bundle —
  14 metrics, Gemini as judge where the property is not mechanical. See *Scoring a run*.
- **Three viewers** (7860 / 8601 / 8602) browse a run, compare models with their metrics, and
  show the Stage-1 critic before and after. See *Inspecting a run*.
- **[docs/critic_evidence/](docs/critic_evidence/README.md)** records what the Stage-1 critic
  actually fixed, with before/after renders. Three of five bench samples produced nothing at
  all before that stage existed.

## Running

Everything runs inside the container:

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash
source /environments/img_2_svg_pretraining/bin/activate && cd /code

# inspect before running
python -m img_2_svg_pretraining.pipeline.run_pipeline samples
python -m img_2_svg_pretraining.pipeline.run_pipeline paths --sample SOME_ID

# one agent, one sample
python -m img_2_svg_pretraining.pipeline.run_pipeline convert-code --limit 1

# a whole stage, or everything
python -m img_2_svg_pretraining.pipeline.run_pipeline stage2
python -m img_2_svg_pretraining.pipeline.run_pipeline all --only CVPR_2025_pipe00002

# resume from any stage, or stop at one -- both bounds inclusive
python -m img_2_svg_pretraining.pipeline.run_pipeline all --from sequence
python -m img_2_svg_pretraining.pipeline.run_pipeline all --from parse --to design
```

Common flags: `--config`, `--limit N`, `--only ID...`, `--force` (ignore cache), `--gpu N`.

`--from`/`--to` narrow whatever was named rather than always slicing the full pipeline, so
`stage2 --from sequence` stays inside stage 2. The stage order is `convert-code`,
`integrate-rasters`, `strategize`, `parse`, `sequence`, `critique-sequence`, `design`,
`critique-animation`, `export`.

## Inspecting a run

Every intermediate artifact is on disk, so a finished run can be browsed after the fact.
Three viewers, on the three ports the container publishes:

| Port | Viewer | Shows |
|---|---|---|
| 7860 | `inspector.app` | one run end to end: figure, reconstruction, sequence, frame scrubber, every stage's file |
| 8601 | `inspector.compare` | ground truth + reference animation + one panel per model, **with that panel's AnimateBench metrics underneath** |
| 8602 | `inspector.critic_ab` | Stage-1 critic before/after, with a swipe slider and the findings it diagnosed |

```bash
python -m img_2_svg_pretraining.pipeline.inspector.app        --port 7860
python -m img_2_svg_pretraining.pipeline.inspector.compare    --port 8601
python -m img_2_svg_pretraining.pipeline.inspector.critic_ab  --port 8602
```

All three cache config and sample discovery at startup, so **restart a viewer after adding
samples or writing new eval records** — it will not pick them up live.

On 8601 every metric carries its plain-English meaning, its direction (higher or lower is
better), the design-doc section it comes from, and its caveats; the wording lives in
`animatebench/descriptions.py`, beside the code that computes it. See
[inspector/README.md](inspector/README.md).

## AnimateBench comparison

The benchmark ships one zip per sample: the paper inputs, the reference pipeline's
intermediates, its rendered animations (5 styles x 2 context tiers x tikz/svg), and human
review metadata. Import turns that into a normal pipeline dataset, keeping the reference
material under `reference/` where discovery ignores it:

```bash
python -m img_2_svg_pretraining.pipeline.import_bench \
  --src /path/to/extracted --dest /code/data/animatebench
```

Three configs differ *only* in which model every agent uses, so a side-by-side isolates the
model rather than the setup — `bench_gemini.yaml`, `bench_gemma4.yaml`, `bench_qwen.yaml`.
All have both critics off. Style is a per-run flag, since artifacts are keyed by it:

```bash
python -m img_2_svg_pretraining.pipeline.run_pipeline all \
  --config configs/bench_gemini.yaml --style colour_pop
```

Then the four-panel viewer — ground truth, the benchmark's own animation, and one panel per
model, with synchronized playback:

```bash
python -m img_2_svg_pretraining.pipeline.inspector.compare --port 8601
```

**The bench dialect.** `tikz_sequencer.yaml` / `svg_sequencer.yaml` emit the benchmark's own
schema rather than ours:

```json
{"metadata": {"traversal_order": ..., "animation_style": ...},
 "sequence": [{"timestamp": 1, "narrative": ...,
               "to_be_animated": {"blocks": [], "nodes": [], "text": [], "arrows": []}}]}
```

`schema.py` reads both and `to_bench_dict()` writes it back, verified lossless against 19 of
the bundle's own narration files (the 20th is malformed in the bundle — an element missing its
`"id"` key — and is skipped rather than fatal). The four element buckets are preserved
separately in `SequenceNode.element_classes` rather than flattened into `focus`, because the
bounding-box styles are defined partly by `text` and `arrows` being empty at every timestamp.

These prompts carry no `{placeholders}`: they name their inputs in prose, so the sequencer
appends them as a labelled suffix instead of interpolating. See `planner/sequencer.py`.

## Scoring a run

`animatebench/` scores a run's artifacts against the reference bundle — 14 metrics across the
four suites, Gemini as judge where the property is not mechanical:

```bash
python -m img_2_svg_pretraining.animatebench.run_eval all \
  --config configs/bench_gemini.yaml --style progressive_reveal
python -m img_2_svg_pretraining.animatebench.run_eval report --config ...   # aggregate table
```

`--no-judge` runs the programmatic half with no API key at all, which is most of the suite:
compilation, depth consistency, DOVR, SSCR, TOF and AIF need no model.

The hard part is that ground-truth and predicted element ids never match textually, and the
correspondence is not 1:1 — the reference draws three RGB thumbnails where our pipeline emits
one image. A judged, mechanically validated **alignment** (one call per sample, style
independent) maps between them, and every GT-dependent metric contracts through that one
artifact so PAA, edge P/R and coverage cannot disagree about element identity. Full design:
[animatebench/README.md](../animatebench/README.md).

## Configuration

One YAML drives a run (`configs/default.yaml`). Each agent independently names a backend, so a
run can mix a commercial API for one agent with local open weights for another:

```yaml
backends:
  gemini_pro:  {type: gemini, model: gemini-2.5-pro, api_key_env: GOOGLE_API_KEY}
  gemma4_12b:  {type: hf_local, hf_repo: google/gemma-4-12B-it, model: gemma4-12b}

planner:
  parser:    {backend: gemma4_12b, prompt: "animation_planner/tikz_parser.yaml#prompt"}
  sequencer: {backend: gemini_pro, prompt: "animation_planner/sequencer.yaml#prompt"}
```

Prompts are addressed `file.yaml#key`, matching the flat `{variant: text}` shape the prompt
files already use. The config is validated eagerly: unknown agents, unresolvable backends,
missing prompt files/keys, typo'd options and bad enum values all fail at startup.

### Backends

| type | use |
|---|---|
| `hf_local` | open weights in-process via transformers — the local route |
| `gemini` | Google Gemini |
| `anthropic` | Anthropic Claude |
| `openai_compat` | OpenAI, or a vendor's hosted OpenAI-shaped API |

Retry with backoff, a concurrency limit, and response caching are handled once in
`backends/base.py`, so adding a provider means implementing `_generate` plus one registry line.

### Running on local GPUs

[`configs/hf_local.yaml`](configs/hf_local.yaml) runs every agent on Gemma 4. Swapping a stage
off Gemini is a one-line change:

```yaml
backends:
  gemma4_12b: {type: hf_local, hf_repo: google/gemma-4-12B-it, model: gemma4-12b}

transmuter:
  code_converter: {backend: gemma4_12b, ...}     # was gemini_flash
```

Run it under the **`gemma4` venv** — the main venv's transformers predates the architecture
(`AutoConfig` raises `KeyError: 'gemma4'`):

```bash
source /environments/gemma4/bin/activate && cd /code
python -m img_2_svg_pretraining.pipeline.run_pipeline all \
    --config src/img_2_svg_pretraining/pipeline/configs/hf_local.yaml --gpu 0
```

**Inference is in-process, not a server.** vLLM was tried and does not work in this container:
its `flash_attn` is built against a different CUDA toolkit than the installed torch, and its
NCCL segfaults above one GPU. `benchmark/models.py` had already recorded the first of those.

Two attention settings are therefore not optional, and `hf_local` defaults to both:
`attn_implementation="eager"` and cuDNN disabled. `sdpa` routes into cuDNN, which has no
execution plan for Gemma 4's head shapes (`cuDNN Frontend error: No valid execution plans
built`).

Sizing: 12B is ~24GB in bf16 and fits comfortably; 31B is ~62GB and needs a genuinely idle
80GB card. **This node is shared** — check `nvidia-smi` before choosing `--gpu`, and note that
inside the container it shows another job's memory but not the process that owns it.

**Credentials** resolve in this order: an inline `api_key` in the config, the `api_key_env` var
(default `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), then `api_keys.csv` at the
repo root — a gitignored file with a `Key, Service` header and one row per key.

Gemini quotas are **per key**, so every matching key is pooled and the ring rotates on a 429
before the base class retries. This matters: the free tier allows only 20 requests/day/model,
and a single sample through all stages costs ~7. With 11 keys a handful of samples is fine; a
full 11-sample run needs a paid tier, or local weights via `hf_local`.

Model availability drifts — `gemini-2.5-flash` is already retired for new keys. Check with
`client.models.list()` before pinning one. `gemini-3.6-flash` worked as of 2026-07-29.

## Data layout

Per-sample directories:

```
<root>/<id>/
    <id>.png          the figure
    caption.txt       figure caption
    abstract.tex|txt  paper abstract
    methods.tex|txt   method section
    arxiv_src/        optional, used for the paper title
```

The extension varies per sample (`.tex` or `.txt`), and one sample has a typo'd image filename,
so discovery falls back rather than dropping it. The four context tiers — `image_only`,
`caption`, `abstract`, `full` — map onto the four variants in `strategizer.yaml`; a sample
missing a field simply cannot run the tiers that need it.

## Contracts

**Animation sequence** (`schema.py`) — the artifact Stage 3 and the benchmark both consume:

```json
{"style": "progressive_reveal", "traversal_style": "OVERVIEW_FIRST",
 "nodes": [{"id": "node_01", "parent": null, "depth": 1,
            "focus": ["block_encoder"], "action": "reveal",
            "narration": "...", "duration": 4.0}],
 "traversal": ["node_01"]}
```

`focus` entries are `xml id` values from the diagram code, which is what makes reference validity
checkable against the parsed XML. `validate()` returns the same violations the benchmark's
validity metrics report, so the critic and the scorer agree on what "invalid" means.

**Animation styles** (`styles.py`) — defined once, shared by the sequencer (which plans within
them), the designer (which realizes them), and the benchmark (which scores them).

The five AnimateBench styles, with constraints taken from Ruleset 3 of the benchmark's own
sequencer prompt so a run of ours is comparable with the reference:

| Style | Mechanically checked |
|---|---|
| `progressive_reveal` | each element revealed exactly once, never re-revealed |
| `colour_pop` | all element types, one logical step per timestamp |
| `alpha_masking` | strict DAG — an element may never appear in two timestamps |
| `hopping_bounding_box` | exactly ONE element per timestamp |
| `sliding_bounding_box` | exactly ONE element per timestamp |

Plus our own earlier three, kept so cached artifacts stay valid: `sliding_bbox`,
`highlight_dim`, `zoom_pan`. Note `sliding_bbox` and `sliding_bounding_box` are different
entries — the bench spelling is its own style with the stricter one-element rule.

## Open-weights roster

Configured in `configs/open_weights.yaml`. Weight sizes are **measured** from the HuggingFace
repos (bf16, weights only — add 10-20% for activations and KV cache), not estimated.

Host budget is GPUs 4-7, four H100 80GB = **320GB**. GPUs 0-3 belong to another job; select
cards with `--gpu 4,5,6,7`, which sets `CUDA_VISIBLE_DEVICES` before torch initializes.

| Model | Repo | Weights | Placement |
|---|---|---|---|
| Qwen3.6 27B | `Qwen/Qwen3.6-27B` | 55.6 GB | 1 GPU |
| Gemma 4 31B | `google/gemma-4-31B-it` | 62.5 GB | 1 GPU |
| Qwen3.6 35B-A3B | `Qwen/Qwen3.6-35B-A3B` | 71.9 GB | 2 GPUs (MoE headroom) |
| GLM-4.6V | `zai-org/GLM-4.6V` | 215.4 GB | 4 GPUs, `max_memory` 70GiB/card |

**Too large for this host** — declared in the config so the sizes are recorded rather than
rediscovered, but they will OOM at bf16:

| Model | Repo | Weights | Needs |
|---|---|---|---|
| InternVL3.5 241B-A28B-Flash | `OpenGVLab/InternVL3_5-241B-A28B-Flash` | 483.4 GB | ~7-8 H100 |
| MiniMax-M3 | `MiniMaxAI/MiniMax-M3` | 854.2 GB | ~12 H100 |

**Not wired** — no verified vision-capable repo id, and far past budget regardless: Kimi K2.5
(~1T), Qwen3.5-397B-A17B (~800GB), DeepSeek-V4 Pro, Llama4-Scout.

Run under `/environments/gemma4` (transformers 5.13.0), which was verified to carry all four
runnable architectures — `glm4v_moe`, `qwen3_5_moe`, `qwen3_5`, `gemma4`. The main venv (5.3.0)
lacks `gemma4`. No new venv is needed.

```bash
source /environments/gemma4/bin/activate
python -u -m img_2_svg_pretraining.pipeline.run_pipeline all \
  --config src/img_2_svg_pretraining/pipeline/configs/open_weights.yaml \
  --gpu 4,5,6,7 --limit 1
```

Sharding is accelerate's `device_map: auto`, not vLLM. vLLM 0.24.0 is installed but unusable
here — it requires `torch==2.11.0` against the base image's `2.7.0a0`, and the earlier attempt
recorded in `backends/hf_local.py` hit a flash-attn/CUDA mismatch plus an NCCL that segfaulted
above one GPU. The cost is throughput: no continuous batching, so requests serialize.


## Artifacts

Under `cache/<dataset>/`, with paths encoding the producing models so configs never collide:

```
code/<converter>/<id>.tex                    1a
rasters/<code-lineage>/<id>/{crop_*.png,detections.json}   1b
code_final/<code-lineage>/<id>.tex           1b output
code_reviewed/<code-lineage>/<id>.tex        1c output  + <id>.critic.json (what it fixed)

strategy/<code-lineage>__<strategizer>__<tier>/<id>.json
xml/<code-lineage>__<parser>/<id>.xml
sequence/<merged-lineage>/<id>.json          + <id>.roundN.json per critic round
sequence_final/<merged-lineage>/<id>.json
sequence_narrated/<...>__<writer>__<tier>/<id>.json        2e
narration/<...>/<id>.jsonl                   timestamp-indexed script, TTS input
audio/<...>/<id>/audio_ts_N.wav              synthesized narration

animation/<...>/<id>.tex                     3a
animation_final/<...>/<id>.tex               3c
exports/<...>/<id>/{animation.pdf,.mp4,.gif,.pptx, animation_narrated.mp4, frames/}

evals/alignment/<config>/<id>.json           GT<->pred element alignment (style independent)
evals/<config>/<style>/<id>/{stage1,xml,sequence,stage3}.json
evals/report.{json,md}                       aggregate table
raw/<agent>/<model>/<id>.txt                 unparsed output, for debugging extraction
marked/<...>/<id>.roundN.png                 Set-of-Mark overlays
renders/<hash>.png                           content-addressed compile cache
```

`resolve_code` reads the newest usable version — `code_reviewed`, else `code_final`, else
`code` — so enabling or disabling a Stage-1 agent changes what every later stage plans against
without any of them knowing about it.

## Notes

- **Critic loops** are bounded by `max_rounds`. Every round's artifact is persisted so
  refinement is inspectable. A sample that exhausts its budget is reported `unresolved`, not
  `ok` — its artifact is still written, but it must not be mistaken for a success.
- **The animation critic's strongest signal is the compiler**, not a model judgment. A latexmk
  log names the failing line; that goes straight back into the repair prompt.
- **`\documentclass[tikz]{standalone}` is incompatible with `animateinline`** (fails with
  `! Improper \prevdepth.`). Animation code must use bare `\documentclass{standalone}` plus
  `\usepackage{tikz}`. Both animator prompts state this explicitly.
- **`viewer/compile.py::compile_tikz` passes `-singlefile`**, so it rasterizes page 1 only.
  Correct for the compile check; the exporter makes its own `pdftoppm` call for all frames.
- **Stage 1a must emit `\usepackage{amsmath, amssymb}`.** Scientific figures are full of math
  notation; a single `\text{x}` in a label otherwise fails the whole document with
  `! Undefined control sequence.` The converter prompt requires it.
- **Token budgets matter more than they look.** The sequencer emits a full node list with
  narration and the planner critic echoes the whole corrected sequence back, so both need
  ~32k. At 8192 they truncate mid-JSON; `extract.looks_truncated` distinguishes that from a
  model ignoring the format, since the two need different fixes.
- **Stage 1b uses one model, the same one every other agent uses.** It was two vision models
  in separate subprocesses; detection now happens in the chat backend, so only one model is
  ever loaded. See [vision/README.md](vision/README.md).

## Stage 1b (raster integration)

**One VLM call per sample.** The model sees the source figure plus the placeholders Stage 1a
emitted, and returns a bounding box per placeholder it can locate; crops are cut from the
source and spliced in as `\includegraphics`.

This replaced a Molmo2 → SoM verify → SAM3 → SoM map chain. That route needed two vision
checkpoints on GPU under a second venv (Molmo2 pins `transformers==4.57.1`), so the pipeline
could never hold only one model at a time. Detection collapses into one call because Stage 1a
already names what to look for — the model returns the `xml id` it matched, so locating a
region and deciding where it belongs are the same answer.

Boxes arrive in Gemini's `[ymin, xmin, ymax, xmax]` convention on a 0–1000 grid, decoded in
one place (`vision/gemini_boxes.py::to_pixels`).

Verified on `CVPR_2025_pipe00002`: 7 placeholders → 7 located → 7 filled, compiles before and
after, and the render shows the real panels in the right boxes with no layout shift. The
Φ(x,δ) composite came back whole rather than as one sub-panel.

⚠️ It filled **7 of 7**, where the old route correctly rejected 4 as line art. **"Placeholders
filled" is not a metric to maximise** — naming a placeholder in the prompt biases the model
toward finding something for it. The omission path is currently unexercised.

Details, the coordinate convention, and the failure behaviour:
[vision/README.md](vision/README.md).

## Stage 1c (diagram critic)

Stages 1a and 1b are write-only: nothing ever looked at what the generated code actually
*draws*. The critic closes that loop — render, compare against the source figure, diagnose,
repair — and it is the only check in Stage 1 that can see a defect the compiler cannot.

```
score render -> below threshold? -> diagnose -> repair -> re-score -> keep if better
```

Four rules make the loop safe to run unattended:

| Rule | Why |
|---|---|
| Gate on fidelity before any repair work | Code already at `fidelity_threshold` (0.7) costs one scoring call and stops. Good samples stay cheap. |
| Keep a repair only if it scores **higher** | The critic can degrade code. The best version seen wins, never the last one tried. |
| Reject a repair that drops an `xml id` | Every later stage addresses elements by id; losing one breaks the animation plan for that element. |
| Reject a repair that does not compile | Never trade a rendering defect for a broken document. |

Scoring and diagnosis are separate calls on purpose: a model asked to grade and fix in one
breath tends to justify its grade instead of finding the defect, and the score has to stay
honest because it is both the entry gate and the accept test. The rubric is the same one
AnimateBench's Rendering Fidelity metric uses, so the loop optimises against the measure it
will later be judged by.

The loop runs in two phases. **Compile repair first** — a document that produces no PDF cannot
be compared with anything, and these failures are self-contained (the code used a colour, style
or node it never declared, and the LaTeX log names which). **Then the render/score loop.**

**Measured across all five AnimateBench samples** (2026-08-06, gemini-3.6-flash):

| Sample | Before | After | Defect |
|---|---|---|---|
| pipe00002 | 0.52 | 0.52 | cosmetic; repair **rejected for not improving** |
| pipe00010 | *did not compile* | **0.88** | undefined shape, then 3 overpainting containers |
| pipe00041 | 0.20 | **0.76** | 4 overpainting containers |
| pipe00045 | *did not compile* | **0.83** | undefined node ref, then 2 overpainting containers |
| pipe00137 | *did not compile* | **0.72** | undefined TikZ style |

**Three of five samples produced nothing at all before this stage existed.** The dominant
rendering defect — a `fit` container declared after its contents with an opaque fill, painting
over its own children — appeared independently in three samples, so it is a systematic
converter habit rather than a one-off. AnimateBench, scoring the artifacts separately, agrees
with the critic's own numbers (`rendering_fidelity` 0.300 → 0.760 on pipe00041).

Full findings, the LaTeX errors, and before/after renders:
[docs/critic_evidence/](docs/critic_evidence/README.md). Interactive swipe comparison:
`inspector/critic_ab.py` on port 8602.

## Full run: 5 samples x 5 styles (2026-08-06, gemini-3.6-flash)

`bench_gemini.yaml` over the five AnimateBench samples, all five animation styles, with the
Stage-1 critic on and both other critics off. 25 animations, 100 eval records.

**Stage 1 and Stage-2 XML are identical across all five styles**, because those artifacts are
style-independent and produced once. That they come out identical is a consistency check on
the cache lineage, not a coincidence:

| | 00002 | 00010 | 00041 | 00045 | 00137 |
|---|---|---|---|---|---|
| Diagram CSR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rendering fidelity | 0.580 | 0.850 | 0.760 | 0.820 | 0.770 |
| Component accuracy | 1.000 | 0.976 | 1.000 | 0.937 | 0.986 |
| PAA | 1.000 | 0.886 | 0.842 | 1.000 | 1.000 |
| Match coverage | 1.000 | 0.921 | 1.000 | 0.900 | 0.808 |
| Edge F1 | 1.000 | 0.833 | 0.938 | 0.757 | 0.826 |

**Per style**, where the differences actually live:

| style | TOF | worst DOVR | animation CSR | mean AIF |
|---|---|---|---|---|
| `sliding_bounding_box` | **1.000** | — | **5/5** | 0.065 |
| `hopping_bounding_box` | **1.000** | — | **5/5** | **0.036** |
| `alpha_masking` | 0.815 | 0.143 | 2/5 | 0.95 |
| `progressive_reveal` | 0.771 | 0.000 | 3/5 | 1.16 |
| `colour_pop` | 0.770 | **0.857** | 2/5 | 1.98 |

Element coverage recall is 1.000 nearly everywhere and SSCR passes 25/25.

Three things this run establishes:

- **The bounding-box styles dominate on every axis.** Their one-element-per-timestamp contract
  leaves no room to violate traversal order (TOF 1.000 on all ten runs), and the box is a
  genuinely additive overlay — AIF ~0.03-0.07 against 1.0-2.0 for the reveal styles, which
  rewrite the diagram body to gate opacity per frame. Every one of their animations compiles.
- **7 of 25 animations fail to compile**, all with the same Stage-3 designer defect (`! File
  ended while scanning use of \pgffor@collectargument` — an unclosed `\foreach`), and all
  concentrated in the reveal styles whose AIF is highest. This is what the *animation* critic
  repairs, and it is disabled in this config. The Stage-1 critic demonstrates the pattern
  works.
- **Worst single planning result:** `colour_pop` on pipe00002, DOVR 0.857 — six of seven
  arrows revealed before the elements they connect. The same sample scores 0.000 under
  `progressive_reveal`, so this is the style interacting badly with the sequencer, not a bad
  sample.

DOVR is reported as `—` for the bounding-box styles rather than 0: those styles animate no
arrows at all, so there is nothing to order-check, and a 0 would read as a passing grade for
something never tested.

Aggregate table: `cache/<dataset>/evals/report.md`.
