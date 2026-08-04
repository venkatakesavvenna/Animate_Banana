# AnimateBanana Pipeline

Static scientific figure -> animation-aware code -> animation plan -> rendered animation.

Three stages, each a set of agents. Every agent runs over the whole sample set as its own
CLI subcommand and hands off to the next purely through on-disk artifacts, so only the model
a given agent needs is ever loaded, and any agent can be re-run alone after a prompt change.

```
STAGE 1  DIAGRAM TRANSMUTER
  1a convert-code        image -> animation-aware diagram code (TikZ/SVG)
  1b integrate-rasters   VLM locates each placeholder's region -> crop -> splice

STAGE 2  ANIMATION PLANNER
  2a strategize          traversal strategy: OVERVIEW_FIRST | DETAIL_FIRST   ┐ independent
  2b parse               diagram structure/hierarchy XML                     ┘
  2c sequence            animation sequence, conditioned on the style
  2d critique-sequence   review + repair, Set-of-Mark grounded

STAGE 3  DIAGRAM ANIMATOR
  3a design              animation code from the sequence + style
  3c critique-animation  compile-driven error detection + repair
  3d export              PDF / MP4 / GIF / PPTX
```

Stages 2a and 2b are independent — neither consumes the other — so their cache paths omit each
other's model and re-running one leaves the other's artifacts valid. The two lineages merge at
the sequencer.

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

Every intermediate artifact is on disk, so a finished run can be browsed after the fact:

```bash
python -m img_2_svg_pretraining.pipeline.inspector.app   # then open http://<host>:7860
```

Source figure beside the reconstruction, the animation sequence step by step, a frame
scrubber over the exported animation, and every stage's file. See
[inspector/README.md](inspector/README.md).

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
them), the designer (which realizes them), and the benchmark (which scores them):
`progressive_reveal`, `sliding_bbox`, `highlight_dim`, `zoom_pan`.

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
(~1T), Qwen3.5-397B-A17B (~800GB), DeepSeek-V4 Pro, Llama4-Scout. **Mistral-Large** is excluded
on a second ground: it is text-only, and every stage of this pipeline sends an image.

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
code/<converter>/<id>.tex
strategy/<code-lineage>__<strategizer>__<tier>/<id>.json
xml/<code-lineage>__<parser>/<id>.xml
sequence/<merged-lineage>/<id>.json          + <id>.roundN.json per critic round
sequence_final/<merged-lineage>/<id>.json
animation_final/<...>/<id>.tex
exports/<...>/<id>/{animation.pdf,.mp4,.gif,.pptx, frames/}
raw/<agent>/<model>/<id>.txt                 unparsed output, for debugging extraction
marked/<...>/<id>.roundN.png                 Set-of-Mark overlays
```

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

## Verified against live models (2026-07-29, gemini-3.6-flash)

**4 of 11 samples ran end to end**; the other 7 are blocked on API quota, not on pipeline
defects. Stage 1a succeeded on all 11. For the 4 that completed: Stage 2 12/12 agent-runs,
Stage 3 designer 3/3 and exporter 3/3, and every `animation_final` compiles.

The block is the free tier's 20 requests/day/model. One sample through all stages costs ~7
calls, so 11 samples need roughly 77 — well past the cap even pooled across keys. Finishing
the set needs a paid tier, or local weights via the `hf_local` backend. Re-running is
cheap: completed samples are cached and skipped, so a later run only pays for what is missing.

Both critics did real work on real output: the planner critic caught the sequencer inventing
`traversal_style: "STEP_BY_STEP"` and corrected it to a schema-valid value; the animation
critic repaired two non-compiling animations to compiling in one round each, driven by the
latexmk log.

Frame counts track sequence length (8-node plan -> 8 frames; a denser figure -> 41), and the
rendered frames show correct cumulative progressive reveal.
