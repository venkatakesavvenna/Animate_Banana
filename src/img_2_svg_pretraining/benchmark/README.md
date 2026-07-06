# Stage 1 VLM Benchmark

Benchmarks vision-language models on the Stage 1 task (architecture/pipeline
diagram image -> TikZ) against the annotated ground truth in `data/`.
Inference runs via plain `transformers` generation (`AutoModelForImageTextToText`
+ `.generate()`), not vLLM — see "Why not vLLM" below.

## Models

Registered in [models.py](models.py):

| key | HF repo |
|---|---|
| `qwen-3.5-9b` | `Qwen/Qwen3.5-9B` |
| `qwen-3.5-4b` | `Qwen/Qwen3.5-4B` |
| `gemma-4-12b` | `google/gemma-4-12B` |
| `gemma-4-e4b` | `google/gemma-4-E4B` |
| `kimi-vl` | `moonshotai/Kimi-VL-A3B-Instruct` |
| `phi-4` | `microsoft/Phi-4-multimodal-instruct` |

All confirmed to exist on the HF Hub and tagged `image-text-to-text`
(vision-capable) as of setup time.

Add more models by adding a `ModelSpec` entry — see the dataclass fields for
`trust_remote_code` / `attn_implementation` / `dtype`.

## Why not vLLM

The original design used vLLM for fast served/batched inference. In practice,
on this container:
- vLLM's PyPI install pulls its own `torch` build (here, `2.11.0+cu130`),
  shadowing the NGC base image's `torch` inside the project venv.
- That new torch's CUDA version (13.0) doesn't match this container's system
  CUDA toolkit (`nvcc`, 12.8), and the base image's precompiled `flash_attn`
  (built against the old NGC torch) breaks with an `undefined symbol` import
  error — it can't be reinstalled either, since compiling it requires a CUDA
  toolkit matching the new torch, and no prebuilt wheel exists for this
  torch/CUDA/Python combination.

Rather than fight the container's CUDA toolkit/torch version matrix, we
dropped vLLM and run inference directly through `transformers`, using
`attn_implementation="sdpa"` (torch's built-in scaled-dot-product-attention,
part of torch itself — no separate compiled extension, so no version-matching
problem). This is slower than vLLM's batched serving for very large sweeps,
but works today without fighting the environment. If GPUs/CUDA toolkit
versions get reconciled later, reintroducing vLLM would mean restoring a
server-based `infer.py` (an earlier version of this harness had one) and
would only require touching `infer.py`/`run_benchmark.py` — `models.py`,
`compile.py` integration, and the metrics are unaffected either way.

## Metrics

- **DINO visual similarity** ([metrics/dino.py](metrics/dino.py)): cosine
  similarity between DINOv2 (`facebook/dinov2-base`) embeddings of the
  ground-truth diagram image and the PNG rendered from the model's
  generated TikZ. Cheap, fully local, no LLM judgment involved — a proxy
  for "does it look right."
- **VLM-as-judge** ([metrics/judge.py](metrics/judge.py)): a separate local
  VLM (loaded via `transformers`, same as the models under test — any model
  from the registry works, or a dedicated judge model) is shown both images
  and asked for a 1-5 structured score plus a one-sentence rationale.
  Slower and needs its own GPU memory/model load, but catches semantic
  mismatches (wrong labels, wrong connections) that DINO similarity can miss.
- Samples where the model's output **fails to compile** are recorded with
  `compiled_ok: false` and `dino_score`/`judge_score` left as `null` (not
  zero) — they're excluded from the mean rather than silently dragging it
  down as a zero, so `compile_rate` and `dino_mean`/`judge_mean` need to be
  read together, not `dino_mean` alone.

## Running a benchmark

Everything runs inside the project docker container (needs the LaTeX/
poppler toolchain from `viewer/compile.py`, plus transformers/torch).

**1. Inference pass** — loads one model, runs it over N samples, compiles
each output:

```bash
python -m img_2_svg_pretraining.benchmark.run_benchmark --model qwen-3.5-9b --limit 100
```

- `--limit 100` (default) does a quick 100-sample pass; use `--limit 0` for
  the full dataset once the harness is validated.
- `--data-root` defaults to `/code/data/test_extracted/test` (the extracted
  `test.zip` set) — point elsewhere for a different snapshot.
- `--gpus` defaults to `1,2,3,4,5,6,7` (GPU 0 is excluded by default since
  it may be in use by another job on this shared host — pass `--gpus
  0,1,2,...` explicitly if you've confirmed it's free). See "Batching and
  multi-GPU data parallelism" below for what passing more than one id does.
- First run for a given model downloads its weights from HF Hub (can be
  several GB, slow depending on network) before inference starts.
- Writes `benchmark/results/<model>.jsonl` (one JSON record per line, raw
  model output/extracted TikZ/compile status/rendered PNG path/latency). See
  "Streaming output" below for how this is written.

**2. Scoring pass** — DINO always; judge optionally:

```bash
python -m img_2_svg_pretraining.benchmark.score --model qwen-3.5-9b --judge-repo Qwen/Qwen3.5-9B
```

Pass `--skip-judge` to only compute DINO scores (no judge model load needed,
much faster). `--judge-repo` can be any HF repo from the registry above, or
a different model entirely.

This updates `benchmark/results/<model>.jsonl` in place with `dino_score` /
`judge_score` / `judge_rationale` per sample, and writes
`benchmark/results/<model>_summary.json` with aggregates (`compile_rate`,
`dino_mean`, `judge_mean`).

**3. Repeat** steps 1-2 for each model in the registry. Inference and
scoring are deliberately separate passes/processes so the model-under-test
never has to share GPU memory with the DINO model or the judge model.

## Batching and multi-GPU data parallelism

- `--batch-size N` (default 1) batches N samples per `.generate()` call —
  left-pads the tokenizer and runs one forward pass over the batch instead
  of one sample at a time. Applies per GPU replica.
- `--gpus` accepts a comma-separated list of physical GPU ids. More than one
  id data-parallelizes: **one full model replica per GPU, each in its own
  process**, with the sample set sharded round-robin across them. This is
  throughput scaling via replicas (N models each doing 1/N of the work), not
  tensor-parallel splitting of a single model across GPUs for memory
  capacity — every listed GPU needs enough memory to hold the whole model on
  its own.
- Workers are spawned with `multiprocessing`'s `"spawn"` start method (not
  `fork`) so each child gets its own clean CUDA context; `CUDA_VISIBLE_DEVICES`
  is pinned to a single physical GPU inside each worker before any
  torch/transformers import happens in that process — torch locks in device
  visibility at first CUDA-touching import, so this has to happen before that.

## Streaming output (JSONL, not one big JSON)

Records are written as they complete rather than accumulated in a list for
the whole run and dumped once at the end — with `--limit 0` (full dataset)
and long generations, holding every sample's raw output/TikZ in memory for
the entire run doesn't scale.

With multiple `--gpus`, each worker process appends to its **own** private
`<model>.gpu<id>.jsonl` file, never a shared one. This was a deliberate
finding, not a stylistic choice: `/code` in this container is a **Lustre**
mount, and Lustre does not give `O_APPEND` the atomicity a local disk (ext4)
gives — concurrent writers opening the same file in append mode can
silently clobber each other's writes with no exception raised. This was
verified experimentally: 4 processes appending to one shared file (even
behind a `multiprocessing.Lock`) lost 2 whole workers' output with zero
errors. Switching each worker to its own file eliminated the loss entirely
(verified the same way). Once all workers finish, their per-GPU files are
concatenated (line-level, no re-parsing) into the final `<model>.jsonl` and
the per-worker files are deleted. Line order in the merged file follows
GPU/shard completion order, not original sample order — every record
carries its own `sample_id`, so downstream consumers (`score.py`, the
viewer) don't need file order to mean anything.

## Why this exists

See [viewer/README.md](../viewer/README.md#motivation) for the shared
motivation across the annotator and this benchmark: verifying/cleaning
ground truth, understanding model failure modes visually (not just via
aggregate scores), and giving every new model we test a standard,
side-by-side evaluation path.

## Viewing results in the annotator

The rendered PNGs and generated TikZ from a benchmark run aren't wired into
the `viewer/` UI yet — today, results live as JSON + cached PNGs under
`benchmark/results/` and `benchmark/render_cache/<model>/`. If you want a
given model's outputs browsable side-by-side with ground truth in the
annotator UI itself, that's a follow-up, not yet implemented here.
