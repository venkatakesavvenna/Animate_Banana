# Animation Evaluation Tree — implementation status

Plan: `~/.claude/plans/so-now-please-look-joyful-wreath.md`
Specs: `ingest/animate-docs/` (6 PDFs, 448pp)

## Done

**Prompts** — `src/img_2_svg_pretraining/animatebench/prompts/`, all transcribed
verbatim from the PDFs, all rendering clean with no leftover `{placeholder}`:

| file | keys | notes |
|---|---|---|
| `animation_fidelity.yaml` | 12 | E1a frames + E1b video, 5 adapters each. All 10 generic bodies verified byte-identical after unwrapping |
| `animation_style.yaml` | 11 | E2 ASCS; per-style rules **and** per-style schema |
| `animation_checklist.yaml` | 1 | the shared E3/SC3 first call |
| `animation_omission.yaml` | 11 | E3, refolded to one frame per call |
| `animation_repetition.yaml` | 7 | SC3, **3 styles only** (no PR, no colour_pop) |
| `animation_selection.yaml` | 9 | SC1 SSS |
| `animation_pacing.yaml` | 9 | SC2 GPS |

**Code**

- `animatebench/frames.py` — numeric sort (reuses `export.render._frame_index`),
  selection policies `last`/`all`/`every_4`, `step_frames()` (1:1 or `None`),
  downscale to 1568px JPEG-85 with a manifest. Verified: 3446×1889/575 KB →
  1568×860/137 KB, still legible.
- `animatebench/checklist.py` — `sequence_view()` buckets ids by **XML class**
  (27/30 sequences are native dialect, where `gt.sequence_elements` would put
  everything in `nodes`), `Checklist`, `validate()`, `build()`, `checklist_path()`.
- `animatebench/metrics/animation_quality.py` — `fold_frames` spine (one image
  per call, state in the caller's closure, per-frame errors isolated), all six
  nodes, `parse_bands` using the docs' own regexes, `run()`, `_would_eliminate`.
- `animatebench/judge.py` — added `ask_text()` (SSS/GPS answer in `### HEADER:`
  form, not JSON); **removed the `type != "gemini"` pin** in `make_judge` so a
  served Qwen judge is reachable by config. Verified no existing config declares
  a non-Gemini `gemini_flash`, so behaviour is unchanged for current runs.

**Env** — `/fsxvision_new/venkat.kesav/environments/ocr_env_vllm` copied from
`aryanjain.intern`. Built at container path `/environments/ocr_env_vllm`, so it
resolves inside `img-2-svg-pretraining-singlenode-venkat.kesav` without shebang
rewriting. Verified there: **vllm 0.19.0, torch 2.10.0+cu128**.
Unresolved: `du` reports 2.7 G against the source's 11 G. vllm imports fine, but
the file-count comparison was cancelled — **re-run it before trusting the copy**.

**Wiring — done**

- `run_eval.py` — `animation` in `SUITES`, `_run_animation`, `_get_checklist`
  (cached per config/style/sample), `--stages`, `--frame-px`. Fixed a bug where
  `needs_judge` omitted `animation`, so `run_eval animation` alone would have
  built no judge and recorded a whole run of `judge_skipped`.
- `results.HEADLINES["animation"]`; `descriptions.py` METRICS/SUITE_ORDER/
  SUITE_NOTES (+ worked example); `compare.py` suite tuple, `_TABLE_SUITES`,
  `_EVIDENCE` (7 keys).
- Tests: **49 passing**, up from 33. The suite tuple is centralised as `SUITES`
  at the top of the test file, so a sixth suite is covered automatically.

**Verified end to end** on `CVPR_2025_pipe00010 / progressive_reveal` with a stub
judge, against real artifacts: all six stages run, none skipped; VFS sends the
last frame only; ASCS 17 frames; SSS/GPS 17 timesteps each; repetition correctly
`not_specified` for progressive reveal. **69 judge calls per cell** → ~2,000 for
a 29-cell sweep. The four existing suites still read unchanged.

**Weights — downloaded.** `Qwen/Qwen3-VL-235B-A22B-Instruct`, full bf16, all 96
shards, 440 GB at
`/opt/dlami/nvme/venkat.kesav/hf_cache/hub/models--Qwen--Qwen3-VL-235B-A22B-Instruct`
(local NVMe, 26 TB free). Ungated; arch `Qwen3VLMoeForConditionalGeneration`,
confirmed supported by vLLM 0.19.0.

**Judge config — wired.** `bench_v2_gemini.yaml` gains a `qwen_judge`
`openai_compat` backend at `http://127.0.0.1:8000/v1`. Verified `make_judge`
resolves it to Qwen with no silent downgrade. `openai` installed into the eval
venv (it was missing).

**Env note (new node).** `ocr_env_vllm/bin/python3` is a symlink to
`/usr/bin/python3`, which was 3.12 on the old node and is 3.10 here — so the
venv is unusable from the host. It works inside any container with 3.12; I used
`vlm-ingest-pipeline-aryanjain.intern`, which mounts `/fsxvision_new` through
(so this repo is visible) and has all 8 GPUs. The copy at
`/fsxvision_new/venkat.kesav/environments/ocr_env_vllm` turned out unnecessary —
that container already mounts the original at `/environments`.

## Live — the judge works

`Qwen/Qwen3-VL-235B-A22B-Instruct` served on 8×H100 (TP=8, bf16, ~10 min to
load) inside `vlm-ingest-pipeline-aryanjain.intern`, which uses **host
networking** — so `http://127.0.0.1:8000/v1` in the config reaches it from the
host with no port mapping.

First scored cell, `CVPR_2025_pipe00010 / progressive_reveal`:
**VFS = 1.000 (raw 10.0/10)**. Read the prose before believing it: the judge
names real elements of that figure (the `101101` key, JPEG Compression / Crop /
Rotation, Watermark Decoder, Matching Messages, Identification) and explicitly
tolerates the animation's added green border and blue connector *under the
style adapter's rules* — i.e. the adapter is doing its job. Substantive, not a
lazy perfect score.

### Full cascade, one cell

All six stages ran on `CVPR_2025_pipe00010 / progressive_reveal`:

| node | result |
|---|---|
| VFS | **1.000** (10.0/10) |
| ASCS | **fail** — 4 of 17 frames DISCARDed (`frame-02, 08, 14, 16`) → `would_eliminate: ["ascs"]` |
| Omission | **0.000** — 0 of 27 checklist items never appeared; 0 arrows |
| SSS | **0.889** (band mean 3.56) but over **9 of 17** steps |
| GPS | **null** — 0 of 17 steps |
| Repetition | `not_specified` (correct: no prompt for progressive reveal) |

Checklist: 3 blocks, 11 nodes, 13 edges, one duplicate edge dropped by
`validate` — the shared artifact and its distrust of the judge both working.

### Complete run — Gemini judge

Same cell, all six stages, **zero errors**:

| node | Gemini 3.6 | Qwen 235B |
|---|---|---|
| VFS | 0.540 | 1.000 |
| ASCS | fail, 4/17 discarded | fail, 4/17 discarded |
| Omission | 0.000 (0 of 35) | 0.000 (0 of 27) |
| SSS | 0.750 (17/17) | 0.889 (9/17, server died) |
| GPS | 0.779 (17/17) | — |
| Repetition | `not_specified` | `not_specified` |

Both judges independently discard exactly 4 of 17 frames on style compliance,
and both find zero omissions — the mechanical half agrees across model families.
VFS is where they diverge hardest (0.54 vs 1.00), which is the judged half doing
what judged halves do.

### Three real bugs found by running it

1. **Gemini never rotated past a dead API key.** `_RETRIABLE_MARKERS` in
   `backends/gemini.py` covered 429 and 5xx but not 401/403, so a disabled
   service account raised a non-retriable error and the ring never advanced —
   with roughly a third of the 64-key pool dead, runs died partway through
   (SSS stopped at step 10 of 17). Auth failures now mark the key exhausted and
   rotate, exactly as quota errors already did. Fix verified: SSS and GPS went
   from 9/17 and 0/17 to **17/17 and 17/17**. This affects every judged metric
   in the repo, not just this suite.
2. **A partial `--stages` run overwrote the whole record.** Re-running
   `--stages sss gps` erased that cell's VFS, ASCS and omission scores. At ~70
   judged calls per cell, building a record incrementally is the normal way to
   work, so `_run_animation` now merges into the previous record.
3. **The judge can hallucinate a frame diff.** Gemini scored SSS t6 band 2 on
   "the rendered frame is visually identical to timestep 5". It is not: t5→t6
   changes 0.49% of the frame, *more* than several steps it scored higher
   (t11→t12 changes 0.24%). Qwen saw the same step correctly and scored it 4,
   naming the elements revealed. Worth carrying as a caveat — at 1568 px a
   small reveal may sit below the judge's perception, and SSS bands are
   sensitive to exactly that.

### The container deaths were external, not a fault

Both containers died with **exit 137**, and I initially suspected a node-level
reaper of 8-GPU containers. Wrong: a teammate killed the run. Nothing in the
vLLM logs shows an OOM, a CUDA fault, or any engine error — the serving itself
was clean for every call it was allowed to answer.

Note: `docker stop` / `docker rm` are denied to this session, so
`img2svg-animbench-venkat.kesav` (the first container, which mounted aryan's
env) is still running and should be removed by hand.

### Two flags added while getting there

Both exist because the bench configs and the cache tree assume a container that
is not the one this runs in:

- `--dataset` — the configs name `/code/data/...`, which exists in no container
  here.
- `--evals-root` — **the whole pipeline cache tree is root-owned** (written by
  a root container; no passwordless sudo anywhere to chown it back). Records,
  the checklist cache, prepared frames and the judge's own response cache all
  follow this override. Currently writing to
  `/fsxvision_new/venkat.kesav/animatebench_evals`.

  Consequence: the animation records live outside the tree the 8601 viewer
  reads, so the Metrics tab will not show them until either the cache is
  chowned back to `venkat.kesav` and the records moved in, or `compare.py`
  gains the same override.

## Previously blocked (resolved)

**Serving.** `vllm serve` with TP=8 failed: `Free memory on device cuda:N
(22.1/79.18 GiB) is less than desired GPU memory utilization`. Another user
(`sreevatsa.s`) started an 8-GPU vLLM job during the download and now holds
**74.5 GB on every card**. Their processes are untouched; our tmux session is
killed and nothing of ours holds GPU memory.

bf16 needs ~471 GB of weights plus KV cache, so it needs essentially the whole
node. Resume by either waiting for that job, or moving to a genuinely free node —
the weights are on this node's local NVMe, so a different node means either
re-downloading (~7 min at 1.2 GB/s) or pointing `HF_HUB_CACHE` at a shared path.

Serve command that was used (adjust `--gpu-memory-utilization` once free):

```bash
docker exec vlm-ingest-pipeline-aryanjain.intern bash -lc \
 'D=/opt/dlami/nvme/venkat.kesav/hf_cache; HF_HOME=$D HF_HUB_CACHE=$D/hub \
  /environments/ocr_env_vllm/bin/vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --limit-mm-per-prompt.image 3 --max-num-seqs 16 --port 8000'
```

Then:

```bash
python -m img_2_svg_pretraining.animatebench.run_eval animation \
  --config src/img_2_svg_pretraining/pipeline/configs/bench_v2_gemini.yaml \
  --style progressive_reveal --only CVPR_2025_pipe00010 \
  --judge-backend qwen_judge --stages vfs
```

## Deviation from the plan: `ingest/` was not used

The plan made serving-and-driving through vision-ingest the load-bearing half.
In practice vLLM was launched directly (`python -m
vllm.entrypoints.openai.api_server`) and the sweep was fanned out with five
shell processes. No `vision_ingest` import exists in any of this work.

Left as-is by decision, after the sweep had already completed on the direct
setup. The cost is real but bounded: no crash-safe resumability, no progress
visibility, and lost in-flight work when a container was killed. See
`docs/INFRA.md` for what retrofitting would involve.

## Also not done

1. E1b (video) — deferred by your call; `animation_fidelity.yaml` already
   carries the video prompts, but `VideoPart` + the Gemini branch are unbuilt.

## Open items

- **`{rubric}` is in `system`, not `user`** for SSS/GPS — `build_system_instruction()`
  concatenates the rubric. `_banded()` still passes `rubric` into the `user`
  render; harmless (unknown placeholders are left alone) but should be dropped.
- **Sliding rule 5 "Hopping Boxes" (HIGHLY CRITICAL) is sequence-level** and no
  per-frame call can enforce it. Recorded in `UNENFORCEABLE_RULES` alongside the
  two rules the docs give no schema key for. Needs a decision in the aggregator.
- **Thresholds and the combination formula are still yours.** `run()` scores
  every node and records `would_eliminate`; only ASCS can fire today.
- `BAND_LABELS` is the docs' fallback when the model omits `### LABEL` — not yet
  mirrored in `parse_bands`.
- A subagent installed `pypdfium2` into the active venv during transcription.
