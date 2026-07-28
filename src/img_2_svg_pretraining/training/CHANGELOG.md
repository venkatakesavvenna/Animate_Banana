# Changelog

## v0.9.0 - 2026-07-14

### Scope

Fixed the actual root cause of Molmo's degenerate/repetitive captioning-eval output (an image-unaware label-masking boundary that let image-patch tokens and role-marker text leak into the supervised loss); fixed several compounding generation-path bugs found while diagnosing it; repository layout and documentation restructuring.

### Done

- **Fixed `tokenize_molmo_instance`'s label-masking boundary** (`training_core/data_modules/vlms/molmo/molmo_data.py`):
  The prompt/answer boundary was computed by tokenizing the prompt text *without* images, then applied directly against a sequence tokenized *with* images. Since images inject 100+ patch tokens, this left most of the image block plus the literal `" User: ... Assistant:"` role-marker text as real, unmasked, supervised training targets — the model was being trained with real gradient to predict `"Assistant:"` as valid next-token output. Verified on a real sample: the old boundary masked only 16 of 754 tokens that should have been masked.
  Fixed by searching for the `" Assistant:"` role marker directly inside the already-correctly-tokenized (with-images) sequence instead of comparing against a separately tokenized fragment. Also detects and strips a second, spurious `" Assistant:"` the processor appends right after the answer, and appends an explicit EOS token marked as supervised in its place — previously EOS was never supervised at all, so the model had no training signal for when to stop generating.

- **Fixed `VLMOnly.generate()` / `VLMSam._build_generate_kwargs()` silently dropping `use_cache`, `synced_gpus`, and `max_new_tokens`** (`training_core/models/vlm_only.py`, `training_core/models/vlm_sam.py`):
  The kwargs allowlist didn't include these keys, so caller-specified values were discarded and `max_new_tokens` always fell back to a hardcoded 512. As a result Molmo's manual decode loop always ran with `use_cache=False` (dropping all image conditioning after the first generated token) and `synced_gpus=False` (no ZeRO-3 straggler protection — a latent multi-rank deadlock risk).

- **Fixed image conditioning loss in `MolmoModel.generate()`'s no-cache decode path**: the per-step forward call didn't pass `images`/`image_masks`/`image_input_idx` after step 0, so without a KV cache the model generated every token past the first one with zero visual grounding.

- **Fixed the captioning-eval ground-truth-leakage guard being a silent no-op for Molmo** (`training_core/train/captioning_eval_callback.py`):
  `_truncate_and_left_pad_for_generation` bailed out immediately whenever `attention_mask` was absent from the batch — and Molmo's collator never produces one at all. Rewritten to derive the prompt boundary and content length from `labels` (always present) and to freshly construct a correct `attention_mask` for the left-padded output. Also fixed `image_input_idx` handling: its real shape is `[B, num_crops, patches_per_crop]` (absolute text-position values), not sequence-aligned like `input_ids` — the previous code tried to left-pad/truncate it as if it were.

- **Fixed a broken MetaCLIP checkpoint path** used by 33 encoder-swap configs/scripts (Molmo/Qwen/Gemma × `raw_pt`/`vlm_pt`/`ft`): the referenced HF cache snapshot was missing `config.json`, causing `CLIPVisionModel.from_pretrained` to silently fall back to the default `hidden_size=768` config and then fail loading ViT-L/14 (`hidden_size=1024`) weights.

- **Repository restructuring:**
  - Root now has no stray `.py` scripts or extra `.md` reports — moved to `scripts/adhoc/` and `docs/reports/` respectively.
  - The detailed reference doc moved from root `README.md` to `src/README.md`, with large sections (dataset registry, known gaps, tests, CI) collapsed behind `<details>` blocks and a new `Troubleshooting` section.
  - Root `README.md` is now a short landing page (logo placeholder, badges, at-a-glance summary, links to quickstart/docs/changelog).
  - `QUICKSTART.md` (root) trimmed from ~565 to ~80 lines — covers only the happy path (build → launch → watch); everything else links to `src/README.md`.

## v0.8.0 - 2026-07-13

### Scope

Encoder-swap smoke test infrastructure for Molmo, Gemma3-12B, and Gemma3-27B across 6 vision encoders (126 configs/scripts); two bug fixes found during smoke testing (OLMo norm checkpoint, left-padded generate slice); training efficiency monitoring; opt-in PyTorch Profiler + NVTX integration.

### Done

- Added 126 encoder-swap SLURM scripts and smoke configs under `scripts/encoder_swap/` and `src/configs/smoke/encoder_swap/`:
  All three VLM families (Molmo-7B, Gemma3-12B, Gemma3-27B) × 6 encoder kinds (`clip`, `metaclip`, `metaclip2`, `openvision`, `siglip`, `siglip2`) × 3 training modes (`vlm_pt`, `raw_pt`, `ft`) = 126 jobs.
  All scripts include `#SBATCH --nodelist` (pinned to 4 healthy nodes) and `#SBATCH --exclude` (known-broken node).
  Configs: `max_new_tokens: 64`, `captioning eval_steps: 100` to keep smoke runs under ~1 min of eval overhead.

- Fixed pretrain-init OLMo checkpoint (`raw_pt` mode):
  Changed `lm_raw_checkpoint` from `allenai/OLMo-7B-0724-Instruct-hf` (OLMo 1.x — no learnable norms; all `attn_norm`, `ff_norm`, `q_norm`, `k_norm` stay at random init) to `allenai/OLMo-7B-1024-preview` (OLMo 2 — has all learnable norm parameters).
  Using OLMo-0724 produced degenerate outputs ("the the the…") because the absent norm weights forced the transformer into a pathological fixed point.
  Affected: `scripts/generate_encoder_swap_configs.py` and all 6 `src/configs/encoder_swap/mn_molmo_7b_v*_raw_pt.yaml` + their smoke variants.

- Fixed prompt-slice bug in multi-sample generation (`vlm_pt` mode):
  Root cause: `VLMOnly.generate()` and `VLMSam._get_prompt_lengths()` used `attention_mask.sum(dim=-1)` (actual non-padded length) as the continuation slice offset. With left-padded batches, shorter samples in the same batch had their tail-of-prompt tokens (e.g. ` Assistant:`) included in the decoded output, producing "Assistant: Assistant: …" repetitions.
  Fix: use `input_ids.shape[1]` (uniform padded sequence length) as the slice offset in both `vlm_only.py` and `vlm_sam.py`.

- Removed deprecated native scripts/configs:
  `scripts/mn_molmo_{clip_olmo,pretrain,pt_replication}.sh`, `scripts/mn_qwen25vl_siglip_qwen25_{ft,pt}.sh`, and their corresponding YAML configs.

- Added `TrainingEfficiencyCallback` (`src/training_core/train/training_efficiency_callback.py`):
  Always-on; registered unconditionally in `run_training`. Logs to W&B under `efficiency/*` namespace.
  Token counts come from `CustomTrainer.training_step` (text/active from batch) and an encoder forward hook (visual tokens from output shape).
  - `step_time_ms`, `steps_per_sec` — 50-step rolling average, first 5 warm-up steps excluded
  - `samples_per_sec` — across all GPUs
  - `text_tokens_per_sec[_per_gpu]` — `input_ids.numel()` per step
  - `active_tokens_per_sec[_per_gpu]` — `(labels != -100).sum()` per step (loss tokens only)
  - `visual_tokens_per_sec` — from encoder output shape, first forward call per step
  - `encoder_time_ms` / `encoder_time_fraction` — wall time inside encoder `forward` (includes gradient-checkpointing recompute); encoder auto-discovered by walking common backbone attribute paths
  - `data_stall_ms` — gap between `on_step_end` and next `on_step_begin` (≈ dataloader fetch time)
  - `peak_gpu_mem_gb` — `max_memory_allocated()` with `all_reduce(MAX)` across all ranks
  - `mfu` / `mfu_pct` — Model FLOP Utilization via 6ND approximation; hardware peak TFLOPS looked up from device name (A100=312, H100=989, …); uses `ds_numel` for ZeRO-3 sharded params

- Added `ProfilingCallback` (`src/training_core/train/profiling_callback.py`):
  Opt-in; enabled via `profiling: {enabled: true}` in config.
  - Wraps selected steps with `torch.profiler.profile` on rank 0; writes TensorBoard trace to `{logging_dir}/profiler_traces/`
  - Registers NVTX range markers (`backbone_forward`, `encoder_forward`) on all ranks so full multi-GPU GPU timelines are annotated in Nsight Systems
  - Config fields: `wait_steps`, `warmup_steps`, `active_steps`, `with_stack`, `nvtx`, `output_dir`
  - `CustomTrainer.training_step` now always emits a `training_step` NVTX range (no-op without a profiler, essentially free)

### Config example

```yaml
# always-on efficiency monitoring — no config needed, registered automatically

# opt-in profiling (profile steps 2–4):
profiling:
  enabled: true
  wait_steps: 1
  warmup_steps: 1
  active_steps: 3
  with_stack: false
  nvtx: true
  # output_dir: /code/outputs/my_run/profiler_traces  # optional override
```

---

## v0.7.2 - 2026-07-08

### Scope

Training-stability fix for the Molmo multi-node pretraining run (`mn_molmo_pretrain.yaml`): gradient-norm spikes and a slowly rising training loss traced to a learning-rate imbalance and a loss-masking change, plus miscellaneous cleanup.

### Done

- Lowered the vision-backbone connector param group's LR from `2.0e-4` to `2.0e-5` (was 10x the LLM trunk's `2.0e-5`, and 33x the ViT's `5.0e-6`) — this LR imbalance was producing training-time `grad_norm` spikes above 40.
- Reverted `tokenize_molmo_instance` in `data_modules/vlms/molmo/molmo_data.py` to compute the loss-mask `prompt_len` boundary from text-only tokenization (matching the pre-v0.7.0 behavior), restoring the last known-good training loss curve.
  - **Caveat:** an interim change (part of v0.7.0) had computed `prompt_len` by re-tokenizing the prompt *with* images, which correctly excludes Molmo's repeated image-patch placeholder tokens from the loss. That correctness fix is what made the reported loss jump from a ~0.3 plateau to a ~1.4 plateau between runs — the ~0.3 numbers were partly diluted by scoring loss on trivially-predictable, repeated placeholder tokens. Reverting restores the old (less precise) masking to match the historical curve; revisit if training quality (not just the loss number) needs the stricter masking back.
- Fixed `pixmo_cap_local.py` manifest/dataset construction to build via `Dataset.from_dict` instead of `Dataset.from_list`, and guard against an empty record set.
- Updated the `pixmo_cap_local` manifest cache path in `mn_molmo_pretrain.sh`.
- Switched `train.py` and `captioning_eval_callback.py` from bare `print()` calls to a configured logger.

## v0.7.1 - 2026-07-07

### Scope

Follow-up to v0.7.0's ZeRO-3 generation hardening — three iterations to find a `generate()` path for Molmo that actually completes under DeepSpeed ZeRO-3 sharding, rather than failing before the first token.

### Done

- All ranks now fully participate in generation (previously only rank 0 collected outputs and non-zero ranks exited early, risking a hang); results are gathered back to rank 0 via `torch.distributed.all_gather_object`.
- Diagnosed the actual crash: `HF GenerationMixin._validate_model_kwargs()` inspects `model.forward`'s signature, but under ZeRO-3 the `DeepSpeedEngine`-wrapped model's apparent forward signature drops Molmo-specific kwargs (`images`, `image_masks`, `image_input_idx`), raising a `ValueError` on every rank before any token was generated.
- First attempted fix: pre-compute `inputs_embeds` with image features baked in before calling `generate()`, plus class-level monkey-patches of `prepare_inputs_for_generation`/`_update_model_kwargs_for_generation` to strip empty `DynamicCache` instances and safely advance `position_ids`/`attention_mask`.
- Final fix: replaced HF `generate()` entirely with a self-contained manual autoregressive decode loop in `MolmoModel.generate()` that calls `self.molmo()` (forward) directly every step, fully sidestepping `_validate_model_kwargs`. Handles images/masks only on the first step, greedy and multinomial sampling, EOS detection with pad-filling for finished sequences, `synced_gpus` for ZeRO-3 straggler protection, and an optional KV cache.

## v0.7.0 - 2026-07-07

### Scope

Standardization of the inference pipeline, FastAPI production serving hardening, and multi-node DeepSpeed generation optimizations for extreme evaluation speedups.

### Done

- **Inference Pipeline Standardization:**
  - Introduced `InferenceRunner` and `InferenceResult` contracts in `inference/contract.py` and `inference/runner.py`.
  - Standardized text and mask generation output structures, hiding VLM-specific generation quirks behind a unified runner abstraction.
  - Re-wrote `fastapi.py` to completely rely on the new runner abstraction for production-ready, standardized inference serving.
  
- **Multi-Node & ZeRO-3 Generative Evaluation Hardening:**
  - Fixed a major DeepSpeed ZeRO-3 deadlock in `GenerativeEvalCallback` where only rank 0 would call `model.generate()`, leaving ranks 1..N hanging at a barrier and starving the AllGather collective. Now all ranks execute the forward pass safely.
  - Implemented `DistributedSampler` in the validation generative callback to chunk the dataset across all ranks. 64 validation samples are now distributed evenly across all 48 GPUs (creating a 16x speedup over sequential duplication), gathered back to rank 0 via `torch.distributed.all_gather_object` for metric logging.
  - Added `synced_gpus=is_distributed` to model generation kwargs to protect against ZeRO-3 straggler hangs when different batches reach `<EOS>` at varying step counts.

- **Molmo Inference Enhancements:**
  - Temporarily disabled KV cache generation (`use_cache=False`) for Molmo-7B during evaluation to bypass an `AttributeError` caused by changing KV cache structures in `transformers > 4.50.3`.
  - Safely monkey-patched `MolmoForCausalLM.prepare_inputs_for_generation` to strip empty `DynamicCache` instances injected by newer HF transformers before the first forward pass, preventing index-out-of-bounds crashes during generation.
  - Synthesised proper `position_ids` directly from `attention_mask` inside `MolmoModel.generate` to satisfy HF generation requirements.

- **FastAPI Inference Generalization:**
  - Generalized `run_inference` in `fastapi.py` to seamlessly handle text-only tasks (e.g. captioning) on VLM-only architectures without crashing on missing layout extraction functions or missing SAM masks.

## v0.6.0 - 2026-06-28

### Scope

Molmo pre-training readiness: local PixMo-Cap loader, config-driven per-layer learning rates, and generation-based captioning eval callback.

### Done

- Added `datasets/pixmo/pixmo_cap_local.py` — registry key `pixmo_cap_local`:
  Loads locally downloaded PixMo-Cap from sharded flat directories at `/fsxvision_new/pratyush.jena/Datasets/pixmo-cap-images/extracted_dataset` (75 shards, 00000–00074).
  Scans `{shard}/{stem}.json` + `{stem}.{jpg,png}` pairs; silently skips missing images.
  Kwargs: `data_path`, `mode` (`"captions"` | `"transcripts"` | `"transcript_and_caption"`), `shards` (subset list), `sample_limit`.
- Added `build_param_groups` in `train/train_utils.py`:
  Config-driven per-layer LR via pattern matching (glob + regex against `model.named_parameters()`).
  Each spec: `{pattern, lr, weight_decay}`. Unmatched params fall into a default group at `base_lr`.
  Decay/no-decay split per group (2-D+ tensors get WD, biases/norms don't). All groups logged at startup.
- Per-layer LR wired end-to-end:
  `CustomTrainer.create_optimizer` overrides base HF optimizer when `param_groups_cfg` is non-empty.
  `run_training` reads `param_groups:` list from config and passes it to both VLM-only and VLMSam trainers.
  VLM-only path now uses `CustomTrainer` (was `Trainer`) for parity.

  YAML example:
  ```yaml
  param_groups:
    - pattern: "backbone.vision_model.*"
      lr: 2.0e-6
      weight_decay: 0.0
    - pattern: "backbone.multi_modal_projector.*"
      lr: 1.0e-5
      weight_decay: 0.01
    - pattern: "backbone.language_model.*"
      lr: 5.0e-6
      weight_decay: 0.01
  ```
- Added `train/captioning_eval_callback.py` — `GenerativeEvalCallback(TrainerCallback)`:
  Runs `model.generate()` on a fixed random subset of the val set every `eval_steps` global steps.
  Computes BLEU-4 via `sacrebleu`; logs score + generated examples as W&B table.
  Config: `captioning_eval: {eval_steps, num_samples, max_new_tokens, do_sample, temperature}`.
  Gracefully no-ops if `sacrebleu` / `wandb` not installed (logs warnings).
  Registered automatically in `run_training` when `captioning_eval:` block is present.

  YAML example:
  ```yaml
  captioning_eval:
    eval_steps: 500
    num_samples: 64
    max_new_tokens: 256
  ```
- `sacrebleu` added to `requirements-test.txt`.
- 40/40 tests pass.

### Todo

- Task-wise metric monitoring — `ComputeMetrics` dispatch per dataset/task type.

---

## v0.5.0 - 2026-06-28

### Scope

Generic multi-task dataset infrastructure: AnnotationSpec, canonical coordinate normalization, layout→layout/ reorganization, PixMo suite (10 keys), public/academic suite (23 keys), weighted data mixing. Total registry: 46 keys.

### Done

- Added `AnnotationSpec` dataclass to `registry/utils.py`:
  `has_localization`, `localization_type` (`"none"` | `"bbox"` | `"point"` | `"bbox+mask"` | `"point+mask"`), `label_classes`, `parse_predictions_fn`.
  `DataArguments.__post_init__` promotes legacy `layout_classes` / `extract_from_labels_fn` into `annotation_spec` automatically — all existing layout datasets work without changes.
- Added `datasets/normalization.py`:
  canonical `[0, 1]` float coordinate space; `pixel_bbox_to_01`, `pixel_point_to_01`, format converters (`xywh_to_xyxy`, `cxcywh_to_xyxy`); VLM-family serializers (`bbox_01_to_qwen_tokens`, `bbox_01_to_generic_tokens`, `point_01_to_generic_tokens`); Molmo point-space converters.
- Reorganized `datasets/` into three subpackages:
  `datasets/layout/` — 18 existing dataset files + `utils.py`; `__init__.py` registers all on import, guards `hierlay`/`hierlay_v2` with try/except for absent local mapping files.
  `datasets/pixmo/` — new (see below).
  `datasets/public/` — new (see below).
  All import sites updated (`train.py`, `inference/`, `archive/`).
- Added `localization_mode` kwarg flow:
  `build_single_dataset` injects `localization_mode` (`"sam"` | `"autoregressive"`) into `DataArguments.get_source_kwargs`. All 16 layout `format_source` functions accept and silently ignore it (backward compat). New datasets use it to branch coordinate serialization.
- Added data mixing infrastructure (`train/train_utils.py`):
  `WeightedMixDataset` — map-style weighted sampling via shuffled flat index, same total length as `ConcatDataset`.
  `IterableDatasetMixture` — port of `allenai/molmo` iterable mixture; `weighted` (probabilistic) and `stratified` (count-tracking) strategies; `total_samples` cap.
  `build_mixed_dataset` — selects implementation based on weights uniformity and dataset type; falls back to `ConcatDataset` when all weights equal (backward compat).
  Config: `datasets.mixing.{strategy, seed, total_samples}` block.
- Added `datasets/pixmo/` — 10 registry keys from `allenai/pixmo-*`:
  `pixmo_cap` (modes: `captions`, `transcripts`, `transcript_and_caption`), `pixmo_cap_qa`, `pixmo_ask_model_anything`, `pixmo_points` (Molmo `<point>` token format), `pixmo_count`, `pixmo_point_explanations`, `pixmo_docs_{charts,diagrams,tables,other}`.
  Pointing datasets: `AnnotationSpec(has_localization=True, localization_type="point")`.
- Added `datasets/public/` — 23 registry keys:
  HF-hosted (streamable): `chartqa`, `textvqa`, `docvqa`, `vqa2`, `okvqa`, `aokvqa`, `scienceqa`, `infovqa`, `mathvista`, `realworldqa`, `mmmu` (multi-subject, `subjects` kwarg), `ai2d`, `vsr`.
  Local-download (includes `download()` helper, `data_path` required): `dvqa`, `tallyqa`, `plotqa`, `figureqa`, `tabwmp`, `scenetextqa`, `countbenchqa`, `clockbench`, `vizwiz`, `hateful_memes` (DUA-gated, `download()` raises `NotImplementedError`).

### Todo

- Organize `datasets/public/` into domain subfolders (`vqa/`, `document/`, `science/`, `chart/`).
- Task-wise metric monitoring — `ComputeMetrics` dispatch per dataset/task type.
- Add `sam2`/`sam3` versioned data/model paths.
- Production inference serving: harden FastAPI entrypoint.

## v0.4.1 - 2026-06-26

### Scope

VLM-only training path — skip SAM entirely, train on CE loss only. Validated across all three VLM families and the encoder swap path. 80/80 encoder×decoder matrix confirmed with no regressions in either mode.

### Done

- Added `src/training_core/data_modules/sam/noop/noop_sam_data.py`:
  `NoopSAMDataModule` registered as `"none"` in `SAMDataModuleRegistry`.
  `format_source` returns structurally valid dict with empty masks; `get_collator` returns a noop that leaves the batch unchanged.
  Requires no changes to `build_single_dataset` — the registry lookup just resolves to the noop implementation.
- Added `src/training_core/models/vlm_only.py`:
  `VLMOnly(PreTrainedModel)` — wraps VLM backbone, CE loss only, no `text_hidden_fcs_layout`, no SAM head.
  Pops SAM keys from `forward` kwargs so absent SAM batch fields do not crash.
  Exposes `.backbone` identically to `VLMSam` for full encoder swap compatibility with `swap_vision_encoder`.
- Updated `src/training_core/train/train.py`:
  `vlm_only = cfg.get("sam_version", "sam1") == "none"` gates two branches:
  VLM-only builds `VLMOnly` + base HF `Trainer` (base `Trainer` required — `CustomTrainer.prediction_step` hardcodes mask output indices incompatible with `VLMOnly`);
  SAM path unchanged — `VLMSam` + `CustomTrainer` + `ComputeMetrics`.
  Encoder swap block shared; both paths expose `.backbone`.
- Added smoke configs under `src/configs/smoke/`:
  `smoke_qwen25_vlm_only.yaml`, `smoke_molmo_o_vlm_only.yaml`, `smoke_gemma3_vlm_only.yaml`,
  `smoke_qwen25_vlm_only_ve_swap.yaml` (VLM-only + SigLIP encoder swap; `skip_final_model_save: true` to avoid Qwen shared-tensor safetensors error — production use: add `save_safetensors: false` under trainer).

### Validated

GPU smoke tests on 8× H100 host (DocLayNet v1.2, 3 steps, DeepSpeed ZeRO-3):

| Smoke | Model | Encoder | train_loss | EXIT_CODE |
|---|---|---|---|---|
| `smoke_qwen25_vlm_only` | Qwen2.5-VL-7B | native | 2.9540 | 0 |
| `smoke_gemma3_vlm_only` | Gemma3-12B | native | 2.6874 | 0 |
| `smoke_molmo_o_vlm_only` | Molmo-7B-O | native | 5.1440 | 0 |
| `smoke_qwen25_vlm_only_ve_swap` | Qwen2.5-VL-7B | SigLIP (swapped) | 3.1065 | 0 |

Encoder×decoder swap matrix rerun (2 previously stale-pickle jobs, `--force-reextract`):
`Completed 2 runs; failures=0` — confirming full 80/80 passing with VLM-only mode included.

### Notes

- **Shared tensor save (Qwen + VE swap)**: `backbone.qwen.visual` and `backbone.qwen.model.visual` are aliases to the same object. After encoder swap, safetensors errors on shared tensors at save time. Workaround for smoke: `skip_final_model_save: true`. Production: add `save_safetensors: false` under trainer in config.
- **Stale pickle on node switch**: `ExtractedVisionEncoder` saves `weights.pt` with node-local HF dynamic module references. Loading on a new node fails. Fix: pass `--force-reextract` to `run_encoder_swap_matrix.py`.

## v0.4.0 - 2026-06-25

### Scope

Encoder×decoder swap matrix validation — full 80-job matrix (4 decoders × 10 encoders × 2 modes) passing with W&B logging to `img-2-svg-pretraining-encoder-swap-matrix`. Infrastructure for the matrix runner, `MolmoVisionBackboneAdapter`, and two bug fixes that unblocked the previously failing Molmo cross-encoder pairs.

### Done

- Added [src/training_core/matrix/encoder_swap_matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/matrix/encoder_swap_matrix.py):
  `MatrixRunSpec` dataclass encoding one (decoder, encoder, mode) cell;
  `ENCODER_SPECS` — registry of all 10 encoder kinds (`native`, `clip`, `siglip`, `siglip2`, `metaclip`, `metaclip2`, `openvision`, `openvision2`, `extracted`, family-specific adapter path for Molmo);
  `DECODER_SPECS` — registry of all 4 decoder kinds (`qwenvlm`, `gemmavlm`, `molmo7b-d`, `molmo7b-o`).
- Added [scripts/run_encoder_swap_matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/run_encoder_swap_matrix.py):
  generates per-job YAML configs at runtime,
  runs 80 jobs in parallel across 8 GPUs,
  logs each run to W&B under project `img-2-svg-pretraining-encoder-swap-matrix`,
  and writes a CSV summary to `outputs/encoder_swap_matrix/`.
  Supports `--encoders` and `--decoders` flags for subset runs.
- Added `MolmoVisionBackboneAdapter` to the Molmo model path:
  handles cross-encoder swaps for Molmo decoders by wrapping the new encoder alongside the original `image_projector`,
  routes through `_molmo_patchify_images` for non-native encoders so the OLMo backbone receives correctly shaped patch tensors.
- Fixed [commit 775b2b7]: `image_masks` in `_molmo_patchify_images` was shape `[B,1]` instead of `[B,1,grid_h*grid_w]`.
  This caused a broadcast explosion (`[B,1,N,D]` → `[B,B,N,D]`) in the OLMo backbone's `pad_and_partial_pad` attention path.
  Fixed to per-patch shape `[B,1,N_patches]` so the attention mask broadcasts correctly.
- Fixed [commit a59ae02]: `OpenVisionEncoder` was calling `open_clip.create_model_from_pretrained(arch, "hf-hub:...")` with the hub path as the `pretrained` argument.
  The correct call form is `open_clip.create_model_from_pretrained("hf-hub:...")` with the full hub path as `model_name`.
  Also installed `open_clip_torch 3.3.0` in the container environment.
- Added `encoder_swap_matrix_report.md` at the repository root with full 80/80 per-job results and pass/fail details.

### Validated

- Full encoder×decoder swap matrix on the shared 8x H100 host:
  `python scripts/run_encoder_swap_matrix.py --report-to wandb`
- Result:
  `80/80 jobs passed`
- W&B run logged under project:
  `img-2-svg-pretraining-encoder-swap-matrix`
- CSV summary and per-job logs:
  `outputs/encoder_swap_matrix/`
- Previously failing pairs (Molmo cross-encoder) now pass:
  `molmo7b-d + extracted-molmo7bo` and `molmo7b-o + extracted-molmo7bd`

### Todo

- Add `sam2`/`sam3` versioned data/model paths.
- Production inference serving: harden the FastAPI entrypoint for real serving loads.
- Add a distinct `openvision2` checkpoint (currently shares the `openvision` hub path).

## v0.3.2 - 2026-05-20

### Scope

Builders checkpoint for the modular VLM + SAM upgrade, adding the three builder scripts: `extract_vision_encoder`, `swap_vision_encoder` (with Linear adapter for dimension mismatch), and `build_vlm_sam`. CPU-safe contract tests cover the no-adapter and adapter-inserted paths; GPU-gated integration tests cover the full swap + forward and the extract + reload round-trip.

### Done

- Added [src/training_core/builders/extract_vision_encoder.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/builders/extract_vision_encoder.py):
  `extract_vision_encoder(vlm_family, vlm_checkpoint, encoder_name, output_dir)` — loads the VLM, detaches the vision module, saves `weights.pt` + `preprocessor.json` to `output_dir/encoder_name/`.
  `embed_dim` is probed from `config.hidden_size` for Qwen/Gemma and from `image_projector.in_features` for Molmo.
  CLI entrypoint: `python -m training_core.builders.extract_vision_encoder ...`.
- Added [src/training_core/builders/swap_vision_encoder.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/builders/swap_vision_encoder.py):
  `swap_vision_encoder(vlm_wrapper, new_encoder, vlm_family)` — replaces the VLM-internal vision backbone.
  When `new_encoder.embed_dim != projector.in_features`, inserts `nn.Linear(new_dim, original_dim)` before the VLM-internal projector (`qwen.visual.merger` / `gemma.multi_modal_projector` / `molmo.vision_backbone.image_projector`).
  The adapter is stored as `vlm_wrapper.vision_adapter` and returned for optimizer-group routing.
  The VLM-to-SAM projector `text_hidden_fcs_layout` is unaffected because it lives in LLM hidden-state space.
- Added [src/training_core/builders/build_vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/builders/build_vlm_sam.py):
  `build_vlm_sam(vlm_family, vlm_checkpoint, sam_version, sam_checkpoint, ...)` — constructs a fully wired `VLMSam` composite plus its `DataModule`.
  Optional `vision_encoder` + `vision_encoder_checkpoint` args trigger a vision encoder swap after construction.
  Logs integrity warnings for hidden_size mismatches and missing seg_token_idx.
  CLI entrypoint: `python -m training_core.builders.build_vlm_sam ...`.
- Updated [tests/contracts/test_contracts.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/contracts/test_contracts.py) with three new CPU-safe builder tests:
  `test_builder_modules_importable`,
  `test_swap_no_dim_change_no_adapter` (stub projector in_dim=8, stub encoder embed_dim=8 → adapter is None),
  `test_swap_dim_mismatch_creates_adapter` (stub encoder embed_dim=1152, projector in_dim=8 → `nn.Linear(1152, 8)` inserted).
- Added [tests/model/test_vision_encoder_swap.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/model/test_vision_encoder_swap.py):
  GPU-gated tests for SigLIP-into-Qwen (dims match, no adapter) and CLIP-into-Qwen (dim mismatch, adapter inserted), both including a full `VLMSam.forward()` call with finite loss assertion.
  GPU-gated extract + reload round-trip for `ExtractedVisionEncoder`.

### Validated

- CPU PR profile in Docker after builder addition:
  `python scripts/run_test_suite.py --profile cpu-pr --pytest-args -vv`
- Result:
  `24 passed, 44 deselected`

### Todo

- Run GPU builder integration tests once SigLIP and CLIP checkpoints are cached.
- Add `vision_encoder` + `vision_encoder_checkpoint` to the main training config schema once the first encoder-swapped experiment runs successfully.

## v0.3.1 - 2026-05-20

### Scope

Vision encoder infrastructure checkpoint for the modular VLM + SAM upgrade, adding `VisionEncoderBase`, `VisionEncoderRegistry`, and six encoder families (CLIP, SigLIP v1, SigLIP v2, MetaCLIP v1 + v2, OpenVision v1 + v2) to the registry, plus the `ExtractedVisionEncoder` wrapper for encoders pulled out of VLMs.

### Done

- Added [src/training_core/vision_encoders/base.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/vision_encoders/base.py) with `VisionEncoderBase(ABC, nn.Module)`:
  abstract `embed_dim` property (output feature dimension),
  abstract `preprocessor_config` property (image_mean, image_std, image_size).
- Added `VisionEncoderRegistry` to [src/training_core/registry/registry.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/registry/registry.py) as a sixth registry alongside the existing five.
- Added [src/training_core/vision_encoders/clip/clip_encoder.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/vision_encoders/clip/clip_encoder.py):
  `CLIPVisionEncoder` wrapping `transformers.CLIPVisionModel`,
  registered as `"clip"`.
- Added [src/training_core/vision_encoders/siglip/siglip_encoder.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/vision_encoders/siglip/siglip_encoder.py):
  `SigLIPVisionEncoder` wrapping `SiglipVisionModel`, registered as `"siglip"`;
  `SigLIP2VisionEncoder` wrapping `Siglip2VisionModel`, registered as `"siglip2"`.
  These are separate HF classes and cannot share a loading code path.
- Added [src/training_core/vision_encoders/metaclip/metaclip_encoder.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/vision_encoders/metaclip/metaclip_encoder.py):
  `MetaCLIPVisionEncoder` wrapping `CLIPVisionModel` for Meta's MetaCLIP checkpoints,
  registered under both `"metaclip"` (v1: l14-fullcc2.5b, h14-fullcc2.5b) and `"metaclip2"` (v2: worldwide-huge-quickgelu).
- Added [src/training_core/vision_encoders/openvision/openvision_encoder.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/vision_encoders/openvision/openvision_encoder.py):
  `OpenVisionEncoder` wrapping `open_clip` models for the UCSC-VLAA OpenVision collection,
  loaded via `open_clip.create_model_from_pretrained("hf-hub:<checkpoint>")`,
  `embed_dim` from `model.visual.output_dim`,
  registered as both `"openvision"` and `"openvision2"`.
  `open_clip_torch` is an optional dependency — guarded with a helpful `ImportError` message.
- Added [src/training_core/vision_encoders/extracted/extracted_encoder.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/vision_encoders/extracted/extracted_encoder.py):
  `ExtractedVisionEncoder` wrapping an `nn.Module` extracted from a VLM,
  with `save(dir)` / `from_saved(dir)` round-trip using `weights.pt` and `preprocessor.json`.
- Added `test_vision_encoder_registry_available_keys` to [tests/contracts/test_contracts.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/contracts/test_contracts.py) asserting all 7 keys (`clip`, `siglip`, `siglip2`, `metaclip`, `metaclip2`, `openvision`, `openvision2`) are registered.
- Added [tests/model/test_vision_encoders.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/model/test_vision_encoders.py):
  GPU-gated load + forward smoke for each family,
  parametrized over representative HF checkpoints with expected `embed_dim`.

### Validated

- CPU PR profile in Docker after vision encoder addition:
  `python scripts/run_test_suite.py --profile cpu-pr --pytest-args -vv`
- Result:
  `21 passed, 31 deselected`

### Todo

- Run GPU vision encoder load tests once relevant HF checkpoints are cached.
- Continue to Phase C (builders: extract, swap, build_vlm_sam).

## v0.3.0 - 2026-05-20

### Scope

Molmo VLM family checkpoint for the modular VLM + SAM upgrade, adding `molmovlm` as a third first-class VLM family alongside `qwenvlm` and `gemmavlm`. Covers all three Molmo variants (7B-D with Qwen2 backbone, 7B-O with OLMo backbone, MolmoE-1B with OLMo-1B MoE backbone), resolves the `images` key naming conflict between the Molmo processor output and the SAM collator, and adds the full test matrix entry for the family.

### Done

- Added [src/training_core/data_modules/vlms/molmo/molmo_data.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/vlms/molmo/molmo_data.py) implementing `MolmoCollator` and the `molmovlm` data module registration.
  Molmo requires `apply_chat_template(tokenize=False)` + `processor(text=..., images=...)` rather than the `tokenize=True` path used by Gemma and Qwen.
  Handles variable-crop batching via `_pad_to_max_crops` along the crops dimension.
  Renames the Molmo processor's `images` output to `molmo_images` so it does not collide with the SAM collator's `batch["images"]` (SAM-preprocessed images).
- Added [src/training_core/models/vlms/molmo/molmo_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/molmo/molmo_model.py) implementing `MolmoModel` and the `molmovlm` VLM model registration.
  Uses `trust_remote_code=True` for both model loading and processor.
  Probes `text_config.hidden_size`, `config.hidden_size`, and `config.d_model` in order to read `hidden_size` correctly across all three variants.
  The `set_model()` method uses `hasattr` fallback attribute probing (`"model"`, `"language_model"`, `"transformer"`) to handle the Qwen2-based (7B-D) and OLMo-based (7B-O, MolmoE-1B) LLM trunks.
  `MolmoModel.forward()` renames `molmo_images` back to `images` before passing to the underlying HF model to preserve the Molmo processor contract.
- Updated [src/training_core/models/vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlm_sam.py) so `_build_generate_kwargs` now passes through `molmo_images`, `image_input_idx`, and `image_masks` in addition to the existing Qwen/Gemma keys.
- Updated [src/training_core/train/train.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/train/train.py) with Molmo module imports for the side-effect registry pattern.
- Updated [tests/support/matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/support/matrix.py) with three new `VLMCheckpointSpec` entries:
  `MOLMO_D_SPEC` (representative, `require_existing_path=True`),
  `MOLMO_O_SPEC`, and `MOLMO_1B_SPEC`.
  Added `selected_molmo_specs()` helper and extended `SUPPORTED_VLM_SPECS` and `SUPPORTED_MOLMO_SPECS`.
- Updated [tests/support/builders.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/support/builders.py) with Molmo module imports.
- Updated [tests/conftest.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/conftest.py) with `molmo_d_model_id`, `molmo_o_model_id`, and `molmo_1b_model_id` session fixtures.
- Added [tests/data/test_molmo_data.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/data/test_molmo_data.py):
  CPU-safe registration check (no checkpoint needed),
  and external data tests gated behind `TRAINING_RUN_LARGE_MODEL_TESTS`:
  collator output keys (including `molmo_images` renamed key and SAM `images` coexistence),
  label masking,
  and batch image padding shape.
- Added [tests/model/test_molmo_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/model/test_molmo_model.py):
  GPU-gated load tests for all three variants (hidden_size assertions: 4096 for 7B-D and 7B-O, 2048 for MolmoE-1B),
  and a forward-loss test for the 7B-D model.
- Added Molmo composite test in [tests/composite/test_vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/composite/test_vlm_sam.py):
  `test_vlm_sam_constructs_with_molmo_d_and_sam1` builds the full VLMSam composite with Molmo-D backbone and SAM1 head and asserts finite loss.
- Updated [tests/contracts/test_contracts.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/contracts/test_contracts.py) so `test_registry_keys_are_available` now asserts `molmovlm` is present in both `DataModuleRegistry` and `VLMModelRegistry`.
- Updated [scripts/discover_test_checkpoints.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/discover_test_checkpoints.py) to discover `TRAINING_TEST_MOLMO_D_MODEL`, `TRAINING_TEST_MOLMO_O_MODEL`, and `TRAINING_TEST_MOLMO_1B_MODEL` from local HF snapshot caches.
- Updated [scripts/run_parallel_test_matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/run_parallel_test_matrix.py) with Molmo granular GPU jobs:
  `cpu-molmo-data` (CPU, priority 0),
  `gpu-molmo-load` (single-GPU, priority 2),
  and `gpu-molmo-composite-contract` (single-GPU, priority 2).
- Added smoke config [src/configs/smoke/real_data_doclaynet_molmo7bd_sam1.yaml](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/configs/smoke/real_data_doclaynet_molmo7bd_sam1.yaml) mirroring the Gemma3 and Qwen2.5 smoke configs.

### Validated

- CPU PR profile in Docker after Molmo addition:
  `python scripts/run_test_suite.py --profile cpu-pr --pytest-args -vv`
- Result:
  `20 passed, 31 deselected`

### Todo

- Run GPU Molmo model load tests once `allenai/Molmo-7B-D-0924`, `allenai/Molmo-7B-O-0924`, and `allenai/MolmoE-1B-0924` are cached on the node under `TRAINING_TEST_MOLMO_*_MODEL` env vars.
- Add Molmo to the representative GPU profile once the 7B-D checkpoint download is confirmed.
- Run the real-data smoke once the Molmo checkpoint path is finalized.
- Continue to Phase B (vision encoder infrastructure).

## v0.2.12 - 2026-05-20

### Scope

Dynamic full-node test-matrix scheduler checkpoint for the modular VLM + SAM upgrade, focused on removing the remaining fixed-phase GPU idle gaps, keeping all 8 assigned H100s busy with granular jobs, and tuning the CPU-side job environment so model loads and tokenization stop bottlenecking the GPU queue.

### Done

- Reworked [scripts/run_parallel_test_matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/run_parallel_test_matrix.py) from a rigid phase launcher into a dynamic resource scheduler.
- Split the old coarse GPU waves into granular jobs:
  CPU contracts/data/stub jobs,
  single-GPU wrapper/composite/trainer/eval tasks,
  per-spec Qwen load tasks,
  and 2-GPU DDP/DataParallel trainer jobs.
- Changed the GPU scheduler so freed GPUs are immediately reused for the next pending 1-GPU job, instead of waiting for a full phase barrier to clear.
- Changed the 2-GPU scheduling so DDP/DataParallel jobs start as soon as two GPUs are free, even while other long single-GPU jobs are still running.
- Added throughput-oriented job env defaults in the matrix runner:
  `HF_ENABLE_PARALLEL_LOADING=true`,
  `HF_PARALLEL_LOADING_WORKERS`,
  `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`,
  `TOKENIZERS_PARALLELISM=false`,
  and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Added [tests/contracts/test_parallel_test_matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/contracts/test_parallel_test_matrix.py) to assert the scheduler keeps granular 1-GPU jobs, preserves the four 2-GPU trainer jobs, and keeps CPU-only jobs out of the GPU pool.
- Updated [README.md](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/README.md) so the documented full-node matrix flow matches the dynamic scheduler rather than the old phased design.

### Validated

- Scheduler contract slice in Docker:
  `python -m pytest -q tests/contracts/test_parallel_test_matrix.py tests/contracts/test_contracts.py -vv`
- Result:
  `10 passed`

- Dynamic full-node matrix on the shared 8x H100 host:
  `python3 scripts/run_parallel_test_matrix.py`
- Result:
  all `29` scheduled jobs passed
- Representative live behavior during the run:
  all 8 GPUs were assigned immediately at startup,
  freed GPUs were backfilled with later Qwen and Gemma jobs,
  and 2-GPU DDP/DataParallel jobs were launched while the last long Qwen load was still running.
- Green log bundle:
  [logs/test_matrix/20260519T202752Z](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/logs/test_matrix/20260519T202752Z)

### Todo

- Decide whether [scripts/run_test_suite.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/run_test_suite.py) should gain an explicit "delegate to dynamic full-node scheduler" mode for local power users, while keeping GitHub Actions on the simpler single-process profile path.
- Consider teaching the dynamic scheduler to estimate per-job memory class explicitly, so the queue can distinguish "load-heavy" and "trainer-heavy" jobs without relying only on manual priority ordering.

## v0.2.11 - 2026-05-20

### Scope

Inference-generalization and documentation-refresh checkpoint for the modular VLM + SAM upgrade, focused on removing the last Qwen-only utility assumptions from the checked-in inference path and bringing the README diagrams in line with the current registry-driven architecture.

### Done

- Added [src/training_core/inference/factory.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/inference/factory.py) to centralize inference-time registry setup, family-aware datamodule construction, and composite model loading for the checked-in `qwenvlm`, `gemmavlm`, and `sam1` paths.
- Updated [src/training_core/inference/utils.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/inference/utils.py) with family-agnostic generation helpers:
  prompt truncation from masked `labels`,
  filtered generate-batch preparation,
  and supervised-target decoding shared by Qwen and Gemma inference flows.
- Updated [src/training_core/inference/fastapi.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/inference/fastapi.py) so single-image inference now routes through registered `vlm_family` and `sam_version` builders instead of assuming an old Qwen-only utility path.
- Updated [src/training_core/inference/val_set_with_generate.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/inference/val_set_with_generate.py) so validation-time generation uses the same family-aware model/data loading and prompt-truncation helpers, including the fixed local CUDA-device binding for spawned workers.
- Added inference utility coverage in [tests/contracts/test_contracts.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/contracts/test_contracts.py) for both Qwen-style and Gemma-style prompt truncation from masked labels.
- Refreshed [README.md](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/README.md), [docs/diagrams/system-architecture.md](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/docs/diagrams/system-architecture.md), and [docs/diagrams/training-flow.md](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/docs/diagrams/training-flow.md) so the documented flow is VLM-generic rather than Qwen-specific.
- Added [scripts/render_diagrams.sh](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/render_diagrams.sh) so Mermaid diagram SVGs can be refreshed from the checked-in `.mmd` sources with one repo-local command.

### Validated

- CPU PR profile in Docker after the inference-helper additions:
  `python scripts/run_test_suite.py --profile cpu-pr --pytest-args -vv`
- Result:
  `16 passed, 23 deselected`

- Representative GPU profile in Docker after the inference generalization:
  `python scripts/run_test_suite.py --profile gpu-representative --pytest-args -vv`
- Result:
  `20 passed, 3 deselected, 2 warnings in 642.91s`

### Todo

- Add dedicated inference integration tests that exercise the public FastAPI-style entrypoints end to end, not just the shared helper and composite-generate layers.
- Decide whether the example inference entrypoints should gain a small config file of their own so serving-time defaults stop carrying historical Qwen-named script conventions.

## v0.2.10 - 2026-05-19

### Scope

GPU-only real-model test-policy checkpoint for the modular VLM + SAM upgrade, focused on removing CPU execution of actual model paths from the checked-in suite, fixing the remaining stale inference imports around the restored `generate()` path, and revalidating both the named runner profiles and the full-node parallel matrix.

### Done

- Updated real-model wrapper/composite tests so actual model execution now runs only on GPU:
  [tests/model/test_qwen_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/model/test_qwen_model.py),
  [tests/model/test_gemma_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/model/test_gemma_model.py),
  [tests/model/test_sam1_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/model/test_sam1_model.py),
  [tests/composite/test_vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/composite/test_vlm_sam.py),
  and [tests/composite/test_vlm_sam_generate.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/composite/test_vlm_sam_generate.py).
- Kept CPU coverage limited to contracts, data modules/processors, SAM batching utilities, and stubbed composite paths.
- Updated [src/training_core/inference/val_set_with_generate.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/inference/val_set_with_generate.py) and [src/training_core/inference/fastapi.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/inference/fastapi.py) so they now import checked-in local inference helpers instead of stale `training_core.without_teacher_forcing.*` modules.
- Tightened [src/training_core/inference/fastapi.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/inference/fastapi.py) return typing/docstrings so they match the current implementation.
- Added an inference-import contract check in [tests/contracts/test_contracts.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/contracts/test_contracts.py) to guard against future regressions to missing-module imports.
- Added automatic CUDA cache cleanup between tests in [tests/conftest.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/conftest.py), which keeps long sequential GPU runs stable and prevents false OOMs in later trainer tests.
- Updated [scripts/run_parallel_test_matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/run_parallel_test_matrix.py) so model-bearing files no longer sit in CPU lanes:
  CPU lanes now cover contracts, data, and stubs,
  GPU phase 1 now covers real model loads plus single-GPU wrapper/composite/trainer checks,
  and GPU phase 2 still covers DDP and DataParallel trainer checks.

### Validated

- Inference import contract in Docker:
  `python -m pytest -q tests/contracts/test_contracts.py::test_inference_modules_import_from_checked_in_package -vv`
- Result:
  `1 passed`

- GPU-only real-model wrapper/composite slice in Docker:
  `python -m pytest -q tests/model/test_qwen_model.py tests/model/test_gemma_model.py tests/model/test_sam1_model.py tests/composite/test_vlm_sam.py tests/composite/test_vlm_sam_generate.py -vv`
- Result:
  `13 passed, 2 warnings in 1046.37s`

- CPU PR profile in Docker after the policy change:
  `python scripts/run_test_suite.py --profile cpu-pr --pytest-args -vv`
- Result:
  `14 passed, 23 deselected`

- Representative GPU profile in Docker after adding CUDA cleanup:
  `python scripts/run_test_suite.py --profile gpu-representative --pytest-args -vv`
- Result:
  `20 passed, 3 deselected, 2 warnings in 577.93s`

- Full-node phased matrix in Docker with the updated scheduler:
  `python3 scripts/run_parallel_test_matrix.py`
- Result:
  `43 passed`
- Green log bundle:
  [logs/test_matrix/20260519T152920Z](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/logs/test_matrix/20260519T152920Z)

### Todo

- Generalize the inference package beyond the current Qwen-oriented utility flow now that the checked-in imports are self-contained again.
- Decide whether the representative GPU profile should remain a single-process sequential pytest run, or whether it should delegate to the phased matrix launcher for even closer parity with the full-node path.

## v0.2.9 - 2026-05-19

### Scope

Legacy composite-generate parity checkpoint for the modular VLM + SAM upgrade, focused on restoring the old `QwenSam.generate()` text-plus-mask behavior onto generalized `VLMSam`, covering both real `Qwen2-VL` and real `Gemma3` paths, and tightening the contract/composite tests around that interface.

### Done

- Updated [src/training_core/models/vlms/base.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/base.py) so `VLMBase` now requires a `generate(...)` path alongside `forward(...)` and embedding resize support.
- Updated [src/training_core/models/vlms/qwen/qwen_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/qwen/qwen_model.py) and [src/training_core/models/vlms/gemma/gemma_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/gemma/gemma_model.py) so both wrappers expose `generate(...)` through the modular VLM interface.
- Updated [src/training_core/models/vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlm_sam.py) to restore generalized composite generation:
  prompt-time model-input filtering,
  continuation decoding,
  generated hidden-state reconstruction,
  `[SEG]` embedding extraction from generated tokens,
  and SAM mask decoding from generated segment embeddings.
- Tightened `VLMSam.generate()` input filtering so training-only extras such as `position_ids` are not forwarded into generation for families that do not need them, which restores the Qwen2-VL prompt-time attention semantics seen in the old path.
- Added [tests/composite/test_vlm_sam_generate.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/composite/test_vlm_sam_generate.py), which covers:
  the generated-hidden-state reconstruction helper,
  real `Qwen2-VL` composite generation,
  and real `Gemma3` composite generation.
- Updated [tests/contracts/test_contracts.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/contracts/test_contracts.py) and [tests/composite/test_stubbed_composite.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/composite/test_stubbed_composite.py) so their VLM test doubles implement the now-required `generate(...)` interface.
- Updated [tests/composite/test_vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/composite/test_vlm_sam.py) so its representative Qwen constructor test uses a CPU-safe attention backend when CUDA is not available, instead of defaulting to `flash_attention_2` on CPU.

### Validated

- Real-model composite generate regression in Docker:
  `python -m pytest -q tests/composite/test_vlm_sam_generate.py -vv`
- Result:
  `3 passed in 324.06s`

- Broader contract-plus-composite slice in Docker:
  `python -m pytest -q tests/contracts/test_contracts.py tests/composite/test_stubbed_composite.py tests/composite/test_vlm_sam.py tests/composite/test_vlm_sam_generate.py -vv`
- Result:
  `9 passed, 1 skipped in 1272.99s`
- The single skip was the legacy-gated representative `Qwen2.5-VL + SAM1` constructor test, which remains behind `TRAINING_RUN_LARGE_MODEL_TESTS=1`.

- Explicit representative `Qwen2.5-VL + SAM1` constructor rerun in Docker with the large-model flag enabled:
  `TRAINING_RUN_LARGE_MODEL_TESTS=1 python -m pytest -q tests/composite/test_vlm_sam.py::test_vlm_sam_constructs_with_representative_qwen_and_sam1 -vv`
- Result:
  `1 passed in 9.75s`

### Todo

- Clean up the remaining stale `training_core.without_teacher_forcing.*` imports in the inference package so the restored model-side `generate()` path is matched by a self-contained checked-in inference utility layer.
- Decide whether prompt truncation for future composite-generate tests should be centralized through a shared inference helper rather than kept as a test-local helper.

## v0.2.8 - 2026-05-19

### Scope

Real-data eval-artifact checkpoint for the modular VLM + SAM upgrade, focused on proving that `Qwen2.5-VL + SAM1` and `Gemma3 + SAM1` both complete a real trainer train+eval smoke on actual `DocLayNet v1.2` data, write validation visualizations to disk, and keep that path covered by a dedicated regression test.

### Done

- Updated [src/training_core/train/train.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/train/train.py) and [src/training_core/train/train_utils.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/train/train_utils.py) so dataset specs can carry `dataset_kwargs`, which makes smoke runs configurable without hardcoding more local paths into code.
- Updated [src/training_core/datasets/doclaynet.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/datasets/doclaynet.py) so `doclaynet` can consume a streamed, materialized subset of `ds4sd/DocLayNet-v1.2` and normalize both the older `objects` layout and the newer `bboxes`/`category_id` layout into the same formatter.
- Added real-data smoke configs:
  [src/configs/smoke/real_data_doclaynet_qwen25_sam1.yaml](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/configs/smoke/real_data_doclaynet_qwen25_sam1.yaml)
  and
  [src/configs/smoke/real_data_doclaynet_gemma3_sam1.yaml](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/configs/smoke/real_data_doclaynet_gemma3_sam1.yaml).
- Fixed two shared evaluation-path assumptions in [src/training_core/train/custom_trainer.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/train/custom_trainer.py):
  fallback to input `labels` when Hugging Face returns `labels=None`,
  and fallback to the logits device when Hugging Face returns `loss=None` during eval.
- Added [tests/trainer/test_vlm_trainer_eval_artifacts.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/trainer/test_vlm_trainer_eval_artifacts.py), a real-model regression that runs `CustomTrainer` with eval enabled and asserts that validation PNGs are written for both representative families.

### Validated

- Real-data `Qwen2.5-VL + SAM1` smoke in Docker with streamed `DocLayNet v1.2`:
  `TRAINING_CONFIG_PATH=/code/src/configs/smoke/real_data_doclaynet_qwen25_sam1.yaml /environments/docgrounding_env/bin/python -m training_core.train.train`
- Result:
  completed one train step plus eval, saved final checkpoint under
  `outputs/smoke_real_data_doclaynet_qwen25_sam1/checkpoints/final`,
  and wrote validation images under
  `outputs/smoke_real_data_doclaynet_qwen25_sam1/validation_steps/smoke_real_data_doclaynet_qwen25_sam1/1/0/`.

- Real-data `Gemma3 + SAM1` smoke in Docker with streamed `DocLayNet v1.2`:
  `TRAINING_CONFIG_PATH=/code/src/configs/smoke/real_data_doclaynet_gemma3_sam1.yaml /environments/docgrounding_env/bin/python -m training_core.train.train`
- Result:
  completed one train step plus eval, saved final checkpoint under
  `outputs/smoke_real_data_doclaynet_gemma3_sam1/checkpoints/final`,
  and wrote validation images under
  `outputs/smoke_real_data_doclaynet_gemma3_sam1/validation_steps/smoke_real_data_doclaynet_gemma3_sam1/1/0/`.

- Real-model eval-artifact regression in Docker:
  `python -m pytest -q tests/trainer/test_vlm_trainer_eval_artifacts.py -vv`
- Result:
  `2 passed in 68.30s`

### Todo

- Restore the old custom `generate()` parity from the original `QwenSam` path onto the generalized `VLMSam`, so inference helpers that expect text-plus-mask generation are not relying on stale behavior.
- Decide whether the real-data smoke configs should remain `DocLayNet v1.2`-specific examples or be generalized into a reusable "streamed subset" pattern for other public datasets.

## v0.2.7 - 2026-05-19

### Scope

HF-unblocked full-matrix checkpoint for the modular VLM + SAM test system, focused on downloading the missing official checkpoints, using the whole node more effectively, and getting one green end-to-end run across CPU, single-GPU, DDP, and DataParallel coverage.

### Done

- Stored the local Hugging Face token in ignored secrets storage at `api_keys/hf_token` with `0600` permissions so gated checkpoint downloads can be done from inside the container without hardcoding credentials into commands.
- Downloaded and cached the missing real checkpoints needed for the broader matrix:
  `Qwen/Qwen2-VL-2B-Instruct`,
  `Qwen/Qwen3-VL-4B-Instruct`,
  and official `google/gemma-3-4b-it`.
- Extended [scripts/discover_test_checkpoints.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/discover_test_checkpoints.py) so it now discovers:
  `TRAINING_TEST_QWEN2_MODEL`,
  `TRAINING_TEST_QWEN25_MODEL`,
  `TRAINING_TEST_QWEN3_MODEL`,
  `TRAINING_TEST_QWEN3_MOE_MODEL`,
  `TRAINING_TEST_GEMMA3_MODEL`,
  and `TRAINING_TEST_SAM1_CHECKPOINT`.
- Added [scripts/run_parallel_test_matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/run_parallel_test_matrix.py) to use the node more effectively:
  4 CPU lanes,
  a first GPU phase for 1-GPU tests,
  and a second GPU phase for 2-GPU trainer tests.
- Fixed [tests/trainer/test_vlm_trainer_ddp.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/trainer/test_vlm_trainer_ddp.py) and [tests/trainer/test_vlm_trainer_dataparallel.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/trainer/test_vlm_trainer_dataparallel.py) so their worker subprocesses now honor the outer `CUDA_VISIBLE_DEVICES` assignment instead of always snapping back to `0,1`.
- Updated [tests/composite/test_vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/composite/test_vlm_sam.py) so the real `Gemma3 + SAM1` composite external test uses CUDA when available, which removes the previous CPU-only long pole for that case.
- Updated the parallel launcher to phase GPU work instead of launching every GPU-heavy lane at once, avoiding avoidable OOM noise while still using the node aggressively.

### Validated

- Official Gemma cache fill in Docker with the local token:
  `google/gemma-3-4b-it`
- Result:
  downloaded to
  `/fsxvision_new/anirudh.srinivasan/hf_cache/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767`

- Focused rerun of the fixed 2-GPU lanes:
  `tests/trainer/test_vlm_trainer_ddp.py`
  and
  `tests/trainer/test_vlm_trainer_dataparallel.py`
- Result:
  both files passed after the device-assignment fix.

- Focused rerun of the accelerated Gemma composite file:
  `tests/composite/test_vlm_sam.py`
- Result:
  `2 passed`

- Final full-node phased matrix in Docker:
  `python3 scripts/run_parallel_test_matrix.py`
- Result:
  `39 passed`
- Green log bundle:
  [logs/test_matrix/20260519T082819Z](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/logs/test_matrix/20260519T082819Z)

### Todo

- Consider teaching the parallel launcher to inspect free GPU memory before assigning phases, in case the node is shared with other workloads.
- Decide whether the 1-GPU phase should keep using dedicated GPUs `0,1,2` by default or accept a configurable GPU map.
- Add the same phased strategy to the self-hosted GPU GitHub Actions workflow once the runner-side checkpoint variables are finalized.

## v0.2.6 - 2026-05-19

### Scope

Test-selection cleanup checkpoint for the modular VLM + SAM matrix, focused on removing noisy optional skips from plain `pytest` runs, keeping heavyweight suites opt-in, and adding a local checkpoint discovery helper for manual/nightly paths.

### Done

- Updated [tests/conftest.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/conftest.py) so optional `external`, `gpu`, `ddp`, and `dataparallel` suites are now deselected by default unless they are explicitly enabled through the structured runner, legacy env flags, or marker selection.
- Updated [scripts/run_test_suite.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/run_test_suite.py) to export `TRAINING_TEST_INCLUDE_EXTERNAL=1` whenever a profile selects external checkpoint-backed coverage.
- Added [scripts/discover_test_checkpoints.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/discover_test_checkpoints.py) to print suggested `TRAINING_TEST_*` exports from local HF cache snapshots and the known `sam1` checkpoint path.
- Documented that `Qwen2.5-VL + SAM1` and `Gemma3 + SAM1` remain the representative default matrix, while heavier `Qwen3` and `Qwen3-MoE` paths stay manual/nightly inputs.

### Validated

- Plain layer run in Docker:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub PYTHONPATH=/code/src python -m pytest -q tests/contracts tests/data tests/model tests/composite -rs -vv`
- Result:
  `12 passed, 21 deselected`

- Representative GPU profile in Docker after the selection cleanup:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub PYTHONPATH=/code/src python scripts/run_test_suite.py --profile gpu-representative --pytest-args -rs -vv`
- Result:
  `10 passed, 8 deselected, 2 warnings in 358.55s`

- Local checkpoint discovery helper:
  `python scripts/discover_test_checkpoints.py`
- Result:
  found local exports for `TRAINING_TEST_QWEN25_MODEL`, `TRAINING_TEST_QWEN3_MODEL`, `TRAINING_TEST_QWEN3_MOE_MODEL`, and `TRAINING_TEST_SAM1_CHECKPOINT`, with `TRAINING_TEST_QWEN2_MODEL` still missing.

### Todo

- Decide whether `qwen2vl` should stay env-only until a stable local checkpoint is available, or be backed by a deliberately chosen smaller cached representative.
- Keep larger `Qwen3` and `Qwen3-MoE` validation in manual/nightly profiles unless we explicitly carve out a heavyweight profile with different runtime expectations.
- Wire the GPU workflow to real repository or organization variables for all checkpoint env vars on the target self-hosted runners.

## v0.2.5 - 2026-05-19

### Scope

Systematic cross-family test-matrix checkpoint for the modular VLM + SAM upgrade, focused on replacing the ad hoc test layout with layer-based suites, adding representative Qwen + Gemma trainer coverage, and checking in a hybrid GitHub Actions model.

### Done

- Reorganized the suite into stable layers under:
  [tests/contracts](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/contracts),
  [tests/data](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/data),
  [tests/model](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/model),
  [tests/composite](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/composite),
  and [tests/trainer](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/trainer).
- Added shared matrix/runtime helpers in:
  [tests/support/matrix.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/support/matrix.py),
  [tests/support/builders.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/support/builders.py),
  and [tests/support/runtime.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/support/runtime.py).
- Added explicit SAM data coverage for packed masks, `mask_counts`, unpack round-trips, and shared `sam_collator` behavior.
- Added explicit Qwen data routing coverage for `qwen2vl`, `qwen2.5vl`, and `qwen3vl`.
- Added representative GPU wrapper/composite smokes for both `Qwen2.5-VL` and `Gemma3`.
- Added representative trainer smokes for both families across:
  single-GPU,
  2-GPU `torchrun` / DDP,
  and 2-GPU single-process `DataParallel`.
- Fixed a Qwen-specific `DataParallel` bug by making collated `position_ids` batch-first in [src/training_core/data_modules/qwen/qwen_data.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/qwen/qwen_data.py) and normalizing back to the HF Qwen layout in [src/training_core/models/vlms/qwen/qwen_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/qwen/qwen_model.py).
- Added the canonical runner at [scripts/run_test_suite.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/run_test_suite.py) with:
  layer/runtime selection,
  legacy env-flag compatibility,
  and external-checkpoint preflight.
- Added repo-owned CI environment bootstrap files:
  [requirements-test.txt](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/requirements-test.txt)
  and [scripts/setup_test_env.sh](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/scripts/setup_test_env.sh).
- Added hybrid GitHub Actions workflows:
  [ci-cpu.yml](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/.github/workflows/ci-cpu.yml)
  and [ci-gpu.yml](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/.github/workflows/ci-gpu.yml).

### Validated

- CPU PR profile in Docker:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub PYTHONPATH=/code/src python scripts/run_test_suite.py --profile cpu-pr --pytest-args -vv`
- Result:
  `12 passed, 21 deselected`

- Representative GPU profile in Docker:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub PYTHONPATH=/code/src python scripts/run_test_suite.py --profile gpu-representative --pytest-args -vv`
- Result:
  `10 passed, 8 deselected, 2 warnings in 280.41s`

- Layer-focused contracts/data/model/composite validation in Docker:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 PYTHONPATH=/code/src python -m pytest -q tests/contracts tests/data tests/model tests/composite -vv`
- Result:
  `22 passed, 6 skipped, 2 warnings in 41.27s`

### Todo

- Wire the GPU workflow to real repository or organization variables for all checkpoint env vars on the target self-hosted runners.
- Expand the nightly/manual matrix to broader Qwen version coverage once `TRAINING_TEST_QWEN2_MODEL`, `TRAINING_TEST_QWEN3_MODEL`, and `TRAINING_TEST_QWEN3_MOE_MODEL` are populated in CI.
- Add Molmo into the same matrix once the runtime implementation lands.
- Add `sam2` and `sam3` into the same matrix once those versioned data/model paths exist.

## v0.2.4 - 2026-05-18

### Scope

SAM batching checkpoint for the modular Gemma3 path, focused on fixing the remaining single-process multi-GPU `DataParallel` failure without regressing the existing DDP, trainer, or inference-adjacent paths.

### Done

- Updated [src/training_core/data_modules/sam/sam_data.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/sam/sam_data.py) so the SAM collator now packs per-sample mask lists into a padded batch tensor plus `mask_counts`.
- Added reusable SAM mask unpacking helpers so downstream code can recover per-sample `[N_i, H_i, W_i]` tensors from the padded batch.
- Updated [src/training_core/models/vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlm_sam.py) and [src/training_core/models/sam/sam1/sam1_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/sam/sam1/sam1_model.py) to consume the packed SAM batch, crop GT masks back to each sample's original image size, and preserve eval/debug outputs in per-sample form.
- Updated [src/training_core/inference/val_set_with_generate.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/inference/val_set_with_generate.py) to unpack GT masks before evaluation/debug export.
- Added a dedicated single-process multi-GPU smoke worker at [tests/gemma_trainer_dataparallel_worker.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/gemma_trainer_dataparallel_worker.py).
- Added an opt-in 2-GPU `DataParallel` smoke test at [tests/test_gemma_trainer_dataparallel_integration.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/test_gemma_trainer_dataparallel_integration.py).
- Hardened the existing DDP smoke launcher in [tests/test_gemma_trainer_ddp_integration.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/test_gemma_trainer_ddp_integration.py) so it launches through the active Python environment instead of depending on the container-default `torchrun` interpreter.
- Registered the `dataparallel` pytest marker in [pytest.ini](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/pytest.ini).

### Validated

- Focused multi-GPU launch regression suite:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_DDP_TESTS=1 TRAINING_RUN_DATAPARALLEL_TESTS=1 PYTHONPATH=/code/src python -m pytest -q tests/test_gemma_trainer_ddp_integration.py tests/test_gemma_trainer_dataparallel_integration.py -vv`
- Result:
  `2 passed in 75.41s`

- Full Docker suite with module GPU smokes, trainer smoke, DDP smoke, and single-process `DataParallel` smoke enabled:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_GPU_TESTS=1 TRAINING_GEMMA_TEST_ATTN_BACKENDS=eager,sdpa,flash_attention_2 TRAINING_RUN_GPU_TRAINER_TESTS=1 TRAINING_GEMMA_TRAINER_ATTN_BACKENDS=eager TRAINING_RUN_DDP_TESTS=1 TRAINING_RUN_DATAPARALLEL_TESTS=1 PYTHONPATH=/code/src python -m pytest -q tests`
- Result:
  `17 passed, 2 warnings in 187.45s`

### Todo

- Keep `torchrun` / DDP as the recommended production launch path even though the checked-in single-process `DataParallel` smoke now passes.
- Add Molmo as the next first-class VLM family when its dependency/runtime story is agreed.
- Add builder scripts for composition and checkpoint-aware model assembly.
- Add the standalone vision encoder registry plus encoder swap/extraction flows.
- Add `sam2` and `sam3` versioned data/model paths.

## v0.2.3 - 2026-05-18

### Scope

Distributed-training checkpoint for the modular Gemma3 path, focused on validating the repo's intended multi-GPU launch mode and making the `DDP works / DataParallel does not` distinction explicit in tests and docs.

### Done

- Added a dedicated DDP worker script at [tests/gemma_trainer_ddp_worker.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/gemma_trainer_ddp_worker.py).
- Added an opt-in distributed smoke test at [tests/test_gemma_trainer_ddp_integration.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/test_gemma_trainer_ddp_integration.py) that launches:
  `torchrun --standalone --nproc_per_node=2`
- Registered the `ddp` pytest marker in [pytest.ini](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/pytest.ini).
- Documented that the supported multi-GPU smoke path is `torchrun` / DDP, while the remaining failure is the single-process `torch.nn.DataParallel` scatter path.

### Validated

- Dedicated 2-GPU DDP smoke:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_DDP_TESTS=1 PYTHONPATH=/code/src python -m pytest -q tests/test_gemma_trainer_ddp_integration.py -vv`
- Result:
  `1 passed in 40.18s`

- Full Docker suite with module GPU smokes, trainer smoke, and DDP smoke enabled:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_GPU_TESTS=1 TRAINING_GEMMA_TEST_ATTN_BACKENDS=eager,sdpa,flash_attention_2 TRAINING_RUN_GPU_TRAINER_TESTS=1 TRAINING_GEMMA_TRAINER_ATTN_BACKENDS=eager TRAINING_RUN_DDP_TESTS=1 PYTHONPATH=/code/src python -m pytest -q tests`
- Result:
  `16 passed in 157.05s`

### Todo

- Decide whether to normalize scalar batch fields so single-process multi-GPU `DataParallel` stops failing, or keep DDP as the only supported multi-GPU path.
- Add Molmo as the next first-class VLM family when its dependency/runtime story is agreed.
- Add builder scripts for composition and checkpoint-aware model assembly.
- Add the standalone vision encoder registry plus encoder swap/extraction flows.
- Add `sam2` and `sam3` versioned data/model paths.

## v0.2.2 - 2026-05-18

### Scope

Trainer/config checkpoint for the modular Gemma3 path, focused on wiring attention backend selection into the training entrypoint and validating a real one-step `CustomTrainer` GPU smoke, including an online W&B run.

### Done

- Updated [src/training_core/train/train.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/train/train.py) to:
  auto-load `/code/api_keys/wandb_key` when `report_to` includes `wandb`,
  honor optional `wandb_project` and `wandb_entity`,
  read `TRAINING_CONFIG_PATH` for alternate config files,
  and pass `attn_implementation` through to `VLMArguments`.
- Updated [src/configs/run_pramana.yaml](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/configs/run_pramana.yaml) with `attn_implementation`, `wandb_project`, and `wandb_entity` placeholders.
- Added trainer-level GPU smoke coverage in [tests/test_gemma_trainer_gpu_integration.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/test_gemma_trainer_gpu_integration.py).
- Kept the trainer smoke single-GPU scoped to avoid the current multi-GPU `DataParallel` scatter issue with scalar tensors in the collated batch.

### Validated

- Trainer-level GPU smoke without W&B:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_GPU_TRAINER_TESTS=1 TRAINING_GEMMA_TRAINER_ATTN_BACKENDS=eager PYTHONPATH=/code/src python -m pytest -q tests/test_gemma_trainer_gpu_integration.py -vv`
- Result:
  `1 passed in 34.03s`

- Trainer-level GPU smoke with W&B logging:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub WANDB_API_KEY=$(cat /code/api_keys/wandb_key) WANDB_PROJECT=img-2-svg-pretraining-smoke TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_GPU_TRAINER_TESTS=1 TRAINING_LOG_WANDB=1 TRAINING_GEMMA_TRAINER_ATTN_BACKENDS=eager PYTHONPATH=/code/src python -m pytest -q tests/test_gemma_trainer_gpu_integration.py -vv -s`
- Result:
  `1 passed in 38.45s`
- Logged W&B run under project:
  `img-2-svg-pretraining-smoke`

- Full Docker suite with GPU module-smoke matrix plus trainer smoke enabled:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_GPU_TESTS=1 TRAINING_GEMMA_TEST_ATTN_BACKENDS=eager,sdpa,flash_attention_2 TRAINING_RUN_GPU_TRAINER_TESTS=1 TRAINING_GEMMA_TRAINER_ATTN_BACKENDS=eager PYTHONPATH=/code/src python -m pytest -q tests`
- Result:
  `15 passed in 120.19s`

### Todo

- Decide whether to harden the trainer path for true multi-GPU smoke coverage or keep the trainer smoke explicitly single-GPU.
- Add Molmo as the next first-class VLM family when its dependency/runtime story is agreed.
- Add builder scripts for composition and checkpoint-aware model assembly.
- Add the standalone vision encoder registry plus encoder swap/extraction flows.
- Add `sam2` and `sam3` versioned data/model paths.

## v0.2.1 - 2026-05-18

### Scope

Gemma3 hardening checkpoint for the modular VLM + SAM upgrade, focused on narrowing `gemmavlm` to the required Gemma3 path and validating GPU smoke coverage across multiple attention backends.

### Done

- Removed the temporary `PaliGemma`-specific fallback path from [src/training_core/data_modules/vlms/common.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/vlms/common.py), [src/training_core/data_modules/vlms/gemma/gemma_data.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/vlms/gemma/gemma_data.py), and [src/training_core/models/vlms/gemma/gemma_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/gemma/gemma_model.py).
- Narrowed `gemmavlm` to chat-template-capable `Gemma3` checkpoints only.
- Extended [src/training_core/registry/utils.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/registry/utils.py) with `VLMArguments.attn_implementation`.
- Wired configurable attention backend selection through [src/training_core/models/vlms/gemma/gemma_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/gemma/gemma_model.py) and [src/training_core/models/vlms/qwen/qwen_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/qwen/qwen_model.py).
- Added opt-in GPU smoke coverage in [tests/test_gemma_gpu_integration.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/tests/test_gemma_gpu_integration.py) for:
  `eager`, `sdpa`, and `flash_attention_2`
- Registered the `gpu` pytest marker in [pytest.ini](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/pytest.ini).

### Validated

- Standard Docker suite:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 PYTHONPATH=/code/src python -m pytest -q tests`
- Result:
  `8 passed, 2 skipped in 41.91s`

- GPU backend-matrix smoke suite:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_GPU_TESTS=1 TRAINING_GEMMA_TEST_ATTN_BACKENDS=eager,sdpa,flash_attention_2 PYTHONPATH=/code/src python -m pytest -q tests/test_gemma_gpu_integration.py -vv`
- Result:
  `6 passed in 116.58s`

- Full Docker suite with GPU backend matrix enabled:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 TRAINING_RUN_GPU_TESTS=1 TRAINING_GEMMA_TEST_ATTN_BACKENDS=eager,sdpa,flash_attention_2 PYTHONPATH=/code/src python -m pytest -q tests`
- Result:
  `14 passed in 108.57s`

### Todo

- Add Molmo as the next first-class VLM family when its dependency/runtime story is agreed.
- Add builder scripts for composition and checkpoint-aware model assembly.
- Add the standalone vision encoder registry plus encoder swap/extraction flows.
- Add `sam2` and `sam3` versioned data/model paths.
- Expand config examples once there is at least one validated non-Qwen experiment config checked in.

## v0.2.0 - 2026-05-18

### Scope

Phase 1 Gemma checkpoint for the modular VLM + SAM upgrade, added on top of the Phase 1 Qwen + SAM1 baseline.

### Done

- Added shared multimodal batching helpers at [src/training_core/data_modules/vlms/common.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/vlms/common.py) so non-Qwen VLM families can reuse message construction, processor tokenization, and multimodal padding logic.
- Extended [src/training_core/registry/utils.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/registry/utils.py) so a datamodule can report `tokenizer_vocab_size` back to the composite model.
- Generalized [src/training_core/models/vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlm_sam.py) to forward family-specific backbone kwargs and resize backbone token embeddings after tokenizer extension.
- Registered `gemmavlm` as a first-class family in:
  [src/training_core/data_modules/vlms/gemma/gemma_data.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/vlms/gemma/gemma_data.py)
  and [src/training_core/models/vlms/gemma/gemma_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/gemma/gemma_model.py).
- Added a processor fallback for `PaliGemma`-style checkpoints that do not ship a chat template.
- Kept `gemmavlm` on a stable default attention backend so the wrapper works in the Docker CPU integration path as well as on GPU-backed training hosts.
- Updated [src/training_core/train/train.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/train/train.py) imports so the Gemma registry entries are available in normal training runs.

### Validated

- Docker test run on `img-2-svg-pretraining-singlenode-anirudh.srinivasan`:
  `HF_HOME=/tmp/training_hf_tests HUGGINGFACE_HUB_CACHE=/tmp/training_hf_tests/hub TRAINING_RUN_LARGE_MODEL_TESTS=1 PYTHONPATH=/code/src python -m pytest -q tests`
- Result:
  `10 passed in 47.84s`

Covered by tests:

- registry availability now includes `gemmavlm`
- real tiny `Gemma3` datamodule + collator path
- real tiny `PaliGemma` datamodule + collator fallback path
- real tiny `Gemma3` wrapper load
- real tiny `PaliGemma` wrapper load
- real `VLMSam` construction and Gemma3 backbone forward with a SAM1 checkpoint

### Todo

- Add Molmo as the next first-class VLM family.
- Add builder scripts for composition and checkpoint-aware model assembly.
- Add the standalone vision encoder registry plus encoder swap/extraction flows.
- Add `sam2` and `sam3` versioned data/model paths.
- Expand config examples once there is at least one validated non-Qwen experiment config checked in.

## v0.1.0 - 2026-05-18

### Scope

Phase 1 restart of the modular VLM + SAM upgrade, intentionally limited to the existing Qwen + SAM1 path.

### Done

- Added generalized registries in [src/training_core/registry/registry.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/registry/registry.py):
  `VLMModelRegistry`, `SAMDataModuleRegistry`, and `SAMModelRegistry`.
- Added modular interface base classes under:
  [src/training_core/data_modules/vlms/base.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/vlms/base.py),
  [src/training_core/data_modules/sam/base.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/data_modules/sam/base.py),
  [src/training_core/models/vlms/base.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlms/base.py),
  and [src/training_core/models/sam/base.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/sam/base.py).
- Extended config/dataclass support for `vlm_family`, `sam_version`, and versioned `SamModelArguments` in [src/training_core/registry/utils.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/registry/utils.py).
- Registered the existing Qwen datamodule under both `qwenvl` and `qwenvlm`.
- Split the current SAM path into a versioned `sam1` datamodule/model registration pair.
- Added a generalized composite model at [src/training_core/models/vlm_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/vlm_sam.py).
- Kept legacy imports working through compatibility shims at:
  [src/training_core/models/qwen_sam.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/qwen_sam.py),
  [src/training_core/models/qwen/qwen_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/qwen/qwen_model.py),
  and [src/training_core/models/sam/sam_model.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/models/sam/sam_model.py).
- Updated [src/training_core/train/train.py](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/training_core/train/train.py) and [src/configs/run_pramana.yaml](/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/configs/run_pramana.yaml) to use `vlm_family`, `sam_version`, and `sam_checkpoint`.

### Validated

- Docker test run on `img-2-svg-pretraining-singlenode-anirudh.srinivasan`:
  `TRAINING_RUN_LARGE_MODEL_TESTS=1 PYTHONPATH=/code/src python -m pytest -q tests`
- Result:
  `5 passed in 52.08s`

Covered by tests:

- registry availability and modular interface smoke checks
- `VLMSam` eval return-order contract
- real Qwen processor/datamodule/collator integration
- real cached Qwen model-wrapper load
- real `VLMSam` construction using cached Qwen weights plus a SAM1 checkpoint

### Todo

- Add Gemma as a first-class VLM family with its own datamodule and wrapper.
- Add Molmo as a first-class VLM family with its own datamodule and wrapper.
- Add builder scripts for composition and checkpoint-aware model assembly.
- Add standalone vision encoder registry entries and encoder swap logic.
- Add extraction/reload support for vision encoders.
- Add `sam2` and `sam3` versioned data/model paths.
- Expand README once Phase 2 lands so the modular architecture section covers more than the Qwen compatibility path.
