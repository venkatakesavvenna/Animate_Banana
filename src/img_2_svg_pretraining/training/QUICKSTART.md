# Quickstart

Get a training run going in a few minutes. For architecture, the full config reference, and everything not covered here, see **[docs/architecture.md](docs/architecture.md)**.

## Prerequisites

- A SLURM cluster with GPU nodes and Docker on each node (the launch scripts build/reuse a per-user image automatically — no manual `docker build`/`run` needed)
- The repo checked out on a filesystem shared across nodes (e.g. FSx/NFS)
- `api_keys/hf_token` (for gated HF checkpoints) and `api_keys/wandb_key` (optional, for W&B logging)

## 1. Add your API keys

```bash
mkdir -p api_keys
printf '%s' '<your-hf-token>'    > api_keys/hf_token
printf '%s' '<your-wandb-key>'   > api_keys/wandb_key
```

## 2. Pick a config — with a vision-encoder swap

Every run is one YAML config + one matching `.sh` launch script under `configs/encoder_swap/` and `scripts/encoder_swap/`. Here's a real one — Molmo-7B, native vision backbone swapped out for CLIP, LLM initialized from a raw (non-instruction-tuned) OLMo checkpoint:

**[configs/encoder_swap/mn_molmo_7b_veclip_raw_pt.yaml](configs/encoder_swap/mn_molmo_7b_veclip_raw_pt.yaml)**

```yaml
vlm_family: molmovlm
base_model: <path-to-Molmo-7B-O-checkpoint>
run_name: "mn_molmo_7b_veclip_raw_pt"
sam_version: none              # no SAM head — pure captioning/text loss

# --- vision-encoder swap: this is what makes it a "swap" run ---
vision_encoder: clip                                    # registered encoder kind
vision_encoder_checkpoint: <path-to-openai/clip-vit-large-patch14-336>

checkpoint:
  resume: false
  mode: weights_only

pretrain_init:                 # omit this whole block for a plain fine-tune
  ve_init_mode: none
  reset_connector: true        # random-init the vision->LLM projector
  lm_init_mode: raw_llm        # replace the LLM with a checkpoint below...
  lm_checkpoint: allenai/OLMo-7B-1024-preview

trainer:
  deepspeed: /code/src/img_2_svg_pretraining/training/configs/deepspeed/zero3.json
  per_device_train_batch_size: 2
  learning_rate: 2.0e-5
  max_grad_norm: 1.0
  bf16: true
  gradient_checkpointing: true
  report_to: "wandb"

datasets:
  train:
    - name: pixmo_cap_local
      dataset_kwargs: {data_path: <path-to-pixmo-cap-images>, mode: captions}
  val:
    - name: pixmo_cap_local
      dataset_kwargs: {data_path: <path-to-pixmo-cap-images>, mode: captions}

captioning_eval:                # opt-in BLEU-4 + W&B table during eval
  eval_steps: 500
  num_samples: 64
  max_new_tokens: 256
```

The matching launch script is **[scripts/encoder_swap/mn_molmo_7b_veclip_raw_pt.sh](scripts/encoder_swap/mn_molmo_7b_veclip_raw_pt.sh)** — it's a `#SBATCH`-annotated script that builds/reuses the training container on each allocated node and runs `torchrun -m img_2_svg_pretraining.training.training_core.train.train` inside it.

To try a **different** encoder, decoder, or training mode, copy this pair and change `vision_encoder`/`vision_encoder_checkpoint` (any of `clip`, `siglip`, `siglip2`, `metaclip`, `metaclip2`, `openvision`) and `base_model`/`vlm_family` (`molmovlm`, `qwenvlm`, `gemmavlm`). The full parameter reference — every field in every block above — is in [docs/architecture.md § Configuration](docs/architecture.md#configuration).

## 3. Launch

```bash
sbatch --export=ALL,TRAINING_CONFIG=/code/src/img_2_svg_pretraining/training/configs/encoder_swap/mn_molmo_7b_veclip_raw_pt.yaml \
  scripts/encoder_swap/mn_molmo_7b_veclip_raw_pt.sh
```

## 4. Watch it run

```bash
tail -f logs/slurm_mn_molmo_7b_veclip_raw_pt_<job_id>.out   # SLURM stdout
tail -f logs/mn_molmo_7b_veclip_raw_pt_rank_0.log           # training log, rank 0
ls outputs/mn_molmo_7b_veclip_raw_pt/checkpoints/                # checkpoints
```

## Next steps

| Topic | Where |
|---|---|
| Every configurable parameter, grouped by block | [docs/architecture.md § Configuration](docs/architecture.md#configuration) |
| Resuming from a checkpoint, debug passes, multi-node internals | [docs/architecture.md](docs/architecture.md) |
| The full encoder×decoder swap matrix | [docs/architecture.md § Encoder×Decoder Swap Matrix](docs/architecture.md#encoderdecoder-swap-matrix) |
| Training throughput / MFU monitoring, profiling | [docs/architecture.md § Training Monitoring](docs/architecture.md#training-monitoring) |
| Troubleshooting (offline HF cache, DeepSpeed OOM, path errors) | [docs/architecture.md § Troubleshooting](docs/architecture.md#troubleshooting) |
| Adding a dataset / VLM family / vision encoder | [docs/architecture.md § Extending The Repo](docs/architecture.md#extending-the-repo) |
