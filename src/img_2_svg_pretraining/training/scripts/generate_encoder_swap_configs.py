#!/usr/bin/env python3
"""Generate encoder-swap training configs and SLURM launch scripts.

Produces configs in src/configs/encoder_swap/ and scripts in scripts/encoder_swap/.

Naming convention:
  PT raw LM:  mn_{fam}_{size}_ve{enc}_raw_pt.yaml
  PT from VLM: mn_{fam}_{size}_ve{enc}_vlm_pt.yaml
  FT:         mn_{fam}_{size}_ve{enc}_ft.yaml
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "src" / "configs" / "encoder_swap"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "encoder_swap"
HF_CACHE = "/fsxvision_new/anirudh.srinivasan/hf_cache/hub"

DATA_PATH = "/fsxvision_new/pratyush.jena/Datasets/pixmo-cap-images/extracted_dataset"
MANIFEST_CACHE = "/fsxvision_new/anirudh.srinivasan/DATASETS/pixmo_cap_manifest"


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass
class EncoderSpec:
    slug: str
    registry_key: str
    hf_id: str
    local_snapshot: str
    arch: str | None = None  # for openvision

    @property
    def checkpoint_path(self) -> str:
        return self.local_snapshot if self.local_snapshot else self.hf_id


ENCODER_SPECS: list[EncoderSpec] = [
    EncoderSpec(
        slug="clip",
        registry_key="clip",
        hf_id="openai/clip-vit-large-patch14-336",
        local_snapshot=f"{HF_CACHE}/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
    ),
    EncoderSpec(
        slug="siglip",
        registry_key="siglip",
        hf_id="google/siglip-so400m-patch14-384",
        local_snapshot=f"{HF_CACHE}/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    ),
    EncoderSpec(
        slug="siglip2",
        registry_key="siglip2",
        hf_id="google/siglip2-so400m-patch14-384",
        local_snapshot=f"{HF_CACHE}/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    ),
    EncoderSpec(
        slug="metaclip",
        registry_key="metaclip",
        hf_id="facebook/metaclip-l14-fullcc2.5b",
        local_snapshot=f"{HF_CACHE}/models--facebook--metaclip-l14-fullcc2.5b/snapshots/d1d23eb55ad314a6091ac3351fa021508e4c677a",
    ),
    EncoderSpec(
        slug="metaclip2",
        registry_key="metaclip2",
        hf_id="facebook/metaclip-2-worldwide-huge-quickgelu",
        local_snapshot=f"{HF_CACHE}/models--facebook--metaclip-2-worldwide-huge-quickgelu/snapshots/c139061af7b10fdb2e754b60d2b1182a3d5526c2",
    ),
    EncoderSpec(
        slug="openvision",
        registry_key="openvision",
        hf_id="UCSC-VLAA/openvision-vit-large-patch14-224",
        local_snapshot=f"{HF_CACHE}/models--UCSC-VLAA--openvision-vit-large-patch14-224/snapshots/8586d8213024e2946cf1c7db6071d430a286cb5b",
        arch="ViT-L-14",
    ),
]


@dataclass
class VLMSpec:
    slug: str           # used in config name
    vlm_family: str
    model_family_name: str
    base_model: str     # local path or HF ID
    attn_impl: str | None
    deepspeed: str
    grad_ckpt: bool
    nodes: int          # 2 or 4
    pdb: int            # per_device_batch_size
    ga: int             # gradient_accumulation_steps
    # LM raw checkpoint for pretrain_init
    lm_raw_checkpoint: str
    # param group patterns — NATIVE encoder
    pg_native_enc_blocks: str
    pg_native_enc_all: str
    pg_native_rest: str
    # param group patterns — SWAPPED encoder
    pg_swap_enc_blocks: str
    pg_swap_connector: str
    pg_swap_rest: str
    # extra top-level fields (e.g. molmo_max_crops)
    extra_fields: dict = field(default_factory=dict)
    is_72b: bool = False


VLM_SPECS: list[VLMSpec] = [
    VLMSpec(
        slug="molmo_7b",
        vlm_family="molmovlm",
        model_family_name="molmo",
        base_model=f"{HF_CACHE}/models--allenai--Molmo-7B-O-0924/snapshots/7a8c4bf80c839c243a6908c6ebbb0f1ee576d7ca",
        attn_impl="null",
        deepspeed="zero3.json",
        grad_ckpt=True,
        nodes=2,
        pdb=2,
        ga=1,
        lm_raw_checkpoint="allenai/OLMo-7B-1024-preview",
        pg_native_enc_blocks="backbone\\.molmo\\.model\\.vision_backbone\\.image_vit\\.",
        pg_native_enc_all="backbone\\.molmo\\.model\\.vision_backbone\\.",
        pg_native_rest="backbone\\.molmo\\.model\\.",
        pg_swap_enc_blocks="backbone\\.molmo\\.model\\.vision_backbone\\.encoder\\.",
        pg_swap_connector="backbone\\.molmo\\.model\\.vision_backbone\\.",
        pg_swap_rest="backbone\\.molmo\\.",
        extra_fields={"molmo_max_crops": 6},
    ),
    VLMSpec(
        slug="qwen25vl_3b",
        vlm_family="qwenvlm",
        model_family_name="qwen2.5vl",
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        attn_impl="flash_attention_2",
        deepspeed="zero3.json",
        grad_ckpt=True,
        nodes=2,
        pdb=2,
        ga=1,
        lm_raw_checkpoint="Qwen/Qwen2.5-3B-Instruct",
        pg_native_enc_blocks="backbone\\.qwen\\.visual\\.blocks\\.",
        pg_native_enc_all="backbone\\.qwen\\.visual\\.",
        pg_native_rest="backbone\\.qwen\\.",
        pg_swap_enc_blocks="backbone\\.qwen\\.visual\\.encoder\\.",
        pg_swap_connector="backbone\\.qwen\\.visual\\.",
        pg_swap_rest="backbone\\.qwen\\.",
    ),
    VLMSpec(
        slug="qwen25vl_7b",
        vlm_family="qwenvlm",
        model_family_name="qwen2.5vl",
        base_model=f"{HF_CACHE}/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        attn_impl="flash_attention_2",
        deepspeed="zero3.json",
        grad_ckpt=True,
        nodes=2,
        pdb=2,
        ga=1,
        lm_raw_checkpoint="Qwen/Qwen2.5-7B-Instruct",
        pg_native_enc_blocks="backbone\\.qwen\\.visual\\.blocks\\.",
        pg_native_enc_all="backbone\\.qwen\\.visual\\.",
        pg_native_rest="backbone\\.qwen\\.",
        pg_swap_enc_blocks="backbone\\.qwen\\.visual\\.encoder\\.",
        pg_swap_connector="backbone\\.qwen\\.visual\\.",
        pg_swap_rest="backbone\\.qwen\\.",
    ),
    VLMSpec(
        slug="qwen25vl_72b",
        vlm_family="qwenvlm",
        model_family_name="qwen2.5vl",
        base_model=f"{HF_CACHE}/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/89c86200743eec961a297729e7990e8f2ddbc4c5",
        attn_impl="flash_attention_2",
        deepspeed="zero3.json",
        grad_ckpt=True,
        nodes=4,
        pdb=1,
        ga=1,
        lm_raw_checkpoint="Qwen/Qwen2.5-72B-Instruct",
        pg_native_enc_blocks="backbone\\.qwen\\.visual\\.blocks\\.",
        pg_native_enc_all="backbone\\.qwen\\.visual\\.",
        pg_native_rest="backbone\\.qwen\\.",
        pg_swap_enc_blocks="backbone\\.qwen\\.visual\\.encoder\\.",
        pg_swap_connector="backbone\\.qwen\\.visual\\.",
        pg_swap_rest="backbone\\.qwen\\.",
        is_72b=True,
    ),
    VLMSpec(
        slug="gemma3_4b",
        vlm_family="gemmavlm",
        model_family_name="gemma3",
        base_model=f"{HF_CACHE}/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
        attn_impl="eager",
        deepspeed="zero3_gemma.json",
        grad_ckpt=True,
        nodes=2,
        pdb=2,
        ga=1,
        # Gemma has no separate text-only checkpoint; raw_llm loads from the same VLM checkpoint
        lm_raw_checkpoint="google/gemma-3-4b-it",
        pg_native_enc_blocks="backbone\\.gemma\\.vision_tower\\.vision_model\\.encoder\\.",
        pg_native_enc_all="backbone\\.gemma\\.vision_tower\\.",
        pg_native_rest="backbone\\.gemma\\.",
        pg_swap_enc_blocks="backbone\\.gemma\\.vision_tower\\.encoder\\.",
        pg_swap_connector="backbone\\.gemma\\.vision_tower\\.",
        pg_swap_rest="backbone\\.gemma\\.",
    ),
    VLMSpec(
        slug="gemma3_12b",
        vlm_family="gemmavlm",
        model_family_name="gemma3",
        base_model="google/gemma-3-12b-it",
        attn_impl="eager",
        deepspeed="zero3_gemma.json",
        grad_ckpt=True,
        nodes=2,
        pdb=1,
        ga=2,
        lm_raw_checkpoint="google/gemma-3-12b-it",
        pg_native_enc_blocks="backbone\\.gemma\\.vision_tower\\.vision_model\\.encoder\\.",
        pg_native_enc_all="backbone\\.gemma\\.vision_tower\\.",
        pg_native_rest="backbone\\.gemma\\.",
        pg_swap_enc_blocks="backbone\\.gemma\\.vision_tower\\.encoder\\.",
        pg_swap_connector="backbone\\.gemma\\.vision_tower\\.",
        pg_swap_rest="backbone\\.gemma\\.",
    ),
    VLMSpec(
        slug="gemma3_27b",
        vlm_family="gemmavlm",
        model_family_name="gemma3",
        base_model="google/gemma-3-27b-it",
        attn_impl="eager",
        deepspeed="zero3_gemma.json",
        grad_ckpt=True,
        nodes=2,
        pdb=1,
        ga=2,
        lm_raw_checkpoint="google/gemma-3-27b-it",
        pg_native_enc_blocks="backbone\\.gemma\\.vision_tower\\.vision_model\\.encoder\\.",
        pg_native_enc_all="backbone\\.gemma\\.vision_tower\\.",
        pg_native_rest="backbone\\.gemma\\.",
        pg_swap_enc_blocks="backbone\\.gemma\\.vision_tower\\.encoder\\.",
        pg_swap_connector="backbone\\.gemma\\.vision_tower\\.",
        pg_swap_rest="backbone\\.gemma\\.",
    ),
]


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

def _datasets_block() -> str:
    return textwrap.dedent(f"""\
        datasets:
          train:
            - name: pixmo_cap_local
              dataset_kwargs:
                data_path: {DATA_PATH}
                mode: captions
          val:
            - name: pixmo_cap_local
              dataset_kwargs:
                data_path: {DATA_PATH}
                mode: captions

        captioning_eval:
          eval_steps: 500
          num_samples: 64
          max_new_tokens: 256
          do_sample: false
        """)


def _trainer_block_pt(vlm: VLMSpec) -> str:
    ds = f"/code/src/img_2_svg_pretraining/training/configs/deepspeed/{vlm.deepspeed}"
    gc = "true" if vlm.grad_ckpt else "false"
    return textwrap.dedent(f"""\
        trainer:
          deepspeed: {ds}

          num_train_epochs: 1

          per_device_train_batch_size: {vlm.pdb}
          per_device_eval_batch_size: {vlm.pdb}
          gradient_accumulation_steps: {vlm.ga}

          eval_strategy: "steps"
          eval_steps: 500
          save_strategy: "steps"
          save_steps: 1000
          save_total_limit: 3

          learning_rate: 2.0e-5
          lr_scheduler_type: "cosine_with_min_lr"
          lr_scheduler_kwargs:
            min_lr_rate: 0.1
          warmup_steps: 200
          weight_decay: 0.0
          max_grad_norm: 1.0

          optim: "adamw_torch"
          adam_beta1: 0.9
          adam_beta2: 0.95
          adam_epsilon: 1.0e-6

          bf16: true
          gradient_checkpointing: {gc}

          logging_steps: 10
          dataloader_num_workers: 4
          report_to: "wandb"
          run_name: ${{run_name}}
        """)


def _trainer_block_ft(vlm: VLMSpec) -> str:
    ds = f"/code/src/img_2_svg_pretraining/training/configs/deepspeed/{vlm.deepspeed}"
    gc = "true" if vlm.grad_ckpt else "false"
    return textwrap.dedent(f"""\
        trainer:
          deepspeed: {ds}

          num_train_epochs: 1

          per_device_train_batch_size: {vlm.pdb}
          per_device_eval_batch_size: {vlm.pdb}
          gradient_accumulation_steps: {vlm.ga}

          eval_strategy: "steps"
          eval_steps: 500
          save_strategy: "steps"
          save_steps: 1000
          save_total_limit: 3

          learning_rate: 2.0e-5
          lr_scheduler_type: "cosine_with_min_lr"
          lr_scheduler_kwargs:
            min_lr_rate: 0.1
          warmup_steps: 200
          weight_decay: 0.0
          max_grad_norm: 1.0

          optim: "adamw_torch"
          adam_beta1: 0.9
          adam_beta2: 0.95
          adam_epsilon: 1.0e-6

          bf16: true
          gradient_checkpointing: {gc}

          logging_steps: 10
          dataloader_num_workers: 4
          report_to: "wandb"
          run_name: ${{run_name}}
        """)


def _yq(pattern: str) -> str:
    """Escape a regex pattern for use inside a YAML double-quoted string.
    Backslashes must be doubled: regex \. → YAML \\. in double-quoted context.
    """
    return pattern.replace("\\", "\\\\")


def _param_groups_pt_swap(vlm: VLMSpec) -> str:
    return textwrap.dedent(f"""\
        param_groups:
          - pattern: "{_yq(vlm.pg_swap_enc_blocks)}"
            lr: 6.0e-6
            weight_decay: 0.0
          - pattern: "{_yq(vlm.pg_swap_connector)}"
            lr: 2.0e-4
            weight_decay: 0.0
          - pattern: "{_yq(vlm.pg_swap_rest)}"
            lr: 2.0e-5
            weight_decay: 0.0
        """)


def _param_groups_ft_swap(vlm: VLMSpec) -> str:
    return textwrap.dedent(f"""\
        param_groups:
          - pattern: "{_yq(vlm.pg_swap_enc_blocks)}"
            lr: 5.0e-6
            weight_decay: 0.0
          - pattern: "{_yq(vlm.pg_swap_connector)}"
            lr: 2.0e-5
            weight_decay: 0.0
          - pattern: "{_yq(vlm.pg_swap_rest)}"
            lr: 2.0e-5
            weight_decay: 0.0
        """)


def _header_fields(vlm: VLMSpec, run_name: str) -> str:
    attn = vlm.attn_impl if vlm.attn_impl != "null" else "null"
    lines = [
        f"vlm_family: {vlm.vlm_family}",
        f"base_model: {vlm.base_model}",
        f"model_family_name: {vlm.model_family_name}",
        f"attn_implementation: {attn}",
    ]
    for k, v in vlm.extra_fields.items():
        lines.append(f"{k}: {v}")
    lines += [
        "",
        "sam_version: none",
        "out_dim: 256",
        "",
        f'run_name: "{run_name}"',
        "split_ratio: 0.995",
        "seed: 42",
        "",
        "wandb_project: img-2-svg-pretraining",
        f"logging_dir: /code/src/img_2_svg_pretraining/training/outputs/{run_name}/logs",
        f"output_dir: /code/src/img_2_svg_pretraining/training/outputs/{run_name}/checkpoints",
        "",
        "checkpoint:",
        "  resume: false",
        "  path: null",
        "  mode: weights_only",
        "",
    ]
    return "\n".join(lines)


def _ve_fields(enc: EncoderSpec) -> str:
    lines = [
        f"vision_encoder: {enc.registry_key}",
        f"vision_encoder_checkpoint: {enc.checkpoint_path}",
    ]
    if enc.arch:
        lines.append(f"vision_encoder_arch: {enc.arch}")
    lines.append("")
    return "\n".join(lines)


def gen_pt_raw_config(vlm: VLMSpec, enc: EncoderSpec) -> str:
    run_name = f"mn_{vlm.slug}_ve{enc.slug}_raw_pt"
    comment = (
        f"# {vlm.slug.replace('_', ' ').title()} | {enc.hf_id} (swap) | raw LM | Pre-Train\n"
        f"#\n"
        f"#   vision encoder  ← {enc.hf_id} (standalone swap)\n"
        f"#   connector       ← random N(0, 0.02)\n"
        f"#   text model      ← {vlm.lm_raw_checkpoint} (raw_llm)\n"
        f"#\n"
        f"# Nodes: {vlm.nodes}\n"
    )
    pretrain_init = textwrap.dedent(f"""\
        pretrain_init:
          ve_init_mode: none
          reset_connector: true
          lm_init_mode: raw_llm
          lm_checkpoint: {vlm.lm_raw_checkpoint}
        """)
    return (
        comment + "\n"
        + _header_fields(vlm, run_name) + "\n"
        + _ve_fields(enc) + "\n"
        + _trainer_block_pt(vlm) + "\n"
        + "loss:\n  ce_loss_weight: 1.0\n\n"
        + pretrain_init + "\n"
        + _param_groups_pt_swap(vlm) + "\n"
        + _datasets_block()
    )


def gen_pt_vlm_config(vlm: VLMSpec, enc: EncoderSpec) -> str:
    run_name = f"mn_{vlm.slug}_ve{enc.slug}_vlm_pt"
    comment = (
        f"# {vlm.slug.replace('_', ' ').title()} | {enc.hf_id} (swap) | VLM LM | Pre-Train\n"
        f"#\n"
        f"#   vision encoder  ← {enc.hf_id} (standalone swap)\n"
        f"#   connector       ← random N(0, 0.02)\n"
        f"#   text model      ← from base_model VLM (from_vlm)\n"
        f"#\n"
        f"# Nodes: {vlm.nodes}\n"
    )
    pretrain_init = textwrap.dedent("""\
        pretrain_init:
          ve_init_mode: none
          reset_connector: true
          lm_init_mode: from_vlm
        """)
    return (
        comment + "\n"
        + _header_fields(vlm, run_name) + "\n"
        + _ve_fields(enc) + "\n"
        + _trainer_block_pt(vlm) + "\n"
        + "loss:\n  ce_loss_weight: 1.0\n\n"
        + pretrain_init + "\n"
        + _param_groups_pt_swap(vlm) + "\n"
        + _datasets_block()
    )


def gen_ft_config(vlm: VLMSpec, enc: EncoderSpec) -> str:
    run_name = f"mn_{vlm.slug}_ve{enc.slug}_ft"
    comment = (
        f"# {vlm.slug.replace('_', ' ').title()} | {enc.hf_id} (swap) | Fine-Tune\n"
        f"#\n"
        f"# End-to-end fine-tuning with {enc.hf_id} as vision encoder.\n"
        f"#\n"
        f"# Nodes: {vlm.nodes}\n"
    )
    return (
        comment + "\n"
        + _header_fields(vlm, run_name) + "\n"
        + _ve_fields(enc) + "\n"
        + _trainer_block_ft(vlm) + "\n"
        + "loss:\n  ce_loss_weight: 1.0\n\n"
        + _param_groups_ft_swap(vlm) + "\n"
        + _datasets_block()
    )


# ---------------------------------------------------------------------------
# Script generator
# ---------------------------------------------------------------------------

SCRIPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --partition=dev
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --exclusive
#SBATCH --exclude=ip-10-20-218-187
#SBATCH --output=/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/img_2_svg_pretraining/training/logs/slurm_{job_name}_%j.out
#SBATCH --error=/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/img_2_svg_pretraining/training/logs/slurm_{job_name}_%j.err
#SBATCH --mail-user=venkatakesavvenna@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

REPO_ROOT="/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/img_2_svg_pretraining/training"
TRAINING_CONFIG="${{TRAINING_CONFIG:-/code/src/img_2_svg_pretraining/training/configs/encoder_swap/{config_name}.yaml}}"

MANIFEST_CACHE="{manifest_cache}"
DATA_PATH="{data_path}"

NNODES="${{SLURM_JOB_NUM_NODES}}"
nodes=( $(scontrol show hostnames "${{SLURM_JOB_NODELIST}}") )
head_node="${{nodes[0]}}"

head_node_ip="$(
    srun -N1 -n1 -w "${{head_node}}" bash -lc "hostname -I | awk '{{for (i=1; i<=NF; i++) if (\\$i ~ /^10\\./) {{print \\$i; exit}}}}'"
)"
head_node_port=$((10000 + SLURM_JOB_ID % 50000))

USER_NAME="$(whoami)"
CONTAINER_NAME="img-2-svg-pretraining-multinode-${{USER_NAME}}"

GPUS_PER_NODE="$(nvidia-smi --list-gpus 2>/dev/null | wc -l || true)"
[[ -z "${{GPUS_PER_NODE}}" || "${{GPUS_PER_NODE}}" -le 0 ]] && GPUS_PER_NODE=8

mkdir -p "${{REPO_ROOT}}/logs"

export MASTER_ADDR="${{head_node_ip}}"
export MASTER_PORT="${{head_node_port}}"

echo "======================================================"
echo "SLURM_JOB_ID        : ${{SLURM_JOB_ID}}"
echo "SLURM_JOB_NODELIST  : ${{SLURM_JOB_NODELIST}}"
echo "NNODES              : ${{NNODES}}"
echo "MASTER_ADDR         : ${{MASTER_ADDR}}"
echo "TRAINING_CONFIG       : ${{TRAINING_CONFIG}}"
echo "======================================================"

srun -N1 -n1 -w "${{head_node}}" bash -lc "
set -euo pipefail
cd '${{REPO_ROOT}}/docker'
bash init_multinode_docker.sh
if ! docker exec '${{CONTAINER_NAME}}' bash -lc \\"test -f '{manifest_cache}/dataset_info.json'\\"; then
    docker exec -e DATA_PATH='{data_path}' -e MANIFEST_CACHE='{manifest_cache}' '${{CONTAINER_NAME}}' \\
        bash -lc \\"PYTHONPATH=/code/src python3 -c 'import os,sys;sys.path.insert(0,\\\\\\"/code/src\\\\\\");from img_2_svg_pretraining.training.training_core.datasets.pixmo.pixmo_cap_local import build_manifest;build_manifest(os.environ[\\\\\\"DATA_PATH\\\\\\"],os.environ[\\\\\\"MANIFEST_CACHE\\\\\\"])'\\"
fi
"

GIT_COMMIT="$(git -C "${{REPO_ROOT}}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
GIT_BRANCH="$(git -C "${{REPO_ROOT}}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_DIRTY="$(git -C "${{REPO_ROOT}}" status --short 2>/dev/null | wc -l | tr -d ' ')"

srun -N "${{NNODES}}" -n "${{NNODES}}" --ntasks-per-node=1 --label bash -lc '
set -euo pipefail
cd "'"${{REPO_ROOT}}"'/docker"
bash init_multinode_docker.sh
docker exec \\
-e MASTER_ADDR="'"${{MASTER_ADDR}}"'" \\
-e MASTER_PORT="'"${{MASTER_PORT}}"'" \\
-e NNODES="'"${{NNODES}}"'" \\
-e NODE_RANK="${{SLURM_NODEID}}" \\
-e GPUS_PER_NODE="'"${{GPUS_PER_NODE}}"'" \\
-e TRAINING_CONFIG_PATH="'"${{TRAINING_CONFIG}}"'" \\
-e GIT_COMMIT="'"${{GIT_COMMIT}}"'" \\
-e GIT_BRANCH="'"${{GIT_BRANCH}}"'" \\
-e GIT_DIRTY="'"${{GIT_DIRTY}}"'" \\
-e DS_NVTX=0 \\
-e DEEPSPEED_NVTX_ENABLED=0 \\
-e NCCL_NVLS_ENABLE=0 \\
"'"${{CONTAINER_NAME}}"'" \\
bash -lc "
set -euo pipefail
[ -f /code/.env ] && set -a && source /code/.env && set +a
mkdir -p /code/src/img_2_svg_pretraining/training/logs
cd /code/src/img_2_svg_pretraining/training
export PYTHONPATH=/code/src
torchrun \\
  --nproc_per_node=\\${{GPUS_PER_NODE}} \\
  --nnodes=\\${{NNODES}} \\
  --node_rank=\\${{NODE_RANK}} \\
  --master_addr=\\${{MASTER_ADDR}} \\
  --master_port=\\${{MASTER_PORT}} \\
  -m img_2_svg_pretraining.training.training_core.train.train \\
  2>&1 | tee logs/{job_name}_rank_\\${{NODE_RANK}}.log
"
'

echo "Job {job_name} completed."
"""


def gen_script(vlm: VLMSpec, config_name: str) -> str:
    return SCRIPT_TEMPLATE.format(
        job_name=config_name,
        nodes=vlm.nodes,
        config_name=config_name,
        manifest_cache=MANIFEST_CACHE,
        data_path=DATA_PATH,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, str]] = []  # (config_name, vlm_slug)

    for vlm in VLM_SPECS:
        for enc in ENCODER_SPECS:
            # PT raw LM
            name_raw = f"mn_{vlm.slug}_ve{enc.slug}_raw_pt"
            cfg_raw = gen_pt_raw_config(vlm, enc)
            (CONFIGS_DIR / f"{name_raw}.yaml").write_text(cfg_raw)
            (SCRIPTS_DIR / f"{name_raw}.sh").write_text(gen_script(vlm, name_raw))
            (SCRIPTS_DIR / f"{name_raw}.sh").chmod(0o755)
            generated.append((name_raw, vlm.slug))

            # PT from VLM
            name_vlm = f"mn_{vlm.slug}_ve{enc.slug}_vlm_pt"
            cfg_vlm = gen_pt_vlm_config(vlm, enc)
            (CONFIGS_DIR / f"{name_vlm}.yaml").write_text(cfg_vlm)
            (SCRIPTS_DIR / f"{name_vlm}.sh").write_text(gen_script(vlm, name_vlm))
            (SCRIPTS_DIR / f"{name_vlm}.sh").chmod(0o755)
            generated.append((name_vlm, vlm.slug))

            # FT
            name_ft = f"mn_{vlm.slug}_ve{enc.slug}_ft"
            cfg_ft = gen_ft_config(vlm, enc)
            (CONFIGS_DIR / f"{name_ft}.yaml").write_text(cfg_ft)
            (SCRIPTS_DIR / f"{name_ft}.sh").write_text(gen_script(vlm, name_ft))
            (SCRIPTS_DIR / f"{name_ft}.sh").chmod(0o755)
            generated.append((name_ft, vlm.slug))

    print(f"Generated {len(generated)} config+script pairs in:")
    print(f"  {CONFIGS_DIR}")
    print(f"  {SCRIPTS_DIR}")

    # Print summary table
    pt_raw = sum(1 for n, _ in generated if n.endswith("_raw_pt"))
    pt_vlm = sum(1 for n, _ in generated if n.endswith("_vlm_pt"))
    ft = sum(1 for n, _ in generated if n.endswith("_ft"))
    print(f"\n  PT (raw LM):   {pt_raw}")
    print(f"  PT (from VLM): {pt_vlm}")
    print(f"  FT:            {ft}")
    print(f"  Total:         {len(generated)}")

    # Write a manifest for the smoke test launcher
    manifest_path = REPO_ROOT / "scripts" / "encoder_swap_manifest.txt"
    with manifest_path.open("w") as f:
        for name, slug in generated:
            f.write(f"{name}\t{slug}\n")
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
