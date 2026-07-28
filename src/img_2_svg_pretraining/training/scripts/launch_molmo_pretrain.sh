#!/bin/bash
# Launch Molmo-7B-O pre-training replication on GPUs 4,5,6,7.
#
# Usage (from repo root, inside or outside container):
#   NPROC_PER_NODE=4 bash scripts/launch_molmo_pretrain.sh
#
# To resume a full run from a checkpoint:
#   RESUME_FROM=/code/src/img_2_svg_pretraining/training/outputs/molmo_pt/checkpoints/checkpoint-XXXX \
#   NPROC_PER_NODE=4 bash scripts/launch_molmo_pretrain.sh

set -e

source /environments/training_core/bin/activate

# W&B key
if [ -f "/code/api_keys/wandb_key" ]; then
    export WANDB_API_KEY=$(cat /code/api_keys/wandb_key)
fi

export HF_HUB_OFFLINE=False
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=4,5,6,7

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

export TRAINING_CONFIG_PATH=${TRAINING_CONFIG_PATH:-"/code/src/img_2_svg_pretraining/training/configs/molmo_pretrain.yaml"}

MANIFEST_CACHE="/fsxvision_new/anirudh.srinivasan/DATASETS/pixmo_cap_manifest"
DATA_PATH="/fsxvision_new/pratyush.jena/Datasets/pixmo-cap-images/extracted_dataset"

cd /code/src/img_2_svg_pretraining/training
export PYTHONPATH=/code/src

# ── Preflight: build manifest if not cached ───────────────────────────────────
# Must run single-process before torchrun to avoid all 4 ranks racing to build.
if [ ! -f "${MANIFEST_CACHE}/dataset_info.json" ]; then
    echo "================================================================"
    echo "  [Preflight] Manifest not found — building from shards."
    echo "  This runs once and takes ~40 min. Subsequent starts are <2s."
    echo "================================================================"
    PYTHONPATH=/code/src python3 -c "
import sys, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from img_2_svg_pretraining.training.training_core.datasets.pixmo.pixmo_cap_local import build_manifest
ds = build_manifest('${DATA_PATH}', '${MANIFEST_CACHE}')
print(f'Manifest ready: {len(ds)} rows')
"
else
    echo "[Preflight] Manifest already cached at ${MANIFEST_CACHE} — skipping scan."
fi

# ── Launch ───────────────────────────────────────────────────────────────────
LOG_FILE="/code/src/img_2_svg_pretraining/training/logs/molmo_pt_$(date +"%Y%m%d_%H%M%S").log"
mkdir -p /code/src/img_2_svg_pretraining/training/logs

echo "================================================================"
echo "  Molmo-7B-O Pre-Training Replication"
echo "  Config  : ${TRAINING_CONFIG_PATH}"
echo "  GPUs    : ${CUDA_VISIBLE_DEVICES}  (${NPROC_PER_NODE} processes)"
echo "  Log     : ${LOG_FILE}"
echo "================================================================"

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m img_2_svg_pretraining.training.training_core.train.train \
    2>&1 | tee "${LOG_FILE}"
