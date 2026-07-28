#!/bin/bash
#SBATCH --partition=dev
#SBATCH --job-name=mn_molmo_7b_vemetaclip2_vlm_pt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --exclusive
#SBATCH --exclude=ip-10-20-218-187
#SBATCH --nodelist=ip-10-20-205-50,ip-10-20-208-160,ip-10-20-213-4,ip-10-20-216-9
#SBATCH --output=/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/img_2_svg_pretraining/training/logs/slurm_mn_molmo_7b_vemetaclip2_vlm_pt_%j.out
#SBATCH --error=/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/img_2_svg_pretraining/training/logs/slurm_mn_molmo_7b_vemetaclip2_vlm_pt_%j.err
#SBATCH --mail-user=venkatakesavvenna@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

REPO_ROOT="/fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/img_2_svg_pretraining/training"
TRAINING_CONFIG="${TRAINING_CONFIG:-/code/src/img_2_svg_pretraining/training/configs/encoder_swap/mn_molmo_7b_vemetaclip2_vlm_pt.yaml}"

MANIFEST_CACHE="/fsxvision_new/anirudh.srinivasan/DATASETS/pixmo_cap_manifest"
DATA_PATH="/fsxvision_new/pratyush.jena/Datasets/pixmo-cap-images/extracted_dataset"

NNODES="${SLURM_JOB_NUM_NODES}"
nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
head_node="${nodes[0]}"

head_node_ip="$(
    srun -N1 -n1 -w "${head_node}" bash -lc "hostname -I | awk '{for (i=1; i<=NF; i++) if (\$i ~ /^10\./) {print \$i; exit}}'"
)"
head_node_port=$((10000 + SLURM_JOB_ID % 50000))

USER_NAME="$(whoami)"
CONTAINER_NAME="img-2-svg-pretraining-multinode-${USER_NAME}"

GPUS_PER_NODE="$(nvidia-smi --list-gpus 2>/dev/null | wc -l || true)"
[[ -z "${GPUS_PER_NODE}" || "${GPUS_PER_NODE}" -le 0 ]] && GPUS_PER_NODE=8

mkdir -p "${REPO_ROOT}/logs"

export MASTER_ADDR="${head_node_ip}"
export MASTER_PORT="${head_node_port}"

echo "======================================================"
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST  : ${SLURM_JOB_NODELIST}"
echo "NNODES              : ${NNODES}"
echo "MASTER_ADDR         : ${MASTER_ADDR}"
echo "TRAINING_CONFIG       : ${TRAINING_CONFIG}"
echo "======================================================"

srun -N1 -n1 -w "${head_node}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}/docker'
bash init_multinode_docker.sh
if ! docker exec '${CONTAINER_NAME}' bash -lc \"test -f '/fsxvision_new/anirudh.srinivasan/DATASETS/pixmo_cap_manifest/dataset_info.json'\"; then
    docker exec -e DATA_PATH='/fsxvision_new/pratyush.jena/Datasets/pixmo-cap-images/extracted_dataset' -e MANIFEST_CACHE='/fsxvision_new/anirudh.srinivasan/DATASETS/pixmo_cap_manifest' '${CONTAINER_NAME}' \
        bash -lc \"PYTHONPATH=/code/src python3 -c 'import os,sys;sys.path.insert(0,\\\"/code/src\\\");from img_2_svg_pretraining.training.training_core.datasets.pixmo.pixmo_cap_local import build_manifest;build_manifest(os.environ[\\\"DATA_PATH\\\"],os.environ[\\\"MANIFEST_CACHE\\\"])'\"
fi
"

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
GIT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_DIRTY="$(git -C "${REPO_ROOT}" status --short 2>/dev/null | wc -l | tr -d ' ')"

srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --label bash -lc '
set -euo pipefail
cd "'"${REPO_ROOT}"'/docker"
bash init_multinode_docker.sh
docker exec \
-e MASTER_ADDR="'"${MASTER_ADDR}"'" \
-e MASTER_PORT="'"${MASTER_PORT}"'" \
-e NNODES="'"${NNODES}"'" \
-e NODE_RANK="${SLURM_NODEID}" \
-e GPUS_PER_NODE="'"${GPUS_PER_NODE}"'" \
-e TRAINING_CONFIG_PATH="'"${TRAINING_CONFIG}"'" \
-e GIT_COMMIT="'"${GIT_COMMIT}"'" \
-e GIT_BRANCH="'"${GIT_BRANCH}"'" \
-e GIT_DIRTY="'"${GIT_DIRTY}"'" \
-e DS_NVTX=0 \
-e DEEPSPEED_NVTX_ENABLED=0 \
-e NCCL_NVLS_ENABLE=0 \
"'"${CONTAINER_NAME}"'" \
bash -lc "
set -euo pipefail
[ -f /code/.env ] && set -a && source /code/.env && set +a
mkdir -p /code/src/img_2_svg_pretraining/training/logs
cd /code/src/img_2_svg_pretraining/training
export PYTHONPATH=/code/src
torchrun \
  --nproc_per_node=\${GPUS_PER_NODE} \
  --nnodes=\${NNODES} \
  --node_rank=\${NODE_RANK} \
  --master_addr=\${MASTER_ADDR} \
  --master_port=\${MASTER_PORT} \
  -m img_2_svg_pretraining.training.training_core.train.train \
  2>&1 | tee logs/mn_molmo_7b_vemetaclip2_vlm_pt_rank_\${NODE_RANK}.log
"
'

echo "Job mn_molmo_7b_vemetaclip2_vlm_pt completed."
