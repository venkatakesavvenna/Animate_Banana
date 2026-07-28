#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodelist=ip-10-20-216-9,ip-10-20-218-142,ip-10-20-238-24
#SBATCH --job-name=layout_training_multinode
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=/fsxvision/venkat.kesav/backup/DocGrounding/src/logs/slurm_%j.out
#SBATCH --error=/fsxvision/venkat.kesav/backup/DocGrounding/src/logs/slurm_%j.err
#SBATCH --mail-user=venkat.kesav@tihiitb.org
#SBATCH --mail-type=ALL
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1

set -euo pipefail

IMAGE_NAME="docgrounding-latest"
CONTAINER_NAME="docgrounding-latest-container"
CODE_MOUNT="/fsxvision/venkat.kesav/backup/DocGrounding"
DATA_MOUNT="/fsxvision/raghuveer.r"
HF_CACHE="/fsxvision/venkat.kesav/backup/hf_cache"
ENVIRONMENT_MOUNT="/fsxvision/venkat.kesav/backup/Environments"
DOGR_DATA_MOUNT="/fsxvision/venkat.kesav/backup"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -lc \
  "hostname -I | tr ' ' '\n' | grep '^10\.' | head -n 1")

head_node_port=$((10000 + SLURM_JOBID%10000))

# Build image ONLY on head node
IMAGE_TAR="/fsxvision/venkat.kesav/backup/docker_images/${IMAGE_NAME}.tar"
mkdir -p /fsxvision/venkat.kesav/backup/docker_images

srun --nodes=1 --ntasks=1 -w ${head_node} bash -lc "
  set -e

  if docker image inspect ${IMAGE_NAME} >/dev/null 2>&1; then
    echo '[head] Image ${IMAGE_NAME} already exists'
  else
    echo '[head] Building image ${IMAGE_NAME}'
    docker build \
      -t ${IMAGE_NAME} \
      -f /fsxvision/venkat.kesav/backup/DocGrounding/docker/Dockerfile.docgrounding \
      /fsxvision/venkat.kesav/backup/DocGrounding
  fi

  echo '[head] Saving image to ${IMAGE_TAR}'
  docker save ${IMAGE_NAME} > ${IMAGE_TAR}
"

srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} bash -lc "
  set -e
  echo \"[\$(hostname)] Loading image ${IMAGE_NAME}\"
  docker load < ${IMAGE_TAR}
"

export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export WORLD_SIZE=$(($SLURM_NNODES * 8))
export MASTER_ADDR=${head_node_ip}
export MASTER_PORT=${head_node_port}
export FI_PROVIDER=efa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RDMAV_FORK_SAFE=1 

SECRETS_FILE="/fsxvision/venkat.kesav/secrets/wandb.env"
CNTR_NAME="layout_multi_node_container"

echo "[host] Running docker on $(hostname)"
echo "[host] MASTER_ADDR=${MASTER_ADDR}"
echo "[host] MASTER_PORT=${MASTER_PORT}"

# Run training
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} bash -lc "
docker run --rm --net=host --shm-size=512g --gpus all \
  --name ${CNTR_NAME} \
  --env-file /fsxvision/venkat.kesav/secrets/wandb.env \
  -e WORLD_SIZE=${WORLD_SIZE} \
  -e RANK=\${SLURM_PROCID} \
  -e LOCAL_RANK=\${SLURM_LOCALID} \
  -e NCCL_DEBUG=INFO \
  -e TRITON_CACHE_DIR=/triton_cache \
  -e MASTER_ADDR=${MASTER_ADDR} \
  -e MASTER_PORT=${MASTER_PORT} \
  -e NCCL_NET=OFI \
  -e NCCL_ASYNC_ERROR_HANDLING=1 \
  -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface \
  -e TMPDIR=/tmp_cache \
  -e LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:\$LD_LIBRARY_PATH \
  -v $CODE_MOUNT:/code \
  -v $DATA_MOUNT:/fsxvision/raghuveer.r \
  -v $HF_CACHE:/root/.cache/huggingface \
  -v $ENVIRONMENT_MOUNT:/environments \
  -v /opt/amazon/efa:/opt/amazon/efa \
  -v /usr/lib/x86_64-linux-gnu/libefa.so.1:/opt/amazon/efa/lib/libefa.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libibverbs.so.1:/opt/amazon/efa/lib/libibverbs.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libibverbs:/usr/lib/x86_64-linux-gnu/libibverbs \
  -v /usr/lib/x86_64-linux-gnu/librdmacm.so.1:/opt/amazon/efa/lib/librdmacm.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libnl-3.so.200:/opt/amazon/efa/lib/libnl-3.so.200 \
  -v /usr/lib/x86_64-linux-gnu/libnl-route-3.so.200:/opt/amazon/efa/lib/libnl-route-3.so.200 \
  -v /usr/lib/x86_64-linux-gnu/libefa.so.1.4.60.0:/usr/lib/x86_64-linux-gnu/libefa.so.1.4.60.0 \
  -v /usr/lib/x86_64-linux-gnu/libhwloc.so.15:/usr/lib/x86_64-linux-gnu/libhwloc.so.15 \
  -v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl \
  --device /dev/infiniband \
  -w /code \
  $IMAGE_NAME bash -lc \"\$(cat << 'EOF'
set -euo pipefail

echo \"[inside] Hostname: \$(hostname)\"
echo \"[inside] RANK=\${RANK} LOCAL_RANK=\${LOCAL_RANK}\"
echo \"[inside] MASTER_ADDR=\${MASTER_ADDR} MASTER_PORT=\${MASTER_PORT}\"

# Activate environment (same as single node)
source /environments/doc_grounding_cluster/bin/activate

# Original env vars
export HF_HUB_OFFLINE=False
export WANDB_PROJECT=\"Pre-Training-Run\"

cd /code/src/img_2_svg_pretraining/training
export PYTHONPATH=/code/src

torchrun \
  --nnodes ${SLURM_NNODES} \
  --node_rank \${RANK} \
  --nproc_per_node 8 \
  --master_addr \${MASTER_ADDR} \
  --master_port \${MASTER_PORT} \
  -m img_2_svg_pretraining.training.training_core.train.train \
  2>&1 | tee /code/src/img_2_svg_pretraining/training/logs/train_multinode_rank_v2\${RANK}.log
EOF
)\"
"
