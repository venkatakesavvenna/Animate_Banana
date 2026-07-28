#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodelist=bharatgpt155,bharatgpt157
#SBATCH --job-name=test_run
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --error=/projects/data/vision-team/venkat_kesav/mn_logs/error_log_DEBUG.txt
#SBATCH --output=/projects/data/vision-team/venkat_kesav/mn_logs/output_log_DEBUG.txt
#SBATCH --mail-user=venkat.kesav@tihiitb.org
#SBATCH --mail-type=ALL

export IMAGE_NAME="docgrounding-latest"
export CONTAINER_NAME="docgrounding-latest-container"
export CODE_MOUNT="/projects/data/vision-team/venkat_kesav/DocGrounding"
export DATA_MOUNT="/projects/data/vision-team/venkat_kesav/DOCGROUNDING-BASE/Grounding_Dataset"
export HF_CACHE="/projects/data/vision-team/venkat_kesav/hf_cache"
export ENVIRONMENT_MOUNT="/projects/data/vision-team/venkat_kesav/DOCGROUNDING-BASE/Environments"
export DOGR_DATA_MOUNT="/projects/data/vision-team/sai_gunda/DATAENGINE/DATASETS"
export WANDB_API_KEY="90d709f2ae703749fbc09f794f7d4a925e3b775f"
export ENTRY_FILE="/code/docgrounding/train/main.py"

# Exporting all the required multi-node env variables
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "hostname -I | tr ' ' '\n' | grep '^10\.' | head -n 1")
head_node_port=$((10000 + SLURM_JOBID%10000)) 

export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export WORLD_SIZE=$(($SLURM_NNODES * 8))
export MASTER_ADDR=${head_node_ip}
export MASTER_PORT=${head_node_port}
export NPROC_PER_NODE=8
export HF_HUB_OFFLINE=1

read -r -d '' SPAWN_CONTAINER_AND_RUN <<-EOM

if docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
    echo "Image $IMAGE_NAME already exists."
else
    echo "Building image $IMAGE_NAME..."
    docker build -t $IMAGE_NAME -f $CODE_MOUNT/docker/Dockerfile $CODE_MOUNT/docker
fi

echo "Just Testing"

if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
    echo "Container $CONTAINER_NAME already exists. Removing it"
    docker rm -f $CONTAINER_NAME
fi

docker run --shm-size=64g -detach -it --gpus all \
  --network=host --ipc=host \
  --name "$CONTAINER_NAME" \
  -v "$CODE_MOUNT":/code \
  -v "$DATA_MOUNT":/data \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$ENVIRONMENT_MOUNT":/environments \
  -e MASTER_ADDR="$MASTER_ADDR" \
  -e MASTER_PORT="$MASTER_PORT" \
  -e WORLD_SIZE="$WORLD_SIZE" \
  -e SLURM_NNODES="$SLURM_NNODES" \
  -e SLURM_NODEID="$SLURM_NODEID" \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}" \
  -w /code \
  "$IMAGE_NAME" \
  bash -c 'echo "[DEBUG container ready] $(hostname)"; sleep infinity'
EOM

export SPAWN_CONTAINER_AND_RUN

srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} sg docker -c 'bash -c "$SPAWN_CONTAINER_AND_RUN"' >&1
