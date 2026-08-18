#!/bin/bash
set -euo pipefail

# =============================================================================
# Docker Container Initialization Script
# =============================================================================
# This script accepts all configuration as arguments from run_pipeline.sh
# DO NOT hardcode values here - all config should come from run_pipeline.sh

# Accept parameters from run_pipeline.sh
USER_NAME="$1"
IMAGE_NAME="$2"
CONTAINER_NAME="$3"
CODE_MOUNT="$4"
DATAMOUNT1="$5"
DATAMOUNT2="$6"
DATAMOUNT3="$7"
DATAMOUNT4="$8"
HF_CACHE_HOST="$9"
TRITON_CACHE="${10}"
TMP_CACHE="${11}"
VLLM_CACHE="${12}"
ENVIRONMENT_MOUNT="${13}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Docker Container Initialization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "User:       $USER_NAME"
echo "Image:      $IMAGE_NAME"
echo "Container:  $CONTAINER_NAME"
echo "Code Mount: $CODE_MOUNT"
echo "HF Cache:   $HF_CACHE_HOST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ---- Build image if not exists ----
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Building image $IMAGE_NAME..."
  docker build \
    --build-arg UID="$(id -u ${USER_NAME})" \
    --build-arg GID="$(id -g ${USER_NAME})" \
    --build-arg USER_NAME="${USER_NAME}" \
    -t "$IMAGE_NAME" .
else
  echo "Image $IMAGE_NAME already exists."
fi

# ---- Ensure cache dirs exist on host ----
mkdir -p "$HF_CACHE_HOST"/{hub,transformers,xdg,datasets}
mkdir -p "$TRITON_CACHE"
mkdir -p "$TMP_CACHE"
mkdir -p "$VLLM_CACHE"/torch_inductor

# ---- Create or Start container ----
if docker ps -aq --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
  echo "Container $CONTAINER_NAME already exists — starting..."
  docker start "$CONTAINER_NAME" >/dev/null
else
  echo "Creating container $CONTAINER_NAME..."
  docker run -dit --gpus all \
    --network=host \
    --shm-size=64g \
    --ipc=host \
    \
    -v "$DATAMOUNT1":"$DATAMOUNT1" \
    -v "$DATAMOUNT2":"$DATAMOUNT2" \
    -v "$DATAMOUNT3":"$DATAMOUNT3" \
    -v "$DATAMOUNT4":"$DATAMOUNT4" \
    -v "$CODE_MOUNT":/code \
    \
    -v "$HF_CACHE_HOST":/hf_cache \
    \
    -v "$TRITON_CACHE":/triton_cache \
    -v "$TMP_CACHE":/tmp \
    -v "$VLLM_CACHE":/vllm_cache \
    -v "$ENVIRONMENT_MOUNT":/environments \
    \
    -e HF_HOME=/hf_cache \
    -e HF_HUB_CACHE=/hf_cache/hub \
    -e HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    -e XDG_CACHE_HOME=/hf_cache/xdg \
    -e HF_DATASETS_CACHE=/hf_cache/datasets \
    \
    -e TRITON_CACHE_DIR="/triton_cache" \
    -e TMPDIR="/tmp" \
    \
    -e VLLM_CACHE_DIR="/vllm_cache" \
    -e TORCHINDUCTOR_CACHE_DIR="/vllm_cache/torch_inductor" \
    \
    --name "$CONTAINER_NAME" \
    -w /code \
    "$IMAGE_NAME" \
    bash
fi

# ---- SSH Keys copy ----
## this will fail as username is not created in dockerfile.. needs to be fixed
echo "Copying SSH keys into container..."
docker exec "$CONTAINER_NAME" bash -lc "mkdir -p /home/$USER_NAME/.ssh"
docker cp "/home/${USER_NAME}/.ssh" "$CONTAINER_NAME":/home/$USER_NAME/.ssh
docker cp "/home/${USER_NAME}/.ssh/." "$CONTAINER_NAME":/home/$USER_NAME/.ssh/
docker exec "$CONTAINER_NAME" bash -lc "
  chmod 700 /home/$USER_NAME/.ssh &&
  chmod 600 /home/$USER_NAME/.ssh/* || true
"

# ---- Cleanup bad container-layer caches (CRITICAL) ----
docker exec "$CONTAINER_NAME" bash -lc "
  rm -rf /root/.triton /root/.cache/vllm /projects 2>/dev/null || true
"

echo ""
echo "🚀 READY — attach using:"
echo "docker exec -it $CONTAINER_NAME bash"
echo ""
echo "🔍 Verify inside container:"
echo "docker exec -it $CONTAINER_NAME bash -lc 'echo HF_HOME=\$HF_HOME; echo HF_HUB_CACHE=\$HF_HUB_CACHE; echo TRANSFORMERS_CACHE=\$TRANSFORMERS_CACHE; echo TRITON_CACHE_DIR=\$TRITON_CACHE_DIR; echo VLLM_CACHE_DIR=\$VLLM_CACHE_DIR; echo TMPDIR=\$TMPDIR; df -h / /tmp /fsxvision_new /hf_cache | tail -n +1'"
echo ""
