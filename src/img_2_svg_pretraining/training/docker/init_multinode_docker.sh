#!/bin/bash

# =============================================================================
# init_multinode_docker.sh — Idempotent container initializer for img_2_svg_pretraining multi-node training
#                            multi-node training on p5.48xlarge (H100) with
#                            EFA networking.
#
# Usage:
#   bash init_multinode_docker.sh
#
# After init, exec into the container:
#   docker exec -it $CONTAINER_NAME bash
# =============================================================================

set -euo pipefail

# =============================================================================
# ── USER CONFIGURATION — edit these before running ───────────────────────────
# =============================================================================

USER_NAME=$(whoami)
IMAGE_NAME="img-2-svg-pretraining-multinode-${USER_NAME}:latest"
CONTAINER_NAME="img-2-svg-pretraining-multinode-${USER_NAME}"
DOCKERFILE_PATH="Dockerfile.multinode"

# Workspace mount — becomes /code inside the container. This is the outer
# repo root (img_2_svg_pretraining/), not training/ itself, so that
# `img_2_svg_pretraining.training.training_core...` imports resolve the same
# way here as they do in the single-node container (see repo root's
# docker/init.sh, which mounts the same root).
# Derived from the script's own location so the correct repo is always mounted
# regardless of which user's directory the script lives in.
WORKSPACE_MOUNT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Shared filesystem mount (FSx for Lustre or NFS).
# Mounted at the same path inside the container so absolute paths stay valid.
FSX_MOUNT="/fsxvision_new"
NVME_MOUNT="/opt/dlami/nvme"

# HuggingFace model cache — avoids re-downloading weights on every run.
HF_CACHE="/fsxvision_new/venkat.kesav/backup/hf_cache"

# HuggingFace token — required for gated models (e.g. Gemma family).
HF_TOKEN_FILE="/fsxvision_new/venkat.kesav/img_2_svg_pretraining/api_keys/hf_token"
HF_TOKEN=""
if [[ -f "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HF_TOKEN_FILE}")"
fi

WANDB_KEY_FILE="/fsxvision_new/venkat.kesav/img_2_svg_pretraining/api_keys/wandb_key"
WANDB_API_KEY=""
if [[ -f "${WANDB_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
fi

# =============================================================================
# ── PATHS INSIDE THE DOCKERFILE (do not change unless dockerfile changes) ────
# =============================================================================

OFI_NCCL_LIB="/opt/amazon/ofi-nccl/lib"
EFA_LIB="/opt/amazon/efa/lib"
OPENMPI_BIN="/opt/amazon/openmpi/bin"
EFA_BIN="/opt/amazon/efa/bin"

# =============================================================================
# ── EFA DEVICE DISCOVERY ─────────────────────────────────────────────────────
# p5.48xlarge exposes 32 EFA interfaces (efa0..efa31) as
# /dev/infiniband/uverbs0..uverbs31. We mount the whole directory rather than
# listing all 32 --device flags individually.
# =============================================================================

EFA_DEVICE_DIR="/dev/infiniband"

# =============================================================================
# ── MAIN SCRIPT ──────────────────────────────────────────────────────────────
# =============================================================================

echo ""
echo "=== img_2_svg_pretraining Multi-Node Docker Initialization ==="
echo "    Image         : ${IMAGE_NAME}"
echo "    Container     : ${CONTAINER_NAME}"
echo "====================================================="
echo ""

# ── Verify the image exists ──────────────────────────────────────────────────
if docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
  echo "✓ Image $IMAGE_NAME already exists."
else
  echo "Building image $IMAGE_NAME..."
  docker build --network host -f $DOCKERFILE_PATH -t $IMAGE_NAME .
  echo "✓ Image built successfully."
fi

# ── Create directories on the host that we'll mount ─────────────────────────
mkdir -p "${HF_CACHE}"

# ── Idempotent container lifecycle ──────────────────────────────────────────
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
echo "✓ Previous container removed (if any)."

echo "  Creating new container '${CONTAINER_NAME}'..."

docker run \
    --detach \
    --interactive \
    --tty \
    --name "${CONTAINER_NAME}" \
    \
    --gpus all \
    \
    `# ── Privileged + host namespaces ────────────────────────────────` \
    `# --privileged:   required for EFA verbs device access            ` \
    `# --network host: EFA interfaces are not visible inside bridge net ` \
    `# --ipc host:     NVLink P2P across GPUs uses shared memory        ` \
    `# --pid host:     required for correct CUDA IPC handle sharing     ` \
    --privileged \
    --network host \
    --ipc host \
    --pid host \
    \
    `# ── Memory limits ───────────────────────────────────────────────` \
    `# --shm-size 512g:  NCCL uses /dev/shm for intra-node collectives;` \
    `#                   default 64MB causes silent OOM crashes.        ` \
    `# memlock=-1:       EFA/RDMA requires unlimited pinned memory.     ` \
    `# stack=67108864:   Large stack for MPI/OFI thread safety.         ` \
    --shm-size=512g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    \
    `# ── Device mounts ───────────────────────────────────────────────` \
    `# Mounts all 32 EFA uverbs devices in one shot efa0 through efa31. ` \
    -v "${EFA_DEVICE_DIR}:${EFA_DEVICE_DIR}" \
    \
    `# ── Filesystem mounts ───────────────────────────────────────────` \
    -v "${WORKSPACE_MOUNT}:/code" \
    -v "${FSX_MOUNT}:${FSX_MOUNT}" \
    -v "${NVME_MOUNT}:${NVME_MOUNT}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v /home:/home \
    \
    `# ── EFA / libfabric settings ────────────────────────────────────` \
    `# FI_PROVIDER=efa:         force libfabric to use EFA provider, not fallback TCP.` \
    `# FI_EFA_USE_HUGE_PAGE=0:  avoids ENOMEM in containers where huge pages aren't pre-allocated.` \
    `# FI_EFA_FORK_SAFE=1:      prevents hangs when PyTorch dataloader workers fork inside the container.` \
    `# FI_LOG_LEVEL=warn:       surface EFA warnings without drowning output in debug noise.` \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_HUGE_PAGE=0 \
    -e FI_EFA_FORK_SAFE=1 \
    -e FI_LOG_LEVEL=warn \
    \
    `# ── NCCL settings ───────────────────────────────────────────────` \
    `# NCCL_SOCKET_IFNAME:      exclude virtual/loopback interfaces so NCCL uses the real EFA-backed NIC.` \
    `# NCCL_P2P_NET_CHUNKSIZE:  2MB chunks, tuned for EFA on p5.        ` \
    `# NCCL_BUFFSIZE:           8MB NCCL comm buffer.                    ` \
    `# NCCL_TUNER_PLUGIN:       lets NCCL auto-select the best algorithm per collective Ring/Tree/etc.` \
    `#                          Without this NCCL may pick Tree which hangs on EFA.` \
    `# NCCL_ASYNC_ERROR_HANDLING: surface NCCL errors immediately rather than hanging indefinitely.` \
    `# NCCL_DEBUG=WARN:         only log warnings/errors, not verbose INFO which floods logs.` \
    -e NCCL_SOCKET_IFNAME=^docker,lo,veth \
    -e NCCL_P2P_NET_CHUNKSIZE=2097152 \
    -e NCCL_BUFFSIZE=8388608 \
    -e NCCL_TUNER_PLUGIN="${OFI_NCCL_LIB}/libnccl-ofi-tuner.so" \
    -e NCCL_ASYNC_ERROR_HANDLING=1 \
    -e NCCL_DEBUG=WARN \
    -e HF_DATASETS_OFFLINE=1 \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    \
    `# ── PyTorch memory allocator ────────────────────────────────────` \
    `# Capital T is required in pytorch:25.04-py3. Lowercase true causes a RuntimeError.` \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    \
    `# ── Library paths (matching dockerfile ENV) ─────────────────────` \
    -e LD_LIBRARY_PATH="${OFI_NCCL_LIB}:${EFA_LIB}:/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib" \
    -e PATH="${OPENMPI_BIN}:${EFA_BIN}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    \
    -w /code \
    \
    "${IMAGE_NAME}" \
    bash

echo "✓ Container '${CONTAINER_NAME}' created and running."

# ── Runtime Python packages not in the base image ───────────────────────────
# Install packages that are required by img_2_svg_pretraining training but missing from the
# current image (omegaconf, open-clip-torch). These will be baked into the
# Dockerfile on the next image rebuild; this step is an immediate fallback so
# jobs don't fail while the image is being rebuilt.
echo ""
echo "Installing runtime Python packages..."
docker exec "${CONTAINER_NAME}" pip3 install --no-cache-dir --quiet \
    "omegaconf>=2.3,<3.0" \
    "open-clip-torch>=2.20,<3.0" \
    "datasets>=2.0,<4.0"
echo "✓ Runtime packages installed."

# ── SSH key setup (required for torchrun / pdsh cross-node communication) ────
echo ""
echo "Setting up SSH keys..."
if [ -d ~/.ssh ]; then
    docker cp ~/.ssh/. "${CONTAINER_NAME}":/root/.ssh/
    docker exec "${CONTAINER_NAME}" bash -c \
        "chmod 700 /root/.ssh && chmod 600 /root/.ssh/* && chown -R root:root /root/.ssh"
    echo "✓ SSH keys configured inside container."
else
    echo "⚠  No ~/.ssh directory found on host — skipping SSH key setup."
    echo "   Cross-node SSH (needed for pdsh/torchrun) will not work until you add keys."
fi

# ── Sanity checks inside the container ──────────────────────────────────────
echo ""
echo "Running sanity checks inside container..."

docker exec -i "${CONTAINER_NAME}" bash << 'EOF'

echo ""
echo "  [1/4] EFA devices visible:"
if ls /dev/infiniband/ >/dev/null 2>&1; then
    echo "  ✓ EFA devices found"
else
    echo "  ✗ No EFA devices — check host EFA driver and --privileged flag"
fi

echo ""
echo "  [2/4] NCCL version:"
if python3 -c "import torch; print(f'  ✓ NCCL {torch.cuda.nccl.version()}')" 2>/dev/null; then
    :
else
    echo "  ✗ Could not query NCCL version"
fi

echo ""
echo "  [3/4] OFI NCCL tuner plugin:"
if ls /opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so >/dev/null 2>&1; then
    echo "  ✓ Tuner plugin found"
else
    echo "  ✗ Tuner plugin missing — NCCL algorithm selection will be unguided"
fi

echo ""
echo "  [4/4] GPU count:"
if python3 -c "import torch; print(f'  ✓ {torch.cuda.device_count()} GPUs visible')" 2>/dev/null; then
    :
else
    echo "  ✗ Could not query GPU count"
fi

EOF

echo "=== Initialization Complete ==="
echo "To access the container, run: docker exec -it $CONTAINER_NAME bash"
