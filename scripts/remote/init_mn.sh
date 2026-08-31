#!/usr/bin/env bash
# Multi-node serving container for Kimi K2.6. Run on EVERY participating node.
#
# WHY A SEPARATE CONTAINER FROM animatebanana-v5
# ----------------------------------------------
# v5 was created with --network host but WITHOUT --privileged and without the
# EFA verbs devices. Single-node NCCL works over NVLink and never needs them;
# cross-node NCCL goes through libfabric/EFA and does. Those flags can only be
# set at container CREATION, so a new container is required rather than an
# edit.
#
# Every EFA/NCCL setting below is load-bearing:
#   --privileged + /dev/infiniband  EFA verbs devices; without them libfabric
#                                   silently falls back to TCP (slow) or hangs
#   --network host                  EFA interfaces are invisible from a bridge
#                                   network, and Ray needs stable host IPs
#   --ipc host + --shm-size 512g    NCCL uses /dev/shm for intra-node
#                                   collectives; the 64MB default OOMs silently
#   --ulimit memlock=-1             EFA/RDMA needs unlimited pinned memory
#   NCCL_NVLS_ENABLE=0              NVLink SHARP multicast needs Fabric Manager,
#                                   which is NOT available inside Docker; without
#                                   this you get "Failed to bind NVLink SHARP
#                                   (NVLS) Multicast memory"
#   NCCL_SOCKET_IFNAME=^docker,lo,veth  keeps NCCL off virtual interfaces so it
#                                   picks the real EFA-backed NIC
#
# WEIGHTS COME FROM LUSTRE, NOT NODE-LOCAL NVMe: every rank must read the same
# checkpoint, and /opt/dlami/nvme is node-local (the second node has no copy).
set -u
C=${C:-animatebanana-mn}
IMAGE=${IMAGE:-vlm-ingest-pipeline-raghuveer.r:latest}
SHARED_HF=/fsxvision_new/venkat.kesav/hf_cache_shared

if docker ps -a --format '{{.Names}}' | grep -qx "$C"; then
  docker ps --format '{{.Names}}' | grep -qx "$C" || docker start "$C" >/dev/null
  echo "✓ $C already exists on $(hostname)"
else
  docker run -d --name "$C" \
    --gpus all --privileged --network host --ipc host \
    --shm-size=512g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/infiniband:/dev/infiniband \
    -e HF_HOME=$SHARED_HF \
    -e HF_HUB_CACHE=$SHARED_HF/hub \
    -e HUGGINGFACE_HUB_CACHE=$SHARED_HF/hub \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_CACHE_ROOT=/tmp/vllm_cache \
    -e TRITON_CACHE_DIR=/tmp/triton_cache \
    -e FI_PROVIDER=efa -e FI_EFA_USE_HUGE_PAGE=0 -e FI_EFA_FORK_SAFE=1 \
    -e NCCL_SOCKET_IFNAME='^docker,lo,veth' \
    -e NCCL_NVLS_ENABLE=0 \
    -e NCCL_DEBUG=WARN \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v /fsxvision_new:/fsxvision_new \
    -v /opt/dlami/nvme:/opt/dlami/nvme \
    -v /fsxvision_new/venkat.kesav/environments:/environments \
    -v /fsxvision_new/venkat.kesav/img_2_svg_pretraining:/code \
    -w /code "$IMAGE" sleep infinity >/dev/null
  echo "✓ $C created on $(hostname)"
fi

echo -n "  python: "; docker exec "$C" python3 -V 2>&1
echo -n "  efa:    "; docker exec "$C" bash -c 'ls /dev/infiniband/ 2>/dev/null|head -1' 2>&1
echo -n "  gpus:   "; docker exec "$C" /environments/v5serve/bin/python -c 'import torch;print(torch.cuda.device_count())' 2>&1|tail -1
echo -n "  weights:"; docker exec "$C" bash -c "ls $SHARED_HF/hub/models--moonshotai--Kimi-K2.6/snapshots/*/ 2>/dev/null|wc -l" 2>&1
