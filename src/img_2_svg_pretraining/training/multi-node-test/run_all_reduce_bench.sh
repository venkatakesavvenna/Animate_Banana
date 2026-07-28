#!/bin/bash
# =============================================================================
# run_bench.sh — Run all_reduce_bench.py INSIDE the Docker container
#
# Copy this script into the container and run it directly:
#   docker cp run_bench.sh <container>:/code/run_bench.sh
#   docker exec -it <container> bash /code/run_bench.sh
#
# --- INTRANODE (single node, run on node 0 only) ---
#   bash run_bench.sh
#
# --- INTERNODE (run on BOTH nodes at the same time) (also change master address and port.) ---
#   Node 0:  NNODES=1 NODE_RANK=0 bash run_all_reduce_bench.sh
#   Node 1:  NNODES=1 NODE_RANK=1 bash run_all_reduce_bench.sh
# =============================================================================

set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:-10.20.204.252}"   # node 0's IP — always node 0 on both nodes
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-1}"                          # set to 1 for intranode, 2 for internode
NODE_RANK="${NODE_RANK:-0}"                    # 0 on node 0, 1 on node 1
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
BENCH_SCRIPT="${BENCH_SCRIPT:-/code/src/multi-node-test/all_reduce_bench.py}"

# Disable NVLS (NVLink SHARP multicast) — requires Fabric Manager which is
# not available inside Docker containers. Without this you get:
#   ncclUnhandledCudaError: Failed to bind NVLink SHARP (NVLS) Multicast memory
export NCCL_NVLS_ENABLE=0

echo ""
echo "=== all_reduce_bench ==="
echo "  MASTER_ADDR   : ${MASTER_ADDR}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  NNODES        : ${NNODES}"
echo "  NODE_RANK     : ${NODE_RANK}"
echo "  GPUS_PER_NODE : ${GPUS_PER_NODE}"
echo "  BENCH_SCRIPT  : ${BENCH_SCRIPT}"
if [ "${NNODES}" -eq 1 ]; then
    echo "  MODE          : INTRANODE (NVLink / PCIe)"
else
    echo "  MODE          : INTERNODE (EFA / network)"
fi
echo "========================"
echo ""

python -u -m torch.distributed.run \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_backend c10d \
    --max_restarts 0 \
    ${BENCH_SCRIPT}