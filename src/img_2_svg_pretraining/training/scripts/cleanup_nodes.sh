#!/bin/bash
# Remove img_2_svg_pretraining Docker containers and images from a list of nodes.
#
# Usage:
#   bash scripts/cleanup_nodes.sh [--dry-run]
#
# Expands SLURM node range notation and SSHes to each node.

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

USER_NAME=$(whoami)
CONTAINER_NAME="img-2-svg-pretraining-multinode-${USER_NAME}"
IMAGE_NAME="img-2-svg-pretraining-multinode-${USER_NAME}:latest"

# Full list of nodes that ran failed smoke test jobs (2075-2146, excluding 2074 molmo).
# ip-10-20-218-187 is excluded from future job scheduling (Docker build cache corruption)
# but is still cleaned here so it's left in a tidy state.
NODES=(
    ip-10-20-202-110
    ip-10-20-204-252
    ip-10-20-216-9
    ip-10-20-218-142
    ip-10-20-218-187
    ip-10-20-224-174
    ip-10-20-225-177
    ip-10-20-227-52
    ip-10-20-227-97
    ip-10-20-229-109
    ip-10-20-230-169
    ip-10-20-235-133
    ip-10-20-238-191
    ip-10-20-238-24
    ip-10-20-239-162
    ip-10-20-239-233
    ip-10-20-241-80
)

echo "=== Node cleanup: remove Docker container + image ==="
echo "    Container : ${CONTAINER_NAME}"
echo "    Image     : ${IMAGE_NAME}"
echo "    Nodes     : ${#NODES[@]}"
echo ""

cleaned=0
skipped=0
failed=0

for node in "${NODES[@]}"; do
    echo -n "  ${node} ... "
    if $DRY_RUN; then
        echo "[DRY] ssh ${node} 'docker rm -f ${CONTAINER_NAME} 2>/dev/null; docker rmi ${IMAGE_NAME} 2>/dev/null'"
        cleaned=$(( cleaned + 1 ))
        continue
    fi

    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "${node}" \
        "docker rm -f '${CONTAINER_NAME}' 2>/dev/null || true; \
         docker rmi '${IMAGE_NAME}' 2>/dev/null || true; \
         docker builder prune -f 2>/dev/null || true; \
         echo done" 2>/dev/null; then
        cleaned=$(( cleaned + 1 ))
    else
        echo "SSH failed (node may be offline)"
        failed=$(( failed + 1 ))
    fi
done

echo ""
echo "=== Summary ==="
echo "  Cleaned : ${cleaned}"
echo "  Failed  : ${failed}"
$DRY_RUN && echo "  [DRY RUN]"
