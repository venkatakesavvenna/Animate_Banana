#!/usr/bin/env bash
# torchrun launcher for the cross-node NCCL check. NODE_RANK differs per node.
set -u
C=animatebanana-mn
MASTER=${MASTER:-10.20.238.24}
NR=${NODE_RANK:?set NODE_RANK}
docker exec -e NODE_RANK=$NR animatebanana-mn bash -lc \
  "NCCL_NVLS_ENABLE=0 /environments/v5serve/bin/python -m torch.distributed.run \
     --nproc_per_node 8 --nnodes 2 --node_rank $NR \
     --rdzv_endpoint $MASTER:29533 --rdzv_backend c10d --max_restarts 0 \
     /code/scripts/remote/nccl_mn_test.py 2>&1 | tail -12"
