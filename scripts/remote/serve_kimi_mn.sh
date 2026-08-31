#!/usr/bin/env bash
# Serve Kimi K2.6 across 2 nodes (16 x H100) via the running Ray cluster.
#
# WHY MULTI-NODE AT ALL: the checkpoint is 555GB. On one node (640GB raw,
# ~608GB at util 0.95) the weights load but vLLM then dies with "No available
# memory for the cache blocks" -- measured, not assumed. 16 GPUs give 1280GB,
# leaving ~700GB of KV cache.
#
# TP=8, PP=2 is vLLM's canonical multi-node shape: tensor-parallel WITHIN a node
# over NVLink, pipeline-parallel ACROSS nodes so only stage activations cross
# the network instead of an all-reduce per layer. 64 attention heads divide by 8
# cleanly (the divisibility trap that caps GLM-4.6V at TP=4).
#
# --disable-custom-all-reduce is REQUIRED here. vLLM's hand-written all-reduce
# kernel assumes every participating GPU can peer-access the others, which holds
# over NVLink inside one node and NOT across nodes. With it enabled the workers
# load all 64 shards and then die at init with
#     Cuda error custom_all_reduce.cuh:164 'invalid argument'
# surfacing only as the generic "worker died ... unexpected system error".
# Disabling it falls back to NCCL, which this cluster already passes a 16-rank
# all_reduce on (scripts/remote/nccl_mn_test.py).
#
# --trust-remote-code: Kimi ships its modelling code in the repo
# (KimiK25ForConditionalGeneration -> DeepseekV3ForCausalLM inside), so the
# loader must execute it or config load fails outright.
#
# WEIGHTS READ FROM NODE-LOCAL NVMe, NOT LUSTRE. Measured on this exact model:
# 16 ranks pulling one checkpoint off shared Lustre ran ~35-40s per shard
# (~60 min); the same shards off local NVMe ran ~7s (~10 min). Each node holds
# its own copy at the SAME path, so every rank reads its own disk and the ranks
# never contend. scripts/remote/stage_nvme.sh puts the copy there.
set -u
C=animatebanana-mn
NVME_HUB=/opt/dlami/nvme/venkat.kesav/hf_cache
docker exec -d -e HF_HOME=$NVME_HUB -e HF_HUB_CACHE=$NVME_HUB/hub \
  -e HUGGINGFACE_HUB_CACHE=$NVME_HUB/hub -e HF_HUB_OFFLINE=1 "$C" bash -lc \
  '/environments/v5serve/bin/python -m vllm.entrypoints.openai.api_server \
     --model moonshotai/Kimi-K2.6 --served-model-name moonshotai/Kimi-K2.6 \
     --trust-remote-code \
     --tensor-parallel-size 8 --pipeline-parallel-size 2 \
     --distributed-executor-backend ray \
     --disable-custom-all-reduce \
     --gpu-memory-utilization 0.90 \
     --max-model-len 131072 --limit-mm-per-prompt.image 2 --max-num-seqs 16 \
     --port 8011 --host 0.0.0.0 > /tmp/vllm_kimi_mn.log 2>&1'
echo "launched on $(hostname)"
