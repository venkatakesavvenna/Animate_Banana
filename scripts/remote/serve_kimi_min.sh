#!/usr/bin/env bash
# Kimi K2.6 on 2 nodes -- MINIMAL profile. Escalation target if the stable
# profile (see serve_kimi_stable.sh) also deadlocks: additionally drops
# --enable-expert-parallel (cross-node all-to-all over EFA is the remaining
# collective that can wedge) and halves max-num-seqs. Slower, but every flag
# left is one the plain TP+PP path exercises constantly.
set -u
C=animatebanana-mn
NVME_HUB=/opt/dlami/nvme/venkat.kesav/hf_cache
docker exec -d -e HF_HOME=$NVME_HUB -e HF_HUB_CACHE=$NVME_HUB/hub \
  -e HUGGINGFACE_HUB_CACHE=$NVME_HUB/hub -e HF_HUB_OFFLINE=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 "$C" bash -lc \
  '/environments/v5serve/bin/python -m vllm.entrypoints.openai.api_server \
     --model moonshotai/Kimi-K2.6 --served-model-name moonshotai/Kimi-K2.6 \
     --trust-remote-code \
     --tensor-parallel-size 8 --pipeline-parallel-size 2 \
     --distributed-executor-backend ray \
     --disable-custom-all-reduce \
     --gpu-memory-utilization 0.92 \
     --max-model-len 131072 --max-num-batched-tokens 16384 \
     --limit-mm-per-prompt.image 2 --max-num-seqs 32 \
     --port 8011 --host 0.0.0.0 > /tmp/vllm_kimi_min.log 2>&1'
echo "launched (min profile) on $(hostname)"
