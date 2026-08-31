#!/usr/bin/env bash
# Kimi K2.6 on 2 nodes -- STABLE profile for the SSS/GPS/NAS judging run.
#
# WHY THIS EXISTS. serve_kimi_fast.sh deadlocked under concurrent judging
# load: 12 requests "running" / 36 waiting frozen for 6+ minutes, GPU util 0%
# on BOTH nodes, API server still answering /metrics 200, and it did NOT
# recover when every client was killed. That is the engine core losing
# requests it believes are in flight -- a scheduler/worker race, not queue
# pressure. Serial judging through the same server worked (210 real calls).
#
# The fast profile carried three throughput flags never validated under
# sustained concurrent load; the two dropped here are the known-racy ones:
#
#   --async-scheduling            DROPPED. Overlaps scheduling with decode;
#                                 with pipeline parallelism this is the
#                                 classic source of exactly this hang:
#                                 scheduler marks requests running that a
#                                 PP stage never completes. Prime suspect.
#   --decode-context-parallel-size 8
#                                 DROPPED. Newest feature of the three; MLA
#                                 takes the unconstrained (least-tested)
#                                 branch of its checks. Costs KV headroom,
#                                 not correctness: judge calls are ~10k
#                                 prompt + tiny thinking:false completions,
#                                 so the un-split 29.9GiB KV cache still
#                                 holds ~85 such requests.
#   --enable-expert-parallel      KEPT. Weight layout, present in every run
#                                 that worked; removing it doubles per-rank
#                                 weight traffic for 384 experts.
#
#   --max-num-seqs 256 -> 64      The judging clients present at most ~32
#                                 concurrent requests; 256 only widens the
#                                 race window.
#
# max-model-len stays 131072: vLLM rejects prompt+max_tokens > max-model-len
# outright (does not clamp), and the roster config asks 32768 completion for
# kimi. VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 kept -- a slow multi-node
# step over 300s must be a wait, not an EngineDeadError.
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
     --enable-expert-parallel \
     --gpu-memory-utilization 0.92 \
     --max-model-len 131072 --max-num-batched-tokens 16384 \
     --limit-mm-per-prompt.image 2 --max-num-seqs 64 \
     --port 8011 --host 0.0.0.0 > /tmp/vllm_kimi_stable.log 2>&1'
echo "launched (stable profile) on $(hostname)"
