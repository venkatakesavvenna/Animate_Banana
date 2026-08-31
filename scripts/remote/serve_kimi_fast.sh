#!/usr/bin/env bash
# Kimi K2.6 on 2 nodes, tuned for THROUGHPUT rather than max context.
#
# WHAT THE FIRST ATTEMPT ACTUALLY MEASURED
# ----------------------------------------
# The model is NOT inherently slow. With a full batch it sustains
#   Avg generation throughput: 736-760 tok/s, Running: 16
# which is the same league as Gemma-4's 963 tok/s on 8 GPUs. The 105 tok/s
# figure that made it look hopeless was the TAIL of a batch: one straggler
# request decoding alone with 15 slots idle. Averaged over a run the tail
# dominates, so the fix is to stop the tail being long, not to add hardware.
#
# THE THREE CHANGES
#
# 1. --max-model-len 131072 (was 131072)
#    vLLM reserves KV by max-model-len, so a 131072 window let it hold only
#    "Maximum concurrency for 131,072 tokens per request: 6.86x" against
#    --max-num-seqs 16 -- it could never fill the batch it was told to run.
#    Shrinking this was tried twice and BOTH failed, because the window must
#    exceed max_tokens (65536) by the whole prompt -- vLLM rejects
#    prompt+max_tokens > max_model_len outright rather than clamping:
#      32768 -> 400 "prompt 16385 + 16384 output > 32768", and silent
#               "no svg document" -- 67KB of pure reasoning truncated before
#               any SVG, because Kimi spends completion tokens thinking first
#      65536 -> 400 "requested 65536 output tokens" with NO room for a prompt
#    So the roster's 65536 completion budget requires a 131072 window, exactly
#    as the other five models are served.
#
#    Concurrency is instead recovered by the flags below -- max-num-seqs and
#    DCP -- not by shrinking the window. That is the real lesson: the
#    "Maximum concurrency 6.86x" warning was never the binding constraint;
#    --max-num-seqs 16 was.
#
# 2. --enable-expert-parallel
#    384 routed experts. Under plain TP every rank holds a slice of EVERY
#    expert; EP shards experts across ranks instead, which is the layout vLLM
#    recommends for DeepSeek-V3-family MoE and cuts per-rank weight traffic.
#
# 3. --async-scheduling
#    Overlaps the scheduler with decode, which matters most exactly when the
#    batch is draining -- the regime that was costing us the throughput.
#
# 4. --max-num-seqs 256 (was 16)
#    THE ACTUAL CEILING. "Running: 16 reqs" was never the model saturating --
#    16 is the value the first attempt passed. The "6.86x concurrency" warning
#    that seemed to justify it is computed against max-model-len 131072; at the
#    real ~10.5k tokens per request the 898,752-token KV cache holds ~86
#    concurrent requests, not 6.86.
#
# 5. --decode-context-parallel-size 8
#    Kimi uses MLA, which has effectively ONE kv head, so TP=8 does not split
#    the KV cache -- all 8 ranks store an identical copy. The arithmetic is
#    exact: 898,752 tok x 576 (kv_lora_rank 512 + rope 64) x 2 bytes x 31
#    layers = 29.89 GiB, matching the per-worker figure vLLM logged. DCP splits
#    that replicated cache 8 ways. vLLM's divisibility checks on DCP are
#    GQA-only; MLA models take the unconstrained branch, so DCP=8 under TP=8 is
#    legal here.
#
# max-num-batched-tokens 16384 keeps chunked prefill from letting one long
# prompt monopolise a step and stall the decode of everything else.
set -u
C=animatebanana-mn
NVME_HUB=/opt/dlami/nvme/venkat.kesav/hf_cache
# VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: 300 (default) -> 1800.
#
# THE CRASH THIS FIXES. Under judging load the server died three times with a
# clean shutdown and no OOM. The real signal was buried under it:
#     TimeoutError: RPC call to sample_tokens timed out.
#     -> EngineDeadError -> 500s -> shutdown
# The engine declares itself dead when a worker misses this deadline. Across
# two nodes, with many image-heavy prompts queued behind each other, a single
# step can exceed 300s -- so the server was killing itself over a slow step,
# not a broken one. Raising the deadline turns a fatal into a wait.
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
     --async-scheduling \
     --gpu-memory-utilization 0.92 \
     --decode-context-parallel-size 8 \
     --max-model-len 131072 --max-num-batched-tokens 16384 \
     --limit-mm-per-prompt.image 2 --max-num-seqs 256 \
     --port 8011 --host 0.0.0.0 > /tmp/vllm_kimi_fast.log 2>&1'
echo "launched (fast profile) on $(hostname)"
