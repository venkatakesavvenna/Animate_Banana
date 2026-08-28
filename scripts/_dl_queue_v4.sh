#!/bin/bash
# Sequential, not parallel: the link already saturates at ~1.2GB/s on one repo,
# so concurrent pulls only trade one finished model for several half-finished
# ones. Order matches the roster's run order, so generation can start on model 1
# while model 2 is still arriving.
for m in Qwen/Qwen3-VL-32B-Instruct \
         google/gemma-3-27b-it \
         google/gemma-4-31B-it \
         Qwen/Qwen3.6-27B \
         Qwen/Qwen3.6-35B-A3B \
         zai-org/GLM-4.6V \
         Qwen/Qwen3-VL-235B-A22B-Instruct; do
  tag=$(echo "$m" | tr '/.' '__')
  echo "=== $(date +%H:%M:%S) $m"
  /environments/gemma4/bin/hf download "$m" --max-workers 16 \
    > "/code/logs/bench_v4/dl_${tag}.log" 2>&1
  echo "    exit=$? size=$(du -sh /hf_cache/hub 2>/dev/null | cut -f1)"
done
echo "=== $(date +%H:%M:%S) all downloads done"
