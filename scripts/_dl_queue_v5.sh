#!/usr/bin/env bash
# v5 weight download. Sequential BY DESIGN: the link saturates at ~1.2GB/s on a
# single repo, so parallel pulls trade one finished model for several
# half-finished ones. Order is smallest-first so bring-up can start on model 1
# while the big ones are still arriving.
#
# Weights land in /hf_cache/hub -> /opt/dlami/nvme/venkat.kesav/hf_cache/hub.
# All three HF cache vars are set by the container: the image PRESETS
# HF_HUB_CACHE and it OVERRIDES HF_HOME, which once sent a 200GB pull into a
# colleague's directory.
set -u
C=${C:-animatebanana-v5}
PY=/environments/gemma4/bin/python
LOG=/code/logs/bench_v5

REPOS=${REPOS:-"
Qwen/Qwen3.8-27B
zai-org/GLM-4.6V
moonshotai/Kimi-K2.6
"}

for r in $REPOS; do
  tag=$(echo "$r" | tr '/.' '__')
  echo "[$(date +%H:%M:%S)] downloading $r"
  docker exec -e HF_TOKEN="${HF_TOKEN:-}" "$C" bash -c \
    "$PY -m huggingface_hub.commands.huggingface_cli download '$r' --max-workers 16" \
    > "$LOG/dl_${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)]   exit=$? cache=$(docker exec $C du -sh /hf_cache/hub 2>/dev/null | cut -f1)"
done
echo "[$(date +%H:%M:%S)] queue complete"
