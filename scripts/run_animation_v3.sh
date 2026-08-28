#!/usr/bin/env bash
# Wait, bring up the Qwen judge, then score every v3 cell's animation tree as
# it becomes ready. Two phases:
#
#   1. Sleep DELAY_S (default 10 minutes -- the node the GPUs are shared on is
#      still finishing another job), then launch vLLM inside the
#      already-running img2svg-qwen-venkat container (host networking, 440GB
#      of Qwen3-VL-235B-A22B-Instruct already cached on local NVMe -- no
#      re-download). ~10 minutes to load.
#   2. Hand off to scripts/animation_sweep.py, which polls all 48 GT-covered
#      cells and scores each one's animation tree the moment its pipeline
#      run (bench_v3_or.sh, running independently) has produced frames.
#
# Runs from the BARE HOST throughout -- see animation_sweep.py's docstring
# for why (network-mode mismatch between the two containers in play).
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
QWEN_CONTAINER=img2svg-qwen-venkat
DELAY_S=${DELAY_S:-600}
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/animation_sweep.log"; }

say "=== waiting ${DELAY_S}s for the GPU node to free up"
sleep "$DELAY_S"

say "=== launching vLLM (Qwen3-VL-235B-A22B-Instruct, TP=8, bf16, port 8010)"
docker exec "$QWEN_CONTAINER" bash -lc '
  D=/opt/dlami/nvme/venkat.kesav/hf_cache
  tmux kill-session -t qwen_serve 2>/dev/null
  tmux new-session -d -s qwen_serve "HF_HOME=$D HF_HUB_CACHE=$D/hub \
    HUGGINGFACE_HUB_CACHE=$D/hub VLLM_CACHE_ROOT=/tmp/vllm_cache_venkat \
    /environments/ocr_env_vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 --max-model-len 32768 \
    --limit-mm-per-prompt.image 3 --max-num-seqs 16 \
    --port 8010 --host 127.0.0.1 > /tmp/qwen_serve.log 2>&1"
' 2>&1 | tee -a "$LOGDIR/animation_sweep.log"

say "=== vLLM launching in the background (tmux session qwen_serve); handing off to the Python driver, which waits for /health itself"

exec env PYTHONPATH="$REPO/src" python3 "$REPO/scripts/animation_sweep.py"
