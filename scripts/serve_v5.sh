#!/usr/bin/env bash
# Serve one v5 model on :8011, or stop whatever is serving.
#
#   scripts/serve_v5.sh qwen38_27b     # bring up
#   scripts/serve_v5.sh stop           # tear down and wait for VRAM to drain
#
# Two behaviours carried over from run_bench_v4_oss.sh, both of which cost real
# time to learn:
#
#  * Reuse is decided by the SERVED MODEL ID, never by /health alone. /health is
#    equally true of a server still holding the PREVIOUS model, and reusing that
#    would attribute one model's output to another -- silently.
#  * The kill happens INSIDE the container. vLLM is launched via `docker exec`
#    and runs as root in the container's pid namespace, so a host-side kill
#    returns EPERM; with stderr discarded that looks exactly like "nothing to
#    kill", and the driver sails on to fail binding the port.
set -u
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=animatebanana-v5
PORT=8011
CONFIGS=src/img_2_svg_pretraining/pipeline/configs
set -a; [ -f "$REPO/.env" ] && . "$REPO/.env"; set +a

say() { echo "[$(date +%H:%M:%S)] $*"; }

serving_model() {
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null
}

gpus_idle() {
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${u:-99999}" -lt 2000 ]
}

stop_server() {
  local pids
  pids=$(docker exec "$C" bash -c \
    'ps -eo pid=,cmd= | grep "[v]llm.entrypoints.openai.api_server" | awk "{print \$1}"' \
    2>/dev/null | tr '\n' ' ')
  [ -n "${pids// /}" ] && { say "stopping (pids: ${pids% })"; docker exec "$C" bash -c "kill $pids" 2>/dev/null; }
  for _ in $(seq 1 60); do gpus_idle && { say "GPUs drained"; return 0; }; sleep 5; done
  say "WARNING: GPUs still holding memory after 5min"
}

[ "${1:-}" = "stop" ] && { stop_server; exit 0; }

MODEL=${1:?usage: serve_v5.sh <stem>|stop}
CFG=$CONFIGS/bench_v5_svg_${MODEL}.yaml
[ -f "$REPO/$CFG" ] || { say "no config for $MODEL"; exit 1; }

# Read the repo id from the config so this script and the YAML cannot disagree
# about which model is served. Parse it as YAML rather than by regex: a regex
# for `model:` after `served:` also matches the JUDGE block further down (it has
# its own `model:`), which silently serves the wrong 471GB model.
REPO_ID=$(cd "$REPO" && python3 -c "
import yaml,sys
print(yaml.safe_load(open('$CFG'))['backends']['served']['model'])")

# gemma4's venv is the only transformers that knows the gemma4 architecture
# (5.13.0). Its torch is cu130 on a cu128 image, so its NCCL is the wrong
# generation and multi-GPU SIGSEGVs -- 62.5GB fits one card, so TP=1 avoids
# NCCL entirely. Everything else serves from ocr_env_vllm at TP=8.
# WHICH VENV, AND WHY IT IS NOT UNIFORM
#
# ocr_env_vllm (torch 2.10+cu128, vllm 0.19.0, transformers 4.57.6) is the
# default: its CUDA matches the image, so multi-GPU NCCL works and TP=8 is
# available.
#
# Two models cannot use it, both because vLLM defers config/processor parsing
# to transformers and 4.57.6 is too old:
#   gemma-4-31B  "Transformers does not recognize this architecture: gemma4"
#   GLM-4.6V     returns a bare PreTrainedTokenizerFast where vLLM demands a
#                ProcessorMixin -> "Invalid type of HuggingFace processor"
# transformers 5.13.0 in /environments/gemma4 handles both (verified:
# AutoProcessor gives Glm46VProcessor).
#
# The cost of that venv is that its torch is cu130 on a CUDA 12.8 image, so its
# bundled NCCL is the wrong generation and ANY tensor-parallel launch dies in
# WorkerProc init. Hence TP=1 there -- fine for gemma-4 (62.5GB on one 80GB
# card) but IMPOSSIBLE for GLM-4.6V at 215GB, which needs the memory of several
# cards. GLM is therefore attempted on ocr_env_vllm first and is expected to
# fail on the processor; see docs/V5_BRINGUP.md.
case "$MODEL" in
  gemma4_31b) VENV=/environments/gemma4; TP=1 ;;
  glm46v)     VENV=${GLM_VENV:-/environments/gemma4}; TP=${GLM_TP:-8} ;;
  *)          VENV=/environments/ocr_env_vllm; TP=8 ;;
esac
CTX=${CTX:-131072}

if [ "$(serving_model)" = "$REPO_ID" ]; then say "$REPO_ID already serving -- reusing"; exit 0; fi
stop_server
say "serving $REPO_ID (venv=$VENV tp=$TP ctx=$CTX)"
# HF_HUB_OFFLINE for the symlinked caches. gemma-4-31B and Qwen3-VL-235B are
# symlinks into COLLEAGUES' cache dirs -- readable, not writable. transformers
# still reaches the hub for small files (chat_template.jinja) even with weights
# present, then fails writing the .incomplete blob, and the error names a
# permission problem rather than the missing file it wanted. Both snapshots are
# verified complete, so offline is safe and turns a confusing failure into no
# request at all.
docker exec -d -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e PYTHONPATH=/environments/v5_shims "$C" bash -lc \
  "$VENV/bin/python -m vllm.entrypoints.openai.api_server \
     --model '$REPO_ID' --served-model-name '$REPO_ID' \
     --tensor-parallel-size $TP --gpu-memory-utilization 0.90 \
     --max-model-len $CTX --limit-mm-per-prompt.image 2 --max-num-seqs 16 \
     ${EXTRA_ARGS:-} --port $PORT --host 127.0.0.1 > /tmp/vllm_${MODEL}.log 2>&1"

W=0; T=${SERVE_TIMEOUT:-2400}
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  sleep 10; W=$((W+10))
  # Fail fast on a dead process: waiting the full timeout for something that no
  # longer exists burns the whole budget after the answer is already knowable.
  if ! docker exec "$C" bash -c 'ps -eo cmd=|grep -q "[v]llm.entrypoints.openai.api_server"' 2>/dev/null; then
    say "FAILED: server died after ${W}s"
    docker exec "$C" grep -iE "Error|Exception|error:" /tmp/vllm_${MODEL}.log 2>/dev/null | tail -8
    exit 1
  fi
  [ $W -ge $T ] && { say "FAILED: no /health after ${T}s"; exit 1; }
done
say "loaded in ${W}s"
