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
# CFG_OVERRIDE lets a non-v5 suite (e.g. bench_v6_svg_*) reuse this script
# unchanged. The served model id is still read FROM THAT CONFIG below, so the
# override cannot desync the script from the YAML.
CFG=${CFG_OVERRIDE:-$CONFIGS/bench_v5_svg_${MODEL}.yaml}
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
# WHICH VENV
#
# v5serve is a CLEAN venv (built WITHOUT --system-site-packages) holding
# torch 2.13+cu130 / vllm 0.28.0 / transformers 5.16.1. It replaces the earlier
# split between ocr_env_vllm and gemma4, and the reason it can is that the NCCL
# SIGSEGV which forced gemma-4 to TP=1 was never really about CUDA versions --
# it was --system-site-packages letting the IMAGE's libraries half-shadow the
# venv's. Verified directly: a 2-rank all_reduce SIGSEGVs under /environments/
# gemma4 and returns "rank 0 OK / rank 1 OK" here, on the same driver and the
# same image.
#
# So every model gets TP=8. transformers 5.16.1 also knows every architecture
# in this roster (Gemma4, Glm4vMoe, KimiK25, Qwen3_5, Qwen3VLMoe all present in
# vllm 0.28.0's registry, and AutoProcessor returns a real Glm46VProcessor for
# GLM-4.6V rather than the bare tokenizer 4.57.6 produced).
VENV=${VENV_OVERRIDE:-/environments/v5serve}
TP=${TP_OVERRIDE:-8}

# TENSOR PARALLELISM MUST DIVIDE THE HEAD COUNT.
#
# GLM-4.6V declares 12 attention heads, and vLLM shards heads across ranks:
#   AssertionError: 12 is not divisible by 8
# raised inside dist_utils.divide during worker init. This surfaces as the
# generic "WorkerProc initialization failed" that the v4 sweep also hit and
# never diagnosed -- the real assertion is in the WORKER's stderr, which the
# EngineCore/APIServer wrapper does not echo. TP=4 divides 12 and gives
# 4x80 = 320GB against 215GB of weights, which is ample.
case "$MODEL" in
  glm46v) TP=${TP_OVERRIDE:-4} ;;
esac

# Pinned revisions, where a shared cache's refs/main points at an INCOMPLETE
# snapshot. gemma-4-31B's main is 842da379... with ZERO safetensors while two
# complete revisions sit beside it; under HF_HUB_OFFLINE that ref is followed
# blindly and the server dies with "Cannot find any model weights".
case "$MODEL" in
  gemma4_31b) EXTRA_ARGS="${EXTRA_ARGS:-} --revision 3548789868c5356dbf307c98e6f609007b82b3eb" ;;
  # Kimi ships its modelling code IN THE REPO rather than in transformers, so
  # the loader must be told to run it; without this it dies at config load with
  # "contains custom code which must be executed".
  kimi_k26)   EXTRA_ARGS="${EXTRA_ARGS:-} --trust-remote-code" ;;
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
# NO SHIMS on v5serve. /environments/v5_shims exists for the OLD gemma4 venv,
# where the image's ABI-broken flash_attn leaked in through
# --system-site-packages and had to be masked. v5serve is a clean venv with its
# own working flash_attn, so the mask now BREAKS it: vllm 0.28 looks the module
# up and the stub's ModuleNotFoundError kills startup.
docker exec -d -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e PYTHONPATH="${SERVE_PYTHONPATH:-}" "$C" bash -lc \
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
