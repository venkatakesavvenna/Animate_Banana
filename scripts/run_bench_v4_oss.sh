#!/usr/bin/env bash
# v4 generation sweep: one open-weights model at a time, SVG target.
#
# Serves a model under vLLM, runs the pipeline over every measurable cell in
# all three styles, stops the server, waits for the GPUs to drain, moves on.
# Judging is deliberately NOT here -- see run_bench_v4_judge.sh. Generating
# with all six models first means the 440GB judge is loaded once rather than
# six times.
#
#   scripts/run_bench_v4_oss.sh                   # the whole roster
#   MODELS=qwen3vl32b scripts/run_bench_v4_oss.sh # one model
#   STYLES=progressive_reveal BATCH=1 scripts/run_bench_v4_oss.sh
#
# RESUMABILITY, WITHOUT NEW MACHINERY
# -----------------------------------
# docs/INFRA.md records a sweep losing in-flight work when a container was
# stopped, and names `vision_ingest.WorkerPool` as the intended fix. That is
# the wrong tool here: WorkerPool exists to stop N model-holding worker
# processes from deadlocking on a shared mp.Queue when one is SIGKILLed, and
# one long-lived vLLM server plus HTTP concurrency has neither workers nor
# queue. The failure it prevents cannot occur in this topology.
#
# What INFRA.md actually lost was progress and visibility, and the pipeline
# already solves most of that: artifacts are keyed by cache lineage on disk and
# every stage skips work whose output exists, so RE-RUNNING THIS SCRIPT IS
# ALREADY A RESUME. The markers below only avoid re-walking finished batches;
# they are an optimisation, not the correctness mechanism.
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python      # pipeline: playwright, no torch
VLLM_PY=/environments/ocr_env_vllm/bin/python           # serving: see WHY THIS VENV

# WHY ocr_env_vllm AND NOT gemma4
# -------------------------------
# gemma4 has the newer stack (torch 2.11.0+cu130, vllm 0.26.0) and was the
# obvious choice. It does not work here, and the reason is structural rather
# than fixable-once: both venvs are `--system-site-packages`, and gemma4 was
# built against a DIFFERENT base image than the one this container runs. So its
# own packages shadow some of the image's while inheriting others, and the two
# halves disagree:
#
#   numpy 2.4.4 (venv) + scipy 1.12.0 (image, built for numpy 1.x)
#       -> ImportError: numpy.core.multiarray failed to import
#   aiohttp (venv, new) + aiohappyeyeballs (image, old)
#       -> ImportError: cannot import name 'SocketFactoryType'
#
# Each is individually patchable and there is no reason to believe the list
# ends. ocr_env_vllm (torch 2.10.0+cu128, vllm 0.19.0) imports cleanly in this
# image, sees all 8 GPUs, and is the env docs/INFRA.md records actually serving
# Qwen3-VL-235B on this node.
#
# The version gap costs nothing that matters here: vllm 0.19.0's model registry
# was checked directly and already carries every architecture in this roster --
# Gemma3ForConditionalGeneration, Gemma4ForConditionalGeneration,
# Qwen3_5ForConditionalGeneration, Qwen3VLForConditionalGeneration and
# Glm4vMoeForConditionalGeneration.
PORT=8011
CONFIGS=$REPO/src/img_2_svg_pretraining/pipeline/configs

MODELS=${MODELS:-"qwen3vl32b gemma3_27b gemma4_31b qwen36_27b qwen36_35b_a3b glm46v qwen3vl235b internvl35_241b"}
STYLES=${STYLES:-"progressive_reveal alpha_masking colour_pop"}
BATCH=${BATCH:-8}
TARGET=svg
# Generous: a cold 215GB model across 8 cards is minutes, and failing early
# here costs a whole model's slot.
SERVE_TIMEOUT=${SERVE_TIMEOUT:-2100}

# HF_TOKEN. Needed at SERVE time, not only at download time: google/gemma-3-27b-it
# is a gated repo, and while its weights are cached, transformers still reaches
# the hub for `chat_template.jinja` (the snapshot ships `chat_template.json`,
# which newer versions no longer prefer). Without the token that single small
# file 401s and the whole server dies after loading nothing -- an authentication
# failure wearing the costume of a model-loading failure.
set -a; [ -f "$REPO/.env" ] && . "$REPO/.env"; set +a

LOG=$REPO/logs/bench_v4
STATE=$LOG/state
mkdir -p "$LOG" "$STATE"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/driver.log"; }

# `docker exec` for pipeline work. -u matters: without it every artifact lands
# root-owned, and there is no passwordless sudo on this node to undo that.
dexec() { docker exec -u "$(id -u):$(id -g)" "$CONTAINER" bash -lc "cd /code && $*"; }

repo_for() {   # config stem -> HF repo id, read from the config itself so this
                # script and the YAML cannot disagree about which model is served
  python3 - "$1" <<'PY'
import sys, pathlib, re
p = pathlib.Path("src/img_2_svg_pretraining/pipeline/configs") / f"bench_v4_svg_{sys.argv[1]}.yaml"
m = re.search(r"^  served:.*?^    model:\s*(\S+)", p.read_text(), re.S | re.M)
print(m.group(1) if m else "")
PY
}

gpus_idle() {   # every card back under 2GB -- vLLM does not always free promptly,
                # and a leftover holder is what made a previous run OOM at 0.90
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${used:-99999}" -lt 2000 ]
}

# What model, if any, is already answering on $PORT.
serving_model() {
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null
}

stop_server() {
  local pids c
  for c in "$CONTAINER" "$SERVE_CONTAINER_ALT"; do
    pids=$(docker exec "$c" bash -c \
             'ps -eo pid=,cmd= | grep "[v]llm.entrypoints.openai.api_server" | awk "{print \$1}"' \
           2>/dev/null | tr '\n' ' ')
    [ -n "${pids// /}" ] && { say "  stopping server in $c (pids: ${pids% })"
                              docker exec "$c" bash -c "kill $pids" 2>/dev/null; }
  done
  for _ in $(seq 1 60); do gpus_idle && return 0; sleep 5; done
  say "  WARNING: GPUs still holding memory after 5min"
  return 0
}

_stop_server_old() {
  local pids
  # KILL INSIDE THE CONTAINER. vLLM is launched by `docker exec` and therefore
  # runs as ROOT in the container's pid namespace. A host-side `kill` on those
  # pids returns EPERM -- and with stderr discarded that looks exactly like
  # "nothing to kill", so the driver sails on, fails to bind the port, and the
  # health probe then answers from the STALE server. Every symptom points
  # somewhere other than the cause.
  pids=$(docker exec "$CONTAINER" bash -c \
           'ps -eo pid=,cmd= | grep "[v]llm.entrypoints.openai.api_server" | awk "{print \$1}"' \
         2>/dev/null | tr '\n' ' ')
  if [ -n "${pids// /}" ]; then
    say "  stopping server (container pids: ${pids% })"
    # Collected first, then killed literally -- never `pkill -f vllm` from a
    # shell whose own command line contains "vllm" (INFRA.md: it kills the shell).
    docker exec "$CONTAINER" bash -c "kill $pids" 2>/dev/null
  fi
  for _ in $(seq 1 60); do gpus_idle && return 0; sleep 5; done
  say "  WARNING: GPUs still holding memory after 5min"
  docker exec "$CONTAINER" bash -c \
    'ps -eo pid=,cmd= | grep "[v]llm.entrypoints" | head -3' 2>/dev/null | tee -a "$LOG/driver.log"
}

serve() {   # $1 = HF repo id
  # Already up with the right weights? Reuse it. A cold load is 5-8 minutes and
  # a rerun after a driver-level failure would otherwise pay that again for
  # nothing. Checked by MODEL ID, not just /health -- health is equally true of
  # a server holding the previous model, and reusing that would silently
  # attribute one model's output to another.
  if [ "$(serving_model)" = "$1" ]; then
    say "  $1 already serving on :$PORT -- reusing"
  else
    say "  serving $1 on :$PORT"
    _launch "$1" || return 1
  fi

  say "  /health up -- now the real check"
  # /health is true of a server whose multimodal path is broken, and every
  # stage of this pipeline sends an image.
  dexec "PYTHONPATH=src $PY -u scripts/vllm_smoke.py \
           --config $CONFIGS/bench_v4_svg_${MODEL}.yaml --backend served" \
    2>&1 | tee -a "$LOG/serve_${MODEL}.log" | grep -q "SMOKE OK"
}

# THE SECOND SERVING CONTAINER
#
# Three models -- gemma-4-31B, GLM-4.6V and InternVL3.5-241B-Flash -- cannot be
# served from `animatebanana-v4` at all. Each needs a newer stack than that
# image can host:
#   gemma-4-31B    transformers must know `gemma4`      (4.57.6 does not)
#   GLM-4.6V       transformers must know its processor (4.57.6 returns a bare
#                                                        tokenizer, vLLM rejects)
#   InternVL-Flash vLLM must know the `gating` weights  (0.19.0 does not)
#
# The gemma4 venv has those versions but SIGSEGVs in NCCL there, because its
# torch is built for CUDA 13 while that image is CUDA 12.8. Confirmed with a
# two-rank all_reduce and unaffected by every NCCL env workaround.
#
# So these are served from `animatebanana-serve`: nvcr 25.12 (CUDA 13.1) with a
# venv built WITHOUT system site-packages, holding torch 2.11.0+cu130, vllm
# 0.26.0 and transformers 5.15.1. The clean venv is the point -- every failure
# above came from a --system-site-packages venv half-shadowing its image.
#
# --network host means the pipeline in the other container still reaches it on
# 127.0.0.1:$PORT with nothing else to configure.
SERVE_CONTAINER_ALT=animatebanana-serve
SERVE_PY_ALT=/venvs/v4serve/bin/python

serve_container_for() {
  case "$1" in
    *gemma-4*|*Gemma-4*|*GLM-4.6V*|*InternVL*) echo "$SERVE_CONTAINER_ALT" ;;
    *)                                          echo "$CONTAINER" ;;
  esac
}

# Which serving venv can load this model.
#
# ocr_env_vllm is the default and works for everything EXCEPT gemma4: vLLM
# 0.19.0 registers Gemma4ForConditionalGeneration, but it hands config parsing
# to transformers, and ocr_env_vllm has 4.57.6 -- which fails with "Transformers
# does not recognize this architecture". The registry entry is necessary and not
# sufficient.
#
# /environments/gemma4 has transformers 5.13.0 (the reason that venv exists at
# all -- see configs/open_weights.yaml) plus vllm 0.26.0. It needed two small
# packages to import in THIS image (aiohappyeyeballs, distro), now installed by
# docker/init_v4.sh. It is used only where it is needed, because the wider
# version skew that made it unsuitable as the default has not gone away.
vllm_py_for() {
  case "$1" in
    *gemma-4*|*Gemma-4*|*GLM-4.6V*|*InternVL*) echo "$SERVE_PY_ALT" ;;
    *)                                          echo "$VLLM_PY" ;;
  esac
}

_launch() {   # $1 = HF repo id
  # InternVLChatModel is not a native transformers architecture -- it ships its
  # modelling code in the repo, so the loader must be told to run it. Without
  # this the server dies at config load with a KeyError on the arch name.
  local EXTRA_ARGS=""
  case "$1" in *InternVL*) EXTRA_ARGS="--trust-remote-code" ;; esac
  local PY_SERVE; PY_SERVE=$(vllm_py_for "$1")
  local SC; SC=$(serve_container_for "$1")

  # --max-model-len must not exceed what the checkpoint declares. vLLM refuses
  # outright ("User-specified max_model_len (131072) is greater than the derived
  # max_model_len (max_position_embeddings=40960)") rather than clamping, which
  # is correct of it: on a RoPE model, positions past the trained maximum return
  # nan rather than failing loudly. Most of this roster declares 262144; only
  # InternVL3.5 is short at 40960, so it alone is capped -- and its completion
  # budget with it, since prompt+completion share that window.
  local CTX=${MODEL_CTX:-131072}
  case "$1" in *InternVL*) CTX=40960 ;; esac

  # TENSOR PARALLELISM IS NOT FREE HERE.
  #
  # google/gemma-4-31B-it can only be served by the gemma4 venv (it is the only
  # one whose transformers knows the architecture), and that venv CANNOT do
  # multi-GPU in this container: every rank takes SIGSEGV inside NCCL init.
  # Verified directly with a two-rank all_reduce, and unaffected by
  # NCCL_CUMEM_ENABLE=0, NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1, or the
  # cu13 LD_LIBRARY_PATH that configs/open_weights.yaml prescribes. The cause is
  # structural: that venv's torch is built for CUDA 13 while this image is CUDA
  # 12.8, so its bundled NCCL 2.28.9 is the wrong generation. Fixing it properly
  # needs a CUDA-13 base image, not a flag.
  #
  # 62.5GB of weights fit on one 80GB card, so TP=1 avoids NCCL altogether. The
  # cost is KV cache: ~9GB left at 0.90 utilisation, hence the reduced context.
  local TP=8
  say "  (serving venv: $PY_SERVE)"
  docker exec -d -e HF_TOKEN="${HF_TOKEN:-}" "$SC" bash -lc \
    "$PY_SERVE -m vllm.entrypoints.openai.api_server \
       --model '$1' --served-model-name '$1' \
       --tensor-parallel-size $TP --gpu-memory-utilization 0.90 \
       --max-model-len $CTX --limit-mm-per-prompt.image 2 --max-num-seqs 16 \
       $EXTRA_ARGS --port $PORT --host 127.0.0.1 > /tmp/vllm_gen_${MODEL}.log 2>&1"

  local waited=0
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 10; waited=$((waited + 10))
    # FAIL FAST ON A DEAD SERVER. A crash during load (bad weights, an
    # unregistered architecture, a gated repo 401) leaves nothing listening,
    # and waiting the full timeout for a process that no longer exists burns
    # SERVE_TIMEOUT per model -- over three hours across a six-model roster,
    # all of it after the answer was already knowable.
    if ! docker exec "$SC" bash -c \
         'ps -eo cmd= | grep -q "[v]llm.entrypoints.openai.api_server"' 2>/dev/null; then
      say "  FAILED: server process died after ${waited}s"
      docker exec "$SC" grep -iE "Error|Exception|error:" /tmp/vllm_gen_${MODEL}.log 2>/dev/null \
        | grep -v paddlex | tail -8 | tee -a "$LOG/driver.log"
      return 1
    fi
    if [ "$waited" -ge "$SERVE_TIMEOUT" ]; then
      say "  FAILED: no /health after ${SERVE_TIMEOUT}s"
      docker exec "$SC" tail -20 /tmp/vllm_gen_${MODEL}.log | tee -a "$LOG/driver.log"
      return 1
    fi
  done
  say "  loaded in ${waited}s"
}

say "=== v4 OSS sweep | models: $MODELS | styles: $STYLES | target=$TARGET"

for MODEL in $MODELS; do
  CFG=$CONFIGS/bench_v4_svg_${MODEL}.yaml
  [ -f "$CFG" ] || { say "!! no config for $MODEL"; continue; }
  if [ -f "$STATE/${MODEL}.done" ]; then say "== $MODEL already complete"; continue; fi

  REPO_ID=$(cd "$REPO" && repo_for "$MODEL")
  say "== $MODEL ($REPO_ID)"

  # Only tear down if the wrong weights are loaded; see serve().
  [ "$(serving_model)" = "$REPO_ID" ] || stop_server
  if ! serve "$REPO_ID"; then
    say "  SKIPPING $MODEL: server never became usable"
    touch "$STATE/${MODEL}.failed"; stop_server; continue
  fi

  for STYLE in $STYLES; do
    ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py \
            --root data/animatebench_v4 --target $TARGET --style "$STYLE")
    # shellcheck disable=SC2206
    arr=($ids)
    [ ${#arr[@]} -eq 0 ] && { say "  $STYLE: no measurable cells"; continue; }
    say "  $STYLE: ${#arr[@]} cell(s)"

    for ((i = 0; i < ${#arr[@]}; i += BATCH)); do
      batch="${arr[*]:i:BATCH}"
      tag="${MODEL}_${STYLE}_$((i / BATCH))"
      [ -f "$STATE/$tag.done" ] && continue

      dexec "PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
               all --config ${CFG#$REPO/} --style $STYLE --only $batch" \
        >>"$LOG/pipeline_${MODEL}_${STYLE}.log" 2>&1
      rc=$?
      say "    batch $((i / BATCH)) [$batch] exit=$rc"
      [ $rc -eq 0 ] && touch "$STATE/$tag.done"
    done
  done

  touch "$STATE/${MODEL}.done"
  say "== $MODEL done"
  stop_server
done

say "=== sweep complete"
say "next: scripts/run_bench_v4_judge.sh  (loads Qwen3-VL-235B once, scores every model)"
