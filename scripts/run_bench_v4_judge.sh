#!/usr/bin/env bash
# v4 judging: load Qwen3-VL-235B ONCE, score every model against it.
#
# Run after scripts/run_bench_v4_oss.sh has generated for the whole roster.
# Loading a 440GB model takes ~10 minutes and needs all 8 cards, so it cannot
# coexist with a generation server -- doing this per model would spend an hour
# on loading alone.
#
# WHY THIS JUDGE
# --------------
# It is a different model family from every generator in the roster. Judging a
# model's own output with itself is the circular evaluation this project's
# notes repeatedly warn about; one independent judge across all six also means
# the models are compared on identical terms rather than each against itself.
#
# CHEAP SUITES FIRST, ON PURPOSE
# ------------------------------
# stage1/xml/sequence/stage3 are ~6-10 judged calls per cell and fill four of
# the scoreboard's five column groups. `animation` is ~94 calls per cell -- an
# order of magnitude more than the other four combined. Running the cheap pass
# first means an interruption during the expensive one still leaves most of the
# table standing, rather than nothing.
#
#   scripts/run_bench_v4_judge.sh                 # cheap suites, then animation
#   SUITES="stage1 xml" scripts/run_bench_v4_judge.sh
#   SKIP_ANIMATION=1 scripts/run_bench_v4_judge.sh
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
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
JUDGE=Qwen/Qwen3-VL-235B-A22B-Instruct
PORT=8010
CONFIGS=$REPO/src/img_2_svg_pretraining/pipeline/configs
EVALS=$REPO/data/animatebench_v4_cache/animatebench_v4/evals

MODELS=${MODELS:-"qwen3vl32b gemma3_27b gemma4_31b qwen36_27b qwen36_35b_a3b glm46v qwen3vl235b internvl35_241b"}
STYLES=${STYLES:-"progressive_reveal alpha_masking colour_pop"}
SUITES=${SUITES:-"stage1 xml sequence stage3"}
SKIP_ANIMATION=${SKIP_ANIMATION:-0}
SERVE_TIMEOUT=${SERVE_TIMEOUT:-2400}

# HF_TOKEN. Needed at SERVE time, not only at download time: google/gemma-3-27b-it
# is a gated repo, and while its weights are cached, transformers still reaches
# the hub for `chat_template.jinja` (the snapshot ships `chat_template.json`,
# which newer versions no longer prefer). Without the token that single small
# file 401s and the whole server dies after loading nothing -- an authentication
# failure wearing the costume of a model-loading failure.
set -a; [ -f "$REPO/.env" ] && . "$REPO/.env"; set +a

LOG=$REPO/logs/bench_v4
STATE=$LOG/state
mkdir -p "$LOG" "$STATE" "$EVALS"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/judge.log"; }
dexec() { docker exec -u "$(id -u):$(id -g)" "$CONTAINER" bash -lc "cd /code && $*"; }

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  say "=== bringing up the judge ($JUDGE, ~440GB, ~10min)"
  docker exec -d -e HF_TOKEN="${HF_TOKEN:-}" "$CONTAINER" bash -lc \
    "$VLLM_PY -m vllm.entrypoints.openai.api_server \
       --model '$JUDGE' --served-model-name '$JUDGE' \
       --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
       --max-model-len 32768 --limit-mm-per-prompt.image 3 \
       --max-num-seqs 32 --port $PORT --host 127.0.0.1 \
       > /tmp/vllm_judge.log 2>&1"
  waited=0
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 15; waited=$((waited + 15))
    if [ "$waited" -ge "$SERVE_TIMEOUT" ]; then
      say "FAILED: judge never came up"; docker exec "$CONTAINER" tail -20 /tmp/vllm_judge.log
      exit 3
    fi
  done
  say "  up after ${waited}s"
else
  say "=== judge already listening on :$PORT"
fi

# The judge sends up to three images per call (figure + previous frame +
# current frame), so verify the multimodal path before spending hours on it.
dexec "PYTHONPATH=src $PY -u scripts/vllm_smoke.py \
        --config ${CONFIGS#$REPO/}/bench_v4_svg_qwen3vl32b.yaml --backend qwen_judge" \
  2>&1 | tee -a "$LOG/judge.log" | grep -q "SMOKE OK" \
  || { say "FAILED: judge is up but unusable through the backend"; exit 4; }

run_suite() {   # model style suite ids
  local model=$1 style=$2 suite=$3 ids=$4
  local tag="judge_${model}_${style}_${suite}"
  [ -f "$STATE/$tag.done" ] && return 0
  dexec "PYTHONPATH=src $PY -u -m img_2_svg_pretraining.animatebench.run_eval \
           $suite --config src/img_2_svg_pretraining/pipeline/configs/bench_v4_svg_${model}.yaml \
           --style $style --only $ids --judge-backend qwen_judge \
           --evals-root ${EVALS#$REPO/}" \
    >>"$LOG/eval_${model}_${style}.log" 2>&1
  local rc=$?
  say "    $model/$style/$suite exit=$rc"
  [ $rc -eq 0 ] && touch "$STATE/$tag.done"
}

for PASS in cheap animation; do
  [ "$PASS" = animation ] && [ "$SKIP_ANIMATION" = 1 ] && { say "=== skipping animation"; break; }
  suites=$SUITES; [ "$PASS" = animation ] && suites=animation
  say "=== $PASS pass: $suites"

  for MODEL in $MODELS; do
    [ -f "$CONFIGS/bench_v4_svg_${MODEL}.yaml" ] || continue
    for STYLE in $STYLES; do
      ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py \
              --root data/animatebench_v4 --target svg --style "$STYLE")
      [ -z "$ids" ] && continue
      for suite in $suites; do run_suite "$MODEL" "$STYLE" "$suite" "$ids"; done
    done
  done
done

say "=== judging complete -- building the aggregate"
dexec "PYTHONPATH=src $PY -u -m img_2_svg_pretraining.animatebench.aggregate \
        --evals-root ${EVALS#$REPO/} --dataset data/animatebench_v4 \
        --target svg --scope both --format md csv json" \
  2>&1 | tail -60 | tee -a "$LOG/judge.log"
say "=== aggregate at $EVALS/aggregate.{md,csv,json}"
