#!/usr/bin/env bash
# v5 overnight sweep: every working model over all 91 measurable SVG cells.
#
#   scripts/run_v5_overnight.sh
#   MODELS="qwen38_27b" BATCH=8 scripts/run_v5_overnight.sh
#
# RESUMABILITY. Artifacts are keyed by cache lineage on disk and every stage
# skips work whose output already exists, so RE-RUNNING THIS SCRIPT IS ALREADY
# A RESUME. The .done markers below only avoid re-walking finished batches;
# they are an optimisation, not the correctness mechanism. That matters here
# because an overnight run WILL be interrupted -- by an OOM, a shared-node
# eviction, or a model that never loads -- and the recovery must not be "start
# again".
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=animatebanana-v5
PY=/environments/img_2_svg_pretraining/bin/python
CONFIGS=src/img_2_svg_pretraining/pipeline/configs
LOG=$REPO/logs/bench_v5; STATE=$LOG/state
mkdir -p "$LOG" "$STATE"

# Order: cheapest and most-proven first, so the night yields complete rows for
# the models we know work before spending hours on the ones that may not.
MODELS=${MODELS:-"gemma4_31b qwen38_27b glm46v qwen3vl235b kimi_k26"}
STYLES=${STYLES:-"alpha_masking colour_pop progressive_reveal hopping_bounding_box sliding_bounding_box"}
BATCH=${BATCH:-24}

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/overnight.log"; }
# -u: without it every artifact lands root-owned and there is no passwordless
# sudo on this node to chown it back.
dexec(){ docker exec -u "$(id -u):$(id -g)" -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc "cd /code && PYTHONPATH=src $*"; }

# GENERATION ONLY. No run_eval, no judge, no metrics anywhere in this script --
# the morning deliverable is animations on disk. Scoring is a separate pass that
# can run later against these artifacts.
say "=== v5 overnight (GENERATION ONLY) | models: $MODELS | batch=$BATCH"

for M in $MODELS; do
  CFG=$CONFIGS/bench_v5_svg_${M}.yaml
  [ -f "$REPO/$CFG" ] || { say "!! no config for $M"; continue; }
  [ -f "$STATE/${M}.done" ] && { say "== $M already complete"; continue; }

  say "== $M : serving"
  if ! bash "$REPO/scripts/serve_v5.sh" "$M" >>"$LOG/serve_${M}.log" 2>&1; then
    say "   SERVE FAILED -- see logs/bench_v5/serve_${M}.log"
    touch "$STATE/${M}.failed"; continue
  fi
  # /health is equally true of a server whose multimodal path is broken, and
  # every stage of this pipeline sends an image.
  if ! dexec "$PY -u scripts/vllm_smoke.py --config $CFG --backend served" 2>&1 \
       | tee -a "$LOG/serve_${M}.log" | grep -q "SMOKE OK"; then
    say "   SMOKE FAILED"; touch "$STATE/${M}.failed"
    bash "$REPO/scripts/serve_v5.sh" stop >/dev/null 2>&1; continue
  fi
  say "   smoke ok"

  for STYLE in $STYLES; do
    ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py \
            --root data/animatebench_v5 --target svg --style "$STYLE" 2>/dev/null)
    arr=($ids); [ ${#arr[@]} -eq 0 ] && continue
    say "   $STYLE: ${#arr[@]} cell(s)"
    for ((i=0;i<${#arr[@]};i+=BATCH)); do
      tag="${M}_${STYLE}_$((i/BATCH))"
      [ -f "$STATE/$tag.done" ] && continue
      batch="${arr[*]:i:BATCH}"
      dexec "$PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
               all --config $CFG --style $STYLE --only $batch" \
        >>"$LOG/pipe_${M}_${STYLE}.log" 2>&1
      rc=$?
      say "     batch $((i/BATCH)) exit=$rc"
      [ $rc -eq 0 ] && touch "$STATE/$tag.done"
    done
  done
  touch "$STATE/${M}.done"; say "== $M done"
  bash "$REPO/scripts/serve_v5.sh" stop >/dev/null 2>&1
done
say "=== sweep complete"
