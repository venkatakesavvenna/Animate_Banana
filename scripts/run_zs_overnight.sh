#!/usr/bin/env bash
# Zero-shot sweep: 4 open models x 193 samples (animatebench_zs), each sample
# at the ONE style its reference animation html names (data/zs_style_map.json).
#
# Clone of run_v5_overnight.sh with three deltas:
#   * configs bench_zs_svg_* (dataset animatebench_zs, its own cache_root)
#   * cells come from the style map, NOT bench_v3_styles.py -- that probe
#     selects cells with GT reference sequences, and this dataset is judged
#     zero-shot by SSS/GPS/NAS which need no GT; filtering by references
#     would silently drop 101 of 193 samples
#   * no kimi row (kimi is the judge, and its 2-node serve owns these GPUs
#     only after generation is done)
#
# GENERATION ONLY -- judging is scripts/run_v5_judge.sh with
# CELLS=data/zs_judge_cells.json afterwards. Re-running resumes.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=animatebanana-v5
PY=/environments/img_2_svg_pretraining/bin/python
CONFIGS=src/img_2_svg_pretraining/pipeline/configs
LOG=$REPO/logs/bench_zs; STATE=$LOG/state
mkdir -p "$LOG" "$STATE"

MODELS=${MODELS:-"gemma4_31b qwen38_27b glm46v qwen3vl235b"}
STYLES=${STYLES:-"alpha_masking colour_pop progressive_reveal hopping_bounding_box sliding_bounding_box"}
BATCH=${BATCH:-24}

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/overnight.log"; }
dexec(){ docker exec -u "$(id -u):$(id -g)" -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc "cd /code && PYTHONPATH=src $*"; }

say "=== zs overnight (GENERATION ONLY) | models: $MODELS | batch=$BATCH"

for M in $MODELS; do
  CFG=$CONFIGS/bench_zs_svg_${M}.yaml
  [ -f "$REPO/$CFG" ] || { say "!! no config for $M"; continue; }
  [ -f "$STATE/${M}.done" ] && { say "== $M already complete"; continue; }

  say "== $M : serving"
  if ! bash "$REPO/scripts/serve_v5.sh" "$M" >>"$LOG/serve_${M}.log" 2>&1; then
    say "   SERVE FAILED -- see logs/bench_zs/serve_${M}.log"
    touch "$STATE/${M}.failed"; continue
  fi
  if ! dexec "$PY -u scripts/vllm_smoke.py --config $CFG --backend served" 2>&1 \
       | tee -a "$LOG/serve_${M}.log" | grep -q "SMOKE OK"; then
    say "   SMOKE FAILED"; touch "$STATE/${M}.failed"
    bash "$REPO/scripts/serve_v5.sh" stop >/dev/null 2>&1; continue
  fi
  say "   smoke ok"

  for STYLE in $STYLES; do
    ids=$(cd "$REPO" && python3 -c "
import json
sm = json.load(open('data/zs_style_map.json'))
print(' '.join(s for s, st in sorted(sm.items()) if st == '$STYLE'))")
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
