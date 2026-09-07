#!/usr/bin/env bash
# Gemini 3.7 Flash over the 193-sample zero-shot set, via OpenRouter,
# WITH ALL THREE CRITICS ACTIVE. Generation only -- judging is separate.
#
# PARALLEL ACROSS STYLES, and that is the whole point of this script.
#
# The three critics call `.generate()` once per sample and never
# `generate_batch`, so within one pipeline process they run STRICTLY
# SEQUENTIALLY -- the same defect that made the v5 study 26x slower with
# critics on than off. That is a property of the critic drivers, not of the
# backend, so raising max_concurrency does not touch it.
#
# This route needs no GPU, so the fix is process-level: one worker per style,
# five in flight. Each worker owns a disjoint set of samples (the style map is
# a partition -- every sample has exactly one style), so they cannot collide
# on an artifact path.
#
# RESUMABLE: every stage skips work whose output already exists, so re-running
# is a resume; the .done markers only avoid re-walking finished batches.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=animatebanana-v5
PY=/environments/img_2_svg_pretraining/bin/python
CFG=src/img_2_svg_pretraining/pipeline/configs/bench_zs_or_gemini37flash.yaml
LOG=$REPO/logs/bench_zs_gemini; STATE=$LOG/state
mkdir -p "$LOG" "$STATE"
set -a; . "$REPO/.env"; set +a
BATCH=${BATCH:-12}
STYLES=${STYLES:-"progressive_reveal alpha_masking hopping_bounding_box colour_pop sliding_bounding_box"}

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/gemini.log"; }

run_style() {
  local STYLE="$1"
  local ids
  ids=$(cd "$REPO" && python3 -c "
import json
sm = json.load(open('data/zs_style_map.json'))
print(' '.join(s for s, st in sorted(sm.items()) if st == '$STYLE'))")
  local arr=($ids)
  [ ${#arr[@]} -eq 0 ] && return 0
  say "  [$STYLE] ${#arr[@]} cell(s)"
  for ((i=0;i<${#arr[@]};i+=BATCH)); do
    local tag="gemini_${STYLE}_$((i/BATCH))"
    [ -f "$STATE/$tag.done" ] && continue
    docker exec -u "$(id -u):$(id -g)" \
      -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
      -e OPEN_ROUTER_KEY="$OPEN_ROUTER_KEY" "$C" bash -lc \
      "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
         all --config $CFG --style $STYLE --only ${arr[*]:i:BATCH}" \
      >>"$LOG/pipe_gemini_${STYLE}.log" 2>&1
    local rc=$?
    say "    [$STYLE] batch $((i/BATCH)) exit=$rc"
    [ $rc -eq 0 ] && touch "$STATE/$tag.done"
  done
  say "  [$STYLE] done"
}

say "=== gemini 3.7 flash | 193-sample zero-shot | CRITICS ON | styles in parallel"
pids=()
for STYLE in $STYLES; do
  run_style "$STYLE" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
say "=== gemini zero-shot complete"
