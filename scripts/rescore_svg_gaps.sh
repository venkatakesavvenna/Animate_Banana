#!/usr/bin/env bash
# Re-score SVG cells whose eval records were deleted as stale.
#
# The generation sweep only evaluates the batch it just ran, so a cell whose
# artifacts already existed but whose records were removed afterwards is never
# revisited. This walks every style and re-runs the four suites; run_eval
# skips cells that already have valid records, so it costs only the gaps.
#
# Waits for the generation sweep first -- two processes writing the same
# record is a race with no winner worth having.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
CONFIG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg.yaml
LOG=$REPO/logs/bench_v3_svg; mkdir -p "$LOG"
cd "$REPO"; set -a; . "$REPO/.env"; set +a
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/rescore.log"; }

while pgrep -f "run_svg_styles.sh" >/dev/null; do sleep 60; done
say "=== generation sweep finished; filling eval gaps"

for style in progressive_reveal hopping_bounding_box sliding_bounding_box colour_pop alpha_masking; do
  ids=$(python3 scripts/bench_v3_styles.py --style "$style")
  [ -z "$ids" ] && continue
  for suite in stage1 xml sequence stage3; do
    docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$OPEN_ROUTER_KEY" \
      -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$CONTAINER" \
      bash -lc "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.animatebench.run_eval \
        $suite --config $CONFIG --style $style --only $ids \
        --judge-backend openrouter_flash" >>"$LOG/rescore_${style}.log" 2>&1
  done
  say "  $style done"
done
say "=== rescore complete"
