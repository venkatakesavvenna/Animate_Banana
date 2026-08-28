#!/usr/bin/env bash
# One SVG cell per style -- breadth before depth, so a style that fails
# structurally is found on cell 1 rather than cell 7.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
CONFIG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg.yaml
STOP_AT=${STOP_AT:-9.85}
LOG=$REPO/logs/bench_v3_svg; mkdir -p "$LOG"
set -a; . "$REPO/.env"; set +a
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/pilot.log"; }
spend() { curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPEN_ROUTER_KEY" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['total_usage'])" 2>/dev/null || echo 999; }
dexec() { docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$OPEN_ROUTER_KEY" \
  -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$CONTAINER" \
  bash -lc "cd /code && PYTHONPATH=src $PY -u $*"; }

say "=== svg pilot: one cell per style"
while read -r style sample; do
  [ -z "$style" ] && continue
  s=$(spend)
  if ! python3 -c "exit(0 if float('$s') < $STOP_AT else 1)"; then
    say "STOPPING at \$$s (ceiling \$$STOP_AT)"; exit 3; fi
  say "--- $style / $sample   (spend \$$s)"
  dexec -m img_2_svg_pretraining.pipeline.run_pipeline all \
    --config "$CONFIG" --style "$style" --only "$sample" >>"$LOG/pilot_$style.log" 2>&1
  say "    pipeline exit=$?"
  for suite in stage1 xml sequence stage3; do
    dexec -m img_2_svg_pretraining.animatebench.run_eval "$suite" \
      --config "$CONFIG" --style "$style" --only "$sample" \
      --judge-backend openrouter_flash >>"$LOG/pilot_$style.log" 2>&1
  done
  say "    evals done"
done < /tmp/svg_pilot.txt
say "=== pilot done | spend \$$(spend)"
