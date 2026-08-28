#!/usr/bin/env bash
# Remaining SVG styles, in the order the budget can actually reach.
#
# STYLES defaults to the two whose designer-prompt keys already match the
# pipeline's style names AND that fit the money left. The bounding-box styles
# work too now (designer._resolve_prompt maps hopping_box / sliding_box) --
# they are simply not affordable yet, so they are opt-in:
#
#   STYLES="hopping_bounding_box sliding_bounding_box" scripts/run_svg_styles.sh
#
# The ceiling is checked before every batch. Paid billing does not fail safe
# on its own, and a run that dies mid-cell leaves a half-scored style.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
CONFIG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg.yaml
STYLES=${STYLES:-"colour_pop alpha_masking"}
SUITES="stage1 xml sequence stage3"
BATCH=${BATCH:-2}
STOP_AT=${STOP_AT:-9.85}
LOG=$REPO/logs/bench_v3_svg
mkdir -p "$LOG"
set -a; . "$REPO/.env"; set +a

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/styles.log"; }
dexec() {
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$OPEN_ROUTER_KEY" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
    "$CONTAINER" bash -lc "cd /code && PYTHONPATH=src $PY -u $*"
}
spend() {
  curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPEN_ROUTER_KEY" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['total_usage'])" 2>/dev/null \
    || echo "999"
}
budget_ok() {
  local s; s=$(spend)
  python3 -c "exit(0 if float('$s') < $STOP_AT else 1)" && { say "  spend \$$s / \$$STOP_AT"; return 0; }
  say "  STOPPING: spend \$$s reached the \$$STOP_AT ceiling"; return 1
}

say "=== svg styles: $STYLES | ceiling \$$STOP_AT"
for style in $STYLES; do
  ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py --style "$style")
  [ -z "$ids" ] && { say "$style: no samples"; continue; }
  # shellcheck disable=SC2206
  arr=($ids); say "--- $style: ${#arr[@]} cell(s)"
  for ((i=0; i<${#arr[@]}; i+=BATCH)); do
    batch="${arr[*]:i:BATCH}"; tag="${style}_$((i/BATCH))"
    budget_ok || exit 3
    say "  batch $tag"
    dexec -m img_2_svg_pretraining.pipeline.run_pipeline all \
      --config "$CONFIG" --style "$style" --only $batch >>"$LOG/pipe_$tag.log" 2>&1
    say "    pipeline exit=$?"
    for suite in $SUITES; do
      dexec -m img_2_svg_pretraining.animatebench.run_eval "$suite" \
        --config "$CONFIG" --style "$style" --only $batch \
        --judge-backend openrouter_flash >>"$LOG/eval_$tag.log" 2>&1
    done
    say "    scored: $(find "$REPO/data/animatebench_v3_cache/animatebench_v3/evals/bench_v3_or_svg" \
      -name stage1.json 2>/dev/null | wc -l) cell(s) total"
  done
done
say "=== done | final spend \$$(spend)"
