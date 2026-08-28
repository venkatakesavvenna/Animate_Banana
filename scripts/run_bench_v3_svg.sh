#!/usr/bin/env bash
# SVG sweep: pipeline + the four intermediate suites, for one style.
#
# SCOPED TO progressive_reveal ON PURPOSE, for two reasons:
#
# 1. Budget. A measured SVG cell costs ~$0.24 end to end. All 47 remaining
#    cells would be ~$11, and the key had $4.08 left when this was written.
#    progressive_reveal is the largest group (16 cells, ~$3.8) and the one
#    the TikZ table is deepest on, so finishing it gives a complete,
#    like-for-like comparison rather than a scatter of half-covered styles.
#
# 2. The SVG designer prompt is STYLE-KEYED (`svg_designer.yaml#<style>`)
#    where the TikZ one is not, and its keys do not match the pipeline's
#    style names -- the file has `hopping_box` / `sliding_box` where the
#    pipeline says `hopping_bounding_box` / `sliding_bounding_box`. Running
#    another style therefore needs a per-style config or a prompt override,
#    not just `--style`. Worth doing, but not silently and not on a budget
#    that cannot finish it.
#
# The spend ceiling is checked before every batch and the run stops cleanly
# when it is reached: paid billing does not fail safe on its own.
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
CONFIG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg.yaml
STYLE=${STYLE:-progressive_reveal}
SUITES="stage1 xml sequence stage3"
BATCH=${BATCH:-2}
STOP_AT=${STOP_AT:-9.60}

LOGDIR=$REPO/logs/bench_v3_svg
mkdir -p "$LOGDIR"
set -a; . "$REPO/.env"; set +a

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/driver.log"; }

dexec() {
  docker exec -u "$(id -u):$(id -g)" \
    -e OPEN_ROUTER_KEY="$OPEN_ROUTER_KEY" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
    "$CONTAINER" bash -lc "cd /code && PYTHONPATH=src $PY -u $*"
}

spend() {
  curl -s https://openrouter.ai/api/v1/credits \
    -H "Authorization: Bearer $OPEN_ROUTER_KEY" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['total_usage'])" \
    2>/dev/null || echo "999"        # fail closed: unreadable balance halts
}

check_budget() {
  local s; s=$(spend)
  if python3 -c "exit(0 if float('$s') < $STOP_AT else 1)"; then
    say "  spend \$$s / ceiling \$$STOP_AT"; return 0
  fi
  say "  STOPPING: spend \$$s reached the \$$STOP_AT ceiling"; return 1
}

ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py --style "$STYLE")
# shellcheck disable=SC2206
arr=($ids)
say "=== svg sweep | style=$STYLE | ${#arr[@]} cell(s) | ceiling \$$STOP_AT"

for ((i = 0; i < ${#arr[@]}; i += BATCH)); do
  batch="${arr[*]:i:BATCH}"
  tag="${STYLE}_$((i / BATCH))"
  say "  batch $tag: $(echo "$batch" | wc -w) sample(s)"
  check_budget || exit 3

  dexec -m img_2_svg_pretraining.pipeline.run_pipeline all \
    --config "$CONFIG" --style "$STYLE" --only $batch \
    >>"$LOGDIR/pipeline_$tag.log" 2>&1
  say "    pipeline exit=$?"

  for suite in $SUITES; do
    dexec -m img_2_svg_pretraining.animatebench.run_eval "$suite" \
      --config "$CONFIG" --style "$STYLE" --only $batch \
      --judge-backend openrouter_flash >>"$LOGDIR/eval_$tag.log" 2>&1
  done
  say "    evals done | scored: $(find "$REPO/data/animatebench_v3_cache/animatebench_v3/evals/bench_v3_or_svg" \
    -name stage1.json 2>/dev/null | wc -l)/${#arr[@]}"
done

say "=== done | final spend \$$(spend)"
