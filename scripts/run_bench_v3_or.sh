#!/usr/bin/env bash
# Full v3 sweep on Gemini 3.7 Flash via OpenRouter: full pipeline + the four
# intermediate-stage suites, over every (sample, style) cell that has a
# reference to be scored against. This is run_bench_v3.sh's OpenRouter
# sibling -- same batching-by-sample and same GT-restricted style plan
# (scripts/bench_v3_styles.py), but none of the per-day-quota wait logic,
# because a paid endpoint has no daily cliff to sleep through.
#
# What it has INSTEAD is a spend ceiling, because paid billing fails in the
# opposite direction of free tier: free tier stops on its own (a 429) and
# costs nothing extra; paid billing does not stop on its own and a runaway
# loop keeps charging. STOP_AT defaults to $9.50 against a $10 balance,
# leaving headroom the credits API's own settlement lag might eat into --
# OpenRouter's /credits total_usage does not always reflect the very last
# call instantly.
#
# The animation tree is deliberately NOT run here either: ~70 judged calls
# per cell, several times this whole sweep's budget, and it is going to Qwen
# on local GPUs, not a paid API.
#
#   scripts/run_bench_v3_or.sh
#   STYLES=colour_pop scripts/run_bench_v3_or.sh
#   STOP_AT=5.00 scripts/run_bench_v3_or.sh
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
CONFIG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or.yaml
SUITES="stage1 xml sequence stage3"
BATCH=${BATCH:-4}
STYLES=${STYLES:-"progressive_reveal hopping_bounding_box sliding_bounding_box colour_pop alpha_masking"}
STOP_AT=${STOP_AT:-9.50}

LOGDIR=$REPO/logs/bench_v3_or
mkdir -p "$LOGDIR"

set -a; . "$REPO/.env"; set +a

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/driver.log"; }

dexec() {
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$OPEN_ROUTER_KEY" \
    "$CONTAINER" bash -lc "cd /code && PYTHONPATH=src $PY -u $*"
}

spend() {
  curl -s https://openrouter.ai/api/v1/credits \
    -H "Authorization: Bearer $OPEN_ROUTER_KEY" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['total_usage'])" \
    2>/dev/null || echo "999"    # fail closed: an unreadable balance halts the run
}

check_budget() {
  local s; s=$(spend)
  if python3 -c "exit(0 if float('$s') < $STOP_AT else 1)"; then
    say "  spend: \$$s (ceiling \$$STOP_AT)"
    return 0
  fi
  say "  STOPPING: spend \$$s has reached the \$$STOP_AT ceiling"
  return 1
}

say "=== bench v3 (OpenRouter, google/gemini-3.7-flash) start | batch=$BATCH | ceiling=\$$STOP_AT"
check_budget || exit 3

for style in $STYLES; do
  ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py --style "$style")
  [ -z "$ids" ] && { say "$style: no samples, skipping"; continue; }
  # shellcheck disable=SC2206
  arr=($ids)
  say "--- $style: ${#arr[@]} sample(s)"

  for ((i = 0; i < ${#arr[@]}; i += BATCH)); do
    batch="${arr[*]:i:BATCH}"
    tag="${style}_$((i / BATCH))"
    say "  batch $tag: $(echo "$batch" | wc -w) sample(s)"
    check_budget || exit 3

    dexec -m img_2_svg_pretraining.pipeline.run_pipeline all \
      --config "$CONFIG" --style "$style" --only $batch \
      >>"$LOGDIR/pipeline_$tag.log" 2>&1
    say "    pipeline exit=$?"

    for suite in $SUITES; do
      dexec -m img_2_svg_pretraining.animatebench.run_eval "$suite" \
        --config "$CONFIG" --style "$style" --only $batch \
        --judge-backend openrouter_flash \
        >>"$LOGDIR/eval_$tag.log" 2>&1
      say "    eval $suite exit=$?"
    done

    say "    scored so far: $(find "$REPO/data/animatebench_v3_cache/animatebench_v3/evals/bench_v3_or" \
      -name 'stage1.json' 2>/dev/null | wc -l) sample-cell(s)"
  done
done

say "=== rendering report"
dexec -m img_2_svg_pretraining.animatebench.run_eval report --config "$CONFIG" \
  >>"$LOGDIR/report.log" 2>&1
say "=== done | final spend: \$$(spend)"
