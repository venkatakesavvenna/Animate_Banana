#!/usr/bin/env bash
# Drive the v3 bench: full pipeline + the four intermediate-stage suites,
# over every (sample, style) cell that has a reference to be scored against.
#
# BATCHED BY SAMPLE, NOT BY STAGE
# -------------------------------
# `run_pipeline` is agent-major: each agent runs over the whole sample list
# before the next one starts. Handed all 16 samples of a style, it advances
# them in lockstep, so an interrupted run leaves 16 samples half-done and zero
# samples finished. Batching trades a little parallelism for the property that
# actually matters -- at any moment the finished batches are finished all the
# way through, pipeline and metrics both. Styles run largest-coverage-first,
# so the earliest batches cover the most distinct samples.
#
# THE BINDING CONSTRAINT IS A PER-DAY QUOTA
# -----------------------------------------
#   quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20
#
# Twenty requests per key per day for gemini-3.6-flash. A spent key is not
# slow, it is done until reset, so the pool must be pruned to keys that can
# still answer -- otherwise KeyRing (correctly built for per-minute quotas)
# clears its exhausted marks when everything is marked and the run spends
# minutes re-uploading figures to keys that cannot reply. PIPELINE_KEY_FILE
# points at the live subset, refreshed between batches.
#
# Out of quota, the driver WAITS rather than failing: everything here is
# resumable, so sleeping until the daily reset costs nothing and picks up
# where it stopped.
#
# The animation tree is deliberately NOT run: ~70 judged calls per cell, and
# it is going to Qwen later. `run_eval all` would include it, so the four
# suites are named explicitly.
#
#   scripts/run_bench_v3.sh
#   STYLES=progressive_reveal scripts/run_bench_v3.sh
#   BATCH=2 scripts/run_bench_v3.sh
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
CONFIG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_gemini.yaml
KEYFILE=$REPO/api_keys.live.csv
SUITES="stage1 xml sequence stage3"
# Matched to the backend's max_concurrency: a larger batch just queues, and
# every extra sample in flight is one more left half-finished if this stops.
BATCH=${BATCH:-4}
STYLES=${STYLES:-"progressive_reveal hopping_bounding_box sliding_bounding_box colour_pop alpha_masking"}
WAIT_MIN=${WAIT_MIN:-30}      # how long to sleep when the pool is dry
MAX_WAITS=${MAX_WAITS:-48}    # give up after 24h of waiting

LOGDIR=$REPO/logs/bench_v3
mkdir -p "$LOGDIR"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/driver.log"; }

# Run as the invoking user, not root: the default pipeline/cache tree is
# root-owned from an earlier container run and cannot be written by anyone
# else, which is the failure this avoids repeating.
#
# `-u` is not optional. Python block-buffers stdout to a file, so the first
# attempt at this run logged nothing for 30 minutes while landing no calls --
# the stall was only visible by stat'ing the cache.
dexec() {
  docker exec -u "$(id -u):$(id -g)" -e PIPELINE_KEY_FILE="$KEYFILE" "$CONTAINER" \
    bash -lc "cd /code && PYTHONPATH=src $PY -u $*"
}

# Block until at least one key can answer. Refresh costs one request per key
# probed, so it re-probes the live subset and only widens when nearly dry.
ensure_keys() {
  local waits=0
  while true; do
    if (cd "$REPO" && python3 scripts/gemini_keys.py refresh >>"$LOGDIR/keys.log" 2>&1); then
      say "  keys: $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['live'])" \
        "$REPO/logs/key_status.json" 2>/dev/null || echo '?') live"
      return 0
    fi
    waits=$((waits + 1))
    if [ "$waits" -gt "$MAX_WAITS" ]; then
      say "  keys: dry for $((WAIT_MIN * MAX_WAITS)) minutes, giving up"
      return 1
    fi
    say "  keys: pool is dry (per-day quota); sleeping ${WAIT_MIN}m [$waits/$MAX_WAITS]"
    sleep $((WAIT_MIN * 60))
  done
}

say "=== bench v3 start | batch=$BATCH | styles: $STYLES"
ensure_keys || exit 3

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
    ensure_keys || exit 3

    dexec -m img_2_svg_pretraining.pipeline.run_pipeline all \
      --config "$CONFIG" --style "$style" --only $batch \
      >>"$LOGDIR/pipeline_$tag.log" 2>&1
    say "    pipeline exit=$?"

    for suite in $SUITES; do
      dexec -m img_2_svg_pretraining.animatebench.run_eval "$suite" \
        --config "$CONFIG" --style "$style" --only $batch \
        >>"$LOGDIR/eval_$tag.log" 2>&1
      say "    eval $suite exit=$?"
    done

    say "    scored so far: $(find "$REPO/data/animatebench_v3_cache/animatebench_v3/evals" \
      -name 'stage1.json' 2>/dev/null | wc -l) sample-cell(s)"
  done
done

say "=== rendering report"
dexec -m img_2_svg_pretraining.animatebench.run_eval report --config "$CONFIG" \
  >>"$LOGDIR/report.log" 2>&1
say "=== done"
