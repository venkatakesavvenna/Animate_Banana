#!/usr/bin/env bash
# Overnight generation for one (model, set) pair, natively on the host.
#
#   MODEL=dsv4fv SET=zs bash scripts/run_gen_night.sh
#
# NO DOCKER. run_zs_gemini.sh execs into
# `img-2-svg-pretraining-singlenode-venkat.kesav`, which no longer exists on
# this host (CLAUDE.md: "Docker is no longer required"). Everything below runs
# in the host python with PYTHONPATH=src, which is how the pipeline CLI and
# Playwright are both reachable now.
#
# STYLE IS PER SAMPLE. Each set has a style map pinning every sample to the one
# style its reference implements. The config's `animation_style` is only a
# DEFAULT -- running without --style would generate most samples in a style
# their reference never used, and every style-sensitive metric downstream would
# then compare unlike things.
#
# CONCURRENCY. OpenRouter caps CONCURRENT spend, not total: above it the call
# returns HTTP 402 in_flight_budget_exhausted and the cell is lost. In-flight is
# JOBS x backend max_concurrency (4 in these configs). 47 ablation cells died at
# 6 x 12 = 72. Keep JOBS x max_concurrency <= 12, so JOBS=3.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
export PYTHONPATH=src
export PLAYWRIGHT_BROWSERS_PATH=/fsxvision_new/venkat.kesav/environments/playwright-browsers

MODEL=${MODEL:?set MODEL (dsv4fv|kimi_k26)}
SET=${SET:?set SET (zs|v6|abl)}
JOBS=${JOBS:-3}
BATCH=${BATCH:-8}
FLOOR=${FLOOR:-4}                 # stop when REMAINING credit falls below this
CFG=src/img_2_svg_pretraining/pipeline/configs/gen_${MODEL}_${SET}.yaml
[ -f "$CFG" ] || { echo "FATAL: no config $CFG"; exit 1; }

case "$SET" in
  zs)  MAP=data/zs_style_map.json ;;
  v6)  MAP=data/v6_style_map.json ;;
  abl) MAP=data/abl_style_map.json ;;
  *)   echo "FATAL: unknown SET $SET"; exit 1 ;;
esac
[ -f "$MAP" ] || { echo "FATAL: no style map $MAP"; exit 1; }

DOCKER_C=${DOCKER_C:-img-2-svg-pretraining-singlenode-venkat.kesav}
CPY=${CPY:-/environments/img_2_svg_pretraining/bin/python}
docker exec "$DOCKER_C" true 2>/dev/null || { echo "FATAL: container $DOCKER_C not running (export needs it)"; exit 1; }

LOG=$REPO/logs/gen_${MODEL}_${SET}; mkdir -p "$LOG"
PID=$LOG/run.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running as pid $(cat "$PID")"; exit 0
fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/run.log"; }
usage(){ bash scripts/or_remaining.sh 2>/dev/null; }

# THE KEY MUST BE IN THE ENVIRONMENT. The backend reads OPEN_ROUTER_KEY from the
# environment, never from .env directly; without this every call fails instantly
# with "AuthenticationError: 401 - Missing Authentication header" and the runner
# walks the whole style map in ~90s spending $0, which in the log looks almost
# exactly like a completed run. That cost a full night once.
export OPEN_ROUTER_KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')
[ -n "$OPEN_ROUTER_KEY" ] || { echo "FATAL: no OPEN_ROUTER_KEY in .env"; exit 1; }

say "=== gen $MODEL/$SET | cfg=$(basename $CFG) jobs=$JOBS batch=$BATCH | remaining=\$$(usage)"

for STYLE in progressive_reveal alpha_masking hopping_bounding_box colour_pop sliding_bounding_box; do
  mapfile -t ARR < <(python3 -c "
import json; sm=json.load(open('$MAP'))
print('\n'.join(s for s,st in sorted(sm.items()) if st=='$STYLE'))")
  [ "${#ARR[@]}" -eq 0 ] && continue
  say "== $STYLE : ${#ARR[@]} sample(s)"
  i=0
  while [ $i -lt ${#ARR[@]} ]; do
    U=$(usage)
    if [ -n "$U" ]; then
      awk "BEGIN{exit !($U <= $FLOOR)}" && { say "CREDIT FLOOR \$$FLOOR reached (remaining \$$U) -- stopping"; exit 0; }
    fi
    running=0
    while [ $i -lt ${#ARR[@]} ] && [ $running -lt $JOBS ]; do
      batch=$(printf "%s " "${ARR[@]:$i:$BATCH}")
      tag="${STYLE}_$((i/BATCH))"
      say "   batch $tag: $(echo $batch|wc -w) sample(s)"
      # GENERATE ON THE HOST, EXPORT IN THE CONTAINER.
      #
      # `--to critique-animation` stops before the exporter: host chromium
      # cannot start (`libatk-1.0.so.0` missing, and there is no sudo here), so
      # an in-process export fails every cell after the model work is already
      # paid for. The container has playwright and mounts the repo at /code.
      ( python3 -u -m img_2_svg_pretraining.pipeline.run_pipeline \
            all --config "$CFG" --style "$STYLE" --only $batch \
            --to critique-animation \
          >>"$LOG/pipe_${tag}.log" 2>&1 \
        || say "   batch $tag generation returned nonzero (continuing)"
        timeout 3600 docker exec -u "$(id -u):$(id -g)" \
            -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$DOCKER_C" \
            bash -lc "cd /code && PYTHONPATH=src $CPY -u -m \
              img_2_svg_pretraining.pipeline.run_pipeline export \
              --config $CFG --style $STYLE --only $batch" \
          >>"$LOG/export_${tag}.log" 2>&1 \
        || say "   batch $tag export returned nonzero (continuing)" ) &
      i=$((i+BATCH)); running=$((running+1))
    done
    wait
  done
done
say "=== generation done | remaining=\$$(usage)"
