#!/usr/bin/env bash
# Gemini 3.7-Flash (critics ON) over the 193-sample zs set. GENERATION ONLY --
# metrics are a separate, free pass (scripts/zs_metrics_night.sh).
#
# WHY NOT run_zs_overnight.sh: that script serves the model locally first
# (serve_v5.sh + vllm_smoke.py) and resolves configs as bench_zs_svg_<M>.yaml.
# Gemini is an OpenRouter model with no local server and its config is
# bench_zs_or_gemini37flash.yaml, so both steps are wrong for it.
#
# STYLE IS PER SAMPLE. data/zs_style_map.json pins each sample to the one style
# its reference animation implements (98 progressive_reveal, 34 alpha_masking,
# 27 hopping, 20 colour_pop, 14 sliding). The config's `animation_style:
# progressive_reveal` is only a default -- running without --style would
# generate 95 samples in a style their reference never used, and every
# style-sensitive metric downstream would be comparing unlike things.
#
# CONCURRENCY. OpenRouter caps CONCURRENT spend, not total: exceeding it returns
# HTTP 402 in_flight_budget_exhausted and the cell is lost. In-flight is
# JOBS x max_concurrency. 47 ablation cells died at 6 x 12 = 72; 12 has run
# clean since. Keep JOBS x max_concurrency <= 12.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=${C:-img-2-svg-pretraining-singlenode-venkat.kesav}
PY=${PY:-/environments/img_2_svg_pretraining/bin/python}
CFG=${CFG:-src/img_2_svg_pretraining/pipeline/configs/bench_zs_or_gemini37flash.yaml}
STYLES=${STYLES:-"progressive_reveal alpha_masking hopping_bounding_box colour_pop sliding_bounding_box"}
JOBS=${JOBS:-3}
BATCH=${BATCH:-8}
FLOOR=${FLOOR:-6}              # stop when REMAINING credit falls below this
LOG=$REPO/logs/zs_gemini; mkdir -p "$LOG"

PID=$LOG/run.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running as pid $(cat "$PID")"; exit 0
fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/run.log"; }
usage(){ bash scripts/or_remaining.sh; }   # REMAINING credit
# THE KEY MUST BE PASSED INTO THE CONTAINER EXPLICITLY. /code/.env exists and is
# readable in there, but the backend reads OPEN_ROUTER_KEY from the ENVIRONMENT,
# not from the file -- without -e every call dies instantly with
#     AuthenticationError: 401 - Missing Authentication header
# and the runner walks the whole style map in 89 seconds spending $0, which in
# the log looks almost like a completed run. run_ablations.sh:56 does pass it;
# this script did not, and that cost a full night of generation.
KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')
[ -n "$KEY" ] || { echo "FATAL: no OPEN_ROUTER_KEY in .env"; exit 1; }
dexec(){ docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
         -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
         "$C" bash -lc "cd /code && PYTHONPATH=src $*"; }

say "=== zs gemini (critics ON) | jobs=$JOBS batch=$BATCH | remaining=\$$(usage)"
for STYLE in $STYLES; do
  mapfile -t ARR < <(python3 -c "
import json; sm=json.load(open('data/zs_style_map.json'))
print('\n'.join(s for s,st in sorted(sm.items()) if st=='$STYLE'))")
  [ "${#ARR[@]}" -eq 0 ] && continue
  say "== $STYLE : ${#ARR[@]} sample(s)"
  i=0
  while [ $i -lt ${#ARR[@]} ]; do
    U=$(usage)
    if [ -n "$U" ]; then
      awk "BEGIN{exit !($U <= $FLOOR)}" && { say "CREDIT FLOOR \$$FLOOR reached (remaining \$$U) -- stopping"; exit 0; }
    fi
    # RUN $JOBS BATCHES CONCURRENTLY. This loop was sequential in the first
    # version -- JOBS was read and never used -- so only one run_pipeline ran at
    # a time and in-flight sat at max_concurrency (4) instead of JOBS x 4 = 12.
    # Measured cost: 12 cells/hour, i.e. ~15h for 193 rather than ~5h.
    # Keep JOBS x max_concurrency <= 12: OpenRouter caps CONCURRENT spend and
    # returns 402 in_flight_budget_exhausted above it, losing the cell.
    running=0
    while [ $i -lt ${#ARR[@]} ] && [ $running -lt $JOBS ]; do
      batch=$(printf "%s " "${ARR[@]:$i:$BATCH}")
      tag="${STYLE}_$((i/BATCH))"
      say "   batch $tag: $(echo $batch|wc -w) sample(s)"
      ( dexec "$PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
                 all --config $CFG --style $STYLE --only $batch" \
          >>"$LOG/pipe_${STYLE}_$((i/BATCH)).log" 2>&1 \
        || say "   batch $tag returned nonzero (continuing)" ) &
      i=$((i+BATCH)); running=$((running+1))
    done
    wait
  done
done
say "=== generation done | remaining=\$$(usage)"
