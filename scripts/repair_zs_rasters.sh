#!/usr/bin/env bash
# Re-run the raster splice and everything downstream for samples whose
# detections.json holds only an error (38 of them a 402 from 09-02 when credit
# ran out). Those failures were CACHED as if they were results: stage 1b saw a
# detections.json, logged "skipped", and produced animations with no crops.
#
# Every derived artifact for these samples is evicted first (see the eviction
# in data/zs_evict_plan.json) -- rebuilding 1b without evicting code_final,
# xml, sequence*, animation* and exports would leave the new splice unread.
# NOTE sequence artifacts are named <sample>.roundN.json: a matcher that strips
# only the extension misses all of them.
#
# Style is per sample from zs_style_map.json; the config's animation_style is
# only a default. Key must be passed with -e: the backend reads it from the
# environment, not /code/.env. JOBS x max_concurrency(4) <= 12.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=${C:-img-2-svg-pretraining-singlenode-venkat.kesav}
PY=${PY:-/environments/img_2_svg_pretraining/bin/python}
CFG=src/img_2_svg_pretraining/pipeline/configs/bench_zs_or_gemini37flash.yaml
LIST=${LIST:-$REPO/data/zs_raster_repair.txt}
JOBS=${JOBS:-3}; BATCH=${BATCH:-6}; FLOOR=${FLOOR:-5}
LOG=$REPO/logs/zs_repair; mkdir -p "$LOG"
PID=$LOG/run.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT
KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')
[ -n "$KEY" ] || { echo "FATAL: no key"; exit 1; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/run.log"; }
rem(){ bash scripts/or_remaining.sh; }
dexec(){ docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
         -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
         "$C" bash -lc "cd /code && PYTHONPATH=src $*"; }
say "=== zs raster repair | remaining=\$$(rem)"
for STYLE in progressive_reveal alpha_masking hopping_bounding_box colour_pop sliding_bounding_box; do
  mapfile -t ARR < <(LIST="$LIST" STYLE="$STYLE" python3 -c "
import json,os
sm=json.load(open('data/zs_style_map.json'))
todo=[s.strip() for s in open(os.environ['LIST']) if s.strip()]
print('\n'.join(s for s in todo if sm.get(s)==os.environ['STYLE']))")
  [ "${#ARR[@]}" -eq 0 ] && continue
  say "== $STYLE : ${#ARR[@]} sample(s)"
  i=0
  while [ $i -lt ${#ARR[@]} ]; do
    U=$(rem); [ -n "$U" ] && awk "BEGIN{exit !($U <= $FLOOR)}" && { say "FLOOR \$$FLOOR hit (remaining \$$U)"; exit 0; }
    running=0
    while [ $i -lt ${#ARR[@]} ] && [ $running -lt $JOBS ]; do
      b=$(printf "%s " "${ARR[@]:$i:$BATCH}")
      say "   $STYLE batch $((i/BATCH)): $(echo $b|wc -w)"
      ( dexec "$PY -u -m img_2_svg_pretraining.pipeline.run_pipeline all --from integrate-rasters \
                 --config $CFG --style $STYLE --only $b" \
          >>"$LOG/pipe_${STYLE}_$((i/BATCH)).log" 2>&1 || say "   batch $((i/BATCH)) nonzero" ) &
      i=$((i+BATCH)); running=$((running+1))
    done
    wait
  done
done
say "=== repair done | remaining=\$$(rem)"
