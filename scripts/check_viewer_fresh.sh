#!/usr/bin/env bash
# Hourly freshness check for the study viewer on :8606.
#
# WHAT ACTUALLY GOES STALE. `_metrics()` reads each eval record from disk on
# every request, so newly-judged cells appear on refresh with no restart. Two
# things are fixed at startup and do NOT: the pinned --cells list and the
# --configs model list. So the only real staleness is a cell that now has data
# but is not in the viewer's list, or a viewer that has died.
#
# Restarting on every new record would be worse than useless -- it drops the
# reader's open page mid-session to load data that was already visible.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
PORT="${PORT:-8606}"
TS=$(date '+%Y-%m-%d %H:%M:%S')

up=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "localhost:$PORT/api/index" || echo 000)
if [ "$up" != "200" ]; then
  echo "[$TS] viewer DOWN (http $up) -- restarting"
  ./scripts/run_study_viewer.sh "$PORT" >/dev/null 2>&1
  sleep 12
  up=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "localhost:$PORT/api/index" || echo 000)
  echo "[$TS] restart -> http $up"
  exit 0
fi

# Cells the viewer is pinned to, vs cells that now have a scored record.
served=$(curl -s --max-time 20 "localhost:$PORT/api/index" \
  | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('cells') or []))")
scored=$(python3 - <<'PY'
import json, glob
seen=set()
for f in glob.glob("data/animatebench_v3_cache/*/evals_study/*/*/*/animation.json"):
    p=f.split("/evals_study/")[1].split("/")
    if p[0]=="reference": continue
    d=json.load(open(f))
    if any(d.get(k) is not None for k in ("vfs_band","sss")):
        seen.add((p[1],p[2]))
print(len(seen))
PY
)
listed=$(grep -c ":" data/cells38.txt)
rows=$(curl -s --max-time 60 "localhost:$PORT/api/overview" \
  | python3 -c "import json,sys; print(len(json.load(sys.stdin)['rows']))" 2>/dev/null || echo "?")

echo "[$TS] viewer OK | pinned=$served/$listed cells | distinct scored cells=$scored | overview rows=$rows"

# A cell with data that the viewer was never told about is the one case a
# restart genuinely fixes.
if [ "$served" -lt "$listed" ]; then
  echo "[$TS] pinned list is short of the study list -- restarting to pick it up"
  ./scripts/run_study_viewer.sh "$PORT" >/dev/null 2>&1
fi
