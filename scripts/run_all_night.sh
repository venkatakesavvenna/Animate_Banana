#!/usr/bin/env bash
# Sequential orchestrator: finish the ablations, then run the zs Gemini
# generation. Kimi judging of the zs output is a SEPARATE, free pass driven by
# scripts/zs_metrics_night.sh, which waits for this generation to finish.
#
# WHY SEQUENTIAL, not parallel: both stages bill the SAME OpenRouter key, and
# OpenRouter caps CONCURRENT spend -- exceeding it returns HTTP 402
# in_flight_budget_exhausted and the cell is lost (47 ablation cells died that
# way at 6 jobs x 12 concurrency). Running both at once would also make each
# stage's budget ceiling fire on the other's spend, since usage is shared.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
ABL_CEILING=${ABL_CEILING:-145}
ZS_CEILING=${ZS_CEILING:-190}
LOG=$REPO/logs/all_night; mkdir -p "$LOG"
PID=$LOG/run.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running as pid $(cat "$PID")"; exit 0
fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/run.log"; }
usage(){ local k; k=$(grep "^OPEN_ROUTER_KEY=" .env|cut -d= -f2-|tr -d '\r\n '); \
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $k" \
  | python3 -c "import json,sys;print(f\"{json.load(sys.stdin)['data'].get('usage',0):.2f}\")" 2>/dev/null; }

say "=== PHASE 1: ablations (ceiling \$$ABL_CEILING) | usage=\$$(usage)"
rm -f "$REPO/logs/ablation/resume.pid"
setsid nohup env CEILING=$ABL_CEILING "$REPO/scripts/ablation_budget_guard.sh" >/dev/null 2>&1 </dev/null &
setsid nohup "$REPO/scripts/ablation_resume.sh" > "$LOG/ablations.log" 2>&1 </dev/null &
sleep 60
while [ -f "$REPO/logs/ablation/resume.pid" ] && kill -0 "$(cat "$REPO/logs/ablation/resume.pid" 2>/dev/null)" 2>/dev/null; do
  sleep 300
done
say "=== PHASE 1 done | usage=\$$(usage)"
pkill -f ablation_budget_guard 2>/dev/null; true

say "=== PHASE 2: zs gemini generation, critics ON (ceiling \$$ZS_CEILING)"
JOBS=3 BATCH=8 CEILING=$ZS_CEILING bash "$REPO/scripts/run_zs_gemini.sh" >> "$LOG/zs_gen.log" 2>&1
say "=== PHASE 2 done | usage=\$$(usage)"
say "metrics: scripts/zs_metrics_night.sh picks this up (free, local Kimi)"
