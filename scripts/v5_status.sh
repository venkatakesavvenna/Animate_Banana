#!/usr/bin/env bash
# One-line health snapshot of the overnight judging run.
#
# LIVENESS IS THE LISTENING PORT, nothing else. Every other signal lied here:
#   docker exec ... ps  -> animatebanana-mn runs --pid host, so this reads the
#                          wrong namespace and reports 0 for a live server
#   pgrep -fc vllm...   -> matches its own shell, so it reports >0 forever
#   curl /health        -> times out while the engine is merely busy
# `ss -lnt | grep :8011` is true exactly when the server can accept work.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
N0=10.20.238.24
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15"
done=$(find data/animatebench_v5*cache -path '*/evals/*' -name animation.json 2>/dev/null|wc -l)
scored=$(python3 - <<'PY' 2>/dev/null
import json,glob
n=0
for f in glob.glob("data/animatebench_v5*cache/animatebench_v5/evals/*/*/*/animation.json"):
    try:
        d=json.load(open(f))
    except Exception:
        continue
    if d.get("sss") is not None or d.get("nas") is not None: n+=1
print(n)
PY
)
sup=$(pgrep -fc v5_judge_night.sh 2>/dev/null || echo 0)
rem=$($SSH $N0 'p=$(ss -lnt 2>/dev/null | grep -c ":8011")
  if [ "$p" -gt 0 ]; then h=$(curl -sf -m 25 http://127.0.0.1:8011/health >/dev/null 2>&1 && echo UP || echo BUSY); else h=DOWN; fi
  w=$(docker exec animatebanana-v5 bash -c "ps -eo cmd=|grep -c \"[r]un_eval\"" 2>/dev/null || echo ?)
  v=$(curl -s -o /dev/null -w "%{http_code}" -m 8 http://127.0.0.1:8601/ 2>/dev/null)
  echo "server=$h workers=$w viewer=$v"' 2>/dev/null|tail -1)
echo "[$(date '+%m-%d %H:%M')] records=$done scored=${scored:-?}/432 supervisor=$sup $rem"
