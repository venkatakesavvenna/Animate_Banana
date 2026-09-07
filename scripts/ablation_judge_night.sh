#!/usr/bin/env bash
# Unattended supervisor for the Kimi judging pass over the ABLATION cells.
#
# Three failure modes, each seen for real on this stack:
#   1. server gone            -> port 8011 stops listening; restart it
#   2. ENGINE DEADLOCK        -> port still answers, requests "running", but
#      vllm:generation_tokens_total does not move. "Port up" is NOT "engine
#      alive"; the token counter is the only signal that separates a busy
#      engine from a wedged one.
#   3. driver exited early    -> cells remain unjudged; relaunch (run_eval
#      skips finished cells, so a restart costs only the unfinished ones)
#
# RESTARTS USE serve_kimi_lustre.sh WITH HUB SET. serve_kimi_mn.sh and
# serve_kimi_stable.sh both point HF_HOME at /opt/dlami/nvme/venkat.kesav,
# which on these nodes is either absent or owned by another user -- a restart
# through them wedges with an empty log instead of serving.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
HEAD=${HEAD:-10.20.213.4}; WORKER=${WORKER:-10.20.214.12}
C=${C:-animatebanana-mn}
HUB=${HUB:-/opt/dlami/nvme/vkkimi/hf_cache}
CELLS=${CELLS:-$REPO/data/ablation_judge_cells.json}
TOTAL=$(python3 -c "import json;print(len(json.load(open('$CELLS'))))")
LOG=$REPO/logs/ablation_judge; mkdir -p "$LOG"
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no"

# pidfile guard: `pgrep -f <name>` matches its OWN shell, so a self-check with
# pgrep reports "already running" forever and every restart silently no-ops.
PID=$LOG/night.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "supervisor already running as pid $(cat "$PID")"; exit 0
fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/night.log"; }
done_count(){ find "$REPO"/data/ablation_cache -path '*evals*' -name 'animation.json' 2>/dev/null | wc -l; }
server_up(){ local p; p=$($SSH "$HEAD" 'ss -lnt 2>/dev/null | grep -c ":8011"' 2>/dev/null|tail -1); [ "${p:-0}" -gt 0 ] 2>/dev/null; }
snap(){ $SSH "$HEAD" "docker exec $C bash -lc 'curl -s -m 10 http://127.0.0.1:8011/metrics'" 2>/dev/null | awk '
  index($1,"vllm:num_requests_running")==1{r=$2} index($1,"vllm:generation_tokens_total")==1{g=$2}
  END{if(r!=""||g!="")printf "%.0f %.0f\n",r,g}'; }
driver_up(){ local n; n=$($SSH "$HEAD" 'pgrep -fc "run_v5_[j]udge.sh"' 2>/dev/null|tail -1); [ "${n:-0}" -gt 0 ] 2>/dev/null; }

restart_server(){
  say "restarting kimi (lustre profile, HUB=$HUB)"
  HUB="$HUB" HEAD="$HEAD" WORKER="$WORKER" C="$C" bash "$REPO/scripts/remote/serve_kimi_lustre.sh" >/dev/null 2>&1
  for i in $(seq 1 60); do
    sleep 30
    $SSH "$HEAD" "docker exec $C bash -lc 'curl -s -m 5 http://127.0.0.1:8011/v1/models'" 2>/dev/null | grep -q Kimi && { say "kimi back up"; return 0; }
  done
  say "kimi FAILED to come back"; return 1
}
start_driver(){
  say "starting judge driver"
  $SSH "$HEAD" "cd $REPO && setsid nohup env C=$C WORKERS=${WORKERS:-6} CELLS=$CELLS \
      EVALS_GLOB='$REPO/data/ablation_cache/*' bash $REPO/scripts/run_v5_judge.sh \
      > $REPO/logs/bench_v5/judge/driver_ablation.log 2>&1 < /dev/null &" >/dev/null 2>&1
}

FROZEN=0; LASTG=-1
while :; do
  D=$(done_count)
  say "records=$D/$TOTAL"
  [ "$D" -ge "$TOTAL" ] && { say "ALL CELLS JUDGED"; break; }

  if ! server_up; then say "server down"; restart_server || { sleep 300; continue; }; start_driver; sleep 300; continue; fi

  S=$(snap); R=$(echo "$S"|awk '{print $1}'); G=$(echo "$S"|awk '{print $2}')
  if [ -n "${G:-}" ] && [ "${R:-0}" -gt 0 ] 2>/dev/null; then
    if [ "$G" = "$LASTG" ]; then
      FROZEN=$((FROZEN+1))
      say "token counter frozen ($FROZEN) running=$R gen=$G"
      if [ "$FROZEN" -ge 5 ]; then
        say "ENGINE DEADLOCK -- restarting"
        $SSH "$HEAD" "pkill -f 'run_v5_[j]udge.sh'" >/dev/null 2>&1
        restart_server && start_driver; FROZEN=0
      fi
    else FROZEN=0; fi
    LASTG=$G
  fi

  if ! driver_up; then say "driver not running; relaunching"; start_driver; fi
  sleep 300
done
