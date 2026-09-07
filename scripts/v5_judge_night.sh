#!/usr/bin/env bash
# Unattended overnight supervisor for the SSS/GPS/NAS judging pass.
#
# Three jobs now, and the third is new:
#   1. keep the Kimi server alive (restart when the port is gone)
#   2. re-run the judging driver -- run_eval skips cells whose record exists,
#      so every pass is a resume and a restart costs only unfinished cells
#   3. DETECT ENGINE DEADLOCK. The fast profile hung with 12 requests
#      "running", GPUs 0% on both nodes, API still answering /metrics 200 --
#      so "port up" is NOT "engine alive". The one signal that separates a
#      busy engine from a wedged one: vllm:generation_tokens_total. If
#      requests are running and that counter does not move for 5 minutes,
#      the engine has lost them and only a restart recovers.
#
# ESCALATION: first restarts use serve_kimi_stable.sh (no async-scheduling,
# no DCP). If the engine deadlocks TWICE despite that, the remaining suspect
# is expert-parallel's cross-node all-to-all -- switch to serve_kimi_min.sh
# (no EP, max-num-seqs 32) and halve the workers for the rest of the night.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
N0=10.20.238.24; N1=10.20.233.75
LOG=$REPO/logs/bench_v5/judge; mkdir -p "$LOG"
PASSES=${PASSES:-12}
WORKERS=${WORKERS:-5}
CELLS=${CELLS:-$REPO/data/v5_judge_cells.json}
TOTAL=${TOTAL:-432}
EVAL_ROOTS=${EVAL_ROOTS:-"$REPO/data/animatebench_v5_cache $REPO/data/animatebench_v5_or_cache"}
CELLS=${CELLS:-$REPO/data/v5_judge_cells.json}
TOTAL=${TOTAL:-432}
EVALS_GLOB=${EVALS_GLOB:-$REPO/data/animatebench_v5*cache}
PROFILE=serve_kimi_stable.sh
DEADLOCKS=0
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [night] $*" | tee -a "$LOG/night.log"; }

server_up(){
  # The listening port is the only liveness signal that never lied here:
  # docker exec ps reads the wrong PID namespace under --pid host (false
  # DOWN), pgrep -fc matches its own shell (false UP), /health times out
  # while merely busy (false DOWN).
  local p
  p=$($SSH $N0 'ss -lnt 2>/dev/null | grep -c ":8011"' 2>/dev/null|tail -1)
  [ "${p:-0}" -gt 0 ] 2>/dev/null
}

# "running gen_tokens" from /metrics; empty on failure.
engine_snap(){
  $SSH $N0 'curl -s -m 10 http://127.0.0.1:8011/metrics' 2>/dev/null | awk '
    index($1,"vllm:num_requests_running")==1   {r=$2}
    index($1,"vllm:generation_tokens_total")==1 {g=$2}
    END{if (r!="" || g!="") printf "%.0f %.0f\n", r, g}'
}

kill_clients(){
  # Bracket patterns so pkill cannot match its own wrapping shell; the
  # pattern "animatebench.run_eval" can never match the vLLM server cmdline
  # (a bare "run_eval" once did, and killed it).
  $SSH $N0 'pkill -f "run_v5_[j]udge.sh" 2>/dev/null
            docker exec -u 0 animatebanana-v5 bash -c "pkill -9 -f \"animatebench.run_[e]val\"" 2>/dev/null; true' >/dev/null 2>&1
}

restart_server(){
  say "restarting server (profile=$PROFILE)"
  # The container itself can be GONE, not just vLLM: on 08-30 both nodes'
  # animatebanana-mn were SIGKILLed (exit 137, oom=false) within 21s of each
  # other by the image owner's cleanup. init_mn.sh is idempotent -- it
  # docker-starts a stopped container and recreates a removed one -- so run
  # it first; docker exec into a stopped container just fails silently here.
  for n in $N0 $N1; do
    $SSH $n "bash $REPO/scripts/remote/init_mn.sh" >/dev/null 2>&1
  done
  for n in $N0 $N1; do
    $SSH $n 'docker exec animatebanana-mn bash -lc "/environments/v5serve/bin/ray stop --force >/dev/null 2>&1; true"
             docker exec -u 0 animatebanana-mn bash -c "pkill -9 -f vllm.entrypoints; pkill -9 -f \"ray::\"; pkill -9 -f raylet; true" 2>/dev/null; sleep 5' >/dev/null 2>&1
  done
  sleep 10
  $SSH $N0 "ROLE=head bash $REPO/scripts/remote/ray_up.sh"   >/dev/null 2>&1
  $SSH $N1 "ROLE=worker bash $REPO/scripts/remote/ray_up.sh" >/dev/null 2>&1
  sleep 5
  $SSH $N0 "bash $REPO/scripts/remote/$PROFILE"              >/dev/null 2>&1
  for _ in $(seq 1 90); do server_up && { say "server healthy"; return 0; }; sleep 20; done
  say "WARNING: server did not come back"; return 1
}

done_cells(){ find $EVAL_ROOTS -path '*/evals/*' -name animation.json 2>/dev/null | wc -l; }

say "supervisor up: $PASSES pass(es), WORKERS=$WORKERS, $TOTAL cells (cells=$CELLS)"
for p in $(seq 1 $PASSES); do
  server_up || restart_server || { say "abort pass $p"; sleep 60; continue; }
  say "pass $p start (done so far: $(done_cells), profile=$PROFILE, workers=$WORKERS)"

  $SSH $N0 "CELLS=$CELLS EVALS_GLOB='$EVALS_GLOB' WORKERS=$WORKERS bash $REPO/scripts/run_v5_judge.sh" >>"$LOG/driver_pass${p}.log" 2>&1 &
  DRV=$!

  stuck=0; last_g=-1; hung=0
  while kill -0 "$DRV" 2>/dev/null; do
    sleep 60
    snap=$(engine_snap)
    if [ -z "$snap" ]; then
      # Metrics unreachable. One miss can be a busy server; three in a row
      # with the port gone is a dead server, and waiting for the pass
      # boundary means every worker burns its cell on connection errors
      # (the 08-30 container kill cost 13+ minutes of exactly that).
      down=$((${down:-0}+1)); stuck=0
      if [ "$down" -ge 3 ] && ! server_up; then
        hung=1
        say "SERVER DEAD mid-pass -- killing clients + restarting"
        kill_clients
        kill "$DRV" 2>/dev/null
        wait "$DRV" 2>/dev/null
        restart_server
        break
      fi
      continue
    fi
    down=0
    read -r r g <<<"$snap"
    if [ "${r:-0}" -gt 0 ] && [ "$g" = "$last_g" ]; then stuck=$((stuck+1)); else stuck=0; fi
    last_g=$g
    if [ "$stuck" -ge 5 ]; then
      hung=1
      DEADLOCKS=$((DEADLOCKS+1))
      say "DEADLOCK #$DEADLOCKS: running=$r, gen_tokens frozen 5 min -- killing clients + restarting"
      kill_clients
      kill "$DRV" 2>/dev/null
      wait "$DRV" 2>/dev/null
      if [ "$DEADLOCKS" -ge 2 ] && [ "$PROFILE" = serve_kimi_stable.sh ]; then
        PROFILE=serve_kimi_min.sh
        WORKERS=$(( WORKERS>2 ? WORKERS/2 : 1 ))
        say "escalating: profile=$PROFILE workers=$WORKERS"
      fi
      restart_server
      break
    fi
  done
  wait "$DRV" 2>/dev/null
  say "pass $p end (done: $(done_cells), deadlocks so far: $DEADLOCKS)"
  [ "$(done_cells)" -ge "$TOTAL" ] && { say "ALL CELLS DONE"; break; }
  [ "$hung" = 0 ] && sleep 10
done
say "=== supervisor finished: $(done_cells)/$TOTAL cells"
