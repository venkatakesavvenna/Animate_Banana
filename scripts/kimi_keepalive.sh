#!/usr/bin/env bash
# Keep the 2-node Kimi judge alive, restarting it whenever it dies.
#
# WHY THIS IS NEEDED. Kimi is 555 GiB and does not fit on one node (measured:
# at --gpu-memory-utilization 0.97 the weights load and vLLM then dies with
# "No available memory for the cache blocks"), so it must span two. Both nodes
# are shared, and twice now a process on the WORKER node has exited mid-run --
#     NCCL error: remote process exited or there was a network error
#     Connection closed by peer [10.20.197.208]
# -- which kills the whole engine. The containers stay "Up" and GPU memory drops
# to 0, so `docker ps` looks healthy while nothing is served.
#
# The per-cell health gate in the judging drivers already prevents a dead server
# from writing poisoned "Connection error" records; this restores the server so
# those cells stop waiting.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
HEAD=${HEAD:-10.20.188.73}
WORKER=${WORKER:-10.20.197.208}
PORT=${PORT:-8011}
LOG=$REPO/logs/kimi_night/keepalive.log

# HARD NODE GUARD. Using nodes the user had not named cost them their bonus on
# 2026-09-08. HEAD/WORKER are no longer trusted as free-form variables: anything
# outside the pair the user named is refused here, before a single SSH.
ALLOWED_NODES="10.20.235.133 10.20.239.233"
for _n in "${HEAD:-}" "${WORKER:-}"; do
  case " $ALLOWED_NODES " in
    *" $_n "*) ;;
    *) echo "REFUSING: node '$_n' is not one the user named ($ALLOWED_NODES)." >&2; exit 1;;
  esac
done

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

up(){ curl -s -m 8 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q Kimi; }

retune(){
  # Re-open the tunnel. A dead endpoint leaves the ssh process ALIVE, so the
  # tunnel looks fine in `ps` while every request fails -- that is what turned
  # a server crash into 19 poisoned records rather than an obvious outage.
  local old; old=$(ps -eo pid,args | awk '/ssh .*-L '"$PORT"':127.0.0.1:'"$PORT"'/ && !/awk/{print $1}')
  for x in $old; do kill "$x" 2>/dev/null; done
  sleep 2
  ssh -f -N -o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -L "$PORT:127.0.0.1:$PORT" "$HEAD" 2>/dev/null
}

while :; do
  if up; then sleep 120; continue; fi
  say "judge DOWN -- restarting 2-node server"
  HEAD=$HEAD WORKER=$WORKER PORT=$PORT bash scripts/remote/serve_kimi_nvme.sh >>"$LOG" 2>&1
  for i in $(seq 1 45); do
    ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "$HEAD" \
       "curl -s -m 5 http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q Kimi && break
    sleep 20
  done
  retune
  if up; then say "judge back up"; else say "restart did not take; retrying in 5min"; sleep 300; fi
done
