#!/usr/bin/env bash
# Expose a running study server through a Cloudflare quick tunnel.
#
#   bash scripts/tunnel_study.sh 8609
#
# The URL is public and unauthenticated -- anyone with it can register. /admin
# still refuses without X-Admin-Token, so the unblinded views stay closed.
# Tear the tunnel down when the session ends: kill the pid it prints.
set -u
PORT=${1:-8609}
LOG=/tmp/cf_tunnel_$PORT.log
: > "$LOG"
setsid nohup ~/bin/cloudflared tunnel --no-autoupdate \
  --url "http://localhost:$PORT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "cloudflared pid $PID, log $LOG"
for i in $(seq 1 40); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1)
  [ -n "$URL" ] && { echo "$URL"; exit 0; }
  sleep 1
done
echo "no URL after 40s; check $LOG" >&2
tail -5 "$LOG" >&2
exit 1
