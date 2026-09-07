#!/usr/bin/env bash
# Keep the study server (and optionally its tunnel) alive.
#
#   PORT=8609 BUNDLE=data/study_bundles/pilot-demo1 \
#   DB=data/study_runs/study_demo1.db TUNNEL=1 \
#     nohup bash scripts/study_watchdog.sh > data/study_runs/watchdog_8609.log 2>&1 &
#
# Exists because the server and tunnel died mid-session when /tmp was swept and
# nothing noticed until someone opened the link. A study that is quietly down
# looks exactly like a study nobody has started yet.
set -u
PORT=${PORT:-8620}
BUNDLE=${BUNDLE:-data/study_bundles/main-v2}
CONFIG=${CONFIG:-src/img_2_svg_pretraining/pipeline/configs/study_main.yaml}
DB=${DB:-data/study_runs/study_pilot.db}
TUNNEL=${TUNNEL:-0}
EVERY=${EVERY:-30}
mkdir -p data/study_runs
URLFILE=data/study_runs/tunnel_url_$PORT.txt

while true; do
  if ! curl -s -o /dev/null -m 10 "http://localhost:$PORT/"; then
    echo "$(date -Is) server on $PORT is down; restarting"
    PORT="$PORT" BUNDLE="$BUNDLE" DB="$DB" CONFIG="${CONFIG:-}" ADMIN_TOKEN="${ADMIN_TOKEN:-devtoken}" \
      bash scripts/run_study_server.sh || true
  fi
  if [ "$TUNNEL" = "1" ]; then
    URL=$(cat "$URLFILE" 2>/dev/null || true)
    # A quick tunnel gets a NEW hostname each time it restarts, so the old link
    # dies with it. Record the current one rather than assuming it is stable.
    if [ -z "$URL" ] || ! curl -s -o /dev/null -m 20 "$URL/"; then
      echo "$(date -Is) tunnel down; reopening"
      NEW=$(bash scripts/tunnel_study.sh "$PORT" 2>/dev/null | tail -1)
      case "$NEW" in https://*) echo "$NEW" > "$URLFILE"; echo "$(date -Is) new url $NEW";; esac
    fi
  fi
  sleep "$EVERY"
done
