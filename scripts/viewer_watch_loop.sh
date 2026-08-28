#!/usr/bin/env bash
# Run the viewer freshness check every INTERVAL seconds, forever.
# Detached from any session (setsid) so it survives the shell that started it.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
INTERVAL="${INTERVAL:-3600}"
while true; do
  ./scripts/check_viewer_fresh.sh >> logs/scale/viewer_watch.log 2>&1
  sleep "$INTERVAL"
done
