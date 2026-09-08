#!/usr/bin/env bash
# Generate the three sets against an ALREADY-RUNNING Kimi server.
#
# Deliberately separate from kimi_night.sh: that script calls
# serve_kimi_nvme.sh, which RECREATES the containers -- restarting it to pick up
# a config change would tear down a healthy server and pay the 450s load again.
# Use this whenever the server is already up.
#
# Resuming is free: run_pipeline skips any stage whose artifact exists, so a
# re-run only redoes the cells that actually failed.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
PORT=${PORT:-8011}
LOG=$REPO/logs/kimi_night; mkdir -p "$LOG"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/gen_only.log"; }

curl -s -m 8 "http://127.0.0.1:$PORT/v1/models" | grep -q Kimi \
  || { say "FATAL: no Kimi on 127.0.0.1:$PORT -- serve it first"; exit 1; }
say "server reachable; starting generation"

for SET in ${SETS:-abl zs v6}; do
  say "--- kimi_k26/$SET"
  MODEL=kimi_k26 SET=$SET JOBS=${JOBS:-2} bash scripts/run_gen_night.sh 2>&1 | tail -3 | tee -a "$LOG/gen_only.log"
done
say "=== kimi generation done"
