#!/usr/bin/env bash
# Unattended overnight supervisor: keeps both generation streams running until
# every cell is done, and RE-RUNS each stream on exit.
#
# Re-running is safe and is the whole recovery strategy: artifacts are keyed by
# cache lineage on disk and every stage skips work whose output already exists,
# so a second pass costs only the cells that did not finish. That is what makes
# an unattended night survivable -- a transient API error, an OOM, or one bad
# sample cannot end the run, and nothing has to be restarted by hand.
#
# Passes are capped so a genuinely broken stream cannot spin forever.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
LOG=$REPO/logs/bench_v5
PASSES=${PASSES:-4}

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [night] $*" | tee -a "$LOG/night.log"; }

say "supervisor up; $PASSES pass(es) per stream"

( for p in $(seq 1 $PASSES); do
    say "gemini pass $p"
    bash scripts/run_v5_gemini.sh >>"$LOG/gemini_driver.log" 2>&1
    say "gemini pass $p exit=$?"
  done ) &
GPID=$!

( for p in $(seq 1 $PASSES); do
    say "local pass $p"
    bash scripts/run_v5_overnight.sh >>"$LOG/overnight_driver.log" 2>&1
    say "local pass $p exit=$?"
  done ) &
LPID=$!

wait $GPID $LPID
say "=== all streams finished"
say "mp4s: $(find data/animatebench_v5_cache data/animatebench_v5_or_cache -name '*.mp4' 2>/dev/null | wc -l)"
