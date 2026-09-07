#!/usr/bin/env bash
# Resume the ablation suite at a concurrency the OpenRouter in-flight budget
# accepts. The previous pass lost 47 cells to HTTP 402 in_flight_budget_exhausted
# -- transport rejections, not model failures, so nothing was cached and the
# cells only need re-running more slowly. JOBS=3 x max_concurrency=4 caps in-flight
# at ~12 requests against the ~72 that triggered the 402s.
set -uo pipefail
cd /fsxvision_new/venkat.kesav/img_2_svg_pretraining
# A pidfile, because `pgrep -f ablation_resume` also matches the shell running
# the check -- which made a dead suite report as alive for 50 minutes.
ABLATION_PIDFILE=logs/ablation/resume.pid
if [ -f "$ABLATION_PIDFILE" ] && kill -0 "$(cat "$ABLATION_PIDFILE")" 2>/dev/null; then
  echo "resume already running as pid $(cat "$ABLATION_PIDFILE")"; exit 0
fi
echo $$ > "$ABLATION_PIDFILE"
trap 'rm -f "$ABLATION_PIDFILE"' EXIT
export BASELINE_MIN=90
for pass in 1 2 3 4; do
  echo "########## resume pass $pass $(date '+%T')"
  # EVICT BEFORE EACH PASS. A response that cannot parse -- LaTeX where SVG was
  # asked for, or a truncated document -- is cached like any other, so a retry
  # replays it and the cell fails identically forever. Measured: 16 such
  # entries in stage1_no_critic alone, and clearing them is what let earlier
  # "permanent" failures succeed on the next attempt.
  for c in data/ablation_cache/*/; do
    python3 scripts/evict_bad_responses.py "$c" 2>/dev/null | sed "s|^|  $(basename $c): |"
  done
  JOBS=3 ./scripts/run_ablations.sh
done
python3 - <<'PY'
import glob
print("final coverage:")
for n in ["full","stage1_no_critic","stage2_no_image","sequencer_no_xml",
          "stage2_no_critic","narration_no_context","designer_no_image","stage3_no_critic"]:
    c=len(glob.glob(f"data/ablation_cache/{n}/animatebench_v5/exports/*/*/animation.mp4"))
    print(f"  {n:24} {c:3}/91")
PY
