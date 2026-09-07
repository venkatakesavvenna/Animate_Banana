#!/usr/bin/env bash
# Stop the ablation suite cleanly when the key's spend reaches a ceiling.
#
# WHY A GUARD RATHER THAN WATCHING BY HAND. Running out mid-flight does not
# fail loudly -- OpenRouter returns 402 per call, the pipeline treats it as a
# stage failure, and the suite keeps marching through cells burning partial
# pipelines that can never complete. That is exactly how the earlier in-flight
# 402 storm wasted ~$10 for one delivered cell. Better to stop on a number.
#
# Reads the AUTHORITATIVE figure -- the key's own reported usage -- not a local
# heuristic. An nlink-based estimate under-reported by 3.5x earlier in this run,
# so local accounting is not trusted for a stop decision.
set -uo pipefail
cd /fsxvision_new/venkat.kesav/img_2_svg_pretraining
FLOOR="${FLOOR:-52}"              # stop when REMAINING credit falls below this
LOG=logs/ablation/budget.log
KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')

usage() { bash scripts/or_remaining.sh; }   # now REMAINING credit, not usage

while true; do
  u=$(usage)
  if [ -z "$u" ]; then
    echo "$(date '+%T') usage unreadable; will retry" >> "$LOG"
    sleep 300; continue
  fi
  over=$(python3 -c "print(1 if float('$u') <= float('$FLOOR') else 0)")
  cov=$(ls data/ablation_cache/*/animatebench_v5/exports/*/*/animation.mp4 2>/dev/null | wc -l)
  echo "$(date '+%T') remaining=\$$u floor=\$$FLOOR cells=$cov/728" >> "$LOG"
  if [ "$over" = "1" ]; then
    echo "$(date '+%T') CREDIT FLOOR REACHED -- stopping the suite" >> "$LOG"
    python3 - <<'PY' >> "$LOG" 2>&1
import subprocess, os, signal
o=subprocess.run(["ps","-eo","pid=,comm=,args="],capture_output=True,text=True).stdout
k=0
for l in o.splitlines():
    q=l.split(None,2)
    if len(q)<3: continue
    pid,comm,args=q
    if (comm.startswith("bash") and ("ablation_resume" in args or "run_ablations.sh" in args)) \
       or (comm.startswith(("python","docker")) and "gemini_ablation" in args):
        try: os.kill(int(pid), signal.SIGTERM); k+=1
        except Exception: pass
print(f"  stopped {k} process(es)")
PY
    python3 - <<'PY' >> "$LOG"
import glob
print("  coverage at stop:")
for n in ["full","stage1_no_critic","stage2_no_image","sequencer_no_xml",
          "stage2_no_critic","narration_no_context","designer_no_image","stage3_no_critic"]:
    c=len(glob.glob(f"data/ablation_cache/{n}/animatebench_v5/exports/*/*/animation.mp4"))
    print(f"    {n:24} {c:3}/91")
PY
    exit 0
  fi
  sleep 300
done
