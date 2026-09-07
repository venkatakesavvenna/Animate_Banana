#!/usr/bin/env bash
# Unattended overnight plan for the full 91x8 ablation suite.
#
#   1. wait for any in-flight run_ablations.sh (the one-sample demo) to finish
#   2. VERIFY the artifact-seeding fix on one fresh sample before spending at
#      scale: run `full` for a single new sample and log how many calls were
#      fresh -- with the fix, converter and raster are skipped outright and
#      parse/sequence should hit round-one's cache, leaving roughly the three
#      critics. If the count comes back high the fix did not take, and the log
#      says so before 91 samples repeat the mistake.
#   3. run the whole suite (full first, then the seven ablations -- the
#      script's own guard enforces that order)
#   4. point the ablation viewer at all 91 cells
#   5. write a spend/coverage summary
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
LOG=logs/ablation/night.log
exec >> "$LOG" 2>&1
echo "=== night run started $(date '+%F %T')"

# -- 1: never run two suite instances against the same caches ---------------
while pgrep -f "run_ablations.sh" | grep -qv $$; do sleep 30; done
echo "--- demo suite finished $(date '+%T')"

# -- 2: verification sample -------------------------------------------------
VERIFY=$(awk -F: 'NR==1{print $2}' data/ablation_cells.txt)
[ "$VERIFY" = "CVPR_2025_arch01225" ] && VERIFY=$(awk -F: 'NR==2{print $2}' data/ablation_cells.txt)
before=$(find data/ablation_cache/full/animatebench_v5/responses -links 1 -name '*.json' 2>/dev/null | wc -l)
ONLY="$VERIFY" JOBS=1 ./scripts/run_ablations.sh full
after=$(find data/ablation_cache/full/animatebench_v5/responses -links 1 -name '*.json' 2>/dev/null | wc -l)
fresh=$((after - before))
echo "--- verification: sample=$VERIFY fresh_calls=$fresh (expect <=6; unseeded would be ~10)"
if [ "$fresh" -gt 8 ]; then
  echo "--- WARNING: artifact seeding is not producing the expected cache hits;"
  echo "---          continuing anyway, but the run will cost more than projected."
fi

# -- 3: the suite -----------------------------------------------------------
JOBS=6 ./scripts/run_ablations.sh
echo "--- suite finished $(date '+%F %T')"

# -- 4: viewer over all 91 cells -------------------------------------------
CELLS=$(awk -F: '{printf "%s%s:%s", sep, $2, $1; sep=","}' data/ablation_cells.txt)
python3 - <<'PY'
import subprocess, os, signal
o=subprocess.run(["ps","-eo","pid=,comm=,args="],capture_output=True,text=True).stdout
for l in o.splitlines():
    q=l.split(None,2)
    if len(q)==3 and q[1].startswith("python") and "inspector.compare --port 8607" in q[2]:
        os.kill(int(q[0]), signal.SIGTERM)
PY
sleep 2
CELLS="$CELLS" ./scripts/run_ablation_viewer.sh 8607
echo "--- viewer relaunched on :8607 with all 91 cells"

# -- 5: summary -------------------------------------------------------------
python3 - <<'PY'
import glob, os, json
lines=["ablation suite summary"]
total=0.0
for n in ["full","stage1_no_critic","stage2_no_image","sequencer_no_xml",
          "stage2_no_critic","narration_no_context","designer_no_image","stage3_no_critic"]:
    exp=len(glob.glob(f"data/ablation_cache/{n}/animatebench_v5/exports/*/*/animation.mp4"))
    d=f"data/ablation_cache/{n}/animatebench_v5/responses/openrouter_gemini37flash"
    tp=tc=fresh=0
    for f in glob.glob(d+"/*.json"):
        if os.stat(f).st_nlink!=1: continue
        fresh+=1; u=(json.load(open(f)).get("usage") or {})
        tp+=u.get("prompt_tokens",0); tc+=u.get("completion_tokens",0)
    c=tp/1e6*0.375+tc/1e6*1.875; total+=c
    lines.append(f"  {n:24} exports={exp:3}/91  fresh_calls={fresh:5}  ${c:.2f}")
lines.append(f"  TOTAL fresh spend (nlink==1 heuristic): ${total:.2f}")
open("logs/ablation/night_summary.txt","w").write("\n".join(lines)+"\n")
print("\n".join(lines))
PY
echo "=== night run done $(date '+%F %T')"
