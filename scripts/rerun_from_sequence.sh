#!/usr/bin/env bash
# Re-run stage 2c onward after the sequencer / narrative / designer prompt edits.
#
# WHY IT STARTS AT `sequence`, NOT `narrate`.
# Three prompts changed. The latest two (narrative_writer, svg_designer) sit at
# 2e and 3a, but `svg_sequencer.yaml` gained RULESET 4 (legend handling) and
# runs at 2c -- BEFORE both. Its rule changes which elements enter the sequence
# at all: legends pinned to timestamp 1 for the reveal styles, and excluded
# entirely as bounding-box targets. Re-running from `narrate` would leave every
# animation built on a sequence that predates that rule, so the legend fix
# would silently never appear.
#
# WHAT IS REUSED. Everything upstream of the sequencer: stage-1 code, the raster
# splices, the diagram critic's repairs, and the parser XML. Those prompts did
# not change, and re-running them would burn budget to reproduce identical
# artifacts.
#
# METRICS ARE NOT TOUCHED. This script never invokes run_eval. Existing records
# under evals_study are left exactly as they are; a snapshot is taken alongside
# before any of this runs.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
CFG="$1"; PREFIX="$2"; JOBS="${3:-3}"
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')
V2="CVPR_2025_pipe00004 CVPR_2025_pipe00010 CVPR_2025_pipe00011 CVPR_2025_pipe00041 CVPR_2025_pipe00045 CVPR_2025_pipe00137"
mkdir -p logs/rerun

run() {  # run <style> <sample>
  local style="$1" sample="$2" cfg="$CFG"
  case " $V2 " in *" $sample "*) cfg="${CFG/bench_v3_/bench_v2_}" ;; esac
  [ -f "src/img_2_svg_pretraining/pipeline/configs/$cfg" ] || return
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline all \
       --config src/img_2_svg_pretraining/pipeline/configs/$cfg --style $style --only $sample \
       --from sequence --force" \
    > "logs/rerun/${CFG%.yaml}__${style}__${sample}.log" 2>&1
  echo "  done $style/$sample exit=$?"
}

while read -r line; do
  cell=$(echo "$line" | tr -d ' "'); [ -z "$cell" ] && continue
  style="${cell%%:*}"; sample="${cell##*:}"
  # Only cells this model already generated: a cell with no stage-1 code cannot
  # be resumed from `sequence`, and re-running it whole is a different job.
  ls data/animatebench_v3_cache/*/exports/${PREFIX}*__svg__*__${style}__*/${sample}/animation.mp4 >/dev/null 2>&1 || continue
  # Already re-run on the new prompts during the pilot -- do not pay for it twice.
  [ "$sample" = "CVPR_2025_arch01215" ] && [ "$style" = "progressive_reveal" ] &&     { echo "  skip $style/$sample (pilot, already on new prompts)"; continue; }
  run "$style" "$sample" &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done < /fsxvision_new/venkat.kesav/img_2_svg_pretraining/data/cells38.txt
wait
echo "=== $CFG re-run complete"
