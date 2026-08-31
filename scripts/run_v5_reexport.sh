#!/usr/bin/env bash
# Re-export every judgeable cell so it has the `steps/` deck.
#
# WHY THIS IS MANDATORY BEFORE JUDGING
# ------------------------------------
# SSS, GPS and NAS are PER-TIMESTEP judges. `animatebench/frames.step_frames`
# accepts a deck only when len(frames) is exactly n_steps or n_steps+1 --
# deliberately, so a step can never be scored against a frame from the wrong
# animation state. The old exports used a uniform time grid, so their frame
# counts do not match: measured 9 of 60 cells matched, meaning ~85% of cells
# would have had all three metrics SILENTLY SKIPPED, producing a table of
# nulls that looks like data.
#
# The new exporter writes a second `steps/` deck sampled at the MIDPOINT of
# each timestep's slice, restoring the 1:1 mapping. This pass regenerates it.
#
# Pure chromium + ffmpeg -- NO model calls -- so it is cheap (~9s/cell) and can
# run at high parallelism regardless of what the GPUs are doing.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=animatebanana-v5
PY=/environments/img_2_svg_pretraining/bin/python
LOG=$REPO/logs/bench_v5/reexport; mkdir -p "$LOG/cells"
WORKERS=${WORKERS:-16}
CELLS=${CELLS:-$REPO/data/v5_judge_cells.json}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/driver.log"; }

export_one() {
  local cfg="$1" style="$2" sample="$3"
  local tag="${cfg}__${style}__${sample}"
  docker exec -u "$(id -u):$(id -g)" -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
       export --config src/img_2_svg_pretraining/pipeline/configs/${cfg}.yaml \
       --style $style --only $sample --force" \
    >"$LOG/cells/$tag.log" 2>&1
  echo "$? $tag"
}
export -f export_one
export C PY LOG

say "=== v5 re-export (steps/ decks) | workers: $WORKERS"
mapfile -t ROWS < <(python3 -c "
import json;
[print(f\"{r['config']} {r['style']} {r['sample']}\") for r in json.load(open('$CELLS'))]")
say "  cells: ${#ROWS[@]}"
printf '%s\n' "${ROWS[@]}" \
  | xargs -P "$WORKERS" -I{} bash -c 'export_one $0' {} \
  | while read -r rc tag; do
      [ "$rc" = 0 ] || say "  FAIL $tag (rc=$rc)"
    done
say "  steps/ decks now: $(find data/animatebench_v5*cache -path '*exports*' -type d -name steps 2>/dev/null|wc -l)"
say "=== re-export complete"
