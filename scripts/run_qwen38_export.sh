#!/usr/bin/env bash
# Re-run ONLY the export stage for the five study cells.
#
# Free: `export` makes no model calls. It rasterises the animated SVG through
# headless chromium and muxes the frames to mp4, so it is also the one stage
# that needs PLAYWRIGHT_BROWSERS_PATH -- which the container does not set by
# default. A run launched without it completes every paid stage and then dies
# here with "Executable doesn't exist"; this re-runs the free tail rather than
# the whole pipeline.
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
CFG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg_qwen38_27b.yaml

export_cell() {   # export_cell <style> <sample>
  local log="$REPO/logs/qwen38/export__$1__$2.log"
  docker exec -u "$(id -u):$(id -g)" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
       export --config $CFG --style $1 --only $2" > "$log" 2>&1
  echo "  export $1/$2 exit=$?"
}

export_cell progressive_reveal   CVPR_2025_arch01225
export_cell colour_pop           CVPR_2025_arch00389
export_cell hopping_bounding_box CVPR_2025_arch00554
export_cell hopping_bounding_box Paper2Fig_pipe_diff_000000714
export_cell progressive_reveal   Paper2Fig_pipe_diff_000000677
