#!/usr/bin/env bash
# Generate the missing cells for ONE model arm, across the 38-cell study list.
#
# Usage: run_scale_generation.sh <config.yaml> <lineage-prefix> [JOBS]
#
# Skips any cell that already has an animation.mp4 under THIS config's own
# lineage -- checking the model-specific prefix, never a bare exports/*/<sample>
# glob, which would match another model's output and silently skip a cell that
# was never generated for this one.
#
# The v2 samples live in a different dataset root and `run_pipeline` has no
# --dataset override, so they are dispatched through the matching *_v2 config.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
CFG="$1"; PREFIX="$2"; JOBS="${3:-3}"
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')
V2_SAMPLES="CVPR_2025_pipe00004 CVPR_2025_pipe00010 CVPR_2025_pipe00011 CVPR_2025_pipe00041 CVPR_2025_pipe00045 CVPR_2025_pipe00137"
mkdir -p logs/scale

gen() {  # gen <style> <sample>
  local style="$1" sample="$2" cfg="$CFG"
  case " $V2_SAMPLES " in *" $sample "*) cfg="${CFG/bench_v3_/bench_v2_}" ;; esac
  [ -f "src/img_2_svg_pretraining/pipeline/configs/$cfg" ] || { echo "  no config $cfg for $sample"; return; }
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline all \
       --config src/img_2_svg_pretraining/pipeline/configs/$cfg --style $style --only $sample" \
    > "logs/scale/${CFG%.yaml}__${style}__${sample}.log" 2>&1
  echo "  done $style/$sample exit=$?"
}

while read -r line; do
  cell=$(echo "$line" | tr -d ' "')
  [ -z "$cell" ] && continue
  style="${cell%%:*}"; sample="${cell##*:}"
  have=$(ls data/animatebench_v3_cache/*/exports/${PREFIX}*__svg__*__${style}__*/${sample}/animation.mp4 2>/dev/null | wc -l)
  if [ "$have" -gt 0 ]; then echo "  skip $style/$sample (already has mp4)"; continue; fi
  gen "$style" "$sample" &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done < /tmp/cells38.txt
wait
echo "=== $CFG generation complete"
