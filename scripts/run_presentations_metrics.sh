#!/bin/bash
# Video metrics over the 35 Original_Presentations talks.
#
# ONLY vfs_band + ascs_video. SSS/GPS/NAS need a sequence and a structure XML
# per sample; a human talk recording has neither, and `animation_quality.run`
# would record `stages_skipped` and produce nothing. See
# scripts/stage_presentations.py.
#
# STYLE IS PER SAMPLE, read from presentations_style_map.json. The config's
# `animation_style` is only a default -- running everything at that default
# would file all 35 under one style-keyed lineage and judge each against the
# wrong style prompt.
#
# The free Google key pool (75 keys x 20/day = 1500) covers this easily: 35
# cells x 2 calls = 70. `gemini_video` max_concurrency is 1 in the config;
# JOBS parallelises across CELLS instead, which does not race a single key.
set -u
cd /fsxvision_new/venkat.kesav/img_2_svg_pretraining
export PYTHONPATH=src
CFG=src/img_2_svg_pretraining/pipeline/configs/presentations_video.yaml
MAP=data/presentations_style_map.json
LOG=logs/presentations
JOBS=${JOBS:-4}
mkdir -p "$LOG"

mapfile -t CELLS < <(python3 -c "
import json
for k,v in sorted(json.load(open('$MAP')).items()): print(f'{v}:{k}')
")
echo "$(date -Is) starting ${#CELLS[@]} cells, JOBS=$JOBS"

run_one(){
  local style="${1%%:*}" sid="${1##*:}"
  local out="$LOG/${sid}.log"
  python3 -m img_2_svg_pretraining.animatebench.run_eval animation \
    --config "$CFG" --style "$style" --only "$sid" \
    --stages vfs_band ascs_video --rubric letters \
    --judge-backend gemini_video --force > "$out" 2>&1
  if grep -q "animation:" "$out"; then echo "$(date -Is) ok   $sid"
  else echo "$(date -Is) FAIL $sid"; fi
}

i=0
for c in "${CELLS[@]}"; do
  run_one "$c" &
  i=$((i+1))
  if [ $((i % JOBS)) -eq 0 ]; then wait; fi
done
wait
echo "$(date -Is) all cells attempted"
python3 scripts/collect_presentation_scores.py || true
