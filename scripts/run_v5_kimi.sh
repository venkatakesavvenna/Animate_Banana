#!/usr/bin/env bash
# Kimi K2.6 generation over all 91 cells. GENERATION ONLY -- no judging.
#
# Unlike run_v5_overnight.sh this does NOT serve the model: Kimi runs on a
# 2-node Ray cluster brought up separately (scripts/remote/serve_kimi_mn.sh),
# and a serve attempt here would fight the running server for the GPUs.
# Must run ON the node hosting the API server -- both containers are
# --network host, so the pipeline reaches it at 127.0.0.1:8011.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=animatebanana-v5
PY=/environments/img_2_svg_pretraining/bin/python
CFG=src/img_2_svg_pretraining/pipeline/configs/bench_v5_svg_kimi_k26.yaml
LOG=$REPO/logs/bench_v5; STATE=$LOG/state; mkdir -p "$LOG" "$STATE"
BATCH=${BATCH:-24}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/kimi.log"; }

say "=== kimi K2.6 (2-node) | GENERATION ONLY"
for STYLE in alpha_masking colour_pop progressive_reveal hopping_bounding_box sliding_bounding_box; do
  ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py --root data/animatebench_v5 --target svg --style "$STYLE" 2>/dev/null)
  arr=($ids); [ ${#arr[@]} -eq 0 ] && continue
  say "  $STYLE: ${#arr[@]} cells"
  for ((i=0;i<${#arr[@]};i+=BATCH)); do
    tag="kimi_${STYLE}_$((i/BATCH))"; [ -f "$STATE/$tag.done" ] && continue
    docker exec -u "$(id -u):$(id -g)" -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
      "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
         all --config $CFG --style $STYLE --only ${arr[*]:i:BATCH}" \
      >>"$LOG/pipe_kimi_${STYLE}.log" 2>&1
    rc=$?; say "    batch $((i/BATCH)) exit=$rc"
    [ $rc -eq 0 ] && touch "$STATE/$tag.done"
  done
done
say "=== kimi complete"
