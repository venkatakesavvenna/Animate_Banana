#!/usr/bin/env bash
# Gemini 3.7 Flash generation over all 91 cells, via OpenRouter.
# Runs CONCURRENTLY with the local GPU sweep: it needs no GPU, so serialising
# it behind the served models would waste the whole night. Generation only.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=animatebanana-v5; PY=/environments/img_2_svg_pretraining/bin/python
CFG=src/img_2_svg_pretraining/pipeline/configs/bench_v5_or_gemini37flash.yaml
LOG=$REPO/logs/bench_v5; STATE=$LOG/state; mkdir -p "$LOG" "$STATE"
set -a; . "$REPO/.env"; set +a
BATCH=${BATCH:-24}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/gemini.log"; }
say "=== gemini 3.7 flash (OpenRouter) | GENERATION ONLY"
for STYLE in alpha_masking colour_pop progressive_reveal hopping_bounding_box sliding_bounding_box; do
  ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py --root data/animatebench_v5 --target svg --style "$STYLE" 2>/dev/null)
  arr=($ids); [ ${#arr[@]} -eq 0 ] && continue
  say "  $STYLE: ${#arr[@]} cells"
  for ((i=0;i<${#arr[@]};i+=BATCH)); do
    tag="gemini_${STYLE}_$((i/BATCH))"; [ -f "$STATE/$tag.done" ] && continue
    docker exec -u "$(id -u):$(id -g)" -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers -e OPEN_ROUTER_KEY="$OPEN_ROUTER_KEY" "$C" bash -lc \
      "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
         all --config $CFG --style $STYLE --only ${arr[*]:i:BATCH}" \
      >>"$LOG/pipe_gemini_${STYLE}.log" 2>&1
    rc=$?; say "    batch $((i/BATCH)) exit=$rc"
    [ $rc -eq 0 ] && touch "$STATE/$tag.done"
  done
done
say "=== gemini complete"
