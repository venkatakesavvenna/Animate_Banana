#!/usr/bin/env bash
# One sample, end to end, on Gemini 3.7 Flash via OpenRouter: full pipeline
# then the four intermediate-stage suites. The animation tree is not run --
# it is going to Qwen later.
#
#   scripts/run_one_or.sh [SAMPLE_ID] [STYLE]
#
# The judge is pinned to the same OpenRouter backend. Left at its default it
# would resolve `gemini_flash`, find no such backend in this config, and fall
# back to the free-tier Gemini judge -- quietly scoring a paid run with the
# rate-limited endpoint this config exists to escape.
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
CONFIG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or.yaml
SAMPLE=${1:-CVPR_2025_arch01871}
STYLE=${2:-progressive_reveal}
LOG=$REPO/logs/bench_v3_or
mkdir -p "$LOG"

# .env is not auto-loaded: python-dotenv is absent, so the Flask app warns
# about it and nothing else reads it.
set -a; . "$REPO/.env"; set +a

dexec() {
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$OPEN_ROUTER_KEY" \
    "$CONTAINER" bash -lc "cd /code && PYTHONPATH=src $PY -u $*"
}

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/driver.log"; }

spend() {   # OpenRouter reports actual cost; measure it rather than modelling it
  curl -s https://openrouter.ai/api/v1/credits \
    -H "Authorization: Bearer $OPEN_ROUTER_KEY" \
    | python3 -c "import json,sys;d=json.load(sys.stdin)['data'];print(f\"{d['total_usage']:.4f}\")" \
    2>/dev/null || echo "?"
}

before=$(spend)
say "=== $SAMPLE / $STYLE on google/gemini-3.7-flash | spent so far: \$$before"

say "pipeline"
dexec -m img_2_svg_pretraining.pipeline.run_pipeline all \
  --config "$CONFIG" --style "$STYLE" --only "$SAMPLE" >>"$LOG/pipeline.log" 2>&1
say "pipeline exit=$?  spent: \$$(spend)"

for suite in stage1 xml sequence stage3; do
  dexec -m img_2_svg_pretraining.animatebench.run_eval "$suite" \
    --config "$CONFIG" --style "$STYLE" --only "$SAMPLE" \
    --judge-backend openrouter_flash >>"$LOG/eval.log" 2>&1
  say "eval $suite exit=$?"
done

after=$(spend)
say "=== done | total spent \$$after  (this run: \$$(python3 -c "print(f'{$after-$before:.4f}')"))"
