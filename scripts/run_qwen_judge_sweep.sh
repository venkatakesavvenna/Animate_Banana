#!/usr/bin/env bash
# Everything Qwen can judge, on local GPU -- no API spend.
#
# PART 1  animation tree on the SVG pilot cells that lack it.
# PART 2  the JUDGED intermediate suites, re-scored with Qwen for BOTH targets,
#         so Gemini and Qwen can be correlated on identical inputs.
#
# ONLY stage1 AND xml ARE JUDGED. `sequence` is GT comparison plus rule checks
# and `stage3` is compile + line-diff; neither calls a model unless
# --include-quality is passed, which it is not. Re-running them under a
# different judge would produce byte-identical records and burn GPU for
# nothing, so they are deliberately excluded.
#
# PART 2 WRITES TO A SEPARATE EVALS ROOT. Same config/style/sample would
# otherwise overwrite the Gemini records in place -- destroying the baseline
# the correlation is against.
#
# Runs on the BARE HOST: vLLM binds 127.0.0.1 (its container uses
# --network host), which the bridge-networked pipeline container cannot
# reach. No LaTeX is needed here because every render is already content-hash
# cached from the Gemini runs, so compile_tikz/render_svg hit the cache.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
QROOT=$REPO/data/animatebench_v3_cache/animatebench_v3/evals_qwen_judge
LOG=$REPO/logs/qwen_judge; mkdir -p "$LOG" "$QROOT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/driver.log"; }
run() { PYTHONPATH=src python3 -u -m img_2_svg_pretraining.animatebench.run_eval "$@"; }

SVG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg.yaml
TIKZ=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or.yaml

say "=== PART 1: animation tree on SVG pilot cells"
while read -r style sample; do
  [ -z "$style" ] && continue
  say "  $style / $sample"
  run animation --config "$SVG" --style "$style" --only "$sample" \
      --judge-backend qwen_judge >>"$LOG/anim_${style}.log" 2>&1
  say "    exit=$?"
done < /tmp/svg_pilot.txt

say "=== PART 2: judged intermediate suites under Qwen -> $QROOT"
for pair in "tikz:$TIKZ" "svg:$SVG"; do
  tag="${pair%%:*}"; cfg="${pair##*:}"
  for style in progressive_reveal hopping_bounding_box sliding_bounding_box colour_pop alpha_masking; do
    ids=$(python3 scripts/bench_v3_styles.py --style "$style")
    [ -z "$ids" ] && continue
    for suite in stage1 xml; do
      run "$suite" --config "$cfg" --style "$style" --only $ids \
          --judge-backend qwen_judge --evals-root "$QROOT" \
          >>"$LOG/${tag}_${style}_${suite}.log" 2>&1
      say "  $tag/$style/$suite exit=$?"
    done
  done
done
say "=== done | qwen-judged records: $(find "$QROOT" -name '*.json' | wc -l)"
