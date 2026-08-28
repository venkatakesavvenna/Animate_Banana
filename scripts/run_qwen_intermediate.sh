#!/usr/bin/env bash
# Re-score the four INTERMEDIATE suites with Qwen instead of Gemini.
#
# WHY THIS IS WORTH DOING, BEYOND COST
# ------------------------------------
# The intermediate suites are currently judged by google/gemini-3.7-flash --
# the same model that GENERATED the code, XML and sequences being judged.
# That is the circular-evaluation setup the animation tree was deliberately
# built to avoid by using Qwen. So this is not only a cost exercise: it also
# removes a methodological weakness from the numbers that go in the paper.
#
# Two of the four suites are judged at all:
#   stage1  component score + rendering fidelity   (judged)
#   xml     GT<->pred alignment, cached per sample (judged, once per sample)
#   seq     GT comparison + rule checks            (NOT judged)
#   stage3  compile + line-diff AIF                (NOT judged unless
#                                                   --include-quality)
# So the saving is real but modest -- roughly $0.024 of a $0.24 cell -- while
# the independence gain applies to every stage-1 number in the table.
#
# WRITES TO A SEPARATE EVALS ROOT. Same config name, style and sample would
# otherwise overwrite the Gemini-judged records in place, destroying the
# baseline this run exists to be compared against.
#
#   scripts/run_qwen_intermediate.sh                    # tikz, all styles
#   CONFIG=bench_v3_or_svg.yaml STYLES=progressive_reveal scripts/run_qwen_intermediate.sh
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
CONTAINER=animatebanana-v4
PY=/environments/img_2_svg_pretraining/bin/python
CFG=${CONFIG:-bench_v3_or.yaml}
CONFIG=src/img_2_svg_pretraining/pipeline/configs/$CFG
STYLES=${STYLES:-"progressive_reveal hopping_bounding_box sliding_bounding_box colour_pop alpha_masking"}
SUITES=${SUITES:-"stage1 xml sequence stage3"}
QROOT=$REPO/data/animatebench_v3_cache/animatebench_v3/evals_qwen_judge
LOG=$REPO/logs/qwen_intermediate
mkdir -p "$LOG" "$QROOT"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/driver.log"; }

# Runs on the BARE HOST: vLLM listens on the host's 127.0.0.1:8010 (its
# container uses --network host), which the pipeline container cannot reach.
# stage1/stage3 compile, so they need LaTeX -- but only for TikZ. Those are
# available on the host? No: latexmk is container-only. So compile-dependent
# suites run in the container, which CAN reach nothing on 8010... hence the
# split below: judged suites that need the GPU run on the host.
say "=== qwen intermediate | cfg=$CFG | styles: $STYLES"
say "    writing to $QROOT (Gemini records untouched)"

for style in $STYLES; do
  ids=$(cd "$REPO" && python3 scripts/bench_v3_styles.py --style "$style")
  [ -z "$ids" ] && continue
  for suite in $SUITES; do
    (cd "$REPO" && PYTHONPATH=src python3 -u -m img_2_svg_pretraining.animatebench.run_eval \
      "$suite" --config "$CONFIG" --style "$style" --only $ids \
      --judge-backend qwen_judge --evals-root "$QROOT") \
      >>"$LOG/${style}_${suite}.log" 2>&1
    say "  $style/$suite exit=$?"
  done
  say "  $style: $(find "$QROOT" -path "*/$style/*" -name 'stage1.json' 2>/dev/null | wc -l) cell(s) scored"
done
say "=== done"
