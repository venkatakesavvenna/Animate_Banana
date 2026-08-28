#!/usr/bin/env bash
# The four-panel metric-soundness viewer.
#
# Shows, for each of the five study cells, exactly four things:
#
#   1  the source figure          (the pipeline's input)
#   2  the GT reference animation (what a human annotator made)
#   3  ours, Gemini 3.7 Flash     (bench_v3_or_svg.yaml)
#   4  ours, Qwen3.8-27B          (bench_v3_or_svg_qwen38_27b.yaml)
#
# SVG ONLY. Each label names a single svg config, so `api_compare` emits one
# panel per label rather than the tikz/svg pair the 8604 viewer shows. Adding
# a tikz config to either label would silently make this a six-panel view.
#
# `--cells` pins each sample to the one style it was generated for. Without it
# the style selector is free, and selecting a (sample, style) cell nobody ran
# shows empty panels that read as a pipeline failure rather than as "not part
# of this study".
#
# Runs on the HOST, not in the container: the viewer needs Flask (host
# python3.10 has it) and only reads artifacts off the shared filesystem. The
# generating pipeline is the opposite -- it needs playwright and ffmpeg, which
# live in the container.
set -euo pipefail

cd /fsxvision_new/venkat.kesav/img_2_svg_pretraining
PORT="${1:-8606}"

# All five study cells. The comparison is pipeline-vs-pipeline (Gemini 3.7
# Flash against Qwen3.8-27B); the GT reference animation is still shown as a
# panel, but is not scored as a third entity.
CELLS="CVPR_2025_arch00012:progressive_reveal,\
CVPR_2025_arch00389:colour_pop,\
CVPR_2025_arch00554:hopping_bounding_box,\
CVPR_2025_arch00855:alpha_masking,\
CVPR_2025_arch01215:progressive_reveal,\
CVPR_2025_arch01224:sliding_bounding_box,\
CVPR_2025_arch01225:progressive_reveal,\
CVPR_2025_arch01311:sliding_bounding_box,\
CVPR_2025_arch01541:alpha_masking,\
CVPR_2025_arch01707:hopping_bounding_box,\
CVPR_2025_arch01709:sliding_bounding_box,\
CVPR_2025_arch01871:progressive_reveal,\
Paper2Fig_pipe_diff_000000536:progressive_reveal,\
Paper2Fig_pipe_diff_000000542:hopping_bounding_box,\
Paper2Fig_pipe_diff_000000673:progressive_reveal,\
Paper2Fig_pipe_diff_000000677:progressive_reveal,\
Paper2Fig_pipe_diff_000000690:progressive_reveal,\
Paper2Fig_pipe_diff_000000714:hopping_bounding_box,\
arch_arxiv_ai_2025_000000126:sliding_bounding_box,\
arch_arxiv_ai_2025_000000127:colour_pop,\
arch_arxiv_ai_2025_000000151:hopping_bounding_box,\
arch_arxiv_cv_2023_000000298:hopping_bounding_box,\
arch_arxiv_cv_2023_000000303:hopping_bounding_box,\
arch_arxiv_db_ir_2024_000000546:progressive_reveal,\
arch_arxiv_db_ir_2025_000000580:progressive_reveal,\
arch_arxiv_db_ir_2025_000000587:progressive_reveal,\
arch_arxiv_db_ir_2025_000000589:hopping_bounding_box,\
arch_arxiv_db_ir_2025_000000611:progressive_reveal,\
arch_arxiv_db_ir_2025_000000618:colour_pop,\
arch_arxiv_db_ir_2026_000000714:colour_pop,\
arch_arxiv_db_ir_2026_000000715:sliding_bounding_box,\
arch_arxiv_db_ir_2026_000000718:progressive_reveal,\
CVPR_2025_pipe00004:progressive_reveal,\
CVPR_2025_pipe00010:progressive_reveal,\
CVPR_2025_pipe00011:progressive_reveal,\
CVPR_2025_pipe00041:progressive_reveal,\
CVPR_2025_pipe00045:progressive_reveal,\
CVPR_2025_pipe00137:progressive_reveal"

# Only the animation-quality suite, and only the five metrics the decision
# rests on. Code / XML / Sequence / Animation(stage-3) columns are not shown --
# they are about earlier stages and are not what this study is deciding.
SUITES="animation"
METRICS="vfs_band,ascs_video,sss,gps,omission_rate"

mkdir -p logs/viewers

# The revised metrics are written to their own evals root (see
# scripts/run_study_metrics.sh on why the live tree is not touched). The viewer
# has to be pointed at the SAME root or the Scoreboard and Metrics tabs show a
# blank row for a cell that is fully scored -- which reads as "the metric did
# not run" rather than "the viewer is looking somewhere else".
export ANIMATEBENCH_EVALS_ROOT=$PWD/data/animatebench_v3_cache/animatebench_v3/evals_study

PYTHONPATH=$PWD/src nohup python3 -m img_2_svg_pretraining.pipeline.inspector.compare \
  --port "$PORT" \
  --configs "Gemini 3.7 Flash=bench_v3_or_svg.yaml,Qwen3.8-27B=bench_v3_or_svg_qwen38_27b.yaml,Gemma-4-26B=bench_v3_or_svg_gemma4_26b.yaml,Zeroshot Gemini 3.1=bench_v3_zeroshot_gemini31.yaml" \
  --cells "$CELLS" \
  --extra-roots "/fsxvision_new/venkat.kesav/img_2_svg_pretraining/data/animatebench_v2" \
  --suites "$SUITES" --metrics "$METRICS" \
  > "logs/viewers/study_${PORT}.log" 2>&1 &

echo "study viewer starting on :$PORT (log logs/viewers/study_${PORT}.log)"
