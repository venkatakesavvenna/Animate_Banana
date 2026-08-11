#!/bin/bash
# Full v2 run: stage 1 (convert -> rasters -> critic) once, then stages 2-3
# per style with all critics on, then metrics.
CFG=src/img_2_svg_pretraining/pipeline/configs/bench_v2_gemini.yaml
echo "######## STAGE 1 ########"
python -u -m img_2_svg_pretraining.pipeline.run_pipeline stage1 --config $CFG 2>&1 \
  | grep -E --line-buffered "^(===|  ok|  WARN|  FAIL)"
for S in progressive_reveal colour_pop alpha_masking hopping_bounding_box sliding_bounding_box; do
  echo "######## $S pipeline ########"
  python -u -m img_2_svg_pretraining.pipeline.run_pipeline all --config $CFG --style $S \
    --from strategize 2>&1 | grep -E --line-buffered "^(===|  ok|  WARN|  FAIL)"
  echo "######## $S metrics ########"
  python -u -m img_2_svg_pretraining.animatebench.run_eval all --config $CFG --style $S \
    2>&1 | grep -E --line-buffered "^(===|  stage1|  xml|  sequence|  stage3)"
done
echo "######## DONE ########"
