#!/bin/bash
# Full Gemini run: stage 1 (incl. critic) once, then stages 2-3 per style,
# then metrics. Stage 1 is style-independent so it is not repeated.
CFG=src/img_2_svg_pretraining/pipeline/configs/bench_gemini.yaml
echo "######## STAGE 1 (convert -> rasters -> critic) ########"
python -u -m img_2_svg_pretraining.pipeline.run_pipeline stage1 --config $CFG 2>&1 \
  | grep -E --line-buffered "^(===|  ok|  WARN|  FAIL)"

for S in progressive_reveal colour_pop alpha_masking hopping_bounding_box sliding_bounding_box; do
  echo "######## $S : pipeline ########"
  python -u -m img_2_svg_pretraining.pipeline.run_pipeline all --config $CFG --style $S \
    --from strategize --force 2>&1 | grep -E --line-buffered "^(===|  ok|  WARN|  FAIL)"
  echo "######## $S : metrics ########"
  python -u -m img_2_svg_pretraining.animatebench.run_eval all --config $CFG --style $S \
    --force 2>&1 | grep -E --line-buffered "^(===|  stage1|  xml|  sequence|  stage3)"
done
echo "######## DONE ########"
