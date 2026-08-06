#!/bin/bash
# Run one config across all 5 bench styles, both samples.
# --from strategize when stage 1 is shared via `code_from`.
CFG=$1
FROM=${2:-convert-code}
for S in progressive_reveal colour_pop alpha_masking hopping_bounding_box sliding_bounding_box; do
  echo "############ $CFG :: $S ############"
  python -u -m img_2_svg_pretraining.pipeline.run_pipeline all \
    --config src/img_2_svg_pretraining/pipeline/configs/$CFG.yaml \
    --style $S --from $FROM 2>&1 | grep -E --line-buffered "^(===|  ok|  WARN|  FAIL)"
done
echo "############ DONE $CFG ############"
