#!/usr/bin/env bash
# Viewer for the ablation suite: one panel per ablation config.
#
# Reuses the comparison viewer unchanged -- each ablation config carries its own
# cache_root, so pointing the viewer at the eight configs as eight "models"
# resolves each panel to that ablation's own export. No new viewer code.
set -euo pipefail
cd /fsxvision_new/venkat.kesav/img_2_svg_pretraining
PORT="${1:-8607}"
CELLS="${CELLS:-CVPR_2025_arch01225:progressive_reveal}"
mkdir -p logs/viewers
PYTHONPATH=$PWD/src nohup python3 -m img_2_svg_pretraining.pipeline.inspector.compare \
  --port "$PORT" \
  --configs "AnimateBanana (full)=gemini_ablation_full.yaml,\
S1 no Critic=gemini_ablation_stage1_no_critic.yaml,\
S2 no Image=gemini_ablation_stage2_no_image.yaml,\
Seq no XML=gemini_ablation_sequencer_no_xml.yaml,\
S2 no Critic=gemini_ablation_stage2_no_critic.yaml,\
Narr no Context=gemini_ablation_narration_no_context.yaml,\
Designer no Image=gemini_ablation_designer_no_image.yaml,\
S3 no Critic=gemini_ablation_stage3_no_critic.yaml" \
  --cells "$CELLS" \
  > "logs/viewers/ablation_${PORT}.log" 2>&1 &
echo "ablation viewer starting on :$PORT"
