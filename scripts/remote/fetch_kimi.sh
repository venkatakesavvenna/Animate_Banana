#!/bin/bash
# Stage Kimi-K2.6 (595GB) to NVMe for local serving.
#
# LOCAL, NOT OpenRouter: Kimi via API is ~$40/set (measured 6 calls/cell at
# 0.95/4.00 per M) -- $101 for all three sets against a $30 balance. Served
# locally it is free, which is the only way all three sets fit.
#
# NVMe, NEVER Lustre: /fsxvision_new was 100% full (395G free) on 2026-09-06.
set -u
DEST=/opt/dlami/nvme/vkkimi/hf
mkdir -p "$DEST"
docker run --rm \
  -v /opt/dlami/nvme/vkkimi:/opt/dlami/nvme/vkkimi \
  -e HF_HOME="$DEST" \
  --entrypoint bash vllm/vllm-openai:nightly -c "
    python3 -c \"
from huggingface_hub import snapshot_download
p=snapshot_download('moonshotai/Kimi-K2.6', max_workers=16)
print('DONE', p)
\"
  "
