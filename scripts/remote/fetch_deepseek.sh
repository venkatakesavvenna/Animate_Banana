#!/bin/bash
# Stage DeepSeek-V4-Flash-Vision-Exp to NVMe on the serving node.
#
# TO NVMe, NEVER LUSTRE. /fsxvision_new was 100% full (395G free) on 2026-09-06;
# a 168GB checkpoint does not fit and a partial download there would also stall
# every other job on the filesystem. /opt/dlami/nvme has 14T.
#
# Runs INSIDE the vllm image: the host has no huggingface_hub (and host python
# is 3.10). The image ships its own hf_transfer-capable hub client.
set -u
DEST=/opt/dlami/nvme/vkds/hf
mkdir -p "$DEST"
docker run --rm \
  -v /opt/dlami/nvme/vkds:/opt/dlami/nvme/vkds \
  -e HF_HOME="$DEST" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  --entrypoint bash vllm/vllm-openai:nightly -c "
    pip install -q hf_transfer 2>/dev/null
    python3 -c \"
from huggingface_hub import snapshot_download
p=snapshot_download('deepseek-ai/DeepSeek-V4-Flash-Vision-Exp',
                    max_workers=16, resume_download=True)
print('DONE', p)
\"
  "
