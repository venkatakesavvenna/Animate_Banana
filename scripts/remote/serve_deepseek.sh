#!/bin/bash
# Serve DeepSeek-V4-Flash-Vision-Exp on ONE node, 8xH100.
#
# WHY THE vllm-openai IMAGE AND NOT THE v5serve VENV. The venv's python is a
# symlink chain ending at /usr/bin/python3.12, which most images on this
# cluster do not have -- it fails as "No such file or directory" on the venv's
# OWN python, which reads like a broken venv and is actually a missing
# interpreter. vllm/vllm-openai:nightly ships its own python + vLLM 0.26.1 +
# torch 2.13/cu130 and needs no venv at all.
#
# --kv-cache-dtype fp8 IS MANDATORY, not a tuning knob. DeepSeek-V4's MLA
# attention uses the `fp8_ds_mla` KV layout, which asserts on anything else:
#     AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache,
#     got auto
# All 8 workers die at init and the API server exits with the generic
# "Engine core initialization failed", which names no cause -- the real line is
# only in the per-worker ERROR output.
#
# WEIGHTS LIVE ON NVMe, NOT LUSTRE. /fsxvision_new was 100% full (395G free);
# 168GB does not fit and a partial write there stalls every other job.
set -u
SNAP=/opt/dlami/nvme/vkds/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/6821d6ad3681a4b137b066b76094fa82ebd0a380
NAME=${NAME:-ds-serve}
PORT=${PORT:-8012}

docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --gpus all --ipc=host --network host \
  -v /opt/dlami/nvme/vkds:/opt/dlami/nvme/vkds \
  -e HF_HOME=/opt/dlami/nvme/vkds/hf \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e VLLM_LOGGING_LEVEL=INFO \
  --entrypoint python3 vllm/vllm-openai:nightly \
    -m vllm.entrypoints.openai.api_server \
    --model "$SNAP" \
    --served-model-name deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8 \
    --limit-mm-per-prompt.image 2 \
    --max-num-seqs 16 \
    --trust-remote-code \
    --port "$PORT" --host 0.0.0.0
echo "started $NAME on :$PORT"
