#!/bin/bash

source /environments/docgrounding_env/bin/activate

# Load wandb API key from api_keys folder
if [ -f "/code/api_keys/wandb_key" ]; then
    export WANDB_API_KEY=$(cat /code/api_keys/wandb_key)
fi

export HF_HUB_OFFLINE=True
# export WANDB_PROJECT="Qwen_Sam_Debug" # give a project name under which all runs are grouped

# export NCCL_IB_DISABLE=1
# export NCCL_NET=Socket
# # optional but often good:
# export NCCL_SOCKET_IFNAME=eth0   # or whatever `ip -br a` shows

# Clear multi-node networking settings
# unset NCCL_NET
# unset FI_PROVIDER
# unset FI_EFA_USE_DEVICE_RDMA
# unset NCCL_IB_DISABLE

# Optional debug
export NCCL_DEBUG=WARN

# Distributed Training Configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-"8"}

cd /code/src/img_2_svg_pretraining/training
export PYTHONPATH=/code/src
entry_file="-m img_2_svg_pretraining.training.training_core.train.train"

# Launch training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file}  2>&1 | tee /code/src/img_2_svg_pretraining/training/logs/train_$(date +"%Y%m%d_%H%M%S").log