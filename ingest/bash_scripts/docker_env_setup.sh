#!/bin/bash

# RUN as: bash docker_env_setup.sh

USER_NAME="srihari.bandarupalli"  # Docker user name (must match Dockerfile user)
IMAGE_NAME="vlm-ingest-pipeline-${USER_NAME}"
CONTAINER_NAME="vlm-ingest-pipeline-${USER_NAME}"

# Python Environment Path (HOST path - full path on the server)
# This will be split into: Directory (for Docker mount) and Name (for init_env.sh inside container)
ENV_PATH="/fsxvision_new/${USER_NAME}/environments/vllm_env"

# Option 2: Shared environment for all users (uncomment to use)
# ENV_PATH="/fsxvision/shared_environments/vlm_ingest_pipeline_env"

# Extract directory and environment name from ENV_PATH
ENVIRONMENT_MOUNT="$(dirname "$ENV_PATH")"  # e.g., /fsxvision/user/environments
ENV_NAME="$(basename "$ENV_PATH")"          # e.g., vlm_ingest_pipeline_env

# Mount Paths
CODE_MOUNT="/fsxvision_new/${USER_NAME}/Vision-Ingestion-Engine"

# Data Mounts (mounted as their own paths inside container)
DATAMOUNT1="/fsxvision"
DATAMOUNT2="/fsxvision_new"
DATAMOUNT3="/opt/dlami/nvme"
# DATAMOUNT4="/home/raghuveer.r/Bench/data" # some custom data directory that u might need
# dont mount entire home directory as /home inside docker is different and if u mount entire home, it will mess up permissions and stuff. Instead mount only the specific data directories u need.

# Cache Directories (using fast NVMe storage). We can maybe setup a common cache later.
HF_CACHE_HOST="/opt/dlami/nvme/${USER_NAME}/hf_cache"
TMP_CACHE="/opt/dlami/nvme/${USER_NAME}/tmp"
VLLM_CACHE="/opt/dlami/nvme/${USER_NAME}/vllm_cache"
TRITON_CACHE="/opt/dlami/nvme/${USER_NAME}/triton_cache"



# --- Initialize Docker Container and Environment ---
bash init_docker.sh \
    "$USER_NAME" \
    "$IMAGE_NAME" \
    "$CONTAINER_NAME" \
    "$CODE_MOUNT" \
    "$DATAMOUNT1" \
    "$DATAMOUNT2" \
    "$DATAMOUNT3" \
    "$DATAMOUNT4" \
    "$HF_CACHE_HOST" \
    "$TRITON_CACHE" \
    "$TMP_CACHE" \
    "$VLLM_CACHE" \
    "$ENVIRONMENT_MOUNT"

# --- Initialize Environment inside Docker container ---
docker exec -i $CONTAINER_NAME /bin/bash << EOF
cd /code/bash_scripts
source init_env.sh "$ENV_NAME"
EOF

echo "Docker container and environment setup completed."
