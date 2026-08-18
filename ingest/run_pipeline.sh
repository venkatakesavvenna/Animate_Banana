#!/bin/bash

# =============================================================================
# DOCKER & ENVIRONMENT CONFIGURATION
# =============================================================================
# These variables control Docker container setup and Python environment.
# Modify these to match your infrastructure and user setup.
VLLM_WORKER_MULTIPROC_METHOD="spawn"
# VLLM_ATTENTION_BACKEND="FLASHINFER" #"FLASH_ATTN" 
USER_NAME="srihari.bandarupalli"  # Docker user name (must match Dockerfile user)
IMAGE_NAME="vlm-ingest-pipeline"
CONTAINER_NAME="vlm-ingest-pipeline-${USER_NAME}"

# Python Environment Path (HOST path - full path on the server)
# This will be split into: Directory (for Docker mount) and Name (for init_env.sh inside container)
# Option 1: User-specific environment (default)
ENV_PATH="/fsxvision/${USER_NAME}/environments/vision_ingestion_engine_env"
# Option 2: Shared environment for all users (uncomment to use)
# ENV_PATH="/fsxvision/shared_environments/vision_ingestion_env"

# Extract directory and environment name from ENV_PATH
ENVIRONMENT_MOUNT="$(dirname "$ENV_PATH")"  # e.g., /fsxvision/user/environments
ENV_NAME="$(basename "$ENV_PATH")"          # e.g., vision_ingestion_engine_env

# Mount Paths
CODE_MOUNT="/fsxvision/${USER_NAME}/Vision-Ingestion-Engine"
DATA_MOUNT="/fsxvision"

# Cache Directories (using fast NVMe storage). We can maybe setup a common cache later.
HF_CACHE_HOST="/opt/dlami/nvme/${USER_NAME}/hf_cache"
TMP_CACHE="/opt/dlami/nvme/${USER_NAME}/tmp"
VLLM_CACHE="/opt/dlami/nvme/${USER_NAME}/vllm_cache"
TRITON_CACHE="/opt/dlami/nvme/${USER_NAME}/triton_cache"
# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================
PROJECT_NAME="test_layout"  # REQUIRED - Give your project a unique name
OUTPUTS="layout_v2"             # Base directory for different versions(prompts) of outputs
PROMPT_PATH="prompts/layout_v2.md"

# Module that we are running
MODULE_TO_RUN="examples.example1.main" # python -m examples.example1.main

# Constructed Paths (based on PROJECT_NAME and OUTPUTS)
DB_PATH="${PROJECT_NAME}/${OUTPUTS}/db"
LOGS_PATH="${PROJECT_NAME}/${OUTPUTS}/logs"
JSONL_OUTPUT_PATH="${PROJECT_NAME}/${OUTPUTS}/jsonl_outputs"

# MODE SELECTION - "ingest" (add to db), "pipeline" (process from db), "both" (ingest + process), or "full_maintenance" (reset stuck + verify)
MODE="both"  # Options: ingest, pipeline, both, full_maintenance

# Parallel execution - only applicable with MODE="both"
PARALLEL=false  # Set to true to run ingest and pipeline in parallel processes

# For ingest and both modes:
IMAGE_PATHS_SOURCE="/${CODE_MOUNT}/images.txt"  # Path to file with image paths OR folder to walk

# PERFORMANCE TIP: Using a pre-generated file list is MUCH faster than os.walk for large datasets!
# Generate the file list once with:
#   find /path/to/images -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) > images.txt
# This can save hours on datasets with millions of images. os.walk is VERY slow at scale!

# Processing Configuration
BATCH_SIZE=8
LOCAL_JSON_THREADS=16
FSYNC_EVERY_LINES=8

# VLM Configuration
VLM_MODEL_NAME="qwen_3_235b" # make sure corresponding model full path is set in vllm_module/vllm_config.py
VLM_GPUS="0,1,2,3,4,5,6,7"  # Comma-separated GPU IDs
VLM_CONFIG_PATH="/fsxvision/srihari.bandarupalli/Vision-Ingestion-Engine/src/vision_ingest/config/vllm_model.yaml" # has corresponding vllm params for full model paths.

GRACEFUL_TIMEOUT=5

# =============================================================================
# Initialize Docker Container and Environment
# =============================================================================
# Pass all configuration variables as arguments to init_docker.sh
bash bash_scripts/init_docker.sh \
    "$USER_NAME" \
    "$IMAGE_NAME" \
    "$CONTAINER_NAME" \
    "$CODE_MOUNT" \
    "$DATA_MOUNT" \
    "$HF_CACHE_HOST" \
    "$TRITON_CACHE" \
    "$TMP_CACHE" \
    "$VLLM_CACHE" \
    "$ENVIRONMENT_MOUNT"
# =============================================================================
# OUTER CLEANUP - Forwards Ctrl+C to docker container
# =============================================================================
# CRITICAL: This trap ensures Ctrl+C in the host terminal triggers cleanup
# inside the Docker container. Without this, processes will linger.

DOCKER_EXEC_PID=""

cleanup_outer() {
    echo ""
    echo "Host: Received signal, running in-container cleanup..."

    # 1) Invoke cleanup inside the container explicitly.. kill all the processes with the pattern and also "VLLM"
    docker exec -i "$CONTAINER_NAME" /bin/bash -lc "
        cd /code
        source bash_scripts/cleanup_processes.sh
        PROCESS_PATTERN=\"python -m $MODULE_TO_RUN\"
        cleanup_processes $GRACEFUL_TIMEOUT \"\$PROCESS_PATTERN\"
    " || true

    # 2) Now wait for the main docker exec that was running the pipeline
    if [ -n \"$DOCKER_EXEC_PID\" ] && kill -0 \"$DOCKER_EXEC_PID\" 2>/dev/null; then
        echo \"Host: Waiting for main pipeline exec to exit...\"
        wait \"$DOCKER_EXEC_PID\" 2>/dev/null || true
    fi

    echo \"Host: Cleanup completed\"
}


# Set up the outer trap
trap cleanup_outer EXIT INT TERM

# =============================================================================
# Run commands inside Docker container
# =============================================================================
# The & at the end runs docker exec in background so we can capture its PID
# This allows us to forward signals to it when Ctrl+C is pressed

docker exec -i $CONTAINER_NAME /bin/bash << EOF &
DOCKER_EXEC_PID=\$!

# Source the cleanup function
source bash_scripts/cleanup_processes.sh

# Set up trap with custom timeout and process pattern
# This will cleanup processes matching the example module AND all vLLM processes
PROCESS_PATTERN="python -m $MODULE_TO_RUN"
trap 'cleanup_processes $GRACEFUL_TIMEOUT "\$PROCESS_PATTERN"' EXIT INT TERM

cd /code
source bash_scripts/init_env.sh "$ENV_NAME"

VLLM_WORKER_MULTIPROC_METHOD="spawn"
if [ "$MODE" = "ingest" ] || [ "$MODE" = "pipeline" ] || [ "$MODE" = "both" ]; then
    echo "Running in $MODE mode..."
    echo "Database path: $DB_PATH"
    [ "$MODE" != "pipeline" ] && echo "Image paths source: $IMAGE_PATHS_SOURCE"
    echo ""
    
    python -m $MODULE_TO_RUN \\
        --mode "$MODE" \\
        --db-path "$DB_PATH" \\
        ${IMAGE_PATHS_SOURCE:+--image-paths-source "$IMAGE_PATHS_SOURCE"} \\
        --logs-path "$LOGS_PATH" \\
        --jsonl-output-path "$JSONL_OUTPUT_PATH" \\
        --batch-size $BATCH_SIZE \\
        --local-json-threads $LOCAL_JSON_THREADS \\
        --fsync-every-lines $FSYNC_EVERY_LINES \\
        --vlm-model-name "$VLM_MODEL_NAME" \\
        --vlm-gpus "$VLM_GPUS" \\
        --vlm-config-path "$VLM_CONFIG_PATH" \\
        --prompt-path "$PROMPT_PATH" \\
        $([ "$PARALLEL" = "true" ] && echo "--parallel")

elif [ "$MODE" = "full_maintenance" ]; then
    echo "Running in FULL_MAINTENANCE mode..."
    echo "Database path: $DB_PATH"
    echo ""
    echo "⚠️  WARNING: Ensure NO other processes on ANY node are accessing this database!"
    echo ""
    
    python -m drivers.db_driver \\
        --db-path "$DB_PATH" \\
        --mode full-maintenance

else
    echo "Error: Invalid MODE. Set MODE to either 'ingest', 'pipeline', 'both', or 'full_maintenance'"
    exit 1
fi

EOF

# Capture the PID of the docker exec command
DOCKER_EXEC_PID=$!

# Wait for the docker exec process to complete
# This will be interrupted if cleanup_outer receives a signal
wait $DOCKER_EXEC_PID
EXIT_CODE=$?

echo ""
echo "Pipeline exited with code: $EXIT_CODE"
exit $EXIT_CODE
# =============================================================================
# Alternative: Run individual commands
# =============================================================================
#
# Ingest images:
# python -m drivers.db_driver --db-path "$DB_PATH" --mode ingest --folder "/path/to/images"
#
# Run pipeline:
# python -m drivers.cli --db-path "$DB_PATH" --logs-path "$LOGS_PATH" --jsonl-output-path "$JSONL_OUTPUT_PATH"
#
# Database maintenance:
# python -m drivers.db_driver --db-path "$DB_PATH" --mode full-maintenance
#
# =============================================================================