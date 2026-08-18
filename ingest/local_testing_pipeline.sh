#!/bin/bash

PROJECT_NAME="test_layout"  # REQUIRED - Give your project a unique name
OUTPUTS="outputs"             # Base directory for different versions(prompts) of outputs

# Constructed Paths (based on PROJECT_NAME and OUTPUTS)
DB_PATH="${PROJECT_NAME}/${OUTPUTS}/db"
LOGS_PATH="${PROJECT_NAME}/${OUTPUTS}/logs"
JSONL_OUTPUT_PATH="${PROJECT_NAME}/${OUTPUTS}/jsonl_outputs"

# MODE SELECTION - "ingest" (add to db), "pipeline" (process from db), "both" (ingest + process), or "full_maintenance" (reset stuck + verify)
MODE="both"  # Options: ingest, pipeline, both, full_maintenance

# Parallel execution - only applicable with MODE="both"
PARALLEL=false  # Set to true to run ingest and pipeline in parallel processes

# For ingest and both modes:
IMAGE_PATHS_SOURCE="images.txt"  # Path to file containing image paths (one per line) OR folder path to walk

# PERFORMANCE TIP: Using a pre-generated file list is MUCH faster than os.walk for large datasets!
# Generate the file list once with:
#   find /path/to/images -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) > images.txt
# This can save hours on datasets with millions of images.

# Processing Configuration
BATCH_SIZE=2
LOCAL_JSON_THREADS=2
FSYNC_EVERY_LINES=2

# VLM Configuration
VLM_MODEL_NAME="qwen_3_32b" # make sure corresponding model full path is set in vllm_module/vllm_config.py
VLM_GPUS="0,1,2,3,4,5,6,7"  # Comma-separated GPU IDs
VLM_CONFIG_PATH="config/vllm_model.yaml" # has corresponding vllm params for full model paths.
PROMPT_PATH="prompts/layout.md"


if [ "$MODE" = "ingest" ] || [ "$MODE" = "pipeline" ] || [ "$MODE" = "both" ]; then
    echo "Running in $MODE mode..."
    echo "Database path: $DB_PATH"
    [ "$MODE" != "pipeline" ] && echo "Image paths source: $IMAGE_PATHS_SOURCE"
    echo ""

    python -m main \
        --mode "$MODE" \
        --db-path "$DB_PATH" \
        ${IMAGE_PATHS_SOURCE:+--image-paths-source "$IMAGE_PATHS_SOURCE"} \
        --logs-path "$LOGS_PATH" \
        --jsonl-output-path "$JSONL_OUTPUT_PATH" \
        --batch-size $BATCH_SIZE \
        --local-json-threads $LOCAL_JSON_THREADS \
        --fsync-every-lines $FSYNC_EVERY_LINES \
        --vlm-model-name "$VLM_MODEL_NAME" \
        --vlm-gpus "$VLM_GPUS" \
        --vlm-config-path "$VLM_CONFIG_PATH" \
        --prompt-path "$PROMPT_PATH" \
        ${PARALLEL:+--parallel}

elif [ "$MODE" = "full_maintenance" ]; then
    echo "Running in FULL_MAINTENANCE mode..."
    echo "Database path: $DB_PATH"
    echo ""
    echo "⚠️  WARNING: Ensure NO other processes on ANY node are accessing this database!"
    echo ""

    python -m drivers.db_driver \
        --db-path "$DB_PATH" \
        --mode full-maintenance

else
    echo "Error: Invalid MODE. Set MODE to either 'ingest', 'pipeline', 'both', or 'full_maintenance'"
    exit 1
fi
