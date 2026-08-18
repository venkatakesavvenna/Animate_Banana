#!/usr/bin/env bash
set -e

# =============================================================================
# Python Environment Initialization Script
# =============================================================================
# This script runs INSIDE the Docker container.
# It receives the environment NAME (not full path) from run_pipeline.sh.
# The /environments directory is mounted from the host.
#
# Usage: init_env.sh <env_name>
# Example: init_env.sh vision_ingestion_engine_env
#
# This will create/activate: /environments/<env_name>

if [ -z "$1" ]; then
  echo "❌ Error: No environment name provided"
  echo "Usage: $0 <env_name>"
  echo "Example: $0 vision_ingestion_engine_env"
  exit 1
fi

ENV_NAME="$1"
ENV_PATH="/environments/${ENV_NAME}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Python Environment Initialization (Inside Container)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Environment name: ${ENV_NAME}"
echo "Container path:   ${ENV_PATH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

if [ -d "${ENV_PATH}" ]; then
  echo "🔹 Environment already exists — activating..."
else
  echo "🟢 Environment not found — creating new venv..."
  python3 -m venv "${ENV_PATH}"
  echo "✅ Created virtual environment at ${ENV_PATH}"

  # Optional: install base packages
  "${ENV_PATH}/bin/pip" install --upgrade pip setuptools wheel
  "${ENV_PATH}/bin/pip" install pydantic psutil nvitop jupyter ipython packaging
  "${ENV_PATH}/bin/pip" install "psycopg[binary,pool]"
  "${ENV_PATH}/bin/pip" install vllm --extra-index-url https://download.pytorch.org/whl/cu128
  "${ENV_PATH}/bin/pip" install ninja
  MAX_JOBS=48 "${ENV_PATH}/bin/pip" install flash-attn --no-build-isolation -v

  # "${ENV_PATH}/bin/pip" install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126 
  # "${ENV_PATH}/bin/pip" install --upgrade flashinfer-python 
  # "${ENV_PATH}/bin/pip" install flashinfer-cubin
  # "${ENV_PATH}/bin/pip" install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
  
  # Install the vision-ingest package in editable mode
  echo "📦 Installing vision-ingest package in editable mode..."
  "${ENV_PATH}/bin/pip" install -e /code
fi

echo
echo "🚀 Activating environment..."
# shellcheck disable=SC1090
source "${ENV_PATH}/bin/activate"

# Add auto-activation to .bashrc if not already present
BASHRC="$HOME/.bashrc"
ACTIVATION_LINE="source ${ENV_PATH}/bin/activate"

if [ -f "$BASHRC" ]; then
  if ! grep -qF "$ACTIVATION_LINE" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# Auto-activate Python environment" >> "$BASHRC"
    echo "$ACTIVATION_LINE" >> "$BASHRC"
    echo "✅ Added auto-activation to ~/.bashrc"
  else
    echo "ℹ️  Auto-activation already in ~/.bashrc"
  fi
else
  echo "# Auto-activate Python environment" > "$BASHRC"
  echo "$ACTIVATION_LINE" >> "$BASHRC"
  echo "✅ Created ~/.bashrc with auto-activation"
fi

# Detect and add PostgreSQL binaries to PATH
PG_VERSION=$(ls -d /usr/lib/postgresql/* 2>/dev/null | head -1 | xargs basename)
if [ -n "$PG_VERSION" ]; then
  PG_PATH_LINE="export PATH=\$PATH:/usr/lib/postgresql/$PG_VERSION/bin"
  if [ -f "$BASHRC" ]; then
    if ! grep -qF "postgresql" "$BASHRC"; then
      echo "" >> "$BASHRC"
      echo "# PostgreSQL binaries" >> "$BASHRC"
      echo "$PG_PATH_LINE" >> "$BASHRC"
      echo "✅ Added PostgreSQL $PG_VERSION binaries to PATH"
    else
      echo "ℹ️  PostgreSQL PATH already in ~/.bashrc"
    fi
  fi
  # Also export immediately for current session
  export PATH=$PATH:/usr/lib/postgresql/$PG_VERSION/bin
else
  echo "⚠️  PostgreSQL not found in /usr/lib/postgresql/"
fi

echo "👍 Done. Active Python: $(which python)"
echo "   Env path: ${ENV_PATH}"
echo "   Future docker exec sessions will auto-activate this environment"
