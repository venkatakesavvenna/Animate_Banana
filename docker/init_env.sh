#!/bin/bash
set -e

ENV_NAME="$1"
if [ -z "$ENV_NAME" ]; then
    echo "Usage: init_env.sh <env_name>"
    exit 1
fi

ENV_PATH="/environments/${ENV_NAME}"

if [ -d "$ENV_PATH" ]; then
    echo "✓ Environment $ENV_NAME already exists at $ENV_PATH."
else
    echo "Creating virtual environment at $ENV_PATH..."
    python3 -m venv --system-site-packages "$ENV_PATH"
    echo "✓ Virtual environment created."
fi

source "$ENV_PATH/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing img_2_svg_pretraining (editable) and its dependencies..."
pip install -e /code

deactivate

echo "✓ Environment $ENV_NAME is ready."
