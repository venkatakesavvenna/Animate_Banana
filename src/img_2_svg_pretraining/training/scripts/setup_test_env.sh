#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${1:-${ROOT_DIR}/.venv-tests}"
TORCH_FLAVOR="${2:-cpu}"

case "${TORCH_FLAVOR}" in
  cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  cu126)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
    ;;
  *)
    echo "Unsupported torch flavor: ${TORCH_FLAVOR}"
    echo "Expected one of: cpu, cu126"
    exit 1
    ;;
esac

python3 -m venv "${ENV_PATH}"
source "${ENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
python -m pip install -r "${ROOT_DIR}/requirements-test.txt"

echo "Test environment ready at ${ENV_PATH}"
python --version
