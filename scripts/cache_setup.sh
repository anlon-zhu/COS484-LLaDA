#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Cache setup script for pip, HuggingFace, and environment mix under Conda
# Usage: source scripts/cache_setup.sh
# -----------------------------------------------------------------------------

# Add loading of .env from project root for HUGGINGFACE_HUB_TOKEN
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/../.env"
if [[ -f "$ENV_FILE" ]]; then
  # Export variables from .env (ignoring comments)
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: CONDA_PREFIX is not set. Please activate your Conda environment first."
  return 1
fi

# Determine base cache directory (override by setting CACHE_DIR env var)
export CACHE_DIR="/n/fs/vl/anlon/model_cache"
echo "Using cache directory: $CACHE_DIR"

# Set cache environment variables
export XDG_CACHE_HOME="${CACHE_DIR}"
export PIP_CACHE_DIR="${CACHE_DIR}/pip_cache"
export TMPDIR="${CACHE_DIR}/pip_build"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export HF_METRICS_CACHE="${CACHE_DIR}/metrics"
export HF_MODULES_CACHE="${CACHE_DIR}/modules"
# export ACCELERATE_LOG_LEVEL=DEBUG
# export TRANSFORMERS_VERBOSITY=debug
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"

# Create cache directories
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${HF_METRICS_CACHE}" "${HF_MODULES_CACHE}" "${TRITON_CACHE_DIR}"

# Login to HuggingFace Hub if token provided
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HUGGINGFACE_HUB_TOKEN}"
fi

cat <<EOF

Cache setup complete!
EOF
