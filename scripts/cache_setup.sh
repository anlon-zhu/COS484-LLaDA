#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Cache setup script for pip, HuggingFace, and environment mix under Conda
# Usage: source scripts/cache_setup.sh
# -----------------------------------------------------------------------------

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
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export HF_METRICS_CACHE="${CACHE_DIR}/metrics"
export HF_MODULES_CACHE="${CACHE_DIR}/modules"

# Create cache directories
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${HF_METRICS_CACHE}" "${HF_MODULES_CACHE}"

cat <<EOF

Cache setup complete!  Run your Python script now in this same shell:
  python chat.py --model_name ... --torch_dtype ... --trust_remote_code
EOF
