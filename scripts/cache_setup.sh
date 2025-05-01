#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Cache setup script for pip, HuggingFace, and environment mix under Conda
# Uses persistent environment-level cache to avoid ephemeral /tmp
# Usage: source scripts/cache_setup.sh
# -----------------------------------------------------------------------------

# Ensure this script is run inside an activated Conda environment
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: CONDA_PREFIX is not set. Please activate your Conda environment first."
  return 1
fi

# Persistent cache location under the Conda environment prefix
env_cache_root="$CONDA_PREFIX/hf_cache"

# Create cache directories if missing
mkdir -p "$env_cache_root/pip_cache" \
         "$env_cache_root/pip_build" \
         "$env_cache_root/transformers" \
         "$env_cache_root/hub" \
         "$env_cache_root/datasets" \
         "$env_cache_root/metrics"

echo "Using persistent cache at $env_cache_root"

# 1) Override XDG cache home for HF and other libraries to use persistent cache\export XDG_CACHE_HOME="$env_cache_root"

# 2) Pip cache and build directories
export PIP_CACHE_DIR="$env_cache_root/pip_cache"
export TMPDIR="$env_cache_root/pip_build"
echo "PIP_CACHE_DIR set to $PIP_CACHE_DIR"
echo "TMPDIR set to $TMPDIR"

# 3) Transformers cache\export TRANSFORMERS_CACHE="$env_cache_root/transformers"
echo "TRANSFORMERS_CACHE set to $TRANSFORMERS_CACHE"

# 4) HuggingFace hub cache
export HF_HOME="$env_cache_root"
export HUGGINGFACE_HUB_CACHE="$env_cache_root/hub"
echo "HUGGINGFACE_HUB_CACHE set to $HUGGINGFACE_HUB_CACHE"

# 5) Datasets and metrics caches
export HF_DATASETS_CACHE="$env_cache_root/datasets"
export HF_METRICS_CACHE="$env_cache_root/metrics"
echo "HF_DATASETS_CACHE set to $HF_DATASETS_CACHE"
echo "HF_METRICS_CACHE set to $HF_METRICS_CACHE"

cat <<EOF

Persistent cache setup complete! 
EOF