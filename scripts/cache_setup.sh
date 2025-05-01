#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Cache setup script for pip, HuggingFace, and environment mix under Conda
# Usage: source scripts/cache_setup.sh
# -----------------------------------------------------------------------------

# Ensure this script is run inside an activated Conda environment
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: CONDA_PREFIX is not set. Please activate your Conda environment first."
  return 1
fi

# Base cache root under your env prefix
export CACHE_ROOT="$CONDA_PREFIX/hf_cache"

# 1) XDG cache override (for HuggingFace hub and others)
export XDG_CACHE_HOME="$CACHE_ROOT"

# 2) Pip cache and build directories
export PIP_CACHE_DIR="$CACHE_ROOT/pip_cache"
export TMPDIR="$CACHE_ROOT/pip_build"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

echo "PIP_CACHE_DIR set to $PIP_CACHE_DIR"
echo "TMPDIR set to $TMPDIR"

# 3) Transformers cache
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
mkdir -p "$TRANSFORMERS_CACHE"
echo "TRANSFORMERS_CACHE set to $TRANSFORMERS_CACHE"

# 4) HuggingFace hub cache
# (HF_HOME may be respected, but XDG_CACHE_HOME covers default ~/.cache)
export HF_HOME="$CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
echo "HUGGINGFACE_HUB_CACHE set to $HUGGINGFACE_HUB_CACHE"

# 5) Datasets and metrics caches
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export HF_METRICS_CACHE="$CACHE_ROOT/metrics"
mkdir -p "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE"
echo "HF_DATASETS_CACHE set to $HF_DATASETS_CACHE"
echo "HF_METRICS_CACHE set to $HF_METRICS_CACHE"

cat <<EOF

Cache setup complete!  Run your Python script now (in the same shell):
  python chat.py --model_name ... --torch_dtype ... --trust_remote_code
EOF
