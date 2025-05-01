#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Cache setup script for pip, HuggingFace, and environment mix under Conda
# Supports using local scratch (/tmp) for faster I/O when available
# Usage: source scripts/cache_setup.sh
# -----------------------------------------------------------------------------

# Ensure this script is run inside an activated Conda environment
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: CONDA_PREFIX is not set. Please activate your Conda environment first."
  return 1
fi

# Determine fast I/O scratch if possible (/tmp assumed local)
USER_TMP="/tmp/$(whoami)"
if mkdir -p "$USER_TMP" && [[ -w "$USER_TMP" ]]; then
  SCRATCH_ROOT="$USER_TMP/hf_cache"
  echo "Using local scratch cache at $SCRATCH_ROOT"
else
  SCRATCH_ROOT="$CONDA_PREFIX/hf_cache"
  echo "Local scratch unavailable, using env prefix cache at $SCRATCH_ROOT"
fi

# Base cache root under chosen location
export CACHE_ROOT="$SCRATCH_ROOT"

# 1) Override XDG cache home (for HuggingFace hub & other libs)
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

Cache setup complete! Run your Python script now (in the same shell):
  python chat.py --model_name ... --torch_dtype ... --trust_remote_code
EOF
