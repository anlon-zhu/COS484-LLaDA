#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Cache setup script for pip and HuggingFace when using Conda environments
# Usage: activate your conda env, then run:
#    bash scripts/cache_setup.sh
# -----------------------------------------------------------------------------

# Ensure this script is run inside an activated Conda environment
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: CONDA_PREFIX is not set. Please activate your Conda environment first."
  exit 1
fi

# 1) Pip cache and build directories under the Conda env prefix
export PIP_CACHE_DIR="$CONDA_PREFIX/pip_cache"
export TMPDIR="$CONDA_PREFIX/pip_build"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

echo "Set PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "Set TMPDIR=$TMPDIR"

# 2) HuggingFace cache directories under the Conda env prefix
export HF_HOME="$CONDA_PREFIX/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE"

echo "Set HF_HOME=$HF_HOME"
echo "Set TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "Set HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "Set HF_METRICS_CACHE=$HF_METRICS_CACHE"

# 3) Confirmation
cat <<EOF

Cache setup complete!  You can now run your Python scripts without hitting
"No space left on device" errors.
EOF
