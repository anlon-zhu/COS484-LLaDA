#!/usr/bin/env bash
set -euo pipefail

# 1) load the module
module load anaconda3/2024

# 2) define your env prefix and name
ENV_PREFIX=/n/fs/vl/anlon/envs/llada
ENV_NAME=llada

# 3) create if it doesn’t already exist
if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "Creating Conda env at $ENV_PREFIX ..."
  conda create -p "$ENV_PREFIX" python=3.10 pip -y
else
  echo "Conda env already exists at $ENV_PREFIX, skipping creation."
fi

# 4) point Conda at your shared envs dir and activate by name
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
# adjust this path if your sysadmin installs Anaconda elsewhere
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# 5) upgrade pip & install your packages
pip install --upgrade pip
pip install \
  transformers==4.38.2 \
  torch accelerate \
  lm_eval==0.4.5 \
  gradio \
  huggingface_hub

echo "✅ llada env is ready and activated."
