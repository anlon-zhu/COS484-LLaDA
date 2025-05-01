#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=prefetch_%j.log

export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

source scripts/cache_setup.sh

cd /n/fs/pvl-progen/anlon/COS484-LLaDA/
# force a full download + local‐scratch copy
python - <<'PYCODE'
from transformers import AutoModel
# this will pull shards into CACHE_ROOT (likely /tmp/…)
AutoModel.from_pretrained(
  "GSAI-ML/LLaDA-8B-Instruct",
  trust_remote_code=True,
  torch_dtype=None,
  low_cpu_mem_usage=True
)
PYCODE
