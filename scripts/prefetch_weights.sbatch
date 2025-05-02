#!/bin/bash
#SBATCH --job-name=prefetch_weights
#SBATCH --output=prefetch_weights.%j.out
#SBATCH --error=prefetch_weights.%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL

set -euo pipefail

# activate conda environment
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

source "$(dirname "$0")/cache_setup.sh"

# prefetch weights for Base and Instruct
python - <<PYCODE
from huggingface_hub import snapshot_download
snapshot_download(repo_id="GSAI-ML/LLaDA-8B-Base", cache_dir="$CACHE_DIR")
snapshot_download(repo_id="GSAI-ML/LLaDA-8B-Instruct", cache_dir="$CACHE_DIR")
PYCODE

echo "  Prefetch complete!  
Interactive sessions can now load instantly from the persistent cache under $CACHE_DIR.  
Example:
  conda activate llada
  source scripts/cache_setup.sh
  python chat.py --model_name GSAI-ML/LLaDA-8B-Instruct --torch_dtype bfloat16 --trust_remote_code"
