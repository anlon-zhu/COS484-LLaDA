#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=prefetch_%j.log

set -euo pipefail

# 1) Activate Conda & persistent cache setup
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada
source scripts/cache_setup.sh    # sets CACHE_ROOT="$CONDA_PREFIX/hf_cache"

# 2) Prepare a fresh local‐scratch cache
LOCAL_CACHE="/tmp/$(whoami)/hf_cache"
echo "🗂  Clearing & creating local cache at $LOCAL_CACHE"
rm -rf "$LOCAL_CACHE"
mkdir -p "$LOCAL_CACHE"

# 3) Download & mmap‐load the model shards
echo "⏱  Downloading + loading LLaDA-8B into GPU via local cache…"
python - <<'PYCODE'
import time, torch
from transformers import AutoModel

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
cache_dir = "$LOCAL_CACHE"

start = time.time()
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=cache_dir,            # download here
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True          # mmap+stream to GPU
)
device = torch.device("cuda")
model.to(device)
torch.cuda.synchronize()
print(f"✅ Loaded in {time.time() - start:.2f}s via {cache_dir}")
PYCODE

# 4) Sync local shards → persistent hub cache
PERSISTENT_HUB="$CACHE_ROOT/hub"
echo "🔁  Syncing local cache → $PERSISTENT_HUB"
mkdir -p "$PERSISTENT_HUB"
rsync -a --info=progress2 "$LOCAL_CACHE/hub/" "$PERSISTENT_HUB/"

echo "🎉  Prefetch complete!  
Interactive sessions can now load instantly from the persistent cache under $PERSISTENT_HUB.  
Example:
  conda activate llada
  source scripts/cache_setup.sh
  python chat.py --model_name GSAI-ML/LLaDA-8B-Instruct --torch_dtype bfloat16 --trust_remote_code"
