#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=prefetch_%j.log

# -----------------------------------------------------------------------------
# Prefetch HuggingFace model shards:
#   1) Stage to node-local /tmp for fast I/O
#   2) Time a low_cpu_mem_usage load onto GPU
#   3) Sync the staged shards back to persistent cache
# -----------------------------------------------------------------------------

set -euo pipefail

# 1) Activate Conda & cache setup
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada
source scripts/cache_setup.sh   # defines CACHE_ROOT="$CONDA_PREFIX/hf_cache"

# 2) Define paths
PERSISTENT_DIR="$CONDA_PREFIX/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct"
LOCAL_DIR="/tmp/$(whoami)/LLaDA-8B-Instruct"

# 3) Stage shards to local scratch
echo "📦  Staging shards from persistent cache → local scratch"
mkdir -p "$LOCAL_DIR"
rsync -a --info=progress2 "$PERSISTENT_DIR/" "$LOCAL_DIR/"

# 4) Time the memory-mapped load
echo "⏱  Timing model load from local scratch ($LOCAL_DIR)"
python - <<'PYCODE'
import time, torch
from transformers import AutoModel

model_path = "/tmp/$(whoami)/LLaDA-8B-Instruct"
start = time.time()

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
device = torch.device("cuda")
model.to(device)
torch.cuda.synchronize()
print(f"✅ Loaded in {time.time()-start:.2f}s from {model_path}")
PYCODE

# 5) Sync staged shards back to persistent cache
echo "🔁  Syncing staged shards back to persistent cache"
rsync -a --info=progress2 "$LOCAL_DIR/" "$PERSISTENT_DIR/"

echo "🎉  Prefetch complete. In interactive sessions, point to:"
echo "    --model_name $LOCAL_DIR"
