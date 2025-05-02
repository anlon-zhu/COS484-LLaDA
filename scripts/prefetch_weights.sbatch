#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=prefetch_%j.log

set -euo pipefail

# 1) Activate your llada Conda env
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

# 2) Set up caches (persistent only)
source scripts/cache_setup.sh    # defines CACHE_ROOT="$CONDA_PREFIX/hf_cache"

# 3) Find the on-disk model cache via HF API
echo "🔍 Locating persistent cache directory..."
PERSISTENT_DIR=$(python - <<'PYCODE'
from huggingface_hub import snapshot_download
# local_files_only=True returns the existing cache path instead of redownloading
path = snapshot_download(
    repo_id="GSAI-ML/LLaDA-8B-Instruct",
    local_files_only=True
)
print(path)
PYCODE
)

echo "🏠  Persistent cache found at: $PERSISTENT_DIR"

# 4) Stage into fast local scratch
LOCAL_DIR="/tmp/$(whoami)/LLaDA-8B-Instruct"
echo "📦  Staging shards to local scratch → $LOCAL_DIR"
rm -rf "$LOCAL_DIR"            # start clean
mkdir -p "$LOCAL_DIR"
rsync -a --info=progress2 "$PERSISTENT_DIR/" "$LOCAL_DIR/"

# 5) Time a memory-mapped, low-CPU-mem load
echo "⏱  Timing model load from local scratch..."
python - <<'PYCODE'
import time, torch
from transformers import AutoModel

model_path = "$LOCAL_DIR"
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

print(f"✅ Model loaded in {time.time() - start:.2f}s from {model_path}")
PYCODE

# 6) Sync back to persistent cache
echo "🔁  Syncing staged shards back to persistent cache"
rsync -a --info=progress2 "$LOCAL_DIR/" "$PERSISTENT_DIR/"

echo "🎉  Prefetch complete."
echo " In interactive sessions, point your script at:"
echo "   --model_name $LOCAL_DIR"
