#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=prefetch_%j.log

set -euo pipefail

# 1) Activate your environment and persistent cache
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada
source scripts/cache_setup.sh    # CACHE_ROOT="$CONDA_PREFIX/hf_cache"

# 2) Prepare node-local cache dir
LOCAL_CACHE="/tmp/$(whoami)/hf_cache"
echo "🗂  Clearing & creating local cache at $LOCAL_CACHE"
rm -rf "$LOCAL_CACHE"
mkdir -p "$LOCAL_CACHE"

# 3) Prefetch + mmap‐load all shards into local cache
echo "⏱  Downloading + mmap-loading shards into local cache (no GPU transfer)…"
python - <<'PYCODE'
import time
from transformers import AutoModel

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
cache_dir = "$LOCAL_CACHE"

start = time.time()
# low_cpu_mem_usage=True will memory-map shards rather than load full into RAM
AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True
)
print(f"✅ All shards fetched and mmap-loaded in {time.time() - start:.2f}s")
PYCODE

# 4) Sync local cache back into persistent hub cache
PERSISTENT_HUB="$CACHE_ROOT/hub"
echo "🔁  Syncing local cache → $PERSISTENT_HUB"
mkdir -p "$PERSISTENT_HUB"
rsync -a --info=progress2 "$LOCAL_CACHE/hub/" "$PERSISTENT_HUB/"

echo "🎉  Prefetch complete!"
echo "Interactive sessions can now load directly from the persistent cache under:"
echo "  $PERSISTENT_HUB/models--GSAI-ML--LLaDA-8B-Instruct"
