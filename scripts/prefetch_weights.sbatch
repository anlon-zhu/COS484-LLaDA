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

export HF_HOME="$LOCAL_CACHE"
export HUGGINGFACE_HUB_CACHE="$LOCAL_CACHE/hub"
export XDG_CACHE_HOME="$LOCAL_CACHE"


# 2.5) Wipe out any old model shards in the persistent cache
PERSISTENT_DIR="$CACHE_ROOT/hub/models--GSAI-ML--LLaDA-8B-Instruct"
if [[ -d "$PERSISTENT_DIR" ]]; then
  echo "🧹 Removing old persistent cache at $PERSISTENT_DIR"
  rm -rf "$PERSISTENT_DIR"
fi

# 4) Download & mmap‐load the model shards into local cache
echo "⏱  Downloading + mmap-loading fresh shards into local cache…"
python - <<'PYCODE'
import time
from transformers import AutoModel

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
cache_dir = "$LOCAL_CACHE"

start = time.time()
AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True
)
print(f"✅ Fresh shards fetched in {time.time() - start:.2f}s")
PYCODE

# 5) Sync local shards → persistent cache
echo "🔁  Syncing fresh shards back → $CACHE_ROOT/hub"
rsync -a --info=progress2 "$LOCAL_CACHE/hub/" "$CACHE_ROOT/hub/"

echo "🎉  Prefetch complete!"
echo "Interactive sessions can now load directly from the persistent cache under:"
echo "  $CACHE_ROOT/hub/models--GSAI-ML--LLaDA-8B-Instruct"
