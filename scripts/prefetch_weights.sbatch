#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=prefetch_%j.log

set -euo pipefail

# 1) Activate Conda & persistent-cache env
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada
source scripts/cache_setup.sh   # now CACHE_ROOT="$CONDA_PREFIX/hf_cache"

# 2) Find the persistent HF‐hub model dir
HF_HUB="$CONDA_PREFIX/hf_cache/hub"
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
PERSISTENT_DIR=$(find "$HF_HUB" -type d -path "*models--${MODEL_NAME//\//--}*" | head -n1)

if [[ ! -d "$PERSISTENT_DIR" ]]; then
  echo "❌ Cannot find persisted shards under $HF_HUB"
  exit 1
fi

echo "📂  Persistent shards found at: $PERSISTENT_DIR"

# 3) Rsync directly to fast local scratch
LOCAL_DIR="/tmp/$(whoami)/models--${MODEL_NAME//\//--}"
echo "🚀  Staging shards → $LOCAL_DIR"
rm -rf "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR"
# --whole-file avoids delta checks; -z enables compression if your NFS benefits
rsync -az --info=progress2 "$PERSISTENT_DIR/" "$LOCAL_DIR/"

# 4) (Optional) quick sanity check that all 6 shards are there
echo "📝  Local shard count: $(ls -1 $LOCAL_DIR/*.safetensors | wc -l) / expected 6"

# 5) Sync back to persistent cache (no-op if nothing changed)
echo "🔁  Syncing local → persistent (fast NFS copy)"
rsync -az --info=progress2 "$LOCAL_DIR/" "$PERSISTENT_DIR/"

echo "✅  Prefetch complete."
echo "In your interactive shell, just point your model at:"
echo "  --model_name $LOCAL_DIR"
