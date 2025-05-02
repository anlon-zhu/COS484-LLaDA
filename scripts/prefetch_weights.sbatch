#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=prefetch_%j.log

# 1) Set up Conda env
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

# 2) Set up all caches under your env prefix (persistent)
source scripts/cache_setup.sh

# 3) Stage the already-downloaded shards into local scratch
LOCAL_MODEL_DIR="/tmp/$(whoami)/LLaDA-8B-Instruct"
echo "Staging model shards → $LOCAL_MODEL_DIR"
mkdir -p "$LOCAL_MODEL_DIR"
rsync -a --info=progress2 \
  "$CONDA_PREFIX/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/" \
  "$LOCAL_MODEL_DIR/"

# 4) Time a memory-mapped load directly onto GPU
echo "Timing model load from local scratch..."
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
# move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# ensure all transfers complete
torch.cuda.synchronize()

elapsed = time.time() - start
print(f"✅ Model loaded in {elapsed:.2f} seconds from {model_path}")
PYCODE

echo "Done prefetch job. In your interactive session, point to:"
echo "  --model_name /tmp/$(whoami)/LLaDA-8B-Instruct"
