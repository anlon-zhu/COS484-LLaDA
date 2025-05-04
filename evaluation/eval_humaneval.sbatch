#!/bin/bash
#SBATCH --job-name=eval_llada
#SBATCH --output=logs/eval_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=END,FAIL

# 1) Load conda & cache once
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

cd /n/fs/vl/anlon/COS484-LLaDA/evaluation
source ../scripts/cache_setup.sh

# 3) Prime registry and run BigCode harness
export PYTHONPATH=/n/fs/vl/anlon/COS484-LLaDA:$PYTHONPATH
python -c "import evaluation.eval_llada"
accelerate launch \
  bigcode-evaluation-harness/main.py \
  --model llada_dist \
  --tasks humaneval \
  --n_samples 20 \
  --max_length_generation 512 \
  --temperature 0.2 \
  --batch_size 1 \
  --allow_code_execution
