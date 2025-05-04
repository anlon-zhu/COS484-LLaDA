#!/bin/bash
#SBATCH --job-name=eval_llada
#SBATCH --output=logs/eval_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # ← use 1 GPU
#SBATCH --ntasks-per-node=1     # ← single process
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL

# — Load conda once —
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

# — Warm up the cache just once —
cd /n/fs/vl/anlon/COS484-LLaDA/evaluation
source ../scripts/cache_setup.sh

# — GPQA few‐shot (single GPU) —
echo "GPQA few-shot evaluation"
accelerate launch \
  --mixed_precision fp16 \
  eval_llada.py \
    --tasks gpqa_main_n_shot \
    --num_fewshot 5 \
    --model llada_dist \
    --batch_size 2 \
    --model_args "model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=32"

# — BigCode fused evaluation (single GPU) —
echo "BigCode fused evaluation"
accelerate launch \
  --mixed_precision fp16 \
  bigcode-evaluation-harness/main.py \
    --model llada_dist \
    --tasks humaneval,mbpp,multiple-py \
    --n_samples 200,15,200 \
    --max_length_generation 512 \
    --temperature 0.2,0.1,0.2 \
    --batch_size 2 \
    --allow_code_execution
