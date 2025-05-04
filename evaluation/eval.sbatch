#!/bin/bash
#SBATCH --job-name=eval_llada
#SBATCH --output=logs/eval_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL

# 1) Load conda & cache once
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

cd /n/fs/vl/anlon/COS484-LLaDA/evaluation
source ../scripts/cache_setup.sh

# 2) GPQA few‐shot under FSDP
echo "GPQA few-shot (FSDP full-shard across 2 GPUs)"
accelerate launch \
  --num_processes 2 \
  --mixed_precision fp16 \
  --main_process_port 29501 \
  --fsdp=FULL_SHARD \
  eval_llada.py \
    --tasks gpqa_main_n_shot \
    --num_fewshot 5 \
    --model llada_dist \
    --batch_size 2 \
    --model_args "model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=True,mc_num=1"

# 3) BigCode fused under the same FSDP setup
echo "BigCode fused (FSDP full-shard across 2 GPUs)"
accelerate launch \
  --num_processes 2 \
  --mixed_precision fp16 \
  --main_process_port 29501 \
  --fsdp=FULL_SHARD \
  bigcode-evaluation-harness/main.py \
    --model llada_dist \
    --tasks humaneval,mbpp,multiple-py \
    --n_samples 200,15,200 \
    --max_length_generation 512 \
    --temperature 0.2,0.1,0.2 \
    --batch_size 2 \
    --allow_code_execution
