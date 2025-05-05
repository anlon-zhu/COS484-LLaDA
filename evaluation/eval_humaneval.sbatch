#!/bin/bash
#SBATCH --job-name=eval_humaneval
#SBATCH --output=logs/%j_eval_humaneval.out
#SBATCH --time=24:00:00
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --gres=gpu:a6000:5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=END,FAIL

export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

cd /n/fs/vl/anlon/COS484-LLaDA/evaluation
source ../scripts/cache_setup.sh

export HF_ALLOW_CODE_EVAL="1"

accelerate launch \
  --multi_gpu \
  --num_processes 5 \
  eval_llada.py \
  --tasks humaneval \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size 4 \
  --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',cfg=0.0,is_check_greedy=True,mc_num=16 \
  --confirm_run_unsafe_code \
  --output_path ../results/humaneval_instruct_zero_shot.json