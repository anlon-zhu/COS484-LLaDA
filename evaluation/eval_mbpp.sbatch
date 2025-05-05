#!/bin/bash
#SBATCH --job-name=eval_zero_shot_mbpp
#SBATCH --output=logs/%j_eval_zero_shot_mbpp.out
#SBATCH --time=24:00:00
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-1%2       # split into 2 shards

export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

cd /n/fs/vl/anlon/COS484-LLaDA/evaluation
source ../scripts/cache_setup.sh

export HF_ALLOW_CODE_EVAL="1"

# compute slice bounds
SLOTS=2        # number of array slots

accelerate launch eval_llada.py \
  --tasks mbpp \
  --num_fewshot 0 \
  --model llada_dist \
  --rank $SLURM_ARRAY_TASK_ID \
  --world_size $SLOTS \
  --batch_size 1 \
  --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',cfg=0.0,is_check_greedy=False,mc_num=128 \
  --confirm_run_unsafe_code \
  --output_path ../results/mbpp_instruct_3shot.json