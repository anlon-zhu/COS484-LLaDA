#!/bin/bash
#SBATCH --job-name=eval_mmlu
#SBATCH --output=logs/eval_mmlu_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=END,FAIL

export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

cd /n/fs/vl/anlon/COS484-LLaDA/evaluation
source ../scripts/cache_setup.sh

accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1