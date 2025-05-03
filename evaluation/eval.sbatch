#!/bin/bash
#SBATCH --job-name=eval_llada
#SBATCH --output=logs/eval_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=ALL

# Load environment if needed
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

cd /n/fs/vl/anlon/COS484-LLaDA/
source scripts/cache_setup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd evaluation

# Few-shot and zero-shot evaluations
echo "GPQA few-shot evaluation"
accelerate launch --mixed_precision fp16 eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist \
  --batch_size 4 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=64

# accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_dist \
#   --batch_size 4 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --model llada_dist \
#   --batch_size 4 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --model llada_dist \
#   --batch_size 4 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --model llada_dist \
#   --batch_size 4 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --model llada_dist \
#   --batch_size 4 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist \
#   --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

# accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --model llada_dist \
#   --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

# accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --model llada_dist \
#   --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

# === BigCode Eval Harness ===
echo "BigCode eval"
accelerate launch bigcode-evaluation-harness/main.py \
  --model llada_dist --tasks humaneval --max_length_generation 512 \
  --temperature 0.2 --n_samples 200 --batch_size 4 \
  --allow_code_execution

accelerate launch bigcode-evaluation-harness/main.py \
  --model llada_dist --tasks mbpp --max_length_generation 512 \
  --temperature 0.1 --n_samples 15 --batch_size 4 \
  --allow_code_execution

accelerate launch bigcode-evaluation-harness/main.py \
  --model llada_dist --tasks multiple-py --max_length_generation 512 \
  --temperature 0.2 --n_samples 200 --batch_size 4 \
  --allow_code_execution
