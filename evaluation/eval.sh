accelerate launch eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

# === BigCode Eval Harness ===
# HumanEval (pass@1,10,100)
accelerate launch bigcode-evaluation-harness/main.py \
  --model llada_dist \
  --tasks humaneval \
  --max_length_generation 512 \
  --temperature 0.2 \
  --n_samples 200 \
  --batch_size 10 \
  --allow_code_execution

# MBPP (pass@1)
accelerate launch bigcode-evaluation-harness/main.py \
  --model llada_dist \
  --tasks mbpp \
  --max_length_generation 512 \
  --temperature 0.1 \
  --n_samples 15 \
  --batch_size 5 \
  --allow_code_execution

# MultiPL-E (multiple languages) without Docker (requires language compilers installed in PATH)
accelerate launch bigcode-evaluation-harness/main.py \
  --model llada_dist \
  --tasks multiple-py \
  --max_length_generation 512 \
  --temperature 0.2 \
  --n_samples 200 \
  --batch_size 10 \
  --allow_code_execution
