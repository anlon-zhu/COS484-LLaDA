#!/usr/bin/env bash
cd "$(dirname \"$0\")"

sbatch eval_mbpp.sbatch
sbatch eval_mbpp_3shot.sbatch
sbatch eval_humaneval.sbatch
sbatch eval_humaneval_3shot.sbatch

# === CodeGLUE ===
sbatch eval_codexglue_python.sbatch
sbatch eval_codexglue_java.sbatch
sbatch eval_codexglue_javascript.sbatch
sbatch eval_codexglue_go.sbatch
