module load anaconda3/2024.6
conda create -p /n/fs/vl/anlon/envs/llada python=3.10 pip -y  
conda activate /n/fs/vl/anlon/envs/llada

pip install --upgrade pip
pip install \
  transformers==4.38.2 \
  torch accelerate \
  lm_eval==0.4.5 \
  gradio \
  huggingface_hub
  
