from setuptools import setup, find_packages

setup(
    name='llada',
    version='0.1.0',
    packages=find_packages(),  # includes evaluation/
    py_modules=['generate', 'chat'],  # top-level scripts
    python_requires='>=3.8',
    install_requires=[
        'accelerate==4.38.2',
        'torch',
        'bitsandbytes',
        'lm_eval==0.4.8',
        'gradio',
        'huggingface_hub>=0.15.1,<0.29',
        'peft',
    ],
)