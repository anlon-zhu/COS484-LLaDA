from setuptools import setup, find_packages

setup(
    name='llada',
    version='0.1.0',
    packages=find_packages(),  # includes evaluation/
    py_modules=['generate', 'chat'],  # top-level scripts
    python_requires='>=3.8',
    install_requires=[
        'accelerate',
        'torch',
        'transformers',
        'datasets',
        'lm-eval',
        'tqdm',
        'numpy',
    ],
)
