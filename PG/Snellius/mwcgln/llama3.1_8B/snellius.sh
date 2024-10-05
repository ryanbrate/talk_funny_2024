#!/bin/bash
#SBATCH --job-name=PG_8B
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00

# load modules 
module load 2023
module load  Miniconda3/23.5.2-0

# load the conda env
source ~/.bashrc
conda activate bigone

# run program
python score_with_llama3.py
