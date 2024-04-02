#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=pretrain_dino_s2a_RYAN.out
#SBATCH --job-name=pretrain_dino_s2a_RYAN
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate obdet

srun python src/benchmark/pretrain_ssl/pretrain_dino_s2a_RYAN.py
