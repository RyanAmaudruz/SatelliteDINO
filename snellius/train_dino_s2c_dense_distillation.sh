#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=train_dino_s2c_dense_distillation.out
#SBATCH --job-name=train_dino_s2c_dense_distillation
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate obdet

srun python src/benchmark/pretrain_ssl/pretrain_dino_s2c_dense_distillation.py
