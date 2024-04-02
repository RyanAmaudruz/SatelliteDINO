#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --output=ssl4eo_dataset_new_s2c.out
#SBATCH --job-name=ssl4eo_dataset_new_s2c
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate obdet

srun python src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset_new_s2c.py
