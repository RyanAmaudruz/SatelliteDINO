#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=pretrain_dino_s2c.out
#SBATCH --job-name=pretrain_dino_s2c
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate obdet

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

srun python src/benchmark/pretrain_ssl/pretrain_dino_s2c_neworiginal.py
