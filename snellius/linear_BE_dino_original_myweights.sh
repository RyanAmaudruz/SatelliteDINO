#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=linear_BE_dino_original_myweights.out
#SBATCH --job-name=linear_BE_dino_original_myweights
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

# # define available gpus
# export CUDA_VISIBLE_DEVICES=0,1,2,3

source activate obdet

srun python -m torch.distributed.launch src/benchmark/transfer_classification/linear_BE_dino_original_myweights.py
