#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=linear_BE_dino.out
#SBATCH --job-name=linear_BE_dino
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate obdet

srun python -m torch.distributed.launch src/benchmark/transfer_classification/linear_BE_dino.py
