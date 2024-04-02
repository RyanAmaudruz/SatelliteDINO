#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=pretrain_moco_s2c_mgpu.out
#SBATCH --job-name=pretrain_moco_s2c_mgpu
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate obdet

srun python -m torch.distributed.launch src/benchmark/pretrain_ssl/pretrain_moco_v3_s2c_original_mgpu.py
