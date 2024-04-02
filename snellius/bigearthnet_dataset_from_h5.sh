#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=bigearthnet_dataset_from_h5.out
#SBATCH --job-name=bigearthnet_dataset_from_h5
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate obdet

srun python src/benchmark/transfer_classification/datasets/BigEarthNet/bigearthnet_dataset_from_h5.py