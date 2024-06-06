#!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=18
# #SBATCH -N 1
# #SBATCH --gpus=1
# #SBATCH --nodelist=node[404]
# #SBATCH --partition=defq
# #SBATCH --time=60:00:00
# #SBATCH --output=linear_BE_dino_original_myweights.out
# #SBATCH --job-name=linear_BE_dino_original_myweights
# #SBATCH --exclude=gcn45,gcn59x


# #!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --ntasks-per-node=8

# Execute program located in $HOME

# # define available gpus
# export CUDA_VISIBLE_DEVICES=0,1,2,3

source activate obdet

srun unzip /var/node433/local/ryan_a/data/leopart_ssl.zip -d /var/node433/local/ryan_a/data
