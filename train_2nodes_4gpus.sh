#!/bin/sh
#SBATCH --job-name=train_birds_2nodes_4gpus
#SBATCH --output=train_birds_2nodes_4gpus.out

#SBATCH --time=05:00:00       # job time limit
#SBATCH --nodes=2             # number of nodes
#SBATCH --ntasks-per-node=2   # number of tasks
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12    # number of allocated cores
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=32G     # memory allocation

#module load PyTorch/1.7.1-fosscuda-2020b

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate py310                 # activate the previously created environment

OUT_PATH=/d/hpc/projects/FRI/DL/example/bird_data/

# Run the training script twice with different hyperparameters.
srun --nodes=2 --exclusive --gpus=2 --ntasks=2 torchrun --nproc_per_node=2 train_2nodes_4gpus.py --lr 1e-4 --epochs 10 --batch_size 64 --out_path $OUT_PATH
