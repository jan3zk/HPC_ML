#!/bin/sh
#SBATCH --job-name=train_birds_1node_2gpus
#SBATCH --output=train_birds_1node_2gpus.out
#SBATCH --time=01:00:00       # job time limit
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24    # number of allocated cores
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-gpu=32G     # memory allocation
#SBATCH --nodelist=wn223

#module load PyTorch/1.7.1-fosscuda-2020b

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate py310                 # activate the previously created environment

OUT_PATH=/d/hpc/projects/FRI/DL/example/bird_data/

# Run the training script twice with different hyperparameters.
srun --nodes=1 --gpus=2 --ntasks=1 python train_1node_2gpus.py --lr 1e-4 --epochs 10 --batch_size 256 --out_path $OUT_PATH
