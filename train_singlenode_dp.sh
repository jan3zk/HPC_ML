#!/bin/sh
#SBATCH --job-name=train_singlenode_dp
#SBATCH --output=train_singlenode_dp.out
#SBATCH --time=01:00:00       # job time limit
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24    # number of allocated cores
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-gpu=32G     # memory allocation

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate pytorch_env                 # activate the previously created environment

OUT_PATH=/d/hpc/projects/FRI/DL/example/bird_data/

# Run the training script
srun --nodes=1 --gpus=2 --ntasks=1 python train_singlenode_dp.py --lr 1e-4 --epochs 4 --batch_size 128 --out_path $OUT_PATH
