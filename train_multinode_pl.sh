#!/bin/bash
#SBATCH --job-name=train_multinode_pl  # Job name
#SBATCH --output=train_multinode_pl.out  # Output file
#SBATCH --time=01:00:00  # Time limit
#SBATCH --nodes=2  # Number of nodes
#SBATCH --ntasks-per-node=2  # Number of tasks per node
#SBATCH --partition=gpu  # Partition
#SBATCH --cpus-per-task=12  # Number of allocated cores per task
#SBATCH --gpus-per-task=1  # Number of GPUs per task
#SBATCH --mem-per-gpu=32G  # Memory allocation per GPU

source ~/miniconda3/etc/profile.d/conda.sh  # Activate Conda environment if needed
conda activate py310  # Activate the necessary environment

# Assuming the script with PyTorch Lightning is named your_lightning_script.py
srun --ntasks=4 --gpus-per-task=1 python train_multinode_pl1.py --out_path /d/hpc/projects/FRI/DL/example/bird_data/ --lr 1e-4 --epochs 10 --batch_size 64
