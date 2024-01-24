#!/bin/sh
#SBATCH --job-name=train_singlenode
#SBATCH --output=train_singlenode.out
#SBATCH --time=05:00:00       # job time limit
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2   # number of tasks
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12    # number of allocated cores
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G     # memory allocation

source ~/miniconda3/etc/profile.d/conda.sh  # initialize conda
conda activate pytorch_env  # activate the previously created environment

OUT_PATH=/d/hpc/projects/FRI/DL/example/bird_data/

# Run training twice using different hyperparameters
srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python train_singlenode.py --gpu 0 --lr 1e-4 --epochs 4 --batch_size 64 --out_path $OUT_PATH &
srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python train_singlenode.py --gpu 0 --lr 1e-3 --epochs 4 --batch_size 64 --out_path $OUT_PATH &

wait
