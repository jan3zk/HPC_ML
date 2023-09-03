#!/bin/sh
#SBATCH --job-name=train_birds_1node_1gpu
#SBATCH --output=train_birds_1node_1gpu.out
#SBATCH --time=05:00:00       # job time limit
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=2   # number of tasks
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12    # number of allocated cores
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G     # memory allocation

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate py310                 # activate the previously created environment

OUT_PATH=/d/hpc/projects/FRI/DL/example/bird_data/

srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python train_1node_1gpu.py --lr 1e-4 --epochs 10 --batch_size 128 --out_path $OUT_PATH &
srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python train_1node_1gpu.py --lr 1e-3 --epochs 10 --batch_size 128 --out_path $OUT_PATH &

wait
