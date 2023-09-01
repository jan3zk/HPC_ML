#!/bin/sh
#SBATCH --reservation=fe
#SBATCH --job-name=train_birds_1node_1gpu
#SBATCH --output=train_birds_1node_1gpu.out
#SBATCH --time=05:00:00       # job time limit
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=16    # number of allocated cores
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=12G     # memory allocation

#module load PyTorch/1.7.1-fosscuda-2020b

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate py37                 # activate the previously created environment

OUT_PATH=/ceph/grid/home/$USER/delavnica3/bird_data/

# Run the training script twice with different hyperparameters.
srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python train_1node_1gpu.py --gpu 0 --lr 1e-4 --epochs 10 --batch_size 64 --out_path $OUT_PATH &
#srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python train.py --gpu 0 --lr 1e-3 --epochs 50 --batch_size 64 --out_path $OUT_PATH &
#nvidia-smi --loop=100

wait
