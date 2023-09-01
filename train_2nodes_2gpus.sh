#!/bin/sh
#SBATCH --job-name=train_birds_two_nodes
#SBATCH --output=train_birds_two_nodes.out
#SBATCH --time=05:00:00       # job time limit
#SBATCH --nodes=2             # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=16    # number of allocated cores
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-gpu=12G     # memory allocation

#module load PyTorch/1.7.1-fosscuda-2020b

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate py37                 # activate the previously created environment

OUT_PATH=/ceph/grid/home/$USER/delavnica3/bird_data/

# Run the training script twice with different hyperparameters.
#srun --nodes=1 --exclusive --gpus=2 --ntasks=1 python train_1node_2gpus.py --lr 1e-4 --epochs 1 --batch_size 128 --out_path $OUT_PATH &
python -m torch.distributed.launch --nproc_per_node=2 train_2nodes_2gpus.py --lr 1e-4 --epochs 1 --batch_size 128 --out_path $OUT_PATH

#wait
