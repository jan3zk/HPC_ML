#!/bin/sh
#SBATCH --job-name=train_birds_2nodes_4gpus
#SBATCH --output=train_birds_2nodes_4gpus.out
#SBATCH --time=05:00:00       # job time limit
#SBATCH --nodes=2             # number of nodes
#SBATCH --ntasks=4            # number of tasks
#SBATCH --ntasks-per-node=2   # number of tasks per node
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8     # number of allocated cores
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G     # memory allocation

export OMP_NUM_THREADS=8

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate py310                 # activate the previously created environment

OUT_PATH=/d/hpc/projects/FRI/DL/example/bird_data/

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --partition=gpu -w $head_node hostname -I | awk '{print $1}')

echo Node $head_node IP: $head_node_ip

srun torchrun --nnodes 2 --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint $head_node_ip:34759 train_2nodes_4gpus_arnes.py --lr 1e-4 --epochs 10 --batch_size 64 --out_path $OUT_PATH
