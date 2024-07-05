#!/bin/bash
#SBATCH --job-name=multinode_example
#SBATCH --output=train_multinode_ddp.out
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate pytorch_env                 # activate the previously created environment

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=50202
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
echo "NODELIST="${SLURM_NODELIST}

OUT_PATH=/d/hpc/projects/FRI/DL/example/bird_data/
srun python train_multinode_ddp.py --lr 1e-3 --epochs 2 --batch_size 16 --out_path $OUT_PATH
