#!/bin/bash
#SBATCH --job-name=multinode_example
#SBATCH --output=train_multinode_ddp.out
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate pytorch_env                 # activate the previously created environment

export MASTER_PORT=50202
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
echo $MASTER_ADDR
echo $WORLD_SIZE
echo "NODELIST="${SLURM_NODELIST}
echo "JOB_NODELIST="${SLURM_JOB_NODELIST}

### the command to run
srun python train_multinode_ddp.py --lr 1e-4 --epochs 4 --batch_size 64
