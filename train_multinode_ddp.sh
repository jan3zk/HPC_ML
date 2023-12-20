#!/bin/bash
#SBATCH --job-name=multinode_example
#SBATCH --output=out_train_example2.out
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
## primer za 2 noda, vsak po 2 GPU-ji == Učenje na 4 GPU-jih
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks=4 #per job
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32gb

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate py310                 # activate the previously created environment

export MASTER_PORT=50321
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
echo $MASTER_ADDR
echo $WORLD_SIZE
echo "NODELIST="${SLURM_NODELIST}
echo "JOB_NODELIST="${SLURM_JOB_NODELIST}

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes

OUT_PATH=/d/hpc/projects/FRI/DL/example/bird_data/

### the command to run
srun python train_multinode_ddp.py --lr 1e-4 --epochs 10 --batch_size 64 --out_path $OUT_PATH

#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#export NCCL_BLOCKING_WAIT=1
