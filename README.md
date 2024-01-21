# Efficient Usage of HPC for Deep Learning - Workshop Examples

You can install all required libraries using the following command:
```bash
conda env create -f environment.yml
```

## Bird Classification Example

Data is available at the [100 Bird Species Dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

### Task Parallelization on a Single Node

The scripts [train_singlenode.sh](train_singlenode.sh) and [train_singlenode.py](train_singlenode.py) demonstrate running multiple training tasks in parallel with distinct parameters. Each task operates independently on a separate GPU. Execute the example using the following command:
```bash
sbatch train_singlenode.sh
```
Details for each `srun` experiment can be found at [Run 1](https://wandb.ai/janezk/bird_example_arnes/runs/0asl9ekr/) and [Run 2](https://wandb.ai/janezk/bird_example_arnes/runs/77qvmk0m). The expected running time is approximately 30 minutes on Nvidia V100s GPU.

### Data Parallelization on a Single Node using PyTorch DataParallel ([constrained](https://pytorch.org/tutorials/beginner/ddp_series_theory.html#why-you-should-prefer-ddp-over-dataparallel-dp))

The scripts [train_singlenode_dp.sh](train_singlenode_dp.sh) and [train_singlenode_dp.py](train_singlenode_dp.py) illustrate data parallelization using the DataParallel approach. PyTorch's DataParallel replicates the model across available GPUs within a single machine, dividing input data into smaller batches for parallel processing on each GPU. It independently computes gradients on these batches, aggregates them, and synchronizes the model's parameters across all GPUs after each update. Run the example using the following command:
```bash
sbatch train_singlenode_dp.sh
```
Details are available at [Run](https://wandb.ai/janezk/bird_example_arnes/runs/s4amb7l6/). The expected running time is approximately 20 minutes on Nvidia V100s GPU.

### Data Parallelization on Multiple Nodes using PyTorch DistributedDataParallel

The scripts [train_multinode_ddp.sh](train_multinode_ddp.sh) and [train_multinode_ddp.py](train_multinode_ddp.py) illustrate data parallelization using the DistributedDataParallel approach. DistributedDataParallel allows efficient parallelization across multiple nodes. It distributes the workload among different nodes, leveraging collective communication operations to manage gradient synchronization and model parameter updates efficiently. This approach significantly reduces the communication overhead between nodes, enabling faster training and scaling for larger models and datasets.

Utilizing PyTorch's DistributedDataParallel incurs less overhead compared to DataParallel. Execute the following command to run the example on multiple nodes:
```bash
sbatch train_multinode_ddp.sh
```
Refer to the experiment's details at [Run](https://wandb.ai/janezk/bird_example_arnes/runs/trrp5ou3). The expected running time is approximately 10 minutes on Nvidia V100s GPU.
