# Efficient usage of HPC for deep learning - Workshop examples

## Bird classification example

Data available at https://www.kaggle.com/datasets/gpiosenka/100-bird-species.

### Task parallelization on a single node

Scripts [train_1node_1gpu_arnes.sh](train_1node_1gpu_arnes.sh) and [train_1node_1gpu_arnes.py](train_1node_1gpu_arnes.py) depict an example of executing multiple training runs in parallel using different training parameters. Each task runs independently on a separate GPU. Run the example using the following command:
```bash
sbatch train_1node_1gpu_arnes.sh
```

### Data parallelization on a single node using Pytorch DataParallel ([obsolete](https://pytorch.org/tutorials/beginner/ddp_series_theory.html#why-you-should-prefer-ddp-over-dataparallel-dp))


Scripts [train_1node_2gpus.sh](train_1node_2gpus.sh) and [train_1node_1gpu_arnes.py](train_1node_2gpus.py) depict an example of data parallelization using DataParallel approach. PyTorch's DataParallel replicates the model across available GPUs within a single machine, dividing input data into smaller batches for parallel processing on each GPU. It independently computes gradients on these batches, aggregates them, and synchronizes the model's parameters across all GPUs after each update. Run the example using the following command:
```bash
sbatch train_1node_2gpus.sh
```

### Data parallelization on multiple nodes using Pytorch DistributedDataParallel

```bash
sbatch train_multinode_sbatch.sh
```
https://wandb.ai/janezk/bird_example_arnes/runs/trrp5ou3


### Data oarallelization on multiple nodes using torchrun instead of srun

```bash
sbatch train_2nodes_4gpus_arnes.sh
```