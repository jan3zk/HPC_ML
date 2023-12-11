# Efficient usage of HPC for deep learning - Workshop examples

## Bird classification example

Data available at https://www.kaggle.com/datasets/gpiosenka/100-bird-species.

### Running on a single GPU

### Parallelization using Pytorch DataParallel (obsolete)

An example of training using DataParallel (an older approach to data parallelism) (https://pytorch.org/tutorials/beginner/ddp_series_theory.html#why-you-should-prefer-ddp-over-dataparallel-dp)
train_1node_2gpus.sh
train_1node_2gpus.py

### Parallelization using Pytorch DistributedDataParallel

train_multinode.sh: SLURM script to launch the training job