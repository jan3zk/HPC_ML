# Efficient usage of HPC for deep learning - Workshop examples

## Bird classification example

Data available at https://www.kaggle.com/datasets/gpiosenka/100-bird-species.

### Running on a single GPU

```
sbatch train_1node_1gpu_arnes.sh
```

### Parallelization using Pytorch DataParallel ([obsolete](https://pytorch.org/tutorials/beginner/ddp_series_theory.html#why-you-should-prefer-ddp-over-dataparallel-dp))

```
sbatch train_1node_2gpus.sh
```

### Parallelization using Pytorch DistributedDataParallel

```
sbatch train_multinode.sh
```