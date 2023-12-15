import os
import argparse
import torch
from torch import nn
import numpy as np
import random
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args

def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    ### model ###
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 400) # should be equal to len(train_dataset.classes)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    if args.rank == 0:
        wandb.init(mode="online", project="bird_example_arnes", entity="janezk", name="bird_example_multi_gpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root="/d/hpc/projects/FRI/DL/example/bird_data/train/", transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_dataset = torchvision.datasets.ImageFolder(root="/d/hpc/projects/FRI/DL/example/bird_data/valid/", transform=transform)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    torch.backends.cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss()

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        validate(val_loader, model, criterion, epoch, args)

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    print(f"Epoch {epoch}, Rank {args.rank}")
    for i,batch in enumerate(train_loader):
        image_batch = batch[0].cuda(args.gpu, non_blocking=True)
        annotation_batch = batch[1].cuda(args.gpu, non_blocking=True)
        output = model(image_batch)
        loss = criterion(output, annotation_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(val_loader, model, criterion, epoch, args):
    counter = torch.zeros((2,), device=torch.device(f'cuda:{args.gpu}'))
    model.eval()
    first_batch_images = None
    first_batch_predictions = None
    first_batch_gt = None
    for i,batch in enumerate(val_loader):
        image_batch = batch[0].cuda(args.gpu, non_blocking=True)
        annotation_batch = batch[1].cuda(args.gpu, non_blocking=True)
        output = model(image_batch)
        labels = torch.argmax(output, dim=1)
        counter[0] += torch.sum((labels == annotation_batch).float())
        counter[1] += image_batch.shape[0]
        if args.rank == 0 and i==0:
            first_batch_images = image_batch
            first_batch_predictions = labels
            first_batch_gt = annotation_batch

    dist.reduce(counter, 0)
    if args.rank == 0:
        pred_table = wandb.Table(columns=["Image", "Prediction", "GT"])
        for j in range(first_batch_images.shape[0]):
            pred_class_name = val_loader.dataset.classes[first_batch_predictions[j].item()]
            gt_class_name = val_loader.dataset.classes[first_batch_gt[j].item()]
            row = [wandb.Image(first_batch_images[j].detach().cpu().numpy().transpose((1, 2, 0))),
                   pred_class_name, gt_class_name]
            pred_table.add_data(*row)
        print("Epoch ",epoch," Cls. Acc: ", (counter[0]/counter[1]).item())
        wandb.log({"Val. CA": (counter[0]/counter[1]), "Val. Table": pred_table}, step=epoch)

if __name__ == '__main__':
    args = parse_args()
    main(args)
