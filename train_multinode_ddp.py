import os
import argparse
import torch
from torch import nn
import numpy as np
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"
WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
WORLD_RANK = int(os.environ['SLURM_PROCID'])
LOCAL_RANK = int(os.environ['SLURM_LOCALID'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--out_path', default='./bird_data/', type=str)
    # DDP configs:
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    args = parser.parse_args()
    return args

def main(args):
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=WORLD_SIZE, rank=WORLD_RANK)

    ### model ###
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 400) # should be equal to len(train_dataset.classes)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    torch.cuda.set_device(LOCAL_RANK)
    model.cuda(LOCAL_RANK)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])
    
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    if WORLD_RANK == 0:
        wandb.init(mode="disabled", project="bird_example",
                   entity=os.environ["USER"], name="bird_example_multinode_ddp")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root=args.out_path+"train/", transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_dataset = torchvision.datasets.ImageFolder(root=args.out_path+"valid/", transform=transform)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    torch.backends.cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss()

    ### main loop ###
    for epoch in range(args.epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        validate(val_loader, model, criterion, epoch)

def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i,batch in enumerate(train_loader):
        image_batch = batch[0].cuda(LOCAL_RANK, non_blocking=True)
        annotation_batch = batch[1].cuda(LOCAL_RANK, non_blocking=True)
        output = model(image_batch)
        loss = criterion(output, annotation_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(val_loader, model, criterion, epoch):
    counter = torch.zeros((2,), device=torch.device(f'cuda:{LOCAL_RANK}'))
    model.eval()
    first_batch_images = None
    first_batch_predictions = None
    first_batch_gt = None
    for i,batch in enumerate(val_loader):
        image_batch = batch[0].cuda(LOCAL_RANK, non_blocking=True)
        annotation_batch = batch[1].cuda(LOCAL_RANK, non_blocking=True)
        output = model(image_batch)
        labels = torch.argmax(output, dim=1)
        counter[0] += torch.sum((labels == annotation_batch).float())
        counter[1] += image_batch.shape[0]
        if WORLD_RANK == 0 and i==0:
            first_batch_images = image_batch
            first_batch_predictions = labels
            first_batch_gt = annotation_batch

    dist.reduce(counter, 0)
    if WORLD_RANK == 0:
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

