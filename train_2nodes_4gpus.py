import os
import argparse
import torch
from torch import nn
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='./bird_data/', type=str)
    #parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    args = parser.parse_args()
    return args


def main(args):

    # Initialize the distributed environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    if dist.get_rank() == 0:
        wandb.init(project="bird_example_arnes", entity="janezk", name="bird_example_2nodes_4gpus")
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }

    model = torchvision.models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 400)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Use DistributedDataParallel instead of DataParallel
    model = model.cuda()
    model = DistributedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root=args.out_path+"train/", transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, drop_last=True, sampler=train_sampler)

    val_dataset = torchvision.datasets.ImageFolder(root=args.out_path+"valid/", transform=transform)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, drop_last=False, sampler=val_sampler)


    criterion = torch.nn.CrossEntropyLoss()

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        #print("Epoch ",epoch)
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image_batch = batch[0].cuda()
            annotation_batch = batch[1].cuda()
            output = model(image_batch)
            loss = criterion(output, annotation_batch)
            loss.backward()
            optimizer.step()

        validate(val_loader, model, epoch, local_rank)

def validate(val_loader, model, epoch, local_rank):
    counter = torch.zeros((2,), device=torch.device(f'cuda:{local_rank}'))
    model.eval()
    first_batch_images = None
    first_batch_predictions = None
    first_batch_gt = None
    for i,batch in enumerate(val_loader):
        image_batch = batch[0].cuda(local_rank, non_blocking=True)
        annotation_batch = batch[1].cuda(local_rank, non_blocking=True)
        output = model(image_batch)
        labels = torch.argmax(output, dim=1)
        counter[0] += torch.sum((labels == annotation_batch).float())
        counter[1] += image_batch.shape[0]
        if local_rank == 0 and i==0:
            first_batch_images = image_batch
            first_batch_predictions = labels
            first_batch_gt = annotation_batch
    dist.reduce(counter, 0)
    if dist.get_rank() == 0:
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
