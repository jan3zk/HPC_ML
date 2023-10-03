import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torchvision
import torchvision.transforms as transforms
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='./bird_data/', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    device = torch.device("cuda")
    
    wandb.init(project="bird_example_arnes", entity="janezk", name=f"bird_example_2nodes_4gpus")
    wandb.config.update(args)
    
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 400)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)
    model = DDP(model, device_ids=[dist.get_rank() % torch.cuda.device_count()])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root=args.out_path+"train/", transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, drop_last=True, sampler=train_sampler)

    val_dataset = torchvision.datasets.ImageFolder(root=args.out_path+"valid/", transform=transform)
    #val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        print(f"Epoch {epoch}, Rank {dist.get_rank()}")
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image_batch, annotation_batch = batch[0].to(device), batch[1].to(device)
            output = model(image_batch)
            loss = criterion(output, annotation_batch)
            loss.backward()
            optimizer.step()

        validate(val_loader, model, device, epoch)

def validate(val_loader, model, device, epoch):
    model.eval()
    tp_sum = 0
    cnt = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            image_batch, annotation_batch = batch[0].to(device), batch[1].to(device)
            output = model(image_batch)
            # ... rest of the validation logic remains same
            labels = torch.argmax(output, dim=1)
            tp_sum += torch.sum((labels == annotation_batch).float())
            cnt += image_batch.shape[0]
            if i==0:
                first_batch_images = image_batch
                first_batch_predictions = labels
                first_batch_gt = annotation_batch

    pred_table = wandb.Table(columns=["Image","Prediction","GT"])
    for j in range(first_batch_images.shape[0]):
        pred_class_name = val_loader.dataset.classes[first_batch_predictions[j].item()]
        gt_class_name = val_loader.dataset.classes[first_batch_gt[j].item()]
        row = [wandb.Image(first_batch_images[j].detach().cpu().numpy().transpose((1, 2, 0))),
               pred_class_name, gt_class_name]
        pred_table.add_data(*row)
    print("Cls. Acc: ",(tp_sum/cnt).item())
    wandb.log({"Val. CA": tp_sum/cnt, "Val. Table":pred_table})

if __name__ == '__main__':
    main()
