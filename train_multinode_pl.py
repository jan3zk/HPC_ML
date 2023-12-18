import argparse
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import wandb

class BirdClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, batch_size=16, workers=24, out_path='./bird_data/', epochs=10):
        super(BirdClassifier, self).__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.workers = workers
        self.out_path = out_path
        self.epochs = epochs

        self.model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 400)

        self.predictions = []
        self.labels = []

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    #def validation_step(self, batch, batch_idx):
    #    x, y = batch
    #    y_hat = self.forward(x)
    #    val_loss = self.criterion(y_hat, y)
    #    self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.ImageFolder(root=self.out_path + "train/", transform=transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        val_dataset = torchvision.datasets.ImageFolder(root=self.out_path + "valid/", transform=transform)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss)

        # Store predictions and labels for later use
        self.predictions.append(torch.argmax(y_hat, dim=1))
        self.labels.append(y)

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            all_predictions = torch.cat(self.predictions)
            all_labels = torch.cat(self.labels)
    
            num_images_to_log = min(len(all_predictions), 10)
            indices = torch.randperm(len(all_predictions))[:num_images_to_log]
    
            images_to_log = []
            dataset = self.val_dataloader().dataset
    
            for idx in indices:
                # Ensure the index is within the valid range of the dataset
                if idx < len(dataset):
                    image = dataset[idx][0]
                    pred_class_name = dataset.classes[all_predictions[idx].item()]
                    gt_class_name = dataset.classes[all_labels[idx].item()]
    
                    images_to_log.append(wandb.Image(image, caption=f"Pred: {pred_class_name}, GT: {gt_class_name}"))
    
            if images_to_log:
                wandb.log({'Validation Images': images_to_log})

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--out_path', default='./bird_data/', type=str)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
args = parser.parse_args()

# WandB initialization
wandb.init(project="bird_example", entity="janezk", name="bird_example_multinode_pl")

# Init LightningModule
model = BirdClassifier(lr=args.lr, batch_size=args.batch_size, out_path=args.out_path, epochs=args.epochs, workers=args.workers)

# Init Trainer for DDP
trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp", num_nodes=2, max_epochs=args.epochs)
trainer.fit(model)