import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from mnist_resnet50 import dataset
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from mnist_resnet50.Lmodel import MNISTResnet50

# Data preprocessing
transform = transforms.ToTensor()
train = dataset.MyData("./dataset/train/train", transform=transform)
train, valid = random_split(train, [int(len(train) * 0.9), len(train) - int(len(train) * 0.9)])
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False, num_workers=8)

net = MNISTResnet50()
wandb_logger = WandbLogger(log_model=True)
trainer = L.Trainer(max_epochs=15, accelerator='cuda', logger=wandb_logger)
trainer.fit(net, train_loader, valid_loader)
trainer.save_checkpoint("./model/model.ckpt")
x = torch.randn(1, 1, 28, 28)
net.to_onnx("./model/Lmodel.onnx", x)
