import torch
import torchvision
import torchvision.transforms as transforms
from mnist_resnet50 import dataset
import lightning as L
from mnist_resnet50.Lmodel import MNISTResnet50

# Data preprocessing
transform = transforms.ToTensor()
test = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

checkpoint = "./model/model.ckpt"
net = MNISTResnet50.load_from_checkpoint(checkpoint)
trainer = L.Trainer(accelerator='cuda')
trainer.test(net, test_loader)

