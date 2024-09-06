import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
from torchvision.models import ResNet50_Weights


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.backbone = resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        return x
