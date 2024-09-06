import torch
import torch.nn as nn
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm, trange
from mnist_resnet50 import model
from mnist_resnet50 import dataset
import lightning as L


class MNISTResnet50(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model.MyModel()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x_tr, y_tr = batch
        hypothesis = self.model(x_tr)
        loss = self.loss_fn(hypothesis, y_tr)
        self.log("training loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val = batch
        hypothesis = self.model(x_val)
        loss = self.loss_fn(hypothesis, y_val)
        self.log("validation loss", loss.item())

    def test_step(self, batch, batch_idx):
        x_test, y_test = batch
        hypothesis = self.model(x_test)
        correct_pred = torch.argmax(hypothesis, dim=1)
        acc = torch.sum(correct_pred == y_test).item() / (len(y_test) * 1.0)
        self.log("test accuracy: ", acc)
