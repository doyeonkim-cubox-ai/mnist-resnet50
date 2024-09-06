import torch
import torch.nn as nn
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm, trange
from mnist_resnet50 import model
from mnist_resnet50 import dataset


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # hyperparameters
    learning_rate = 0.001
    # learning_rate = {0.01, 0.005, 0.0025, 0.002, 0.001}
    training_epochs = 15
    batch_size = 32
    # lr: 0.005, epoch: 40 -> 0.008
    # lr: 0.001, epoch: 30 -> 0.011
    # lr: 0.0025, epoch: 20 -> 0.018
    # lr: 0.002, epoch: 20 -> 0.021
    # lr: 0.002, epoch: 30 -> 0.022
    # lr: 0.002, epoch: 15 -> 0.029
    # lr: 0.001, epoch: 15 -> 0.017

    # data load
    transform = transforms.Compose([transforms.ToTensor()])
    ## using MNIST
    # train = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
    # train, valid = random_split(train, [int(len(train)*0.9), int(len(train)*0.1)])
    # trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    # validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    ## using custom dataset: dataset.py
    train = dataset.MyData("./dataset/train/train", transform=transform)
    train, valid = random_split(train, [int(len(train)*0.9), len(train)-int(len(train)*0.9)])
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # model
    net = model.MyModel().to(device)

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # train
    train_batch = len(train_loader)
    valid_batch = len(valid_loader)

    wandb.init(project='mnist_resnet50', name='240905')

    print('Learning started')

    for epoch in trange(training_epochs):
        avg_train_cost = 0
        avg_valid_cost = 0
        # set to train
        net.train()
        for x_tr, y_tr in tqdm(train_loader):
            x_tr = x_tr.to(device)
            y_tr = y_tr.to(device)

            optimizer.zero_grad()
            hypothesis = net(x_tr)
            tr_cost = criterion(hypothesis, y_tr)
            tr_cost.backward()
            optimizer.step()

            avg_train_cost += tr_cost / train_batch
        # switch to eval
        net.eval()
        # Check overfitting
        for x_val, y_val in tqdm(valid_loader):
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            optimizer.zero_grad()
            hypothesis = net(x_val)
            val_cost = criterion(hypothesis, y_val)

            avg_valid_cost += val_cost / valid_batch

        print('Epoch: {} Train error: {} Valid error: {}'.format(epoch+1, avg_train_cost, avg_valid_cost))
        wandb.log({'train_loss': avg_train_cost, 'valid_loss': avg_valid_cost})
    print('Learning finished')

    torch.save(net.state_dict(), './model/model.pth')

    # onnx
    x = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(net, x, "./model/model.onnx")


if __name__ == "__main__":
    main()
