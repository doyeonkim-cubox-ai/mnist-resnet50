import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
learning_rate = 0.002
# learning_rate = {0.01, 0.005, 0.0025, 0.002, 0.001}
training_epochs = 30
batch_size = 32
# lr: 0.005, epoch: 40 -> 0.008
# lr: 0.001, epoch: 30 -> 0.011
# lr: 0.0025, epoch: 20 -> 0.018
# lr: 0.002, epoch: 20 -> 0.021
# lr: 0.002, epoch: 30 -> 0.022

# data load
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# model
model = resnet.resnet50().to(device)
# resnet default input channel 수는 3, mnist는 1이므로 layer 수정
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
model = model.to(device)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
total_batch = len(trainloader)

print('Learning started')

for epoch in range(training_epochs):
    avg_cost = 0
    for X, Y in trainloader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch: {} Cost: {}'.format(epoch+1, avg_cost))
print('Learning finished')

torch.save(model.state_dict(), './model/model.pth')