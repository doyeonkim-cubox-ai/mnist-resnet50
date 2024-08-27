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
# learning_rate = 0.01
# training_epochs = 100
batch_size = 32

# data load
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# model load
net = resnet.resnet50().to(device)
# resnet default input channel 수는 3, mnist는 1이므로 layer 수정
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
net = net.to(device)
net.load_state_dict(torch.load('./model/model.pth', weights_only=False))

with torch.no_grad():
    for num, data in enumerate(testloader):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)

        pred = net(imgs)
        correct_pred = torch.argmax(pred, 1) == label

        acc = correct_pred.float().mean()
print('Accuracy: {}'.format(acc.item()))