import torch
import torchvision
import torchvision.transforms as transforms
import model


def main():
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
    test = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    # model load
    net = model.Mymodel().to(device)
    net.load_state_dict(torch.load('./model/model.pth', weights_only=False))

    # set to eval
    net.eval()
    # compute accuracy
    with torch.no_grad():
        for num, data in enumerate(testloader):
            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)

            pred = net(imgs)
            correct_pred = torch.argmax(pred, 1) == label

            acc = correct_pred.float().mean()
    print('Accuracy: {}'.format(acc.item()))


if __name__ == "__main__":
    main()