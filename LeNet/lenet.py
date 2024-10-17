import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class LeNet_task2(LeNet):
    def __init__(self, num_classes=10):
        super(LeNet_task2, self).__init__(num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 8, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )
        self.reset_parameters()


class LeNet_task3(LeNet):
    def __init__(self, num_classes=10, dropout=0.1):
        super(LeNet_task3, self).__init__(num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )
        self.reset_parameters()


def train(train_loader, test_loader, model, optimizer, epochs, device):
    best_test_loss = float('inf')
    train_loss_log = []
    test_loss_log = []
    for e in range(epochs):
        train_loss = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description("(Epoch {}, iteration {})".format((e + 1), i + 1))
            pbar.set_postfix_str("loss={:.4f}".format(np.mean(train_loss)))

        test_loss = test(model, device, test_loader)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(model)
        train_loss_log.append(np.mean(train_loss))
        test_loss_log.append(test_loss)

        plt.clf()
        plt.plot(train_loss_log, label="Train")
        plt.plot(test_loss_log, label="Test")
        plt.legend()
        plt.title("Learning Curve")
        plt.savefig('learning_curve.png')


def test(model, device, test_loader):
    model.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss.append(F.cross_entropy(output, target).item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = np.mean(test_loss)
    acc = 100 * correct / len(test_loader.dataset)
    print(f'Test loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.3f}%)')
    return test_loss


def save_model(model, path='models/best_model.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    data_train = MNIST('../MNIST_DATA/', train=True, download=True, transform=transform)
    data_test = MNIST('../MNIST_DATA/', train=False, download=True, transform=transform)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print(model)

    train(train_loader, test_loader, model, optimizer, args.epochs, device)


if __name__ == "__main__":
    main()
