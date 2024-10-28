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
import random
from PIL import Image


class SEG_MNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data
        self.targets = self.targets
        self.transform = transform
        self.target_transform = target_transform
        self.bg_img = np.array(Image.open("background.jpg").convert("L"))

    def __getitem__(self, index):
        img, _ = self.data[index], int(self.targets[index])
        img = img.numpy().astype(np.float32) / 255.0
        foreground = (img > 0.2).astype(np.float32)

        x_start = np.random.randint(0, self.bg_img.shape[0] - 28)
        y_start = np.random.randint(0, self.bg_img.shape[1] - 28)
        background = self.bg_img[x_start:x_start+28, y_start:y_start+28] / 255.0

        synthetic_img = foreground * img + (1 - foreground) * background
        segmentation_gt = foreground.copy()

        synthetic_img = Image.fromarray((synthetic_img * 255).astype(np.uint8))
        segmentation_gt = Image.fromarray((segmentation_gt * 255).astype(np.uint8))

        if self.transform is not None:
            synthetic_img = self.transform(synthetic_img)
        else:
            synthetic_img = transforms.ToTensor()(synthetic_img)

        if self.target_transform is not None:
            segmentation_gt = self.target_transform(segmentation_gt)
        else:
            segmentation_gt = transforms.ToTensor()(segmentation_gt)

        return synthetic_img, segmentation_gt


class LeNetSegmentation(nn.Module):
    def __init__(self):
        super(LeNetSegmentation, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.deconv1 = nn.ConvTranspose2d(16, 6, kernel_size=5)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv2 = nn.ConvTranspose2d(6, 1, kernel_size=5, padding=2)
        self.up2 = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.deconv1(x))
        x = self.up1(x)
        x = self.deconv2(x)
        x = self.up2(x)

        x = torch.sigmoid(x)
        return x

def test(model, device, test_loader):
    model.eval()
    test_loss = []
    criterion = nn.BCELoss()
    with torch.no_grad():
        for img, gt in test_loader:
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            test_loss.append(criterion(output, gt).item())

    test_loss = np.mean(test_loss)
    print(f'Test loss: {test_loss:.4f}')
    return test_loss

def save_model(model, path='models/best_model.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def train(train_loader, test_loader, model, optimizer, epochs, device):
    best_test_loss = float('inf')
    train_loss_log = []
    test_loss_log = []
    criterion = nn.BCELoss()
    for e in range(epochs):
        train_loss = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (img, gt) in pbar:
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
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

def visualization(model, test_loader, device):
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()
    with torch.no_grad():
        for img, gt in test_loader:
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            output = output.cpu().numpy()
            img = img.cpu().numpy()
            gt = gt.cpu().numpy()
            break

    fig, ax = plt.subplots(3, 5, figsize=(12, 7))
    for i in range(5):
        ax[0, i].imshow(img[i].squeeze(), cmap='gray')
        ax[0, i].set_title('Input Image')
        ax[0, i].axis('off')

        ax[1, i].imshow(gt[i].squeeze(), cmap='gray')
        ax[1, i].set_title('Ground Truth')
        ax[1, i].axis('off')

        ax[2, i].imshow(output[i].squeeze() > 0.5, cmap='gray')
        ax[2, i].set_title('Predicted Foreground')
        ax[2, i].axis('off')

    plt.tight_layout()
    plt.savefig("test.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    data_train = SEG_MNIST('../MNIST_DATA/', train=True, download=True, transform=transform)
    data_test = SEG_MNIST('../MNIST_DATA/', train=False, download=True, transform=transform)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNetSegmentation().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print(model)

    train(train_loader, test_loader, model, optimizer, args.epochs, device)

    visualization(model, test_loader, device)

if __name__ == "__main__":
    main()
