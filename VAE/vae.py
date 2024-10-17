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


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
        )
        self.enc_linear = nn.Sequential(
            nn.Linear(latent_dim, int(latent_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(latent_dim / 2), 4))
        self.dec_linear = nn.Sequential(
            nn.Linear(2, int(latent_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(latent_dim / 2), latent_dim))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.encoder(x)
        latent_shape = h.shape
        h = h.view(batch_size, -1)
        h = self.enc_linear(h)
        mu, logvar = h.chunk(2, dim=1)  # batch_size, latent_dim
        z = self.reparameterize(mu, logvar)
        z = self.dec_linear(z)
        z = z.view(latent_shape)
        out = self.decoder(z)
        return out, mu, logvar


def train(train_loader, test_loader, model, optimizer, epochs, device):
    best_test_loss = float('inf')
    train_loss_log = []
    test_loss_log = []
    for e in range(epochs):
        train_loss = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (x, _) in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            recon_loss = F.mse_loss(x_hat, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            pbar.set_description("(Epoch {}, iteration {})".format((e + 1), i + 1))
            pbar.set_postfix({'loss': loss.item()})

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
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            recon_loss = F.mse_loss(x_hat, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            test_loss.append(loss.item())

    test_loss = np.mean(test_loss)
    print(f'Test loss: {test_loss:.4f}')
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
    parser.add_argument("--latent_dim", type=int, default=196)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    data_train = MNIST('../MNIST_DATA/', train=True, download=True, transform=transform)
    data_test = MNIST('../MNIST_DATA/', train=False, download=True, transform=transform)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print(model)

    train(train_loader, test_loader, model, optimizer, args.epochs, device)


if __name__ == "__main__":
    main()
