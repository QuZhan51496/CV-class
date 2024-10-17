import argparse
import torch
import numpy as np
from scipy.stats import norm
from vae import VAE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=196)
    args = parser.parse_args()


    state = torch.load("models/best_model.pth")
    model = VAE(args.latent_dim)
    model.load_state_dict(state)
    model.eval()

    n = 20
    digit_size = 28

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    figure = np.zeros((digit_size * n, digit_size * n))
    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):
            t = np.array([x, y])
            # t = torch.randn((1, 2))
            z_sampled = torch.tensor(t, dtype=torch.float32)
            z_sampled = z_sampled.view(1, 2)
            with torch.no_grad():
                z_sampled = model.dec_linear(z_sampled)
                z_sampled = z_sampled.view((4, 7, 7))
                digit = model.decoder(z_sampled)
                figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.savefig('vis.png')
