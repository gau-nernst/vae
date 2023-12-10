import os
import time

import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, Flowers102


class VAE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, dim: int) -> None:
        super().__init__()
        # predict parameters of the approximate posterior q(z|x) from x
        # we choose factorized Gaussian -> parametrized by mean and diagonal co-variance
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, z_dim * 2),
        )

        # predict parameters of the model likelihood p(x|z) from z
        # we choose unit variance Gaussian -> parametrized by mean
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, x_dim),
            nn.Sigmoid(),
        )
        self.z_dim = z_dim
        self.x_dim = x_dim

    def forward(self, x: Tensor) -> Tensor:
        # NOTE: we don't include constant terms in log(prob)
        # inference model q(z|x)
        mean_qz_given_x, log_std_qz_given_x = self.encoder(x).chunk(2, dim=-1)
        e = torch.randn(x.shape[0], self.z_dim, device=x.device, dtype=x.dtype)  # reparam trick
        std_qz_given_x = log_std_qz_given_x.exp()
        z = mean_qz_given_x + e * std_qz_given_x

        # generative model p(x,z) = p(x|z) * p(z)
        mean_px_given_z = self.decoder(z)
        log_px_given_z = -0.5 * (x - mean_px_given_z).square().sum(-1)  # reconstruction loss

        kl_div_q_given_x_px = 0.5 * (
            mean_qz_given_x.square().sum(-1) + std_qz_given_x.square().sum(-1) - 2 * log_std_qz_given_x.sum(-1)
        )
        elbo = log_px_given_z - kl_div_q_given_x_px
        return -elbo.mean()


def main():
    dataset = "flowers102"

    if dataset == "mnist":
        img_size = 28
        n_channels = 1
        z_dim = 16
        hidden_dim = 128

        transform = T.Compose([T.ToTensor(), nn.Flatten(0)])
        ds = MNIST("data", download=True, transform=transform)

    elif dataset == "flowers102":
        img_size = 64
        n_channels = 3
        z_dim = 16
        hidden_dim = 512

        transform = T.Compose(
            [
                T.Resize(img_size, T.InterpolationMode.LANCZOS),
                T.CenterCrop(img_size),
                T.ToTensor(),
                nn.Flatten(0),
            ]
        )
        ds = Flowers102("data", download=True, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset={dataset}")

    dloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)

    vae = VAE(img_size * img_size * n_channels, z_dim, hidden_dim)
    optim = torch.optim.AdamW(vae.parameters(), 1e-3, betas=(0.9, 0.95), weight_decay=1e-3)

    fixed_z = torch.randn(8 * 8, z_dim)

    @torch.no_grad()
    def generate(epoch_idx: int):
        generated = vae.decoder(fixed_z)
        grid = (
            generated.view(8, 8, n_channels, img_size, img_size)
            .permute(0, 3, 1, 4, 2)
            .reshape(8 * img_size, 8 * img_size, n_channels)
            .squeeze(-1)
        )
        grid_u8 = (grid * 256).clip(0, 255.9999).to(torch.uint8)

        os.makedirs(dataset, exist_ok=True)
        Image.fromarray(grid_u8.numpy()).save(f"{dataset}/epoch_{epoch_idx:04d}.png")

    generate(0)
    for epoch_idx in range(100):
        time0 = time.perf_counter()

        for images, labels in dloader:
            loss = vae(images)

            loss.backward()
            optim.step()
            optim.zero_grad(True)

        throughput = len(dloader) / (time.perf_counter() - time0)
        print(f"Epoch {epoch_idx+1}: loss={loss.item():.4f} | throughput={throughput:.2f}it/s")
        generate(epoch_idx + 1)

    torch.save(vae.state_dict(), f"model_{dataset}.pth")


if __name__ == "__main__":
    main()
