import os
import time

import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


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

    def log_prob(self, x: Tensor) -> Tensor:
        mean_qz_given_x, log_std_qz_given_x = self.encoder(x).chunk(2, dim=-1)
        e = torch.randn(x.shape[0], self.z_dim, device=x.device, dtype=x.dtype)  # reparam trick
        std_qz_given_x = log_std_qz_given_x.exp()
        z = mean_qz_given_x + e * std_qz_given_x

        mean_px_given_z = self.decoder(z)
        log_px_given_z = -0.5 * (x - mean_px_given_z).square().sum(-1)  # reconstruction loss
        log_pz = -0.5 * z.square().sum(-1)
        log_pxz = log_px_given_z + log_pz

        log_qz_x = -0.5 * (e.square().sum(-1) + log_std_qz_given_x.sum(-1))
        return (log_pxz - log_qz_x).mean()


def main():
    device = "cuda"
    batch_size = 256

    img_size = 28
    n_channels = 1
    z_dim = 16
    hidden_dim = 128

    transform = T.Compose([T.ToTensor(), nn.Flatten(0)])
    data = MNIST("data", download=True, transform=transform).data.to(device) / 256
    data_val = MNIST("data", train=False, download=True, transform=transform).data.to(device) / 256
    print(f"Dataset: {data.shape}")

    vae = VAE(img_size * img_size * n_channels, z_dim, hidden_dim).to(device)
    optim = torch.optim.AdamW(vae.parameters(), 1e-3, betas=(0.9, 0.95), weight_decay=1e-2)

    fixed_z = torch.randn(8 * 8, z_dim, device=device)

    @torch.no_grad()
    def generate(epoch_idx: int):
        generated = vae.decoder(fixed_z)
        grid = (
            generated.view(8, 8, n_channels, img_size, img_size)
            .cpu()
            .permute(0, 3, 1, 4, 2)
            .reshape(8 * img_size, 8 * img_size, n_channels)
            .squeeze(-1)
        )
        grid_u8 = (grid * 256).clip(0, 255.9999).to(torch.uint8)

        os.makedirs("mnist", exist_ok=True)
        Image.fromarray(grid_u8.numpy()).save(f"mnist/epoch_{epoch_idx:04d}.png")

    generate(0)
    for epoch_idx in range(100):
        time0 = time.perf_counter()

        # shuffle
        indices = torch.randperm(data.shape[0], device=device)
        data = data[indices]

        for i in range(data.shape[0] // batch_size):
            loss = vae(data[i * batch_size : (i + 1) * batch_size].view(batch_size, -1))

            loss.backward()
            optim.step()
            optim.zero_grad(True)

        throughput = data.shape[0] // batch_size / (time.perf_counter() - time0)
        with torch.no_grad():
            val_log_prob = vae.log_prob(data_val.view(data_val.shape[0], -1))
        print(
            f"Epoch {epoch_idx+1}: loss={loss.item():.4f} | val log-prob={val_log_prob.item()} | throughput={throughput:.2f}it/s"
        )
        generate(epoch_idx + 1)

    torch.save(vae.state_dict(), f"model_mnist.pth")


if __name__ == "__main__":
    main()
