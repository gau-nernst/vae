import time

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


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
        z = mean_qz_given_x + e * log_std_qz_given_x.exp()
        log_qz_given_x = -0.5 * e.square().sum(-1) - log_std_qz_given_x.sum(-1)

        # generative model p(x,z) = p(x|z) * p(z)
        log_pz = -0.5 * z.square().sum(-1)  # prior
        mean_px_given_z = self.decoder(z)
        log_px_given_z = -0.5 * (x - mean_px_given_z).square().sum(-1)  # reconstruction loss

        elbo = log_px_given_z + log_pz - log_qz_given_x
        return -elbo.mean()


def main():
    ds = MNIST("mnist", download=True, transform=Compose([ToTensor(), nn.Flatten(0)]))
    dloader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

    vae = VAE(28 * 28, 16, 128)
    optim = torch.optim.AdamW(vae.parameters(), 1e-3, betas=(0.9, 0.95), weight_decay=1e-3)

    fixed_z = torch.randn(8 * 8, 16)

    @torch.no_grad()
    def generate(epoch_idx: int):
        generated = vae.decoder(fixed_z)
        grid = generated.view(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
        grid_u8 = (grid * 256).clip(0, 255.9999).to(torch.uint8)
        Image.fromarray(grid_u8.numpy()).save(f"epoch_{epoch_idx:04d}.png")

    generate(0)
    for epoch_idx in range(10):
        time0 = time.perf_counter()

        for images, labels in dloader:
            loss = vae(images)

            loss.backward()
            optim.step()
            optim.zero_grad(True)

        throughput = len(dloader) / (time.perf_counter() - time0)
        print(f"Epoch {epoch_idx+1: 2d}: loss={loss.item():.4f} | throughput={throughput:.2f}it/s")
        generate(epoch_idx + 1)


if __name__ == "__main__":
    main()
