import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, LowRankMultivariateNormal


class GaussianCopulaDecoder(nn.Module):
    def __init__(self, latent_dim=10, output_channels=1, rank=1):
        """
        Low-rank Gaussian decoder:
            Sigma(z) = omega * I + W(z) W(z)^T

        - omega: global scalar (learned)
        - W(z):   low-rank factors in pixel space, rank << dim
        """
        super().__init__()
        self.rank = rank

        # Must match StrongEncoder flatten_dim
        self.reshape_dim = (256, 4, 4)
        self.flatten_dim = 256 * 4 * 4
        self.output_channels = output_channels

        # 1. Linear projection from z to feature map
        self.fc = nn.Linear(latent_dim, self.flatten_dim)

        # 2. ConvTranspose backbone (mirror of StrongEncoder)
        self.cnn_body = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 3. Heads
        # Mean image; if your data is in [0,1], you can add sigmoid in forward
        self.head_mu = nn.Conv2d(32, output_channels, kernel_size=1)

        # Low-rank factors: one covariance over all pixels, independent of channels
        self.head_factors = nn.Conv2d(32, rank, kernel_size=1)

        # Global log-variance (omega); scalar, not z-dependent (simpler / stable)
        self.log_omega = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        """
        z: [..., latent_dim]  (e.g. [B, D] or [K, B, D])

        returns:
            mu_flat:     [..., N]       (N = C * H * W)
            diag_scale:  [..., N]       (sqrt of diag variance)
            factors:     [..., N, R]    (low-rank factors)
        """
        # keep leading batch dims
        input_shape = z.shape[:-1]      # e.g. (B,) or (K, B)
        latent_dim = z.shape[-1]

        # flatten leading dims so backbone sees [Total_Batch, latent_dim]
        z_flat = z.view(-1, latent_dim)

        # 1. Project and reshape
        h = self.fc(z_flat)                     # [T, 256*4*4]
        h = F.relu(h)
        h = h.view(-1, *self.reshape_dim)       # [T, 256, 4, 4]

        # 2. ConvTranspose backbone
        features = self.cnn_body(h)             # [T, 32, 64, 64]

        # 3a. Mean
        mu = self.head_mu(features)             # [T, C, 64, 64]
        # if your x is in [0,1], uncomment:
        # mu = torch.sigmoid(mu)

        # flatten to [..., N]
        mu_flat = mu.view(*input_shape, -1)     # [..., N]
        N = mu_flat.shape[-1]

        # 3b. Diagonal variance: omega * I (global scalar)
        omega = F.softplus(self.log_omega) + 1e-4   # scalar > 0
        diag_var = omega * torch.ones_like(mu_flat) # [..., N]
        diag_scale = diag_var.sqrt()                # std

        # 3c. Low-rank factors W(z)
        # head_factors: [T, R, 64, 64]
        factors = self.head_factors(features)
        # [T, R, N]
        factors = factors.view(-1, self.rank, N)
        # [T, N, R]
        factors = factors.permute(0, 2, 1)
        # restore leading dims: [..., N, R]
        factors = factors.view(*input_shape, N, self.rank)

        return mu_flat, diag_scale, factors

    def get_distribution(self, z):
        """
        Creates LowRankMultivariateNormal over pixels with full batch support.
        """
        mu_flat, diag_scale, factors = self.forward(z)
        cov_diag = diag_scale.pow(2)

        dist = LowRankMultivariateNormal(
            loc=mu_flat,          # [..., N]
            cov_factor=factors,   # [..., N, R]
            cov_diag=cov_diag     # [..., N]
        )
        return dist

class GaussianMFDecoder(nn.Module):
    def __init__(self, latent_dim=10, output_channels=1):
        """
        Mean-field Gaussian decoder:
            p(x|z) = N(mu(z), sigma^2 I)

        - Same fc + cnn_body as the copula decoder.
        - Only per-pixel mean is learned.
        - Diagonal variance is a single global scalar (stable).
        """
        super().__init__()

        self.output_channels = output_channels

        # Must match StrongEncoder flatten_dim
        self.reshape_dim = (256, 4, 4)
        self.flatten_dim = 256 * 4 * 4

        # 1. Linear projection from z to feature map
        self.fc = nn.Linear(latent_dim, self.flatten_dim)

        # 2. ConvTranspose backbone (same as GaussianCopulaDecoder)
        self.cnn_body = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 3. Mean head
        self.head_mu = nn.Conv2d(32, output_channels, kernel_size=1)

        # Global log-std (scalar); more stable than per-pixel logvar
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        """
        z: [..., latent_dim]  (e.g. [B, D] or [K, B, D])

        returns:
            mu_flat:    [..., N]
            diag_scale: [..., N]  (std for each pixel; all equal)
        """
        input_shape = z.shape[:-1]      # e.g. (B,) or (K, B)
        latent_dim = z.shape[-1]

        # flatten leading dims
        z_flat = z.view(-1, latent_dim)

        # 1. Project and reshape
        h = self.fc(z_flat)
        h = F.relu(h)
        h = h.view(-1, *self.reshape_dim)  # [T, 256, 4, 4]

        # 2. Backbone
        features = self.cnn_body(h)        # [T, 32, 64, 64]

        # 3. Mean
        mu = self.head_mu(features)        # [T, C, 64, 64]
        # if x in [0,1], you can optionally squash:
        # mu = torch.sigmoid(mu)

        mu_flat = mu.view(*input_shape, -1)  # [..., N]
        N = mu_flat.shape[-1]

        # Global diagonal std
        sigma = F.softplus(self.log_sigma) + 1e-4   # scalar > 0
        diag_scale = sigma * torch.ones_like(mu_flat)

        return mu_flat, diag_scale

    def get_distribution(self, z):
        """
        Independent Gaussian over pixels (mean-field).
        """
        mu_flat, diag_scale = self.forward(z)
        dist = Independent(
            Normal(loc=mu_flat, scale=diag_scale),
            reinterpreted_batch_ndims=1  # last dim is "event"
        )
        return dist
