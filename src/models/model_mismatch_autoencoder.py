#!/usr/bin/env python3
"""
model_mismatch_autoencoder.py
=============================
Dual-Branch Latent Autoencoder Architecture for Multimodal Spatial-Temporal Alignment.

Resolves the dimensionality and sampling rate mismatch between:
1. Spatial Branch: Sentinel-2 multi-spectral imagery [B2, B3, B4, B8, NDWI] (B, 5, 256, 256)
2. Temporal Branch: Oceanographic hydrodynamic time-series (B, 72, F)

Produces:
- Synchronized latent representations:
  * z_spatial  in R^128 (Spatial Denoised Bottleneck)
  * z_temporal in R^64  (Temporal Hydrodynamic Bottleneck)
  * z_unified  in R^192 (Unified Multimodal Node Embedding)
"""

from __future__ import annotations
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable 2D Structural Similarity Index (SSIM)
# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def create_window_2d(window_size: int, channel: int) -> torch.Tensor:
    _1d_window = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    channel: int = 5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Differentiable 2D Multi-Channel SSIM calculation.
    Supports 5-channel satellite composite [RGB, NIR, NDWI].
    """
    if img1.size(1) != channel:
        channel = img1.size(1)

    window = create_window_2d(window_size, channel).to(img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )

    if reduction == "mean":
        return ssim_map.mean()
    elif reduction == "none":
        return ssim_map
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss: 1.0 - SSIM."""
    def __init__(self, window_size: int = 11, channel: int = 5):
        super().__init__()
        self.window_size = window_size
        self.channel = channel

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(img1, img2, window_size=self.window_size, channel=self.channel, reduction="mean")


class SpatialLoss(nn.Module):
    """
    Combined Spatial Loss: MSE + Structural Similarity Index (SSIM).
    Handles cloud masking and spectral noise removal.
    """
    def __init__(self, alpha: float = 0.7, window_size: int = 11, channel: int = 5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=window_size, channel=channel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_mse = self.mse(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        loss = self.alpha * l_mse + (1.0 - self.alpha) * l_ssim
        return loss, l_mse, l_ssim


# ---------------------------------------------------------------------------
# 1. Spatial Denoising Branch: Encoder, Bottleneck & Decoder
# ---------------------------------------------------------------------------
class SpatialEncoder(nn.Module):
    """
    4-stage 2D Convolutional downsampling encoder with BatchNorm, LeakyReLU, and Dropout.
    Input:  (B, in_channels, 256, 256)
    Output: z_spatial in R^latent_dim (default: 128)
    """
    def __init__(
        self,
        in_channels: int = 5,
        latent_dim: int = 128,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # 4 Downsampling Stages: 256 -> 128 -> 64 -> 32 -> 16
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
        )  # (B, 32, 128, 128)

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
        )  # (B, 64, 64, 64)

        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
        )  # (B, 128, 32, 32)

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
        )  # (B, 256, 16, 16)

        self.flatten_dim = 256 * 16 * 16  # 65,536
        self.fc_latent = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        flat = torch.flatten(out, start_dim=1)
        z = self.fc_latent(flat)
        return z


class SpatialDecoder(nn.Module):
    """
    Symmetrical Transposed Convolutional decoder reconstructing 256x256 multi-spectral image.
    Input:  z_spatial in R^latent_dim (default: 128)
    Output: (B, out_channels, 256, 256) with valid bounded range:
            - Channels 0..3 (RGB + NIR) in [0, 1] via Sigmoid
            - Channel 4 (NDWI) in [-1, 1] via Tanh
    """
    def __init__(
        self,
        out_channels: int = 5,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.flatten_dim = 256 * 16 * 16

        self.fc_unflatten = nn.Linear(latent_dim, self.flatten_dim)

        # 4 Upsampling Stages: 16 -> 32 -> 64 -> 128 -> 256
        self.up_stage1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 128, 32, 32)

        self.up_stage2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 64, 64, 64)

        self.up_stage3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 32, 128, 128)

        self.up_stage4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # (B, 5, 256, 256)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        unflat = self.fc_unflatten(z)
        feat = unflat.view(-1, 256, 16, 16)
        out = self.up_stage1(feat)
        out = self.up_stage2(out)
        out = self.up_stage3(out)
        raw = self.up_stage4(out)

        # Enforce physical spectral constraints
        # Bands 0-3 (Blue, Green, Red, NIR): [0, 1]
        # Band 4 (NDWI): [-1, 1]
        rgb_nir = torch.sigmoid(raw[:, :4, :, :])
        ndwi = torch.tanh(raw[:, 4:5, :, :])
        return torch.cat([rgb_nir, ndwi], dim=1)


class SpatialAutoencoder(nn.Module):
    """Complete Spatial Denoising Autoencoder."""
    def __init__(self, in_channels: int = 5, latent_dim: int = 128, dropout_p: float = 0.2):
        super().__init__()
        self.encoder = SpatialEncoder(in_channels=in_channels, latent_dim=latent_dim, dropout_p=dropout_p)
        self.decoder = SpatialDecoder(out_channels=in_channels, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


# ---------------------------------------------------------------------------
# 2. Temporal Signal Branch: 1D Conv Encoder, Bottleneck & Decoder
# ---------------------------------------------------------------------------
class TemporalEncoder(nn.Module):
    """
    3-layer 1D Convolutional network with kernel sizes [7, 5, 3] downsampling along time dimension.
    Input:  (B, seq_len=72, num_features=F) -> transposed to (B, F, 72)
    Downsampling sequence: 72 -> 36 -> 18 -> 9
    Bottleneck Linear projection: 128 * 9 = 1152 -> z_temporal in R^latent_dim (default: 64)
    """
    def __init__(
        self,
        num_features: int = 7,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim

        # Layer 1: k=7, s=2, p=3 -> (72 + 6 - 7)//2 + 1 = 36
        self.layer1 = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 2: k=5, s=2, p=2 -> (36 + 4 - 5)//2 + 1 = 18
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 3: k=3, s=2, p=1 -> (18 + 2 - 3)//2 + 1 = 9
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = 128 * 9  # 1152
        self.fc_latent = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 72, F) -> permute to (B, F, 72)
        x_t = x.permute(0, 2, 1)
        h = self.layer1(x_t)
        h = self.layer2(h)
        h = self.layer3(h)
        flat = torch.flatten(h, start_dim=1)
        z = self.fc_latent(flat)
        return z


class TemporalDecoder(nn.Module):
    """
    Symmetrical 1D Transposed Convolutional decoder reconstructing 72-hour sequence.
    Input:  z_temporal in R^latent_dim (default: 64)
    Output: (B, 72, num_features)
    """
    def __init__(
        self,
        num_features: int = 7,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.flatten_dim = 128 * 9  # 1152

        self.fc_unflatten = nn.Linear(latent_dim, self.flatten_dim)

        # Upsampling Stage 1: 9 -> 18
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Upsampling Stage 2: 18 -> 36
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Upsampling Stage 3: 36 -> 72
        self.up3 = nn.ConvTranspose1d(32, num_features, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        unflat = self.fc_unflatten(z)
        h = unflat.view(-1, 128, 9)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)  # (B, F, 72)
        # Permute back to standard format: (B, 72, F)
        return h.permute(0, 2, 1)


class TemporalAutoencoder(nn.Module):
    """Complete Temporal Signal Autoencoder."""
    def __init__(self, num_features: int = 7, latent_dim: int = 64):
        super().__init__()
        self.encoder = TemporalEncoder(num_features=num_features, latent_dim=latent_dim)
        self.decoder = TemporalDecoder(num_features=num_features, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


# ---------------------------------------------------------------------------
# 3. Dual-Branch Master Model: Unified Latent Node Embeddings
# ---------------------------------------------------------------------------
class DualBranchAutoencoder(nn.Module):
    """
    Dual-Branch Latent Autoencoder for multimodal coastal monitoring.
    - Branch 1: Spatial Denoising Autoencoder (5, 256, 256) -> z_spatial in R^128
    - Branch 2: Temporal Signal Autoencoder (72, F)         -> z_temporal in R^64
    - Unified Bottleneck: Concatenated latent vector z_unified in R^192
    """
    def __init__(
        self,
        spatial_channels: int = 5,
        temporal_features: int = 7,
        spatial_latent_dim: int = 128,
        temporal_latent_dim: int = 64,
        spatial_dropout: float = 0.2,
    ):
        super().__init__()
        self.spatial_latent_dim = spatial_latent_dim
        self.temporal_latent_dim = temporal_latent_dim
        self.unified_latent_dim = spatial_latent_dim + temporal_latent_dim  # 192

        self.spatial_branch = SpatialAutoencoder(
            in_channels=spatial_channels,
            latent_dim=spatial_latent_dim,
            dropout_p=spatial_dropout,
        )
        self.temporal_branch = TemporalAutoencoder(
            num_features=temporal_features,
            latent_dim=temporal_latent_dim,
        )

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            rec_spatial:  (B, 5, 256, 256)
            rec_temporal: (B, 72, F)
            z_spatial:    (B, 128)
            z_temporal:   (B, 64)
            z_unified:    (B, 192)
        """
        rec_spatial, z_spatial = self.spatial_branch(x_spatial)
        rec_temporal, z_temporal = self.temporal_branch(x_temporal)
        z_unified = torch.cat([z_spatial, z_temporal], dim=-1)  # (B, 192)
        return rec_spatial, rec_temporal, z_spatial, z_temporal, z_unified

    def encode(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fast inference encoding forward pass."""
        z_spatial = self.spatial_branch.encoder(x_spatial)
        z_temporal = self.temporal_branch.encoder(x_temporal)
        z_unified = torch.cat([z_spatial, z_temporal], dim=-1)
        return z_spatial, z_temporal, z_unified

    def get_parameter_counts(self) -> Dict[str, int]:
        """Calculates detailed parameter counts across branches."""
        spatial_enc = sum(p.numel() for p in self.spatial_branch.encoder.parameters())
        spatial_dec = sum(p.numel() for p in self.spatial_branch.decoder.parameters())
        temporal_enc = sum(p.numel() for p in self.temporal_branch.encoder.parameters())
        temporal_dec = sum(p.numel() for p in self.temporal_branch.decoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "spatial_encoder": spatial_enc,
            "spatial_decoder": spatial_dec,
            "spatial_total": spatial_enc + spatial_dec,
            "temporal_encoder": temporal_enc,
            "temporal_decoder": temporal_dec,
            "temporal_total": temporal_enc + temporal_dec,
            "total_trainable": total,
        }

    def get_compression_ratios(
        self,
        spatial_shape: Tuple[int, ...] = (5, 256, 256),
        temporal_shape: Tuple[int, ...] = (72, 7),
    ) -> Dict[str, Dict[str, float]]:
        """Calculates exact data compression ratios."""
        spatial_raw_dim = math.prod(spatial_shape)
        temporal_raw_dim = math.prod(temporal_shape)
        total_raw_dim = spatial_raw_dim + temporal_raw_dim

        return {
            "spatial": {
                "raw_dim": spatial_raw_dim,
                "latent_dim": self.spatial_latent_dim,
                "compression_ratio": spatial_raw_dim / self.spatial_latent_dim,
                "reduction_percent": (1.0 - self.spatial_latent_dim / spatial_raw_dim) * 100.0,
            },
            "temporal": {
                "raw_dim": temporal_raw_dim,
                "latent_dim": self.temporal_latent_dim,
                "compression_ratio": temporal_raw_dim / self.temporal_latent_dim,
                "reduction_percent": (1.0 - self.temporal_latent_dim / temporal_raw_dim) * 100.0,
            },
            "unified": {
                "raw_dim": total_raw_dim,
                "latent_dim": self.unified_latent_dim,
                "compression_ratio": total_raw_dim / self.unified_latent_dim,
                "reduction_percent": (1.0 - self.unified_latent_dim / total_raw_dim) * 100.0,
            },
        }


if __name__ == "__main__":
    # Self-test unit check
    print("Testing DualBranchAutoencoder architecture...")
    model = DualBranchAutoencoder()
    print("Parameter counts:", model.get_parameter_counts())
    print("Compression ratios:", model.get_compression_ratios())

    # Forward pass sanity check
    x_s = torch.randn(2, 5, 256, 256)
    x_t = torch.randn(2, 72, 7)
    rec_s, rec_t, z_s, z_t, z_u = model(x_s, x_t)

    assert rec_s.shape == (2, 5, 256, 256), f"Wrong spatial shape: {rec_s.shape}"
    assert rec_t.shape == (2, 72, 7), f"Wrong temporal shape: {rec_t.shape}"
    assert z_s.shape == (2, 128), f"Wrong spatial latent shape: {z_s.shape}"
    assert z_t.shape == (2, 64), f"Wrong temporal latent shape: {z_t.shape}"
    assert z_u.shape == (2, 192), f"Wrong unified latent shape: {z_u.shape}"
    print("DualBranchAutoencoder sanity check passed successfully!")
