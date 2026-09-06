#!/usr/bin/env python3
"""
train_mismatch_autoencoder.py
=============================
Dual-Branch Latent Autoencoder Training Pipeline.

Pairs Sentinel-2 multi-spectral imagery with oceanographic time-series,
and trains a synchronized latent autoencoder with:
1. Spatial Branch: MSE + SSIM reconstruction with synthetic cloud/noise masking.
2. Temporal Branch: Huber Loss (Smooth L1) reconstruction against surge outliers.
3. Synchronized checkpointing: saves `autoencoder_best.pt`.
4. Generates visual reconstruction check: `data/reconstruction_check.png`.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_mismatch_autoencoder import (
    DualBranchAutoencoder,
    SpatialLoss,
    ssim,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("TrainMismatchAutoencoder")

# Constants & Default Paths
DEFAULT_SPATIAL_H5 = Path("data/changeformer_dataset/dataset.h5")
DEFAULT_OCEAN_H5 = Path("data/ocean/ocean_timeseries_72h.h5")
DEFAULT_OUTPUT_DIR = Path("data")
CHECKPOINT_NAME = "autoencoder_best.pt"
RECONSTRUCTION_PLOT = "reconstruction_check.png"


# ---------------------------------------------------------------------------
# Custom MultiModal Dataset
# ---------------------------------------------------------------------------
class MultiModalMismatchDataset(Dataset):
    """
    Synchronized Multi-Modal Dataset pairing Sentinel-2 spatial patches
    (B2, B3, B4, B8, NDWI) with co-located coastal oceanographic time-series.

    Each item returns:
    - x_spatial_clean:  (5, 256, 256) float32 in [0, 1] / [-1, 1]
    - x_spatial_noisy:  (5, 256, 256) float32 (with optional cloud/noise masking)
    - x_temporal:       (72, 7) float32 (Z-score standardized ocean window)
    - metadata:         dict with zone, year, pair, patch_idx, timestamp
    """
    def __init__(
        self,
        spatial_h5_path: Path = DEFAULT_SPATIAL_H5,
        ocean_h5_path: Path = DEFAULT_OCEAN_H5,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        augment_noise: bool = True,
    ):
        super().__init__()
        self.spatial_h5_path = Path(spatial_h5_path)
        self.ocean_h5_path = Path(ocean_h5_path)
        self.split = split
        self.augment_noise = augment_noise

        self.spatial_h5: Optional[h5py.File] = None
        self.ocean_h5: Optional[h5py.File] = None

        self._index: List[Dict[str, Any]] = []
        self._build_index(val_ratio=val_ratio, seed=seed)

    def _build_index(self, val_ratio: float, seed: int):
        """Index spatial patches and pair with co-located ocean windows."""
        if not self.spatial_h5_path.is_file():
            raise FileNotFoundError(f"Spatial dataset not found: {self.spatial_h5_path}")
        if not self.ocean_h5_path.is_file():
            raise FileNotFoundError(f"Ocean dataset not found: {self.ocean_h5_path}")

        all_samples = []

        with h5py.File(self.spatial_h5_path, "r") as sp_h5, h5py.File(self.ocean_h5_path, "r") as oc_h5:
            zones = sorted(list(sp_h5.keys()))
            for zone in zones:
                if zone not in oc_h5:
                    continue

                ts_arr = oc_h5[zone]["timestamps_end"][:]
                timestamps = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in ts_arr]
                n_ocean_windows = len(timestamps)

                # Pre-index ocean window timestamps by year
                year_to_indices: Dict[int, List[int]] = {}
                for idx, ts in enumerate(timestamps):
                    year = int(ts[:4])
                    year_to_indices.setdefault(year, []).append(idx)

                for pair_key in sorted(sp_h5[zone].keys()):
                    # pair_key format: "pair_2019_2020"
                    parts = pair_key.split("_")
                    y1, y2 = int(parts[1]), int(parts[2])
                    n_patches = sp_h5[zone][pair_key]["T1"].shape[0]

                    # Map T1 patches to year y1 ocean windows, T2 patches to year y2 ocean windows
                    for t_tag, year in [("T1", y1), ("T2", y2)]:
                        avail_indices = year_to_indices.get(year, list(range(n_ocean_windows)))
                        n_avail = len(avail_indices)

                        for patch_idx in range(n_patches):
                            # Deterministic balanced mapping for consistent temporal association
                            ocean_idx = avail_indices[(patch_idx * 37) % n_avail]
                            all_samples.append({
                                "zone": zone,
                                "pair_key": pair_key,
                                "t_tag": t_tag,
                                "year": year,
                                "patch_idx": patch_idx,
                                "ocean_idx": ocean_idx,
                                "timestamp": timestamps[ocean_idx],
                            })

        # Train/Validation split
        rng = random.Random(seed)
        shuffled = list(all_samples)
        rng.shuffle(shuffled)

        n_val = int(len(shuffled) * val_ratio)
        if self.split == "val":
            self._index = shuffled[:n_val]
        elif self.split == "train":
            self._index = shuffled[n_val:]
        else:
            self._index = shuffled  # 'all'

        log.info(
            "Indexed %d samples for split '%s' across %d zones",
            len(self._index),
            self.split,
            len(set(s["zone"] for s in self._index)),
        )

    def __len__(self) -> int:
        return len(self._index)

    def _inject_cloud_and_noise(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Synthetically simulates cloud masking and sensor noise on input patch:
        - Additive Gaussian noise (reflectance jitter)
        - Dense rectangular cloud occlusion (high reflectance in RGB, zero NDWI)
        """
        noisy = patch.clone()

        # 1. Additive Gaussian noise (50% probability)
        if random.random() < 0.5:
            noise = torch.randn_like(noisy) * 0.04
            noisy = noisy + noise

        # 2. Synthetic Cloud / Obstruction Mask (40% probability)
        if random.random() < 0.4:
            c_size = random.randint(24, 64)
            y0 = random.randint(0, 256 - c_size)
            x0 = random.randint(0, 256 - c_size)
            # Cloud: high visible reflectance, low/zero NDWI
            noisy[:3, y0:y0 + c_size, x0:x0 + c_size] = torch.clamp(
                noisy[:3, y0:y0 + c_size, x0:x0 + c_size] + random.uniform(0.3, 0.7),
                0.0, 1.0,
            )
            noisy[4, y0:y0 + c_size, x0:x0 + c_size] = 0.0

        # Enforce bounds
        noisy[:4] = torch.clamp(noisy[:4], 0.0, 1.0)
        noisy[4] = torch.clamp(noisy[4], -1.0, 1.0)
        return noisy

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Lazily open HDF5 files per-process for thread/worker safety
        if self.spatial_h5 is None:
            self.spatial_h5 = h5py.File(self.spatial_h5_path, "r")
        if self.ocean_h5 is None:
            self.ocean_h5 = h5py.File(self.ocean_h5_path, "r")

        sample = self._index[idx]
        zone = sample["zone"]
        pair_key = sample["pair_key"]
        t_tag = sample["t_tag"]
        patch_idx = sample["patch_idx"]
        ocean_idx = sample["ocean_idx"]

        # Read spatial patch: (5, 256, 256)
        clean_spatial = self.spatial_h5[zone][pair_key][t_tag][patch_idx]
        clean_spatial = torch.from_numpy(clean_spatial).to(torch.float32)

        # Apply noise/cloud augmentation if training
        if self.augment_noise and self.split == "train":
            noisy_spatial = self._inject_cloud_and_noise(clean_spatial)
        else:
            noisy_spatial = clean_spatial.clone()

        # Read temporal window: (72, 7) Z-score normalized
        temporal_win = self.ocean_h5[zone]["X_zscore"][ocean_idx]
        temporal_win = torch.from_numpy(temporal_win).to(torch.float32)

        meta = {
            "zone": zone,
            "pair_key": pair_key,
            "t_tag": t_tag,
            "year": sample["year"],
            "patch_idx": patch_idx,
            "ocean_idx": ocean_idx,
            "timestamp": sample["timestamp"],
        }

        return clean_spatial, noisy_spatial, temporal_win, meta


# ---------------------------------------------------------------------------
# Training & Validation Loops
# ---------------------------------------------------------------------------
def train_epoch(
    model: DualBranchAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    spatial_criterion: SpatialLoss,
    temporal_criterion: nn.Module,
    lambda_spatial: float,
    lambda_temporal: float,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss_accum = 0.0
    sp_loss_accum = 0.0
    sp_mse_accum = 0.0
    sp_ssim_accum = 0.0
    temp_loss_accum = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training Batch", leave=False)
    for i, (clean_sp, noisy_sp, temp_win, _) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        clean_sp = clean_sp.to(device)
        noisy_sp = noisy_sp.to(device)
        temp_win = temp_win.to(device)

        optimizer.zero_grad()

        # Forward pass
        rec_sp, rec_temp, z_s, z_t, z_u = model(noisy_sp, temp_win)

        # Spatial loss (MSE + SSIM) against clean ground truth
        loss_sp, l_mse, l_ssim = spatial_criterion(rec_sp, clean_sp)

        # Temporal loss (Huber / Smooth L1)
        loss_temp = temporal_criterion(rec_temp, temp_win)

        # Weighted multi-modal objective
        total_loss = lambda_spatial * loss_sp + lambda_temporal * loss_temp

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss_accum += total_loss.item()
        sp_loss_accum += loss_sp.item()
        sp_mse_accum += l_mse.item()
        sp_ssim_accum += (1.0 - l_ssim.item())  # Store actual SSIM value
        temp_loss_accum += loss_temp.item()
        n_batches += 1

        pbar.set_postfix({
            "Loss": f"{total_loss.item():.4f}",
            "SpMSE": f"{l_mse.item():.4f}",
            "TempHuber": f"{loss_temp.item():.4f}",
        })

    return {
        "total_loss": total_loss_accum / max(1, n_batches),
        "spatial_loss": sp_loss_accum / max(1, n_batches),
        "spatial_mse": sp_mse_accum / max(1, n_batches),
        "spatial_ssim": sp_ssim_accum / max(1, n_batches),
        "temporal_huber": temp_loss_accum / max(1, n_batches),
    }


def validate(
    model: DualBranchAutoencoder,
    dataloader: DataLoader,
    spatial_criterion: SpatialLoss,
    temporal_criterion: nn.Module,
    lambda_spatial: float,
    lambda_temporal: float,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    model.eval()
    total_loss_accum = 0.0
    sp_loss_accum = 0.0
    sp_mse_accum = 0.0
    sp_ssim_accum = 0.0
    temp_loss_accum = 0.0
    n_batches = 0

    saved_visual_samples = None

    with torch.no_grad():
        for i, (clean_sp, noisy_sp, temp_win, _) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            clean_sp = clean_sp.to(device)
            noisy_sp = noisy_sp.to(device)
            temp_win = temp_win.to(device)

            rec_sp, rec_temp, z_s, z_t, z_u = model(noisy_sp, temp_win)

            loss_sp, l_mse, l_ssim = spatial_criterion(rec_sp, clean_sp)
            loss_temp = temporal_criterion(rec_temp, temp_win)
            total_loss = lambda_spatial * loss_sp + lambda_temporal * loss_temp

            total_loss_accum += total_loss.item()
            sp_loss_accum += loss_sp.item()
            sp_mse_accum += l_mse.item()
            sp_ssim_accum += (1.0 - l_ssim.item())
            temp_loss_accum += loss_temp.item()
            n_batches += 1

            if saved_visual_samples is None:
                saved_visual_samples = (
                    clean_sp[0].cpu(),
                    noisy_sp[0].cpu(),
                    rec_sp[0].cpu(),
                    temp_win[0].cpu(),
                    rec_temp[0].cpu(),
                )

    metrics = {
        "total_loss": total_loss_accum / max(1, n_batches),
        "spatial_loss": sp_loss_accum / max(1, n_batches),
        "spatial_mse": sp_mse_accum / max(1, n_batches),
        "spatial_ssim": sp_ssim_accum / max(1, n_batches),
        "temporal_huber": temp_loss_accum / max(1, n_batches),
    }
    return metrics, saved_visual_samples


# ---------------------------------------------------------------------------
# Visual Reconstruction Quality Plotter
# ---------------------------------------------------------------------------
def plot_visual_reconstruction(
    clean_sp: torch.Tensor,
    noisy_sp: torch.Tensor,
    rec_sp: torch.Tensor,
    clean_temp: torch.Tensor,
    rec_temp: torch.Tensor,
    history: Dict[str, List[float]],
    out_path: Path,
):
    """
    Creates a comprehensive validation figure displaying:
    1. Spatial RGB reconstruction: Clean vs Noisy vs Reconstructed vs Absolute Error.
    2. Spatial NDWI channel reconstruction.
    3. Temporal Hydrodynamic reconstruction for wave height (swh) and wind magnitude.
    4. Training Loss progression curves.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25)

    # RGB channels: Band 2 (Red), Band 1 (Green), Band 0 (Blue)
    def to_rgb(t: torch.Tensor) -> np.ndarray:
        rgb = np.stack([t[2].numpy(), t[1].numpy(), t[0].numpy()], axis=-1)
        return np.clip(rgb * 3.0, 0.0, 1.0)  # Enhance visual contrast

    rgb_clean = to_rgb(clean_sp)
    rgb_noisy = to_rgb(noisy_sp)
    rgb_rec = to_rgb(rec_sp)
    rgb_err = np.abs(rgb_clean - rgb_rec).mean(axis=-1)

    # Row 0: Spatial RGB Reconstruction
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_clean)
    ax0.set_title("Target Sentinel-2 (Clean RGB)", fontsize=11, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(rgb_noisy)
    ax1.set_title("Input (Synthetic Cloud/Noise)", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(rgb_rec)
    ax2.set_title("Reconstructed (Spatial Decoder)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    im_err = ax3.imshow(rgb_err, cmap="magma")
    ax3.set_title("Reconstruction Error Map", fontsize=11, fontweight="bold")
    ax3.axis("off")
    fig.colorbar(im_err, ax=ax3, fraction=0.046, pad=0.04)

    # Row 1: NDWI channel & Temporal Signals
    ndwi_clean = clean_sp[4].numpy()
    ndwi_rec = rec_sp[4].numpy()

    ax4 = fig.add_subplot(gs[1, 0])
    im_n1 = ax4.imshow(ndwi_clean, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax4.set_title("Clean NDWI Channel", fontsize=11, fontweight="bold")
    ax4.axis("off")
    fig.colorbar(im_n1, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 1])
    im_n2 = ax5.imshow(ndwi_rec, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax5.set_title("Reconstructed NDWI", fontsize=11, fontweight="bold")
    ax5.axis("off")
    fig.colorbar(im_n2, ax=ax5, fraction=0.046, pad=0.04)

    # Temporal Feature 0: Significant Wave Height (swh)
    ax6 = fig.add_subplot(gs[1, 2])
    hours = np.arange(1, 73)
    ax6.plot(hours, clean_temp[:, 0].numpy(), "b-", lw=1.8, label="Ground Truth (swh)")
    ax6.plot(hours, rec_temp[:, 0].numpy(), "r--", lw=1.5, label="Reconstructed")
    ax6.set_title("Significant Wave Height (swh)", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Lookback Hours (t-72 to t-1)", fontsize=9)
    ax6.set_ylabel("Standardized Value (Z-score)", fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)

    # Temporal Feature 5: Wind Magnitude
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.plot(hours, clean_temp[:, 5].numpy(), "g-", lw=1.8, label="Ground Truth (W_mag)")
    ax7.plot(hours, rec_temp[:, 5].numpy(), "m--", lw=1.5, label="Reconstructed")
    ax7.set_title("Wind Magnitude (W_mag)", fontsize=11, fontweight="bold")
    ax7.set_xlabel("Lookback Hours (t-72 to t-1)", fontsize=9)
    ax7.set_ylabel("Standardized Value (Z-score)", fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=8)

    # Row 2: Loss Curves
    epochs = range(1, len(history["train_total_loss"]) + 1)

    ax8 = fig.add_subplot(gs[2, 0:2])
    ax8.plot(epochs, history["train_total_loss"], "b-o", label="Train Total Loss", lw=2)
    ax8.plot(epochs, history["val_total_loss"], "r--s", label="Val Total Loss", lw=2)
    ax8.set_title("Dual-Branch Total Loss Convergence", fontsize=11, fontweight="bold")
    ax8.set_xlabel("Epoch", fontsize=10)
    ax8.set_ylabel("Total Loss", fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=9)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(epochs, history["train_spatial_mse"], "c-o", label="Train Sp MSE")
    ax9.plot(epochs, history["val_spatial_mse"], "m--s", label="Val Sp MSE")
    ax9.set_title("Spatial MSE Progression", fontsize=11, fontweight="bold")
    ax9.set_xlabel("Epoch", fontsize=10)
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=9)

    ax10 = fig.add_subplot(gs[2, 3])
    ax10.plot(epochs, history["train_temporal_huber"], "g-o", label="Train Temp Huber")
    ax10.plot(epochs, history["val_temporal_huber"], "orange", linestyle="--", marker="s", label="Val Temp Huber")
    ax10.set_title("Temporal Huber Loss Progression", fontsize=11, fontweight="bold")
    ax10.set_xlabel("Epoch", fontsize=10)
    ax10.grid(True, alpha=0.3)
    ax10.legend(fontsize=9)

    plt.suptitle("Dual-Branch Latent Autoencoder Reconstruction Quality & Convergence", fontsize=15, fontweight="bold", y=0.99)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    log.info("Saved visual reconstruction check plot to: %s", out_path)


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Dual-Branch Latent Autoencoder")
    parser.add_argument("--spatial-h5", type=Path, default=DEFAULT_SPATIAL_H5)
    parser.add_argument("--ocean-h5", type=Path, default=DEFAULT_OCEAN_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--spatial-latent-dim", type=int, default=128, help="Spatial latent dimension z_spatial")
    parser.add_argument("--temporal-latent-dim", type=int, default=64, help="Temporal latent dimension z_temporal")
    parser.add_argument("--lambda-spatial", type=float, default=1.0, help="Weight for spatial loss")
    parser.add_argument("--lambda-temporal", type=float, default=1.0, help="Weight for temporal loss")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cuda or cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test mode with fewer steps")
    args = parser.parse_args()

    # Determine execution hardware
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 68)
    log.info("    Dual-Branch Latent Autoencoder Training Pipeline")
    log.info("=" * 68)
    log.info("  Device               : %s", device)
    if device.type == "cuda":
        log.info("  GPU Name             : %s", torch.cuda.get_device_name(0))
        log.info("  GPU Memory           : %.2f GB", torch.cuda.get_device_properties(0).total_memory / (1024**3))
    log.info("  Spatial HDF5 Path    : %s", args.spatial_h5)
    log.info("  Ocean HDF5 Path      : %s", args.ocean_h5)
    log.info("  Latent Dimensions    : Spatial=%d, Temporal=%d, Unified=%d",
             args.spatial_latent_dim, args.temporal_latent_dim, args.spatial_latent_dim + args.temporal_latent_dim)
    log.info("  Hyperparameters      : Epochs=%d, BatchSize=%d, LR=%.1e", args.epochs, args.batch_size, args.lr)
    log.info("  Loss Weights         : Lambda_Spatial=%.1f, Lambda_Temporal=%.1f", args.lambda_spatial, args.lambda_temporal)

    # 1. Datasets and Loaders
    train_dataset = MultiModalMismatchDataset(
        spatial_h5_path=args.spatial_h5,
        ocean_h5_path=args.ocean_h5,
        split="train",
        val_ratio=0.15,
        augment_noise=True,
    )
    val_dataset = MultiModalMismatchDataset(
        spatial_h5_path=args.spatial_h5,
        ocean_h5_path=args.ocean_h5,
        split="val",
        val_ratio=0.15,
        augment_noise=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Safe and reliable cross-platform execution
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model Initialization
    model = DualBranchAutoencoder(
        spatial_channels=5,
        temporal_features=7,
        spatial_latent_dim=args.spatial_latent_dim,
        temporal_latent_dim=args.temporal_latent_dim,
        spatial_dropout=0.2,
    ).to(device)

    param_counts = model.get_parameter_counts()
    compression = model.get_compression_ratios()
    log.info("Trainable Parameters : Total=%s (Spatial=%s, Temporal=%s)",
             f"{param_counts['total_trainable']:,}",
             f"{param_counts['spatial_total']:,}",
             f"{param_counts['temporal_total']:,}")
    log.info("Compression Ratios   : Spatial=%.1fx, Temporal=%.2fx, Unified=%.1fx",
             compression["spatial"]["compression_ratio"],
             compression["temporal"]["compression_ratio"],
             compression["unified"]["compression_ratio"])

    # 3. Loss Functions & Optimizer
    spatial_loss_fn = SpatialLoss(alpha=0.7, window_size=11, channel=5).to(device)
    temporal_loss_fn = nn.SmoothL1Loss(beta=1.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # 4. Training Loop
    history: Dict[str, List[float]] = {
        "train_total_loss": [],
        "train_spatial_loss": [],
        "train_spatial_mse": [],
        "train_spatial_ssim": [],
        "train_temporal_huber": [],
        "val_total_loss": [],
        "val_spatial_loss": [],
        "val_spatial_mse": [],
        "val_spatial_ssim": [],
        "val_temporal_huber": [],
    }

    best_val_loss = float("inf")
    best_checkpoint_path = args.output_dir / CHECKPOINT_NAME
    last_visual_sample = None

    max_train_batches = 12 if args.smoke_test else None
    max_val_batches = 4 if args.smoke_test else None
    if args.smoke_test:
        log.info("Smoke test active: limiting to %d train batches and %d val batches per epoch",
                 max_train_batches, max_val_batches)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        log.info("─── Epoch %d/%d ───", epoch, args.epochs)

        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            spatial_criterion=spatial_loss_fn,
            temporal_criterion=temporal_loss_fn,
            lambda_spatial=args.lambda_spatial,
            lambda_temporal=args.lambda_temporal,
            device=device,
            max_batches=max_train_batches,
        )

        val_metrics, visual_sample = validate(
            model=model,
            dataloader=val_loader,
            spatial_criterion=spatial_loss_fn,
            temporal_criterion=temporal_loss_fn,
            lambda_spatial=args.lambda_spatial,
            lambda_temporal=args.lambda_temporal,
            device=device,
            max_batches=max_val_batches,
        )
        last_visual_sample = visual_sample
        scheduler.step()

        # Record history
        history["train_total_loss"].append(train_metrics["total_loss"])
        history["train_spatial_loss"].append(train_metrics["spatial_loss"])
        history["train_spatial_mse"].append(train_metrics["spatial_mse"])
        history["train_spatial_ssim"].append(train_metrics["spatial_ssim"])
        history["train_temporal_huber"].append(train_metrics["temporal_huber"])

        history["val_total_loss"].append(val_metrics["total_loss"])
        history["val_spatial_loss"].append(val_metrics["spatial_loss"])
        history["val_spatial_mse"].append(val_metrics["spatial_mse"])
        history["val_spatial_ssim"].append(val_metrics["spatial_ssim"])
        history["val_temporal_huber"].append(val_metrics["temporal_huber"])

        log.info(
            "Epoch %d Results | Train Loss: %.4f (Sp MSE: %.4f, SSIM: %.4f, Temp Huber: %.4f) | "
            "Val Loss: %.4f (Sp MSE: %.4f, SSIM: %.4f, Temp Huber: %.4f)",
            epoch,
            train_metrics["total_loss"],
            train_metrics["spatial_mse"],
            train_metrics["spatial_ssim"],
            train_metrics["temporal_huber"],
            val_metrics["total_loss"],
            val_metrics["spatial_mse"],
            val_metrics["spatial_ssim"],
            val_metrics["temporal_huber"],
        )

        # Checkpoint Best Model
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": {
                    "spatial_channels": 5,
                    "temporal_features": 7,
                    "spatial_latent_dim": args.spatial_latent_dim,
                    "temporal_latent_dim": args.temporal_latent_dim,
                    "unified_latent_dim": args.spatial_latent_dim + args.temporal_latent_dim,
                },
                "history": history,
            }, best_checkpoint_path)
            log.info(" ★ New best model saved to: %s (Val Loss: %.4f)", best_checkpoint_path, best_val_loss)

    elapsed = time.time() - t0
    log.info("Training completed in %.2f seconds.", elapsed)

    # 5. Visual Reconstruction Check Plot
    if last_visual_sample is not None:
        plot_path = args.output_dir / RECONSTRUCTION_PLOT
        plot_visual_reconstruction(
            clean_sp=last_visual_sample[0],
            noisy_sp=last_visual_sample[1],
            rec_sp=last_visual_sample[2],
            clean_temp=last_visual_sample[3],
            rec_temp=last_visual_sample[4],
            history=history,
            out_path=plot_path,
        )

    # Print summary metrics JSON
    summary = {
        "epochs": args.epochs,
        "elapsed_seconds": elapsed,
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_total_loss"][-1],
        "final_val_loss": history["val_total_loss"][-1],
        "final_spatial_ssim": history["val_spatial_ssim"][-1],
        "checkpoint_path": str(best_checkpoint_path),
        "parameter_counts": param_counts,
        "compression_ratios": compression,
    }
    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved training summary to: %s", args.output_dir / "training_summary.json")


if __name__ == "__main__":
    main()
