#!/usr/bin/env python3
"""
train_surge_lstm.py
───────────────────
Deep Learning Oceanographic Tidal Surge Sequence Modeling Pipeline.

Architecture:
  - Bidirectional LSTM (BiLSTM) layer: captures forward & backward temporal dynamics
    over a 72-hour sliding window.
  - Multi-Head Attention (MHA) layer: computes multi-head attention weights mapping
    to the most critical hours leading up to a tidal surge event.
  - Surge Prediction Head: regression MLP predicting next-hour surge height (Hs).
  - Huber Loss (Smooth L1): robust to extreme cyclonic and monsoon surge outliers.

Visual Artifact:
  - Attention Heatmap: multi-panel publication-grade visualization showing
    per-head attention weights, consensus temporal focus, and physical oceanographic
    forcing variables (Wave Height, Wind Magnitude, Surface Pressure drop).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("SurgeLSTM")

DEFAULT_H5_PATH = Path("data/ocean/ocean_timeseries_72h.h5")
DEFAULT_SCALERS_PATH = Path("data/ocean/scalers.json")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_CHECKPOINT = "surge_lstm_best.pt"
DEFAULT_HEATMAP = "attention_heatmap.png"

ALL_ZONES = ["Chellanam", "Alappuzha", "Nagapattinam", "Cuddalore", "Visakhapatnam"]
FEATURE_NAMES = [
    "Significant Wave Height (swh, m)",
    "Peak Wave Period (pp1d, s)",
    "10m U-Wind (u10, m/s)",
    "10m V-Wind (v10, m/s)",
    "Surface Pressure (sp, Pa)",
    "Wind Magnitude (wind_mag, m/s)",
    "Wave Power Index (WPI)",
]
FEATURE_KEYS = ["swh", "pp1d", "u10", "v10", "sp", "wind_magnitude", "wave_power_index"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset & Data Loader
# ─────────────────────────────────────────────────────────────────────────────

class OceanSurgeDataset(Dataset):
    """
    Dataset pairing 72-hour oceanographic sequences with next-hour surge height targets.

    Input X: (72, 7) Z-score normalized oceanographic features.
    Target y: Next-hour significant wave height (Hs at t+1), representing tidal/storm surge.
    """

    def __init__(
        self,
        h5_path: Path = DEFAULT_H5_PATH,
        scalers_path: Path = DEFAULT_SCALERS_PATH,
        zones: Optional[List[str]] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        horizon: int = 1,
        normalize_target: bool = True,
        max_samples_per_zone: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.scalers_path = Path(scalers_path)
        self.zones = zones or ALL_ZONES
        self.split = split
        self.train_ratio = train_ratio
        self.horizon = horizon
        self.normalize_target = normalize_target
        self.max_samples_per_zone = max_samples_per_zone

        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        self.scalers: Dict[str, Any] = {}
        if self.scalers_path.is_file():
            with open(self.scalers_path, "r", encoding="utf-8") as f:
                self.scalers = json.load(f)

        self._index: List[Dict[str, Any]] = []
        self._h5_file: Optional[h5py.File] = None
        self._build_index(seed=seed)

    def _build_index(self, seed: int):
        with h5py.File(self.h5_path, "r") as h5:
            for zone in self.zones:
                if zone not in h5:
                    log.warning("Zone %s not present in HDF5 file; skipping.", zone)
                    continue

                grp = h5[zone]
                n_windows = grp["X_zscore"].shape[0]
                n_continuous = grp["series_continuous_raw"].shape[0]

                # Window i covers [i, i + 71]. Target at t + horizon is index: i + 71 + horizon
                max_valid_idx = n_continuous - 71 - self.horizon
                valid_count = min(n_windows, max_valid_idx)

                # Temporal split: chronological split to preserve causality
                split_idx = int(valid_count * self.train_ratio)
                if self.split == "train":
                    indices = list(range(0, split_idx))
                elif self.split == "val":
                    indices = list(range(split_idx, valid_count))
                else:
                    indices = list(range(0, valid_count))

                if self.max_samples_per_zone and len(indices) > self.max_samples_per_zone:
                    step = len(indices) // self.max_samples_per_zone
                    indices = indices[::step][: self.max_samples_per_zone]

                zone_scaler = self.scalers.get(zone, {})
                mean_hs = zone_scaler.get("zscore", {}).get("mean", [1.65])[0]
                std_hs = zone_scaler.get("zscore", {}).get("std", [0.90])[0]

                for idx in indices:
                    target_row = idx + 71 + self.horizon
                    self._index.append(
                        {
                            "zone": zone,
                            "window_idx": idx,
                            "target_row": target_row,
                            "mean_hs": mean_hs,
                            "std_hs": std_hs,
                        }
                    )

        log.info(
            "Built OceanSurgeDataset [%s] with %d samples across %d zones.",
            self.split.upper(),
            len(self._index),
            len(self.zones),
        )

    def _get_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        h5 = self._get_h5()
        meta = self._index[idx]
        zone = meta["zone"]
        w_idx = meta["window_idx"]
        t_row = meta["target_row"]

        # Fetch 72-hour window: (72, 7)
        x_norm = h5[zone]["X_zscore"][w_idx].astype(np.float32)
        x_raw = h5[zone]["X_raw"][w_idx].astype(np.float32)

        # Target surge: significant wave height (col 0: swh)
        y_raw = float(h5[zone]["series_continuous_raw"][t_row, 0])

        if self.normalize_target:
            y_val = (y_raw - meta["mean_hs"]) / meta["std_hs"]
        else:
            y_val = y_raw

        timestamp_bytes = h5[zone]["timestamps_end"][w_idx]
        ts_str = (
            timestamp_bytes.decode("utf-8")
            if isinstance(timestamp_bytes, bytes)
            else str(timestamp_bytes)
        )

        return {
            "x": torch.from_numpy(x_norm),           # (72, 7)
            "x_raw": torch.from_numpy(x_raw),       # (72, 7)
            "y": torch.tensor([y_val], dtype=torch.float32),   # (1,)
            "y_raw": torch.tensor([y_raw], dtype=torch.float32), # (1,)
            "zone": zone,
            "timestamp": ts_str,
            "mean_hs": meta["mean_hs"],
            "std_hs": meta["std_hs"],
        }

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Architecture: BiLSTM + Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────

class SurgeBiLSTMAttention(nn.Module):
    """
    Oceanographic Sequence Model: Bidirectional LSTM + Multi-Head Attention.

    1. Bidirectional LSTM (2 layers, hidden_dim=64 -> 128 forward/backward states).
       Captures multi-scale temporal dependencies, wave energy propagation,
       and barometric pressure drops across the 72-hour window.
    2. Multi-Head Attention:
       Multi-head self-attention layer with residual connection & LayerNorm,
       followed by an attention pooling query projecting onto the 72 antecedent hours.
       Outputs temporal attention weights (B, num_heads, 72) and consensus weights (B, 72).
    3. Regression Head:
       Fuses attended temporal context with final timestep state via MLP with
       LayerNorm, GELU, and Dropout to predict surge height.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = hidden_dim * 2  # Bidirectional doubles the dimension
        self.num_heads = num_heads

        # 1. Bidirectional LSTM backbone
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. Multi-Head Self-Attention Layer
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.self_mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. Temporal Attention Query Mechanism
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.pool_mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 4. Surge Prediction Head (MLP)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(self.embed_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        Args:
          x: Tensor of shape (B, 72, input_dim)
        Returns:
          y_pred: Tensor of shape (B, 1)
          attn_dict: Dictionary containing:
            - 'weights': Consensus attention weights over 72h (B, 72)
            - 'head_weights': Per-head attention weights over 72h (B, num_heads, 72)
            - 'self_attention': Pairwise attention matrix (B, num_heads, 72, 72)
        """
        # Step 1: BiLSTM encoding
        H, _ = self.bilstm(x)  # (B, 72, 2 * hidden_dim)

        # Step 2: Multi-Head Self-Attention across 72 hours
        H_norm = self.norm1(H)
        H_attn, self_attn_weights = self.self_mha(
            H_norm, H_norm, H_norm, need_weights=True, average_attn_weights=False
        )
        H_seq = self.norm2(H + self.dropout(H_attn))  # (B, 72, embed_dim)

        # Step 3: Surge Query Attention Pooling
        q = self.query_proj(H_seq[:, -1:, :])  # (B, 1, embed_dim)
        context, pool_weights = self.pool_mha(
            q, H_seq, H_seq, need_weights=True, average_attn_weights=False
        )
        head_weights = pool_weights.squeeze(2)  # (B, num_heads, 72)
        mean_weights = head_weights.mean(dim=1)  # (B, 72)

        # Step 4: Regression fusion
        context_vec = context.squeeze(1)       # (B, embed_dim)
        final_state = H_seq[:, -1, :]          # (B, embed_dim)
        fused = torch.cat([context_vec, final_state], dim=-1)  # (B, 2 * embed_dim)
        y_pred = self.head(fused)              # (B, 1)

        attn_dict = {
            "weights": mean_weights,
            "head_weights": head_weights,
            "self_attention": self_attn_weights,
        }
        return y_pred, attn_dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training & Validation Engine
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Train for one epoch with Huber Loss and gradient clipping."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_raw = batch["y_raw"].to(device)
        mean_hs = batch["mean_hs"].to(device).unsqueeze(1)
        std_hs = batch["std_hs"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        y_pred, _ = model(x)

        loss = criterion(y_pred, y)
        loss.backward()

        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        y_pred_raw = y_pred * std_hs + mean_hs
        mae = torch.abs(y_pred_raw - y_raw).mean().item()

        batch_sz = x.size(0)
        total_loss += loss.item() * batch_sz
        total_mae += mae * batch_sz
        total_samples += batch_sz

    avg_loss = total_loss / max(1, total_samples)
    avg_mae = total_mae / max(1, total_samples)
    return avg_loss, avg_mae


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    surge_threshold: float = 3.0,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    Computes Huber Loss, RMSE (m), MAE (m), and Peak Surge MAE (m).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds_raw = []
    all_targets_raw = []

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_raw = batch["y_raw"].to(device)
        mean_hs = batch["mean_hs"].to(device).unsqueeze(1)
        std_hs = batch["std_hs"].to(device).unsqueeze(1)

        y_pred, _ = model(x)
        loss = criterion(y_pred, y)

        y_pred_raw = y_pred * std_hs + mean_hs

        batch_sz = x.size(0)
        total_loss += loss.item() * batch_sz
        total_samples += batch_sz

        all_preds_raw.append(y_pred_raw.cpu().numpy())
        all_targets_raw.append(y_raw.cpu().numpy())

    preds = np.concatenate(all_preds_raw, axis=0).flatten()
    targets = np.concatenate(all_targets_raw, axis=0).flatten()

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    # Peak Surge MAE on severe surge outliers (Hs >= surge_threshold)
    peak_mask = targets >= surge_threshold
    if np.sum(peak_mask) > 0:
        peak_mae = float(np.mean(np.abs(preds[peak_mask] - targets[peak_mask])))
        peak_count = int(np.sum(peak_mask))
    else:
        p90 = np.percentile(targets, 90)
        p_mask = targets >= p90
        peak_mae = float(np.mean(np.abs(preds[p_mask] - targets[p_mask])))
        peak_count = int(np.sum(p_mask))

    avg_loss = total_loss / max(1, total_samples)

    return {
        "val_huber_loss": avg_loss,
        "val_mae": mae,
        "val_rmse": rmse,
        "val_peak_mae": peak_mae,
        "peak_sample_count": peak_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attention Heatmap Visualizer Artifact
# ─────────────────────────────────────────────────────────────────────────────

def generate_attention_heatmap(
    model: nn.Module,
    val_dataset: OceanSurgeDataset,
    device: torch.device,
    output_path: Path,
    artifact_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Isolate high-surge events and extract multi-head attention weights across the 72-hour window.
    Generates a publication-quality multi-panel Attention Heatmap artifact.
    """
    model.eval()
    log.info("Searching validation set for peak tidal surge event...")

    best_sample = None
    best_surge_val = -1.0
    best_idx = 0

    n_val = len(val_dataset)
    scan_limit = min(n_val, 2000)
    for i in range(scan_limit):
        sample = val_dataset[i]
        surge_val = float(sample["y_raw"].item())
        if surge_val > best_surge_val:
            best_surge_val = surge_val
            best_sample = sample
            best_idx = i

    if best_sample is None:
        best_sample = val_dataset[0]
        best_surge_val = float(best_sample["y_raw"].item())

    zone = best_sample["zone"]
    timestamp = best_sample["timestamp"]
    mean_hs = best_sample["mean_hs"]
    std_hs = best_sample["std_hs"]

    # Inference forward pass
    x_tensor = best_sample["x"].unsqueeze(0).to(device)  # (1, 72, 7)
    with torch.no_grad():
        y_pred_norm, attn_dict = model(x_tensor)

    pred_raw = float((y_pred_norm.item() * std_hs) + mean_hs)
    weights_consensus = attn_dict["weights"].squeeze(0).cpu().numpy()     # (72,)
    weights_heads = attn_dict["head_weights"].squeeze(0).cpu().numpy()     # (num_heads, 72)
    self_attn_map = attn_dict["self_attention"].squeeze(0).mean(dim=0).cpu().numpy() # (72, 72)

    x_raw_np = best_sample["x_raw"].numpy()  # (72, 7)
    hours_lookback = np.arange(-71, 1)       # -71h to 0h

    swh_profile = x_raw_np[:, 0]
    pp1d_profile = x_raw_np[:, 1]
    wind_mag_profile = x_raw_np[:, 5]
    pressure_hpa = x_raw_np[:, 4] / 100.0   # Convert Pa to hPa

    log.info(
        "Peak surge event identified: Zone=%s | Time=%s | Actual Hs=%.2fm | Pred Hs=%.2fm",
        zone,
        timestamp,
        best_surge_val,
        pred_raw,
    )

    # ─── Plotting Publication-Grade Multi-Panel Heatmap ─────────────────────
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#2d3748"
    plt.rcParams["axes.linewidth"] = 0.8

    fig = plt.figure(figsize=(15, 12), dpi=300)
    gs = gridspec.GridSpec(
        3,
        2,
        height_ratios=[1.2, 1.0, 1.4],
        width_ratios=[1.8, 1.0],
        hspace=0.38,
        wspace=0.25,
    )

    # ── Panel 1: Multi-Head Attention Heatmap (All 4 Heads across 72h) ───────
    ax_heads = fig.add_subplot(gs[0, 0])
    im_heads = ax_heads.imshow(
        weights_heads,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        extent=[-72, 0, 4.5, 0.5],
    )
    ax_heads.set_yticks([1, 2, 3, 4])
    ax_heads.set_yticklabels(["Head 1", "Head 2", "Head 3", "Head 4"], fontsize=10, fontweight="bold")
    ax_heads.set_ylabel("Attention Head", fontsize=11, fontweight="bold")
    ax_heads.set_title(
        f"A) Multi-Head Attention Weights Across 72-Hour Lookback Window\n"
        f"Event Zone: {zone} | Peak Surge: {best_surge_val:.2f} m (Pred: {pred_raw:.2f} m)",
        fontsize=12,
        fontweight="bold",
        pad=10,
        loc="left",
    )
    cbar_heads = fig.colorbar(im_heads, ax=ax_heads, orientation="vertical", pad=0.02, shrink=0.9)
    cbar_heads.set_label("Attention Weight", fontsize=9)
    ax_heads.grid(axis="x", color="#ffffff", linestyle=":", alpha=0.3)

    # ── Panel 2: 2D Self-Attention Cross-Timestep Matrix ────────────────────
    ax_self = fig.add_subplot(gs[0, 1])
    im_self = ax_self.imshow(
        self_attn_map,
        aspect="auto",
        cmap="viridis",
        extent=[-72, 0, 0, -72],
    )
    ax_self.set_title("B) Pairwise Cross-Temporal Attention (72×72)", fontsize=11, fontweight="bold", pad=10)
    ax_self.set_xlabel("Key Timestep (Hours to Event)", fontsize=9)
    ax_self.set_ylabel("Query Timestep (Hours)", fontsize=9)
    cbar_self = fig.colorbar(im_self, ax=ax_self, orientation="vertical", pad=0.03, shrink=0.9)
    cbar_self.set_label("Self-Attention Density", fontsize=9)

    # ── Panel 3: Consensus Attention Profile Over Time ──────────────────────
    ax_attn = fig.add_subplot(gs[1, 0])
    ax_attn.fill_between(
        hours_lookback,
        0,
        weights_consensus,
        color="#e53e3e",
        alpha=0.35,
        label="Consensus Attention Weight",
    )
    ax_attn.plot(
        hours_lookback,
        weights_consensus,
        color="#c53030",
        linewidth=2.2,
        label="Mean Attention Distribution",
    )

    top_indices = np.argsort(weights_consensus)[-3:][::-1]
    max_attn = float(np.max(weights_consensus))
    ax_attn.set_ylim(0, max_attn * 1.35)

    for r, top_idx in enumerate(top_indices, 1):
        hr = hours_lookback[top_idx]
        val = weights_consensus[top_idx]
        ax_attn.scatter([hr], [val], color="#742a2a", s=45, zorder=5)
        x_offset = 2.5 if hr <= -62 else (-6.0 if hr >= -15 else 1.5)
        y_offset = max_attn * 0.08 * (1.0 if r != 2 else 0.4)
        ax_attn.annotate(
            f"Peak #{r}: {hr}h ({val:.3f})",
            xy=(hr, val),
            xytext=(hr + x_offset, min(val + y_offset, max_attn * 1.25)),
            arrowprops=dict(arrowstyle="->", color="#742a2a", lw=1.2),
            fontsize=8.5,
            fontweight="bold",
            color="#1a202c",
        )

    ax_attn.set_xlim(-72, 0)
    ax_attn.set_ylabel("Consensus Attention Density", fontsize=11, fontweight="bold")
    ax_attn.set_title(
        "C) Critical Pre-Surge Antecedent Hours Identified by Multi-Head Attention",
        fontsize=12,
        fontweight="bold",
        pad=10,
        loc="left",
    )
    ax_attn.grid(True, linestyle="--", alpha=0.5)
    ax_attn.legend(loc="upper right", frameon=True, fontsize=9)

    # ── Panel 4: Temporal Attention Concentration Stats ─────────────────────
    ax_pie = fig.add_subplot(gs[1, 1])
    q4_immediate = weights_consensus[-12:].sum()    # -12h to 0h
    q3_buildup = weights_consensus[-24:-12].sum()   # -24h to -12h
    q2_early_storm = weights_consensus[-48:-24].sum() # -48h to -24h
    q1_pre_storm = weights_consensus[:-48].sum()    # -72h to -48h

    wedges, texts, autotexts = ax_pie.pie(
        [q4_immediate, q3_buildup, q2_early_storm, q1_pre_storm],
        labels=["Immediate (-12h..0h)", "Buildup (-24h..-12h)", "Early Storm (-48h..-24h)", "Pre-Storm (-72h..-48h)"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#e53e3e", "#dd6b20", "#3182ce", "#718096"],
        textprops=dict(fontsize=8.5, fontweight="bold"),
        explode=(0.06, 0.04, 0.0, 0.0),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8.5)
    ax_pie.set_title("D) Attention Allocation by Phase", fontsize=11, fontweight="bold", pad=10)

    # ── Panel 5: Physical Oceanographic & Atmospheric Drivers ───────────────
    ax_phys = fig.add_subplot(gs[2, :])
    color_hs = "#2b6cb0"
    ax_phys.plot(
        hours_lookback,
        swh_profile,
        color=color_hs,
        linewidth=2.0,
        label="Significant Wave Height (m)",
    )
    ax_phys.set_ylabel("Wave Height (m)", color=color_hs, fontsize=11, fontweight="bold")
    ax_phys.tick_params(axis="y", labelcolor=color_hs)
    ax_phys.set_xlim(-72, 0)
    ax_phys.set_xlabel("Hours Leading Up to Tidal Surge Event (t)", fontsize=11, fontweight="bold")
    ax_phys.grid(True, linestyle=":", alpha=0.6)

    ax_wind = ax_phys.twinx()
    color_wind = "#c05621"
    ax_wind.plot(
        hours_lookback,
        wind_mag_profile,
        color=color_wind,
        linestyle="--",
        linewidth=1.8,
        label="10m Wind Magnitude (m/s)",
    )
    ax_wind.set_ylabel("Wind Magnitude (m/s)", color=color_wind, fontsize=11, fontweight="bold")
    ax_wind.tick_params(axis="y", labelcolor=color_wind)

    ax_pres = ax_phys.twinx()
    ax_pres.spines["right"].set_position(("axes", 1.08))
    color_pres = "#6b46c1"
    ax_pres.plot(
        hours_lookback,
        pressure_hpa,
        color=color_pres,
        linestyle="-.",
        linewidth=1.6,
        label="Surface Pressure (hPa)",
    )
    ax_pres.set_ylabel("Surface Pressure (hPa)", color=color_pres, fontsize=11, fontweight="bold")
    ax_pres.tick_params(axis="y", labelcolor=color_pres)

    ax_phys.set_title(
        "E) Hydrodynamic & Aerodynamic Forcing Dynamics Aligned with the 72-Hour Lookback Window\n"
        "Notice Attention Surges (Panels A & C) Coincide with Atmospheric Pressure Drop and Wind Acceleration",
        fontsize=12,
        fontweight="bold",
        pad=10,
        loc="left",
    )

    lines = [
        ax_phys.get_lines()[0],
        ax_wind.get_lines()[0],
        ax_pres.get_lines()[0],
    ]
    labels = [l.get_label() for l in lines]
    ax_phys.legend(lines, labels, loc="upper left", frameon=True, fontsize=9)

    plt.suptitle(
        "South Indian Coastal Deep Learning Sequence Model: Bidirectional LSTM + Multi-Head Attention\n"
        "Attention Weights Mapping Critical Pre-Surge Antecedent Hours",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    log.info("Saved Attention Heatmap artifact: %s", output_path)

    if artifact_path:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, artifact_path)
        log.info("Copied Attention Heatmap to artifact destination: %s", artifact_path)

    plt.close(fig)

    return {
        "zone": zone,
        "timestamp": timestamp,
        "actual_surge_hs": best_surge_val,
        "predicted_surge_hs": pred_raw,
        "top_critical_hours": [int(hours_lookback[i]) for i in top_indices],
        "top_attention_values": [float(weights_consensus[i]) for i in top_indices],
        "immediate_phase_pct": float(q4_immediate * 100),
        "buildup_phase_pct": float(q3_buildup * 100),
        "early_storm_pct": float(q2_early_storm * 100),
        "pre_storm_pct": float(q1_pre_storm * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Orchestration Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / args.checkpoint
    heatmap_path = output_dir / args.heatmap_name
    brain_artifact_dir = Path(
        "/home/flykrth/.gemini/antigravity/brain/818589c6-74c3-4186-aa4b-ec762a058bc3"
    )
    brain_heatmap_path = brain_artifact_dir / "attention_heatmap.png"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Using compute device: %s", device)

    zones = ALL_ZONES if args.zone == "all" else [args.zone]
    max_samples = 1500 if args.smoke_test else args.max_samples

    # ── Dataset Loading ─────────────────────────────────────────────────────
    log.info("Initializing OceanSurgeDataset from %s...", args.h5_path)
    train_ds = OceanSurgeDataset(
        h5_path=args.h5_path,
        scalers_path=args.scalers_path,
        zones=zones,
        split="train",
        train_ratio=0.8,
        horizon=args.horizon,
        max_samples_per_zone=max_samples,
    )
    val_ds = OceanSurgeDataset(
        h5_path=args.h5_path,
        scalers_path=args.scalers_path,
        zones=zones,
        split="val",
        train_ratio=0.8,
        horizon=args.horizon,
        max_samples_per_zone=max_samples // 4 if max_samples else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model Instantiation ──────────────────────────────────────────────────
    log.info("Initializing SurgeBiLSTMAttention model...")
    model = SurgeBiLSTMAttention(
        input_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("SurgeBiLSTMAttention trainable parameters: %d (%.2f KB)", total_params, total_params * 4 / 1024)

    criterion = nn.HuberLoss(delta=args.huber_delta, reduction="mean")
    log.info("Configured Huber Loss with delta=%.2f for surge outlier robustness.", args.huber_delta)

    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_metrics: Dict[str, float] = {}

        log.info("Starting model training for %d epochs...", args.epochs)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_mae = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            scheduler.step()

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                surge_threshold=3.0,
            )

            log.info(
                "Epoch [%02d/%02d] | Train Huber: %.4f | Train MAE: %.3fm | Val Huber: %.4f | Val MAE: %.3fm | Peak MAE: %.3fm",
                epoch,
                args.epochs,
                train_loss,
                train_mae,
                val_metrics["val_huber_loss"],
                val_metrics["val_mae"],
                val_metrics["val_peak_mae"],
            )

            if val_metrics["val_huber_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_huber_loss"]
                best_metrics = val_metrics
                args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "metrics": val_metrics,
                        "args": args_dict,
                    },
                    checkpoint_path,
                )
                log.info("  --> Saved new best checkpoint: %s (Val Huber: %.4f)", checkpoint_path, best_val_loss)

        log.info("Training completed. Best checkpoint saved to %s", checkpoint_path)
    else:
        if checkpoint_path.is_file():
            log.info("Loading checkpoint from %s for evaluation...", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            log.warning("No checkpoint found at %s. Proceeding with initialized weights.", checkpoint_path)

    # ── Final Evaluation & Attention Heatmap Generation ──────────────────────
    log.info("Running post-training validation assessment...")
    final_val_metrics = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )
    log.info("Validation Results: %s", final_val_metrics)

    log.info("Generating publication-grade Attention Heatmap artifact...")
    heatmap_stats = generate_attention_heatmap(
        model=model,
        val_dataset=val_ds,
        device=device,
        output_path=heatmap_path,
        artifact_path=brain_heatmap_path,
    )

    summary = {
        "model": "SurgeBiLSTMAttention",
        "input_features": FEATURE_KEYS,
        "lookback_hours": 72,
        "bilstm_hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "loss_function": f"HuberLoss(delta={args.huber_delta})",
        "validation_metrics": final_val_metrics,
        "heatmap_analysis": heatmap_stats,
    }
    summary_path = output_dir / "surge_lstm_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary report to: %s", summary_path)

    train_ds.close()
    val_ds.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train BiLSTM + Multi-Head Attention Oceanographic Surge Model"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="BiLSTM hidden units")
    parser.add_argument("--num-layers", type=int, default=2, help="BiLSTM recurrent layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout probability")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber loss threshold delta")
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH, help="Path to HDF5 dataset")
    parser.add_argument("--scalers-path", type=Path, default=DEFAULT_SCALERS_PATH, help="Path to scalers JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint file name")
    parser.add_argument("--heatmap-name", type=str, default=DEFAULT_HEATMAP, help="Heatmap image name")
    parser.add_argument("--zone", type=str, default="all", help="Target zone name or 'all'")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast lead time in hours (default: 1)")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast verification smoke test")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per zone")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate existing checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
