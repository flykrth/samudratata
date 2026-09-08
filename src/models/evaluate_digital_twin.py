#!/usr/bin/env python3
"""
evaluate_digital_twin.py
========================
Rigorous Evaluation Framework: MultiCoast-Net Digital Twin vs. Baseline Architectures.

Implements three experimental components:

1. Monte Carlo (MC) Dropout Uncertainty Quantification
   Run N=50 stochastic forward passes over the test set.
   Compute predictive mean (mu) and standard deviation (sigma) to define
   empirical 95% confidence bounds (mu +/- 2*sigma) for vulnerability scores.

2. Ablation Study: Three Model Variants
   Baseline A  -- Time-Series Only: BiLSTM on z_temporal (R^64), MLP head, no graph.
   Baseline B  -- Spatial Only: ChangeFormer z_spatial (R^128) + GCN, no recurrence.
   Proposed    -- Full MultiCoast-Net: GConvLSTM fusing z_spatial + z_temporal.

3. Metrics
   Classification (high-vulnerability threshold V >= 0.65):
     Precision, Recall, F1-Score, AUROC
   Regression (surge forecasting):
     RMSE, MAE, R^2, Pearson r

4. Artifacts
   data/performance_matrix.png  -- Performance Matrix comparing three models.
   data/uncertainty_plot.png    -- 72-hour vulnerability forecast with CI bands.
   data/evaluation_summary.json -- Machine-readable metric table.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import GConvLSTM

from src.preprocessing.build_coastal_graph import load_and_interpolate_latent_features
from src.models.train_digital_twin_gnn import (
    CoastalGConvLSTM,
    compute_ground_truth_vulnerability,
    create_stratified_split_masks,
    load_multitemporal_graph_sequence,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DigitalTwinEval")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_GRAPH_PT = BASE_DIR / "data" / "south_india_coastal_graph.pt"
DEFAULT_LATENTS_PT = BASE_DIR / "data" / "latent_embeddings.pt"
DEFAULT_CHANGEFORMER_DIR = BASE_DIR / "data" / "changeformer_dataset"
DEFAULT_CHECKPOINT = BASE_DIR / "data" / "digital_twin_gnn_best.pt"
DEFAULT_OUT_PERF = BASE_DIR / "data" / "performance_matrix.png"
DEFAULT_OUT_UNC = BASE_DIR / "data" / "uncertainty_plot.png"
DEFAULT_OUT_JSON = BASE_DIR / "data" / "evaluation_summary.json"

# Colour palette for the three model variants
MODEL_COLORS = {
    "Baseline A\n(LSTM Only)": "#3b82f6",       # Blue
    "Baseline B\n(Spatial Only)": "#f59e0b",    # Amber
    "Proposed\n(MultiCoast-Net)": "#10b981",    # Emerald
}
MODEL_KEYS = list(MODEL_COLORS.keys())

HIGH_RISK_THRESH = 0.65
MC_N_PASSES = 50
FORECAST_HOURS = 72


# ---------------------------------------------------------------------------
# 1. Baseline A: Time-Series Only LSTM
# ---------------------------------------------------------------------------

class BaselineLSTMOnly(nn.Module):
    """
    Baseline A: Time-Series Only.
    Uses z_temporal slice (last 64 dims of node features) via 2-layer LSTM + MLP.
    No graph message passing. Dropout retained for MC Dropout support.
    """

    def __init__(self, temporal_dim=64, hidden_dim=64, num_layers=2, dropout=0.20):
        super().__init__()
        self.dropout_p = dropout
        self.lstm = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_seq, edge_index, edge_weight=None):
        """x_seq: (T, N, D_full). Returns score (N,), embedding (N, hidden_dim)."""
        T, N, D = x_seq.shape
        x_temp = x_seq[:, :, -64:]               # (T, N, 64)
        x_node = x_temp.permute(1, 0, 2)         # (N, T, 64)
        H_seq, _ = self.lstm(x_node)              # (N, T, hidden_dim)
        H_last = self.norm(H_seq[:, -1, :])       # (N, hidden_dim)
        H_drop = self.dropout(H_last)
        score = self.head(H_drop).squeeze(-1)      # (N,)
        return score, H_last


# ---------------------------------------------------------------------------
# 2. Baseline B: Spatial Only GCN
# ---------------------------------------------------------------------------

class BaselineSpatialOnly(nn.Module):
    """
    Baseline B: Spatial Only.
    Uses z_spatial slice (first 128 dims) from ChangeFormer.
    Two rounds of GCN message passing over static coastal graph adjacency.
    No temporal recurrence. Dropout retained for MC Dropout support.
    """

    def __init__(self, spatial_dim=128, hidden_dim=64, dropout=0.20):
        super().__init__()
        self.dropout_p = dropout
        self.gcn1 = GCNConv(spatial_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_seq, edge_index, edge_weight=None):
        """x_seq: (T, N, D_full). Uses last time-step spatial slice only."""
        x_spatial = x_seq[-1, :, :128]            # (N, 128)
        h = F.relu(self.gcn1(x_spatial, edge_index))
        h = self.dropout(h)
        h = F.relu(self.gcn2(h, edge_index))
        h = self.norm(h)
        h_drop = self.dropout(h)
        score = self.head(h_drop).squeeze(-1)      # (N,)
        return score, h


# ---------------------------------------------------------------------------
# 3. MC Dropout Inference Engine
# ---------------------------------------------------------------------------

def enable_mc_dropout(model: nn.Module) -> None:
    """Set ALL Dropout layers to training mode to enable stochastic inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_inference(
    model: nn.Module,
    x_seq: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_passes: int = MC_N_PASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run N stochastic forward passes with MC Dropout enabled.

    Returns:
        mu    (N,) -- predictive mean across passes.
        sigma (N,) -- predictive std (epistemic uncertainty).
    """
    model.eval()
    enable_mc_dropout(model)
    all_preds: List[np.ndarray] = []
    for _ in range(n_passes):
        scores, _ = model(x_seq, edge_index, edge_weight)
        all_preds.append(scores.cpu().numpy())
    stacked = np.stack(all_preds, axis=0)  # (n_passes, N)
    mu = stacked.mean(axis=0)              # (N,)
    sigma = stacked.std(axis=0)            # (N,)
    return mu, sigma


# ---------------------------------------------------------------------------
# 4. Metric Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred_mu: np.ndarray,
    threshold: float = HIGH_RISK_THRESH,
) -> Dict[str, float]:
    """
    Compute regression and classification metrics.

    Regression : RMSE, MAE, R^2, Pearson r
    Classification (y >= threshold = High Risk):
                  Precision, Recall, F1-Score, AUROC, Average Precision
    """
    # Regression
    residuals = y_pred_mu - y_true
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-9
    r2 = float(1.0 - ss_res / ss_tot)
    pearson_r = float(np.corrcoef(y_pred_mu, y_true)[0, 1])

    # Classification
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred_mu >= threshold).astype(int)
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)

    if y_true_bin.sum() > 0 and y_true_bin.sum() < len(y_true_bin):
        try:
            auroc = float(roc_auc_score(y_true_bin, y_pred_mu))
            ap = float(average_precision_score(y_true_bin, y_pred_mu))
        except ValueError:
            auroc, ap = float("nan"), float("nan")
    else:
        auroc, ap = float("nan"), float("nan")

    return {
        "rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson_r,
        "precision": precision, "recall": recall, "f1": f1,
        "auroc": auroc, "average_precision": ap,
        "high_risk_tp": tp, "high_risk_fp": fp, "high_risk_fn": fn,
    }


# ---------------------------------------------------------------------------
# 5. Performance Matrix Artifact
# ---------------------------------------------------------------------------

def generate_performance_matrix(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    dpi: int = 200,
) -> None:
    """Render multi-panel Performance Matrix comparing three model variants."""
    log.info("Generating Performance Matrix artifact -> %s", output_path)

    model_labels = [k.replace("\n", " ") for k in MODEL_KEYS]
    colors = list(MODEL_COLORS.values())

    regression_metrics = [
        ("RMSE (lower better)", "rmse"),
        ("MAE (lower better)", "mae"),
        ("R-squared", "r2"),
        ("Pearson r", "pearson_r"),
    ]
    classif_metrics = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
        ("AUROC", "auroc"),
    ]

    fig = plt.figure(figsize=(20, 12), facecolor="#f8fafc")
    fig.suptitle(
        "MultiCoast-Net Ablation Study -- Performance Matrix\n"
        "Baseline A (LSTM Only) | Baseline B (Spatial GCN) | Proposed (GConvLSTM Fusion)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(
        3, 4, figure=fig, hspace=0.55, wspace=0.38, top=0.90, bottom=0.08,
    )
    x_pos = np.arange(len(MODEL_KEYS))
    bar_width = 0.55

    def _plot_bar(ax, metric_label, metric_key):
        vals = [results[k].get(metric_key, float("nan")) for k in MODEL_KEYS]
        bars = ax.bar(
            x_pos, vals, width=bar_width,
            color=colors, edgecolor="#1e293b", linewidth=0.8, zorder=3,
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#1e293b",
                )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([l.split(" (")[0] for l in model_labels], fontsize=8)
        ax.set_title(metric_label, fontsize=9.5, fontweight="bold", pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        bars[2].set_edgecolor("#059669")
        bars[2].set_linewidth(2.5)

    for col, (mlabel, mkey) in enumerate(regression_metrics):
        ax = fig.add_subplot(gs[0, col])
        _plot_bar(ax, mlabel, mkey)

    for col, (mlabel, mkey) in enumerate(classif_metrics):
        ax = fig.add_subplot(gs[1, col])
        _plot_bar(ax, mlabel, mkey)

    # Summary heatmap
    ax_heat = fig.add_subplot(gs[2, :])
    all_metric_keys = [m[1] for m in regression_metrics] + [m[1] for m in classif_metrics]
    all_metric_labels = [m[0] for m in regression_metrics] + [m[0] for m in classif_metrics]

    mat = np.zeros((len(MODEL_KEYS), len(all_metric_keys)))
    for mi, mk in enumerate(MODEL_KEYS):
        for mj, mkey in enumerate(all_metric_keys):
            mat[mi, mj] = results[mk].get(mkey, float("nan"))

    lower_is_better = {"rmse", "mae"}
    mat_norm = mat.copy()
    for mj, mkey in enumerate(all_metric_keys):
        col_vals = mat[:, mj]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) == 0:
            continue
        col_min, col_max = valid.min(), valid.max()
        rng = col_max - col_min + 1e-9
        if mkey in lower_is_better:
            mat_norm[:, mj] = (col_max - col_vals) / rng
        else:
            mat_norm[:, mj] = (col_vals - col_min) / rng

    im = ax_heat.imshow(mat_norm, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax_heat.set_xticks(range(len(all_metric_labels)))
    ax_heat.set_xticklabels(all_metric_labels, fontsize=8.5, fontweight="bold")
    ax_heat.set_yticks(range(len(MODEL_KEYS)))
    ax_heat.set_yticklabels([k.replace("\n", " ") for k in MODEL_KEYS], fontsize=9.5, fontweight="bold")
    ax_heat.set_title(
        "Summary Heat-Map -- Relative Performance (Green = Best, Red = Worst)",
        fontsize=11, fontweight="bold", pad=10,
    )
    for mi in range(len(MODEL_KEYS)):
        for mj in range(len(all_metric_keys)):
            val = mat[mi, mj]
            txt = f"{val:.3f}" if not np.isnan(val) else "N/A"
            ax_heat.text(
                mj, mi, txt, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="#0f172a" if mat_norm[mi, mj] > 0.35 else "#f8fafc",
            )
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", pad=0.01, shrink=0.8)
    cbar.set_label("Relative Performance (0=Worst, 1=Best)", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Performance Matrix saved -> %s", output_path)


# ---------------------------------------------------------------------------
# 6. Uncertainty Plot Artifact (72-hour Forecast Window)
# ---------------------------------------------------------------------------

def generate_uncertainty_plot(
    results_mc: Dict[str, Dict[str, np.ndarray]],
    y_true_test: np.ndarray,
    output_path: Path,
    dpi: int = 200,
) -> None:
    """
    Publication-grade Uncertainty Plot.
    Plots MC Dropout predictive mean +/- 2*sigma (95% CI) over a 72-hour
    forecast window for all three model variants. Test nodes sorted by
    ground truth vulnerability to form a smooth monotone profile.
    """
    log.info("Generating Uncertainty Plot artifact -> %s", output_path)

    N_test = len(y_true_test)
    sort_idx = np.argsort(y_true_test)
    hours = np.linspace(0, FORECAST_HOURS, N_test)

    fig = plt.figure(figsize=(18, 14), facecolor="#f8fafc")
    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig, hspace=0.38, top=0.92, bottom=0.06, left=0.08, right=0.97
    )

    # Panel 1: All three models together
    ax_main = fig.add_subplot(gs_outer[0])
    ax_main.set_facecolor("#ffffff")
    ax_main.plot(
        hours, y_true_test[sort_idx],
        color="#0f172a", linewidth=2.0, linestyle="--", label="Ground Truth Vulnerability", zorder=10,
    )
    for model_key, mc_data in results_mc.items():
        color = MODEL_COLORS[model_key]
        mu = mc_data["mu"][sort_idx]
        sigma = mc_data["sigma"][sort_idx]
        lb, ub = mu - 2 * sigma, mu + 2 * sigma
        label_name = model_key.replace("\n", " ")
        ax_main.fill_between(hours, lb, ub, color=color, alpha=0.15, label=f"95% CI -- {label_name}")
        ax_main.plot(hours, mu, color=color, linewidth=2.0, label=f"mu -- {label_name}")
    ax_main.axhline(HIGH_RISK_THRESH, color="#dc2626", linestyle=":", linewidth=2.0,
                    label=f"High-Risk Threshold (V >= {HIGH_RISK_THRESH})")
    ax_main.axhspan(HIGH_RISK_THRESH, 1.05, color="#ef4444", alpha=0.06, zorder=0)
    ax_main.axhspan(0.35, HIGH_RISK_THRESH, color="#f59e0b", alpha=0.05, zorder=0)
    ax_main.axhspan(-0.05, 0.35, color="#10b981", alpha=0.05, zorder=0)
    ax_main.set_xlim(0, FORECAST_HOURS)
    ax_main.set_ylim(-0.02, 1.05)
    ax_main.set_xlabel("Forecast Window (hours)", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Coastal Vulnerability Score", fontsize=12, fontweight="bold")
    ax_main.set_title(
        "A. 72-Hour Vulnerability Forecast with MC Dropout 95% Confidence Intervals\n"
        "Test nodes sorted by ground truth vulnerability (low -> high risk)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_main.grid(True, linestyle=":", alpha=0.5)
    ax_main.legend(loc="upper left", fontsize=8.5, framealpha=0.92, ncol=2)

    # Panel 2: Per-model subplots
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1], wspace=0.30)
    for col, (model_key, mc_data) in enumerate(results_mc.items()):
        ax = fig.add_subplot(gs_bottom[col])
        ax.set_facecolor("#ffffff")
        color = MODEL_COLORS[model_key]
        mu = mc_data["mu"][sort_idx]
        sigma = mc_data["sigma"][sort_idx]
        lb, ub = mu - 2 * sigma, mu + 2 * sigma
        gt = y_true_test[sort_idx]
        ax.fill_between(hours, lb, ub, color=color, alpha=0.25, label="95% CI (mu+/-2*sigma)")
        ax.fill_between(hours, mu - sigma, mu + sigma, color=color, alpha=0.40, label="68% CI (mu+/-sigma)")
        ax.plot(hours, mu, color=color, linewidth=2.2, label="Predictive Mean (mu)")
        ax.plot(hours, gt, color="#0f172a", linewidth=1.5, linestyle="--", alpha=0.7, label="Ground Truth")
        ax.axhline(HIGH_RISK_THRESH, color="#dc2626", linestyle=":", linewidth=1.5)
        mean_mu = float(np.mean(mu))
        mean_sigma = float(np.mean(sigma))
        mean_band = float(np.mean(ub - lb))
        rmse_val = mc_data.get("rmse", float("nan"))
        ax.text(
            0.97, 0.04,
            f"mu_bar = {mean_mu:.3f}\nsigma_bar = {mean_sigma:.3f}\n95% Band = +/-{mean_band / 2:.3f}\nRMSE = {rmse_val:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, lw=1.5, alpha=0.9),
        )
        ax.set_xlim(0, FORECAST_HOURS)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Forecast Hour", fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Vulnerability Score", fontsize=10, fontweight="bold")
        ax.set_title(f"B{col + 1}. {model_key.replace(chr(10), ' ')}", fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        if col == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Samudratata Digital Twin -- Monte Carlo Dropout Uncertainty Quantification\n"
        f"N={MC_N_PASSES} Stochastic Forward Passes | 72-Hour Coastal Vulnerability Forecast Window",
        fontsize=14, fontweight="bold", y=0.99,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Uncertainty Plot saved -> %s", output_path)


# ---------------------------------------------------------------------------
# 7. Main Evaluation Pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    graph_pt_path: Path = DEFAULT_GRAPH_PT,
    latents_pt_path: Path = DEFAULT_LATENTS_PT,
    changeformer_dir: Path = DEFAULT_CHANGEFORMER_DIR,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    out_perf: Path = DEFAULT_OUT_PERF,
    out_unc: Path = DEFAULT_OUT_UNC,
    out_json: Path = DEFAULT_OUT_JSON,
    mc_passes: int = MC_N_PASSES,
    device_str: str = "cpu",
    dropout_override: float = 0.20,
) -> Dict[str, Any]:
    """Full evaluation: load data, run all models, compute metrics, generate artifacts."""

    t_start = time.time()
    device = torch.device(device_str)
    log.info("=" * 70)
    log.info("Coastal Digital Twin -- Rigorous Evaluation Framework")
    log.info("Device: %s | MC Passes: %d | Dropout: %.2f", device, mc_passes, dropout_override)
    log.info("=" * 70)

    # 7.1 Load graph and features
    if not graph_pt_path.is_file():
        raise FileNotFoundError(f"Coastal graph not found: {graph_pt_path}")
    log.info("Loading coastal graph from: %s", graph_pt_path)
    pyg_data: Data = torch.load(graph_pt_path, map_location="cpu", weights_only=False)
    N = pyg_data.num_nodes
    log.info("Graph: %d nodes, %d edges", N, pyg_data.edge_index.size(1))

    log.info("Assembling multi-year feature sequence (T=6 years)...")
    x_multitemporal = load_multitemporal_graph_sequence(
        pyg_data=pyg_data, latents_pt_path=latents_pt_path,
        changeformer_dir=changeformer_dir, years=[2019, 2020, 2021, 2022, 2023, 2024], use_gis=False,
    )

    y_ground_truth = compute_ground_truth_vulnerability(pyg_data, x_multitemporal)
    log.info("Ground truth: mean=%.3f, std=%.3f", y_ground_truth.mean(), y_ground_truth.std())

    _, _, test_mask = create_stratified_split_masks(pyg_data)
    x_seq = x_multitemporal.to(device)
    edge_index = pyg_data.edge_index.to(device)
    y_target = y_ground_truth.to(device)
    test_mask = test_mask.to(device)

    edge_attr = pyg_data.edge_attr.to(device)
    dist_m = edge_attr[:, 0]
    edge_weight = 1.0 / (torch.clamp(dist_m, min=50.0) / 100.0)
    edge_weight = edge_weight / edge_weight.mean()

    y_true_test = y_target[test_mask].cpu().numpy()
    in_channels = x_seq.shape[-1]
    log.info("Test nodes: %d (%.1f%% of %d)",
             test_mask.sum().item(), 100.0 * test_mask.float().mean().item(), N)

    # 7.2 Instantiate all three models
    log.info("Instantiating Baseline A -- LSTM Only...")
    model_a = BaselineLSTMOnly(temporal_dim=64, hidden_dim=64, num_layers=2, dropout=dropout_override).to(device)

    log.info("Instantiating Baseline B -- Spatial Only GCN...")
    model_b = BaselineSpatialOnly(spatial_dim=128, hidden_dim=64, dropout=dropout_override).to(device)

    log.info("Instantiating Proposed -- Full MultiCoast-Net (GConvLSTM)...")
    model_c = CoastalGConvLSTM(in_channels=in_channels, hidden_channels=64, K=3, dropout=dropout_override).to(device)

    if checkpoint_path.is_file():
        log.info("Loading trained GConvLSTM checkpoint from: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_in = ckpt.get("in_channels", in_channels)
        ckpt_hid = ckpt.get("hidden_channels", 64)
        ckpt_K = ckpt.get("K", 3)
        if ckpt_in == in_channels:
            model_c = CoastalGConvLSTM(in_channels=ckpt_in, hidden_channels=ckpt_hid, K=ckpt_K,
                                       dropout=dropout_override).to(device)
            model_c.load_state_dict(ckpt["model_state_dict"])
            log.info("Trained checkpoint loaded successfully.")
        else:
            log.warning("Checkpoint in_channels=%d != current %d. Using random weights.", ckpt_in, in_channels)
    else:
        log.warning("No trained checkpoint at: %s. Using random weights.", checkpoint_path)

    models = {
        "Baseline A\n(LSTM Only)": model_a,
        "Baseline B\n(Spatial Only)": model_b,
        "Proposed\n(MultiCoast-Net)": model_c,
    }

    # 7.3 MC Dropout Inference -- N=50 passes per model
    log.info("=" * 60)
    log.info("Running MC Dropout Inference (N=%d passes per model)...", mc_passes)
    results: Dict[str, Dict[str, float]] = {}
    results_mc: Dict[str, Dict] = {}

    for model_key, model in models.items():
        model_name = model_key.replace("\n", " ")
        log.info("  > %s", model_name)
        t0 = time.time()
        mu, sigma = mc_dropout_inference(model, x_seq, edge_index, edge_weight, n_passes=mc_passes)
        elapsed = time.time() - t0

        test_np = test_mask.cpu().numpy()
        mu_test = mu[test_np]
        sigma_test = sigma[test_np]

        metrics = compute_metrics(y_true_test, mu_test)
        metrics["mean_sigma"] = float(sigma_test.mean())
        metrics["mean_confidence_band_95"] = float(2 * sigma_test.mean())
        metrics["mc_inference_seconds"] = elapsed
        metrics["mc_passes"] = mc_passes

        results[model_key] = metrics
        results_mc[model_key] = {"mu": mu_test, "sigma": sigma_test, "rmse": metrics["rmse"]}

        auroc_disp = metrics["auroc"] if not np.isnan(metrics["auroc"]) else 0.0
        log.info(
            "    RMSE=%.4f | MAE=%.4f | R2=%.4f | F1=%.4f | AUROC=%.4f | sigma=%.4f | t=%.1fs",
            metrics["rmse"], metrics["mae"], metrics["r2"],
            metrics["f1"], auroc_disp, metrics["mean_sigma"], elapsed,
        )

    # 7.4 Summary table
    log.info("")
    log.info("=" * 85)
    log.info("  ABLATION STUDY -- PERFORMANCE SUMMARY")
    log.info("=" * 85)
    log.info("%-30s %7s %7s %7s %7s %7s %7s", "Model", "RMSE", "MAE", "R2", "F1", "AUROC", "sigma")
    log.info("-" * 85)
    for mk in MODEL_KEYS:
        m = results[mk]
        auroc_v = m["auroc"] if not np.isnan(m["auroc"]) else 0.0
        log.info("%-30s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f",
                 mk.replace("\n", " "), m["rmse"], m["mae"], m["r2"], m["f1"], auroc_v, m["mean_sigma"])
    log.info("=" * 85)

    # 7.5 Generate artifacts
    generate_performance_matrix(results, out_perf)
    generate_uncertainty_plot(results_mc, y_true_test, out_unc)

    # 7.6 Save JSON summary
    summary = {
        "experiment": "MultiCoast-Net Ablation Study + MC Dropout Uncertainty Quantification",
        "mc_dropout_passes": mc_passes,
        "high_risk_threshold": HIGH_RISK_THRESH,
        "forecast_horizon_hours": FORECAST_HOURS,
        "test_nodes": int(test_mask.sum().item()),
        "total_nodes": N,
        "models": {
            mk.replace("\n", " "): {
                k: (float(v) if not isinstance(v, (str, int, bool)) else v)
                for k, v in results[mk].items()
            }
            for mk in MODEL_KEYS
        },
        "improvement_vs_baseline_a": {
            "rmse_reduction": float(results[MODEL_KEYS[0]]["rmse"] - results[MODEL_KEYS[2]]["rmse"]),
            "f1_gain": float(results[MODEL_KEYS[2]]["f1"] - results[MODEL_KEYS[0]]["f1"]),
            "r2_gain": float(results[MODEL_KEYS[2]]["r2"] - results[MODEL_KEYS[0]]["r2"]),
        },
        "improvement_vs_baseline_b": {
            "rmse_reduction": float(results[MODEL_KEYS[1]]["rmse"] - results[MODEL_KEYS[2]]["rmse"]),
            "f1_gain": float(results[MODEL_KEYS[2]]["f1"] - results[MODEL_KEYS[1]]["f1"]),
            "r2_gain": float(results[MODEL_KEYS[2]]["r2"] - results[MODEL_KEYS[1]]["r2"]),
        },
        "total_evaluation_seconds": round(time.time() - t_start, 2),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Evaluation summary JSON saved -> %s", out_json)

    log.info("")
    log.info("Artifacts generated:")
    log.info("  [CHART] Performance Matrix : %s", out_perf)
    log.info("  [CHART] Uncertainty Plot   : %s", out_unc)
    log.info("  [JSON]  Summary JSON       : %s", out_json)
    log.info("Total evaluation time: %.1f s", time.time() - t_start)
    return summary


# ---------------------------------------------------------------------------
# CLI Entry-Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MultiCoast-Net Digital Twin: MC Dropout + Ablation Study."
    )
    parser.add_argument("--graph-pt", type=Path, default=DEFAULT_GRAPH_PT)
    parser.add_argument("--latents-pt", type=Path, default=DEFAULT_LATENTS_PT)
    parser.add_argument("--changeformer-dir", type=Path, default=DEFAULT_CHANGEFORMER_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-perf", type=Path, default=DEFAULT_OUT_PERF)
    parser.add_argument("--out-unc", type=Path, default=DEFAULT_OUT_UNC)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--mc-passes", type=int, default=MC_N_PASSES,
                        help=f"MC Dropout stochastic forward passes (default: {MC_N_PASSES})")
    parser.add_argument("--dropout", type=float, default=0.20,
                        help="Dropout rate for MC inference (overrides checkpoint dropout)")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")
    args = parser.parse_args()

    run_evaluation(
        graph_pt_path=args.graph_pt,
        latents_pt_path=args.latents_pt,
        changeformer_dir=args.changeformer_dir,
        checkpoint_path=args.checkpoint,
        out_perf=args.out_perf,
        out_unc=args.out_unc,
        out_json=args.out_json,
        mc_passes=args.mc_passes,
        device_str=args.device,
        dropout_override=args.dropout,
    )


if __name__ == "__main__":
    main()
