#!/usr/bin/env python3
"""
train_changeformer_coastal.py
=============================
Fine-Tuning ChangeFormer (Siamese Vision Transformer) for Binary Coastal Change Detection.

Architecture & Transfer Learning:
1. Backbone: ChangeFormerV6 with Siamese ViT encoder and multi-scale MLP decoder head.
2. Transfer Learning: Pre-trained on LEVIR-CD building change detection.
3. Selective Fine-Tuning:
   - FROZEN: Initial transformer encoder blocks (Tenc_x2: ~28.9M parameters).
   - TRAINABLE: Multi-layer perceptron decoder head (TDec_x2: ~12.1M parameters).
4. Data Pipeline:
   - Ingests 256x256 Sentinel-2 .pt bitemporal tensors (B2, B3, B4, B8, NDWI).
   - Maps optical bands [B4_red, B3_green, B2_blue] to standard 3-channel Siamese input.
   - Derives binary coastal change ground truth via bitemporal NDWI land/water boundary dynamics.
5. Metrics & Tracking:
   - Logs validation metrics (F1-Score, IoU, Precision, Recall, Overall Accuracy) to Weights & Biases (WandB).
   - Generates publication-ready "Loss Curve" artifact upon completion of 10 epochs.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add ChangeFormer submodule to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHANGEFORMER_DIR = REPO_ROOT / "ChangeFormer"
if str(CHANGEFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(CHANGEFORMER_DIR))

try:
    from models.ChangeFormer import ChangeFormerV6
except ImportError as e:
    raise ImportError(f"Failed to import ChangeFormerV6 from {CHANGEFORMER_DIR}: {e}")

# Weights & Biases
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("CoastalChangeFormer")

# Default paths
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "changeformer_dataset"
DEFAULT_CHECKPOINT_PATH = (
    CHANGEFORMER_DIR
    / "checkpoints"
    / "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256"
    / "best_ckpt.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"
DEFAULT_LOSS_CURVE_PATH = REPO_ROOT / "loss_curve.png"
BEST_MODEL_PATH = DEFAULT_OUTPUT_DIR / "changeformer_coastal_best.pt"


# ---------------------------------------------------------------------------
# 1. Dataset & DataLoaders for 256x256 Sentinel-2 .pt Tensors
# ---------------------------------------------------------------------------
class CoastalSentinelDataset(Dataset):
    """
    Loads pre-processed 256x256 Sentinel-2 bitemporal patches from .pt files.
    
    Each .pt file contains:
    - 'T1': (N, 5, 256, 256) float32 [B2, B3, B4, B8, NDWI]
    - 'T2': (N, 5, 256, 256) float32 [B2, B3, B4, B8, NDWI]
    - 'coords': (N, 2)
    - 'meta': dict of metadata
    
    The Siamese ViT encoder expects 3-channel optical inputs:
    - Red:   B4 (Band index 2)
    - Green: B3 (Band index 1)
    - Blue:  B2 (Band index 0)
    
    Binary coastal change mask is derived via physical NDWI water-land boundary dynamics:
    - Water mask at T1: W1 = (NDWI_1 > 0.0)
    - Water mask at T2: W2 = (NDWI_2 > 0.0)
    - Coastal change mask: L = (W1 != W2).long() (1 = shoreline erosion / accretion change, 0 = unchanged)
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        val_ratio: float = 0.20,
        seed: int = 42,
        augment: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.samples: List[Dict[str, Any]] = []

        self._discover_and_split(val_ratio=val_ratio, seed=seed)

    def _discover_and_split(self, val_ratio: float, seed: int):
        pt_files = sorted(self.data_dir.rglob("pair_*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No pair_*.pt files found in {self.data_dir}")

        all_entries: List[Tuple[Path, int]] = []
        for pt_path in pt_files:
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                n_patches = data["T1"].shape[0]
                for idx in range(n_patches):
                    all_entries.append((pt_path, idx))
            except Exception as e:
                log.warning("Could not read %s: %s", pt_path, e)

        if not all_entries:
            raise ValueError(f"No valid patches found in {self.data_dir}")

        # Deterministic stratified or seeded shuffle
        rng = random.Random(seed)
        rng.shuffle(all_entries)

        split_idx = int(len(all_entries) * (1.0 - val_ratio))
        if self.split == "train":
            selected = all_entries[:split_idx]
        else:
            selected = all_entries[split_idx:]

        # Pre-cache indexed records
        for pt_path, idx in selected:
            self.samples.append({"file": pt_path, "idx": idx})

        log.info(
            "CoastalSentinelDataset [%s]: loaded %d samples (out of %d total)",
            self.split.upper(),
            len(self.samples),
            len(all_entries),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.samples[index]
        data = torch.load(entry["file"], map_location="cpu", weights_only=False)
        patch_idx = entry["idx"]

        t1 = data["T1"][patch_idx].float()  # (5, 256, 256)
        t2 = data["T2"][patch_idx].float()  # (5, 256, 256)

        # Extract RGB [B4, B3, B2]
        # Sentinel-2 indices: 2: B4 (Red), 1: B3 (Green), 0: B2 (Blue)
        rgb_t1 = t1[[2, 1, 0], :, :]  # (3, 256, 256)
        rgb_t2 = t2[[2, 1, 0], :, :]  # (3, 256, 256)

        # Compute Binary Ground Truth Coastal Change Mask from NDWI transitions
        # Index 4: NDWI = (Green - NIR) / (Green + NIR)
        if "L" in data:
            label = data["L"][patch_idx].long()
        else:
            ndwi1 = t1[4, :, :]
            ndwi2 = t2[4, :, :]
            water1 = ndwi1 > 0.0
            water2 = ndwi2 > 0.0
            label = (water1 != water2).long()  # (256, 256)

        # Optional data augmentations during training
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                rgb_t1 = torch.flip(rgb_t1, dims=[-1])
                rgb_t2 = torch.flip(rgb_t2, dims=[-1])
                label  = torch.flip(label, dims=[-1])
            # Random vertical flip
            if random.random() > 0.5:
                rgb_t1 = torch.flip(rgb_t1, dims=[-2])
                rgb_t2 = torch.flip(rgb_t2, dims=[-2])
                label  = torch.flip(label, dims=[-2])

        return {
            "A": rgb_t1,
            "B": rgb_t2,
            "L": label,
        }


# ---------------------------------------------------------------------------
# 2. Evaluation Metric Computation (F1-Score, IoU, Accuracy)
# ---------------------------------------------------------------------------
class CoastalMetricMeter:
    """
    Computes precision, recall, F1-Score, and Intersection over Union (IoU)
    for binary change detection (Class 0: No Change, Class 1: Coastal Change).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: (B, H, W) binary predictions {0, 1}
        targets: (B, H, W) binary ground truth {0, 1}
        """
        preds = preds.detach().cpu().long()
        targets = targets.detach().cpu().long()

        self.tp += float(((preds == 1) & (targets == 1)).sum())
        self.fp += float(((preds == 1) & (targets == 0)).sum())
        self.fn += float(((preds == 0) & (targets == 1)).sum())
        self.tn += float(((preds == 0) & (targets == 0)).sum())

    def get_metrics(self) -> Dict[str, float]:
        eps = 1e-7
        total = self.tp + self.fp + self.fn + self.tn + eps
        accuracy = (self.tp + self.tn) / total

        # Metrics for coastal change class (1)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1_score = 2.0 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)

        # Background class (0) metrics for mean calculations
        prec_0 = self.tn / (self.tn + self.fn + eps)
        rec_0 = self.tn / (self.tn + self.fp + eps)
        f1_0 = 2.0 * prec_0 * rec_0 / (prec_0 + rec_0 + eps)
        iou_0 = self.tn / (self.tn + self.fn + self.fp + eps)

        mean_f1 = 0.5 * (f1_score + f1_0)
        mean_iou = 0.5 * (iou + iou_0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "iou": iou,
            "mean_f1": mean_f1,
            "mean_iou": mean_iou,
        }


# ---------------------------------------------------------------------------
# 3. Multi-Scale Deep Supervision Loss
# ---------------------------------------------------------------------------
def multi_scale_loss(
    preds: List[torch.Tensor],
    targets: torch.Tensor,
    weights: Tuple[float, ...] = (0.5, 0.5, 0.5, 0.8, 1.0),
) -> torch.Tensor:
    """
    Computes multi-scale deep supervision cross-entropy loss across ChangeFormer stages:
    preds: list of [ (B, 2, 8, 8), (B, 2, 16, 16), (B, 2, 32, 32), (B, 2, 64, 64), (B, 2, 256, 256) ]
    targets: (B, 256, 256)
    """
    total_loss = torch.tensor(0.0, device=targets.device)
    targets_float = targets.unsqueeze(1).float()  # (B, 1, 256, 256)

    for pred, w in zip(preds, weights):
        h, w_size = pred.shape[2], pred.shape[3]
        if targets.shape[-2:] != (h, w_size):
            downsampled = F.interpolate(
                targets_float, size=(h, w_size), mode="nearest"
            ).squeeze(1).long()
        else:
            downsampled = targets

        stage_loss = F.cross_entropy(pred, downsampled)
        total_loss = total_loss + w * stage_loss

    return total_loss


# ---------------------------------------------------------------------------
# 4. Model Builder with Frozen Encoder and Trainable Decoder Head
# ---------------------------------------------------------------------------
def build_and_freeze_changeformer(
    checkpoint_path: Path,
    embed_dim: int = 256,
    device: torch.device = torch.device("cpu"),
) -> ChangeFormerV6:
    """
    Instantiates ChangeFormerV6, loads pre-trained LEVIR-CD weights,
    freezes the Siamese Vision Transformer encoder (Tenc_x2), and
    enables gradient updates strictly on the MLP decoder head (TDec_x2).
    """
    model = ChangeFormerV6(
        input_nc=3,
        output_nc=2,
        decoder_softmax=False,
        embed_dim=embed_dim,
    )

    if checkpoint_path.exists():
        log.info("Loading pre-trained LEVIR-CD checkpoint from: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_G_state_dict"] if "model_G_state_dict" in ckpt else ckpt
        msg = model.load_state_dict(state_dict, strict=False)
        log.info("Checkpoint loaded successfully: %s", msg)
    else:
        log.warning(
            "Pretrained checkpoint not found at %s. Initializing from scratch.",
            checkpoint_path,
        )

    # ── FREEZE Transformer Encoder Blocks ───────────────────────────────────
    for param in model.Tenc_x2.parameters():
        param.requires_grad = False

    # ── ONLY Fine-tune MLP Decoder Head ─────────────────────────────────────
    for param in model.TDec_x2.parameters():
        param.requires_grad = True

    model.to(device)

    # Parameter accounting
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = frozen_params + trainable_params

    log.info("=" * 68)
    log.info("     Transfer Learning Architecture (ChangeFormerV6)")
    log.info("=" * 68)
    log.info("  Frozen Encoder Parameters (Tenc_x2) : %10s (%5.2f%%)", f"{frozen_params:,}", 100 * frozen_params / total_params)
    log.info("  Trainable Decoder Head (TDec_x2)    : %10s (%5.2f%%)", f"{trainable_params:,}", 100 * trainable_params / total_params)
    log.info("  Total Model Parameters               : %10s", f"{total_params:,}")
    log.info("=" * 68)

    return model


# ---------------------------------------------------------------------------
# 5. Plot Loss Curve Artifact
# ---------------------------------------------------------------------------
def generate_loss_curve(
    history: Dict[str, List[float]],
    save_path: Path,
    artifact_dir: Optional[Path] = None,
) -> Path:
    """
    Generates and saves a two-panel publication-grade loss and metric curve.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    # Left: Training vs Validation Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", lw=2, markersize=5)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", lw=2, markersize=5)
    ax1.set_title("ChangeFormer Coastal Fine-Tuning Loss Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Multi-Scale Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.set_xticks(epochs)

    # Right: Validation F1-Score & IoU
    ax2.plot(epochs, history["val_f1"], "g-^", label="Validation F1-Score (Change)", lw=2, markersize=6)
    ax2.plot(epochs, history["val_iou"], "m--d", label="Validation IoU (Change)", lw=2, markersize=6)
    ax2.set_title("Validation Coastal Change Detection Metrics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score [0.0 - 1.0]", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("✓ Loss curve generated and saved to: %s", save_path)

    # Copy to agent artifact directory if provided
    if artifact_dir and artifact_dir.exists():
        dest = artifact_dir / save_path.name
        shutil.copy2(save_path, dest)
        log.info("✓ Copied loss curve to artifact directory: %s", dest)

    return save_path


# ---------------------------------------------------------------------------
# 6. Main Training & Evaluation Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune ChangeFormer for Coastal Change Detection")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to Sentinel-2 dataset")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Path to pretrained weights")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for MLP decoder head")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--smoke-test", action="store_true", help="Quick run with limited batches per epoch")
    parser.add_argument("--wandb-project", type=str, default="samudratata-changeformer-coastal", help="WandB project")
    parser.add_argument("--wandb-mode", type=str, default=None, help="WandB mode: online, offline, or disabled")
    args = parser.parse_args()

    # Hardware setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine WandB mode
    wandb_mode = args.wandb_mode
    if wandb_mode is None:
        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"

    log.info("Initializing Weights & Biases (mode: %s)...", wandb_mode)
    wandb.init(
        project=args.wandb_project,
        mode=wandb_mode,
        config={
            "architecture": "ChangeFormerV6",
            "pretraining": "LEVIR-CD",
            "frozen_encoder": True,
            "trainable_head": "TDec_x2 (MLP)",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "smoke_test": args.smoke_test,
        },
    )

    # DataLoaders
    log.info("Constructing Coastal Sentinel-2 Datasets...")
    train_dataset = CoastalSentinelDataset(
        data_dir=args.data_dir,
        split="train",
        val_ratio=0.20,
        seed=42,
        augment=True,
    )
    val_dataset = CoastalSentinelDataset(
        data_dir=args.data_dir,
        split="val",
        val_ratio=0.20,
        seed=42,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Build Model & Freeze Encoder
    model = build_and_freeze_changeformer(
        checkpoint_path=args.checkpoint,
        embed_dim=256,
        device=device,
    )

    # Optimizer (only decoder parameters require grad)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # Batch limits for smoke test
    max_train_batches = 6 if args.smoke_test else None
    max_val_batches = 3 if args.smoke_test else None

    # History recording
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
        "val_accuracy": [],
    }

    best_val_f1 = -1.0
    best_model_path = args.output_dir / "changeformer_coastal_best.pt"
    t_start = time.time()

    log.info("Starting fine-tuning for %d epochs...", args.epochs)

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        # ── TRAIN EPOCH ─────────────────────────────────────────────────────
        model.train()
        # Keep encoder in eval mode to preserve batchnorm/layernorm stats
        model.Tenc_x2.eval()

        train_loss_total = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", unit="batch")
        for b_idx, batch in enumerate(pbar):
            if max_train_batches and b_idx >= max_train_batches:
                break

            img_a = batch["A"].to(device)
            img_b = batch["B"].to(device)
            target = batch["L"].to(device)

            optimizer.zero_grad()
            preds = model(img_a, img_b)

            loss = multi_scale_loss(preds, target)
            loss.backward()
            optimizer.step()

            bs = img_a.size(0)
            train_loss_total += loss.item() * bs
            train_count += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_total / max(train_count, 1)

        # ── VALIDATION EPOCH ────────────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        val_count = 0
        metric_meter = CoastalMetricMeter()

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                if max_val_batches and b_idx >= max_val_batches:
                    break

                img_a = batch["A"].to(device)
                img_b = batch["B"].to(device)
                target = batch["L"].to(device)

                preds = model(img_a, img_b)
                loss = multi_scale_loss(preds, target)

                bs = img_a.size(0)
                val_loss_total += loss.item() * bs
                val_count += bs

                # Final resolution prediction: preds[-1] has shape (B, 2, 256, 256)
                pred_binary = torch.argmax(preds[-1], dim=1)
                metric_meter.update(pred_binary, target)

        val_loss = val_loss_total / max(val_count, 1)
        metrics = metric_meter.get_metrics()
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(metrics["f1_score"])
        history["val_iou"].append(metrics["iou"])
        history["val_precision"].append(metrics["precision"])
        history["val_recall"].append(metrics["recall"])
        history["val_accuracy"].append(metrics["accuracy"])

        epoch_duration = time.time() - epoch_t0
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch summary
        log.info(
            "Epoch %02d/%02d (%4.1fs) | Train Loss: %.4f | Val Loss: %.4f | "
            "Val F1: %.4f | Val IoU: %.4f | Val Acc: %.4f | LR: %.2e",
            epoch,
            args.epochs,
            epoch_duration,
            train_loss,
            val_loss,
            metrics["f1_score"],
            metrics["iou"],
            metrics["accuracy"],
            current_lr,
        )

        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/f1_score": metrics["f1_score"],
            "val/iou": metrics["iou"],
            "val/precision": metrics["precision"],
            "val/recall": metrics["recall"],
            "val/accuracy": metrics["accuracy"],
            "val/mean_iou": metrics["mean_iou"],
            "val/mean_f1": metrics["mean_f1"],
            "learning_rate": current_lr,
        })

        # Save Best Checkpoint
        if metrics["f1_score"] > best_val_f1:
            best_val_f1 = metrics["f1_score"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "val_iou": metrics["iou"],
                "val_loss": val_loss,
                "history": history,
            }, best_model_path)
            log.info(" ★ New best model saved: %s (Val F1: %.4f, IoU: %.4f)", best_model_path, best_val_f1, metrics["iou"])

    total_time = time.time() - t_start
    log.info("Completed %d epochs in %.2f seconds.", args.epochs, total_time)

    # ── GENERATE LOSS CURVE ARTIFACT ────────────────────────────────────────
    loss_curve_dest = REPO_ROOT / "loss_curve.png"
    artifact_brain_dir = (
        Path(os.environ["ANTIGRAVITY_ARTIFACT_DIR"])
        if "ANTIGRAVITY_ARTIFACT_DIR" in os.environ
        else None
    )
    generate_loss_curve(history, loss_curve_dest, artifact_brain_dir)
    
    # Also save in data/
    shutil.copy2(loss_curve_dest, args.output_dir / "loss_curve.png")

    # Log artifact to WandB
    curve_artifact = wandb.Artifact("loss_curve", type="evaluation")
    curve_artifact.add_file(str(loss_curve_dest))
    wandb.log_artifact(curve_artifact)
    wandb.log({"Loss Curve": wandb.Image(str(loss_curve_dest))})

    # Save summary metadata JSON
    summary_path = args.output_dir / "changeformer_coastal_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": "ChangeFormerV6",
            "epochs_completed": args.epochs,
            "best_val_f1": best_val_f1,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_iou": history["val_iou"][-1],
            "final_val_accuracy": history["val_accuracy"][-1],
            "history": history,
            "total_time_seconds": total_time,
        }, f, indent=2)

    wandb.finish()
    log.info("✓ Training completed successfully. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
