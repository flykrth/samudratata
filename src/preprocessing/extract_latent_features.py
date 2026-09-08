#!/usr/bin/env python3
"""
extract_latent_features.py
==========================
Unified Latent Feature Extraction Pipeline.

Loads the trained Dual-Branch Latent Autoencoder (`autoencoder_best.pt`)
and encodes all paired spatial-temporal coastal node observations:
- Spatial Bottleneck:  z_spatial  in R^128 (Denoised Sentinel-2 visual features)
- Temporal Bottleneck: z_temporal in R^64  (Hydrodynamic wave/wind surge dynamics)
- Unified Embedding:   z_unified  in R^192 (Concatenated Multimodal Node Representation)

Outputs:
- data/latent_embeddings.pt: PyTorch dictionary with tensors and sample metadata.
- data/latent_embeddings.h5: Hierarchical HDF5 storage for downstream GNN & Transformer ingestion.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model_mismatch_autoencoder import DualBranchAutoencoder
from src.models.train_mismatch_autoencoder import (
    DEFAULT_OCEAN_H5,
    DEFAULT_SPATIAL_H5,
    MultiModalMismatchDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ExtractLatentFeatures")

DEFAULT_CHECKPOINT = Path("data/autoencoder_best.pt")
OUTPUT_PT = Path("data/latent_embeddings.pt")
OUTPUT_H5 = Path("data/latent_embeddings.h5")


def extract_features(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    spatial_h5: Path = DEFAULT_SPATIAL_H5,
    ocean_h5: Path = DEFAULT_OCEAN_H5,
    output_pt: Path = OUTPUT_PT,
    output_h5: Path = OUTPUT_H5,
    batch_size: int = 32,
    device_str: str = None,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")

    # Determine device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading checkpoint from: %s on device: %s", checkpoint_path, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {
        "spatial_channels": 5,
        "temporal_features": 7,
        "spatial_latent_dim": 128,
        "temporal_latent_dim": 64,
        "unified_latent_dim": 192,
    })

    # Initialize model with saved architecture configuration
    model = DualBranchAutoencoder(
        spatial_channels=config["spatial_channels"],
        temporal_features=config["temporal_features"],
        spatial_latent_dim=config["spatial_latent_dim"],
        temporal_latent_dim=config["temporal_latent_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log.info("Model loaded successfully. Trained up to epoch %d with val loss %.4f",
             checkpoint.get("epoch", -1), checkpoint.get("best_val_loss", float("nan")))

    # Load all paired dataset samples
    dataset = MultiModalMismatchDataset(
        spatial_h5_path=spatial_h5,
        ocean_h5_path=ocean_h5,
        split="all",  # Extract for entire corpus
        augment_noise=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    log.info("Extracting embeddings for %d paired multimodal samples...", len(dataset))

    z_spatial_list: List[torch.Tensor] = []
    z_temporal_list: List[torch.Tensor] = []
    z_unified_list: List[torch.Tensor] = []
    metadata_list: List[Dict[str, Any]] = []

    with torch.no_grad():
        for clean_sp, _, temp_win, meta in tqdm(loader, desc="Encoding Latents"):
            clean_sp = clean_sp.to(device)
            temp_win = temp_win.to(device)

            z_s, z_t, z_u = model.encode(clean_sp, temp_win)

            z_spatial_list.append(z_s.cpu())
            z_temporal_list.append(z_t.cpu())
            z_unified_list.append(z_u.cpu())

            # Collate metadata
            batch_sz = clean_sp.size(0)
            for i in range(batch_sz):
                metadata_list.append({
                    "zone": meta["zone"][i],
                    "pair_key": meta["pair_key"][i],
                    "t_tag": meta["t_tag"][i],
                    "year": int(meta["year"][i]),
                    "patch_idx": int(meta["patch_idx"][i]),
                    "ocean_idx": int(meta["ocean_idx"][i]),
                    "timestamp": meta["timestamp"][i],
                })

    z_spatial_all = torch.cat(z_spatial_list, dim=0)    # (N, 128)
    z_temporal_all = torch.cat(z_temporal_list, dim=0)  # (N, 64)
    z_unified_all = torch.cat(z_unified_list, dim=0)    # (N, 192)

    log.info("Extracted Latent Shapes:")
    log.info("  z_spatial  : %s", tuple(z_spatial_all.shape))
    log.info("  z_temporal : %s", tuple(z_temporal_all.shape))
    log.info("  z_unified  : %s", tuple(z_unified_all.shape))

    # 1. Save to PyTorch binary (.pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    payload_pt = {
        "z_spatial": z_spatial_all,
        "z_temporal": z_temporal_all,
        "z_unified": z_unified_all,
        "metadata": metadata_list,
        "config": config,
    }
    torch.save(payload_pt, output_pt)
    log.info("Saved PyTorch latent embeddings to: %s (%.2f MB)",
             output_pt, output_pt.stat().st_size / (1024 * 1024))

    # 2. Save to HDF5 (.h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as h5f:
        h5f.attrs["num_samples"] = len(metadata_list)
        h5f.attrs["spatial_latent_dim"] = config["spatial_latent_dim"]
        h5f.attrs["temporal_latent_dim"] = config["temporal_latent_dim"]
        h5f.attrs["unified_latent_dim"] = config["unified_latent_dim"]
        h5f.attrs["config"] = json.dumps(config)

        h5f.create_dataset("z_spatial", data=z_spatial_all.numpy(), compression="gzip", compression_opts=4)
        h5f.create_dataset("z_temporal", data=z_temporal_all.numpy(), compression="gzip", compression_opts=4)
        h5f.create_dataset("z_unified", data=z_unified_all.numpy(), compression="gzip", compression_opts=4)

        # Zone-specific hierarchical views
        for idx, item in enumerate(metadata_list):
            z_grp = h5f.require_group(f"zones/{item['zone']}/{item['year']}")
            if "indices" not in z_grp:
                z_grp.create_dataset("indices", data=[idx], maxshape=(None,), dtype="int32")
            else:
                curr = list(z_grp["indices"][:])
                curr.append(idx)
                del z_grp["indices"]
                z_grp.create_dataset("indices", data=curr, dtype="int32")

    log.info("Saved HDF5 latent embeddings to: %s (%.2f MB)",
             output_h5, output_h5.stat().st_size / (1024 * 1024))

    return {
        "num_samples": len(metadata_list),
        "z_spatial_shape": list(z_spatial_all.shape),
        "z_temporal_shape": list(z_temporal_all.shape),
        "z_unified_shape": list(z_unified_all.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract unified latent embeddings")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--spatial-h5", type=Path, default=DEFAULT_SPATIAL_H5)
    parser.add_argument("--ocean-h5", type=Path, default=DEFAULT_OCEAN_H5)
    parser.add_argument("--output-pt", type=Path, default=OUTPUT_PT)
    parser.add_argument("--output-h5", type=Path, default=OUTPUT_H5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    extract_features(
        checkpoint_path=args.checkpoint,
        spatial_h5=args.spatial_h5,
        ocean_h5=args.ocean_h5,
        output_pt=args.output_pt,
        output_h5=args.output_h5,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
