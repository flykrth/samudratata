#!/usr/bin/env python3
"""
preprocess_changeformer.py
==========================
Deep-Learning Data Architect pipeline for ChangeFormer (Siamese-ViT) training.

Converts Sentinel-2 annual-median GeoTIFFs into structured bitemporal patch
datasets suitable for PyTorch and Vision Transformer architectures.

Pipeline Stages
---------------
1. Discovery   -- iterates through downloaded folders (data/sentinel2/<Zone>/<Year>/)
                  and cross-checks download_manifest.json.
2. NDWI        -- computes (Green - NIR) / (Green + NIR), stacks as channel 5 -> (5, H, W).
3. Normalise   -- bands 0-3 clamped [0, 10000] -> [0, 1]; NDWI clipped [-1, 1].
4. Pair        -- builds temporal pairs (T1, T2) for each bounding box.
5. Patch       -- 256x256 sliding window, stride=128 (50% overlap), >=80% valid pixels.
6. Save        -- per-pair .pt tensors + consolidated dataset.h5 + metadata JSON.
7. Dataset     -- ChangeFormerDataset PyTorch Dataset class for seamless DataLoader ingestion.

Output Tensor Shapes (per sample)
----------------------------------
T1 : (5, 256, 256)  float32  -- [B2, B3, B4, B8, NDWI] at time T1
T2 : (5, 256, 256)  float32  -- same channel layout at time T2
Batch for Siamese-ViT: (B, 5, 256, 256) for each branch
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Default configuration constants
BASE_DIR      = Path(__file__).resolve().parent
SENTINEL_DIR  = BASE_DIR / "data" / "sentinel2"
MANIFEST_PATH = SENTINEL_DIR / "download_manifest.json"
OUTPUT_DIR    = BASE_DIR / "data" / "changeformer_dataset"

PATCH_SIZE    = 256
STRIDE        = 128
VALID_FRAC    = 0.80
BAND_CLIP_MAX = 10_000.0
EPS           = 1e-6

# Sentinel-2 band indices in downloaded GeoTIFF:
# Band 0: B2 (Blue, 490 nm)
# Band 1: B3 (Green, 560 nm)
# Band 2: B4 (Red, 665 nm)
# Band 3: B8 (NIR, 842 nm)
IDX_BLUE  = 0
IDX_GREEN = 1
IDX_RED   = 2
IDX_NIR   = 3

CHANNEL_NAMES = ["B2_blue", "B3_green", "B4_red", "B8_nir", "NDWI"]


# ---------------------------------------------------------------------------
# 1. Discovery: Iterate downloaded folders & cross-reference manifest
# ---------------------------------------------------------------------------
def discover_data(
    sentinel_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[int, Path]], Dict[str, Any]]:
    """
    Discovers all Sentinel-2 TIFFs by scanning the directory tree
    <sentinel_dir>/<Zone>/<Year>/*.tif, supplemented with metadata from manifest.
    """
    index: Dict[str, Dict[int, Path]] = {}
    zone_metadata: Dict[str, Any] = {}

    # Load manifest metadata if available
    if manifest_path and manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
                zone_metadata = manifest_data.get("zones", {})
        except Exception as e:
            print(f"  [Warning] Could not read manifest: {e}")

    # Walk directory structure directly to ensure all downloaded files are found
    for zone_path in sorted(sentinel_dir.iterdir()):
        if not zone_path.is_dir() or zone_path.name.startswith("."):
            continue
        zone_name = zone_path.name

        for year_path in sorted(zone_path.iterdir()):
            if not year_path.is_dir() or year_path.name.startswith("."):
                continue
            try:
                year = int(year_path.name)
            except ValueError:
                continue

            tifs = sorted(year_path.glob("*.tif"))
            if tifs:
                index.setdefault(zone_name, {})[year] = tifs[0].resolve()

    total_tifs = sum(len(v) for v in index.values())
    print(f"\n[Discovery] Discovered {total_tifs} GeoTIFFs across {len(index)} zones:")
    for zone, year_map in sorted(index.items()):
        years = sorted(year_map.keys())
        print(f"  - {zone:<18} ({len(years)} years): {years}")

    return index, zone_metadata


# ---------------------------------------------------------------------------
# 2. NDWI Computation & Raster Reading
# ---------------------------------------------------------------------------
def read_tif_with_ndwi(path: Path) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Reads 4-band GeoTIFF, calculates NDWI = (Green - NIR) / (Green + NIR),
    and concatenates along channel dimension -> shape (5, H, W).
    """
    with rasterio.open(path) as src:
        data   = src.read().astype(np.float32)  # (4, H, W)
        nodata = src.nodata
        bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
        crs    = str(src.crs)
        transform = list(src.transform)[:6]

    if nodata is not None:
        data[data == nodata] = np.nan
    data[data == 0] = np.nan

    valid_ratio = np.isfinite(data).mean()
    if valid_ratio < 0.01:
        return None

    green = data[IDX_GREEN]  # B3
    nir   = data[IDX_NIR]    # B8

    # McFeeters (1996) NDWI: (Green - NIR) / (Green + NIR)
    ndwi = (green - nir) / (green + nir + EPS)

    # 5-channel composite: (5, H, W)
    composite = np.concatenate([data, ndwi[np.newaxis, ...]], axis=0)

    meta = {
        "bounds": bounds,
        "crs": crs,
        "transform": transform,
        "raw_shape": list(data.shape),
    }
    return composite, meta


# ---------------------------------------------------------------------------
# 3. Normalization
# ---------------------------------------------------------------------------
def normalise(arr: np.ndarray) -> torch.Tensor:
    """
    Normalizes multi-spectral bands [0, 10000] -> [0.0, 1.0] and clips NDWI [-1.0, 1.0].
    Fills NaNs with 0.0 for stable Transformer attention.
    """
    t = torch.from_numpy(arr.copy())
    # Bands 0-3: Sentinel-2 Surface Reflectance (scaled by 10000)
    t[:4] = torch.clamp(t[:4], 0.0, BAND_CLIP_MAX) / BAND_CLIP_MAX
    # Band 4: NDWI index in [-1.0, 1.0]
    t[4]  = torch.clamp(t[4], -1.0, 1.0)
    # Replace NaNs / Infs with 0.0
    t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    return t.to(torch.float32)


# ---------------------------------------------------------------------------
# 4. Temporal Pairing
# ---------------------------------------------------------------------------
def build_pairs(index: Dict[str, Dict[int, Path]]) -> List[Dict[str, Any]]:
    """
    Pairs consecutive annual median images (T1, T2) for each coastal zone.
    """
    pairs = []
    for zone, year_map in sorted(index.items()):
        years = sorted(year_map.keys())
        for t1_yr, t2_yr in zip(years[:-1], years[1:]):
            pairs.append({
                "zone":    zone,
                "t1_year": t1_yr,
                "t2_year": t2_yr,
                "t1_path": year_map[t1_yr],
                "t2_path": year_map[t2_yr],
            })

    print(f"\n[Pairing] Created {len(pairs)} temporal pairs:")
    for p in pairs:
        print(f"  {p['zone']:<18}  {p['t1_year']} -> {p['t2_year']}")
    return pairs


# ---------------------------------------------------------------------------
# 5. Spatial Alignment & Overlapping Patch Extraction
# ---------------------------------------------------------------------------
def align_tensors(t1: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensures spatial dimensions match between T1 and T2 via bilinear interpolation if needed."""
    if t1.shape[-2:] == t2.shape[-2:]:
        return t1, t2
    h, w = t1.shape[-2], t1.shape[-1]
    t2 = F.interpolate(
        t2.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)
    return t1, t2


def extract_patches(
    t1: torch.Tensor,
    t2: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    valid_frac: float = VALID_FRAC,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts overlapping (patch_size x patch_size) patches using a sliding window.
    Filters out patches with < valid_frac non-zero pixels in either T1 or T2.

    Returns:
        patches_t1: (N_valid, 5, patch_size, patch_size)
        patches_t2: (N_valid, 5, patch_size, patch_size)
        coords:     (N_valid, 2) [row_offset, col_offset]
    """
    C, H, W = t1.shape
    ps, s   = patch_size, stride

    pad_h = (s - (H - ps) % s) % s
    pad_w = (s - (W - ps) % s) % s
    if pad_h > 0 or pad_w > 0:
        t1 = F.pad(t1, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        t2 = F.pad(t2, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    H_pad, W_pad = t1.shape[-2], t1.shape[-1]
    n_h = (H_pad - ps) // s + 1
    n_w = (W_pad - ps) // s + 1

    # Unfold along H and W
    def unfold_with_coords(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = t.unfold(1, ps, s).unfold(2, ps, s)  # (C, n_h, n_w, ps, ps)
        patches = patches.contiguous().view(C, n_h * n_w, ps, ps)
        patches = patches.permute(1, 0, 2, 3)  # (N_grid, C, ps, ps)
        
        # Grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, n_h * s, s),
            torch.arange(0, n_w * s, s),
            indexing="ij"
        )
        coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
        return patches, coords

    raw_t1, coords = unfold_with_coords(t1)
    raw_t2, _      = unfold_with_coords(t2)

    # Valid mask: at least valid_frac pixels must be non-zero across spectral channels
    def compute_valid_mask(patches: torch.Tensor) -> torch.Tensor:
        any_nonzero = (patches[:, :4] != 0.0).any(dim=1)  # (N, ps, ps)
        frac = any_nonzero.float().mean(dim=(-1, -2))     # (N,)
        return frac >= valid_frac

    mask = compute_valid_mask(raw_t1) & compute_valid_mask(raw_t2)
    return raw_t1[mask], raw_t2[mask], coords[mask]


# ---------------------------------------------------------------------------
# 6. Structured Saving: .pt and .h5
# ---------------------------------------------------------------------------
def save_pt(
    patches_t1: torch.Tensor,
    patches_t2: torch.Tensor,
    coords: torch.Tensor,
    meta: Dict[str, Any],
    out_dir: Path,
) -> Path:
    """Saves per-pair PyTorch binary file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"pair_{meta['t1_year']}_{meta['t2_year']}.pt"
    out_path = out_dir / fname
    torch.save({
        "T1": patches_t1,
        "T2": patches_t2,
        "coords": coords,
        "meta": meta,
    }, out_path)
    return out_path


def save_to_hdf5(
    h5_file: h5py.File,
    patches_t1: torch.Tensor,
    patches_t2: torch.Tensor,
    coords: torch.Tensor,
    meta: Dict[str, Any],
):
    """Saves patches and metadata into hierarchical HDF5 structure."""
    zone     = meta["zone"]
    pair_key = f"pair_{meta['t1_year']}_{meta['t2_year']}"
    group    = h5_file.require_group(f"{zone}/{pair_key}")

    for key, tensor in [("T1", patches_t1), ("T2", patches_t2)]:
        arr = tensor.numpy()
        if key in group:
            del group[key]
        ds = group.create_dataset(
            key, data=arr, dtype="float32",
            compression="gzip", compression_opts=4,
            chunks=(min(16, arr.shape[0]), 5, PATCH_SIZE, PATCH_SIZE)
        )
        ds.attrs["shape"]         = str(arr.shape)
        ds.attrs["channel_names"] = ", ".join(CHANNEL_NAMES)

    if "coords" in group:
        del group["coords"]
    group.create_dataset("coords", data=coords.numpy(), dtype="int32")
    group.attrs["meta"] = json.dumps({k: str(v) for k, v in meta.items()})


# ---------------------------------------------------------------------------
# 7. Verification & Sanity Checks
# ---------------------------------------------------------------------------
def verify_pair(t1: torch.Tensor, t2: torch.Tensor, label: str):
    """Rigorous assertion of tensor properties."""
    assert t1.dtype == torch.float32,  f"[{label}] T1 dtype must be float32"
    assert t2.dtype == torch.float32,  f"[{label}] T2 dtype must be float32"
    assert t1.ndim == 4,               f"[{label}] T1 must be 4-D (N, C, H, W)"
    assert t2.shape == t1.shape,       f"[{label}] Shape mismatch: T1={t1.shape} vs T2={t2.shape}"
    assert t1.shape[1] == 5,          f"[{label}] Expected 5 channels, got {t1.shape[1]}"
    assert t1.shape[2] == PATCH_SIZE, f"[{label}] Patch H must be {PATCH_SIZE}"
    assert t1.shape[3] == PATCH_SIZE, f"[{label}] Patch W must be {PATCH_SIZE}"
    assert t1.shape[0] > 0,           f"[{label}] Zero valid patches extracted"

    # Value ranges
    for i, name in enumerate(CHANNEL_NAMES[:4]):
        lo, hi = t1[:, i].min().item(), t1[:, i].max().item()
        assert -1e-5 <= lo and hi <= 1.0 + 1e-5, \
            f"[{label}] Band {name} out of [0, 1]: [{lo:.4f}, {hi:.4f}]"

    ndwi_lo, ndwi_hi = t1[:, 4].min().item(), t1[:, 4].max().item()
    assert -1.0 - 1e-5 <= ndwi_lo and ndwi_hi <= 1.0 + 1e-5, \
        f"[{label}] NDWI out of [-1, 1]: [{ndwi_lo:.4f}, {ndwi_hi:.4f}]"

    assert not torch.isnan(t1).any(), f"[{label}] NaNs detected in T1"
    assert not torch.isnan(t2).any(), f"[{label}] NaNs detected in T2"


# ---------------------------------------------------------------------------
# 8. PyTorch ChangeFormerDataset Class
# ---------------------------------------------------------------------------
class ChangeFormerDataset(Dataset):
    """
    PyTorch Dataset for feeding bitemporal patches to ChangeFormer.

    Args:
        data_source: Path to consolidated dataset.h5 OR folder containing .pt files.
        transform: Optional augmentation/transform callable applied to (T1, T2).
    """
    def __init__(self, data_source: Path, transform=None):
        self.data_source = Path(data_source)
        self.transform   = transform
        self.samples     = []
        self.h5_file     = None

        if self.data_source.is_file() and self.data_source.suffix == ".h5":
            self.use_h5 = True
            with h5py.File(self.data_source, "r") as h5:
                for zone in sorted(h5.keys()):
                    for pair_key in sorted(h5[zone].keys()):
                        n_patches = h5[zone][pair_key]["T1"].shape[0]
                        for idx in range(n_patches):
                            self.samples.append((zone, pair_key, idx))
        else:
            self.use_h5 = False
            self.pt_pairs = []
            pt_files = sorted(self.data_source.rglob("pair_*.pt"))
            for pt_p in pt_files:
                loaded = torch.load(pt_p, weights_only=False)
                t1 = loaded["T1"]
                t2 = loaded["T2"]
                meta = loaded["meta"]
                for idx in range(t1.shape[0]):
                    self.samples.append((t1[idx], t2[idx], meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.use_h5:
            if self.h5_file is None:
                self.h5_file = h5py.File(self.data_source, "r")
            zone, pair_key, patch_idx = self.samples[idx]
            grp = self.h5_file[zone][pair_key]
            t1 = torch.from_numpy(grp["T1"][patch_idx])
            t2 = torch.from_numpy(grp["T2"][patch_idx])
            meta = json.loads(grp.attrs.get("meta", "{}"))
        else:
            t1, t2, meta = self.samples[idx]

        if self.transform:
            t1, t2 = self.transform(t1, t2)

        return t1, t2, meta


# ---------------------------------------------------------------------------
# 9. Main Pipeline Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess Sentinel-2 data for ChangeFormer")
    parser.add_argument("--data-dir", type=Path, default=SENTINEL_DIR, help="Input directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE, help="Patch size (pixels)")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Sliding window stride")
    parser.add_argument("--valid-frac", type=float, default=VALID_FRAC, help="Minimum valid pixel ratio")
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 68)
    print("      ChangeFormer Preprocessing Pipeline (Siamese-ViT)")
    print("=" * 68)
    print(f"  Input Directory   : {args.data_dir}")
    print(f"  Output Directory  : {args.output_dir}")
    print(f"  Patch Size        : {args.patch_size} x {args.patch_size}")
    print(f"  Stride (Overlap)  : {args.stride} ({(1 - args.stride/args.patch_size)*100:.0f}% overlap)")
    print(f"  Valid Ratio Thresh: {args.valid_frac * 100:.0f}%")
    print(f"  Channels (5)      : {', '.join(CHANNEL_NAMES)}")

    # Stage 1: Discovery
    print("\n[Stage 1] Discovering TIFFs ...")
    index, zone_geo_meta = discover_data(args.data_dir, MANIFEST_PATH)
    if not index:
        sys.exit(f"ERROR: No valid GeoTIFFs found in {args.data_dir}")

    # Stage 4: Pairing
    print("\n[Stage 4] Building temporal pairs ...")
    pairs = build_pairs(index)
    if not pairs:
        sys.exit("ERROR: No valid temporal pairs could be constructed.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    h5_path      = args.output_dir / "dataset.h5"
    meta_all     = []
    summary_rows = []

    # Stages 2, 3, 5, 6: Process Pairs
    print("\n[Stages 2, 3, 5, 6] Processing and Patching Image Pairs ...")
    with h5py.File(h5_path, "w") as h5:
        for pair in tqdm(pairs, desc="Processing Pairs", unit="pair"):
            zone    = pair["zone"]
            t1_year = pair["t1_year"]
            t2_year = pair["t2_year"]
            label   = f"{zone} ({t1_year}->{t2_year})"

            res_t1 = read_tif_with_ndwi(pair["t1_path"])
            res_t2 = read_tif_with_ndwi(pair["t2_path"])

            if res_t1 is None or res_t2 is None:
                tqdm.write(f"  [SKIP] {label}: Invalid or empty raster")
                continue

            arr_t1, meta_t1 = res_t1
            arr_t2, meta_t2 = res_t2

            # Normalise
            t1 = normalise(arr_t1)
            t2 = normalise(arr_t2)
            t1, t2 = align_tensors(t1, t2)

            # Patch extraction
            patches_t1, patches_t2, coords = extract_patches(
                t1, t2,
                patch_size=args.patch_size,
                stride=args.stride,
                valid_frac=args.valid_frac,
            )

            if patches_t1.shape[0] == 0:
                tqdm.write(f"  [SKIP] {label}: No patches satisfied >= {args.valid_frac*100:.0f}% validity")
                continue

            # Verify integrity
            verify_pair(patches_t1, patches_t2, label)

            # Save per-pair .pt
            zone_out = args.output_dir / zone
            meta = {
                "zone":         zone,
                "t1_year":      t1_year,
                "t2_year":      t2_year,
                "t1_path":      str(pair["t1_path"]),
                "t2_path":      str(pair["t2_path"]),
                "patch_count":  int(patches_t1.shape[0]),
                "patch_size":   args.patch_size,
                "stride":       args.stride,
                "channels":     CHANNEL_NAMES,
                "tensor_shape": list(patches_t1.shape),
                "crs":          meta_t1["crs"],
                "bounds":       meta_t1["bounds"],
                "zone_geo":     zone_geo_meta.get(zone, {}),
            }
            pt_path = save_pt(patches_t1, patches_t2, coords, meta, zone_out)

            # Save to consolidated HDF5
            save_to_hdf5(h5, patches_t1, patches_t2, coords, meta)

            meta_all.append(meta)
            summary_rows.append((zone, t1_year, t2_year, patches_t1.shape[0], list(patches_t1.shape)))
            tqdm.write(f"  ✓ {label:<28} -> {patches_t1.shape[0]:3d} patches  shape: {list(patches_t1.shape)}")

    # Write global dataset manifest
    meta_path = args.output_dir / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "patch_size":    args.patch_size,
            "stride":        args.stride,
            "valid_frac":    args.valid_frac,
            "channels":      CHANNEL_NAMES,
            "total_pairs":   len(meta_all),
            "total_patches": sum(m["patch_count"] for m in meta_all),
            "pairs":         meta_all,
        }, f, indent=2)

    elapsed = time.time() - t_start

    # Summary table
    print("\n" + "=" * 68)
    print("                    DATASET GENERATION SUMMARY")
    print("=" * 68)
    print(f"  {'Zone':<18} {'T1':>4}  {'T2':>4}  {'Patches':>8}  {'Output Tensor Shape':<24}")
    print(f"  {'-'*18} {'-'*4}  {'-'*4}  {'-'*8}  {'-'*24}")
    for zone, t1y, t2y, n, shape in summary_rows:
        print(f"  {zone:<18} {t1y:>4}  {t2y:>4}  {n:>8}  {str(shape):<24}")
    total_patches = sum(r[3] for r in summary_rows)
    print("=" * 68)
    print(f"  Total Valid Pairs Generated : {len(summary_rows)}")
    print(f"  Total Paired 256x256 Patches: {total_patches:,}")
    print(f"  Execution Time              : {elapsed:.1f} seconds")
    print(f"\n  Output Artifacts:")
    print(f"    - Consolidated HDF5       : {h5_path}")
    print(f"    - Global Dataset Metadata : {meta_path}")
    print(f"    - Per-Pair PyTorch Binaries: {args.output_dir}/<Zone>/pair_<T1>_<T2>.pt")
    print("=" * 68)

    # Verification: PyTorch DataLoader test
    print("\n[Verification] Testing PyTorch DataLoader with ChangeFormerDataset ...")
    h5_dataset = ChangeFormerDataset(h5_path)
    loader     = DataLoader(h5_dataset, batch_size=8, shuffle=True)
    batch_t1, batch_t2, batch_meta = next(iter(loader))

    print(f"  Dataset Total Samples : {len(h5_dataset)}")
    print(f"  Mini-Batch Size       : {batch_t1.shape[0]}")
    print(f"  Batch T1 Tensor Shape : {tuple(batch_t1.shape)} (dtype: {batch_t1.dtype})")
    print(f"  Batch T2 Tensor Shape : {tuple(batch_t2.shape)} (dtype: {batch_t2.dtype})")
    print(f"  T1 Value Range        : min={batch_t1.min():.4f}, max={batch_t1.max():.4f}")
    print(f"  T2 Value Range        : min={batch_t2.min():.4f}, max={batch_t2.max():.4f}")
    print("\n✓ Pipeline verification successfully completed!")


if __name__ == "__main__":
    main()
