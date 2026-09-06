# samudrataṭa 🛰️

> **Samudra** (समुद्र) = ocean · **Taṭa** (तट) = coast  
> A deep-learning pipeline for Sentinel-2 Level-2A coastal imagery (2019–2024).

## Overview

This repository houses a step-by-step pipeline to:

1. **Authenticate** against either the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu) or [Google Earth Engine (GEE)](https://earthengine.google.com)
2. **Search & download** Sentinel-2 L2A scenes by AOI, date range, and cloud-cover threshold
3. **Preprocess** scenes (band-stacking, cloud masking, normalisation, COG export)
4. **Assemble** a spatially-split DL dataset (256×256 patches, train/val/test)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/flykrth/samudratatata.git
cd samudratatata

# 2. Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env — pick AUTH_BACKEND=cdse or gee and fill in credentials

# 4. Verify authentication (no imagery downloaded)
python config_setup.py
```

## Project Structure

```
samudratatata/
├── config_setup.py                 # Auth verification entry point
├── fetch_sentinel2_data.py         # Sentinel-2 data discovery & acquisition
├── fetch_ocean_timeseries.py       # ERA5 oceanographic time-series acquisition
├── preprocess_changeformer.py      # Sentinel-2 patch extraction & NDWI dataset builder
├── preprocess_ocean_timeseries.py  # Oceanographic cubic-spline interpolation & 72h sliding windows
├── model_mismatch_autoencoder.py   # Dual-Branch Latent Autoencoder & SSIM loss
├── train_mismatch_autoencoder.py   # Multi-modal training & visual reconstruction check
├── extract_latent_features.py      # Unified node embedding extractor (z in R^192)
├── verify_ocean_pipeline.py        # Sanity verification for ocean pipelines
├── requirements.txt                # Pinned Python dependencies
├── .env.example                    # Credential template (safe to commit)
├── .gitignore                      # Excludes .env, *.tif, .venv, etc.
└── data/                           # Processed & raw assets (git-ignored)
```

## Dual-Branch Latent Autoencoder

Reconciles the spatial-temporal mismatch between annual **Sentinel-2 multi-spectral imagery** $(B, 5, 256, 256)$ and hourly **oceanographic time-series** $(B, 72, 7)$:

1. **Spatial Denoising Branch**:
   - 4-stage 2D-CNN with BatchNorm, LeakyReLU, Dropout(0.2)
   - Projects to latent $z_{spatial} \in \mathbb{R}^{128}$ ($2,560\times$ compression)
   - Transposed 2D-CNN reconstructing clean imagery under MSE + SSIM loss
2. **Temporal Signal Branch**:
   - 3-layer 1D-CNN ($k=[7, 5, 3]$) downsampling along time axis
   - Projects to latent $z_{temporal} \in \mathbb{R}^{64}$ ($7.88\times$ compression)
   - Transposed 1D-CNN reconstructing 72h series under Huber Loss (Smooth L1)
3. **Unified Multimodal Latents**:
   - Concatenated node embedding $z = [z_{spatial}; z_{temporal}] \in \mathbb{R}^{192}$ ($1,709.3\times$ overall compression)

### Usage

```bash
# Train Dual-Branch Autoencoder
python train_mismatch_autoencoder.py --epochs 5 --smoke-test

# Extract unified node embeddings (z in R^192)
python extract_latent_features.py
```

## Authentication Backends

| Backend | Notes |
|---------|-------|
| **CDSE** | Free account at dataspace.copernicus.eu · OAuth2 token flow |
| **GEE**  | GCP project required · supports interactive ADC and service-account |

See `.env.example` for all available configuration options.

## Dataset Parameters

| Parameter | Default |
|-----------|---------|
| Date range | 2019-01-01 → 2024-12-31 |
| Max cloud cover | 20 % |
| Product type | Sentinel-2 L2A (Surface Reflectance) |
| Bands | B2, B3, B4, B8 + NDWI |

## Roadmap

## PyTorch Geometric Coastal Graph Dataset

Transforms discretized coastal transects ($100\,\text{m}$ intervals) and multimodal latent representations ($z \in \mathbb{R}^{192}$) into a structured PyG graph dataset:

- **Official Coastline Discretization**: $718$ coastal transects across Chellanam ($115$), Alappuzha ($113$), Nagapattinam ($111$), Cuddalore ($96$), and Visakhapatnam ($283$).
- **Graph Topology**:
  - Linear Chain Edges: Bidirectional edges connecting adjacent segments ($i \leftrightarrow i+1$).
  - Directional Longshore Drift Edges: Asymmetric directed transport edges following seasonal monsoon forcing:
    - **SW Coast (Chellanam & Alappuzha)**: South-to-North drift during SW Monsoon.
    - **SE Coast (Nagapattinam, Cuddalore, Visakhapatnam)**: North-to-South drift during NE Monsoon.
  - Edge Attributes ($E_{\text{attr}} \in \mathbb{R}^{2134 \times 3}$): Euclidean distance ($m$), alongshore bathymetric depth gradient ($\Delta d / \Delta s$), and drift indicator ($\pm 1$).
- **Node Features ($X \in \mathbb{R}^{718 \times 194}$)**:
  - Static GIS properties: Elevation ($z_{\text{topo}}$, meters above MSL) and Baseline Distance to Vegetation Line ($d_{\text{veg}}$, meters, from Sentinel-2 NDVI $> 0.3$).
  - Dynamic Latent Vectors: $z \in \mathbb{R}^{192}$ interpolated from the multimodal autoencoder via spatial inverse distance weighting.

### Usage

```bash
# Build coastal graph dataset (outputs data/south_india_coastal_graph.pt & graph_schema.json)
python build_coastal_graph.py

# Visualize topology overlaid on geographic coordinates
python visualize_graph.py
```

## ChangeFormer Siamese-ViT Coastal Change Detection

Fine-tunes the **ChangeFormerV6** Siamese Vision Transformer architecture for binary coastal change detection (erosion and accretion) using Sentinel-2 bitemporal patch pairs:

- **Transfer Learning**: Pre-trained on LEVIR-CD building change detection.
- **Selective Fine-Tuning**: Initial Siamese transformer encoder (`Tenc_x2`: 28.9M parameters) is frozen, fine-tuning only the multi-scale MLP decoder head (`TDec_x2`: 12.1M parameters).
- **Coastal Land/Water Ground Truth**: Binary coastal change masks derived from multi-spectral NDWI transitions ($W_1 \neq W_2$).
- **Experiment Tracking**: Full validation logging (F1-Score, IoU, Precision, Recall, Accuracy) with Weights & Biases (WandB) and automatic generation of publication-ready `loss_curve.png`.

### Usage

```bash
# Fine-tune ChangeFormerV6 for 10 epochs
python train_changeformer_coastal.py --epochs 10 --smoke-test
```

## Roadmap

- [x] Virtual environment & library installation
- [x] Credential management (`config_setup.py`)
- [x] CDSE & GEE authentication smoke-tests
- [x] Sentinel-2 data acquisition & processing (`preprocess_changeformer.py`)
- [x] ERA5 oceanographic time-series preprocessing (`preprocess_ocean_timeseries.py`)
- [x] Dual-Branch Latent Autoencoder architecture & SSIM loss (`model_mismatch_autoencoder.py`)
- [x] Multimodal mismatch training pipeline (`train_mismatch_autoencoder.py`)
- [x] Unified node latent extraction (`extract_latent_features.py`)
- [x] PyG Coastal Graph generation with longshore drift topology (`build_coastal_graph.py`)
- [x] Geographic graph visualization suite (`visualize_graph.py`)
- [x] ChangeFormer Siamese-ViT coastal fine-tuning & WandB tracking (`train_changeformer_coastal.py`)

## License

MIT
