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

- [x] Virtual environment & library installation
- [x] Credential management (`config_setup.py`)
- [x] CDSE & GEE authentication smoke-tests
- [x] Sentinel-2 data acquisition & processing (`preprocess_changeformer.py`)
- [x] ERA5 oceanographic time-series preprocessing (`preprocess_ocean_timeseries.py`)
- [x] Dual-Branch Latent Autoencoder architecture & SSIM loss (`model_mismatch_autoencoder.py`)
- [x] Multimodal mismatch training pipeline (`train_mismatch_autoencoder.py`)
- [x] Unified node latent extraction (`extract_latent_features.py`)

## License

MIT
