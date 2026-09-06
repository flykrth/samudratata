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
├── config_setup.py     # Auth verification entry point
├── requirements.txt    # Pinned Python dependencies
├── .env.example        # Credential template (safe to commit)
├── .gitignore          # Excludes .env, *.tif, .venv, etc.
└── data/sentinel2/     # Downloaded imagery (auto-created, git-ignored)
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
| Bands | B2, B3, B4, B8, B11, B12 + SCL |

## Roadmap

- [x] Virtual environment & library installation
- [x] Credential management (`config_setup.py`)
- [x] CDSE & GEE authentication smoke-tests
- [ ] AOI definition (`aoi.geojson`)
- [ ] Download pipeline (`download_s2.py`)
- [ ] Preprocessing (`preprocess.py`)
- [ ] DL dataset assembly (`tile_dataset.py`)

## License

MIT
