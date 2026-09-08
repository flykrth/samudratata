# Dataset Card — South India Coastal Graph Dataset

**Filename:** `data/south_india_coastal_graph.pt`  
**Format:** `torch_geometric.data.Data` (PyTorch Geometric)  
**Size:** ~640 KB  
**Collection Date Range:** 2019-01-01 → 2024-12-31  
**Version:** 1.0.0  
**License:** MIT  

---

## Overview

The **South India Coastal Graph Dataset** is a PyTorch Geometric heterogeneous coastal graph
encoding 718 discrete 100-metre transect nodes across five high-vulnerability coastal zones
in peninsular India. Each node aggregates static GIS properties and dynamic multimodal latent
representations learned from six years (2019–2024) of Sentinel-2 multispectral imagery and
ERA5/INCOIS hourly oceanographic time-series.

This dataset is the primary input to the **Coastal Digital Twin GConvLSTM** model
(`src/models/train_digital_twin_gnn.py`) which outputs a per-transect Coastal Vulnerability
Score $V \in [0, 1]$.

---

## Study Zones

| Zone | Coast | Sea | State | Nodes | Edges | Drift Direction | Mean Elevation (m) |
|---|---|---|---|---|---|---|---|
| Chellanam | SW | Arabian Sea | Kerala | 115 | 341 | South → North (SW Monsoon) | 1.60 |
| Alappuzha | SW | Arabian Sea | Kerala | 113 | 335 | South → North (SW Monsoon) | 2.20 |
| Nagapattinam | SE | Bay of Bengal | Tamil Nadu | 111 | 329 | North → South (NE Monsoon) | 2.00 |
| Cuddalore | SE | Bay of Bengal | Tamil Nadu | 96 | 284 | North → South (NE Monsoon) | 2.40 |
| Visakhapatnam | SE | Bay of Bengal | Andhra Pradesh | 283 | 845 | North → South (NE Monsoon) | 4.51 |
| **Total** | — | — | — | **718** | **2,134** | — | — |

---

## Node Features — $X \in \mathbb{R}^{718 \times 194}$

Each of the 718 nodes (coastal transects, 100m spacing) carries a 194-dimensional feature vector:

### Static GIS Properties (dims 0–1)

| Index | Feature | Unit | Description |
|---|---|---|---|
| 0 | `elevation_m` | metres above MSL | Coastal elevation from SRTM DEM, resampled to transect midpoint |
| 1 | `baseline_dist_veg_m` | metres | Euclidean distance from transect to the nearest pixel with NDVI > 0.3 (Sentinel-2 derived permanent vegetation baseline), computed via Euclidean distance transform |

### Dynamic Latent Features (dims 2–193)

The remaining 192 dimensions are the **unified multimodal node embedding** $z \in \mathbb{R}^{192}$
produced by the Dual-Branch Latent Autoencoder and spatially interpolated (Inverse Distance
Weighting) to each transect midpoint.

| Dim range | Sub-embedding | Source | Architecture |
|---|---|---|---|
| 2–129 | $z_{\text{spatial}} \in \mathbb{R}^{128}$ | Sentinel-2 annual-median patches (B2, B3, B4, B8, NDWI) | 4-stage 2D-CNN encoder + bottleneck (2,560× compression) |
| 130–193 | $z_{\text{temporal}} \in \mathbb{R}^{64}$ | ERA5 oceanographic 72h windows (SWH, PP1D, U10, V10, SP, W_mag, WPI) | 3-layer 1D-CNN encoder (7.88× compression) |

---

## Edge Connectivity — $E \in \mathbb{R}^{2134 \times 3}$

### Topology

The graph uses **two complementary edge types** constructed deterministically from official
coastline geometry:

**1. Bidirectional Linear Chain Edges**  
Connect adjacent transect nodes ($i \leftrightarrow i+1$) within each zone, forming a
continuous coastal chain at 100m resolution. These model near-field sediment diffusion
and wave propagation continuity.

**2. Directional Longshore Drift Edges**  
Directed edges encoding seasonal monsoon-driven longshore sediment transport:
- **SW Coast** (Chellanam, Alappuzha): South → North during SW Monsoon (Jun–Sep)
- **SE Coast** (Nagapattinam, Cuddalore, Visakhapatnam): North → South during NE Monsoon (Oct–Dec)

### Edge Attributes — $E_{\text{attr}} \in \mathbb{R}^{2134 \times 3}$

| Index | Attribute | Unit | Description |
|---|---|---|---|
| 0 | `euclidean_distance_m` | metres | Geodetic distance between connected transect midpoints (WGS84 ellipsoid, via `pyproj.Geod`) |
| 1 | `bathymetric_depth_gradient` | m/m | Alongshore bathymetric depth gradient $\Delta d / \Delta s$ between connected nodes (calibrated from CESS/ICMAM zone profiles) |
| 2 | `seasonal_drift_direction_indicator` | ±1 | +1 = drift edge aligned with dominant longshore transport, −1 = opposing edge, 0 = chain edge |

### Edge Statistics

| Metric | Value |
|---|---|
| Total edges | 2,134 |
| Graph sparsity | 99.59% |
| Min inter-transect distance | 91.3 m |
| Max inter-transect distance | 201.7 m |
| Mean inter-transect distance | 133.0 m |
| Mean total node degree | 5.94 |
| Min / max node degree | 3 / 6 |

---

## Data Collection

### Sentinel-2 Imagery
- **Collection:** Copernicus Data Space Ecosystem (CDSE) API / Google Earth Engine
- **Product:** Sentinel-2 L2A Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`)
- **Bands used:** B2 (Blue, 490nm), B3 (Green, 560nm), B4 (Red, 665nm), B8 (NIR, 842nm) + derived NDWI
- **Spatial resolution:** 10 m native → 256×256 patch tiles
- **Temporal coverage:** Annual median composites, 2019–2024 (6 composites per zone)
- **Cloud cover filter:** ≤20%
- **Zones:** 5 coastal zones × 6 years = 30 annual scenes

### Oceanographic Time-Series
- **Source:** ERA5 reanalysis (ECMWF Copernicus Climate Data Store via `cdsapi`)
- **Variables:** Significant wave height (SWH), peak wave period (PP1D), 10m wind components (U10, V10), surface pressure (SP)
- **Derived:** Wind magnitude $W_{\text{mag}} = \sqrt{u10^2 + v10^2}$, Wave power index $P \approx 0.5 H_s^2 T_p$
- **Temporal coverage:** Hourly, 2019-01-01 → 2024-12-31 (~52,560 timesteps per zone)
- **Sliding window:** 72-hour windows → tensors of shape (52537, 72, 7)
- **Preprocessing:** Bidirectional cubic-spline interpolation for missing gaps; MinMax + Z-score normalisation

### Coastline Geometry
- **Source:** `data/coastline/south_india_coastline.geojson`
- **Discretisation:** Official coastline GeoJSON → 100m transects via `shapely` + `pyproj.Geod` geodetic interpolation
- **Coordinate system:** WGS84 (EPSG:4326)
- **Zone calibration profiles:** Based on published CESS, ICMAM, and coastal geomorphology literature for peninsular India

---

## PyG Data Object Schema

```python
import torch
from torch_geometric.data import Data

graph = torch.load("data/south_india_coastal_graph.pt", weights_only=False)

print(graph)
# Data(x=[718, 194], edge_index=[2, 2134], edge_attr=[2134, 3],
#      pos=[718, 2], zone_ids=[718], zone_names=[5])

# Node feature matrix
graph.x          # shape (718, 194) — float32
graph.pos        # shape (718, 2)   — [lon, lat] in decimal degrees
graph.zone_ids   # shape (718,)     — int64, zone index 0–4

# Edge connectivity (COO format)
graph.edge_index  # shape (2, 2134)  — int64
graph.edge_attr   # shape (2134, 3)  — float32
```

---

## Zone Node Offsets

| Zone | Start idx | End idx |
|---|---|---|
| Chellanam | 0 | 114 |
| Alappuzha | 115 | 227 |
| Nagapattinam | 228 | 338 |
| Cuddalore | 339 | 434 |
| Visakhapatnam | 435 | 717 |

---

## Reproducibility

To reproduce this dataset from scratch:

```bash
# 1. Fetch Sentinel-2 imagery (requires CDSE or GEE credentials in .env)
python src/preprocessing/fetch_sentinel2_data.py --live

# 2. Fetch ERA5 oceanographic time-series (requires CDS API key in .env)
python src/preprocessing/fetch_ocean_timeseries.py

# 3. Preprocess imagery → changeformer patches
python src/preprocessing/preprocess_changeformer.py

# 4. Preprocess ocean time-series → 72h sliding windows
python src/preprocessing/preprocess_ocean_timeseries.py

# 5. Train Dual-Branch Autoencoder and extract latent embeddings
python src/models/train_mismatch_autoencoder.py --epochs 5
python src/preprocessing/extract_latent_features.py

# 6. Build the coastal graph
python src/preprocessing/build_coastal_graph.py
# → writes data/south_india_coastal_graph.pt and data/graph_schema.json
```

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{samudratata2024,
  title        = {Samudrata\d{t}a: South India Coastal Graph Dataset},
  author       = {flykrth},
  year         = {2024},
  howpublished = {\url{https://github.com/flykrth/samudratata}},
  note         = {PyTorch Geometric coastal graph, 718 transects, 5 zones, 2019--2024}
}
```

---

## License

MIT — see [LICENSE](../LICENSE) for full terms.
