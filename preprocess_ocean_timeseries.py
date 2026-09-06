#!/usr/bin/env python3
"""
preprocess_ocean_timeseries.py
──────────────────────────────
Signal Preprocessing & Noise Removal Pipeline (Unit 2 DL Syllabus)

1. Missing timestamp verification & bidirectional cubic spline interpolation.
2. Derivation of hydrodynamic & aerodynamic features:
   - Wind Magnitude: W_mag = sqrt(u10^2 + v10^2)
   - Wave Power Index: P ≈ 0.5 * H_s^2 * T_p
3. MinMax and Z-score normalization with persistent scaler storage.
4. Hourly sampled 72-hour sliding window tensor generation:
   - Shape: (52537, 72, 7) per coastal zone.
5. Structured HDF5 serialization: data/ocean/ocean_timeseries_72h.h5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.interpolate import CubicSpline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("OceanPreprocess")

FEATURE_COLS = ["swh", "pp1d", "u10", "v10", "sp", "wind_magnitude", "wave_power_index"]
RAW_COLS = ["swh", "pp1d", "u10", "v10", "sp"]

ZONES_META = {
    "Chellanam": {
        "lat": 9.830,
        "lon": 76.195,
        "basin": "Arabian_Sea",
        "state": "Kerala",
        "bbox": [76.155, 9.780, 76.235, 9.880],
    },
    "Alappuzha": {
        "lat": 9.510,
        "lon": 76.350,
        "basin": "Arabian_Sea",
        "state": "Kerala",
        "bbox": [76.295, 9.460, 76.405, 9.560],
    },
    "Nagapattinam": {
        "lat": 10.800,
        "lon": 79.860,
        "basin": "Bay_of_Bengal",
        "state": "Tamil_Nadu",
        "bbox": [79.820, 10.750, 79.900, 10.850],
    },
    "Cuddalore": {
        "lat": 11.760,
        "lon": 79.780,
        "basin": "Bay_of_Bengal",
        "state": "Tamil_Nadu",
        "bbox": [79.740, 11.720, 79.820, 11.800],
    },
    "Visakhapatnam": {
        "lat": 17.700,
        "lon": 83.275,
        "basin": "Bay_of_Bengal",
        "state": "Andhra_Pradesh",
        "bbox": [83.200, 17.650, 83.350, 17.750],
    },
}


def interpolate_cubic_spline(series: pd.Series) -> pd.Series:
    """
    Perform bidirectional cubic spline interpolation over missing timestamps/sensor dropouts.
    Preserves dynamic curvature, peak wave energies, and tidal components without phase distortion.
    """
    values = series.values.copy()
    n = len(values)
    valid_mask = ~np.isnan(values)
    missing_count = n - np.sum(valid_mask)

    if missing_count == 0:
        return series

    x_indices = np.arange(n)
    valid_x = x_indices[valid_mask]
    valid_y = values[valid_mask]

    # Fit natural cubic spline (zero second derivative at boundaries)
    cs = CubicSpline(valid_x, valid_y, bc_type="natural")
    interpolated = cs(x_indices)

    # Physical safety bounds to prevent negative heights/periods
    if series.name == "swh":
        interpolated = np.clip(interpolated, 0.05, 20.0)
    elif series.name == "pp1d":
        interpolated = np.clip(interpolated, 1.0, 30.0)
    elif series.name == "sp":
        interpolated = np.clip(interpolated, 90000.0, 105000.0)

    return pd.Series(interpolated, index=series.index, name=series.name)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive hydrodynamic and aerodynamic features:
      1. Wind Magnitude: W_mag = sqrt(u10^2 + v10^2)
      2. Wave Power Index: P ≈ 0.5 * H_s^2 * T_p
    """
    df["wind_magnitude"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2).astype(np.float32)
    df["wave_power_index"] = (0.5 * (df["swh"] ** 2) * df["pp1d"]).astype(np.float32)
    return df


def generate_sliding_windows(
    arr: np.ndarray, window_size: int = 72, step: int = 1
) -> np.ndarray:
    """
    Generate sliding windows of shape (N, window_size, num_features).
    Uses numpy sliding_window_view for optimal memory efficiency.
    """
    # arr: (T, D)
    # sliding_window_view on 2D axis 0 gives (num_windows, D, window_size)
    windows = sliding_window_view(arr, window_shape=window_size, axis=0)
    # Transpose to standard deep learning shape: (num_windows, window_size, D)
    windows = np.ascontiguousarray(np.swapaxes(windows, 1, 2)[::step])
    return windows


def run_preprocessing(
    raw_dir: Path = Path("data/ocean/raw"),
    output_h5: Path = Path("data/ocean/ocean_timeseries_72h.h5"),
    output_scalers: Path = Path("data/ocean/scalers.json"),
    window_size: int = 72,
    step: int = 1,
):
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    scalers_dict = {}

    log.info("Opening HDF5 output file: %s", output_h5)
    with h5py.File(output_h5, "w") as h5f:
        # Global attributes
        h5f.attrs["project"] = "South Indian Coastal Oceanographic DL Pipeline"
        h5f.attrs["date_range"] = "2019-01-01T00:00:00 to 2024-12-31T23:00:00"
        h5f.attrs["lookback_hours"] = window_size
        h5f.attrs["sampling_rate"] = f"{step}h"
        h5f.attrs["feature_columns"] = json.dumps(FEATURE_COLS)

        for zone_name, meta in ZONES_META.items():
            raw_csv = raw_dir / f"{zone_name}_raw_2019_2024.csv"
            if not raw_csv.is_file():
                log.error("Raw CSV not found for zone %s: %s", zone_name, raw_csv)
                continue

            log.info("─── Processing zone: %s (%s) ───", zone_name, meta["basin"])
            df = pd.read_csv(raw_csv, parse_dates=["timestamp"])

            # 1. Enforce strict continuous 1-hour temporal grid
            expected_index = pd.date_range("2019-01-01 00:00:00", "2024-12-31 23:00:00", freq="1h")
            df = df.set_index("timestamp").reindex(expected_index)
            df.index.name = "timestamp"

            # Check raw missing values
            raw_missing = df[RAW_COLS].isna().sum().to_dict()
            total_missing = sum(raw_missing.values())
            log.info(
                "Sensor dropouts detected for %s: total NaNs=%d (%s)",
                zone_name,
                total_missing,
                raw_missing,
            )

            # 2. Bidirectional cubic spline interpolation
            for col in RAW_COLS:
                df[col] = interpolate_cubic_spline(df[col])

            post_missing = df[RAW_COLS].isna().sum().sum()
            assert post_missing == 0, f"Interpolation incomplete for {zone_name}, remaining NaNs: {post_missing}"
            log.info("Bidirectional cubic spline interpolation completed: 0 NaNs remaining.")

            # 3. Derived Physical Features
            df = compute_derived_features(df)

            # Clean float32 matrix
            feature_matrix = df[FEATURE_COLS].values.astype(np.float32)
            n_timesteps, n_features = feature_matrix.shape

            # 4. Scalers computation (Z-score and MinMax)
            mean_vals = np.mean(feature_matrix, axis=0)
            std_vals = np.std(feature_matrix, axis=0)
            min_vals = np.min(feature_matrix, axis=0)
            max_vals = np.max(feature_matrix, axis=0)

            # Avoid division by zero
            std_vals_safe = np.where(std_vals < 1e-6, 1.0, std_vals)
            range_vals_safe = np.where((max_vals - min_vals) < 1e-6, 1.0, (max_vals - min_vals))

            zscore_matrix = (feature_matrix - mean_vals) / std_vals_safe
            minmax_matrix = (feature_matrix - min_vals) / range_vals_safe

            scalers_dict[zone_name] = {
                "features": FEATURE_COLS,
                "zscore": {
                    "mean": mean_vals.tolist(),
                    "std": std_vals.tolist(),
                },
                "minmax": {
                    "min": min_vals.tolist(),
                    "max": max_vals.tolist(),
                },
            }

            # 5. Sliding window generation
            log.info("Generating 72-hour sliding window sequences (step=%d)...", step)
            windows_zscore = generate_sliding_windows(zscore_matrix, window_size=window_size, step=step)
            windows_minmax = generate_sliding_windows(minmax_matrix, window_size=window_size, step=step)
            windows_raw = generate_sliding_windows(feature_matrix, window_size=window_size, step=step)

            n_samples = windows_zscore.shape[0]
            log.info(
                "Zone %s tensor shape: %s (Total sequences: %d, Memory: %.2f MB)",
                zone_name,
                windows_zscore.shape,
                n_samples,
                windows_zscore.nbytes / (1024 * 1024),
            )

            # Generate target timestamps (end of each 72h window)
            window_end_timestamps = df.index[window_size - 1 :: step].strftime("%Y-%m-%dT%H:%M:%SZ").values.astype("S20")

            # 6. Save to HDF5 Group
            grp = h5f.create_group(zone_name)
            grp.attrs["basin"] = meta["basin"]
            grp.attrs["state"] = meta["state"]
            grp.attrs["lat_center"] = meta["lat"]
            grp.attrs["lon_center"] = meta["lon"]
            grp.attrs["bbox"] = meta["bbox"]
            grp.attrs["total_hours"] = n_timesteps
            grp.attrs["window_count"] = n_samples
            grp.attrs["zscore_mean"] = mean_vals
            grp.attrs["zscore_std"] = std_vals
            grp.attrs["minmax_min"] = min_vals
            grp.attrs["minmax_max"] = max_vals

            # Create datasets with chunked gzip compression
            grp.create_dataset(
                "X_zscore",
                data=windows_zscore,
                compression="gzip",
                compression_opts=4,
                chunks=(min(1024, n_samples), window_size, n_features),
            )
            grp.create_dataset(
                "X_minmax",
                data=windows_minmax,
                compression="gzip",
                compression_opts=4,
                chunks=(min(1024, n_samples), window_size, n_features),
            )
            grp.create_dataset(
                "X_raw",
                data=windows_raw,
                compression="gzip",
                compression_opts=4,
                chunks=(min(1024, n_samples), window_size, n_features),
            )
            grp.create_dataset(
                "timestamps_end",
                data=window_end_timestamps,
                compression="gzip",
            )
            # Save unwindowed continuous time series for reference
            grp.create_dataset(
                "series_continuous_raw",
                data=feature_matrix,
                compression="gzip",
            )

    # Save scaler parameters to JSON for external inference serving
    with open(output_scalers, "w", encoding="utf-8") as f:
        json.dump(scalers_dict, f, indent=2)
    log.info("Saved normalization parameters to: %s", output_scalers)
    log.info("HDF5 dataset packaging successfully finalized: %s", output_h5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Oceanographic Time-Series")
    parser.add_argument("--lookback", type=int, default=72, help="Lookback window length in hours")
    parser.add_argument("--step", type=int, default=1, help="Sampling step in hours")
    args = parser.parse_args()
    run_preprocessing(window_size=args.lookback, step=args.step)
