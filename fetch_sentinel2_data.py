#!/usr/bin/env python3
"""
fetch_sentinel2_data.py
───────────────────────
Sentinel-2 Level-2A coastal-zone downloader (2019-2024).

Usage
─────
  # Count images only (no downloads, no auth needed beyond ee.Initialize):
  python fetch_sentinel2_data.py --dry-run

  # Full download → data/sentinel2/<Zone>/<Year>/<file>.tif
  python fetch_sentinel2_data.py --live

Dataset
───────
  Collection : COPERNICUS/S2_SR_HARMONIZED
  Bands      : B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
  Resolution : 10 m
  Cloud max  : 20 %
  Composite  : Annual median per zone
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import zipfile
import io
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import requests

# ─── Load .env ────────────────────────────────────────────────────────────────
_dotenv = find_dotenv(usecwd=True)
if _dotenv:
    load_dotenv(_dotenv, override=False)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Project constants ────────────────────────────────────────────────────────
BANDS           = ["B2", "B3", "B4", "B8"]
BAND_LABELS     = {"B2": "Blue-10m", "B3": "Green-10m", "B4": "Red-10m", "B8": "NIR-10m"}
SCALE           = 10          # metres (native S2 res for B2-B4, B8)
MAX_CLOUD       = int(os.getenv("MAX_CLOUD_COVER", "20"))
DATE_START      = os.getenv("DATE_START",  "2019-01-01")
DATE_END        = os.getenv("DATE_END",    "2024-12-31")
DATA_DIR        = Path(os.getenv("DATA_DIR", "./data/sentinel2"))
GEE_PROJECT     = os.getenv("GEE_PROJECT", "")
GEE_SA          = os.getenv("GEE_SERVICE_ACCOUNT", "")
GEE_KEY         = os.getenv("GEE_KEY_FILE", "")

YEARS = list(range(int(DATE_START[:4]), int(DATE_END[:4]) + 1))

# ─── Coastal zone bounding boxes  [west, south, east, north]  ────────────────
ZONES: dict[str, dict] = {
    "Chellanam": {
        "bbox": [76.155, 9.780, 76.235, 9.880],
        "state": "Kerala",
        "description": "Low-lying fishing village on Arabian Sea coast",
    },
    "Alappuzha": {
        "bbox": [76.295, 9.460, 76.405, 9.560],
        "state": "Kerala",
        "description": "Alappuzha (Alleppey) backwaters & coast",
    },
    "Nagapattinam": {
        "bbox": [79.820, 10.750, 79.900, 10.850],
        "state": "Tamil Nadu",
        "description": "Nagapattinam Bay of Bengal coast",
    },
    "Cuddalore": {
        "bbox": [79.740, 11.720, 79.820, 11.800],
        "state": "Tamil Nadu",
        "description": "Cuddalore port and coastal plain",
    },
    "Visakhapatnam": {
        "bbox": [83.200, 17.650, 83.350, 17.750],
        "state": "Andhra Pradesh",
        "description": "Visakhapatnam Bay of Bengal coast",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────

def authenticate_gee() -> bool:
    """Initialise Earth Engine; return True on success."""
    try:
        import ee  # noqa: PLC0415
    except ImportError:
        log.error("earthengine-api not installed. Run: pip install earthengine-api")
        return False

    try:
        if GEE_SA and GEE_KEY:
            creds = ee.ServiceAccountCredentials(GEE_SA, GEE_KEY)
            ee.Initialize(credentials=creds, project=GEE_PROJECT or None)
            log.info("GEE ✓  service-account flow  (SA=%s)", GEE_SA)
        elif GEE_PROJECT:
            ee.Initialize(project=GEE_PROJECT)
            log.info("GEE ✓  ADC / cached-credentials flow  (project=%s)", GEE_PROJECT)
        else:
            ee.Initialize()
            log.info("GEE ✓  default credentials")
        # Quick smoke-test
        _ = ee.Number(42).getInfo()
        return True
    except Exception as exc:
        log.error("GEE authentication failed: %s", exc)
        log.error(
            "Fix:\n"
            "  1. Run `earthengine authenticate` once in your terminal.\n"
            "  2. Set GEE_PROJECT=<your-gcp-project> in .env.\n"
            "  3. Re-run this script."
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Core query + download helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_collection(ee, zone_name: str, bbox: list[float], year: int):
    """Return a filtered, band-selected ImageCollection for one zone/year."""
    geometry = ee.Geometry.Rectangle(bbox)
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
        .select(BANDS)
    )
    return col, geometry


def download_tif(url: str, dest: Path, retries: int = 3) -> bool:
    """Stream a GEE download URL (possibly a ZIP containing a TIFF) to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            total = int(resp.headers.get("content-length", 0))

            # Read all bytes (needed to detect ZIP vs raw TIFF)
            buf = io.BytesIO()
            with tqdm(
                desc=dest.name,
                total=total if total else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=65536):
                    buf.write(chunk)
                    bar.update(len(chunk))

            raw = buf.getvalue()

            # GEE sometimes wraps the GeoTIFF in a ZIP
            if raw[:2] == b"PK":          # ZIP magic bytes
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
                    if not tif_names:
                        log.error("ZIP had no .tif inside for %s", dest.name)
                        return False
                    tif_data = zf.read(tif_names[0])
                dest.write_bytes(tif_data)
            else:
                dest.write_bytes(raw)

            return True

        except Exception as exc:
            log.warning("  Attempt %d/%d — %s: %s", attempt, retries, dest.name, exc)
            if attempt < retries:
                time.sleep(5 * attempt)

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-zone-year processor
# ─────────────────────────────────────────────────────────────────────────────

def process_zone_year(ee, zone_name: str, zone: dict, year: int, dry_run: bool) -> dict:
    result = {
        "zone": zone_name,
        "year": year,
        "n_images": 0,
        "status": "pending",
        "path": None,
        "size_kb": 0,
    }

    col, geometry = build_collection(ee, zone_name, zone["bbox"], year)

    try:
        n = col.size().getInfo()
    except Exception as exc:
        result["status"] = "query_error"
        log.error("  Query failed %s/%d: %s", zone_name, year, exc)
        return result

    result["n_images"] = n
    log.info("  %-16s %d  → %3d scenes (cloud < %d%%)", zone_name, year, n, MAX_CLOUD)

    if n == 0:
        result["status"] = "no_images"
        return result

    if dry_run:
        result["status"] = "dry_run_ok"
        return result

    # ── Build annual median composite ─────────────────────────────────────────
    composite = col.median().clip(geometry)

    filename = f"{zone_name.lower()}_{year}_S2_B2B3B4B8.tif"
    dest = DATA_DIR / zone_name / str(year) / filename

    # ── Get download URL ──────────────────────────────────────────────────────
    try:
        url = composite.getDownloadURL({
            "bands":       BANDS,
            "region":      geometry,
            "scale":       SCALE,
            "format":      "GEO_TIFF",
            "filePerBand": False,
        })
    except Exception as exc:
        result["status"] = "url_error"
        log.error("  getDownloadURL failed %s/%d: %s", zone_name, year, exc)
        return result

    # ── Download ──────────────────────────────────────────────────────────────
    log.info("  Downloading → %s", dest)
    ok = download_tif(url, dest)
    if ok:
        size_kb = dest.stat().st_size // 1024
        result["status"]  = "success"
        result["path"]    = str(dest.relative_to(DATA_DIR.parent))
        result["size_kb"] = size_kb
        log.info("  ✓ %s  (%d KB)", dest.relative_to(Path.cwd()), size_kb)
    else:
        result["status"] = "download_error"
        log.error("  ✗ Download failed: %s / %d", zone_name, year)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], dry_run: bool) -> None:
    mode = "DRY-RUN (image counts only)" if dry_run else "LIVE DOWNLOAD"
    ok   = [r for r in results if r["status"] in ("success", "dry_run_ok")]
    noi  = [r for r in results if r["status"] == "no_images"]
    err  = [r for r in results if r["status"] not in ("success", "dry_run_ok", "no_images")]
    total_kb = sum(r.get("size_kb", 0) for r in results)

    print()
    print("═" * 78)
    print(f"  Sentinel-2 Coastal Downloader — {mode}")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("═" * 78)
    print(f"  {'Zone':<18} {'Year':>6} {'Scenes':>8}  {'Status':<20} {'KB':>8}")
    print("  " + "─" * 65)
    for r in sorted(results, key=lambda x: (x["zone"], x["year"])):
        if   r["status"] == "success":    icon = "✓"
        elif r["status"] == "dry_run_ok": icon = "◉"
        elif r["status"] == "no_images":  icon = "○"
        else:                              icon = "✗"
        print(
            f"  {r['zone']:<18} {r['year']:>6} {r['n_images']:>8}  "
            f"{icon} {r['status']:<18} {r.get('size_kb', 0):>8}"
        )
    print("  " + "─" * 65)
    print(
        f"  Succeeded: {len(ok)}/{len(results)}   "
        f"No images: {len(noi)}   "
        f"Errors: {len(err)}   "
        + (f"Total size: {total_kb:,} KB" if not dry_run else "")
    )
    print("═" * 78)

    if dry_run and ok:
        print()
        print("  ◉ = images available. Re-run with --live to download all TIFFs.")
        print()


def print_tree(data_dir: Path) -> None:
    """Print directory tree for downloaded files."""
    tifs = sorted(data_dir.rglob("*.tif"))
    if not tifs:
        return
    print("\n  Downloaded files:")
    cur_zone, cur_year = None, None
    for tif in tifs:
        parts = tif.relative_to(data_dir).parts  # (Zone, Year, filename)
        zone, year, fname = parts[0], parts[1], parts[2]
        if zone != cur_zone:
            print(f"    data/sentinel2/{zone}/")
            cur_zone, cur_year = zone, None
        if year != cur_year:
            print(f"      {year}/")
            cur_year = year
        size_kb = tif.stat().st_size // 1024
        print(f"        {fname}  ({size_kb:,} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentinel-2 coastal zone downloader"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Query image counts only — no files downloaded",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Download multi-band GeoTIFFs to data/sentinel2/<Zone>/<Year>/",
    )
    args = parser.parse_args()
    dry_run = args.dry_run

    print()
    print("═" * 78)
    print("  fetch_sentinel2_data.py — Sentinel-2 L2A Coastal Zone Processor")
    print("═" * 78)
    print(f"  Mode        : {'DRY-RUN (count only)' if dry_run else 'LIVE DOWNLOAD'}")
    print(f"  Zones       : {', '.join(ZONES)}")
    print(f"  Years       : {YEARS[0]} – {YEARS[-1]}  ({len(YEARS)} years)")
    print(f"  Bands       : {BANDS}  (Blue, Green, Red, NIR @ 10m)")
    print(f"  Cloud filter: < {MAX_CLOUD}%  CLOUDY_PIXEL_PERCENTAGE")
    print(f"  Collection  : COPERNICUS/S2_SR_HARMONIZED")
    print(f"  Composite   : Annual median per zone")
    print(f"  Output root : {DATA_DIR.resolve()}")
    print("═" * 78)
    print()

    # ── Auth ──────────────────────────────────────────────────────────────────
    if not authenticate_gee():
        sys.exit(1)

    import ee  # noqa: PLC0415

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    total = len(ZONES) * len(YEARS)
    log.info("Starting: %d zone × year combinations...", total)
    print()

    for zone_name, zone in ZONES.items():
        print(f"── {zone_name} ({zone['state']})  |  {zone['description']}")
        for year in YEARS:
            res = process_zone_year(ee, zone_name, zone, year, dry_run)
            results.append(res)
            time.sleep(0.5)   # gentle rate-limiting
        print()

    print_summary(results, dry_run)

    if not dry_run:
        print_tree(DATA_DIR)

    # ── Save JSON manifest ─────────────────────────────────────────────────────
    manifest_path = DATA_DIR / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "mode": "dry_run" if dry_run else "live",
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": BANDS,
        "scale_m": SCALE,
        "max_cloud_pct": MAX_CLOUD,
        "date_range": [DATE_START, DATE_END],
        "zones": {k: {"bbox": v["bbox"], "state": v["state"]} for k, v in ZONES.items()},
        "results": results,
    }
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Manifest saved → %s", manifest_path)


if __name__ == "__main__":
    main()
