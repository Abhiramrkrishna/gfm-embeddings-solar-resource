"""
Extract AlphaEarth Foundations Satellite Embeddings for each DWD station.

Prerequisites:
    1. Google account with Earth Engine access (free, register at earthengine.google.com)
    2. pip install earthengine-api
    3. earthengine authenticate  (one-time, opens browser)

Strategy:
    For each station (lat, lon) and each year 2020-2024:
      - Pull the AlphaEarth annual embedding image at that point
      - Average over a small buffer (default: 100m, so 10x10 pixels) so we get
        a representative descriptor of the station surroundings rather than a
        single 10m pixel that might happen to be a building roof
      - Cache as numpy arrays locally; no need to re-query EE on every run

Output:
    data/alphaearth/{station_id}_{year}.npy  -- shape (64,) per file
    data/alphaearth/index.csv                -- manifest

The embeddings are stable per year, so this is a one-time extraction.
Total volume: 28 stations x 5 years = 140 small calls. Trivial.
"""
from __future__ import annotations
import time
import csv
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import ee
except ImportError:
    raise SystemExit(
        "earthengine-api not installed.\n"
        "  pip install earthengine-api\n"
        "  earthengine authenticate"
    )

STATIONS_CSV = "dwd_core_stations.csv"
OUT_DIR = Path("data/alphaearth")
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_CSV = OUT_DIR / "index.csv"

EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BUFFER_METERS = 100  # 100m buffer = roughly 10x10 AlphaEarth pixels averaged
YEARS = list(range(2020, 2025))

def initialize_ee(project: str | None = None):
    """Initialize Earth Engine. If you have a Google Cloud project ID, pass it.
    Otherwise relies on default credentials from `earthengine authenticate`."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)

def fetch_embedding(lat: float, lon: float, year: int) -> np.ndarray:
    """Pull a single 64-d embedding vector averaged over a small region."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(BUFFER_METERS)
    # The AlphaEarth annual collection: each image covers one year
    collection = ee.ImageCollection(EMBEDDING_COLLECTION).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    ).filterBounds(point)
    image = collection.mosaic()
    # Compute mean over the buffer region
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10,
        maxPixels=1e7,
        bestEffort=True,
    ).getInfo()
    # Band order: A00, A01, ..., A63
    vec = np.array([stats[f"A{i:02d}"] for i in range(64)], dtype=np.float32)
    return vec

def main(project: str | None = None):
    initialize_ee(project=project)
    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})
    manifest_rows = []
    for _, s in stations.iterrows():
        sid = s["station_id"].zfill(5)
        lat, lon = float(s["lat"]), float(s["lon"])
        for year in YEARS:
            target = OUT_DIR / f"{sid}_{year}.npy"
            if target.exists():
                manifest_rows.append({"station_id": sid, "year": year, "path": str(target), "status": "cached"})
                continue
            try:
                vec = fetch_embedding(lat, lon, year)
                np.save(target, vec)
                manifest_rows.append({"station_id": sid, "year": year, "path": str(target), "status": "ok"})
                print(f"  [{sid}] {year}: ok  range=[{vec.min():.3f}, {vec.max():.3f}]")
            except Exception as e:
                print(f"  [{sid}] {year}: FAILED {e}")
                manifest_rows.append({"station_id": sid, "year": year, "path": "", "status": f"error: {e}"})
            time.sleep(0.5)  # be polite to Earth Engine
    pd.DataFrame(manifest_rows).to_csv(INDEX_CSV, index=False)
    print(f"\nManifest: {INDEX_CSV}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=None, help="Google Cloud project ID for EE quota")
    args = p.parse_args()
    main(project=args.project)
