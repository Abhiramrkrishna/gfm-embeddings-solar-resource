"""
Extract AlphaEarth embeddings for the relaxed-pool ADDED stations only
(the core 28 already have them in data/alphaearth/).

Reuses fetch_embedding / initialize_ee from scripts/03 so extraction is
identical to the core run (100 m buffer mean, 10 m scale, years 2020-2024).

Requires the Earth Engine Google Cloud project id:
    python scripts/10_extract_added_embeddings.py --project YOUR_EE_PROJECT
"""
from __future__ import annotations
import argparse
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR    = Path("data/alphaearth")
POOLED_CSV = Path("spatial_holdout/pooled_stations.csv")
YEARS      = list(range(2020, 2025))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main(project):
    ae = _load("scripts/03_extract_alphaearth.py", "ae")
    ae.initialize_ee(project=project)

    pooled = pd.read_csv(POOLED_CSV, dtype={"station_id": str})
    added = pooled[~pooled["in_core"]].copy()
    print(f"Extracting embeddings for {len(added)} added stations × "
          f"{len(YEARS)} years...\n")

    ok = miss = 0
    for _, r in added.iterrows():
        sid = r["station_id"].zfill(5)
        for year in YEARS:
            tgt = OUT_DIR / f"{sid}_{year}.npy"
            if tgt.exists():
                continue
            try:
                vec = ae.fetch_embedding(float(r["lat"]), float(r["lon"]), year)
                np.save(tgt, vec)
                ok += 1
                print(f"  [{sid}] {year}: ok  range=[{vec.min():.3f},{vec.max():.3f}]")
            except Exception as e:
                miss += 1
                print(f"  [{sid}] {year}: FAILED {str(e)[:120]}")
            time.sleep(0.5)
    print(f"\nDone. {ok} extracted, {miss} failed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Earth Engine GCP project id")
    main(p.parse_args().project)
