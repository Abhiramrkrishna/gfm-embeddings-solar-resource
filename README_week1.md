# Week 1: data pipeline

Goal of this week: prove the data flows end-to-end across all 28 core stations.
By Friday we want:
1. DWD hourly solar observations downloaded and parsed
2. Clear-sky GHI computed; clear-sky index `kt_cs` derived
3. AlphaEarth embeddings extracted for each (station, year) pair
4. Sanity-check plots that show: kt_cs is in a physically reasonable range,
   embeddings differ visibly across geographically distinct stations, and PCA
   of embeddings shows latitude/elevation structure

If any of those break, fix before writing model code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy pyarrow matplotlib pvlib scikit-learn requests earthengine-api
```

For AlphaEarth you'll also need:
1. A Google account
2. Sign up at https://earthengine.google.com (free for research, ~24h approval)
3. `earthengine authenticate` (one-time, opens browser)
4. Optional: a Google Cloud project ID for clean quota accounting

## Pipeline

```bash
# 1. Build the station list (already done; produces dwd_core_stations.csv)
python parse_stations.py

# 2. Download DWD hourly solar archives, parse to parquet
python scripts/01_download_dwd.py

# 3. Add clear-sky GHI + clear-sky index
python scripts/02_add_clear_sky.py

# 4. Extract AlphaEarth embeddings (requires EE auth)
python scripts/03_extract_alphaearth.py

# 5. Sanity-check plots
python scripts/04_sanity_plots.py
```

## What "Week 1 done" looks like

- `data/coverage_report.csv` shows >85% valid daytime hours for at least 22 of the 28 stations
- `data/enrichment_report.csv` shows `kt_cs` mean per station between roughly 0.55 and 0.80; standard deviation between 0.15 and 0.35
- `data/alphaearth/index.csv` shows `status=ok` for all 28x5=140 extractions
- `data/figs/embeddings_2023.png` shows visibly different embedding signatures across stations (especially Zugspitze 2956m vs. Bremen 4m vs. Norderney island)
- `data/figs/embedding_pca_2023.png` shows some visible coloured structure — coastal vs. inland, lowland vs. mountain — in the PCA projection

If the PCA plot is just a random cloud with no visible structure, that's an
early warning that the embeddings may not be capturing what we hope. Worth
discussing with Garrett before doing the model work.

## Likely gotchas

- DWD ZIP archives sometimes split into `historical/` and `recent/` folders;
  stations need both for the full 2020-2024 window. The download script tries
  both folders automatically.
- DWD timestamps are in UTC for the hourly product. Don't double-shift.
- The `J/cm^2` to `W/m^2` conversion is `J/cm^2 * 10000 / 3600`. Easy to get
  wrong; sanity-check is whether kt_cs ends up roughly 0.6-0.8 on average.
- Earth Engine quotas: free tier is generous for 140 small `reduceRegion`
  calls. If you hit a quota, attach a billing project (no charge for research).
- `pvlib`'s `linke_turbidity=None` means it auto-loads a global climatology
  from the package data; should work offline once installed.

## What this builds toward

Week 2: turn the per-station parquets into a clean training-ready dataset
(`pv_alphaearth/dataset.py`) with leave-station-out splits.

Week 3+: model code, baselines, etc.

For Week 1 you do NOT need a GPU. Laptop is fine.
