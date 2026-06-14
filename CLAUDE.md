# Project context: AlphaEarth × Solar Resource

## TL;DR for any Claude session

I'm Abhiram, an M.Eng. Mechatronics graduate based in Stuttgart working on this independently as a hook project to land a PhD or research position before my visa runs out in ~7 months. I'm a competent ML engineer but new to macOS dev workflows. Be direct, push back when I'm wrong, don't pad responses.

## Research question

**Do AlphaEarth Foundations embeddings encode physically meaningful information about the local solar resource?**

Specifically:
1. Do they predict residual error in NWP-based irradiance estimates after clear-sky normalisation?
2. Can they improve site-specific irradiance estimation in low-data regimes (few-shot transfer to unseen stations)?
3. Which of the 64 embedding dimensions correlate with measurable physical quantities (clear-sky index variance, persistence, terrain ruggedness, shading frequency)?

The bet: AlphaEarth is trained on Sentinel-2/1, Landsat, GEDI canopy height, ERA5-Land, DEM, etc. — all of which encode terrain, land cover, microclimate. If the 64-d embedding preserves this information usably, it offers a free, globally-consistent site descriptor for any solar application. Nobody has tested this yet for solar; everyone tests it on agriculture/hydrology/land-cover.

## Why this question and not something bigger

- Multimodal foundation models for energy systems (Baguan-solar, SolarFM, SPIRIT, MM-VSF, AlphaEarth) are already crowded; can't compete with DeepMind/Alibaba scale
- Per-site PV generation in Germany is NOT openly available (data protection)
- DWD pyranometer network IS openly available and is high-quality ground truth
- So the cleanest answerable question is "does AlphaEarth contain solar signal at all?", probed by the best ground truth Germany offers
- This is also a natural extension of my published Solar Energy 2026 thesis work (physically-conserving super-resolution of Meteosat irradiation) — so co-author Dr. Garrett H. Good (former IEE supervisor) is a credible fit, not a courtesy.

## Constraints

- **Visa: ~7 months remaining.** This project is a hook to land a PhD/RA position. Must produce demoable results in ~8 weeks, not 12 months.
- **Solo, no funding, single laptop (M5 MacBook).** Can rent a GPU later if needed but design for laptop development.
- **AlphaEarth backbone is FROZEN.** We are not training a foundation model. We extract embeddings and train small heads on top.
- **Honesty over hype.** Negative results go in the paper. Calibrated authors get trusted.

## Collaborators

- **Dr. Garrett H. Good** (`ghg36@cornell.edu`) — former IEE supervisor, co-author on Solar Energy 2026 paper, fluid dynamics PhD, strong on cloud physics + forecasting + radiation imagery. Now between jobs, willing to collaborate. Will be approached AFTER first base results are in hand (not before).
- **Mike Zehner** (TH Rosenheim) — runs HELIOS-AI (sky-imager-based PV forecasting). Email sent, awaiting reply. Friend (his thesis student) can intro if needed. Goal: research-assistant offer OR introduction to his industry contacts. Approached AFTER Garrett is on board and we have results.

## Dataset choices (locked)

- **Time window:** 2020–2024 (5 years, balances AlphaEarth coverage 2017–2025 with station availability)
- **Primary ground truth:** DWD hourly solar radiation, 29 core stations covering full window
  - Source: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/solar/
  - License: CC-BY 4.0
  - Variable: FG_LBERG (hourly sum GHI, J/cm²; convert × 10000/3600 → W/m² mean)
  - Quality: pyranometer + ScaPP, QN_592 quality codes; missing = -999
- **Static input:** AlphaEarth Foundations Satellite Embedding (GEE: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`)
  - 64-d, 10m, annual 2017–2025
  - Extracted per station, averaged over 100m buffer
- **Time-varying input (later weeks):** DWD ICON-D2 NWP forecasts (radiation outputs)
- **Secondary eval (spatial demo, later weeks):** PVGIS-SARAH2 hourly across Germany grid for visual map
- **Clear-sky baseline:** pvlib Ineichen-Perez model

## Evaluation protocol (locked)

- **Primary metric space:** clear-sky index `kt_cs = GHI_observed / GHI_clearsky`, daytime only (clear-sky > 5 W/m²)
- **Cross-validation:** 7-fold leave-station-out across 29 stations, report mean ± std
- **Kill-the-project-early test:** embedding-shuffle ablation in Week 4. If shuffled embeddings perform as well as real ones, the signal isn't there → pivot.

## Baselines (locked, do not change later)

1. Persistence (yesterday's kt_cs = today's)
2. NWP raw (ICON-D2 GHI)
3. NWP + per-station bias correction (constant offset learned from training)
4. NWP + simple site metadata (lat, lon, elevation, slope, aspect from DEM) via small MLP
5. DWD DUETT 1km gridded product (operational satellite+station fusion — the strongest baseline)

## Project structure

```
solar-alphaearth/
├── .venv/                          # Python 3.12 virtualenv (Apple Silicon)
├── CLAUDE.md                       # this file
├── README_week1.md                 # week 1 plan
├── requirements.txt
├── parse_stations.py               # builds dwd_core_stations.csv from DWD metadata
├── dwd_core_stations.csv           # 29 stations covering 2020-2024
├── dwd_relaxed_stations.csv        # 34 stations (relaxed criteria, backup pool)
├── scripts/
│   ├── 01_download_dwd.py          # pull DWD hourly solar archives
│   ├── 02_add_clear_sky.py         # pvlib clear-sky model + kt_cs
│   ├── 03_extract_alphaearth.py    # GEE embedding extraction (requires auth)
│   └── 04_sanity_plots.py          # week 1 validation figures
└── data/                           # gitignored
    ├── raw_dwd/{station_id}/       # downloaded ZIPs
    ├── stations/                   # parsed parquet per station
    ├── stations_enriched/          # + clear-sky + kt_cs
    ├── alphaearth/                 # 64-d embeddings as .npy
    ├── figs/                       # sanity-check plots
    ├── coverage_report.csv
    └── enrichment_report.csv
```

## Environment

- macOS Apple Silicon (M5)
- Python 3.12.7 via pyenv
- venv in `.venv/`, always activate before running
- Earth Engine: authenticated, registered Google Cloud project for noncommercial use
- VS Code with Python + Jupyter extensions
- Shell: zsh

## Current status (as of last update)

- Setup complete, Earth Engine authenticated
- `scripts/01_download_dwd.py` downloads ZIPs for all 29 stations and parses them correctly
- Parsing fixed: MESS_DATUM format is `%Y%m%d%H:%M` (e.g. `1981010100:09`), floored to hour; `skipinitialspace=True` handles leading whitespace
- All 29 station parquets written to `data/stations/`; `data/coverage_report.csv` is up to date
- **Next: run `scripts/02_add_clear_sky.py`** to add pvlib clear-sky + kt_cs
- Two stations to watch: 04642 (Seehausen, 71.5% FG valid) and 07365 (Bochum, 94.5%, data ends 2024-11)

## Working style

- I prefer concise, direct responses. No padding.
- Push back when I'm wrong. Don't just agree.
- I write code daily but I'm new to macOS shell and the M5 environment. Be explicit about Mac-specific gotchas.
- I'm fine with one-line fixes; I'm also fine reading 100 lines of new code if it's well-commented.
- When debugging, FIRST inspect (view file, check data), don't guess.
- For data files: always check actual format before assuming. DWD uses `;` separators, possibly German locale, sometimes header whitespace.
- Never reformulate the science to make a result look better. Negative results are valuable.

## Out of scope right now

- PV-specific forecasting (deferred to potential Phase 2 after we know AlphaEarth carries solar signal)
- Sky imager integration (Zehner-specific extension, only if collaboration materialises)
- Training a foundation model from scratch (we don't have the compute or the need)
- Anything that takes >8 weeks before producing a demoable result

## Week 2 status

- Layer 2 v2 complete. Headline finding: residual_kt_mean MAE — embedding (Ridge) 0.0125 vs geo 0.0182 vs shuffle 0.0169. 26% gain over shuffle on detrended target.
- Conclusion: AlphaEarth carries information beyond (lat, lon, elev) for mean clear-sky index. residual_over_irr_frac shows suggestive but underpowered gap.
- Methodological note for the paper: Ridge with GCV alpha selection outperforms PLS-8 and gradient-boosting trees for n=28 × d=64 regressions. PLS is over-parameterized at n_components=8 in this regime.
- Layer 3 (hourly residual prediction) is the next decisive test.


