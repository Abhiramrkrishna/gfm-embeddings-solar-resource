# Do geospatial-foundation-model embeddings encode solar resource beyond geographic coordinates?

A pre-registered pilot with a built-in replication test, a pre-committed kill
criterion, and adversarial from-raw verification of every reported number.
Methodology is the focus: the negative result is held to the same evidential
standard a positive result would require. The outcome: at pilot scale, AlphaEarth
embeddings add little to latitude, longitude, and elevation for solar-resource
climatology, and this finding survives independent re-derivation.

**Status:** verified repository and optional technical note. Not submitted to, or
under review at, any journal or venue.

Geospatial foundation models (GFMs) such as AlphaEarth turn a location into a
compact embedding used as a frozen feature extractor for environmental
regression. They have been benchmarked on agriculture, land cover, biomass, and
air quality, but not on surface solar resource, one of the most economically
important location-dependent quantities. This study asks, with pre-registration
and explicit kill criteria, whether AlphaEarth's 64-dimensional embedding predicts
the **clear-sky-index climatology** of a site *beyond what latitude, longitude,
and elevation already provide.*

## Result

On 38 German DWD pyranometer stations, predicting the **detrended mean clear-sky
index** (the part of site climatology that latitude, longitude, and elevation do
not explain), under spatial-block leave-stations-out cross-validation:

- A **pilot signal existed** on the first 28 stations: embeddings beat geography
  by **+27.6%** MAE (per-station, Wilcoxon p=0.004), and beat a shuffled-embedding
  null as well.
- It **did not replicate.** On 10 additional stations the embedding beat geography
  at only **3 of 10** sites (versus 22 of 28 in the pilot). A geo-easy confound
  was ruled out (the added sites are equally hard for geography; Mann-Whitney
  p=0.70), so the most consistent interpretation is that the pilot effect is
  genuinely fragile.
- A **power analysis** shows the per-station advantage is small (median +0.00015,
  about 2.6% of the geo baseline error) and never useful-sized: even under the
  optimistic scenario the effect's confidence interval never clears a
  25%-of-baseline "generates a lead" bar at any n up to 160, and under the
  replication-based scenario the decision rule never fires. **Verdict: WORLD 2,
  significant-but-not-useful; do not build the full multi-network study, and adopt
  a limitation framing instead.**
- A **cluster diagnostic** finds no physical site property (terrain ruggedness,
  coastal distance, local variability) that survives FDR correction as a predictor
  of where the embedding helps.
- An **adversarial reproducibility check** re-derives every headline number from
  raw bytes through an independent code path, passes positive and negative
  controls, and confirms determinism. The negative result is real, not a pipeline
  artifact. See **[VERIFICATION.md](VERIFICATION.md)**.

The hourly forecasting angle (Layer 3) is negative by construction and was retired
early: a static annual embedding cannot predict hour-to-hour weather.

## Key numbers

| stage | finding |
|---|---|
| Pilot Layer 2 (n=28, residual kt_cs) | emb MAE 0.0125 vs geo 0.0182 vs shuffle 0.0169 |
| Spatial-block + per-fold detrend (n=28) | emb-vs-geo **+27.6%**, Wilcoxon p=0.004 |
| Pool to n=38 | emb-vs-geo **+13.7%** (p=0.025); strict n=35 **+23.8%** (p=0.070, n.s.) |
| Per-station replication | emb beats geo **22/28** core vs **3/10** added |
| Geo-easy test | added geo-residual ≈ core (MWU p=0.70), not geo-easy |
| Power (per-station advantage) | median **+0.00015**; never clears 25%-of-geo usefulness bar |
| Cluster diagnostic | no physical covariate survives BH-FDR |
| Verification (from raw) | CORE 27.57%, ALL 13.69%, median 0.00015, all MATCH, controls pass |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Earth Engine steps need an authenticated account and a GCP project:
earthengine authenticate
export EE_PROJECT=your-ee-project-id
python parse_stations.py     # builds dwd_core_stations.csv (station list)
./reproduce.sh               # runs scripts 01 -> 15 in order
```

`reproduce.sh` marks which steps need network (DWD download) or Earth Engine
(embedding extraction, SRTM ruggedness). To re-verify the result against the
committed intermediates without re-running the whole pipeline, run
`python scripts/15_verify_all.py`: it re-derives everything from the raw DWD zips
and AlphaEarth `.npy` files and checks it against the reported numbers.

Scripts, in order:

| # | script | purpose | needs |
|---|---|---|---|
| 01 | `01_download_dwd.py` | pull DWD hourly solar archives | network |
| 02 | `02_add_clear_sky.py` | pvlib clear-sky + clear-sky index | — |
| 03 | `03_extract_alphaearth.py` | AlphaEarth embeddings (core) | Earth Engine |
| 04 | `04_sanity_plots.py` | validation figures | — |
| 05 | `05_layer2_regression.py` | pilot per-station regression | — |
| 06 | `06_layer3_residual_hourly.py` | hourly test (negative by design) | — |
| 07 | `07_layer2_spatial_block.py` | spatial-block CV | — |
| 08 | `08_layer2_spatial_block_perfold.py` | + per-fold detrend (leakage-clean) | — |
| 09 | `09_pool_relaxed_stations.py` | add relaxed stations to n=38 | network |
| 10 | `10_extract_added_embeddings.py` | embeddings for added stations | Earth Engine |
| 11 | `11_pool_retest.py` | re-test at n=38 + sensitivity | — |
| 12 | `12_geo_easy_diagnostic.py` | test the geo-easy confound | — |
| 13 | `13_power_analysis.py` | the kill-decision instrument | — |
| 14 | `14_cluster_diagnostic.py` | where, if anywhere, the embedding helps | Earth Engine (SRTM) |
| 15 | `15_verify_all.py` | adversarial re-derivation from raw | — |

### Reproduction notes

- DWD hourly timestamps are UTC; the J/cm² to W/m² conversion is
  `J/cm² × 10000 / 3600`. A useful sanity check is that the mean clear-sky index
  lands near 0.6–0.8 per station.
- `pvlib`'s Ineichen model auto-loads a global Linke-turbidity climatology bundled
  with the package; this works offline once installed.
- Earth Engine's free tier is sufficient for the small per-station `reduceRegion`
  calls used here.

## Data, provenance, and checksums

Raw inputs and large intermediates are not committed (they are regenerable).
Curated, license-clean derived outputs (results CSVs, key figures, environment and
checksums) are tracked so the result can be inspected without re-running. The
38-station AlphaEarth station-mean embeddings are bundled (see
`data/alphaearth/NOTICE`), so the analysis can be reproduced without an Earth
Engine account.

- Inputs are pinned by sha256 in
  [`data/repro/data_checksums.txt`](data/repro/data_checksums.txt) (38 DWD zips and
  38 AlphaEarth `.npy`). Regenerate via scripts 01/03/10 and confirm identical
  hashes to verify identical inputs.
- Exact package versions: [`data/repro/environment.txt`](data/repro/environment.txt).

### Licenses and attribution

- **DWD hourly solar radiation** — Deutscher Wetterdienst, Climate Data Center,
  under **GeoNutzV** (free use and redistribution with a source note). *Source:
  Deutscher Wetterdienst; derived to clear-sky index, modified.*
- **AlphaEarth Foundations Satellite Embedding** — **CC-BY 4.0**. *The AlphaEarth
  Foundations Satellite Embedding dataset is produced by Google and Google
  DeepMind.* (GEE asset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.)
- **SRTM** (terrain ruggedness covariate) — `USGS/SRTMGL1_003`, public domain.

## Limitations and what this does not claim

- It does not claim AlphaEarth is useless for solar. It claims that, at this pilot
  scale (n=38, one country, one network, annual embeddings, one target), the
  per-station advantage over latitude/longitude/elevation is too small and too
  fragile to be useful, and does not replicate to fresh stations.
- It is not a multi-model claim. Only AlphaEarth was tested; Prithvi, Clay, and
  SatCLIP were not.
- A weak real effect is not excluded. The embedding-helps-where-geography-fails
  signature is weakly present (Spearman ρ=+0.35, p=0.03) but is core-driven and
  not robust; confirming or rejecting it requires a larger, more diverse sample.
- The pilot uses a single network. Cross-network harmonization artifacts are not
  tested here; the negative result sidesteps that risk but also limits external
  validity.
- The pre-registered primary rule is on emb-vs-shuffle; this pilot's shuffle-based
  p is seed-fragile at n=38 (see VERIFICATION.md). The emb-vs-geo contrast
  reported above is seed-stable and is the contrast relevant to a beyond-
  coordinates claim.

## Repository map

- `scripts/01..15` — the pipeline, in order.
- `spatial_holdout/FINDINGS.md` — a dated record of each analysis step and result.
- `PREREGISTRATION.md` — the pre-registered primary hypothesis, target, estimator,
  test, and kill criteria that governed the executed work.
- `VERIFICATION.md` — the adversarial reproducibility report.
- `data/` — curated outputs tracked; raw and intermediate files are regenerable
  (see above).
