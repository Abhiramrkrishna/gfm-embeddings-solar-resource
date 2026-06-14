# Station-pooling feasibility — can we grow n=28 → 50+?

Dated 2026-06-14. Question: what additional ground stations are reachable to
strengthen the spatial-transfer claim, and what does each cost?

## Bottom line

**n=28 → ~38–40 is cheap and realistic; n=28 → 50+ is NOT reachable from German
DWD data alone** — it requires pulling in neighbouring-country pyranometers via
SYNOP/WMO exchange or national open-data portals. The embedding side is a
non-issue in every scenario: GEE re-extraction is minutes of compute and free.
The real cost is ground-truth ingest + QC code, and (for cross-border) handling
heterogeneous instruments/formats.

## 1. DWD relaxed-criteria pool — exact, from metadata

The DWD radiation network is small. The full metadata block in
`parse_stations.py` contains **56 stations total, ever**; only ~40 are currently
active. Relaxing our "full 2020–2024" criterion buys little:

| Criterion | Stations | Beyond core 28 |
|---|---|---|
| Core: full 5-yr 2020–2024 | 29* | — |
| Relaxed 3-yr (start ≤ 2022-01) | 34 | **+6** |
| 2-yr (start ≤ 2023-01) | 38 | **+10** |
| All active (bis ≥ 2025) | 40 | **+12** |

\* The core *criterion* yields 29; our working CSV has 28 because **04642
Seehausen was dropped for quality (71.5 % valid FG), not coverage**. It can be
re-added if we accept the gaps.

The +12 active-but-excluded stations (coverage of the 2020–2024 window):

- **3-yr+ usable now (+6):** 04642 Seehausen (quality re-add), 00867 Lautertal,
  02925 Leinefelde, 07370 Waldmünchen, 05142 Ueckermünde, 06197 Lügde
- **2-yr (+4):** 01420 Frankfurt/Main, 15444 Ulm-Mähringen,
  **01346 Feldberg/Schwarzwald (1486 m — valuable high-elevation point)**,
  15000 Aachen-Orsbach
- **<2-yr, too short (+2):** 01639 Gießen (2024-), 02483 Kahler Asten (2025-)

**Realistic DWD-internal ceiling: ~38 stations** (28 core + 6 three-year + the
4 two-year, accepting shorter records). That is the most we get without a second
network. Note these are recent-start stations, so they add *spatial* coverage
but with **shorter time series** → noisier per-station kt_mean climatology, the
exact target of our Layer 2 test. Worth a sensitivity check on minimum record
length before pooling.

## 2. SYNOP — clarification matters here

In the German context, **"DWD SYNOP" does not add stations beyond the CDC solar
set**: the synoptic stations that report global radiation in their BUFR/SYNOP
messages *are* the same pyranometer network already enumerated above. So SYNOP
within Germany ≈ the 40 stations in §1.

The genuine expansion route labelled "SYNOP" is **neighbouring-country
pyranometers** exchanged over WMO/SYNOP, or pulled from national open portals:

- KNMI (Netherlands, ~30 GHI stations, fully open)
- RMI Belgium, DWD-adjacent
- ZAMG/GeoSphere Austria, MeteoSwiss (open since 2025), ČHMÚ Czechia, IMGW Poland
- Météo-France (radiation subset, open data)

A **Central-Europe pyranometer pool of n=50–80 is realistic** this way, and
AlphaEarth coverage is global so embeddings extract identically. Costs/risks:
- Heterogeneous QC, instrument types, timestamp conventions, units → real ingest
  work (each portal is its own parser, like the DWD one we already wrote).
- Reframes the paper from "Germany" to "Central Europe" — defensible, arguably
  stronger (more terrain/climate diversity), but a scope decision to make
  consciously.
- Clear-sky baseline (pvlib Ineichen) is location-agnostic, so no extra modelling.

**Caveat (flagged honestly):** the exact open-station counts per country above
are from memory and should be verified against each portal before committing —
treat them as order-of-magnitude, not exact like the DWD numbers.

## 3. PvLive — not a drop-in station source (recommend against for this purpose)

"PvLive"-type products (e.g. Fraunhofer **PV-Live** / regional feed-in
estimates) are **PV power, regionally aggregated, not point pyranometer GHI**.
For our kt_cs climatology that is the wrong modality:
- It is generation, not irradiance — needs a PV model to invert back to GHI,
  injecting array/tilt/derating assumptions we are trying to avoid.
- It is polygon-aggregated, not point — no clean (lat,lon) to extract a 100 m
  embedding at, which is the whole premise of the site-descriptor test.
- CLAUDE.md already records that per-site PV in Germany is closed (data
  protection); the open products are aggregates.

So PvLive is **not integrable as added ground-truth stations**. It is only
usable as a separate, weaker downstream demo (regional irradiance map), if at
all — not for growing n in the Layer 2 transfer test. **My recommendation: drop
it from the n-growth plan.**

## 4. Embedding re-extraction cost — negligible everywhere

From `scripts/03_extract_alphaearth.py`: per station = 5 years × 1
`reduceRegion` mean over a 100 m buffer at 10 m scale + 0.5 s politeness sleep.
Each call is sub-second on a tiny region.

| Scenario | New stations | New GEE calls | Wall-clock | $ |
|---|---|---|---|---|
| DWD relaxed (+10) | 10 | ~50 | ~1–2 min | free (noncommercial) |
| Central-Europe (+25) | 25 | ~125 | ~3–5 min | free |
| Aggressive (+50) | 50 | ~250 | ~6–10 min | free |

Embeddings are not the bottleneck — they are free and instant. **All the cost is
in the ground-truth pipeline** (parsers + QC for each new network), and for
cross-border, in the scope decision.

## Recommendation

1. **Cheap immediate win:** pool the DWD relaxed +6 (and +4 two-year) → n≈38,
   with a minimum-record-length sensitivity check. Re-run the per-fold-detrend
   spatial-block test; if the GO holds at n≈38 with tighter CI, that alone
   materially hardens the result.
2. **To reach n=50+:** commit to a Central-Europe scope and ingest 1–2
   neighbouring open networks (KNMI is the easiest first). This is the only
   real path past ~40 and the highest-value next step if a reviewer questions
   power at n=28.
3. **Drop PvLive** from the station-growth plan; reconsider only as a separate
   spatial-demo modality later.
