# Research Plan: Do Geospatial Foundation Model Embeddings Encode Solar Resource Beyond Geographic Coordinates?

**A pre-registered, multi-network, multi-model evaluation study**

Author: Abhiram Radha Krishna
Collaborator (domain): Dr. Garrett H. Good (to be engaged once Phase 0 + first Phase 2 results exist)
Status: planning / pre-registration draft
Last updated: (fill in)

---

## 0. The one-paragraph version

Geospatial foundation models (GFMs) such as AlphaEarth, Prithvi, and Clay are increasingly used as frozen feature extractors that turn a location into a compact embedding for downstream environmental regression. They have been benchmarked on agriculture, land cover, biomass, landslides, air quality, and groundwater — but not on **surface solar resource**, despite solar being one of the most economically important location-dependent environmental quantities. This study asks, rigorously and with pre-registration, whether GFM embeddings predict solar resource (clear-sky index climatology and its structure) **beyond what latitude, longitude, and elevation already provide** — and if so, **which models, where, and how much**. A pilot on 28 German DWD stations showed a promising AlphaEarth advantage that did **not** replicate on 10 additional stations; ruling out the "geo-easy" confound left "genuinely fragile" as the honest read. This study scales to a multi-network, multi-model design powered to actually settle the question, with the per-site heterogeneity signal (embeddings help where geography fails) as a pre-specified secondary hypothesis.

---

## 1. Why this is not "just another gap analysis"

Three grounded reasons (each verifiable against the literature, not assumed):

1. **The task is unevaluated.** Published AlphaEarth downstream evaluations enumerate agriculture, crop mapping, land cover, biomass, landslide susceptibility, urban air quality, groundwater — surface solar irradiance does not appear in any benchmark list found as of mid-2026. This is a named blank, not a manufactured one.

2. **The skeptical frame is already legitimate.** The PANGAEA GFM benchmark established that GFMs do not consistently beat supervised baselines. A careful "here is where they help and where they don't, for solar" extends an active debate rather than inventing a contrarian position.

3. **There is a real positive hypothesis to chase.** The pilot's Spearman rho = +0.35 (embedding advantage correlates with geographic-baseline error) is the physically sensible fingerprint of a real-but-conditional effect: embeddings should help at sites where coordinates miss (complex terrain, coastlines, urban, water bodies) and add nothing at "generic" sites. That is a mechanistic hypothesis, testable at scale.

The combination — unevaluated task + legitimate skeptical frame + a concrete positive hypothesis — is what makes this a study, not a gap note. Every outcome is publishable; see Section 7.

---

## 2. The honest design principle (the spine)

The danger in "run everything, let results pick the framing" is post-hoc framing — choosing the story after seeing the data. The fix is NOT to run fewer experiments. It is to **pre-commit a single confirmatory test, and explicitly label everything else exploratory.**

- **One confirmatory hypothesis (H1), one primary target, one primary test, one decision rule — all fixed before touching new data.**
- A rich **exploratory** layer (multi-model, multi-target, heterogeneity, interpretation) that generates the next hypotheses but is never reported as if it were confirmatory.

This structure is what makes a large body of experiments honest rather than a fishing expedition. Reviewers trust it precisely because it does not pretend exploration was confirmation.

---

## 3. Hypotheses

### Primary (confirmatory) — H1
> Across a held-out set of ground stations not used in any pilot, AlphaEarth embeddings predict the **detrended mean clear-sky index** (residual after removing an OLS lat/lon/elevation trend) with lower error than a shuffled-embedding null, evaluated under spatial-block leave-stations-out cross-validation.

- **Primary model:** AlphaEarth (the strongest published GFM; the pilot model).
- **Primary target:** detrended mean clear-sky index (kt_cs), per-fold detrend.
- **Primary estimator:** Ridge regression with GCV alpha selection (justified by pilot: trees overfit, PLS over-parameterized at this n/d).
- **Primary test:** Wilcoxon signed-rank on per-station |error| (embedding vs shuffle null), one-sided.
- **Decision rule (fixed in advance):** H1 is supported iff BOTH
  (a) per-station embedding-beats-shuffle rate > 60%, AND
  (b) Wilcoxon p < 0.01 (stricter than 0.05 to account for the multi-network pooling).
- **Secondary confirmatory check:** embedding must also beat the **geographic** baseline (not just shuffle) on the same test, at p < 0.05. (Beating shuffle shows "contains information"; beating geo shows "adds beyond coordinates" — both required for the claim that matters.)

### Secondary (pre-specified, exploratory) — H2 (the interesting one)
> The embedding's advantage over the geographic baseline is larger at sites where the geographic baseline has higher error (high local heterogeneity). Operationalized as: Spearman rho(geo_residual_magnitude, emb_advantage) > 0 across all stations, with rho and CI reported. Stratify by site-type covariates: terrain ruggedness (DEM-derived), coastal proximity, urban fraction, Koppen climate zone.

### Tertiary (exploratory) — H3, H4
- **H3 (multi-model):** Which GFM (AlphaEarth vs Prithvi vs Clay vs SatCLIP, as feasible) best predicts solar resource, and does any beat geo robustly? Pure comparison; any outcome informative.
- **H4 (temporal-resolution limitation):** Annual embeddings are structurally limited for a sub-annual quantity (cloud climatology). Test whether predictive value concentrates in targets that are more "static" (mean) vs more "dynamic" (variance, persistence, over-irradiance). If embeddings only help static targets, that localizes the limitation and motivates finer-temporal GFMs.

---

## 4. Data (all verified available)

### Ground-truth networks (targets)
| Network | Region | ~Stations | Terrain | Access | Format | Notes |
|---|---|---|---|---|---|---|
| DWD | Germany | ~40 | mixed | CC-BY, opendata | CSV/ZIP | PILOT — already done (28 core + 10 added) |
| KNMI | Netherlands | 50+ | flat | CC-BY-4.0, Open Data API | NetCDF | flat terrain isolates non-terrain signal |
| BSRN | global sparse | ~58 active | diverse | free, read-account (email AWI) via PANGAEA | per-LR files | highest accuracy; contrasting climate zones; one email to Amelie Driemel |
| MeteoSwiss / GeoSphere AT | Alps | tens | mountainous | mostly open | varies | alpine counterpart to flat KNMI |
| (secondary surface) PVGIS / CAMS / SARAH-3 | pan-EU | gridded | all | free | NetCDF/CSV | satellite-derived; baseline + dense map, NOT ground truth |

**Power consideration:** target n >= 100 ground stations across terrain types. The pilot suggests the effect (if real) is small; a power analysis (Phase 0) determines the exact n needed for 80% power.

### Embeddings (inputs)
| Model | Source | Status |
|---|---|---|
| AlphaEarth Foundations | Google Earth Engine, annual 64-d | pipeline done |
| Prithvi (NASA/IBM) | Hugging Face | new extraction pipeline |
| Clay | Hugging Face / open | new extraction pipeline |
| SatCLIP | open | new extraction pipeline (location-only; interesting contrast) |

All consumed FROZEN. No GFM training. Each is "backbone -> embedding vector," same pattern already built once for AlphaEarth.

### Harmonization (a real risk to control, not ignore)
Different networks = different instruments, calibration, QC, reporting. **Network identity must be a controlled variable.** Mandatory: test H1 WITHIN each network separately, not only pooled. If the "advantage" tracks network identity rather than solar physics, that is an artifact and must be caught.

---

## 5. Phased plan

### Phase 0 — Verify the gap and power the study (1-2 weeks). GATE.
This phase exists because a literature gap asserted by an AI assistant is not trustworthy. Abhiram verifies it himself.
- [ ] Systematic literature check: search Google Scholar, Semantic Scholar, arXiv, OpenReview for "AlphaEarth solar", "satellite embedding irradiance", "geospatial foundation model solar resource", "GFM embedding photovoltaic site". Read abstracts of the top ~40. Confirm no existing AlphaEarth-on-solar evaluation. **If one exists, the framing pivots to differentiate from it — do this BEFORE building.**
- [ ] Read in full: the AlphaEarth paper (Brown et al. 2025), PANGAEA GFM benchmark (Marsocci et al.), one or two of the recent AlphaEarth downstream papers (agriculture 2601.00857, landslide 2601.07268). Understand their evaluation protocols so ours is comparable and defensible.
- [ ] **Power analysis:** using the pilot's per-station effect size on detrended kt_cs, simulate Wilcoxon power vs n. Determine n for 80% power. Decide whether n~100 suffices or whether more networks are needed. (This becomes a paper subsection reviewers value.)
- [ ] **Write and commit the pre-registration** (Section 3 decision rules, frozen) with a git timestamp BEFORE any new data is touched.

**Gate:** proceed only if (a) the gap is confirmed by Abhiram's own reading, and (b) power analysis says the achievable n can detect the pilot effect. If the effect is too small to detect at achievable n, that itself reframes the paper toward H4 (limitation study).

### Phase 1 — Build the multi-network ingestion (3-4 weeks)
- [ ] KNMI pipeline: Open Data API, NetCDF parser, global radiation -> hourly GHI -> clear-sky index (reuse pvlib step). New parser; budget for it.
- [ ] BSRN: email AWI for read account; build PANGAEA/ftp ingestion for European + selected global stations; map logical records to GHI.
- [ ] MeteoSwiss/GeoSphere if open access confirmed.
- [ ] Unified schema: every station -> (network, lat, lon, elev, hourly GHI, clear-sky index, coverage stats, site-type covariates). One clean parquet per station, network-tagged.
- [ ] Per-network QC mirroring the DWD pipeline (completeness thresholds, kt_cs sanity, over-irradiance rates). Lock station inclusion by RULE, not by eye.

### Phase 2 — AlphaEarth confirmatory test at scale (1-2 weeks)
- [ ] Extract AlphaEarth embeddings for all qualifying stations across all networks (pipeline exists).
- [ ] Run the PRE-REGISTERED H1 test: detrended kt_cs, Ridge, spatial-block LOO, shuffle null + geo baseline, pooled AND within-network.
- [ ] Report against the fixed decision rule. **This is the headline result, whatever it says.**
- [ ] **First credible artifact exists here -> engage Garrett** (the over-irradiance / clear-sky-model question, physical interpretation, co-authorship).

### Phase 3 — Exploratory landscape (3-5 weeks)
- [ ] H2 heterogeneity: emb_advantage vs geo_residual, stratified by terrain/coast/urban/Koppen. The "where does it help" map.
- [ ] H3 multi-model: extract Prithvi, Clay, SatCLIP; same protocol; comparison table. (Each new extraction is bounded work.)
- [ ] H4 target spectrum: static (mean) vs dynamic (std, p95, persistence, over-irr) targets; localize where embeddings help.
- [ ] Physical interpretation (with Garrett): which embedding dims correlate with which physical quantities; the water/terrain/urban signatures seen in the pilot.

### Phase 4 — Write (3-4 weeks, overlapping Phase 3)
- [ ] Frame chosen by the confirmatory result (Section 7), exploratory layer as support.
- [ ] Preprint -> arXiv (need endorser; Garrett or a Phase-1 contact).
- [ ] Target venue chosen by outcome and strength.

Total: ~3-4 months of focused part-time work. Intermediate artifacts (pre-registration + Phase 2 result) are themselves usable as a proposal for cold-reaching professors even before the full paper.

---

## 6. What makes it rigorous (the checklist reviewers look for)
- Pre-registered primary hypothesis + decision rule, git-timestamped before new data.
- Spatial-block cross-validation (no spatial leakage).
- Shuffle null (controls for "contains info") AND geo baseline (controls for "beyond coordinates").
- Within-network controls (catches network-identity artifacts).
- Power analysis (honest about detectability).
- Multiple-comparisons discipline (one confirmatory target; rest labeled exploratory).
- Multi-model (not a single-model anecdote).
- Negative/null results reported in full; calibrated language.

---

## 7. Outcome -> framing map (decided by data, but frames pre-enumerated so it is not post-hoc)
| Confirmatory H1 result | H2 heterogeneity | Resulting paper |
|---|---|---|
| Robust: emb beats geo across networks | — | "GFM embeddings are useful solar site descriptors: which model, where, how much." Positive contribution. |
| Conditional: fails pooled but H2 holds | supported | "GFM embeddings help at high-heterogeneity sites; coordinates suffice elsewhere." Mechanistic, nuanced. |
| Null: no robust advantage anywhere | not supported | "Current GFM embeddings do not reliably encode solar resource beyond coordinates: a cautionary multi-network evaluation." Honest negative; needed; cites into PANGAEA debate. |
| Null pooled, but one model (e.g. AlphaEarth) > others | partial | "Among GFMs, only X shows solar-relevant signal, and only conditionally." Comparative + conditional. |

Every row is a real paper. The pre-registration + power analysis make even the null row credible rather than "we didn't try hard enough."

---

## 8. Risks and honest failure modes
- **Effect is too small to detect even at n~150.** Then H4 (limitation) becomes the paper. Still publishable; arguably the most useful outcome.
- **Network harmonization artifacts dominate.** Mitigated by within-network analysis; if unfixable, report as a finding about cross-network evaluation difficulty.
- **Multi-model extraction is more work than budgeted.** Mitigation: AlphaEarth confirmatory result (Phase 2) stands alone as a minimum paper; other models are additive.
- **Clear-sky model bias (the parked over-irradiance issue).** Resolve with Garrett in Phase 2; test sensitivity to Ineichen vs McClear vs SOLIS.
- **AI-asserted gap is wrong.** Mitigated by Phase 0 gate — Abhiram confirms by his own reading before committing.

---

## 9. Immediate next actions
1. Phase 0 literature check — Abhiram reads, confirms gap independently. (Do not skip. Do not delegate the judgment.)
2. Power analysis on pilot effect size.
3. Write + commit pre-registration.
4. Only then: start KNMI ingestion.
