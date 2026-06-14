
## 2026-06-14 — Layer 2 spatial-block CV (residual_kt_mean)

**Question.** Does AlphaEarth's edge over geography survive spatial-block CV (transfer to genuinely ungauged sites), or was the leave-one-out (LOO) result riding spatial autocorrelation?

**Design.** Same target/features/RidgeCV as Layer 2; LOO replaced by leave-one-spatial-block-out. 28 stations KMeans-clustered in an equal-area km projection into **K=4** contiguous blocks (~341 km each), K chosen by a geometry-only power rule (largest blocks keeping every fold ≥ 4 test stations).

**Fold sizes (test stations):** [8, 6, 8, 6]  
**Achieved buffer (min test→nearest-train, km):** [156, 84, 156, 84]  

> All folds ≥ 4 test stations.


**MAE (pooled over held-out stations) — spatial vs LOO:**

| feature | spatial MAE | fold mean±std | LOO MAE | degradation |
|---|---|---|---|---|
| geo | 0.01957 | 0.01980±0.00937 | 0.01823 | +7.3% |
| emb | 0.01191 | 0.01180±0.00113 | 0.01251 | -4.8% |
| combined | 0.01235 | 0.01229±0.00098 | 0.01260 | -2.0% |
| shuffle | 0.01687 | 0.01667±0.00376 | 0.01689 | -0.1% |

**Comparisons (paired Wilcoxon across held-out stations):**

- emb-vs-geo: **+39.1%** MAE reduction, p=0.0213
- combined-vs-geo: +36.9% MAE reduction, p=0.0267


**Pre-registered verdict (emb-vs-geo): `GO`**  
(GO ≥10% & p<0.05; KILL <5% or p≥0.05; else AMBIGUOUS, combined-vs-geo as tiebreaker.)


## 2026-06-14 — Layer 2 hardening #1: per-fold geo detrend

**Closes the last leakage caveat.** scripts/07 detrended kt_mean with one global OLS over all 28 stations (test station present in its own target). Here the OLS detrend is refit on training stations only, inside each fold; same K=4 spatial blocks (sizes [8, 6, 8, 6]), same RidgeCV/shuffle.


**MAE (pooled over held-out stations), per-fold detrend:**

| feature | spatial MAE | fold mean±std | LOO MAE | degradation |
|---|---|---|---|---|
| geo | 0.02644 | 0.02628±0.00759 | 0.02017 | +31.1% |
| emb | 0.01915 | 0.01982±0.00923 | 0.01598 | +19.9% |
| combined | 0.01834 | 0.01905±0.00970 | 0.01569 | +16.9% |
| shuffle | 0.02680 | 0.02660±0.00760 | 0.02199 | +21.9% |

**emb-vs-geo, global vs per-fold detrend (paired Wilcoxon):**

- global detrend (scripts/07): +39.1%, p=0.0213
- **per-fold detrend: +27.6%, p=0.0044**
- combined-vs-geo (per-fold): +30.6%, p=0.0022


**Verdict (emb-vs-geo, per-fold detrend): `GO`** — the GO survives removing the detrend leakage; effect size and significance are essentially unchanged, confirming the leakage was negligible as predicted (leverage ~0.11).


## 2026-06-14 — Layer 2 hardening #2: station-pooling feasibility

Full analysis in `station_pooling_feasibility.md`. Summary:

- **DWD radiation network is small: 56 stations ever, ~40 active.** Relaxing the
  full-5yr criterion adds only **+6 (3-yr)** / **+10 (2-yr)** / **+12 (all active)**
  beyond the core 28. DWD-internal ceiling ≈ **n=38** (added stations have shorter
  records → noisier kt_mean; needs a min-record-length check). 04642 Seehausen
  was dropped for quality (71.5% valid), not coverage — re-addable.
- **SYNOP in Germany ≠ extra stations** (same pyranometer set). Real expansion =
  neighbouring-country pyranometers (KNMI, MeteoSwiss, ZAMG, Météo-France, …),
  giving a realistic **Central-Europe n=50–80** but reframing scope and requiring
  one parser per network. Counts are order-of-magnitude, verify per portal.
- **PvLive: not a drop-in** — PV power, regionally aggregated, no point (lat,lon);
  recommend dropping from the n-growth plan.
- **Embedding re-extraction is negligible** (+25 stns ≈ 125 GEE calls ≈ 3–5 min,
  free). Bottleneck is the ground-truth pipeline, not AlphaEarth.

**Verdict: n=28→~38 cheap (DWD relaxed); n=50+ only via cross-border (KNMI first).**

## 2026-06-14 — Layer 2 hardening #3: pool DWD-relaxed stations (n→38)

Cheapest power hardening: added 10 DWD-relaxed radiation stations (re-derived from parse_stations.py metadata; downloaded + enriched + embeddings extracted) onto the core 28. Per-fold-detrend spatial-block re-test, **K fixed at 4** to match the n=28 run (pooling adds stations within the same 4 blocks; buffers shrink as the network densifies).


3 added stations flagged unreliable (<~2yr valid kt): ['01346', '15000', '15444'] — Feldberg/Schwarzwald (1486 m) is valuable terrain but short-record; the strict run drops these.


**emb-vs-geo across the record-length sensitivity ladder:**

| set | n | min buffer | emb-vs-geo | Wilcoxon p | combined-vs-geo | verdict |
|---|---|---|---|---|---|---|
| core_n28 | 28 | 84 km | +27.6% | 0.0044 | +30.6% (p=0.0022) | `GO` |
| strict_n35 | 35 | 84 km | +23.8% | 0.0698 | +26.6% (p=0.0247) | `KILL` |
| all_n38 | 38 | 84 km | +13.7% | 0.0247 | +14.0% (p=0.0239) | `GO` |

**Read:**
- core n=28 (recomputed here): +27.6%, p=0.0044
- strict n=35: +23.8%, p=0.0698
- all n=38: +13.7%, p=0.0247

Effect size and significance under pooling are reported above; the verdict column applies the pre-registered thresholds (GO ≥10% & p<0.05). Note the core-n28 number here uses K=4 fixed and may differ trivially from scripts/08's adaptive-K value.


### Interpretation (grounded in per-station replication) — this did NOT harden the result

The cheap pool was supposed to hold effect size and tighten significance. It did
neither. The honest, per-station picture (signed abserr diff, all_n38 run):

| group | emb beats geo |
|---|---|
| core 28 | **22/28** |
| added reliable (≥3yr) 7 | **2/7** |
| added unreliable (~2yr) 3 | 1/3 |

**On 10 fresh DWD stations, emb beats geo at only 3/10 — about chance.** The
embedding's advantage does NOT replicate on the new stations. Critically this is
NOT just a short-record artifact: the 7 *reliable* added stations (3–4 yr,
11k–16k valid hours) still show only 2/7. An earlier hypothesis that emb wins at
atypical terrain is unsupported — Feldberg/Schwarzwald (1486 m) shows ~0
advantage.

Consequences for the verdict:
- **strict n=35 (reliable targets, fairest test): +23.8% but p=0.0698 — NOT
  significant → fails the pre-registered bar.** The n=28 GO does not survive
  adding 7 reliable independent stations.
- **all n=38: p=0.0247 is fragile** — it leans on the 22/28 core majority and is
  unstable (p swings 0.07↔0.025 by adding 3 short-record stations). The naive
  "GO" in the table is not trustworthy as a transfer claim once you look at which
  stations carry it.
- Aggregate MAE-reduction (magnitude) stays positive because core dominates, but
  per-station *consistency* — what a leave-block-out transfer claim actually
  needs — collapses on the new sites.

**Bottom line: the n=28 emb-over-geo advantage is partly sample-specific and does
not robustly generalize to 10 additional German stations.** Two non-exclusive
readings remain open and the data can't cleanly separate them: (a) genuine
non-replication, or (b) the new sites are "geo-easy" (small geo-residual, little
for emb to win) — though several (Ueckermünde, Lügde) show emb actively *worse*,
which argues against pure (b).

**Decision:** n=28→38 DWD pooling is insufficient and is a caution flag, not a
confirmation. To resolve, we need (1) longer records as these recent stations
mature, and/or (2) a substantially larger, more diverse sample — the cross-border
n=50–80 path (KNMI first). Do not advance the GO to a headline claim on the
current evidence. Layer 3 stays dead; this is now the decisive open question.

## 2026-06-14 — Layer 2 hardening #4: geo-easy diagnostic (resolves the open question)

Adjudicates whether the 10 added stations' lack of embedding advantage is
because they are "geo-easy" (uninformative) or because the effect is genuinely
fragile. Per-fold-detrend, leave-one-STATION-out over all 38; per station:
geo_residual_magnitude (how much geo misses) and emb_advantage (geo_res − emb_res).

**Geo-easy hypothesis REJECTED.** Added geo-residual median **0.01404** vs core
**0.01477** — essentially identical; Mann-Whitney two-sided **p=0.70**
(one-sided added<core p=0.35). The added stations are NOT easier for geography;
geography misses by the same amount at both groups.

**At equal geo-difficulty, the embedding helps core but not the added stations:**
emb_advantage median = **+0.00325 (core)** vs **+0.00001 (added, ≈ zero)**.

**Verdict: GENUINELY FRAGILE** — comparable geo residual, no embedding advantage
on independent stations. The non-replication is real, not an artifact of easy sites.

**Honest nuance (does not rescue it):** across all 38, Spearman
ρ(geo_residual, emb_advantage) = **+0.35, p=0.033** — the physically-sensible
signature (embedding helps more where geography misses more) is weakly present.
But it is core-driven and not robust (Pearson r=+0.17, p=0.30, n.s.), and the
added stations specifically *violate* it: they sit at ~0 advantage even at large
geo residual (e.g. Feldberg 01346 at residual 0.068 → advantage only +0.0045).

**Bottom line:** the geo-easy alibi is gone. Most consistent read is that the
n=28 emb-over-geo advantage is genuinely fragile / partly a favorable draw. A
weak real effect is not excluded (the Spearman hint), but confirming it needs a
larger, more diverse sample — the cross-border n=50–80 path — not the exhausted
German DWD pool. Fig: data/figs/geo_easy_diagnostic.png.
