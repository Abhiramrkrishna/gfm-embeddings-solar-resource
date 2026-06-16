# Findings log

A dated record of each analysis step and its result. Numbers are reported as
produced by the scripts; interpretation is kept separate from the measurements.

## 2026-06-14 — Layer 2 spatial-block CV (residual_kt_mean)

**Question.** Does the AlphaEarth advantage over geography survive spatial-block
CV (transfer to genuinely ungauged sites), or was the leave-one-out (LOO) result
driven by spatial autocorrelation?

**Design.** Same target, features, and RidgeCV as Layer 2; LOO replaced by
leave-one-spatial-block-out. The 28 stations are KMeans-clustered in an equal-area
km projection into **K=4** contiguous blocks (~341 km each); K is chosen by a
geometry-only rule (largest blocks keeping every fold ≥ 4 test stations).

**Fold sizes (test stations):** [8, 6, 8, 6]
**Achieved buffer (min test→nearest-train, km):** [156, 84, 156, 84] (all folds ≥ 4 test stations)

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
(GO ≥10% & p<0.05; KILL <5% or p≥0.05; otherwise AMBIGUOUS, combined-vs-geo as tiebreaker.)

## 2026-06-14 — Layer 2 hardening 1: per-fold geo detrend

Closes the last leakage caveat. scripts/07 detrended kt_mean with one global OLS
over all 28 stations (the test station present in its own target). Here the OLS
detrend is refit on training stations only, inside each fold; same K=4 spatial
blocks (sizes [8, 6, 8, 6]), same RidgeCV and shuffle.

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

**Verdict (emb-vs-geo, per-fold detrend): `GO`.** The GO survives removing the
detrend leakage; effect size and significance are essentially unchanged,
confirming the leakage was negligible (leverage ~0.11).

## 2026-06-14 — Layer 2 hardening 2: station-pooling feasibility

Full analysis in `station_pooling_feasibility.md`. Summary:

- The DWD radiation network is small: 56 stations ever, ~40 active. Relaxing the
  full-5yr criterion adds only +6 (3-yr) / +10 (2-yr) / +12 (all active) beyond the
  core 28. The DWD-internal ceiling is about n=38 (added stations have shorter
  records and noisier kt_mean, so a minimum-record-length check is needed). Station
  04642 Seehausen was dropped for quality (71.5% valid), not coverage, and is
  re-addable.
- SYNOP in Germany does not add stations beyond the CDC solar set (same pyranometer
  network). Real expansion requires neighbouring-country pyranometers (KNMI,
  MeteoSwiss, ZAMG, Météo-France), giving a realistic Central-Europe n=50–80 but
  reframing scope and requiring one parser per network. Counts are order-of-magnitude.
- PvLive is not a drop-in: it is PV power, regionally aggregated, with no point
  (lat, lon), and is removed from the n-growth plan.
- Embedding re-extraction is negligible (+25 stations ≈ 125 GEE calls ≈ 3–5 min,
  free). The bottleneck is the ground-truth pipeline, not AlphaEarth.

**Verdict:** n=28→~38 is cheap (DWD relaxed); n=50+ requires cross-border data (KNMI first).

## 2026-06-14 — Layer 2 hardening 3: pool DWD-relaxed stations (n→38)

Added 10 DWD-relaxed radiation stations (re-derived from parse_stations.py metadata;
downloaded, enriched, embeddings extracted) onto the core 28. Per-fold-detrend
spatial-block re-test, **K fixed at 4** to match the n=28 run (pooling adds stations
within the same 4 blocks; buffers shrink as the network densifies).

3 added stations are flagged unreliable (<~2yr valid kt): 01346, 15000, 15444.
Feldberg/Schwarzwald (01346, 1486 m) is valuable terrain but short-record; the
strict run drops these.

**emb-vs-geo across the record-length sensitivity ladder:**

| set | n | min buffer | emb-vs-geo | Wilcoxon p | combined-vs-geo | verdict |
|---|---|---|---|---|---|---|
| core_n28 | 28 | 84 km | +27.6% | 0.0044 | +30.6% (p=0.0022) | `GO` |
| strict_n35 | 35 | 84 km | +23.8% | 0.0698 | +26.6% (p=0.0247) | `KILL` |
| all_n38 | 38 | 84 km | +13.7% | 0.0247 | +14.0% (p=0.0239) | `GO` |

The verdict column applies the pre-registered thresholds (GO ≥10% & p<0.05). The
core-n28 number here uses K=4 fixed and may differ trivially from scripts/08's
adaptive-K value.

### Interpretation (per-station replication)

Pooling was intended to hold effect size and tighten significance. It did neither.
The per-station picture (signed abserr difference, all_n38 run):

| group | emb beats geo |
|---|---|
| core 28 | 22/28 |
| added reliable (≥3yr) 7 | 2/7 |
| added unreliable (~2yr) 3 | 1/3 |

On 10 fresh DWD stations, emb beats geo at only 3/10 (about chance). The advantage
does not replicate on the new stations, and this is not a short-record artifact: the
7 reliable added stations (3–4 yr, 11k–16k valid hours) still show only 2/7. An
earlier hypothesis that emb wins at atypical terrain is unsupported — Feldberg
(01346, 1486 m) shows ~0 advantage.

Consequences for the verdict:

- strict n=35 (reliable targets, fairest test): +23.8% but p=0.0698, not significant,
  fails the pre-registered bar. The n=28 GO does not survive adding 7 reliable
  independent stations.
- all n=38: p=0.0247 is fragile — it leans on the 22/28 core majority and is unstable
  (p swings 0.07↔0.025 by adding 3 short-record stations). The "GO" in the table is
  not trustworthy as a transfer claim once the carrying stations are examined.
- Aggregate MAE reduction stays positive because the core dominates, but per-station
  consistency — what a leave-block-out transfer claim requires — collapses on the
  new sites.

**Conclusion:** the n=28 emb-over-geo advantage is partly sample-specific and does
not robustly generalize to 10 additional German stations. Two non-exclusive readings
remain and the data cannot cleanly separate them: (a) genuine non-replication, or
(b) the new sites are "geo-easy" (small geo-residual, little for emb to win). Several
sites (Ueckermünde, Lügde) show emb actively worse, which argues against (b). n=28→38
DWD pooling is insufficient. Resolving it requires longer records as the recent
stations mature, or a substantially larger, more diverse sample (the cross-border
n=50–80 path, KNMI first). Layer 3 stays retired; this is the decisive open question.

## 2026-06-14 — Layer 2 hardening 4: geo-easy diagnostic

Adjudicates whether the 10 added stations' lack of embedding advantage is because
they are "geo-easy" (uninformative) or because the effect is genuinely fragile.
Per-fold detrend, leave-one-STATION-out over all 38; per station: geo_residual_magnitude
(how much geography misses) and emb_advantage (geo_res − emb_res).

**Geo-easy hypothesis rejected.** Added geo-residual median **0.01404** vs core
**0.01477** — essentially identical; Mann-Whitney two-sided **p=0.70** (one-sided
added<core p=0.35). The added stations are not easier for geography; geography
misses by the same amount in both groups.

At equal geo-difficulty, the embedding helps the core but not the added stations:
emb_advantage median = **+0.00325 (core)** vs **+0.00001 (added, ≈ zero)**.

**Verdict: genuinely fragile** — comparable geo residual, no embedding advantage on
independent stations. The non-replication is real, not an artifact of easy sites.

**Qualification.** Across all 38, Spearman ρ(geo_residual, emb_advantage) = **+0.35,
p=0.033** — the physically-sensible signature (embedding helps more where geography
misses more) is weakly present. But it is core-driven and not robust (Pearson r=+0.17,
p=0.30, n.s.), and the added stations specifically violate it: they sit at ~0
advantage even at large geo residual (Feldberg 01346 at residual 0.068 → advantage
only +0.0045).

**Conclusion:** the geo-easy explanation is rejected. The most consistent
interpretation is that the n=28 emb-over-geo advantage is genuinely fragile, partly a
favorable draw. A weak real effect is not excluded (the Spearman signature), but
confirming it requires a larger, more diverse sample (the cross-border n=50–80 path),
not the exhausted German DWD pool. Figure: data/figs/geo_easy_diagnostic.png.

## 2026-06-15 — Power analysis: kill-decision for the multi-network study

Decision instrument on the 38 pilot stations (locked per-fold-detrend spatial-block
pipeline). Nonparametric-bootstrap power for the rule *win-rate>60% AND Wilcoxon
one-sided p<0.01*, plus a usefulness bar (median emb_advantage ≥ 25% of the geo
baseline error).

**Effect size (Part A):** ALL-38 median advantage +0.00015 (win 0.66); CORE +0.00050
(win 0.79); ADDED -0.00090 (win 0.30). Geo baseline median error 0.01925, so the
25%-of-geo usefulness bar = **0.00481**.

**Power (Part B) and usefulness (Part C):**

| n | opt power | pess power | opt CI-low | pess CI-low |
|---|---|---|---|---|
| 20 | 0.16 | 0.00 | -0.00006 | -0.00361 |
| 30 | 0.25 | 0.00 | -0.00001 | -0.00355 |
| 40 | 0.37 | 0.00 | +0.00002 | -0.00355 |
| 60 | 0.57 | 0.00 | +0.00005 | -0.00355 |
| 80 | 0.70 | 0.00 | +0.00006 | -0.00263 |
| 120 | 0.85 | 0.00 | +0.00006 | -0.00171 |
| 160 | 0.92 | 0.00 | +0.00007 | -0.00171 |

- Optimistic power reaches 0.8 at n=120; pessimistic never reaches it.
- The optimistic CI-low never clears the 25%-of-geo bar; the pessimistic CI-low never
  clears zero.

**Verdict: WORLD 2 — scope down.** The effect is at best significant-but-not-useful:
optimistic 0.8 power at n=120, pessimistic never; the emb_advantage CI lower bound
never clears the 25%-of-geo usefulness bar (0.00481). Detectable with enough n, but
not lead-generating. The pessimistic curve (new stations resembling the weaker
added-10 sample) is the realistic one given the replication failure. Figure:
data/figs/power_curves.png.

**Reconciliation with PREREGISTRATION.md:**

- This instrument applied the decision-rule thresholds (win>60% AND p<0.01) to the
  emb-vs-geo contrast (emb_advantage = |geo_err|−|emb_err|), per the scripts/13 task.
  The pre-registered primary rule applies those thresholds to emb-vs-shuffle;
  emb-vs-geo is the secondary p<0.05 check. Beating shuffle ("contains information")
  is easier than beating geo ("beyond coordinates"); a shuffle-contrast power curve
  would look more favorable. The usefulness conclusion is intrinsically a
  beyond-coordinates question, so it is unaffected and is measured on emb-vs-geo.
- WORLD 2 ("scope down") maps onto the pre-registration's own contingency: if the
  effect is too small to detect at achievable n, the work becomes a limitation study.
  This verdict therefore triggers a planned change of scope.
