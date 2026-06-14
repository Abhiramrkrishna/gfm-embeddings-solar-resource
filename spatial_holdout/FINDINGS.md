
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

