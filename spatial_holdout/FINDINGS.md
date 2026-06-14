
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

