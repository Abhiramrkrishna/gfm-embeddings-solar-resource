# Reproducibility Verification — AlphaEarth-vs-geo solar pilot

**Date:** 2026-06-15 · **Branch:** spatial-transfer · **Script:** `scripts/15_verify_all.py`

**Question:** is the negative result (WORLD 2 — AlphaEarth's per-station advantage
over geographic coordinates is small/near-zero on detrended mean clear-sky index)
*real*, or a pipeline artifact / reporting error?

**Verdict: CONFIRMED.** Every headline number re-derives from raw bytes via an
independent code path, all positive/negative controls behave as predicted, and
the conclusion is deterministic and seed-stable.

---

## What was checked

1. **Full chain from raw.** kt_cs recomputed for all 38 stations directly from the
   raw DWD zips (inline parse + pvlib Ineichen clear-sky), reading **no** cached
   enriched parquet and **no** results CSV. Station list reconstructed from
   `dwd_core_stations.csv` + the DWD metadata in `parse_stations.py` (not from any
   derived pooled CSV). Embeddings read from the raw `*_2023.npy`.
2. **Independent re-implementation** of the key numbers (median per-station
   emb_advantage; Wilcoxon p on emb-vs-shuffle): my from-scratch spatial-block
   per-fold-detrend loop vs the locked `scripts/08` path; scipy Wilcoxon vs a
   hand-written signed-rank (normal approx, continuity + tie correction).
3. **Positive/negative controls** that must behave as predicted or there is a bug.
4. **Determinism** (same seed twice) and **seed stability** (3 seeds).
5. **Environment + data checksums** captured to `data/repro/`.
6. **Final comparison** of re-derived numbers against the previously reported ones.

The from-raw recompute catches stale-cache and reporting errors; the controls
catch leakage/target-wiring bugs that faithful re-implementation alone could not.

## Results — re-derived from raw vs previously reported

| quantity | re-derived | reported | \|diff\| | |
|---|---|---|---|---|
| CORE-28 emb-vs-geo pooled MAE reduction | **+27.57%** | +27.6% | 0.03 | MATCH |
| ALL-38 emb-vs-geo pooled MAE reduction | **+13.69%** | +13.7% | 0.01 | MATCH |
| ALL-38 median per-station emb_advantage | **+0.00015** | +0.00015 | 0.0000 | MATCH |

## Independent cross-implementation

| key number | inline path | locked/2nd path | \|diff\| |
|---|---|---|---|
| median emb_advantage | +0.00015140 | +0.00015140 | **0.00e+00** |
| Wilcoxon p (emb>shuffle) | 0.20014 (scipy) | 0.20014 (hand) | **0.00e+00** |

Note: scipy's `wilcoxon` omits the continuity correction by default; the initial
run showed p=0.198 vs 0.200 purely from that. Comparing like-for-like
(`correction=True`) the two implementations agree to floating point.

## Controls (all PASS)

| control | expected | observed | result |
|---|---|---|---|
| noise embeddings (random normal) | advantage → ~0 | median_adv = +0.00003 | **PASS** |
| target-as-feature (leak) | emb error → ~0, advantage spikes | emb_err 0.00001 vs geo 0.01925; adv +0.01924 | **PASS** |
| label-shuffled target | all signal vanishes | median_adv −0.00008, p=0.842 (n.s.) | **PASS** |

## Determinism & seed stability

- Same seed twice → **identical** outputs.
- emb-vs-geo is **seed-free**: reduction constant at 13.69%, median advantage
  constant at 0.00015 across seeds {42, 0, 7}. **WORLD-2/negative verdict stable.**
- **Caveat surfaced:** the emb-vs-*shuffle* p is **highly seed-sensitive** —
  {0.2001, 0.2664, 0.0053} across the three seeds (spread 0.26), because it
  depends on which random permutation is drawn. The pre-registered shuffle-based
  decision rule (Plan §3) is therefore **fragile at n=38**; the emb-vs-geo
  contrast (the one that matters) is not. This reinforces the WORLD 2 call and is
  a reason to fix the shuffle seed / average over many permutations at scale.

## Environment & inputs

- python 3.12.7; numpy 2.4.4, pandas 3.0.3, scikit-learn 1.8.0, scipy 1.17.1,
  pvlib 0.15.1, earthengine-api 1.7.26.
- `data/repro/environment.txt` — full `pip freeze`.
- `data/repro/data_checksums.txt` — sha256 of all **76** raw inputs used
  (38 DWD zips + 38 AlphaEarth `*_2023.npy`). Anyone with the same hashes has the
  same inputs.

## Bottom line

The small/near-zero AlphaEarth-over-geography advantage **is real, not a pipeline
artifact or reporting error.** It reproduces exactly from raw data through an
independent implementation, passes all leakage/wiring controls, and is
deterministic. The negative (WORLD 2) result stands. Per `RESEARCH_PLAN.md`, this
supports the H4 limitation framing rather than the full multi-network build.
