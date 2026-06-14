"""
Layer 2 robustness check: does AlphaEarth's advantage over geography survive
SPATIAL-BLOCK cross-validation?

Motivation
----------
The original Layer 2 (scripts/05) used leave-one-station-out (LOO) CV. With
only 28 stations spread over Germany, a held-out station almost always has a
training station ~50-100 km away. Spatial autocorrelation in the target then
leaks across the train/test split and can inflate the embedding's apparent
skill. The honest question for an "ungauged site" claim is: does emb still beat
geo when every test station is geographically isolated from all training
stations?

This script repeats ONLY the residual_kt_mean comparison from Layer 2, with the
same feature sets (geo / emb / combined / shuffle) and the same RidgeCV, but
swaps LOO for leave-one-spatial-block-out CV.

Design (locked before looking at results)
-----------------------------------------
  * Target          : residual_kt_mean  (kt_mean detrended on lat,lon,elev via
                      a single global OLS fit on all 28 stations -- identical to
                      Layer 2 so the numbers are directly comparable).
  * Feature sets    : geo (3-d), emb (64-d), combined (67-d), shuffle (64-d,
                      station rows permuted with seed 42 -- same as Layer 2).
  * Regressor       : RidgeCV(alphas=logspace(-2,4,50)), features z-scored
                      inside each training fold (no test leakage).
  * Spatial blocks  : stations projected to an approx. equal-area km grid, then
                      KMeans(K) -> contiguous geographic blocks; leave-one-block
                      -out. K chosen by a POWER rule, not by the result:
                      smallest number of folds that still keeps every fold at
                      >= MIN_FOLD_SIZE test stations, so blocks are as large
                      (= separation as strong) as the power budget allows.
  * Buffer reported : per fold, the min great-circle distance from any test
                      station to its nearest training station (the achieved
                      spatial separation; blocks are a hard partition with no
                      dead-zone, so boundary stations set this floor).

Verdict thresholds (pre-registered, emb-vs-geo on pooled MAE)
-------------------------------------------------------------
  GO        : MAE reduction >= 10%  AND  paired Wilcoxon p < 0.05
  KILL      : MAE reduction < 5%    OR   not significant (p >= 0.05)
  AMBIGUOUS : in between (5-10% and significant); combined-vs-geo is tiebreaker

Outputs (never touches existing Layer 2/3 files)
  spatial_holdout/results.csv
  spatial_holdout/per_station_errors.csv
  spatial_holdout/spatial_blocks.png
  spatial_holdout/FINDINGS.md   (dated entry appended)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import wilcoxon

ENRICHED_DIR = Path("data/stations_enriched")
AE_DIR       = Path("data/alphaearth")
STATIONS_CSV = "dwd_core_stations.csv"
OUT_DIR      = Path("spatial_holdout")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_YEAR = 2023          # same single-year embedding as Layer 2
RIDGE_ALPHAS   = np.logspace(-2, 4, 50)
RANDOM_STATE   = 42
MIN_FOLD_SIZE  = 4             # power floor: flag any fold with fewer test stns

# Pre-registered verdict thresholds (emb-vs-geo, pooled MAE reduction)
GO_REDUCTION   = 0.10
KILL_REDUCTION = 0.05
ALPHA_SIG      = 0.05


# ---------------------------------------------------------------------------
# Data loading (mirrors scripts/05)
# ---------------------------------------------------------------------------

def compute_kt_mean(sid: str) -> float | None:
    p = ENRICHED_DIR / f"{sid}.parquet"
    if not p.exists():
        return None
    kt = pd.read_parquet(p)["kt_cs"].dropna()
    if len(kt) < 100:
        return None
    return float(kt.mean())


def load_embedding(sid: str, year: int) -> np.ndarray | None:
    p = AE_DIR / f"{sid}_{year}.npy"
    return np.load(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def project_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Approx. equal-area local projection (km) so KMeans sees isotropic space."""
    lat0 = lat.mean()
    x = lon * 111.320 * np.cos(np.radians(lat0))
    y = lat * 110.574
    return np.column_stack([x, y])


def choose_k(coords_km: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    """Pick the SMALLEST fold count (largest = best-separated blocks) such that
    every block still has >= MIN_FOLD_SIZE stations. Decision uses geometry
    only -- never the regression result -- so it can't bias the verdict."""
    chosen_k, chosen_labels = None, None
    for k in range(7, 2, -1):   # try few folds first (big blocks), step down...
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(coords_km)
        sizes = np.bincount(labels, minlength=k)
        if sizes.min() >= MIN_FOLD_SIZE:
            chosen_k, chosen_labels = k, labels
            break
    if chosen_k is None:        # fallback: fewest folds, accept imbalance
        km = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
        chosen_labels = km.fit_predict(coords_km)
        chosen_k = 4
    return chosen_k, chosen_labels


# ---------------------------------------------------------------------------
# CV engines -- both z-score features inside each training fold
# ---------------------------------------------------------------------------

def _ridge():
    return RidgeCV(alphas=RIDGE_ALPHAS)


def oof_predict(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """Out-of-fold predictions given an explicit list of test-index arrays."""
    preds = np.full(len(y), np.nan)
    all_idx = np.arange(len(y))
    for test_idx in folds:
        train_idx = np.setdiff1d(all_idx, test_idx)
        scaler = StandardScaler().fit(X[train_idx])
        m = _ridge().fit(scaler.transform(X[train_idx]), y[train_idx])
        preds[test_idx] = np.asarray(
            m.predict(scaler.transform(X[test_idx]))).ravel()
    return preds


def loo_oof_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    folds = [test_idx for _, test_idx in LeaveOneOut().split(X)]
    return oof_predict(X, y, folds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})

    rows, embs = [], []
    for _, s in stations.iterrows():
        sid = s["station_id"].zfill(5)
        ktm = compute_kt_mean(sid)
        emb = load_embedding(sid, EMBEDDING_YEAR)
        if ktm is None or emb is None:
            print(f"  [{sid}] skipped (missing parquet or embedding)")
            continue
        rows.append({"station_id": sid, "name": s["name"],
                     "kt_mean": ktm, "lat": float(s["lat"]),
                     "lon": float(s["lon"]), "elev_m": float(s["elev_m"])})
        embs.append(emb)

    df = pd.DataFrame(rows).reset_index(drop=True)
    E  = np.stack(embs)
    n  = len(df)
    print(f"Loaded {n} stations.\n")

    # --- target: geo-detrended kt_mean (global OLS, identical to Layer 2) ----
    X_geo = df[["lat", "lon", "elev_m"]].values.astype(float)
    ols   = LinearRegression().fit(X_geo, df["kt_mean"].values)
    y     = df["kt_mean"].values - ols.predict(X_geo)   # residual_kt_mean

    rng       = np.random.default_rng(RANDOM_STATE)
    X_emb     = E
    X_comb    = np.hstack([X_geo, E])
    X_shuf    = E[rng.permutation(n)]
    feature_sets = {"geo": X_geo, "emb": X_emb,
                    "combined": X_comb, "shuffle": X_shuf}

    # --- build spatial blocks (geometry only) -------------------------------
    coords_km = project_km(df["lat"].values, df["lon"].values)
    K, labels = choose_k(coords_km, n)
    df["block"] = labels
    folds = [np.where(labels == b)[0] for b in range(K)]
    fold_sizes = [len(f) for f in folds]

    # achieved buffer: per fold, min test->nearest-train great-circle distance
    fold_buffers = []
    for b, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        dmins = []
        for ti in test_idx:
            d = [haversine_km(df.lat[ti], df.lon[ti], df.lat[tr], df.lon[tr])
                 for tr in train_idx]
            dmins.append(min(d))
        fold_buffers.append(min(dmins))   # worst-case (closest) separation

    # approx block linear size (km) from projected coords spread
    block_extent = np.sqrt((np.ptp(coords_km[:, 0]) * np.ptp(coords_km[:, 1])) / K)

    print(f"{'='*70}")
    print(f"Spatial blocking: K={K} blocks (chosen by power rule, min fold "
          f">= {MIN_FOLD_SIZE})")
    print(f"  approx block linear size : ~{block_extent:.0f} km")
    print(f"  fold sizes (test stns)   : {fold_sizes}")
    print(f"  min test->train buffer/fold (km): "
          f"{[round(b,0) for b in fold_buffers]}")
    underpowered = [b for b, sz in enumerate(fold_sizes) if sz < MIN_FOLD_SIZE]
    if underpowered:
        print(f"  !! UNDERPOWERED folds (<{MIN_FOLD_SIZE} test stns): "
              f"{underpowered}")
    print(f"{'='*70}\n")

    # --- run both CV schemes for every feature set --------------------------
    spatial_pred = {name: oof_predict(X, y, folds)
                    for name, X in feature_sets.items()}
    loo_pred     = {name: loo_oof_predict(X, y)
                    for name, X in feature_sets.items()}

    # per-station absolute errors (for Wilcoxon + scatter)
    err = pd.DataFrame({"station_id": df["station_id"], "name": df["name"],
                        "block": df["block"], "y": y})
    for name in feature_sets:
        err[f"abserr_spatial_{name}"] = np.abs(y - spatial_pred[name])
        err[f"abserr_loo_{name}"]     = np.abs(y - loo_pred[name])
    err.to_csv(OUT_DIR / "per_station_errors.csv", index=False)

    # --- aggregate -----------------------------------------------------------
    def pooled_mae(pred):       # comparable to Layer 2's mean_absolute_error
        return float(mean_absolute_error(y, pred))

    def perfold_mae(pred):      # mean +/- std across folds
        m = [mean_absolute_error(y[f], pred[f]) for f in folds]
        return float(np.mean(m)), float(np.std(m, ddof=1))

    results = []
    for name in feature_sets:
        sp_pool = pooled_mae(spatial_pred[name])
        sp_mean, sp_std = perfold_mae(spatial_pred[name])
        loo_pool = pooled_mae(loo_pred[name])
        degr = 100 * (sp_pool - loo_pool) / loo_pool
        results.append({
            "feature_set": name,
            "spatial_mae_pooled":   round(sp_pool, 5),
            "spatial_mae_foldmean": round(sp_mean, 5),
            "spatial_mae_foldstd":  round(sp_std, 5),
            "loo_mae_pooled":       round(loo_pool, 5),
            "degradation_pct":      round(degr, 1),
        })
    res = pd.DataFrame(results).set_index("feature_set")

    # --- comparisons: emb-vs-geo and combined-vs-geo ------------------------
    def compare(name_a, name_b="geo"):
        a_sp = err[f"abserr_spatial_{name_a}"].values
        b_sp = err[f"abserr_spatial_{name_b}"].values
        mae_a = a_sp.mean()
        mae_b = b_sp.mean()
        red = 100 * (mae_b - mae_a) / mae_b   # % MAE reduction vs geo
        # paired Wilcoxon across held-out stations (one-sided: a < b)
        try:
            stat, p = wilcoxon(a_sp, b_sp, alternative="less")
        except ValueError:
            stat, p = np.nan, np.nan
        return red, float(p)

    emb_red,  emb_p  = compare("emb")
    comb_red, comb_p = compare("combined")

    # --- verdict (pre-registered) -------------------------------------------
    def verdict(red, p):
        if red >= GO_REDUCTION * 100 and p < ALPHA_SIG:
            return "GO"
        if red < KILL_REDUCTION * 100 or p >= ALPHA_SIG:
            return "KILL"
        return "AMBIGUOUS"

    emb_verdict = verdict(emb_red, emb_p)

    # --- console report ------------------------------------------------------
    print("Spatial-block CV vs leave-one-out (residual_kt_mean, RidgeCV)")
    print(res.to_string())
    print()
    print(f"emb-vs-geo      : MAE reduction {emb_red:+.1f}%   "
          f"Wilcoxon p={emb_p:.4f}   [{emb_verdict}]")
    print(f"combined-vs-geo : MAE reduction {comb_red:+.1f}%   "
          f"Wilcoxon p={comb_p:.4f}")
    print()
    print(f"PRE-REGISTERED VERDICT (emb-vs-geo): {emb_verdict}")

    # --- save results.csv (tidy) --------------------------------------------
    res_out = res.reset_index()
    res_out.to_csv(OUT_DIR / "results.csv", index=False)

    summary = pd.DataFrame([
        {"comparison": "emb_vs_geo",      "mae_reduction_pct": round(emb_red, 1),
         "wilcoxon_p": round(emb_p, 4),   "verdict": emb_verdict},
        {"comparison": "combined_vs_geo", "mae_reduction_pct": round(comb_red, 1),
         "wilcoxon_p": round(comb_p, 4),  "verdict": ""},
    ])
    summary.to_csv(OUT_DIR / "comparisons.csv", index=False)

    # --- map of blocks -------------------------------------------------------
    _plot_blocks(df, K, fold_sizes, fold_buffers)

    # --- FINDINGS entry ------------------------------------------------------
    _write_findings(K, block_extent, fold_sizes, fold_buffers, underpowered,
                    res, emb_red, emb_p, comb_red, comb_p, emb_verdict, n)

    print(f"\nSaved: {OUT_DIR/'results.csv'}, comparisons.csv, "
          f"per_station_errors.csv, spatial_blocks.png, FINDINGS.md")


def _plot_blocks(df, K, fold_sizes, fold_buffers):
    fig, ax = plt.subplots(figsize=(7, 8))
    cmap = plt.get_cmap("tab10")
    for b in range(K):
        sub = df[df["block"] == b]
        ax.scatter(sub["lon"], sub["lat"], s=90, color=cmap(b),
                   edgecolors="k", linewidths=0.5,
                   label=f"block {b}: n={fold_sizes[b]}, "
                         f"buf~{fold_buffers[b]:.0f}km")
    for _, r in df.iterrows():
        ax.annotate(r["station_id"], (r["lon"], r["lat"]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title(f"Spatial-block CV partition (K={K})\n"
                 "leave-one-block-out; buf = min test->train great-circle dist")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_blocks.png", dpi=130)
    plt.close(fig)


def _write_findings(K, block_extent, fold_sizes, fold_buffers, underpowered,
                    res, emb_red, emb_p, comb_red, comb_p, emb_verdict, n):
    today = date.today().isoformat()
    r = res
    lines = []
    lines.append(f"\n## {today} — Layer 2 spatial-block CV (residual_kt_mean)\n")
    lines.append(
        f"**Question.** Does AlphaEarth's edge over geography survive spatial-"
        f"block CV (transfer to genuinely ungauged sites), or was the leave-"
        f"one-out (LOO) result riding spatial autocorrelation?\n")
    lines.append(
        f"**Design.** Same target/features/RidgeCV as Layer 2; LOO replaced by "
        f"leave-one-spatial-block-out. {n} stations KMeans-clustered in an "
        f"equal-area km projection into **K={K}** contiguous blocks "
        f"(~{block_extent:.0f} km each), K chosen by a geometry-only power rule "
        f"(largest blocks keeping every fold ≥ {MIN_FOLD_SIZE} test stations).\n")
    lines.append(
        f"**Fold sizes (test stations):** {fold_sizes}  \n"
        f"**Achieved buffer (min test→nearest-train, km):** "
        f"{[round(b) for b in fold_buffers]}  \n")
    if underpowered:
        lines.append(
            f"> ⚠️ Underpowered folds (<{MIN_FOLD_SIZE} test stns): "
            f"{underpowered}. Treat per-fold std with caution.\n")
    else:
        lines.append(
            f"> All folds ≥ {MIN_FOLD_SIZE} test stations.\n")

    lines.append("\n**MAE (pooled over held-out stations) — spatial vs LOO:**\n")
    lines.append("| feature | spatial MAE | fold mean±std | LOO MAE | degradation |")
    lines.append("|---|---|---|---|---|")
    for fs in ["geo", "emb", "combined", "shuffle"]:
        row = r.loc[fs]
        lines.append(
            f"| {fs} | {row.spatial_mae_pooled:.5f} | "
            f"{row.spatial_mae_foldmean:.5f}±{row.spatial_mae_foldstd:.5f} | "
            f"{row.loo_mae_pooled:.5f} | {row.degradation_pct:+.1f}% |")

    lines.append("\n**Comparisons (paired Wilcoxon across held-out stations):**\n")
    lines.append(
        f"- emb-vs-geo: **{emb_red:+.1f}%** MAE reduction, p={emb_p:.4f}\n"
        f"- combined-vs-geo: {comb_red:+.1f}% MAE reduction, p={comb_p:.4f}\n")
    lines.append(
        f"\n**Pre-registered verdict (emb-vs-geo): `{emb_verdict}`**  \n"
        f"(GO ≥10% & p<0.05; KILL <5% or p≥0.05; else AMBIGUOUS, "
        f"combined-vs-geo as tiebreaker.)\n")
    with open(OUT_DIR / "FINDINGS.md", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
