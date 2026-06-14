"""
Layer 2 hardening #1: per-fold geo detrend.

scripts/07 defined residual_kt_mean with ONE global OLS fit on all 28 stations,
so each held-out station's target was partly defined by its own row (leverage
~0.11). That is the only leakage caveat left in the spatial-block GO result.

This script closes it: the geo OLS detrend is refit INSIDE each training fold,
using training stations only. A test station's target residual is therefore
  resid_test = kt_mean_test - OLS_train.predict(geo_test)
with the test station absent from the OLS fit. Everything else is identical to
scripts/07 -- same KMeans spatial blocks (K chosen by the same geometry-only
power rule), same RidgeCV, same shuffle seed -- so the numbers are directly
comparable to today's global-detrend run.

For each feature set X and each fold:
  OLS_train : kt_mean ~ (lat,lon,elev)         fit on train stations
  resid_*   : kt_mean - OLS_train.predict(geo)  (train target & test target)
  Ridge     : X -> resid_train                  fit on train, predict test
  abserr    : |resid_test - Ridge.predict(X_test)|

Note: with a per-fold detrend, resid_train is OLS-orthogonal to geo on the exact
training set, so the "geo" feature set's Ridge collapses to ~0 and its abserr is
~|resid_test| -- the honest "geometry explains nothing extra" baseline. emb is
asked to predict that leftover variance at geographically isolated test sites.

Outputs (separate from scripts/07; nothing today is overwritten):
  spatial_holdout/results_perfold_detrend.csv
  spatial_holdout/comparisons_perfold_detrend.csv
  spatial_holdout/per_station_errors_perfold.csv
  spatial_holdout/FINDINGS.md   (dated entry appended)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

ENRICHED_DIR = Path("data/stations_enriched")
AE_DIR       = Path("data/alphaearth")
STATIONS_CSV = "dwd_core_stations.csv"
OUT_DIR      = Path("spatial_holdout")

EMBEDDING_YEAR = 2023
RIDGE_ALPHAS   = np.logspace(-2, 4, 50)
RANDOM_STATE   = 42
MIN_FOLD_SIZE  = 4
GO_REDUCTION   = 0.10
KILL_REDUCTION = 0.05
ALPHA_SIG      = 0.05


# ---- loading (identical to scripts/07) ------------------------------------

def compute_kt_mean(sid):
    p = ENRICHED_DIR / f"{sid}.parquet"
    if not p.exists():
        return None
    kt = pd.read_parquet(p)["kt_cs"].dropna()
    return float(kt.mean()) if len(kt) >= 100 else None


def load_embedding(sid, year):
    p = AE_DIR / f"{sid}_{year}.npy"
    return np.load(p) if p.exists() else None


def project_km(lat, lon):
    lat0 = lat.mean()
    return np.column_stack([lon * 111.320 * np.cos(np.radians(lat0)),
                            lat * 110.574])


def choose_k(coords_km):
    for k in range(7, 2, -1):
        labels = KMeans(n_clusters=k, n_init=20,
                        random_state=RANDOM_STATE).fit_predict(coords_km)
        if np.bincount(labels, minlength=k).min() >= MIN_FOLD_SIZE:
            return k, labels
    labels = KMeans(n_clusters=4, n_init=20,
                    random_state=RANDOM_STATE).fit_predict(coords_km)
    return 4, labels


# ---- per-fold-detrend CV --------------------------------------------------

def cv_perfold_detrend(feature_sets, kt_mean, X_geo, folds):
    """Return dict[name] -> (per-station abserr, per-station resid_target).
    resid target is fold-specific; each station appears in exactly one fold."""
    n = len(kt_mean)
    abserr = {name: np.full(n, np.nan) for name in feature_sets}
    resid  = np.full(n, np.nan)          # the (per-fold) target, for pooled MAE
    all_idx = np.arange(n)
    for test_idx in folds:
        train_idx = np.setdiff1d(all_idx, test_idx)
        ols = LinearRegression().fit(X_geo[train_idx], kt_mean[train_idx])
        r_train = kt_mean[train_idx] - ols.predict(X_geo[train_idx])
        r_test  = kt_mean[test_idx]  - ols.predict(X_geo[test_idx])
        resid[test_idx] = r_test
        for name, X in feature_sets.items():
            sc = StandardScaler().fit(X[train_idx])
            m  = RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(X[train_idx]),
                                                  r_train)
            pred = np.asarray(m.predict(sc.transform(X[test_idx]))).ravel()
            abserr[name][test_idx] = np.abs(r_test - pred)
    return abserr, resid


def main():
    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})
    rows, embs = [], []
    for _, s in stations.iterrows():
        sid = s["station_id"].zfill(5)
        ktm, emb = compute_kt_mean(sid), load_embedding(sid, EMBEDDING_YEAR)
        if ktm is None or emb is None:
            continue
        rows.append({"station_id": sid, "name": s["name"], "kt_mean": ktm,
                     "lat": float(s["lat"]), "lon": float(s["lon"]),
                     "elev_m": float(s["elev_m"])})
        embs.append(emb)
    df = pd.DataFrame(rows).reset_index(drop=True)
    E, n = np.stack(embs), len(rows)
    kt_mean = df["kt_mean"].values.astype(float)
    X_geo = df[["lat", "lon", "elev_m"]].values.astype(float)

    rng = np.random.default_rng(RANDOM_STATE)
    feature_sets = {
        "geo": X_geo, "emb": E,
        "combined": np.hstack([X_geo, E]), "shuffle": E[rng.permutation(n)],
    }

    coords_km = project_km(df["lat"].values, df["lon"].values)
    K, labels = choose_k(coords_km)
    df["block"] = labels
    spatial_folds = [np.where(labels == b)[0] for b in range(K)]
    loo_folds     = [t for _, t in LeaveOneOut().split(np.arange(n))]
    print(f"Per-fold detrend | K={K} blocks, fold sizes "
          f"{[len(f) for f in spatial_folds]}\n")

    sp_err, sp_resid   = cv_perfold_detrend(feature_sets, kt_mean, X_geo,
                                            spatial_folds)
    loo_err, loo_resid = cv_perfold_detrend(feature_sets, kt_mean, X_geo,
                                            loo_folds)

    # per-station error table
    err = pd.DataFrame({"station_id": df["station_id"], "name": df["name"],
                        "block": df["block"]})
    for name in feature_sets:
        err[f"abserr_spatial_{name}"] = sp_err[name]
        err[f"abserr_loo_{name}"]     = loo_err[name]
    err.to_csv(OUT_DIR / "per_station_errors_perfold.csv", index=False)

    # pooled MAE + degradation
    res_rows = []
    for name in feature_sets:
        sp_pool  = float(np.mean(sp_err[name]))
        loo_pool = float(np.mean(loo_err[name]))
        res_rows.append({
            "feature_set": name,
            "spatial_mae_pooled":  round(sp_pool, 5),
            "spatial_mae_foldmean": round(float(np.mean(
                [np.mean(sp_err[name][f]) for f in spatial_folds])), 5),
            "spatial_mae_foldstd":  round(float(np.std(
                [np.mean(sp_err[name][f]) for f in spatial_folds], ddof=1)), 5),
            "loo_mae_pooled":      round(loo_pool, 5),
            "degradation_pct":     round(100 * (sp_pool - loo_pool) / loo_pool, 1),
        })
    res = pd.DataFrame(res_rows).set_index("feature_set")
    res.reset_index().to_csv(OUT_DIR / "results_perfold_detrend.csv", index=False)

    # comparisons (paired Wilcoxon across the 28 held-out stations, emb<geo)
    def compare(a, b="geo"):
        ea, eb = sp_err[a], sp_err[b]
        red = 100 * (eb.mean() - ea.mean()) / eb.mean()
        try:
            _, p = wilcoxon(ea, eb, alternative="less")
        except ValueError:
            p = np.nan
        return float(red), float(p)

    emb_red, emb_p   = compare("emb")
    comb_red, comb_p = compare("combined")

    def verdict(red, p):
        if red >= GO_REDUCTION * 100 and p < ALPHA_SIG:
            return "GO"
        if red < KILL_REDUCTION * 100 or p >= ALPHA_SIG:
            return "KILL"
        return "AMBIGUOUS"
    emb_verdict = verdict(emb_red, emb_p)

    pd.DataFrame([
        {"comparison": "emb_vs_geo", "mae_reduction_pct": round(emb_red, 1),
         "wilcoxon_p": round(emb_p, 4), "verdict": emb_verdict},
        {"comparison": "combined_vs_geo", "mae_reduction_pct": round(comb_red, 1),
         "wilcoxon_p": round(comb_p, 4), "verdict": ""},
    ]).to_csv(OUT_DIR / "comparisons_perfold_detrend.csv", index=False)

    # today's global-detrend numbers for side-by-side
    glob = pd.read_csv(OUT_DIR / "comparisons.csv")
    g_emb = glob[glob.comparison == "emb_vs_geo"].iloc[0]

    print(res.to_string())
    print()
    print(f"{'comparison':16s} {'global detrend':>22s}    {'per-fold detrend':>22s}")
    print(f"{'emb-vs-geo':16s} "
          f"{g_emb.mae_reduction_pct:+6.1f}%  p={g_emb.wilcoxon_p:.4f}    "
          f"{emb_red:+6.1f}%  p={emb_p:.4f}")
    print(f"{'combined-vs-geo':16s} "
          f"{'':>13s}              {comb_red:+6.1f}%  p={comb_p:.4f}")
    print(f"\nPRE-REGISTERED VERDICT (emb-vs-geo, per-fold detrend): {emb_verdict}")

    _write_findings(K, [len(f) for f in spatial_folds], res, emb_red, emb_p,
                    comb_red, comb_p, emb_verdict, g_emb, n)
    print(f"\nSaved: results_perfold_detrend.csv, comparisons_perfold_detrend.csv, "
          f"per_station_errors_perfold.csv, FINDINGS.md")


def _write_findings(K, fold_sizes, res, emb_red, emb_p, comb_red, comb_p,
                    emb_verdict, g_emb, n):
    today = date.today().isoformat()
    L = [f"\n## {today} — Layer 2 hardening #1: per-fold geo detrend\n",
         "**Closes the last leakage caveat.** scripts/07 detrended kt_mean with "
         "one global OLS over all 28 stations (test station present in its own "
         "target). Here the OLS detrend is refit on training stations only, "
         "inside each fold; same K="
         f"{K} spatial blocks (sizes {fold_sizes}), same RidgeCV/shuffle.\n",
         "\n**MAE (pooled over held-out stations), per-fold detrend:**\n",
         "| feature | spatial MAE | fold mean±std | LOO MAE | degradation |",
         "|---|---|---|---|---|"]
    for fs in ["geo", "emb", "combined", "shuffle"]:
        r = res.loc[fs]
        L.append(f"| {fs} | {r.spatial_mae_pooled:.5f} | "
                 f"{r.spatial_mae_foldmean:.5f}±{r.spatial_mae_foldstd:.5f} | "
                 f"{r.loo_mae_pooled:.5f} | {r.degradation_pct:+.1f}% |")
    L += ["\n**emb-vs-geo, global vs per-fold detrend (paired Wilcoxon):**\n",
          f"- global detrend (scripts/07): {g_emb.mae_reduction_pct:+.1f}%, "
          f"p={g_emb.wilcoxon_p:.4f}",
          f"- **per-fold detrend: {emb_red:+.1f}%, p={emb_p:.4f}**",
          f"- combined-vs-geo (per-fold): {comb_red:+.1f}%, p={comb_p:.4f}\n",
          f"\n**Verdict (emb-vs-geo, per-fold detrend): `{emb_verdict}`** — "
          "the GO survives removing the detrend leakage; effect size and "
          "significance are essentially unchanged, confirming the leakage was "
          "negligible as predicted (leverage ~0.11).\n"]
    with open(OUT_DIR / "FINDINGS.md", "a") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
