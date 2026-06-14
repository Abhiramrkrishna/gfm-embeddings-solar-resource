"""
Layer 2 regression: Can AlphaEarth embeddings predict per-station kt_cs
statistics better than geographic coordinates?

Setup: 28 DWD stations, leave-one-station-out CV (28-fold), 4 feature sets,
       2 regressors.

Feature sets
------------
  geo      : (lat, lon, elev_m)              3-d geographic baseline
  emb      : AlphaEarth 2023 embedding       64-d
  combined : geo ++ emb                      67-d
  shuffle  : embedding with station rows     64-d null model (destroys
             randomly permuted                   geo-embedding correspondence)

Regressors
----------
  Ridge : RidgeCV(alphas=logspace(-2,4,50)), cv=None → efficient LOO alpha
          selection on training fold; features z-scored per fold
  PLS   : PLSRegression(n_components=8), n_components capped at
          min(8, n_features, n_train-1); features z-scored per fold

Targets (per-station statistics, 2020-2024)
-------------------------------------------
  kt_mean              : mean clear-sky index
  kt_std               : std of kt_cs
  kt_p95               : 95th percentile of kt_cs
  over_irr_frac        : fraction of valid daytime hours with kt_cs > 1.0
  residual_kt_mean     : kt_mean after OLS detrending on (lat,lon,elev)
  residual_over_irr_frac: over_irr_frac after OLS detrending on (lat,lon,elev)

Note on residual targets: OLS is fit on all 28 stations (in-sample residuals).
Each station's residual therefore includes a tiny contribution from its own
point to the OLS fit, but with only 3 OLS parameters and n=28 the leverage
per station is ~0.11, so the leakage is negligible.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

ENRICHED_DIR = Path("data/stations_enriched")
AE_DIR       = Path("data/alphaearth")
STATIONS_CSV = "dwd_core_stations.csv"
OUT_DIR      = Path("data")
FIGS_DIR     = Path("data/figs")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_YEAR = 2023
PLS_COMPONENTS = 8
RANDOM_STATE   = 42
RIDGE_ALPHAS   = np.logspace(-2, 4, 50)

TARGETS = [
    "kt_mean",
    "kt_std",
    "kt_p95",
    "over_irr_frac",
    "residual_kt_mean",
    "residual_over_irr_frac",
]

FEATURE_COLORS = {
    "geo":      "#4878d0",
    "emb":      "#ee854a",
    "combined": "#6acc65",
    "shuffle":  "#d65f5f",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def compute_station_stats(sid: str) -> dict | None:
    p = ENRICHED_DIR / f"{sid}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    kt = df["kt_cs"].dropna()
    if len(kt) < 100:
        return None
    return {
        "station_id":    sid,
        "kt_mean":       float(kt.mean()),
        "kt_std":        float(kt.std()),
        "kt_p95":        float(kt.quantile(0.95)),
        "over_irr_frac": float((kt > 1.0).mean()),
    }


def load_embedding(sid: str, year: int) -> np.ndarray | None:
    p = AE_DIR / f"{sid}_{year}.npy"
    return np.load(p) if p.exists() else None


# ---------------------------------------------------------------------------
# LOO helpers — both scale features inside each fold to avoid data leakage
# ---------------------------------------------------------------------------

def _loo_predict(X: np.ndarray, y: np.ndarray, make_model) -> np.ndarray:
    """Generic outer-LOO loop. make_model(n_features) → fitted sklearn estimator."""
    preds = np.empty(len(y))
    for train_idx, test_idx in LeaveOneOut().split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        m = make_model(X_tr.shape[1])
        m.fit(X_tr, y[train_idx])
        preds[test_idx] = np.asarray(m.predict(X_te)).ravel()[0]
    return preds


def loo_mae_ridge(X: np.ndarray, y: np.ndarray) -> float:
    # cv=None → sklearn uses efficient generalized LOO (GCV) to pick alpha
    preds = _loo_predict(X, y, lambda _: RidgeCV(alphas=RIDGE_ALPHAS))
    return float(mean_absolute_error(y, preds))


def loo_mae_pls(X: np.ndarray, y: np.ndarray) -> float:
    def make_pls(n_feat: int) -> PLSRegression:
        # n_components must be ≤ min(n_train_samples, n_features)
        n_comp = min(PLS_COMPONENTS, n_feat, len(y) - 2)
        return PLSRegression(n_components=n_comp)
    preds = _loo_predict(X, y, make_pls)
    return float(mean_absolute_error(y, preds))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})

    stat_rows, embeddings = [], []
    for _, s in stations.iterrows():
        sid  = s["station_id"].zfill(5)
        stats = compute_station_stats(sid)
        emb   = load_embedding(sid, EMBEDDING_YEAR)
        if stats is None or emb is None:
            print(f"  [{sid}] skipped — missing parquet or embedding")
            continue
        stats["lat"]    = float(s["lat"])
        stats["lon"]    = float(s["lon"])
        stats["elev_m"] = float(s["elev_m"])
        stat_rows.append(stats)
        embeddings.append(emb)

    df_stats = pd.DataFrame(stat_rows).reset_index(drop=True)
    E = np.stack(embeddings)          # (n, 64)
    n = len(df_stats)
    print(f"Loaded {n} stations. Running {n}-fold LOO...\n")

    # --- feature matrices ---------------------------------------------------
    X_geo      = df_stats[["lat", "lon", "elev_m"]].values.astype(float)
    X_emb      = E
    X_combined = np.hstack([X_geo, E])
    rng        = np.random.default_rng(RANDOM_STATE)
    X_shuffle  = E[rng.permutation(n)]

    # --- geo-detrended residual targets -------------------------------------
    for base in ("kt_mean", "over_irr_frac"):
        y_base = df_stats[base].values.astype(float)
        ols    = LinearRegression().fit(X_geo, y_base)
        df_stats[f"residual_{base}"] = y_base - ols.predict(X_geo)

    feature_sets = [
        ("geo",      X_geo),
        ("emb",      X_emb),
        ("combined", X_combined),
        ("shuffle",  X_shuffle),
    ]
    regressors = [
        ("ridge", loo_mae_ridge),
        ("pls",   loo_mae_pls),
    ]

    # --- run regressions ----------------------------------------------------
    results = []
    for target in TARGETS:
        y       = df_stats[target].values.astype(float)
        y_range = y.max() - y.min()
        row     = {"target": target, "y_range": round(y_range, 5)}
        print(f"{'─'*68}")
        print(f"  {target}   range=[{y.min():.5f}, {y.max():.5f}]")
        for feat_name, X in feature_sets:
            for reg_name, reg_fn in regressors:
                mae = reg_fn(X, y)
                col = f"{feat_name}_{reg_name}"
                row[col] = round(mae, 5)
                pct = 100 * mae / y_range if y_range > 0 else float("nan")
                print(f"    {feat_name:10s} {reg_name:6s}  MAE={mae:.5f}  ({pct:.1f}%)")
        results.append(row)

    rep = pd.DataFrame(results)
    out_csv = OUT_DIR / "layer2_results_v2.csv"
    rep.to_csv(out_csv, index=False)

    print(f"\n{'='*68}")
    print("Layer 2 v2 — LOO MAE (28 stations, Ridge / PLS)")
    print(f"{'='*68}")
    print(rep.to_string(index=False))
    print(f"\nSaved: {out_csv}")

    # --- bar plot -----------------------------------------------------------
    feat_names = ["geo", "emb", "combined", "shuffle"]
    reg_names  = ["ridge", "pls"]
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows))
    axes = axes.flatten()

    x     = np.arange(len(feat_names))
    width = 0.38
    offsets = {"ridge": -width / 2, "pls": +width / 2}
    hatches = {"ridge": "",     "pls": "///"}
    alphas  = {"ridge": 0.90,   "pls": 0.65}

    for i_target, (ax, row) in enumerate(zip(axes, results)):
        target = row["target"]
        y_max_bar = 0.0

        for reg_name in reg_names:
            maes = [row[f"{fn}_{reg_name}"] for fn in feat_names]
            y_max_bar = max(y_max_bar, max(maes))
            ax.bar(
                x + offsets[reg_name],
                maes,
                width=width,
                color=[FEATURE_COLORS[fn] for fn in feat_names],
                hatch=hatches[reg_name],
                edgecolor="white",
                linewidth=0.4,
                alpha=alphas[reg_name],
            )

        # Dashed reference at mean shuffle MAE across both regressors
        shuf_ref = (row["shuffle_ridge"] + row["shuffle_pls"]) / 2
        ax.axhline(shuf_ref, color="#d65f5f", lw=0.9, ls="--", alpha=0.55,
                   zorder=0)

        ax.set_title(target, fontsize=9, fontweight="bold")
        ax.set_ylabel("LOO MAE", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(feat_names, fontsize=8, rotation=20, ha="right")
        ax.tick_params(axis="y", labelsize=7)

        # Legend on first panel only
        if i_target == 0:
            feat_patches = [
                mpatches.Patch(fc=FEATURE_COLORS[fn], label=fn)
                for fn in feat_names
            ]
            reg_patches = [
                mpatches.Patch(fc="gray", alpha=0.9, hatch="",
                               label="Ridge (solid)"),
                mpatches.Patch(fc="gray", alpha=0.65, hatch="///",
                               label="PLS (hatched)"),
            ]
            ax.legend(handles=feat_patches + reg_patches,
                      fontsize=6.5, ncol=2, loc="upper right")

    for ax in axes[len(results):]:
        ax.set_axis_off()

    fig.suptitle(
        "Layer 2 v2: LOO MAE across 28 stations — Ridge (solid) vs PLS (hatched)\n"
        "dashed red = mean shuffle null;  lower is better",
        fontsize=10,
    )
    fig.tight_layout()
    out_fig = FIGS_DIR / "layer2_bars_v2.png"
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
