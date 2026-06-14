"""
Layer 3: hourly kt_cs prediction — does adding AlphaEarth embeddings to
time-only features reduce residual error?

CV: 7-fold leave-station-out, stratified by elevation quartile.
Models
------
  TIME_ONLY     : hour/doy sin-cos, solar angles, lat/lon/elev (9-d)
  EMB_ONLY      : AlphaEarth 2020-2024 embeddings (64-d, year-matched)
  TIME_PLUS_EMB : time ++ embedding (73-d)
  SHUFFLE_EMB   : time ++ shuffled embedding (73-d null model)

Backend: LightGBM, 500 trees, early stopping on 10 % val split.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENRICHED_DIR = Path("data/stations_enriched")
AE_DIR       = Path("data/alphaearth")
STATIONS_CSV = "dwd_core_stations.csv"
OUT_DIR      = Path("data")
FIGS_DIR     = Path("data/figs")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS      = 7
N_TREES      = 500
EARLY_STOP   = 50
VAL_FRAC     = 0.10
RANDOM_STATE = 42

TIME_FEATS = [
    "hour_sin", "hour_cos",
    "doy_sin",  "doy_cos",
    "sun_zenith", "sun_apparent_elevation",
    "lat", "lon", "elev_m",
]
EMB_COLS  = [f"emb_{i:02d}" for i in range(64)]
SHUF_COLS = [f"shuf_{i:02d}" for i in range(64)]

MODELS = {
    "TIME_ONLY":     TIME_FEATS,
    "EMB_ONLY":      EMB_COLS,
    "TIME_PLUS_EMB": TIME_FEATS + EMB_COLS,
    "SHUFFLE_EMB":   TIME_FEATS + SHUF_COLS,
}

# Maps model name → prediction column in the output parquet
PRED_COL = {
    "TIME_ONLY":     "y_hat_time_only",
    "EMB_ONLY":      "y_hat_emb_only",
    "TIME_PLUS_EMB": "y_hat_time_plus_emb",
    "SHUFFLE_EMB":   "y_hat_shuffle",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embedding_lookup(sids: list[str], shuf_map: dict[str, str]) -> pd.DataFrame:
    """Build a (station_id, year, emb_00..63, shuf_00..63) lookup DataFrame.
    Using a join is cleaner and avoids row-by-row .loc assignment."""
    rows = []
    for sid in sids:
        shuf_sid = shuf_map[sid]
        for year in range(2020, 2025):
            real_path  = AE_DIR / f"{sid}_{year}.npy"
            shuf_path  = AE_DIR / f"{shuf_sid}_{year}.npy"
            if not real_path.exists() or not shuf_path.exists():
                continue
            real_emb = np.load(real_path)
            shuf_emb = np.load(shuf_path)
            row = {"station_id": sid, "year": year}
            for i in range(64):
                row[EMB_COLS[i]]  = float(real_emb[i])
                row[SHUF_COLS[i]] = float(shuf_emb[i])
            rows.append(row)
    return pd.DataFrame(rows)


def build_master_df(stations: pd.DataFrame, emb_lookup: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, s in stations.iterrows():
        sid = s["station_id"].zfill(5)
        p   = ENRICHED_DIR / f"{sid}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        mask = df["is_daytime"] & df["kt_cs"].notna()
        df   = df.loc[mask, ["timestamp_utc", "kt_cs",
                              "sun_zenith", "sun_apparent_elevation"]].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["year"]     = ts.dt.year
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["doy_sin"]  = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
        df["doy_cos"]  = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
        df["lat"]      = float(s["lat"])
        df["lon"]      = float(s["lon"])
        df["elev_m"]   = float(s["elev_m"])
        df["station_id"] = sid
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    # Join year-matched embeddings
    full = full.merge(emb_lookup, on=["station_id", "year"], how="inner")
    return full


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _make_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )


def _fit_predict(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, int]:
    model = _make_lgbm()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model.predict(X_te), int(model.best_iteration_)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})
    sids = [s["station_id"].zfill(5) for _, s in stations.iterrows()]

    # Fixed station-level shuffle: station A gets station perm[A]'s embedding
    perm      = rng.permutation(len(sids))
    shuf_map  = {sids[i]: sids[perm[i]] for i in range(len(sids))}

    print("Building embedding lookup…")
    emb_lookup = load_embedding_lookup(sids, shuf_map)

    print("Building master DataFrame…")
    df = build_master_df(stations, emb_lookup)
    n_stations = df["station_id"].nunique()
    print(f"  {len(df):,} daytime rows  ×  {n_stations} stations  "
          f"({time.time()-t_start:.1f}s)")

    # --- Station metadata for CV stratification ---
    meta = (
        stations.assign(sid=lambda d: d["station_id"].str.zfill(5))
        .set_index("sid")[["elev_m"]]
        .loc[lambda d: d.index.isin(df["station_id"].unique())]
        .copy()
    )
    # qcut with duplicates="drop" in case of ties at quartile boundaries
    meta["elev_q"] = pd.qcut(meta["elev_m"], q=4, labels=False,
                              duplicates="drop")
    sid_arr = meta.index.values
    q_arr   = meta["elev_q"].values.astype(int)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    fold_metrics = []
    pred_frames  = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(sid_arr, q_arr)):
        train_sids = set(sid_arr[tr_idx])
        test_sids  = set(sid_arr[te_idx])

        df_train = df[df["station_id"].isin(train_sids)]
        df_test  = df[df["station_id"].isin(test_sids)]

        print(f"\nFold {fold_idx+1}/{N_FOLDS} — "
              f"test: {sorted(test_sids)} ({len(df_test):,} rows)")

        # 10 % random val split from training rows (for early stopping only)
        n_train  = len(df_train)
        val_idx  = rng.choice(n_train, size=int(VAL_FRAC * n_train),
                              replace=False)
        val_mask = np.zeros(n_train, dtype=bool)
        val_mask[val_idx] = True

        df_tr = df_train.iloc[~val_mask]
        df_va = df_train.iloc[ val_mask]

        y_tr = df_tr["kt_cs"].values.astype(np.float32)
        y_va = df_va["kt_cs"].values.astype(np.float32)
        y_te = df_test["kt_cs"].values.astype(np.float32)

        fold_preds: dict[str, object] = {
            "station_id":    df_test["station_id"].values,
            "timestamp_utc": df_test["timestamp_utc"].values,
            "y":             y_te,
            "fold":          fold_idx,
        }
        fold_row: dict[str, object] = {
            "fold":           fold_idx,
            "n_test":         len(y_te),
            "test_stations":  ",".join(sorted(test_sids)),
        }

        for model_name, feats in MODELS.items():
            X_tr = df_tr[feats].values.astype(np.float32)
            X_va = df_va[feats].values.astype(np.float32)
            X_te = df_test[feats].values.astype(np.float32)

            y_hat, best_iter = _fit_predict(X_tr, y_tr, X_va, y_va, X_te)

            mae  = mean_absolute_error(y_te, y_hat)
            rmse = root_mean_squared_error(y_te, y_hat)

            pred_key = PRED_COL[model_name]
            fold_preds[pred_key] = y_hat
            fold_row[f"mae_{model_name}"]       = round(mae, 5)
            fold_row[f"rmse_{model_name}"]      = round(rmse, 5)
            fold_row[f"best_iter_{model_name}"] = best_iter
            print(f"  {model_name:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
                  f"best_iter={best_iter}")

        # Variance reduction: 1 - Var(res_emb) / Var(res_time)
        res_time = y_te - fold_preds["y_hat_time_only"]
        res_emb  = y_te - fold_preds["y_hat_time_plus_emb"]
        vr = 1.0 - float(np.var(res_emb)) / float(np.var(res_time))
        fold_row["var_reduction"] = round(vr, 5)
        print(f"  {'VAR_REDUCTION':20s}  {vr:+.4f}")

        fold_metrics.append(fold_row)
        pred_frames.append(pd.DataFrame(fold_preds))

    # --- Aggregate across folds ---
    metrics_df = pd.DataFrame(fold_metrics)

    summary_rows = []
    for model_name in MODELS:
        mae_vals  = metrics_df[f"mae_{model_name}"].values
        rmse_vals = metrics_df[f"rmse_{model_name}"].values
        summary_rows.append({
            "model":     model_name,
            "mae_mean":  round(mae_vals.mean(), 5),
            "mae_std":   round(mae_vals.std(ddof=1), 5),
            "rmse_mean": round(rmse_vals.mean(), 5),
            "rmse_std":  round(rmse_vals.std(ddof=1), 5),
        })
    summary_df = pd.DataFrame(summary_rows)

    vr_vals = metrics_df["var_reduction"].values
    vr_mean, vr_std = vr_vals.mean(), vr_vals.std(ddof=1)

    out_csv = OUT_DIR / "layer3_results.csv"
    summary_df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print("Layer 3 — 7-fold LOO-station results (mean ± std)")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\nVariance reduction (TIME_PLUS_EMB vs TIME_ONLY): "
          f"{vr_mean:+.4f} ± {vr_std:.4f}")
    print(f"Saved: {out_csv}")

    # --- Save full predictions ---
    preds_df = pd.concat(pred_frames, ignore_index=True)
    out_pq = OUT_DIR / "layer3_predictions.parquet"
    preds_df.to_parquet(out_pq, index=False)
    print(f"Saved: {out_pq}")

    # --- Figures ---
    _plot_bars(summary_df, vr_mean, vr_std)
    _plot_residual_scatter(preds_df, stations)
    print(f"Figures saved to {FIGS_DIR}")
    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} min")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "TIME_ONLY":     "#4878d0",
    "EMB_ONLY":      "#a0a0a0",
    "TIME_PLUS_EMB": "#6acc65",
    "SHUFFLE_EMB":   "#d65f5f",
}


def _plot_bars(summary_df: pd.DataFrame, vr_mean: float, vr_std: float):
    models = summary_df["model"].tolist()
    maes   = summary_df["mae_mean"].values
    stds   = summary_df["mae_std"].values

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    bars = ax.bar(
        x, maes, yerr=stds, capsize=5,
        color=[MODEL_COLORS.get(m, "#888") for m in models],
        width=0.55, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=10)
    ax.set_ylabel("LOO MAE (kt_cs)", fontsize=9)
    ax.set_title(
        f"Layer 3: hourly kt_cs — 7-fold LOO-station MAE (mean ± 1 std)\n"
        f"Variance reduction TIME+EMB vs TIME: {vr_mean:+.4f} ± {vr_std:.4f}",
        fontsize=9,
    )
    ax.tick_params(axis="y", labelsize=8)
    for bar, mae, std in zip(bars, maes, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.001,
                f"{mae:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIGS_DIR / "layer3_bars.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_residual_scatter(preds_df: pd.DataFrame, stations: pd.DataFrame):
    elev_map = {s["station_id"].zfill(5): float(s["elev_m"])
                for _, s in stations.iterrows()}
    name_map = {s["station_id"].zfill(5): s["name"]
                for _, s in stations.iterrows()}

    records = []
    for sid, grp in preds_df.groupby("station_id"):
        records.append({
            "station_id": sid,
            "name":       name_map.get(sid, sid),
            "mae_time":   float(np.median(np.abs(grp["y"] - grp["y_hat_time_only"]))),
            "mae_emb":    float(np.median(np.abs(grp["y"] - grp["y_hat_time_plus_emb"]))),
            "elev_m":     elev_map.get(sid, 0.0),
        })
    scat = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    sc = ax.scatter(scat["mae_time"], scat["mae_emb"],
                    c=scat["elev_m"], cmap="plasma", s=75, zorder=3,
                    edgecolors="white", linewidths=0.4)

    lo = min(scat["mae_time"].min(), scat["mae_emb"].min()) * 0.96
    hi = max(scat["mae_time"].max(), scat["mae_emb"].max()) * 1.04
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.45)
    # Shade region where embedding helps
    ax.fill_between([lo, hi], [lo, lo], [hi, hi],
                    alpha=0.04, color="green", label="emb helps (below diagonal)")

    for _, row in scat.iterrows():
        ax.annotate(
            row["station_id"],
            (row["mae_time"], row["mae_emb"]),
            fontsize=6.5, textcoords="offset points", xytext=(4, 2),
            color="#333",
        )

    fig.colorbar(sc, ax=ax, label="elevation (m)")
    ax.set_xlabel("TIME_ONLY  median |residual| (kt_cs)", fontsize=9)
    ax.set_ylabel("TIME_PLUS_EMB  median |residual| (kt_cs)", fontsize=9)
    ax.set_title(
        "Per-station residuals: TIME_ONLY vs TIME_PLUS_EMB\n"
        "Below diagonal → embedding reduces error; colour = elevation",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIGS_DIR / "layer3_residual_scatter.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
