"""
TASK 1 — Cluster diagnostic: does the high-emb_advantage tail share a measurable
site property, or is it just noise?

Per-station emb_advantage (= |geo_err| - |emb_err|) from the locked per-fold-
detrend spatial-block pipeline (scripts/08 + scripts/11), regressed against ~8
candidate site covariates. With n=38 and 8 covariates, ~0.4 false hits are
expected at uncorrected p<0.05 — so Benjamini-Hochberg FDR control is mandatory
and a single uncorrected p<0.05 is NOT a finding.

Covariates:
  terrain_ruggedness : stdDev of SRTM 30 m elevation in a 5 km buffer (GEE)
  coastal_distance   : great-circle km to a coarse hardcoded German coastline
  lat, lon, elev_m   : raw geography
  kt_std, over_irr_frac, kt_p95 : local-variability proxies (from enriched kt_cs)

Outputs: data/cluster_diagnostic.csv, data/figs/cluster_diagnostic.png
"""
from __future__ import annotations
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, mannwhitneyu, false_discovery_control

POOLED   = Path("spatial_holdout/pooled_stations.csv")
ENRICHED = Path("data/stations_enriched")
RUG_CACHE = Path("data/terrain_ruggedness_cache.csv")
OUT_CSV  = Path("data/cluster_diagnostic.csv")
FIGS     = Path("data/figs"); FIGS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE, K_FIXED = 42, 4
RUGGED_BUFFER_M = 5000

COVARIATES = ["terrain_ruggedness", "coastal_distance", "lat", "lon", "elev_m",
              "kt_std", "over_irr_frac", "kt_p95"]

# Coarse German coastline reference points (lon, lat) — North Sea + Baltic.
COASTLINE = [(6.70, 53.60), (7.15, 53.71), (8.10, 53.55), (8.60, 53.87),
             (8.30, 54.90), (8.60, 54.33), (9.55, 54.50), (10.15, 54.45),
             (11.10, 54.45), (12.08, 54.18), (13.43, 54.68), (14.00, 53.90)]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

s08 = _load("scripts/08_layer2_spatial_block_perfold.py", "s08")
s11 = _load("scripts/11_pool_retest.py", "s11")


def haversine_km(a1, o1, a2, o2):
    R = 6371.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = np.radians(a2 - a1), np.radians(o2 - o1)
    h = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(h))


def coastal_distance(lat, lon):
    return min(haversine_km(lat, lon, cy, cx) for cx, cy in COASTLINE)


def kt_variability(sid):
    df = pd.read_parquet(ENRICHED / f"{sid}.parquet")
    kt = df["kt_cs"].dropna()
    return (float(kt.std()), float((kt > 1.0).mean()), float(kt.quantile(0.95)))


def terrain_ruggedness(stations):
    if RUG_CACHE.exists():
        c = pd.read_csv(RUG_CACHE, dtype={"station_id": str})
        if set(stations["station_id"]).issubset(set(c["station_id"])):
            print("  terrain_ruggedness: using cache")
            return dict(zip(c["station_id"], c["terrain_ruggedness"]))
    import ee
    ee.Initialize(project="alpha-earth-x-solar-resources")
    dem = ee.Image("USGS/SRTMGL1_003")
    rug = {}
    print(f"  terrain_ruggedness: querying SRTM stdDev ({RUGGED_BUFFER_M} m buffer)")
    for _, r in stations.iterrows():
        sid = r["station_id"]
        pt = ee.Geometry.Point([float(r["lon"]), float(r["lat"])])
        v = dem.reduceRegion(ee.Reducer.stdDev(), pt.buffer(RUGGED_BUFFER_M),
                             30, bestEffort=True).getInfo()
        rug[sid] = float(v.get("elevation", np.nan))
    pd.DataFrame({"station_id": list(rug), "terrain_ruggedness": list(rug.values())}
                 ).to_csv(RUG_CACHE, index=False)
    return rug


def main():
    pooled = pd.read_csv(POOLED, dtype={"station_id": str})
    pooled["station_id"] = pooled["station_id"].str.zfill(5)

    # --- locked pipeline -> per-station emb_advantage ----------------------
    df, E = s11.build_frame(pooled)
    df["station_id"] = df["station_id"].str.zfill(5)
    n = len(df)
    kt = df["kt_mean"].values.astype(float)
    Xg = df[["lat", "lon", "elev_m"]].values.astype(float)
    rng = np.random.default_rng(RANDOM_STATE)
    feats = {"geo": Xg, "emb": E, "combined": np.hstack([Xg, E]),
             "shuffle": E[rng.permutation(n)]}
    coords = s08.project_km(df["lat"].values, df["lon"].values)
    labels = KMeans(K_FIXED, n_init=20, random_state=RANDOM_STATE).fit_predict(coords)
    folds = [np.where(labels == b)[0] for b in range(K_FIXED)]
    err, _ = s08.cv_perfold_detrend(feats, kt, Xg, folds)
    df["emb_advantage"] = err["geo"] - err["emb"]

    # --- covariates --------------------------------------------------------
    rug = terrain_ruggedness(pooled[["station_id", "lat", "lon"]])
    df["terrain_ruggedness"] = df["station_id"].map(rug)
    df["coastal_distance"] = [coastal_distance(la, lo)
                              for la, lo in zip(df["lat"], df["lon"])]
    var = {sid: kt_variability(sid) for sid in df["station_id"]}
    df["kt_std"]        = df["station_id"].map(lambda s: var[s][0])
    df["over_irr_frac"] = df["station_id"].map(lambda s: var[s][1])
    df["kt_p95"]        = df["station_id"].map(lambda s: var[s][2])
    pooled_core = pooled.set_index("station_id")["in_core"]
    df["in_core"] = df["station_id"].map(pooled_core)
    df["network"] = "DWD"          # pilot is single-network

    df.to_csv(OUT_CSV, index=False)

    # --- Spearman vs emb_advantage, BH-corrected ---------------------------
    adv = df["emb_advantage"].values
    sp_rows = []
    for c in COVARIATES:
        rho, p = spearmanr(df[c], adv)
        sp_rows.append({"covariate": c, "spearman_rho": rho, "p_raw": p})
    sp = pd.DataFrame(sp_rows)
    sp["p_bh"] = false_discovery_control(sp["p_raw"].values, method="bh")

    # --- top-tercile vs rest, Mann-Whitney, BH-corrected -------------------
    thr = np.quantile(adv, 2/3)
    hi = df[adv >= thr]; rest = df[adv < thr]
    mw_rows = []
    for c in COVARIATES:
        try:
            U, p = mannwhitneyu(hi[c], rest[c], alternative="two-sided")
        except ValueError:
            U, p = np.nan, np.nan
        mw_rows.append({"covariate": c, "median_hi": float(np.median(hi[c])),
                        "median_rest": float(np.median(rest[c])), "p_raw": p})
    mw = pd.DataFrame(mw_rows)
    mw["p_bh"] = false_discovery_control(mw["p_raw"].values, method="bh")

    # --- report ------------------------------------------------------------
    print(f"\nn={n}  (high-advantage top tercile: {len(hi)} stations, "
          f"threshold emb_advantage >= {thr:+.5f})\n")
    print("Spearman( covariate , emb_advantage )  [BH-FDR across 8]:")
    print(f"  {'covariate':18s} {'rho':>7} {'p_raw':>9} {'p_BH':>9}  survive?")
    for _, r in sp.sort_values("p_raw").iterrows():
        print(f"  {r.covariate:18s} {r.spearman_rho:>+7.3f} {r.p_raw:>9.4f} "
              f"{r.p_bh:>9.4f}  {'YES' if r.p_bh < 0.05 else 'no'}")
    print("\nMann-Whitney (top-tercile vs rest)  [BH-FDR across 8]:")
    print(f"  {'covariate':18s} {'med_hi':>9} {'med_rest':>9} {'p_raw':>9} {'p_BH':>9}  survive?")
    for _, r in mw.sort_values("p_raw").iterrows():
        print(f"  {r.covariate:18s} {r.median_hi:>9.3f} {r.median_rest:>9.3f} "
              f"{r.p_raw:>9.4f} {r.p_bh:>9.4f}  {'YES' if r.p_bh < 0.05 else 'no'}")

    sp_surv = set(sp.loc[sp.p_bh < 0.05, "covariate"])
    mw_surv = set(mw.loc[mw.p_bh < 0.05, "covariate"])
    survivors = sp_surv | mw_surv
    both = sp_surv & mw_surv
    # raw coordinates are baseline inputs, not independent physical site props
    COORDS = {"lat", "lon", "elev_m"}
    PHYSICAL = {"terrain_ruggedness", "coastal_distance", "kt_std",
                "over_irr_frac", "kt_p95"}
    phys_surv = survivors & PHYSICAL

    print("\n" + "=" * 70)
    if not survivors:
        print("NO covariate survives FDR correction in either test.")
        print("The high-advantage tail is NOT explained by measured site")
        print("properties — consistent with noise.")
    else:
        print("FDR survivors (mechanical):", ", ".join(sorted(survivors)))
        for c in sorted(survivors):
            rr = sp[sp.covariate == c].iloc[0]
            tests = "both tests" if c in both else \
                    ("Spearman only" if c in sp_surv else "Mann-Whitney only")
            kind = "raw COORDINATE (already in baseline)" if c in COORDS \
                   else "physical site property"
            print(f"  {c}: rho={rr.spearman_rho:+.3f}, Spearman BH p={rr.p_bh:.4f} "
                  f"[{tests}] — {kind}")
        print("\nHonest interpretation:")
        if not phys_surv:
            print("  - NO PHYSICAL site property survives FDR. The H2 mechanistic")
            print("    covariates (terrain ruggedness, coastal distance, local")
            print("    variability) show no FDR-surviving relationship.")
        if survivors and not both:
            print("  - The only survivor(s) clear FDR in just ONE of the two test")
            print("    types, i.e. do not replicate across methods — fragile.")
        if survivors & COORDS:
            print("  - A surviving raw coordinate is NOT a site-property finding:")
            print("    it is confounded with the geographic block structure of the")
            print("    spatial CV and is already an input to the geo baseline.")
        print("  => Treat as a hypothesis for independent data, NOT a finding.")
        print("     Consistent with the power analysis (median advantage ~0).")
    print("=" * 70)

    _plot(df)
    print(f"\nSaved: {OUT_CSV}, {FIGS/'cluster_diagnostic.png'}")


def _plot(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, c in zip(axes.flat, COVARIATES):
        for grp, col, mk in [(True, "#4878d0", "o"), (False, "#d65f5f", "^")]:
            sub = df[df["in_core"] == grp]
            ax.scatter(sub[c], sub["emb_advantage"], c=col, marker=mk, alpha=0.75,
                       edgecolors="white", linewidths=0.4,
                       label="CORE" if grp else "ADDED")
        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
        rho, p = spearmanr(df[c], df["emb_advantage"])
        ax.set_xlabel(c, fontsize=8)
        ax.set_ylabel("emb_advantage", fontsize=8)
        ax.set_title(f"{c}\nSpearman ρ={rho:+.2f} (raw p={p:.3f})", fontsize=8)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7, title="network: DWD")
    fig.suptitle("Cluster diagnostic: where does the embedding help? "
                 "(n=38, single network DWD; FDR-corrected — see console)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "cluster_diagnostic.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
