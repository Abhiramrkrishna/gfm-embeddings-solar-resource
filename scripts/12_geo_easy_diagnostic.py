"""
Geo-easy diagnostic: are the 10 added DWD stations systematically easier for
geography to predict than the core 28?

This adjudicates the open question left by scripts/11: the embedding advantage
that held at n=28 did not replicate on the 10 added stations. Two readings were
left open — (a) the added sites are "geo-easy" (geography already nails them, so
there is little residual for the embedding to win on → their lack of advantage
is uninformative), or (b) the effect is genuinely fragile (added sites have just
as much geo residual, but the embedding still fails to help).

Construction (same per-fold detrend as scripts/08/11, but leave-one-STATION-out):
  For each held-out station i over all 38 pooled stations:
    train = other 37
    OLS_train : kt_mean ~ (lat,lon,elev)  fit on train
    target    : kt_mean - OLS_train.predict(geo)   (geo-detrended mean kt_cs)
    Ridge(geo): fit geo->target on train, predict i  -> geo_prediction
    Ridge(emb): fit emb->target on train, predict i  -> emb_prediction
  Per station:
    geo_residual_magnitude = |target_i - geo_prediction_i|   (how much geo MISSES)
    emb_residual_magnitude = |target_i - emb_prediction_i|
    emb_advantage          = geo_residual_magnitude - emb_residual_magnitude
                             (>0 = embedding helps at this station)

Outputs:
  data/figs/geo_easy_diagnostic.png
  spatial_holdout/geo_easy_diagnostic.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from scipy.stats import mannwhitneyu, spearmanr, pearsonr

ENRICHED   = Path("data/stations_enriched")
AE_DIR     = Path("data/alphaearth")
POOLED_CSV = Path("spatial_holdout/pooled_stations.csv")
OUT_DIR    = Path("spatial_holdout")
FIGS       = Path("data/figs")
FIGS.mkdir(parents=True, exist_ok=True)

EMB_YEAR     = 2023
RIDGE_ALPHAS = np.logspace(-2, 4, 50)
DAYTIME_HRS_PER_YEAR = 4380.0     # ~half of 8760 (cs_ghi>5 hours)


def load_station(sid):
    """Return (kt_mean, valid_kt_hours) or (None, None)."""
    p = ENRICHED / f"{sid}.parquet"
    if not p.exists():
        return None, None
    kt = pd.read_parquet(p)["kt_cs"].dropna()
    if len(kt) < 100:
        return None, None
    return float(kt.mean()), int(len(kt))


def ridge_loo_residual(X, target):
    """Per-fold-detrend is already baked into `target` being recomputed per
    fold below; here X->target Ridge under LOO, returns |target - pred|."""
    mag = np.empty(len(target))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(X[tr]), target[tr])
        mag[te] = np.abs(target[te] - m.predict(sc.transform(X[te]))[0])
    return mag


def main():
    pooled = pd.read_csv(POOLED_CSV, dtype={"station_id": str})
    rows, embs = [], []
    for _, r in pooled.iterrows():
        sid = r["station_id"].zfill(5)
        ktm, nh = load_station(sid)
        ep = AE_DIR / f"{sid}_{EMB_YEAR}.npy"
        if ktm is None or not ep.exists():
            print(f"  [{sid}] skipped")
            continue
        rows.append({"station_id": sid, "name": r["name"], "kt_mean": ktm,
                     "lat": float(r["lat"]), "lon": float(r["lon"]),
                     "elev_m": float(r["elev_m"]), "in_core": bool(r["in_core"]),
                     "valid_kt_hours": nh,
                     "record_years": round(nh / DAYTIME_HRS_PER_YEAR, 2)})
        embs.append(np.load(ep))
    df = pd.DataFrame(rows).reset_index(drop=True)
    E  = np.stack(embs)
    n  = len(df)
    kt = df["kt_mean"].values.astype(float)
    Xg = df[["lat", "lon", "elev_m"]].values.astype(float)

    # ---- LOO with per-fold detrend; build geo & emb residual magnitudes ----
    geo_mag = np.empty(n)
    emb_mag = np.empty(n)
    for tr, te in LeaveOneOut().split(Xg):
        ols = LinearRegression().fit(Xg[tr], kt[tr])
        t_tr = kt[tr] - ols.predict(Xg[tr])
        t_te = kt[te] - ols.predict(Xg[te])
        # geo model
        scg = StandardScaler().fit(Xg[tr])
        mg = RidgeCV(alphas=RIDGE_ALPHAS).fit(scg.transform(Xg[tr]), t_tr)
        geo_mag[te] = np.abs(t_te - mg.predict(scg.transform(Xg[te]))[0])
        # emb model
        sce = StandardScaler().fit(E[tr])
        me = RidgeCV(alphas=RIDGE_ALPHAS).fit(sce.transform(E[tr]), t_tr)
        emb_mag[te] = np.abs(t_te - me.predict(sce.transform(E[te]))[0])

    df["geo_residual_magnitude"] = geo_mag
    df["emb_residual_magnitude"] = emb_mag
    df["emb_advantage"] = geo_mag - emb_mag
    df.to_csv(OUT_DIR / "geo_easy_diagnostic.csv", index=False)

    core  = df[df["in_core"]]
    added = df[~df["in_core"]]

    # ---- group comparison: is ADDED geo-easier? ----------------------------
    cg, ag = core["geo_residual_magnitude"].values, added["geo_residual_magnitude"].values
    U, p_two = mannwhitneyu(ag, cg, alternative="two-sided")
    _, p_less = mannwhitneyu(ag, cg, alternative="less")   # added < core
    med_core, med_added = np.median(cg), np.median(ag)

    # ---- physical signature: emb helps where geo misses? -------------------
    rho, p_rho = spearmanr(df["geo_residual_magnitude"], df["emb_advantage"])
    rP, p_rP   = pearsonr(df["geo_residual_magnitude"], df["emb_advantage"])

    med_adv_core  = float(np.median(core["emb_advantage"]))
    med_adv_added = float(np.median(added["emb_advantage"]))

    # ---- verdict -----------------------------------------------------------
    added_smaller   = med_added < med_core
    geo_easy_sig    = (p_less < 0.05) and added_smaller
    comparable_geo  = p_two >= 0.05
    added_no_adv    = med_adv_added <= 0.0005          # ~no help (target units)

    if geo_easy_sig:
        verdict = ("GEO-EASY — added stations have significantly SMALLER geo "
                   "residual (geography already nails them), so their weak "
                   "embedding advantage is partly UNINFORMATIVE, not strong "
                   "evidence against the effect.")
    elif comparable_geo and added_no_adv:
        verdict = ("GENUINELY FRAGILE — added stations have COMPARABLE geo "
                   "residual but the embedding still does not help there; the "
                   "effect does not replicate on independent stations.")
    else:
        verdict = ("AMBIGUOUS — neither cleanly geo-easy nor cleanly fragile; "
                   "see numbers and the geo-residual/emb-advantage correlation.")

    # ---- print -------------------------------------------------------------
    print(f"n = {n}  (core {len(core)}, added {len(added)})\n")
    print("geo_residual_magnitude (how much geography misses):")
    print(f"  CORE  median = {med_core:.5f}")
    print(f"  ADDED median = {med_added:.5f}   "
          f"({'smaller' if added_smaller else 'larger/equal'} than core)")
    print(f"  Mann-Whitney U: two-sided p = {p_two:.4f}, "
          f"one-sided p(added<core) = {p_less:.4f}\n")
    print("emb_advantage (geo_res - emb_res, >0 = embedding helps):")
    print(f"  CORE  median = {med_adv_core:+.5f}")
    print(f"  ADDED median = {med_adv_added:+.5f}\n")
    print("Does the embedding help more where geography misses more?")
    print(f"  Spearman rho(geo_residual, emb_advantage) = {rho:+.3f}  (p={p_rho:.4f})")
    print(f"  Pearson  r  (geo_residual, emb_advantage) = {rP:+.3f}  (p={p_rP:.4f})")
    print(f"  -> {'YES, positive — physically sensible signature of a real effect'
                  if rho > 0 and p_rho < 0.05 else 'not a clear positive relationship'}\n")
    print("VERDICT:", verdict)

    # ---- scatter -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for grp, sub, color, mk in [("CORE", core, "#4878d0", "o"),
                                ("ADDED", added, "#d65f5f", "^")]:
        ax.scatter(sub["geo_residual_magnitude"], sub["emb_advantage"],
                   s=sub["record_years"] * 28, c=color, marker=mk,
                   alpha=0.78, edgecolors="white", linewidths=0.5,
                   label=f"{grp} (n={len(sub)})")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("geo_residual_magnitude  (how much geography misses) →", fontsize=9)
    ax.set_ylabel("emb_advantage  (↑ embedding helps;  ↓ embedding hurts)", fontsize=9)
    ax.set_title(
        f"Geo-easy diagnostic (n={n})\n"
        f"Spearman ρ(geo_residual, emb_advantage) = {rho:+.2f} (p={p_rho:.3f});  "
        f"marker size ∝ record length",
        fontsize=9)
    # annotate the added stations
    for _, r in added.iterrows():
        ax.annotate(r["station_id"], (r["geo_residual_magnitude"], r["emb_advantage"]),
                    fontsize=6.5, color="#7a1f1f",
                    textcoords="offset points", xytext=(4, 2))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIGS / "geo_easy_diagnostic.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nSaved: {out}, {OUT_DIR/'geo_easy_diagnostic.csv'}")


if __name__ == "__main__":
    main()
