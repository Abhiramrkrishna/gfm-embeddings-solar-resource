"""
Pooled re-test: per-fold-detrend spatial-block comparison at the larger n,
with a record-length sensitivity check.

Reuses the locked per-fold-detrend machinery from scripts/08. The ONLY change
vs today's n=28 run is the station set; to keep the spatial blocking structurally
identical (so we compare n, not block geometry) **K is fixed at 4** — the same
number of geographic blocks as the n=28 run. Pooling therefore adds stations
*within* the same 4-block partition; expect per-fold buffers to shrink as the
network densifies (the expected cost of trading buffer for n).

Sensitivity ladder (record length):
  all    : n=38  (core 28 + 10 relaxed)
  strict : n=35  (drop the 3 added stations with <~2yr valid kt data, whose
                  kt_mean target is flagged unreliable)
  core   : n=28  (today's result, recomputed here for a like-for-like baseline)

Outputs (spatial_holdout/):
  results_pooled.csv, comparisons_pooled.csv, FINDINGS.md (dated entry)
"""
from __future__ import annotations
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut
from scipy.stats import wilcoxon

OUT_DIR    = Path("spatial_holdout")
POOLED_CSV = OUT_DIR / "pooled_stations.csv"
RECLEN_CSV = OUT_DIR / "added_record_lengths.csv"
AE_DIR     = Path("data/alphaearth")
EMB_YEAR   = 2023
RANDOM_STATE = 42
K_FIXED    = 4                 # match the n=28 spatial-block run
MIN_RELIABLE_HOURS = 8000      # ~>=2yr daytime valid kt -> reliable target
GO_RED, KILL_RED, ALPHA = 0.10, 0.05, 0.05


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

s08 = _load("scripts/08_layer2_spatial_block_perfold.py", "s08")


def haversine_km(a1, o1, a2, o2):
    R = 6371.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = np.radians(a2 - a1), np.radians(o2 - o1)
    h = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(h))


def build_frame(pooled):
    rows, embs = [], []
    for _, r in pooled.iterrows():
        sid = r["station_id"].zfill(5)
        ktm = s08.compute_kt_mean(sid)
        emb = s08.load_embedding(sid, EMB_YEAR)
        if ktm is None or emb is None:
            print(f"  [{sid}] skipped (kt_mean={ktm is not None}, "
                  f"emb={emb is not None})")
            continue
        rows.append({"station_id": sid, "name": r["name"], "kt_mean": ktm,
                     "lat": float(r["lat"]), "lon": float(r["lon"]),
                     "elev_m": float(r["elev_m"]), "in_core": bool(r["in_core"])})
        embs.append(emb)
    return pd.DataFrame(rows).reset_index(drop=True), np.stack(embs)


def run_one(df, E, label):
    n = len(df)
    kt = df["kt_mean"].values.astype(float)
    Xg = df[["lat", "lon", "elev_m"]].values.astype(float)
    rng = np.random.default_rng(RANDOM_STATE)
    feats = {"geo": Xg, "emb": E, "combined": np.hstack([Xg, E]),
             "shuffle": E[rng.permutation(n)]}

    coords = s08.project_km(df["lat"].values, df["lon"].values)
    labels = KMeans(n_clusters=K_FIXED, n_init=20,
                    random_state=RANDOM_STATE).fit_predict(coords)
    folds = [np.where(labels == b)[0] for b in range(K_FIXED)]
    sizes = [len(f) for f in folds]

    # achieved buffers
    buffers = []
    for test_idx in folds:
        tr = np.setdiff1d(np.arange(n), test_idx)
        buffers.append(min(min(haversine_km(df.lat[t], df.lon[t],
                                            df.lat[j], df.lon[j]) for j in tr)
                           for t in test_idx))

    err, _ = s08.cv_perfold_detrend(feats, kt, Xg, folds)

    def pooled_mae(name): return float(np.mean(err[name]))
    def cmp(a, b="geo"):
        red = 100*(err[b].mean() - err[a].mean())/err[b].mean()
        try: _, p = wilcoxon(err[a], err[b], alternative="less")
        except ValueError: p = np.nan
        return float(red), float(p)

    emb_red, emb_p = cmp("emb")
    comb_red, comb_p = cmp("combined")

    def verdict(red, p):
        if red >= GO_RED*100 and p < ALPHA: return "GO"
        if red < KILL_RED*100 or p >= ALPHA: return "KILL"
        return "AMBIGUOUS"

    return {
        "set": label, "n": n, "K": K_FIXED, "fold_sizes": sizes,
        "min_buffer_km": round(min(buffers), 0),
        "mae_geo": round(pooled_mae("geo"), 5),
        "mae_emb": round(pooled_mae("emb"), 5),
        "mae_combined": round(pooled_mae("combined"), 5),
        "mae_shuffle": round(pooled_mae("shuffle"), 5),
        "emb_vs_geo_pct": round(emb_red, 1), "emb_vs_geo_p": round(emb_p, 4),
        "combined_vs_geo_pct": round(comb_red, 1),
        "combined_vs_geo_p": round(comb_p, 4),
        "verdict": verdict(emb_red, emb_p),
    }


def main():
    pooled = pd.read_csv(POOLED_CSV, dtype={"station_id": str})
    reclen = pd.read_csv(RECLEN_CSV, dtype={"station_id": str})
    unreliable = set(reclen.loc[reclen["valid_kt_hours"] < MIN_RELIABLE_HOURS,
                                "station_id"].str.zfill(5))
    print(f"Unreliable (<{MIN_RELIABLE_HOURS} valid kt hrs): "
          f"{sorted(unreliable)}\n")

    df_all, E_all = build_frame(pooled)
    if len(df_all) < len(pooled):
        print("!! some pooled stations missing embeddings — run scripts/10 first")

    keep = ~df_all["station_id"].isin(unreliable)
    df_strict, E_strict = df_all[keep].reset_index(drop=True), E_all[keep.values]
    df_core,  E_core  = (df_all[df_all["in_core"]].reset_index(drop=True),
                         E_all[df_all["in_core"].values])

    runs = [run_one(df_all,    E_all,    "all_n38"),
            run_one(df_strict, E_strict, "strict_n35"),
            run_one(df_core,   E_core,   "core_n28")]
    res = pd.DataFrame(runs)
    res.to_csv(OUT_DIR / "results_pooled.csv", index=False)
    res[["set", "n", "emb_vs_geo_pct", "emb_vs_geo_p",
         "combined_vs_geo_pct", "combined_vs_geo_p", "verdict"]].to_csv(
        OUT_DIR / "comparisons_pooled.csv", index=False)

    print(res.to_string(index=False))
    _write_findings(res, sorted(unreliable))
    print(f"\nSaved: results_pooled.csv, comparisons_pooled.csv, FINDINGS.md")


def _write_findings(res, unreliable):
    today = date.today().isoformat()
    r = {row["set"]: row for _, row in res.iterrows()}
    L = [f"\n## {today} — Layer 2 hardening #3: pool DWD-relaxed stations (n→38)\n",
         "Cheapest power hardening: added 10 DWD-relaxed radiation stations "
         "(re-derived from parse_stations.py metadata; downloaded + enriched + "
         "embeddings extracted) onto the core 28. Per-fold-detrend spatial-block "
         "re-test, **K fixed at 4** to match the n=28 run (pooling adds stations "
         "within the same 4 blocks; buffers shrink as the network densifies).\n",
         f"\n3 added stations flagged unreliable (<~2yr valid kt): "
         f"{unreliable} — Feldberg/Schwarzwald (1486 m) is valuable terrain but "
         "short-record; the strict run drops these.\n",
         "\n**emb-vs-geo across the record-length sensitivity ladder:**\n",
         "| set | n | min buffer | emb-vs-geo | Wilcoxon p | combined-vs-geo | verdict |",
         "|---|---|---|---|---|---|---|"]
    for key in ["core_n28", "strict_n35", "all_n38"]:
        x = r[key]
        L.append(f"| {x['set']} | {x['n']} | {x['min_buffer_km']:.0f} km | "
                 f"{x['emb_vs_geo_pct']:+.1f}% | {x['emb_vs_geo_p']:.4f} | "
                 f"{x['combined_vs_geo_pct']:+.1f}% (p={x['combined_vs_geo_p']:.4f}) | "
                 f"`{x['verdict']}` |")
    a, s, c = r["all_n38"], r["strict_n35"], r["core_n28"]
    L += ["\n**Read:**",
          f"- core n=28 (recomputed here): {c['emb_vs_geo_pct']:+.1f}%, "
          f"p={c['emb_vs_geo_p']:.4f}",
          f"- strict n=35: {s['emb_vs_geo_pct']:+.1f}%, p={s['emb_vs_geo_p']:.4f}",
          f"- all n=38: {a['emb_vs_geo_pct']:+.1f}%, p={a['emb_vs_geo_p']:.4f}",
          "\nEffect size and significance under pooling are reported above; the "
          "verdict column applies the pre-registered thresholds (GO ≥10% & "
          "p<0.05). Note the core-n28 number here uses K=4 fixed and may differ "
          "trivially from scripts/08's adaptive-K value.\n"]
    with open(OUT_DIR / "FINDINGS.md", "a") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
