"""
TASK 2 — Adversarial reproducibility verification.

Goal: prove the negative result (WORLD 2 / emb barely beats geo, median advantage
~0) is REAL, not a pipeline artifact or a reporting error. This script re-derives
the key numbers FROM RAW DATA (raw DWD zips + raw AlphaEarth .npy), independent of
every cached intermediate (no enriched parquet, no results CSV), and cross-checks
them by a second code path plus positive/negative controls.

Checks:
  1. Full chain from raw: parse -> clear-sky -> kt_cs -> per-fold-detrend
     spatial-block Ridge CV -> emb_advantage, emb-vs-shuffle, emb-vs-geo.
  2. Independent re-implementation of the key number (median emb_advantage and
     Wilcoxon p on emb-vs-shuffle): my inline CV vs the locked s08 path; scipy
     Wilcoxon vs a hand-written signed-rank.
  3. Positive/negative controls (must behave as predicted or there is a bug):
     noise embeddings -> advantage ~0; target-as-feature -> advantage spikes;
     label-shuffled target -> all signal vanishes.
  4. Determinism: same seed twice -> identical; 3 seeds -> verdict stable.
  5. Environment + data checksums -> data/repro/.
  6. Final verdict vs previously reported (27.6% core / 13.7% all / ~0 median /
     WORLD 2).

Reuses transformation *logic* (it is the locked method) but reads only raw bytes;
the controls are what catch leakage/wiring bugs that faithful re-implementation
alone could not.
"""
from __future__ import annotations
import importlib.util
import hashlib
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
from scipy.stats import wilcoxon, norm
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler

RAW_DWD   = Path("data/raw_dwd")
AE_DIR    = Path("data/alphaearth")
CORE_CSV  = "dwd_core_stations.csv"
REPRO     = Path("data/repro"); REPRO.mkdir(parents=True, exist_ok=True)
EMB_YEAR  = 2023
RIDGE_ALPHAS = np.logspace(-2, 4, 50)
K_FIXED   = 4
KMEANS_RS = 42

# Previously reported numbers (to verify against)
REPORTED = {"core_reduction_pct": 27.6, "all_reduction_pct": 13.7,
            "all_median_adv": 0.00015}

s08 = None
def _load_s08():
    global s08
    spec = importlib.util.spec_from_file_location(
        "s08", "scripts/08_layer2_spatial_block_perfold.py")
    s08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s08)


# ---------------------------------------------------------------------------
# Stations from source lists (no derived CSV of results)
# ---------------------------------------------------------------------------

def derive_stations():
    core = pd.read_csv(CORE_CSV, dtype={"station_id": str})
    core["station_id"] = core["station_id"].str.zfill(5)
    core_ids = set(core["station_id"])
    src = open("parse_stations.py").read()
    raw = src.split('raw = """')[1].split('"""')[0]
    meta = []
    for line in raw.strip().split("\n"):
        p = line.split()
        meta.append(dict(station_id=p[0].zfill(5),
                         von=datetime.strptime(p[1], "%Y%m%d"),
                         bis=datetime.strptime(p[2], "%Y%m%d"),
                         elev_m=int(p[3]), lat=float(p[4]), lon=float(p[5])))
    meta = pd.DataFrame(meta)
    added = meta[(meta.von <= datetime(2023, 1, 1)) &
                 (meta.bis >= datetime(2024, 12, 31)) &
                 (~meta.station_id.isin(core_ids))].copy()
    rows = []
    for _, r in core.iterrows():
        rows.append(dict(station_id=r.station_id, lat=float(r.lat),
                         lon=float(r.lon), elev_m=float(r.elev_m), in_core=True))
    for _, r in added.iterrows():
        rows.append(dict(station_id=r.station_id, lat=float(r.lat),
                         lon=float(r.lon), elev_m=float(r.elev_m), in_core=False))
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# RAW -> kt_mean (independent inline parse + clear-sky; reads only zip bytes)
# ---------------------------------------------------------------------------

def kt_mean_from_raw(sid, lat, lon, elev, used_files):
    frames = []
    for zp in sorted((RAW_DWD / sid).glob("*.zip")):
        used_files.append(zp)
        with zipfile.ZipFile(zp) as zf:
            name = [n for n in zf.namelist()
                    if n.lower().startswith("produkt") and n.endswith(".txt")][0]
            with zf.open(name) as f:
                d = pd.read_csv(f, sep=";", skipinitialspace=True,
                                na_values=["-999", -999], dtype={"STATIONS_ID": str})
        d.columns = [c.strip() for c in d.columns]
        ts = d["MESS_DATUM"].astype(str).str.strip()
        d["timestamp_utc"] = pd.to_datetime(ts, format="%Y%m%d%H:%M",
                                            utc=True, errors="coerce").dt.floor("h")
        frames.append(d[["timestamp_utc", "FG_LBERG"]])
    df = (pd.concat(frames, ignore_index=True)
          .dropna(subset=["timestamp_utc"])
          .drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc"))
    m = (df.timestamp_utc >= "2020-01-01") & (df.timestamp_utc < "2025-01-01")
    df = df.loc[m].copy()
    df["FG_WM2"] = df["FG_LBERG"] * (1e4 / 3600.0)

    idx = pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    df = df.set_index("timestamp_utc").reindex(idx)
    loc = pvlib.location.Location(lat, lon, altitude=elev, tz="UTC")
    centers = idx - pd.Timedelta(minutes=30)
    cs = loc.get_clearsky(centers, model="ineichen"); cs.index = idx
    ghi_cs = cs["ghi"].values
    day = ghi_cs > 5.0
    fg = df["FG_WM2"].values
    with np.errstate(divide="ignore", invalid="ignore"):   # night hours -> cs=0
        kt = np.where(day, fg / ghi_cs, np.nan)
    kt = np.where((kt > 1.5) | (kt < 0), np.nan, kt)
    kt = kt[~np.isnan(kt)]
    return float(kt.mean()), int(len(kt))


# ---------------------------------------------------------------------------
# Independent inline spatial-block per-fold-detrend CV (Path A)
# ---------------------------------------------------------------------------

def cv_independent(feature_mats, kt, Xg, folds, leak_target=False):
    n = len(kt)
    out = {k: np.full(n, np.nan) for k in feature_mats}
    alli = np.arange(n)
    for te in folds:
        tr = np.setdiff1d(alli, te)
        ols = LinearRegression().fit(Xg[tr], kt[tr])
        t_tr, t_te = kt[tr] - ols.predict(Xg[tr]), kt[te] - ols.predict(Xg[te])
        for name, X in feature_mats.items():
            if leak_target and name == "emb":
                Xtr, Xte = t_tr.reshape(-1, 1), t_te.reshape(-1, 1)
            else:
                Xtr, Xte = X[tr], X[te]
            sc = StandardScaler().fit(Xtr)
            mdl = RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(Xtr), t_tr)
            out[name][te] = np.abs(t_te - mdl.predict(sc.transform(Xte)))
    return out


def make_folds(lat, lon):
    coords = s08.project_km(lat, lon)
    labels = KMeans(K_FIXED, n_init=20, random_state=KMEANS_RS).fit_predict(coords)
    return [np.where(labels == b)[0] for b in range(K_FIXED)]


# ---------------------------------------------------------------------------
# Stats: hand-written one-sided Wilcoxon signed-rank (normal approx)
# ---------------------------------------------------------------------------

def wilcoxon_hand_greater(d):
    """One-sided signed-rank, H1: median(d) > 0. Normal approx w/ tie+continuity
    correction — matches scipy method='approx'."""
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan
    r = pd.Series(np.abs(d)).rank().values
    W = float(np.sum(r[d > 0]))
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = np.sum(counts**3 - counts)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    z = (W - mu - 0.5) / np.sqrt(var)
    return float(norm.sf(z))


def win_rate(d):
    return float(np.mean(d > 0))


def reduction_pct(geo_err, emb_err):
    return 100.0 * (geo_err.mean() - emb_err.mean()) / geo_err.mean()


# ---------------------------------------------------------------------------
# Core re-derivation for a given station subset + seed
# ---------------------------------------------------------------------------

def derive(df, E, kt, seed, leak=False, noise=False, label_shuffle=False):
    rng = np.random.default_rng(seed)
    n = len(df)
    Xg = df[["lat", "lon", "elev_m"]].values.astype(float)
    ktv = kt.copy()
    if label_shuffle:
        ktv = ktv[rng.permutation(n)]
    Euse = rng.standard_normal(E.shape).astype(E.dtype) if noise else E
    feats = {"geo": Xg, "emb": Euse, "shuffle": Euse[rng.permutation(n)]}
    folds = make_folds(df["lat"].values, df["lon"].values)
    err = cv_independent(feats, ktv, Xg, folds, leak_target=leak)
    adv_geo = err["geo"] - err["emb"]          # emb beats geo
    adv_shuf = err["shuffle"] - err["emb"]     # emb beats shuffle
    return {"err": err, "adv_geo": adv_geo, "adv_shuf": adv_shuf,
            "reduction": reduction_pct(err["geo"], err["emb"]),
            "median_adv": float(np.median(adv_geo)),
            "win_geo": win_rate(adv_geo), "win_shuf": win_rate(adv_shuf),
            # correction=True matches the hand impl's continuity correction
            "p_shuf_scipy": float(wilcoxon(adv_shuf, alternative="greater",
                                           method="approx",
                                           correction=True).pvalue),
            "p_shuf_hand": wilcoxon_hand_greater(adv_shuf)}


# ---------------------------------------------------------------------------
def main():
    _load_s08()
    print("=" * 72)
    print("ADVERSARIAL REPRODUCIBILITY VERIFICATION (from raw)")
    print("=" * 72)

    stations = derive_stations()
    print(f"\nReconstructed {len(stations)} stations from source lists "
          f"({stations.in_core.sum()} core + {(~stations.in_core).sum()} added)")

    # ---- 1. raw -> kt_mean + embeddings ----------------------------------
    print("\n[1] Recomputing kt_cs from raw DWD zips + clear-sky (inline)...")
    used_files, kt_list, emb_list, keep = [], [], [], []
    for _, r in stations.iterrows():
        sid = r.station_id
        ep = AE_DIR / f"{sid}_{EMB_YEAR}.npy"
        try:
            km, nh = kt_mean_from_raw(sid, r.lat, r.lon, r.elev_m, used_files)
        except Exception as e:
            print(f"  [{sid}] raw recompute FAILED: {str(e)[:80]}"); continue
        if not ep.exists():
            print(f"  [{sid}] missing embedding"); continue
        kt_list.append(km); emb_list.append(np.load(ep)); keep.append(r.name)
        used_files.append(ep)
    df = stations.loc[keep].reset_index(drop=True)
    kt = np.array(kt_list); E = np.stack(emb_list)
    print(f"  recomputed kt_mean for {len(df)} stations from raw "
          f"(range {kt.min():.3f}-{kt.max():.3f})")

    core_mask = df.in_core.values
    df_core = df[core_mask].reset_index(drop=True)
    E_core, kt_core = E[core_mask], kt[core_mask]

    base_all  = derive(df,      E,      kt,      seed=42)
    base_core = derive(df_core, E_core, kt_core, seed=42)

    print(f"\n  ALL-38 : reduction {base_all['reduction']:+.1f}%, "
          f"median_adv {base_all['median_adv']:+.5f}, "
          f"win_geo {base_all['win_geo']:.2f}, win_shuf {base_all['win_shuf']:.2f}")
    print(f"  CORE-28: reduction {base_core['reduction']:+.1f}%, "
          f"win_geo {base_core['win_geo']:.2f}")

    # ---- 2. independent cross-implementation -----------------------------
    print("\n[2] Independent cross-implementation of the key numbers:")
    folds = make_folds(df["lat"].values, df["lon"].values)
    Xg = df[["lat", "lon", "elev_m"]].values.astype(float)
    rng = np.random.default_rng(42)
    feats_b = {"geo": Xg, "emb": E, "combined": np.hstack([Xg, E]),
               "shuffle": E[rng.permutation(len(df))]}
    err_b, _ = s08.cv_perfold_detrend(feats_b, kt, Xg, folds)   # locked path B
    adv_geo_b = err_b["geo"] - err_b["emb"]
    med_a, med_b = base_all["median_adv"], float(np.median(adv_geo_b))
    print(f"  median emb_advantage : inline={med_a:+.8f}  locked={med_b:+.8f}  "
          f"|diff|={abs(med_a-med_b):.2e}")
    # NB shuffle perm differs between paths; recompute emb-vs-shuffle on path B
    adv_shuf_b = err_b["shuffle"] - err_b["emb"]
    p_a = base_all["p_shuf_scipy"]
    p_hand = base_all["p_shuf_hand"]
    print(f"  Wilcoxon p(emb>shuf) : scipy={p_a:.5f}  hand={p_hand:.5f}  "
          f"|diff|={abs(p_a-p_hand):.2e}")
    impl_ok = abs(med_a - med_b) < 1e-9 and abs(p_a - p_hand) < 1e-3
    print(f"  => cross-implementation {'AGREE' if impl_ok else 'DISAGREE'}")

    # ---- 3. controls ------------------------------------------------------
    print("\n[3] Positive/negative controls (must behave as predicted):")
    noise = derive(df, E, kt, seed=42, noise=True)
    leak  = derive(df, E, kt, seed=42, leak=True)
    lbl   = derive(df, E, kt, seed=42, label_shuffle=True)
    geo_med_err = float(np.median(base_all["err"]["geo"]))
    c1 = abs(noise["median_adv"]) < 0.001          # noise can't help
    c2 = (np.median(leak["err"]["emb"]) < 0.1 * geo_med_err
          and leak["median_adv"] > 0.5 * geo_med_err)   # leak -> ~perfect emb
    c3 = abs(lbl["median_adv"]) < 0.001 and lbl["p_shuf_scipy"] > 0.05
    print(f"  noise embeddings   : median_adv {noise['median_adv']:+.5f}  "
          f"-> {'PASS' if c1 else 'FAIL'} (expect ~0)")
    print(f"  target-as-feature  : emb_err {np.median(leak['err']['emb']):.5f} "
          f"vs geo_err {geo_med_err:.5f}, adv {leak['median_adv']:+.5f}  "
          f"-> {'PASS' if c2 else 'FAIL'} (expect emb~0, adv spikes)")
    print(f"  label-shuffle      : median_adv {lbl['median_adv']:+.5f}, "
          f"p_shuf {lbl['p_shuf_scipy']:.3f}  "
          f"-> {'PASS' if c3 else 'FAIL'} (expect ~0, n.s.)")
    controls_ok = c1 and c2 and c3

    # ---- 4. determinism + seed stability ---------------------------------
    print("\n[4] Determinism + seed stability:")
    rep = derive(df, E, kt, seed=42)
    det = (rep["median_adv"] == base_all["median_adv"]
           and rep["reduction"] == base_all["reduction"]
           and rep["p_shuf_scipy"] == base_all["p_shuf_scipy"])
    print(f"  same seed twice identical: {'YES' if det else 'NO'}")
    seeds = [42, 0, 7]
    runs = [derive(df, E, kt, seed=s) for s in seeds]
    reds = [r["reduction"] for r in runs]
    meds = [r["median_adv"] for r in runs]
    pshuf = [r["p_shuf_scipy"] for r in runs]
    print(f"  across seeds {seeds}:")
    print(f"    reduction%   : {[round(x,2) for x in reds]} "
          f"(emb-vs-geo is seed-free -> constant)")
    print(f"    median_adv   : {[round(x,5) for x in meds]}")
    print(f"    p(emb>shuf)  : {[round(x,4) for x in pshuf]} "
          f"(varies with shuffle perm; spread {max(pshuf)-min(pshuf):.4f})")
    verdict_stable = (max(reds) - min(reds) < 1e-6 and
                      all(r["median_adv"] < 0.005 for r in runs))
    print(f"  WORLD-2/negative stable across seeds: "
          f"{'YES' if verdict_stable else 'NO'}")

    # ---- 5. environment + checksums --------------------------------------
    print("\n[5] Environment + data checksums -> data/repro/")
    env = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True).stdout
    (REPRO / "environment.txt").write_text(
        f"python {sys.version}\nplatform {sys.platform}\n\n{env}")
    lines = []
    for f in sorted(set(used_files)):
        h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        lines.append(f"{h}  {f}")
    (REPRO / "data_checksums.txt").write_text("\n".join(lines) + "\n")
    print(f"  wrote environment.txt and data_checksums.txt "
          f"({len(lines)} raw files hashed)")

    # ---- 6. final verdict vs reported ------------------------------------
    print("\n" + "=" * 72)
    print("[6] VERIFICATION VERDICT (re-derived from raw vs previously reported)")
    print("=" * 72)
    checks = [
        ("CORE-28 emb-vs-geo reduction", base_core["reduction"],
         REPORTED["core_reduction_pct"], 1.0, "%"),
        ("ALL-38 emb-vs-geo reduction", base_all["reduction"],
         REPORTED["all_reduction_pct"], 1.0, "%"),
        ("ALL-38 median emb_advantage", base_all["median_adv"],
         REPORTED["all_median_adv"], 0.0005, ""),
    ]
    all_match = True
    for name, got, rep_v, tol, unit in checks:
        ok = abs(got - rep_v) <= tol
        all_match &= ok
        print(f"  {name:34s} re-derived={got:+.5f}{unit}  reported={rep_v:+.5f}{unit}"
              f"  |diff|={abs(got-rep_v):.4f}  {'MATCH' if ok else 'DISCREPANCY'}")
    print(f"\n  Cross-implementation agree : {impl_ok}")
    print(f"  Controls pass              : {controls_ok}")
    print(f"  Deterministic + seed-stable: {det and verdict_stable}")
    confirmed = all_match and impl_ok and controls_ok and det and verdict_stable
    print("\n  " + ("NEGATIVE RESULT CONFIRMED under independent re-derivation: "
                    "the small/near-zero emb-over-geo advantage (WORLD 2) is real, "
                    "not a pipeline artifact." if confirmed else
                    "NOT fully confirmed — see discrepancies above."))
    print("=" * 72)
    return confirmed


if __name__ == "__main__":
    main()
