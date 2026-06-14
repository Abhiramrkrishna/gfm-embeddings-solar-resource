"""
Pool DWD-relaxed stations onto the core 28 to reach n~38, then prepare them for
the spatial-block re-test.

The relaxed CSV was never persisted on this machine (parse_stations.py wrote it
to a Linux sandbox path), so we RE-DERIVE the pool from the DWD metadata block
embedded in parse_stations.py.

Pool definition (locked):
  added = radiation stations with >= 2-yr coverage of 2020-2024
          (von <= 2023-01-01 AND bis >= 2024-12-31) that are NOT in core 28.
  -> 10 stations, giving n = 38.

This script (no Earth Engine needed):
  1. re-derives the pool, writes spatial_holdout/pooled_stations.csv (38 rows,
     with calendar window length per station),
  2. downloads + enriches the added stations that lack an enriched parquet
     (reuses functions from scripts/01 and scripts/02),
  3. reports REAL record length per added station: valid-kt hours and calendar
     span, flagging any whose short record makes its kt_mean target unreliable.

Embedding extraction (scripts/10) and the re-test (scripts/11) are separate.
"""
from __future__ import annotations
import importlib.util
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

CORE_CSV   = "dwd_core_stations.csv"
OUT_DIR    = Path("spatial_holdout")
OUT_DIR.mkdir(parents=True, exist_ok=True)
POOLED_CSV = OUT_DIR / "pooled_stations.csv"
ENRICHED   = Path("data/stations_enriched")


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def derive_pool() -> pd.DataFrame:
    src = open("parse_stations.py").read()
    raw = src.split('raw = """')[1].split('"""')[0]
    rows = []
    for line in raw.strip().split("\n"):
        p = line.split()
        rows.append(dict(
            station_id=p[0].zfill(5),
            von=datetime.strptime(p[1], "%Y%m%d"),
            bis=datetime.strptime(p[2], "%Y%m%d"),
            elev_m=int(p[3]), lat=float(p[4]), lon=float(p[5]),
            name=" ".join(p[6:-1]), state=p[-1]))
    return pd.DataFrame(rows)


def main():
    meta = derive_pool()
    core = pd.read_csv(CORE_CSV, dtype={"station_id": str})
    core_ids = set(core["station_id"].str.zfill(5))

    win_start, win_end = datetime(2020, 1, 1), datetime(2025, 1, 1)
    twoyr = meta[(meta.von <= datetime(2023, 1, 1)) &
                 (meta.bis >= datetime(2024, 12, 31))]
    added = twoyr[~twoyr.station_id.isin(core_ids)].copy()

    # calendar window length actually inside 2020-2024
    def win_years(r):
        s = max(r.von, win_start)
        e = min(r.bis, win_end)
        return round((e - s).days / 365.25, 2)
    meta["window_years"] = meta.apply(win_years, axis=1)

    pooled_ids = core_ids | set(added.station_id)
    pooled = meta[meta.station_id.isin(pooled_ids)].copy()
    pooled["in_core"] = pooled.station_id.isin(core_ids)
    pooled = pooled.sort_values(["in_core", "station_id"], ascending=[False, True])
    pooled[["station_id", "name", "state", "lat", "lon", "elev_m",
            "von", "bis", "window_years", "in_core"]].to_csv(POOLED_CSV, index=False)

    print(f"Core: {len(core_ids)}  |  Added (relaxed, >=2yr, not core): "
          f"{len(added)}  |  Pooled n = {len(pooled)}")
    print(f"Wrote {POOLED_CSV}\n")
    print("Added stations:")
    for _, r in added.sort_values("von").iterrows():
        print(f"  {r.station_id}  {r['name']:24s} {r.von:%Y-%m}->{r.bis:%Y-%m}  "
              f"win~{meta.loc[meta.station_id==r.station_id,'window_years'].iloc[0]:.1f}yr "
              f"{r.elev_m:>4d}m")

    # ---- download + enrich added stations lacking an enriched parquet ------
    dl = _load_module("scripts/01_download_dwd.py", "dwd_dl")
    en = _load_module("scripts/02_add_clear_sky.py", "dwd_en")

    need = [r for _, r in added.iterrows()
            if not (ENRICHED / f"{r.station_id}.parquet").exists()]
    print(f"\n{len(need)} added stations need download/enrich.")
    if need:
        print("Discovering DWD remote archives...")
        remote = dl.list_remote_zips()
        for r in need:
            sid = r.station_id
            urls = remote.get(sid, [])
            if not urls:
                print(f"  [{sid}] NO DWD archive found — cannot pool")
                continue
            print(f"  [{sid}] {r['name']:24s} download({len(urls)}) -> parquet -> enrich")
            dl.download_station(sid, urls)
            dl.build_station_parquet(sid)
            en.enrich_station(sid, float(r.lat), float(r.lon), float(r.elev_m))

    # ---- real record-length report for ALL added stations ------------------
    print("\nRecord-length / target-reliability report (added stations):")
    rep = []
    for _, r in added.iterrows():
        sid = r.station_id
        p = ENRICHED / f"{sid}.parquet"
        if not p.exists():
            rep.append({"station_id": sid, "name": r["name"], "status": "MISSING"})
            continue
        df = pd.read_parquet(p)
        kt = df["kt_cs"].dropna()
        ts = pd.to_datetime(df.loc[df["kt_cs"].notna(), "timestamp_utc"], utc=True)
        span_yr = (ts.max() - ts.min()).days / 365.25 if len(ts) else 0.0
        # daytime hours over 5 full years if complete ~ 5*4380 ~ 21900
        reliable = len(kt) >= 8000           # ~>=2 yr of daytime valid hours
        rep.append({
            "station_id": sid, "name": r["name"],
            "valid_kt_hours": int(len(kt)),
            "span_years": round(span_yr, 2),
            "kt_mean": round(float(kt.mean()), 4) if len(kt) else np.nan,
            "reliable_target": reliable,
        })
    rdf = pd.DataFrame(rep)
    rdf.to_csv(OUT_DIR / "added_record_lengths.csv", index=False)
    print(rdf.to_string(index=False))
    flagged = rdf[rdf.get("reliable_target") == False]
    if len(flagged):
        print(f"\n⚠️  {len(flagged)} added station(s) with <~2yr valid data — "
              f"target kt_mean is noisy; excluded under the strict filter.")
    print(f"\nSaved: {POOLED_CSV}, {OUT_DIR/'added_record_lengths.csv'}")


if __name__ == "__main__":
    main()
