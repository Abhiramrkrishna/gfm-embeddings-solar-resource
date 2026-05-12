"""
Augment per-station hourly observations with clear-sky GHI and derived features.

This is the foundation of the entire downstream analysis. The clear-sky index
(observed GHI / clear-sky GHI) is the right unit of analysis because it removes
the deterministic solar geometry and leaves the atmospheric/local-effect signal
that AlphaEarth might or might not encode.

Dependencies: pvlib >= 0.10
    pip install pvlib pandas pyarrow

Usage:
    python scripts/02_add_clear_sky.py
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import pvlib
from pathlib import Path

PARQUET_DIR = Path("data/stations")
STATIONS_CSV = "dwd_core_stations.csv"
OUT_DIR = Path("data/stations_enriched")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_clear_sky(times: pd.DatetimeIndex, lat: float, lon: float, elev_m: float) -> pd.DataFrame:
    """Compute clear-sky GHI/DNI/DHI using pvlib's Ineichen-Perez model.

    Notes:
      - Times must be tz-aware UTC.
      - We label timestamps at the *end* of the hour, matching DWD convention.
      - We integrate at the center of each hour (timestamp - 30min) for a
        reasonable hourly mean approximation; for high-zenith hours this is
        an approximation, but matches the DWD aggregation convention.
    """
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=elev_m, tz="UTC")
    center_times = times - pd.Timedelta(minutes=30)
    cs = loc.get_clearsky(center_times, model="ineichen", linke_turbidity=None)
    cs.index = times  # re-label back to hour-ending
    cs.columns = [f"cs_{c}" for c in cs.columns]
    # Also compute solar position for sanity
    sp = loc.get_solarposition(center_times)
    sp.index = times
    cs["sun_zenith"] = sp["zenith"]
    cs["sun_apparent_elevation"] = sp["apparent_elevation"]
    return cs

def enrich_station(station_id: str, lat: float, lon: float, elev_m: float) -> Path | None:
    pq = PARQUET_DIR / f"{station_id}.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    # Build complete hourly index covering 2020-01-01 to 2024-12-31 23:00
    full_idx = pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    df = df.set_index("timestamp_utc").reindex(full_idx)
    df.index.name = "timestamp_utc"
    # Clear-sky model
    cs = compute_clear_sky(full_idx, lat, lon, elev_m)
    df = df.join(cs)
    # Clear-sky index (KT_cs)
    # FG_WM2 was set in step 1; if missing here, derive from FG_LBERG
    if "FG_WM2" not in df.columns and "FG_LBERG" in df.columns:
        df["FG_WM2"] = df["FG_LBERG"] * (1e4 / 3600.0)
    # Daytime mask: clear-sky > 5 W/m^2 avoids dawn/dusk numerical noise
    df["is_daytime"] = df["cs_ghi"] > 5.0
    df["kt_cs"] = np.where(df["is_daytime"], df["FG_WM2"] / df["cs_ghi"], np.nan)
    # Cap the clear-sky index at 1.5 (over-irradiance is real but extreme values
    # are typically QC failures rather than physics)
    df.loc[df["kt_cs"] > 1.5, "kt_cs"] = np.nan
    df.loc[df["kt_cs"] < 0, "kt_cs"] = np.nan
    df["station_id"] = station_id
    df = df.reset_index()
    out = OUT_DIR / f"{station_id}.parquet"
    df.to_parquet(out, index=False)
    return out

def main():
    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})
    rows = []
    for _, s in stations.iterrows():
        sid = s["station_id"].zfill(5)
        out = enrich_station(sid, float(s["lat"]), float(s["lon"]), float(s["elev_m"]))
        if out is None:
            continue
        df = pd.read_parquet(out)
        n_obs = df["FG_WM2"].notna().sum()
        n_day = df.loc[df["is_daytime"], "FG_WM2"].notna().sum()
        n_kt = df["kt_cs"].notna().sum()
        rows.append({
            "station_id": sid,
            "name": s["name"],
            "n_total_hours": len(df),
            "n_obs_hours": int(n_obs),
            "n_daytime_hours": int(n_day),
            "n_valid_kt": int(n_kt),
            "valid_kt_pct": round(100 * n_kt / df["is_daytime"].sum(), 1) if df["is_daytime"].sum() > 0 else 0.0,
            "kt_mean": round(df["kt_cs"].mean(), 3),
            "kt_std": round(df["kt_cs"].std(), 3),
        })
    rep = pd.DataFrame(rows)
    rep.to_csv("data/enrichment_report.csv", index=False)
    print(rep.to_string(index=False))
    print(f"\nEnriched files in: {OUT_DIR}")
    print("Next: scripts/03_extract_alphaearth.py")

if __name__ == "__main__":
    main()
