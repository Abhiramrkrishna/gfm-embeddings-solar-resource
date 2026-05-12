"""
Download DWD hourly solar radiation data for our 28 core stations.

Source: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/solar/
License: CC-BY 4.0
Variables of interest:
  FG_LBERG : hourly sum of global solar radiation (J/cm^2)
  FD_LBERG : hourly sum of diffuse radiation (J/cm^2)
  SD_LBERG : hourly sum of sunshine duration (min)
  ZENIT    : solar zenith angle at mid-interval (degrees)
Missing value: -999
Units: J/cm^2  ->  multiply by (10000/3600) to get W/m^2 (hourly mean)
"""
from __future__ import annotations
import io
import re
import zipfile
import time
import pandas as pd
import requests
from pathlib import Path

# ---- Config ----
DWD_BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/solar"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_dwd"
PARQUET_DIR = DATA_DIR / "stations"
STATIONS_CSV = "dwd_core_stations.csv"  # produced by parse_stations.py

# Hourly solar radiation is in a single archive per station, named
# stundenwerte_ST_{station_id}_row.zip (current period)
# stundenwerte_ST_{station_id}_{from}_{to}_hist.zip (historical period)
# We try the "row" (recent) archive first; for full 2020-2024 coverage we
# typically also need the historical one. The directory listing tells us which.

def list_remote_zips() -> dict[str, list[str]]:
    """Walk the DWD opendata directory for the hourly/solar product.

    Returns: {station_id: [zip_filename, ...]} for each available archive.
    Notes: DWD splits files into 'historical' and 'recent' folders nowadays.
    """
    folders = {"recent": f"{DWD_BASE}/recent/", "historical": f"{DWD_BASE}/historical/"}
    by_station: dict[str, list[str]] = {}
    for label, url in folders.items():
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            # Fallback: some products have everything at top-level
            r = requests.get(DWD_BASE + "/", timeout=30)
            r.raise_for_status()
        # Parse the Apache directory listing for .zip files
        zips = re.findall(r'href="([^"]+\.zip)"', r.text)
        for z in zips:
            # Filename example: stundenwerte_ST_00183_row.zip
            #                   stundenwerte_ST_00691_20090101_20231231_hist.zip
            m = re.search(r"ST_(\d{5})", z)
            if not m:
                continue
            sid = m.group(1)
            by_station.setdefault(sid, []).append(url + z)
    return by_station

def download_station(station_id: str, zip_urls: list[str]) -> Path:
    """Download all archives for a station into RAW_DIR/{station_id}/."""
    out = RAW_DIR / station_id
    out.mkdir(parents=True, exist_ok=True)
    for url in zip_urls:
        fname = url.rsplit("/", 1)[-1]
        target = out / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        target.write_bytes(r.content)
        time.sleep(0.2)  # be polite to DWD
    return out

def parse_station_zip(zip_path: Path) -> pd.DataFrame:
    """Extract the produkt_st_stunde_*.txt CSV from a DWD station zip."""
    with zipfile.ZipFile(zip_path) as zf:
        produkt_files = [n for n in zf.namelist() if n.lower().startswith("produkt") and n.endswith(".txt")]
        if not produkt_files:
            return pd.DataFrame()
        with zf.open(produkt_files[0]) as f:
            df = pd.read_csv(f, sep=";", na_values=["-999", -999], dtype={"STATIONS_ID": str})
    df.columns = [c.strip() for c in df.columns]
    # Strip trailing 'eor' marker column if present
    df = df.drop(columns=[c for c in df.columns if c.lower() == "eor"], errors="ignore")
    # Parse the timestamp (UTC, YYYYMMDDHHmm or YYYYMMDDHH)
    ts = df["MESS_DATUM"].astype(str).str.zfill(10)
    df["timestamp_utc"] = pd.to_datetime(ts, format="%Y%m%d%H", utc=True, errors="coerce")
    return df

def jcm2_to_wm2(jcm2: pd.Series) -> pd.Series:
    """Convert hourly sum in J/cm^2 to mean W/m^2 over the hour.
    1 J/cm^2 = 1e4 J/m^2; divide by 3600 s -> W/m^2."""
    return jcm2 * (1e4 / 3600.0)

def build_station_parquet(station_id: str) -> Path:
    """Assemble all archives for a station, filter to 2020-2024, save parquet."""
    raw_dir = RAW_DIR / station_id
    frames = []
    for zip_path in sorted(raw_dir.glob("*.zip")):
        df = parse_station_zip(zip_path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    # Filter to project window
    mask = (df["timestamp_utc"] >= "2020-01-01") & (df["timestamp_utc"] < "2025-01-01")
    df = df.loc[mask].copy()
    # Convert to W/m^2 for human-readable downstream use
    for col in ("FG_LBERG", "FD_LBERG", "ATMO_LBERG"):
        if col in df.columns:
            df[col.replace("_LBERG", "_WM2")] = jcm2_to_wm2(df[col])
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    out = PARQUET_DIR / f"{station_id}.parquet"
    df.to_parquet(out, index=False)
    return out

def coverage_report(station_id: str) -> dict:
    """Report hours expected vs. observed for the 2020-2024 window."""
    pq = PARQUET_DIR / f"{station_id}.parquet"
    if not pq.exists():
        return {"station_id": station_id, "status": "missing"}
    df = pd.read_parquet(pq)
    expected_hours = (pd.Timestamp("2025-01-01", tz="UTC") - pd.Timestamp("2020-01-01", tz="UTC")).total_seconds() / 3600
    observed = df["timestamp_utc"].notna().sum()
    fg_valid = df["FG_LBERG"].notna().sum() if "FG_LBERG" in df else 0
    return {
        "station_id": station_id,
        "n_rows": observed,
        "n_hours_expected": int(expected_hours),
        "fg_valid_hours": int(fg_valid),
        "fg_valid_pct": round(100 * fg_valid / expected_hours, 1),
        "first": str(df["timestamp_utc"].min()),
        "last": str(df["timestamp_utc"].max()),
    }

def main():
    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})
    print(f"Targeting {len(stations)} core stations")
    print("Discovering remote archives...")
    remote = list_remote_zips()
    print(f"Found archives for {len(remote)} stations on DWD opendata")
    reports = []
    for _, row in stations.iterrows():
        sid = row["station_id"].zfill(5)
        urls = remote.get(sid, [])
        if not urls:
            print(f"  [{sid}] no archives found")
            reports.append({"station_id": sid, "status": "no_archives"})
            continue
        print(f"  [{sid}] {row['name']:30s} downloading {len(urls)} archive(s)...")
        download_station(sid, urls)
        build_station_parquet(sid)
        reports.append(coverage_report(sid))
    rep = pd.DataFrame(reports)
    rep.to_csv("data/coverage_report.csv", index=False)
    print("\nCoverage report saved to data/coverage_report.csv")
    print(rep.to_string(index=False))

if __name__ == "__main__":
    main()
