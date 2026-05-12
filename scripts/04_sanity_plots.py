"""
Week 1 sanity-check plots. Goal: confirm the data pipeline works end-to-end,
the units are right, and the AlphaEarth embeddings actually differ between
geographically distinct stations.

If station-pair embeddings look identical, something is wrong with extraction.
If kt_cs looks unphysical (e.g., median far from ~0.6-0.8), the unit conversion
or clear-sky model has a bug.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ENRICHED_DIR = Path("data/stations_enriched")
AE_DIR = Path("data/alphaearth")
STATIONS_CSV = "dwd_core_stations.csv"
OUT_DIR = Path("data/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_stations() -> pd.DataFrame:
    return pd.read_csv(STATIONS_CSV, dtype={"station_id": str})

def load_enriched(sid: str) -> pd.DataFrame:
    p = ENRICHED_DIR / f"{sid}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)

def load_embedding(sid: str, year: int) -> np.ndarray | None:
    p = AE_DIR / f"{sid}_{year}.npy"
    if not p.exists():
        return None
    return np.load(p)

def plot_kt_cs_distributions(stations: pd.DataFrame):
    """Histogram of kt_cs for each station -- should peak somewhere between
    0.5 and 0.9 with a long tail; very different distributions across sites
    would already suggest local effects worth modelling."""
    n = len(stations)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (_, s) in zip(axes, stations.iterrows()):
        sid = s["station_id"].zfill(5)
        df = load_enriched(sid)
        if df.empty:
            ax.set_axis_off()
            continue
        vals = df["kt_cs"].dropna()
        ax.hist(vals, bins=60, color="steelblue", alpha=0.8)
        ax.set_title(f"{sid} {s['name'][:18]}\nN={len(vals)} mean={vals.mean():.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle("Clear-sky index (kt_cs) per station, 2020-2024 daytime hours", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "kt_cs_distributions.png", dpi=120)
    plt.close(fig)

def plot_embedding_signature(stations: pd.DataFrame, year: int = 2023):
    """Visualize the 64-d embedding for each station as a horizontal bar.
    Stations far apart geographically should show visibly different signatures."""
    rows = []
    for _, s in stations.iterrows():
        sid = s["station_id"].zfill(5)
        emb = load_embedding(sid, year)
        if emb is None:
            continue
        rows.append((sid, s["name"], emb))
    if not rows:
        print("No embeddings found yet; run scripts/03_extract_alphaearth.py first.")
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(rows))))
    arr = np.stack([r[2] for r in rows])
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]} {r[1][:18]}" for r in rows], fontsize=8)
    ax.set_xlabel("Embedding dimension (A00..A63)")
    ax.set_title(f"AlphaEarth Foundations embeddings, year={year}, 100m buffer")
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"embeddings_{year}.png", dpi=120)
    plt.close(fig)

def plot_embedding_pca(stations: pd.DataFrame, year: int = 2023):
    """PCA the embeddings into 2D and colour by latitude / elevation.
    If AlphaEarth carries terrain/latitude signal as we'd hope, we should see
    structure in this scatter."""
    from sklearn.decomposition import PCA
    embs, sids, lats, elevs = [], [], [], []
    for _, s in stations.iterrows():
        sid = s["station_id"].zfill(5)
        emb = load_embedding(sid, year)
        if emb is None:
            continue
        embs.append(emb)
        sids.append(sid)
        lats.append(float(s["lat"]))
        elevs.append(float(s["elev_m"]))
    if len(embs) < 3:
        print("Not enough embeddings for PCA")
        return
    X = np.stack(embs)
    pcs = PCA(n_components=2).fit_transform(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sc1 = ax1.scatter(pcs[:, 0], pcs[:, 1], c=lats, cmap="viridis", s=60)
    ax1.set_title("PC1 vs PC2 coloured by latitude")
    fig.colorbar(sc1, ax=ax1, label="lat")
    sc2 = ax2.scatter(pcs[:, 0], pcs[:, 1], c=elevs, cmap="plasma", s=60)
    ax2.set_title("PC1 vs PC2 coloured by elevation (m)")
    fig.colorbar(sc2, ax=ax2, label="elev")
    for ax in (ax1, ax2):
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(f"AlphaEarth embedding PCA across {len(embs)} DWD stations, year={year}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"embedding_pca_{year}.png", dpi=120)
    plt.close(fig)

def main():
    stations = load_stations()
    print(f"Loaded {len(stations)} stations")
    print("Plotting kt_cs distributions...")
    plot_kt_cs_distributions(stations)
    print("Plotting embedding signatures...")
    plot_embedding_signature(stations, year=2023)
    print("Plotting embedding PCA...")
    plot_embedding_pca(stations, year=2023)
    print(f"\nFigures saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
