"""
Power analysis as a COLD kill-decision for the multi-network study.

Question: is scaling AlphaEarth-vs-geo to more networks worth it? Not just
"is the pilot effect detectable", but "is it USEFUL-sized at achievable n".

All analysis on the PRE-REGISTERED primary target (detrended mean clear-sky
index, per-fold detrend, RidgeCV-GCV, spatial-block CV) reusing the locked
machinery from scripts/08 + scripts/11. No new data; no re-implementation of the
detrend/CV.

Note: RESEARCH_PLAN.md is not present in the repo; context taken from CLAUDE.md
and the spatial_holdout/FINDINGS.md history (core wins 22/28, added 3/10).

  PART A  effect-size characterization on the 38 pilot stations
  PART B  subsample power curve (optimistic = all-38 dist, pessimistic = added-10)
  PART C  usefulness threshold (>0 bar vs 25%-of-geo "generates a lead" bar)
  PART D  three-world verdict, stated plainly

Outputs:
  data/power_analysis.csv
  data/figs/power_curves.png
"""
from __future__ import annotations
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon

OUT_CSV   = Path("data/power_analysis.csv")
FIGS      = Path("data/figs"); FIGS.mkdir(parents=True, exist_ok=True)
POOLED    = Path("spatial_holdout/pooled_stations.csv")

RANDOM_STATE = 42
K_FIXED      = 4
N_BOOT_A     = 10_000          # PART A effect-size CIs
N_SIM_B      = 2_000           # simulated studies per n
N_GRID       = [20, 30, 40, 60, 80, 120, 160]
WIN_THRESH   = 0.60            # decision rule: per-station win rate > 60% ...
P_THRESH     = 0.01            # ... AND Wilcoxon one-sided p < 0.01
USEFUL_FRAC  = 0.25            # "generates a lead": >= 25% of geo baseline error


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

s08 = _load("scripts/08_layer2_spatial_block_perfold.py", "s08")
s11 = _load("scripts/11_pool_retest.py", "s11")


# ---------------------------------------------------------------------------
# Locked pipeline -> per-station geo & emb errors on all 38 stations
# ---------------------------------------------------------------------------

def per_station_errors():
    pooled = pd.read_csv(POOLED, dtype={"station_id": str})
    df, E = s11.build_frame(pooled)
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
    err, _ = s08.cv_perfold_detrend(feats, kt, Xg, folds)
    df["geo_err"] = err["geo"]
    df["emb_err"] = err["emb"]
    df["emb_advantage"] = err["geo"] - err["emb"]
    return df


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def boot_ci(values, stat, n_boot, rng, lo=2.5, hi=97.5, draw_n=None):
    vals = np.asarray(values)
    m = draw_n or len(vals)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat(rng.choice(vals, size=m, replace=True))
    return float(np.percentile(stats, lo)), float(np.percentile(stats, hi))


def decision_fires(sample):
    """Pre-registered rule: win rate > 60% AND Wilcoxon one-sided p < 0.01."""
    win = float(np.mean(sample > 0))
    if win <= WIN_THRESH:
        return False
    try:
        _, p = wilcoxon(sample, alternative="greater", zero_method="wilcox")
    except ValueError:
        return False
    return p < P_THRESH


# ---------------------------------------------------------------------------
# PART A
# ---------------------------------------------------------------------------

def part_a(df, rng):
    adv = df["emb_advantage"].values
    geo_iqr = np.subtract(*np.percentile(df["geo_err"], [75, 25]))

    def report(name, sub):
        a = sub["emb_advantage"].values
        med = float(np.median(a))
        ci = boot_ci(a, np.median, N_BOOT_A, rng)
        std_eff = med / geo_iqr
        win = float(np.mean(a > 0))
        win_ci = boot_ci(a, lambda v: np.mean(v > 0), N_BOOT_A, rng)
        print(f"  {name:8s} (n={len(a):2d}): "
              f"median adv = {med:+.5f}  CI[{ci[0]:+.5f},{ci[1]:+.5f}]  "
              f"std_eff = {std_eff:+.2f}  "
              f"win = {win:.2f}  CI[{win_ci[0]:.2f},{win_ci[1]:.2f}]")
        return med, ci, win

    print("PART A — effect size on pilot data "
          f"(geo-error IQR = {geo_iqr:.5f} used as std denominator)")
    full  = report("ALL-38", df)
    core  = report("CORE-28", df[df["in_core"]])
    added = report("ADDED-10", df[~df["in_core"]])
    return {"geo_iqr": geo_iqr, "all": full, "core": core, "added": added}


# ---------------------------------------------------------------------------
# PART B + C  (one pass: power + emb_advantage CI per n, both distributions)
# ---------------------------------------------------------------------------

def power_and_ci(source_vals, rng):
    """For each n: power (fraction of sims firing) and median-advantage CI."""
    out = {}
    for n in N_GRID:
        fires = np.empty(N_SIM_B, dtype=bool)
        meds  = np.empty(N_SIM_B)
        for s in range(N_SIM_B):
            sample = rng.choice(source_vals, size=n, replace=True)
            fires[s] = decision_fires(sample)
            meds[s]  = np.median(sample)
        out[n] = {
            "power": float(fires.mean()),
            "med":   float(np.median(meds)),
            "ci_low":  float(np.percentile(meds, 2.5)),
            "ci_high": float(np.percentile(meds, 97.5)),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_STATE)
    df = per_station_errors()

    a = part_a(df, rng)
    geo_med = float(np.median(df["geo_err"]))
    useful_bar = USEFUL_FRAC * geo_med

    all_adv   = df["emb_advantage"].values
    added_adv = df[~df["in_core"]]["emb_advantage"].values

    opt = power_and_ci(all_adv, rng)        # optimistic: same dist as pilot
    pes = power_and_ci(added_adv, rng)      # pessimistic: weaker added-10 dist

    # CSV (emb_adv columns are the PESSIMISTIC = decision-relevant distribution)
    rows = []
    for n in N_GRID:
        clears = pes[n]["ci_low"] >= useful_bar
        rows.append({
            "n": n,
            "optimistic_power":  round(opt[n]["power"], 3),
            "pessimistic_power": round(pes[n]["power"], 3),
            "emb_adv_median":    round(pes[n]["med"], 6),
            "emb_adv_ci_low":    round(pes[n]["ci_low"], 6),
            "emb_adv_ci_high":   round(pes[n]["ci_high"], 6),
            "clears_usefulness_bar": bool(clears),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    # ---- PART C: usefulness thresholds -----------------------------------
    print(f"\nPART C — usefulness (geo baseline median error = {geo_med:.5f})")
    print(f"  zero bar         = 0.0")
    print(f"  25%-of-geo bar   = {useful_bar:.5f}  (\"generates a lead\")")

    def first_n(curve, bar):
        for n in N_GRID:
            if curve[n]["ci_low"] >= bar:
                return n
        return None

    print("\n  CI lower bound on median emb_advantage by n:")
    print(f"  {'n':>4} {'OPT ci_low':>11} {'>0?':>4} {'>bar?':>6}   "
          f"{'PESS ci_low':>11} {'>0?':>4} {'>bar?':>6}")
    for n in N_GRID:
        o, p = opt[n]["ci_low"], pes[n]["ci_low"]
        print(f"  {n:>4} {o:>+11.5f} {'Y' if o>0 else 'n':>4} "
              f"{'Y' if o>=useful_bar else 'n':>6}   "
              f"{p:>+11.5f} {'Y' if p>0 else 'n':>4} "
              f"{'Y' if p>=useful_bar else 'n':>6}")

    opt_n_zero = first_n(opt, 0.0);  opt_n_bar = first_n(opt, useful_bar)
    pes_n_zero = first_n(pes, 0.0);  pes_n_bar = first_n(pes, useful_bar)
    print(f"\n  OPTIMISTIC : CI low clears 0 at n={opt_n_zero}, "
          f"clears 25%-of-geo at n={opt_n_bar}")
    print(f"  PESSIMISTIC: CI low clears 0 at n={pes_n_zero}, "
          f"clears 25%-of-geo at n={pes_n_bar}")

    # ---- power curve console ---------------------------------------------
    print("\nPART B — power (decision rule: win>60% AND Wilcoxon p<0.01)")
    print(f"  {'n':>4} {'optimistic':>11} {'pessimistic':>12}")
    for n in N_GRID:
        print(f"  {n:>4} {opt[n]['power']:>11.3f} {pes[n]['power']:>12.3f}")
    print("  CAVEAT: the bootstrap assumes new stations are drawn from the SAME")
    print("  distribution as the pilot. The pilot replication (added 3/10 vs core")
    print("  22/28) says they are NOT — the PESSIMISTIC curve is the honest one.")

    def reaches(curve, thr=0.8):
        for n in N_GRID:
            if curve[n]["power"] >= thr:
                return n
        return None
    opt_n80 = reaches(opt); pes_n80 = reaches(pes)

    # ---- PART D: verdict --------------------------------------------------
    pes_useful_at_power = (pes_n80 is not None and pes_n80 <= 80 and
                           pes[pes_n80]["ci_low"] >= useful_bar)
    world1 = pes_useful_at_power
    # significant-but-not-useful, or power only at very large n
    never_useful = (opt_n_bar is None and pes_n_bar is None)
    world2 = (not world1) and (
        (pes_n80 is None or pes_n80 >= 150) or never_useful) and (
        opt_n80 is not None and not never_useful or never_useful)
    # mostly-answered: optimistic also struggles or effect washes out
    opt_struggles = (opt_n80 is None or opt_n80 > 160)
    washes_out = (opt[max(N_GRID)]["ci_low"] <= 0)
    world3 = opt_struggles or washes_out

    print("\n" + "=" * 72)
    print("PART D — VERDICT")
    print("=" * 72)
    if world1:
        verdict = "WORLD 1 — BUILD IT"
        why = (f"pessimistic power reaches 0.8 by n={pes_n80} (<=80) AND its "
               f"emb_advantage CI low clears the 25%-of-geo bar ({useful_bar:.5f}).")
    elif world3 and not world2:
        verdict = "WORLD 3 — MOSTLY ANSWERED"
        why = (f"even the OPTIMISTIC power curve "
               f"{'never reaches 0.8 in range' if opt_struggles else 'reaches 0.8 only late'} "
               f"(0.8 at n={opt_n80}) "
               f"{'and the effect washes out (optimistic CI low <= 0 at n=160)' if washes_out else ''}; "
               f"pessimistic power ~{pes[max(N_GRID)]['power']:.2f} at n=160. The "
               f"question has largely answered itself.")
    else:
        verdict = "WORLD 2 — SCOPE DOWN"
        why = (f"effect is at best significant-but-not-useful: optimistic 0.8 "
               f"power at n={opt_n80}, pessimistic at n={pes_n80}; the "
               f"emb_advantage CI lower bound "
               f"{'never' if never_useful else 'does not reliably'} clears the "
               f"25%-of-geo usefulness bar ({useful_bar:.5f}) "
               f"(optimistic clears at n={opt_n_bar}, pessimistic at n={pes_n_bar}). "
               f"Detectable with enough n, but not lead-generating.")
    print(f"\n{verdict}\n\n  Why: {why}")
    print(f"\n  Key numbers: ALL-38 median adv {a['all'][0]:+.5f} "
          f"(win {a['all'][2]:.2f}), CORE {a['core'][0]:+.5f} (win {a['core'][2]:.2f}), "
          f"ADDED {a['added'][0]:+.5f} (win {a['added'][2]:.2f}); "
          f"geo median err {geo_med:.5f}, useful bar {useful_bar:.5f}.")

    _plot(opt, pes, useful_bar, verdict)
    _write_findings(a, geo_med, useful_bar, opt, pes, opt_n80, pes_n80,
                    opt_n_bar, pes_n_bar, verdict, why)
    print(f"\nSaved: {OUT_CSV}, {FIGS/'power_curves.png'}")
    return verdict


def _plot(opt, pes, useful_bar, verdict):
    ns = N_GRID
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.plot(ns, [opt[n]["power"] for n in ns], "o-", color="#4878d0",
             label="optimistic (pilot dist)")
    ax1.plot(ns, [pes[n]["power"] for n in ns], "^-", color="#d65f5f",
             label="pessimistic (added-10 dist)")
    ax1.axhline(0.8, color="k", ls="--", lw=0.9, alpha=0.6, label="0.8 power")
    ax1.set_xlabel("n stations"); ax1.set_ylabel("power (decision rule fires)")
    ax1.set_ylim(-0.03, 1.03); ax1.set_title("Subsample power vs n")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    for cur, col, lab, mk in [(opt, "#4878d0", "optimistic", "o"),
                              (pes, "#d65f5f", "pessimistic", "^")]:
        med = [cur[n]["med"] for n in ns]
        lo  = [cur[n]["ci_low"] for n in ns]
        hi  = [cur[n]["ci_high"] for n in ns]
        ax2.plot(ns, med, mk + "-", color=col, label=f"{lab} median")
        ax2.fill_between(ns, lo, hi, color=col, alpha=0.15)
    ax2.axhline(0.0, color="k", ls="-", lw=0.8, alpha=0.6, label="zero bar")
    ax2.axhline(useful_bar, color="green", ls="--", lw=1.0,
                label=f"25%-of-geo bar ({useful_bar:.4f})")
    ax2.set_xlabel("n stations"); ax2.set_ylabel("median emb_advantage (95% CI)")
    ax2.set_title("Effect size & precision vs n"); ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle(f"Power / usefulness decision instrument — {verdict}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "power_curves.png", dpi=130)
    plt.close(fig)


def _write_findings(a, geo_med, useful_bar, opt, pes, opt_n80, pes_n80,
                    opt_n_bar, pes_n_bar, verdict, why):
    today = date.today().isoformat()
    L = [f"\n## {today} — Power analysis: kill-decision for multi-network study\n",
         f"Decision instrument on the 38 pilot stations (locked per-fold-detrend "
         f"spatial-block pipeline). Nonparametric-bootstrap power for the rule "
         f"*win-rate>60% AND Wilcoxon one-sided p<0.01*, plus a usefulness bar "
         f"(median emb_advantage ≥ 25% of the geo baseline error).\n",
         f"\n**Effect size (PART A):** ALL-38 median advantage "
         f"{a['all'][0]:+.5f} (win {a['all'][2]:.2f}); CORE {a['core'][0]:+.5f} "
         f"(win {a['core'][2]:.2f}); ADDED {a['added'][0]:+.5f} "
         f"(win {a['added'][2]:.2f}). Geo baseline median error {geo_med:.5f} → "
         f"25%-of-geo usefulness bar = **{useful_bar:.5f}**.\n",
         "\n**Power (PART B) & usefulness (PART C):**\n",
         "| n | opt power | pess power | opt CI-low | pess CI-low |",
         "|---|---|---|---|---|"]
    for n in N_GRID:
        L.append(f"| {n} | {opt[n]['power']:.2f} | {pes[n]['power']:.2f} | "
                 f"{opt[n]['ci_low']:+.5f} | {pes[n]['ci_low']:+.5f} |")
    L += [f"\n- Optimistic power reaches 0.8 at n={opt_n80}; pessimistic at n={pes_n80}.",
          f"- Optimistic CI-low clears the 25%-of-geo bar at n={opt_n_bar}; "
          f"pessimistic at n={pes_n_bar}.",
          f"\n**VERDICT: {verdict}**\n\n{why}\n",
          "\nThe pessimistic curve (new stations ~ the weaker added-10 sample) is "
          "the honest one given the pilot's replication failure. "
          "Fig: data/figs/power_curves.png.\n"]
    with open("spatial_holdout/FINDINGS.md", "a") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
