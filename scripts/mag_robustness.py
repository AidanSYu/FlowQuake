"""Magnitude-robustness of the temporal win.

A reviewer concern for the dense-catalog temporal result is that FlowQuake's gain
over ETAS could be an artifact of magnitude-scale heterogeneity (catalogs mix ML,
Md, Mw, mb with different scaling). A direct, model-internal test: stratify the
paired per-event temporal gain dT = tll_FlowQuake - TLL_ETAS by event magnitude.
If the win is an artifact of one magnitude band (e.g. the smallest events, where
magnitude-type conversion errors are largest), dT would be concentrated there;
if it is broad-based, the win is robust to magnitude scale.

We merge FlowQuake per-event temporal LLs (by timestamp) with the ETAS augmented
catalog (which carries magnitude and TLL), bin dT by magnitude, and report the
mean and a stationary block-bootstrap 95% CI per bin (blocks preserve aftershock
autocorrelation), plus the Spearman correlation of dT with magnitude.

Run: python scripts/mag_robustness.py  -> runs/mag_robustness.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.stats import stationary_block_bootstrap_ci

ER = Path("reference/Experiments/ETAS")
REGIONS = {
    "California": dict(etas="output_data_ComCat_25", fq="runs/n1_density/per_event_test.csv", mc=2.5),
    "Italy":      dict(etas="output_data_Italy_25",  fq="runs/italy_n1/per_event_test.csv",  mc=2.5),
    "Chile":      dict(etas="output_data_Chile_25",  fq="runs/chile_n1/per_event_test.csv",  mc=4.0),
}


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def analyze(etas_dir, fq_csv, mc):
    ep = ER / etas_dir / "augmented_catalog.csv"
    if not ep.exists() or not Path(fq_csv).exists():
        return None
    e = pd.read_csv(ep, parse_dates=["time"]).dropna(subset=["TLL", "magnitude"])[["time", "magnitude", "TLL"]]
    f = pd.read_csv(fq_csv, parse_dates=["time"])[["time", "tll"]]
    m = f.merge(e, on="time", how="inner").sort_values("time").reset_index(drop=True)
    if not len(m):
        return None
    m["dT"] = m["tll"] - m["TLL"]
    # magnitude bins from mc upward in 0.5 steps; merge the sparse tail
    hi = float(m["magnitude"].max())
    edges = list(np.arange(mc, min(hi, mc + 3.0) + 1e-9, 0.5)) + [hi + 1e-6]
    edges = sorted(set(round(x, 3) for x in edges))
    bins = []
    for lo, up in zip(edges[:-1], edges[1:]):
        grp = m[(m["magnitude"] >= lo) & (m["magnitude"] < up)]
        if len(grp) < 30:
            continue
        lo_ci, hi_ci = stationary_block_bootstrap_ci(grp["dT"].to_numpy(), seed=7)
        bins.append({"mag_lo": round(lo, 2), "mag_hi": round(up, 2), "n": int(len(grp)),
                     "dT": float(grp["dT"].mean()), "ci": [round(lo_ci, 4), round(hi_ci, 4)],
                     "win": bool(lo_ci > 0)})
    lo_all, hi_all = stationary_block_bootstrap_ci(m["dT"].to_numpy(), seed=7)
    return {"n": int(len(m)), "dT_overall": float(m["dT"].mean()),
            "dT_overall_ci": [round(lo_all, 4), round(hi_all, 4)],
            "spearman_dT_vs_mag": round(spearman(m["magnitude"].to_numpy(), m["dT"].to_numpy()), 4),
            "bins": bins}


def main():
    res = {}
    for reg, c in REGIONS.items():
        a = analyze(c["etas"], c["fq"], c["mc"])
        if a is None:
            print(f"{reg}: no data"); continue
        res[reg] = a
        print(f"\n{reg}: N={a['n']}  overall dT={a['dT_overall']:+.4f} "
              f"CI[{a['dT_overall_ci'][0]:+.3f},{a['dT_overall_ci'][1]:+.3f}]  "
              f"Spearman(dT,mag)={a['spearman_dT_vs_mag']:+.3f}")
        print(f"  {'mag bin':>12}{'n':>7}{'dT':>9}{'95% CI':>20}{'win':>5}")
        for b in a["bins"]:
            ci = f"[{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]"
            print(f"  {b['mag_lo']:>5}-{b['mag_hi']:<5} {b['n']:>6}{b['dT']:>+9.4f}{ci:>20}{'*' if b['win'] else '':>5}")
    Path("runs").mkdir(exist_ok=True)
    json.dump(res, open("runs/mag_robustness.json", "w"), indent=2)
    print("\nwrote runs/mag_robustness.json  (* = bin temporal gain CI strictly > 0)")


if __name__ == "__main__":
    main()
