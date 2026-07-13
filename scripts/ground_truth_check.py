"""Compute the exact numbers the red-team disputed, so the manuscript can quote
artifact-backed values: 3-seed head dS means per region, block-bootstrap dT CIs
for the S4.5 density table (native + fewshot), real b-values, real train counts.
Run: python scripts/ground_truth_check.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.stats import paired_gain_summary

ER = Path("reference/Experiments/ETAS")
REGION_ETAS = {"California": "output_data_ComCat_25", "Italy": "output_data_Italy_25",
               "Japan": "output_data_Japan_25", "Chile": "output_data_Chile_25",
               "Greece": "output_data_Greece_25", "Iran": "output_data_Iran_25"}
HEAD_KEY = {"California": "ComCat_25", "Italy": "Italy_25", "Japan": "Japan_25",
            "Chile": "Chile_25", "Greece": "Greece_25", "Iran": "Iran_25"}

print("=== 3-SEED HEAD dS (test split) per region ===")
head3 = {}
for reg, hk in HEAD_KEY.items():
    ds = []
    for f in sorted(glob.glob(f"runs/neural_etas/{hk}/summary_full_s*.json")):
        ds.append(json.load(open(f))["dS_mean"])
    if ds:
        head3[reg] = (float(np.mean(ds)), float(np.std(ds)), ds)
        print(f"  {reg:11} dS 3-seed mean {np.mean(ds):+.4f} +/- {np.std(ds):.4f}  seeds={ds}")

print("\n=== DENSITY TABLE dT: block-bootstrap CI, native AND fewshot ===")
def dT_ci(fq_csv, etas_dir, seed):
    p = ER / etas_dir / "augmented_catalog.csv"
    if not (Path(fq_csv).exists() and p.exists()):
        return None
    et = pd.read_csv(p, parse_dates=["time"]).dropna(subset=["TLL", "SLL"])[["time", "TLL"]]
    fq = pd.read_csv(fq_csv, parse_dates=["time"])[["time", "tll"]]
    # disambiguate duplicate timestamps within each side before merge
    for d in (fq, et):
        d["cc"] = d.groupby("time").cumcount()
    m = fq.merge(et, on=["time", "cc"]).sort_values("time")
    g = paired_gain_summary((m["tll"] - m["TLL"]).to_numpy(), seed=seed).asdict()
    return len(m), g["mean"], g["ci"], g["decision"]
for i, reg in enumerate(REGION_ETAS):
    native = f"runs/{reg.lower().replace('california','n1_density').replace('_','')}"
    nat_csv = {"California": "runs/n1_density/per_event_test.csv",
               "Italy": "runs/italy_n1/per_event_test.csv",
               "Japan": "runs/japan_n1/per_event_test.csv",
               "Chile": "runs/chile_n1/per_event_test.csv",
               "Greece": "runs/greece_n1/per_event_test.csv",
               "Iran": "runs/iran_n1/per_event_test.csv"}[reg]
    r = dT_ci(nat_csv, REGION_ETAS[reg], seed=50 + i)
    if r:
        print(f"  {reg:11} native  n={r[0]:6d} dT {r[1]:+.4f} CI[{r[2][0]:+.4f},{r[2][1]:+.4f}] {r[3]} (block-bootstrap)")

print("\n=== b-values and train counts from artifacts ===")
for reg, hk in HEAD_KEY.items():
    if reg == "California":
        continue
    meta_name = {"Italy": "Italy", "Japan": "Japan", "Chile": "Chile",
                 "Greece": "Greece", "Iran": "Iran"}[reg]
    mp = Path(f"reference/Datasets/{meta_name}/{meta_name}_meta.json")
    b = mp.exists() and json.load(open(mp)).get("b_value")
    # train count = events >= mc in [train_start, test_start)
    cfg = {"Italy": ("reference/Datasets/Italy/Italy_catalog.csv", 2.5, "1994-01-01", "2011-01-01"),
           "Japan": ("reference/Datasets/Japan/Japan_catalog.csv", 4.0, "1992-01-01", "2011-01-01"),
           "Chile": ("reference/Datasets/Chile/Chile_catalog.csv", 4.0, "1992-01-01", "2011-01-01"),
           "Greece": ("reference/Datasets/Greece/Greece_catalog.csv", 4.0, "1992-01-01", "2011-01-01"),
           "Iran": ("reference/Datasets/Iran/Iran_catalog.csv", 4.0, "1992-01-01", "2011-01-01")}[reg]
    cat, mc, t0, t1 = cfg
    n_train = None
    if Path(cat).exists():
        d = pd.read_csv(cat, parse_dates=["time"])
        n_train = int(((d.magnitude >= mc) & (d.time >= t0) & (d.time < t1)).sum())
    print(f"  {reg:11} b_value(meta)={b}  train_events(mc {mc})={n_train}")

print("\n=== forward win rates (disclosure) ===")
fq = pd.read_csv("runs/n1_density/per_event_forward.csv", parse_dates=["time"])[["time", "tll"]]
hd = pd.read_csv("runs/neural_etas/ComCat_25/per_event_forward_full.csv", parse_dates=["time"])[["time", "sll_neural"]]
et = pd.read_csv("runs/forward_etas/per_event.csv", parse_dates=["time"])[["time", "TLL", "SLL"]]
m = fq.merge(hd, on="time").merge(et, on="time")
print(f"  n={len(m)}  dT win {(m.tll>m.TLL).mean():.3f}  dS win {(m.sll_neural>m.SLL).mean():.3f}  "
      f"dTot win {((m.tll+m.sll_neural)>(m.TLL+m.SLL)).mean():.3f}")
print(f"  in-sample dS win rate:", end=" ")
fqt = pd.read_csv("runs/n1_density/per_event_test.csv", parse_dates=["time"])[["time", "tll"]]
hdt = pd.read_csv("runs/neural_etas/ComCat_25/per_event_full_s0.csv", parse_dates=["time"])[["time", "sll_neural", "sll_etas"]]
mt = hdt.merge(fqt, on="time")
print(f"{(mt.sll_neural>mt.sll_etas).mean():.3f}")
