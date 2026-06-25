"""Stress-test the Japan temporal result: is FlowQuake's temporal gain over ETAS
real and broad, or a Tohoku-sequence artifact? Split the test window into the
Tohoku aftershock sequence (2011-03-11 .. 2011-09-11, 6 months) vs the rest,
and report paired temporal/spatial/total gains in each. Also confirm the
FlowQuake and ETAS event sets are identical (fair paired comparison).

Run: python scripts/japan_verify.py
"""
from pathlib import Path
import numpy as np, pandas as pd

ETAS = Path("reference/Experiments/ETAS/output_data_Japan_25")
TOHOKU = pd.Timestamp("2011-03-11"); TOHOKU_END = pd.Timestamp("2011-09-11")


def main():
    aug = pd.read_csv(ETAS / "augmented_catalog.csv", parse_dates=["time"]).dropna(subset=["TLL", "SLL"])
    for name, path in [("zero-shot", "runs/transfer_japan_per_event.csv"),
                       ("few-shot", "runs/japan_fewshot/per_event_test.csv"),
                       ("native", "runs/japan_n1/per_event_test.csv")]:
        fq = pd.read_csv(path, parse_dates=["time"])
        m = fq.merge(aug[["time", "TLL", "SLL"]], on="time", how="inner")
        # alignment check
        only_fq = len(fq) - len(m); only_etas = len(aug) - len(m)
        m["dt"] = m["tll"] - m["TLL"]; m["ds"] = m["sll"] - m["SLL"]
        toh = (m["time"] >= TOHOKU) & (m["time"] < TOHOKU_END)
        print(f"\n=== {name} === matched {len(m)}  (FQ-only {only_fq}, ETAS-only {only_etas})")
        for lab, sub in [("ALL", m), ("Tohoku 6mo", m[toh]), ("rest", m[~toh])]:
            if not len(sub):
                continue
            dt, ds = sub["dt"], sub["ds"]
            se_t = dt.std(ddof=1)/np.sqrt(len(dt)); se_s = ds.std(ddof=1)/np.sqrt(len(ds))
            print(f"  {lab:11} n={len(sub):>6}  dT {dt.mean():+.4f}+-{se_t:.4f} (win {(dt>0).mean():.1%})"
                  f"   dS {ds.mean():+.4f}   dTot {(dt+ds).mean():+.4f}")
    # how much of the test is the Tohoku sequence?
    tall_m = pd.read_csv("runs/japan_n1/per_event_test.csv", parse_dates=["time"]).merge(
        aug[["time"]], on="time")
    toh_frac = ((tall_m := tall_m if False else tall_m)["time"].between(TOHOKU, TOHOKU_END)).mean() \
        if False else (tall_m["time"].between(TOHOKU, TOHOKU_END)).mean()
    print(f"\nTohoku 6-mo window is {toh_frac:.1%} of test events")


if __name__ == "__main__":
    main()
