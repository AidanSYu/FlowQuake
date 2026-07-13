"""Figure: ONE pooled global FlowQuake vs each region's own ETAS inversion.

Bars = temporal log-likelihood gain over region-fitted ETAS with one shared
checkpoint and no per-region weight fitting after pooling. Data-poor regions
(Greece, Iran) also show the brief fine-tune value. Above 0 means the pooled
deployment model beats locally fitted ETAS temporally.

Run: python scripts/make_global_figure.py  -> figures/global_vs_etas.png
"""
import json
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.stats import paired_gain_summary

ER = Path("reference/Experiments/ETAS")
# region -> (regime, mc, ETAS dir, global pooled per-event, global few-shot per-event or None)
R = {
    "California": ("transform", 2.5, "output_data_ComCat_25", "runs/global_California_per_event.csv", None),
    "Italy":      ("extension", 2.5, "output_data_Italy_25",   "runs/global_Italy_per_event.csv", None),
    "Japan":      ("subduction", 4.0, "output_data_Japan_25",  "runs/global_Japan_per_event.csv", None),
    "Chile":      ("subduction", 4.0, "output_data_Chile_25",  "runs/global_Chile_per_event.csv", None),
    "Greece":     ("extension", 4.0, "output_data_Greece_25",  "runs/global_Greece_per_event.csv", "runs/global_few_greece_n1/per_event_test.csv"),
    "Iran":       ("collision", 4.0, "output_data_Iran_25",    "runs/global_Iran_per_event.csv", "runs/global_few_iran_n1/per_event_test.csv"),
}
COL = {"transform": "#2980B9", "extension": "#E67E22", "subduction": "#27AE60", "collision": "#8E44AD"}


def dT(fq, ed):
    f, a = Path(fq), ER / ed / "augmented_catalog.csv"
    if not f.exists() or not a.exists():
        return None, None
    m = pd.read_csv(f, parse_dates=["time"]).merge(
        pd.read_csv(a, parse_dates=["time"]).dropna(subset=["TLL"])[["time", "TLL"]], on="time", how="inner")
    d = m["tll"] - m["TLL"]
    g = paired_gain_summary(d, seed=sum(ord(ch) for ch in fq) % 10000)
    return g.mean, g.ci_low, g.ci_high, g.decision


def main():
    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels, names = [], list(R)
    for i, reg in enumerate(names):
        regime, mc, ed, zs, fs = R[reg]
        gz, loz, hiz, _ = dT(zs, ed)
        if gz is None:
            continue
        ax.bar(i, gz, 0.62, color=COL[regime],
               yerr=[[gz - loz], [hiz - gz]], capsize=3,
               label=regime if regime not in labels else None)
        labels.append(regime)
        # few-shot marker for data-poor regions
        if fs:
            gf, lof, hif, _ = dT(fs, ed)
            if gf is not None:
                ax.plot(i, gf, "k_", ms=18, mew=2)
                ax.annotate("+few-shot", (i, gf), fontsize=6.5, ha="center",
                            xytext=(0, 6 if gf > gz else -10), textcoords="offset points")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{r}\n({R[r][0]}, mc{R[r][1]})" for r in names], fontsize=7.5)
    ax.set_ylabel("temporal gain over region-fitted ETAS (nats/event)")
    ax.set_title("ONE pooled FlowQuake checkpoint vs each region's own ETAS inversion\n"
                 "positive mean = higher temporal LL; error bars = block-bootstrap CI")
    ax.legend(title="regime", frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig("figures/global_vs_etas.png", dpi=200)
    print("wrote figures/global_vs_etas.png")


if __name__ == "__main__":
    main()
