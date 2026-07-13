"""Cross-regime generalization figure.

A: per region, FlowQuake temporal gain over region-fitted ETAS (dTemp, nats/event)
   for native / zero-shot transfer / few-shot. Zero line = ETAS. Shows native
   wins on data-rich regions but ties/loses on data-poor ones, while few-shot
   (brief warm-started fine-tune) rescues Greece/Iran to parity. Japan remains
   the small-loss/tie boundary case for held-out transfer.
B: data efficiency -- dTemp vs region training-set size; native collapses as
   data shrinks, transfer/few-shot stay above ETAS. ETAS cannot transfer.

Run: python scripts/make_multiregion_figure.py  -> figures/multiregion_transfer.png
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = json.load(open("runs/multiregion_master.json"))
TRAIN = {  # (catalog, mc, train_start, test_start) to count native training events
    "California": ("reference/Datasets/ComCat/ComCat_catalog.csv", 2.5, "1981-01-01", "2007-01-01"),
    "Italy": ("reference/Datasets/Italy/Italy_catalog.csv", 2.5, "1994-01-01", "2011-01-01"),
    "Japan": ("reference/Datasets/Japan/Japan_catalog.csv", 4.0, "1992-01-01", "2011-01-01"),
    "Chile": ("reference/Datasets/Chile/Chile_catalog.csv", 4.0, "1992-01-01", "2011-01-01"),
    "Greece": ("reference/Datasets/Greece/Greece_catalog.csv", 4.0, "1992-01-01", "2011-01-01"),
    "Iran": ("reference/Datasets/Iran/Iran_catalog.csv", 4.0, "1992-01-01", "2011-01-01"),
}


def train_size(reg):
    c, mc, ts, te = TRAIN[reg]
    d = pd.read_csv(c, parse_dates=["time"])
    return int(((d.magnitude >= mc) & (d.time >= ts) & (d.time < te)).sum())


def dT(reg, key):
    v = M[reg].get(key)
    return v["paired"]["dT"] if v and v.get("paired") else None


def main():
    Path("figures").mkdir(exist_ok=True)
    regs = [r for r in M if M[r].get("ETAS")]  # regions with an ETAS baseline
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))

    variants = [("native", "FQ native", "#7F8C8D"),
                ("zeroshot", "FQ transfer (zero-shot)", "#2980B9"),
                ("fewshot", "FQ transfer (few-shot)", "#27AE60")]
    w = 0.26
    for vi, (key, lab, col) in enumerate(variants):
        xs, ys = [], []
        for i, r in enumerate(regs):
            d = dT(r, key)
            if d is not None:
                xs.append(i + (vi - 1) * w); ys.append(d)
        axA.bar(xs, ys, w, color=col, label=lab)
    axA.axhline(0, color="k", lw=1)
    axA.set_xticks(range(len(regs)))
    axA.set_xticklabels([f"{r}\n({M[r]['regime']})" for r in regs], fontsize=8)
    axA.set_ylabel("temporal gain over ETAS (nats/event)")
    axA.set_title("A. FlowQuake vs region-fitted ETAS (temporal)\nabove 0 = beats ETAS")
    axA.legend(fontsize=8, loc="upper right")

    # Panel B: data efficiency
    sizes = {r: train_size(r) for r in regs}
    for key, lab, col, mk in [("native", "native (train from scratch)", "#7F8C8D", "o"),
                              ("fewshot", "few-shot transfer", "#27AE60", "s")]:
        xs, ys = [], []
        for r in regs:
            d = dT(r, key)
            if d is not None:
                xs.append(sizes[r]); ys.append(d)
        order = np.argsort(xs)
        axB.plot(np.array(xs)[order], np.array(ys)[order], mk + "-", color=col, label=lab, ms=8)
    for r in regs:
        d = dT(r, "fewshot") or dT(r, "native")
        if d is not None:
            axB.annotate(r, (sizes[r], d), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axB.axhline(0, color="k", lw=1); axB.set_xscale("log")
    axB.set_xlabel("native training events (log)")
    axB.set_ylabel("temporal gain over ETAS (nats/event)")
    axB.set_title("B. Data efficiency: transfer rescues data-poor regions")
    axB.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figures/multiregion_transfer.png", dpi=140)
    print("wrote figures/multiregion_transfer.png")
    for r in regs:
        print(f"  {r:11} train={sizes[r]:>6}  native dT={dT(r,'native')}  "
              f"zs={dT(r,'zeroshot')}  few={dT(r,'fewshot')}")


if __name__ == "__main__":
    main()
