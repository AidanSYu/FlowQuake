"""Cross-regime transfer figure: California-trained FlowQuake applied to Japan.

Panel A: test nll = -(tll+sll) for Poisson, ETAS(Japan-fit), FlowQuake
zero-shot / few-shot / native (multi-seed mean +- range where available).
Panel B: temporal vs spatial decomposition (where the transferred skill lives).

Run: python scripts/make_transfer_figure.py   -> figures/transfer_japan.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETAS = Path("reference/Experiments/ETAS/output_data_Japan_25")


def fq(path):
    r = json.load(open(path)); return r["tll"], r["sll"]


def seeds(paths):
    vals = [fq(p) for p in paths if Path(p).exists()]
    if not vals: return None
    t = np.array([v[0] for v in vals]); s = np.array([v[1] for v in vals])
    return t.mean(), s.mean(), t, s


def main():
    Path("figures").mkdir(exist_ok=True)
    pois = json.load(open("runs/transfer_japan.json"))
    bars = []  # (label, tll, sll, color, lo_nll, hi_nll)
    bars.append(("Poisson/\nuniform", pois["pois_tll"], pois["unif_sll"], "#999999", None, None))

    if (ETAS / "augmented_catalog.csv").exists():
        import pandas as pd
        a = pd.read_csv(ETAS / "augmented_catalog.csv").dropna(subset=["TLL", "SLL"])
        bars.append(("ETAS\n(Japan-fit)", a["TLL"].mean(), a["SLL"].mean(), "#C0392B", None, None))

    zt, zs = pois["tll"], pois["sll"]
    bars.append(("FlowQuake\nCA zero-shot", zt, zs, "#2980B9", None, None))
    if Path("runs/japan_fewshot/eval_test.json").exists():
        ft, fs = fq("runs/japan_fewshot/eval_test.json")
        bars.append(("FlowQuake\nCA few-shot", ft, fs, "#16A085", None, None))
    nat = seeds(["runs/japan_n1/eval_test.json", "runs/japan_n1_s1553/eval_test.json",
                 "runs/japan_n1_s1554/eval_test.json"])
    if nat:
        tm, sm, tarr, sarr = nat
        nll_arr = -(tarr + sarr)
        bars.append(("FlowQuake\nJapan-native", tm, sm, "#27AE60",
                     nll_arr.min(), nll_arr.max()))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [b[0] for b in bars]; nlls = [-(b[1] + b[2]) for b in bars]
    cols = [b[3] for b in bars]
    axA.bar(range(len(bars)), nlls, color=cols)
    for i, b in enumerate(bars):
        if b[4] is not None:
            axA.errorbar(i, -(b[1]+b[2]), yerr=[[-(b[1]+b[2])-b[4]], [b[5]+(b[1]+b[2])]],
                         fmt="none", ecolor="k", capsize=4)
    axA.set_xticks(range(len(bars))); axA.set_xticklabels(labels, fontsize=8)
    axA.set_ylabel("test NLL  = -(tll+sll)  (lower better)")
    axA.set_title("A. Japan test 2011-2020 (incl. Tohoku M9)")
    for i, v in enumerate(nlls):
        axA.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)

    # Panel B: temporal vs spatial skill above naive baseline
    base_t, base_s = pois["pois_tll"], pois["unif_sll"]
    axB.axhline(0, color="k", lw=0.6)
    for i, b in enumerate(bars[1:], 0):
        axB.bar(i-0.2, b[1]-base_t, 0.4, color="#2980B9", label="temporal" if i==0 else "")
        axB.bar(i+0.2, b[2]-base_s, 0.4, color="#E67E22", label="spatial" if i==0 else "")
    axB.set_xticks(range(len(bars)-1)); axB.set_xticklabels([b[0] for b in bars[1:]], fontsize=8)
    axB.set_ylabel("nats/event above Poisson / uniform")
    axB.set_title("B. Where the skill is (temporal vs spatial)")
    axB.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figures/transfer_japan.png", dpi=140)
    print("wrote figures/transfer_japan.png")
    print("bars:", [(b[0].replace(chr(10),' '), round(-(b[1]+b[2]),3)) for b in bars])


if __name__ == "__main__":
    main()
