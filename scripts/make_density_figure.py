"""Central figure: FlowQuake's temporal gain over region-fitted ETAS vs catalog
completeness magnitude (mc), across tectonic regimes. The neural advantage is
density-dependent: positive on dense (low-mc) catalogs across the tested
regimes, with San Jacinto as the conservative block-bootstrap boundary case,
shrinking to a tie/loss on sparse (mc4.0) catalogs where ETAS's parametric
Omori/GR structure suffices (Japan=Tohoku-Omori; data-poor Greece/Iran rescued
by transfer, shown elsewhere).

Run: python scripts/make_density_figure.py  -> figures/density_dependence.png
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
# (label, regime, mc, FQ per-event, ETAS dir)
PTS = [
    ("WHITE", "transform", 0.6, "runs/WHITE_06_n1/per_event_test.csv", "output_data_WHITE_06"),
    ("Salton Sea", "transform", 1.0, "runs/SaltonSea_10_n1/per_event_test.csv", "output_data_SaltonSea_10"),
    ("San Jacinto", "transform", 1.0, "runs/SanJac_10_n1/per_event_test.csv", "output_data_SanJac_10"),
    ("SCEDC", "transform", 2.0, "runs/SCEDC_20_n1/per_event_test.csv", "output_data_SCEDC_20"),
    ("California", "transform", 2.5, "runs/n1_density/per_event_test.csv", "output_data_ComCat_25"),
    ("Italy", "extension", 2.5, "runs/italy_n1/per_event_test.csv", "output_data_Italy_25"),
    ("Chile", "subduction", 4.0, "runs/chile_n1/per_event_test.csv", "output_data_Chile_25"),
    ("Japan", "subduction", 4.0, "runs/japan_n1/per_event_test.csv", "output_data_Japan_25"),
    ("Greece", "extension", 4.0, "runs/greece_n1/per_event_test.csv", "output_data_Greece_25"),
    ("Iran", "collision", 4.0, "runs/iran_n1/per_event_test.csv", "output_data_Iran_25"),
    ("New Zealand", "subduction", 3.5, "runs/newzealand_n1/per_event_test.csv", "output_data_NewZealand_35"),
]
COL = {"transform": "#2980B9", "extension": "#E67E22", "subduction": "#27AE60", "collision": "#8E44AD"}


def gain(fq, ed):
    f, a = Path(fq), ER / ed / "augmented_catalog.csv"
    if not f.exists() or not a.exists():
        return None
    m = pd.read_csv(f, parse_dates=["time"]).merge(
        pd.read_csv(a, parse_dates=["time"]).dropna(subset=["TLL"])[["time", "TLL"]], on="time", how="inner")
    if not len(m):
        return None
    d = m["tll"] - m["TLL"]
    g = paired_gain_summary(d, seed=sum(ord(ch) for ch in fq) % 10000)
    return g.mean, g.ci_low, g.ci_high, g.decision, len(m)


def main():
    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    seen = set(); rows = []
    rng = np.random.default_rng(0)
    for lab, reg, mc, fq, ed in PTS:
        g = gain(fq, ed)
        if g is None:
            continue
        dT, lo, hi, decision, n = g; rows.append((lab, reg, mc, dT, lo, hi, decision, n))
        xj = mc + (rng.random() - 0.5) * 0.12   # jitter overlapping mc
        ax.errorbar(xj, dT, yerr=[[dT - lo], [hi - dT]], fmt="o", color=COL[reg], ms=7, capsize=3,
                    label=reg if reg not in seen else None)
        seen.add(reg)
        ax.annotate(lab, (xj, dT), fontsize=7, xytext=(5, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=1)
    ax.fill_between([0.2, 4.4], -0.005, 0.005, color="grey", alpha=0.15)  # ~tie band
    ax.set_xlabel("catalog completeness magnitude  mc  (← denser data)")
    ax.set_ylabel("temporal log-lik. gain over ETAS (nats/event)")
    ax.set_title("FlowQuake beats ETAS temporally on dense catalogs across regimes;\nadvantage shrinks as catalogs thin (mc↑)")
    ax.legend(title="tectonic regime", frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("figures/density_dependence.png", dpi=200)
    print("wrote figures/density_dependence.png")
    print(f"{'catalog':13}{'regime':11}{'mc':>5}{'dTemp':>9}{'95%CI':>17}{'N':>7}{'call':>7}")
    for lab, reg, mc, dT, lo, hi, decision, n in sorted(rows, key=lambda r: r[2]):
        print(f"{lab:13}{reg:11}{mc:>5.1f}{dT:>+9.3f}  [{lo:+.3f},{hi:+.3f}]{n:>7}{decision:>7}")


if __name__ == "__main__":
    main()
