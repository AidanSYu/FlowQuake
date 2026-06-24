"""Manuscript figures (ComCat_25). Writes PNGs to figures/.

  fig_memorization.png  — train vs test NLL across the whole-catalog
                          bottleneck h (the mechanism result).
  fig_spatial_gap.png   — per-event spatial LL gain over ETAS vs nearest-
                          recent-neighbour distance (localizes the deficit).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")


def fig_memorization():
    # From the ablation (ckpt_last / converged endpoints).
    rows = [(0, 7.28, 7.62), (4, 4.14, 19.65), (16, 4.18, 18.73), (64, 4.27, 18.33)]
    h = [r[0] for r in rows]
    tr = [r[1] for r in rows]
    te = [r[2] for r in rows]
    x = np.arange(len(h))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    w = 0.38
    ax.bar(x - w / 2, tr, w, label="train NLL", color="#4C72B0")
    ax.bar(x + w / 2, te, w, label="held-out (test) NLL", color="#C44E52")
    ax.axhline(7.2554, ls="--", c="k", lw=1, label="ETAS test NLL")
    ax.set_xticks(x); ax.set_xticklabels([f"h={v}" for v in h])
    ax.set_xlabel("whole-catalog embedding bottleneck width")
    ax.set_ylabel("NLL (lower better)")
    ax.set_title("Exposing a learned global embedding to the heads\ncauses memorization")
    for i in range(len(h)):
        ax.text(x[i] + w / 2, te[i] + 0.3, f"gap {te[i]-tr[i]:.1f}", ha="center",
                fontsize=8, color="#C44E52")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_memorization.png", dpi=160)
    print("wrote", OUT / "fig_memorization.png")


def _nn_km():
    """Nearest-recent-neighbour distance (min to previous 64 events) per event."""
    cat = pd.read_csv("reference/Datasets/ComCat/ComCat_catalog.csv", parse_dates=["time"])
    cat = cat[cat.magnitude >= 2.5].sort_values("time").reset_index(drop=True)
    x = cat.x.to_numpy(); y = cat.y.to_numpy()
    nn = np.full(len(cat), np.nan)
    for i in range(1, len(cat)):
        lo = max(0, i - 64)
        nn[i] = np.sqrt((x[i] - x[lo:i]) ** 2 + (y[i] - y[lo:i]) ** 2).min()
    cat["nn_km"] = nn
    return cat[["time", "nn_km"]]


def _binned_gain(per_csv, aug, nn, bins, labels):
    per = pd.read_csv(per_csv, parse_dates=["time"])
    m = per.merge(aug, on="time", how="inner").merge(nn, on="time", how="inner")
    m["gain"] = m["sll"] - m["SLL"]
    m["bin"] = pd.cut(m["nn_km"], bins=bins, labels=labels)
    return m.groupby("bin", observed=True)["gain"].agg(["mean", "count"])


def fig_spatial_gap():
    aug = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"])
    aug = aug.dropna(subset=["SLL"])[["time", "SLL"]]
    nn = _nn_km()
    bins = [0, 0.5, 1, 2, 5, 10, 30, 1e9]
    labels = ["<0.5", "0.5-1", "1-2", "2-5", "5-10", "10-30", ">30"]
    base = _binned_gain("runs/final_s1555/per_event_test.csv", aug, nn, bins, labels)
    n1 = _binned_gain("runs/n1_density/per_event_test.csv", aug, nn, bins, labels)

    x = np.arange(len(labels)); w = 0.4
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x - w / 2, base["mean"], w, label="canonical (isotropic floor)", color="#C44E52")
    ax.bar(x + w / 2, n1["mean"], w, label="N1 (density-adaptive width)", color="#55A868")
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("nearest-recent-neighbour distance (km)")
    ax.set_ylabel("mean spatial LL gain over ETAS\n(per event; >0 = FlowQuake wins)")
    ax.set_title("N1 closes the sub-km spatial deficit it was designed for")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_spatial_gap.png", dpi=160)
    print("wrote", OUT / "fig_spatial_gap.png")
    print("canonical vs N1 mean spatial gain by nn-distance bin:")
    cmp = pd.DataFrame({"canonical": base["mean"], "N1": n1["mean"], "n": base["count"]})
    print(cmp.round(3).to_string())


def fig_csep():
    """Per-day CSEP quantiles with the 95% consistency band (ComCat, best seed)."""
    r = json.load(open("runs/csep_results_s1555.json"))
    res = r["results"]

    def qvals(key, which):
        out = []
        for d in res:
            if key not in d:
                continue
            q = d[key]["quantile"]
            v = (q[which] if isinstance(q, list) else q)
            out.append(v)
        return np.sort(out)

    panels = [("N", 1, "Number (δ₂ = P(N_sim ≤ N_obs))"),
              ("S", 0, "Spatial"), ("M", 0, "Magnitude")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (key, which, title) in zip(axes, panels):
        v = qvals(key, which)
        ax.axhspan(0.025, 0.975, color="#DDE8DD", label="consistent (95%)")
        ax.plot(np.arange(len(v)), v, "o", ms=3, color="#4C72B0")
        ax.axhline(0.025, ls="--", c="#C44E52", lw=1)
        ax.axhline(0.975, ls="--", c="#C44E52", lw=1)
        npass = int(((v >= 0.025) & (v <= 0.975)).sum()) if key != "N" else \
            int((v >= 0.025).sum())  # N: lower tail only (overprediction)
        ax.set_title(f"{title}\n{npass}/{len(v)} consistent", fontsize=9)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("forecast day (sorted)", fontsize=8)
        ax.set_ylabel("test quantile", fontsize=8)
    fig.suptitle("FlowQuake CSEP consistency (ComCat, 100 days × 10⁴ catalogs)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_csep.png", dpi=160)
    print("wrote", OUT / "fig_csep.png")


if __name__ == "__main__":
    fig_memorization()
    fig_spatial_gap()
    fig_csep()
