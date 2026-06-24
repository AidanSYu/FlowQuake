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
    # Reproducible from runs/ablation_h/memorization_figure.json (the converged
    # ckpt_last endpoints; regenerate with scripts/memorization_eval.py).
    data = json.load(open("runs/ablation_h/memorization_figure.json"))
    last = {d["h"]: d for d in data if d["ckpt"] == "last"}
    h = sorted(last)
    tr = [last[k]["train"]["nll"] for k in h]
    te = [last[k]["test"]["nll"] for k in h]
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


def fig_memorization_curve():
    """Held-out NLL vs training step for each h (the divergence onset).

    Reads the logged val NLL from runs/ablation_h/h*/metrics.jsonl — no model
    eval needed. h=0 stays flat; any h>0 diverges catastrophically after the
    first checkpoint, which is why early stopping cannot rescue it."""
    colors = {0: "#4C72B0", 4: "#DD8452", 16: "#C44E52", 64: "#8172B3"}
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for h in (0, 4, 16, 64):
        path = Path(f"runs/ablation_h/h{h}/metrics.jsonl")
        if not path.exists():
            continue
        steps, nll = [], []
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "val" in r:
                steps.append(r["step"]); nll.append(r["val"]["nll"])
        ax.plot(steps, nll, "-o", ms=2.5, lw=1.3, color=colors[h], label=f"h={h}")
    ax.axhline(7.2554, ls="--", c="k", lw=1, label="ETAS test NLL")
    ax.set_xlabel("training step"); ax.set_ylabel("held-out NLL (lower better)")
    ax.set_title("Any whole-catalog embedding (h>0) diverges on held-out data;\n"
                 "h=0 (relational heads) stays stable")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "fig_memorization_curve.png", dpi=160)
    print("wrote", OUT / "fig_memorization_curve.png")


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


def fig_fullsuite():
    """Per-catalog temporal LL gain over ETAS (3-seed mean ± sd)."""
    s = json.load(open("runs/fullsuite_summary.json"))
    etas = {'ComCat_25':1.4343,'WHITE_06':2.0211,'SanJac_10':1.1325,
            'SaltonSea_10':2.3320,'SCEDC_20':2.5410}
    order = ['ComCat_25','SCEDC_20','SanJac_10','WHITE_06','SaltonSea_10']
    gains = [s[d]['tll']-etas[d] for d in order]
    errs = [2*s[d]['tll_sd'] for d in order]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(range(len(order)), gains, yerr=errs, capsize=4, color="#55A868")
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([d.replace('_','\n') for d in order], fontsize=8)
    ax.set_ylabel("temporal LL gain over ETAS\n(nat/event; >0 = FlowQuake wins)")
    ax.set_title("FlowQuake beats ETAS temporally on all 5 EarthquakeNPP catalogs\n(3-seed mean, error bars ±2σ)")
    for i, g in enumerate(gains):
        ax.text(i, g+errs[i]+0.002, f"+{g:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_fullsuite.png", dpi=160)
    print("wrote", OUT / "fig_fullsuite.png")


def _min_tail(q):
    """Two-sided consistency probability min(δ₁,δ₂) — the csep_summary criterion.

    pyCSEP stores each test quantile as a pair; a day is consistent at the two-
    sided 95% level iff min ≥ 0.025. For S/M the pair sums to 1 (so this is just
    0.025≤γ≤0.975); for the discrete N-test, integer-count ties make δ₁+δ₂>1, so
    min(δ₁,δ₂) is the correct tie-aware statistic (and what the table reports).
    Returns None for unevaluable days (NaN or the (-1,-1) sentinel)."""
    if isinstance(q, (list, tuple)):
        if any(v != v for v in q) or min(q) < -0.5:
            return None
        return min(q)
    if q != q or q < -0.5:
        return None
    return min(q, 1.0 - q)


def fig_csep():
    """Per-day CSEP two-sided consistency probability (ComCat, N1 production)."""
    res = json.load(open("runs/n1_density/csep/csep_results.json"))["results"]
    panels = [("N", "Number"), ("S", "Spatial"), ("M", "Magnitude")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (key, title) in zip(axes, panels):
        v = np.sort([m for d in res if key in d
                     for m in [_min_tail(d[key]["quantile"])] if m is not None])
        ax.axhspan(0.025, 0.5, color="#DDE8DD", label="consistent (95%)")
        ax.plot(np.arange(len(v)), v, "o", ms=3, color="#4C72B0")
        ax.axhline(0.025, ls="--", c="#C44E52", lw=1)
        npass = int((v >= 0.025).sum())
        ax.set_title(f"{title}\n{npass}/{len(v)} consistent", fontsize=9)
        ax.set_ylim(-0.01, 0.52); ax.set_xlabel("forecast day (sorted)", fontsize=8)
        ax.set_ylabel("min(δ₁, δ₂)  (consistency)", fontsize=8)
    fig.suptitle("FlowQuake CSEP consistency (ComCat, 100 days × 10⁴ catalogs)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_csep.png", dpi=160)
    print("wrote", OUT / "fig_csep.png")


def _passrate(res, key):
    """CSEP pass count at the two-sided 95% level (matches csep_forecast)."""
    n = p = 0
    for d in res:
        if key not in d:
            continue
        q = d[key]["quantile"]
        ok = (min(q) >= 0.025) if isinstance(q, (list, tuple)) else (q >= 0.025)
        n += 1; p += int(ok)
    return p, n


def fig_csep_headtohead():
    """FlowQuake vs ETAS through the identical pyCSEP path, same forecast days."""
    fq_path = "runs/n1_density/csep/csep_results.json"
    et_path = "runs/etas_csep/csep_results.json"
    if not Path(et_path).exists():
        print("skip head-to-head: no", et_path); return
    fq = json.load(open(fq_path))["results"]
    et = json.load(open(et_path))["results"]
    fq_by_day = {d["day"]: d for d in fq}
    et_by_day = {d["day"]: d for d in et}

    panels = [("N", "Number"), ("S", "Spatial"), ("M", "Magnitude")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    print("\nCSEP head-to-head (same days, same harness, min(δ₁,δ₂)≥0.025):")
    print(f"{'test':>9} | {'FlowQuake':>12} | {'ETAS':>12}")
    for ax, (key, title) in zip(axes, panels):
        # paired days where the test is evaluable for BOTH models
        days, fq_l, et_l = [], [], []
        for d in fq_by_day:
            if key not in fq_by_day[d] or key not in et_by_day.get(d, {}):
                continue
            mf = _min_tail(fq_by_day[d][key]["quantile"])
            me = _min_tail(et_by_day[d][key]["quantile"])
            if mf is None or me is None:
                continue
            days.append(d); fq_l.append(mf); et_l.append(me)
        fq_v = np.array(fq_l); et_v = np.array(et_l)
        order = np.argsort(fq_v)
        ax.axhspan(0.025, 0.5, color="#DDE8DD", label="consistent (95%)")
        ax.axhline(0.025, ls="--", c="#888", lw=0.8)
        ax.plot(np.arange(len(days)), fq_v[order], "o", ms=3.2,
                color="#4C72B0", label="FlowQuake")
        ax.plot(np.arange(len(days)), et_v[order], "^", ms=3.2,
                color="#DD8452", label="ETAS", alpha=0.8)
        fqp = int((fq_v >= 0.025).sum())
        etp = int((et_v >= 0.025).sum())
        ax.set_title(f"{title}\nFQ {fqp}/{len(days)}  ETAS {etp}/{len(days)}", fontsize=9)
        ax.set_ylim(-0.01, 0.52); ax.set_xlabel("forecast day (FQ-sorted)", fontsize=8)
        ax.set_ylabel("min(δ₁, δ₂)  (consistency)", fontsize=8)
        if key == "N":
            ax.legend(fontsize=7, loc="upper left")
        print(f"{title:>9} | {fqp:>5}/{len(days):<6} | {etp:>5}/{len(days):<6}")
    fig.suptitle("FlowQuake vs ETAS — CSEP consistency, identical pyCSEP path "
                 "(ComCat, same forecast days)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_csep_headtohead.png", dpi=160)
    print("wrote", OUT / "fig_csep_headtohead.png")


if __name__ == "__main__":
    fig_fullsuite()
    fig_memorization()
    fig_memorization_curve()
    fig_spatial_gap()
    fig_csep()
    fig_csep_headtohead()
