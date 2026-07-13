"""CSEP head-to-head figure: FlowQuake vs ETAS through the identical pyCSEP path.

Both models simulate 10^3 one-day catalogs per forecast day on the SAME 100 days
and are scored with the SAME N/S/M consistency tests (flowquake.csep_forecast).
This figure shows the result of §4.2: at a matched simulation budget the two
models are statistically indistinguishable on calibration (a tie), and the ETAS
count forecast is unbiased (the earlier harness under-prediction is gone).

Panels:
  A  N/S/M consistency pass rates, FlowQuake vs ETAS, with the nominal 95% line.
  B  Count calibration: observed daily count vs simulated-mean count, both models,
     with the y=x line. Systematic under-prediction would pull a model below y=x.
  C  N-test reliability: empirical CDF of the lower-tail quantile delta1 across
     days for both models against the Uniform(0,1) diagonal (calibrated = on it).

Run: python scripts/make_csep_h2h_figure.py  -> figures/fig_csep_headtohead.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FQ_DIR = "runs/csep_h2h_fq"
ET_DIR = "runs/csep_h2h_etas"
FQ = FQ_DIR + "/csep_results.json"
ET = ET_DIR + "/csep_results.json"
N_SIMS = 1000  # matched simulation budget for both models
FQ_C, ET_C = "#27AE60", "#C0392B"  # FlowQuake green, ETAS red


def load(path):
    return json.load(open(path))


def rates(r):
    s = r["summary"]
    return {t: (s[t]["n_pass"], s[t]["n_eval"], s[t]["pass_rate"]) for t in ("N", "S", "M")}


def counts(r, out_dir):
    """(observed N, forecast-mean N) per day. n_obs is read from the scored JSON;
    the forecast mean is recomputed from the saved csep_ascii catalogs as
    total_events / N_SIMS (empty simulated catalogs write no rows), because
    --rerun scoring does not repopulate sim_mean."""
    import csv as _csv
    obs, sim = [], []
    for d in r["results"]:
        o = d.get("n_obs")
        day = d.get("day")
        if o is None or not np.isfinite(o):
            continue
        f = Path(out_dir) / f"CSEP_day_{day}_.csv"
        if not f.exists():
            continue
        with open(f, newline="") as fh:
            nrows = sum(1 for _ in fh) - 1  # minus header
        obs.append(o)
        sim.append(max(nrows, 0) / N_SIMS)
    return np.array(obs, float), np.array(sim, float)


def n_delta1(r):
    """N-test lower-tail quantile per day (Uniform(0,1) under calibration)."""
    out = []
    for d in r["results"]:
        q = d.get("N", {}).get("quantile", [None, None])
        v = q[0] if q else None
        if v is not None and np.isfinite(v) and v >= 0:
            out.append(float(v))
    return np.array(out)


def ecdf(x):
    xs = np.sort(x)
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def main():
    Path("figures").mkdir(exist_ok=True)
    fq, et = load(FQ), load(ET)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: pass-rate bars
    rfq, ret = rates(fq), rates(et)
    tests = ["N", "S", "M"]
    x = np.arange(len(tests))
    w = 0.36
    axA.bar(x - w / 2, [rfq[t][2] * 100 for t in tests], w, color=FQ_C, label="FlowQuake")
    axA.bar(x + w / 2, [ret[t][2] * 100 for t in tests], w, color=ET_C, label="ETAS")
    for i, t in enumerate(tests):
        axA.text(i - w / 2, rfq[t][2] * 100 + 0.6, f"{rfq[t][0]}/{rfq[t][1]}", ha="center", fontsize=8)
        axA.text(i + w / 2, ret[t][2] * 100 + 0.6, f"{ret[t][0]}/{ret[t][1]}", ha="center", fontsize=8)
    axA.axhline(95, color="k", ls="--", lw=1, label="nominal 95%")
    axA.set_xticks(x)
    axA.set_xticklabels(["Number\n(N)", "Spatial\n(S)", "Magnitude\n(M)"])
    axA.set_ylabel("CSEP consistency pass rate (%)")
    axA.set_ylim(80, 101)
    axA.set_title("A. Matched-budget consistency: a tie")
    axA.legend(fontsize=8, loc="lower right")

    # Panel B: count calibration scatter
    ofq, sfq = counts(fq, FQ_DIR)
    oet, set_ = counts(et, ET_DIR)
    lim_hi = max(ofq.max(), sfq.max(), oet.max(), set_.max()) * 1.3
    lim_lo = 0.5
    axB.scatter(np.clip(ofq, lim_lo, None), np.clip(sfq, lim_lo, None), s=18, alpha=0.6,
                color=FQ_C, label="FlowQuake")
    axB.scatter(np.clip(oet, lim_lo, None), np.clip(set_, lim_lo, None), s=18, alpha=0.6,
                color=ET_C, label="ETAS")
    axB.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="y = x (unbiased)")
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(lim_lo, lim_hi)
    axB.set_ylim(lim_lo, lim_hi)
    axB.set_xlabel("observed events / day")
    axB.set_ylabel("forecast mean events / day")
    axB.set_title("B. Count calibration (no under-prediction)")
    axB.legend(fontsize=8, loc="upper left")

    # Panel C: N-test reliability ECDF
    xs, ys = ecdf(n_delta1(fq))
    axC.step(xs, ys, where="post", color=FQ_C, lw=2, label="FlowQuake")
    xs, ys = ecdf(n_delta1(et))
    axC.step(xs, ys, where="post", color=ET_C, lw=2, label="ETAS")
    axC.plot([0, 1], [0, 1], "k--", lw=1, label="Uniform (calibrated)")
    axC.set_xlabel("N-test lower-tail quantile $\\delta_1$")
    axC.set_ylabel("empirical CDF")
    axC.set_title("C. Count reliability vs Uniform")
    axC.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/fig_csep_headtohead.png", dpi=140)
    print("wrote figures/fig_csep_headtohead.png")
    print(f"  FlowQuake  N {rfq['N'][2]*100:.0f}%  S {rfq['S'][2]*100:.0f}%  M {rfq['M'][2]*100:.0f}%")
    print(f"  ETAS       N {ret['N'][2]*100:.0f}%  S {ret['S'][2]*100:.0f}%  M {ret['M'][2]*100:.0f}%")
    print(f"  count bias: FQ median(sim/obs)={np.median(sfq/np.clip(ofq,1,None)):.2f}  "
          f"ETAS={np.median(set_/np.clip(oet,1,None)):.2f}  (1.0 = unbiased)")


if __name__ == "__main__":
    main()
