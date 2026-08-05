"""The killer figure: forecast skill for large earthquakes vs catalog completeness.

`MOONSHOT.md`: "If that figure does not exist, there is no paper."

    x   catalog completeness mc, DESCENDING left-to-right (more small events ->)
    y   skill on a FIXED set of M>=M_TARGET events, in nats per target event
    -   FlowQuake, both arms; ETAS re-inverted at the same mc, both backgrounds
    o   3-seed mean with min-max band

Reading it:
  * matched_window is the headline — "what does lowering my detection threshold
    buy me?" — and confounds information with sample size.
  * matched_n holds the training-event count fixed, so a slope THERE is
    information per event, not more data.
  * the slope, in nats per decade of magnitude, is the number the paper claims.
  * ETAS at the same mc is the control: if ETAS improves just as fast, the
    finding is about catalogs, not about the model — which is still publishable,
    but it is a different sentence.

A flat, tight curve is a real result (gate G3): it bounds how much of a large
earthquake is knowable from the seismicity around it. Plot it and say so.

Usage:
    python scripts/make_moonshot_figure.py \
        --curve runs/scaling_curve/california/curve.json \
        --etas runs/etas_by_mc/california/etas_curve.json \
        --out figures/moonshot_scaling_curve.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARM_STYLE = {
    "matched_window": dict(color="#1f77b4", marker="o", ls="-",
                           label="FlowQuake — matched window (N varies)"),
    "matched_n":      dict(color="#2ca02c", marker="s", ls="--",
                           label="FlowQuake — matched N (information per event)"),
}
ETAS_STYLE = {
    "uniform":  dict(color="#d62728", marker="^", ls=":",
                     label="ETAS — uniform background (benchmark incumbent)"),
    "smoothed": dict(color="#ff7f0e", marker="v", ls="-.",
                     label="ETAS — smoothed background (operational practice)"),
}


def group(rows, key):
    """{group: {mc: [values]}} dropping unscored points."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get("ll_per_target_event")
        if v is not None:
            out[r[key]][r["mc"]].append(v)
    return out


def series(by_mc):
    mcs = sorted(by_mc, reverse=True)
    mean = np.array([np.mean(by_mc[m]) for m in mcs])
    lo = np.array([np.min(by_mc[m]) for m in mcs])
    hi = np.array([np.max(by_mc[m]) for m in mcs])
    return np.array(mcs), mean, lo, hi


def slope_per_decade(mcs, vals):
    """Nats gained per decade of magnitude added below the top mc.

    x is mc, and moving DOWN in mc adds a decade of events, so the reported
    slope is negated: positive means small events help.
    """
    if len(mcs) < 2:
        return float("nan"), float("nan")
    A = np.vstack([mcs, np.ones_like(mcs)]).T
    coef, res, *_ = np.linalg.lstsq(A, vals, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((vals - pred) ** 2))
    ss_tot = float(np.sum((vals - vals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-coef[0]), r2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--etas", default=None)
    ap.add_argument("--out", default="figures/moonshot_scaling_curve.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--m-target", type=float, default=4.0)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(args.curve))
    by_arm = group(rows, "arm")
    if not by_arm:
        raise SystemExit(f"{args.curve} has no scored points yet")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    summary = {}

    for arm, by_mc in by_arm.items():
        mcs, mean, lo, hi = series(by_mc)
        st = ARM_STYLE.get(arm, dict(marker="o", ls="-", label=arm))
        ax.plot(mcs, mean, **st, lw=2, ms=6)
        if np.any(hi > lo):
            ax.fill_between(mcs, lo, hi, color=st.get("color"), alpha=0.15, lw=0)
        s, r2 = slope_per_decade(mcs, mean)
        summary[arm] = {"mc": mcs.tolist(), "mean": mean.tolist(),
                        "slope_nats_per_decade": s, "r2": r2,
                        "n_seeds": {str(m): len(by_mc[m]) for m in mcs}}

    if args.etas and Path(args.etas).exists():
        # The two curves MUST cover the same mc grid.
        #
        # They are produced by different machinery: the ETAS control evaluates
        # an analytic rate field, so it always yields every point, while the
        # neural arm simulates and can lose points at large n_sims. Because
        # matched-resolution n_sims grows monotonically with mc, any such loss
        # removes the HIGH-mc end first -- so a silent mismatch would compare a
        # neural curve over, say, {1.5, 1.0} against an ETAS curve over
        # {2.5, 2.0, 1.5, 1.0}, and the difference between them would be partly
        # a difference of support rather than of skill.
        etas_rows = json.load(open(args.etas))
        neural_mc = {r["mc"] for by in by_arm.values() for r in rows
                     if r.get("ll_per_target_event") is not None}
        etas_mc = {r["mc"] for r in etas_rows
                   if r.get("ll_per_target_event") is not None}
        if neural_mc and etas_mc and neural_mc != etas_mc:
            print(f"\n*** mc GRID MISMATCH — the curves are not comparable ***"
                  f"\n    neural: {sorted(neural_mc, reverse=True)}"
                  f"\n    ETAS  : {sorted(etas_mc, reverse=True)}"
                  f"\n    only in one: {sorted(neural_mc ^ etas_mc, reverse=True)}"
                  f"\n    Restricting BOTH to the intersection for the plot.\n")
            common = neural_mc & etas_mc
            etas_rows = [r for r in etas_rows if r["mc"] in common]
        for bg, by_mc in group(etas_rows, "background").items():
            mcs, mean, lo, hi = series(by_mc)
            st = ETAS_STYLE.get(bg, dict(marker="^", ls=":", label=f"ETAS {bg}"))
            ax.plot(mcs, mean, **st, lw=1.8, ms=5, alpha=0.9)
            s, r2 = slope_per_decade(mcs, mean)
            summary[f"etas_{bg}"] = {"mc": mcs.tolist(), "mean": mean.tolist(),
                                     "slope_nats_per_decade": s, "r2": r2}

    ax.invert_xaxis()          # more small events to the right
    ax.set_xlabel("catalog completeness  $m_c$   (more small events " + r"$\rightarrow$" + ")")
    ax.set_ylabel(f"log-likelihood per M$\\geq${args.m_target:g} target event  (nats)")
    ax.set_title(args.title or
                 f"Forecast skill for M$\\geq${args.m_target:g} events vs catalog completeness")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    txt = []
    for k in ("matched_window", "matched_n"):
        if k in summary:
            txt.append(f"{k}: {summary[k]['slope_nats_per_decade']:+.3f} nats/decade "
                       f"($R^2$={summary[k]['r2']:.2f})")
    if txt:
        ax.text(0.02, 0.02, "\n".join(txt), transform=ax.transAxes, fontsize=8,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    json.dump(summary, open(out.with_suffix(".json"), "w"), indent=2)

    print(f"wrote {out}")
    print(f"\n{'series':34}{'slope (nats/decade)':>21}{'R^2':>7}")
    for k, v in summary.items():
        print(f"{k:34}{v['slope_nats_per_decade']:>21.4f}{v['r2']:>7.2f}")

    # Interpretation. Sign matters: a NEGATIVE slope means skill gets worse as
    # small events are added, which is a finding in its own right and must never
    # be described with the language of a gain.
    mw = summary.get("matched_window", {}).get("slope_nats_per_decade")
    mn = summary.get("matched_n", {}).get("slope_nats_per_decade")
    print()
    if mw is None or mn is None:
        print("READ: both arms are required before this figure means anything "
              "(MOONSHOT.md invariant 2).")
    elif mw <= 0.0:
        print(f"READ: the headline (matched-window) slope is {mw:+.3f} nats/decade — "
              f"NOT positive.\n  Small events do not improve the forecast here. "
              f"Either the curve is a genuine\n  NULL (gate G3 says stop and publish "
              f"it) or something upstream is wrong —\n  check completeness per era, "
              f"and check the magnitude-tail control (invariant 1c).")
    elif mn <= 0.0:
        print(f"READ: matched-window gains ({mw:+.3f}) but matched-N does not "
              f"({mn:+.3f}).\n  The improvement is SAMPLE SIZE alone. That is a real "
              f"claim about catalog\n  volume, but it is not 'small earthquakes carry "
              f"information'.")
    elif mn <= 0.2 * mw:
        print(f"READ: matched-N ({mn:+.3f}) is a small fraction of matched-window "
              f"({mw:+.3f}),\n  so most of the gain is sample size rather than "
              f"information per event.\n  Weaker, still publishable — say it plainly.")
    else:
        print(f"READ: the slope survives holding N fixed ({mn:+.3f} vs {mw:+.3f}), "
              f"so small\n  events carry information beyond their count. That is the "
              f"claim\n  MOONSHOT.md is trying to earn.")

    if mw is not None and mw > 0:
        for k in ("etas_uniform", "etas_smoothed"):
            if k in summary:
                e = summary[k]["slope_nats_per_decade"]
                if e >= 0.8 * mw:
                    print(f"  NOTE: {k} improves comparably ({e:+.3f} vs {mw:+.3f}). "
                          f"The finding is then\n  about CATALOGS, not about the "
                          f"neural model — a different, still-valid sentence.")


if __name__ == "__main__":
    main()
