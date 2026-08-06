"""The moonshot answer across two regions, in one figure.

MOONSHOT.md: "If that figure does not exist, there is no paper." The
single-region version (make_moonshot_answer_figure.py) made the claim; this one
has to make it SURVIVE replication, which is a different job and needs a
different layout.

Three panels:

  left/middle  one region each, curves on their own axes. They are NOT shared:
               WHITE lives near -4 to -5 nats and Salton Sea near -1 to -4, and
               forcing a common axis would flatten the very shape difference the
               figure exists to show. The margin is annotated at both ends
               because its COLLAPSE is the replicated finding -- +1.56 -> +0.15
               at WHITE, +1.69 -> +0.12 at Salton Sea.
  right        forest plot: both models, both regions, and the DL pool. This is
               the claim in one panel -- the two model classes' intervals do not
               share a sign, in either region or pooled.

The interior optimum is marked only where it is real. At Salton Sea
P(interior optimum) = 1.0000; at WHITE it is 0.5337, so annotating both the same
way would assert something the WHITE data does not support (invariant 1w: a
figure must not draw a shape its own numbers deny).

Usage:
    python scripts/make_moonshot_two_region_figure.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIONS = [
    ("San Jacinto (WHITE)", "runs/surface_white", 1.5, None),
    ("Salton Sea (QTM)", "runs/surface_saltonsea", 0.8, 2.1),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", default="runs/moonshot_pooled.json")
    ap.add_argument("--out", default="figures/moonshot_two_region.png")
    args = ap.parse_args(argv)

    pooled = json.load(open(args.pooled))
    res = [(nm, json.load(open(f"{p}/moonshot_answer.json")), span, opt)
           for nm, p, span, opt in REGIONS]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)

    for ax, (nm, r, span, opt) in zip(axes[:2], res):
        mcs, n_s, e_s = r["mcs"], r["neural"], r["etas"]
        ax.fill_between(mcs, n_s, e_s, color="C0", alpha=0.10)
        ax.plot(mcs, n_s, "o-", color="C0", lw=2.4, ms=7.5, label="FlowQuake",
                zorder=3)
        ax.plot(mcs, e_s, "s-", color="C1", lw=2.4, ms=6.5, label="ETAS",
                zorder=3)
        for i in (0, len(mcs) - 1):
            ax.annotate(f"margin {n_s[i]-e_s[i]:+.2f}",
                        xy=(mcs[i], (n_s[i] + e_s[i]) / 2), fontsize=8.5,
                        ha="center", color="0.25",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="0.7", alpha=.92))
        if opt is not None:
            j = mcs.index(opt)
            ax.annotate("interior optimum\nP = 1.000", xy=(opt, n_s[j]),
                        xytext=(-6, -46), textcoords="offset points",
                        fontsize=8.5, ha="center", color="C0",
                        arrowprops=dict(arrowstyle="->", color="C0", lw=1.3))
        ax.invert_xaxis()
        ax.margins(x=0.10)      # else the endpoint markers sit on the spine
        ax.set_xlabel("completeness $m_c$   (deeper catalog $\\rightarrow$)")
        ax.set_ylabel("shape log-likelihood per target event (nats)")
        ax.set_title(f"{nm}\n{r['n_targets']} targets, {span:g} decades")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")

    # --- the claim: intervals, by region and pooled ------------------------
    ax = axes[2]
    rows = []
    for nm, r, _, _ in res:
        short = nm.split(" (")[0]
        rows.append((f"{short}", "C1", r["slope_etas"], r["slope_etas_ci"]))
        rows.append((f"{short}", "C0", r["slope_neural"], r["slope_neural_ci"]))
    rows.append(("POOLED", "C1", pooled["ETAS"]["estimate"], pooled["ETAS"]["ci"]))
    rows.append(("POOLED", "C0", pooled["FlowQuake"]["estimate"],
                 pooled["FlowQuake"]["ci"]))

    ax.axvline(0, color="0.35", ls="--", lw=1.2, zorder=1)
    ypos, labels = [], []
    for k, (lab, col, v, ci) in enumerate(rows):
        y = len(rows) - k
        big = lab == "POOLED"
        ax.plot(ci, [y, y], color=col, lw=4.5 if big else 3.0,
                solid_capstyle="round", alpha=1.0 if big else 0.85)
        ax.plot([v], [y], "o", color=col, ms=12 if big else 9, zorder=3)
        ax.annotate(f"{v:+.2f}", xy=(v, y), xytext=(0, 11),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=col,
                    fontweight="bold" if big else "normal")
        ypos.append(y)
        labels.append(f"{lab}\n{'ETAS' if col=='C1' else 'FlowQuake'}")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.9, len(rows) + 0.9)   # headroom below row 1 for the captions
    ax.set_xlabel("nats per decade of magnitude\n(positive = deeper catalog helps)")
    # The title must not overclaim. The POOLED intervals are disjoint and
    # opposite in sign, and so are WHITE's -- but Salton Sea's neural interval
    # runs [-2.94, +0.67] and overlaps its own ETAS interval. Titling this panel
    # "no interval shares a sign" would be a figure asserting what its own bars
    # deny, which is the failure invariant 1w exists to catch.
    ax.set_title("pooled: disjoint intervals,\nopposite signs")
    ax.grid(alpha=0.3, axis="x")
    d = pooled["ETAS - FlowQuake"]
    ax.text(0.5, 0.10, f"pooled gap {d['estimate']:+.2f} "
                       f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f2fff2", ec="C2", alpha=0.95))
    ax.text(0.5, 0.015,
            "Salton Sea's neural interval is wide (70 targets)\nand crosses zero",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="0.4")

    tot = sum(r["n_targets"] for _, r, _, _ in res)
    fig.suptitle(
        "A decade of magnitude buys a fitted ETAS "
        f"{pooled['ETAS']['estimate']:+.2f} nats and a flexible learned model "
        f"{pooled['FlowQuake']['estimate']:+.2f} — "
        f"two regions, {tot} target events, identical frames",
        fontsize=12.5)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
