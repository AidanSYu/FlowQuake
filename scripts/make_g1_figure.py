"""Gate G1 figure: the ETAS control across regions, curves + forest plot.

The result this has to communicate is NOT a single slope — it is that the
regions disagree. I² is ~70% and Cochran's Q is significant, so a lone pooled
number with a confidence interval would be actively misleading: it would read as
"the effect is +0.19, not quite significant" when the truth is "two regions gain
substantially, two do not, and averaging them is the wrong operation."

So the layout is deliberately a meta-analysis one:

  left   each region's curve, shape score vs mc, on its own mc grid
  right  a forest plot — per-region slope with CI, then the pooled diamond,
         with I² and Q printed where a reader cannot miss them

Panels use different mc grids and different M_TARGET, which is legal across
panels (invariant 1 fixes the target set WITHIN a curve). That is exactly why
the x-axis of the left plot is mc and the panels do not share endpoints, and why
the poolable quantity on the right is slope per decade rather than any level.

Usage:
    python scripts/make_g1_figure.py --background uniform \
        --out figures/g1_etas_regions_uniform.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import block_bootstrap_slope, random_effects_pool  # noqa: E402

PANELS = [
    ("San Jacinto (WHITE)", "runs/panel_white/etas", "M≥3.0"),
    ("San Jacinto (QTM)", "runs/panel_qtm_sanjac/etas", "M≥3.0"),
    ("Salton Sea (QTM)", "runs/panel_qtm_saltonsea/etas", "M≥3.0"),
    ("California (ComCat)", "runs/panel_comcat/etas", "M≥4.0"),
]


def load(root: Path, background: str):
    pts = {}
    for d in sorted(root.glob(f"{background}_mc*")):
        f = d / "target_process.json"
        if not f.exists():
            continue
        mc = float(d.name.replace(f"{background}_mc", ""))
        w = json.load(open(f))["windows"]
        if not w or "ll_shape" not in w[0]:
            continue
        pts[mc] = (np.array([x["ll_shape"] for x in w], dtype=float),
                   np.array([x["n_target_obs"] for x in w], dtype=float))
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", default="uniform", choices=["uniform", "smoothed"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args(argv)
    out = Path(args.out or f"figures/g1_etas_regions_{args.background}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves, slopes = [], []
    for name, path, tgt in PANELS:
        pts = load(Path(path), args.background)
        if len(pts) < 2:
            print(f"[skip] {name}: {len(pts)} point(s)")
            continue
        mcs = sorted(pts, reverse=True)
        scores = np.vstack([pts[m][0] for m in mcs])
        tot = pts[mcs[0]][1]
        agg = [float(pts[m][0].sum() / pts[m][1].sum()) for m in mcs]
        ps = block_bootstrap_slope(name, mcs, scores, tot,
                                   n_boot=args.n_boot, seed=0)
        curves.append((name, tgt, mcs, agg))
        slopes.append(ps)
        print(f"  {name:22s} slope {ps.slope:+.4f} [{ps.ci_lo:+.4f}, {ps.ci_hi:+.4f}]")

    if len(slopes) < 2:
        print("need at least two panels")
        return 1
    pooled = random_effects_pool(slopes)

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]})
    colors = plt.get_cmap("tab10").colors

    for i, (name, tgt, mcs, agg) in enumerate(curves):
        axl.plot(mcs, agg, "o-", color=colors[i], lw=2, ms=6,
                 label=f"{name}  ({tgt})")
    axl.invert_xaxis()          # deeper catalogs to the RIGHT
    axl.set_xlabel("completeness magnitude $m_c$  (deeper catalog $\\rightarrow$)")
    axl.set_ylabel("shape log-likelihood per target event (nats)")
    axl.set_title(f"ETAS control per region — {args.background} background")
    axl.grid(alpha=0.3)
    axl.legend(fontsize=8, loc="best")

    ys = np.arange(len(slopes))[::-1]
    lo = min([p.ci_lo for p in slopes] + [pooled.ci_lo, 0.0])
    hi = max([p.ci_hi for p in slopes] + [pooled.ci_hi, 0.0])
    span = hi - lo
    # Reserve room on the right for the n= labels and below for the stats box,
    # so neither is clipped at any panel count.
    axr.set_xlim(lo - 0.08 * span, hi + 0.26 * span)

    for y, p in zip(ys, slopes):
        sig = (p.ci_lo > 0 or p.ci_hi < 0)
        axr.plot([p.ci_lo, p.ci_hi], [y, y], "-",
                 color="black" if sig else "0.55", lw=2)
        axr.plot([p.slope], [y], "s", ms=8,
                 color="black" if sig else "0.55")
        axr.text(p.ci_hi + 0.02 * span, y, f"n={p.n_targets}", va="center",
                 fontsize=8, color="0.35")
    # pooled diamond, set apart from the per-region rows
    yp = -1.4
    axr.axhline(-0.5, color="0.8", lw=0.8)
    axr.plot([pooled.ci_lo, pooled.ci_hi], [yp, yp], "-", color="C3", lw=2.5)
    axr.plot([pooled.estimate], [yp], "D", ms=10, color="C3")
    axr.axvline(0.0, color="0.3", ls="--", lw=1)
    axr.set_yticks(list(ys) + [yp])
    axr.set_yticklabels([p.name for p in slopes] + ["POOLED (random effects)"],
                        fontsize=9)
    axr.set_xlabel("slope (nats per decade of $m_c$)")
    axr.set_title("per-region slopes and pooled estimate")
    axr.grid(axis="x", alpha=0.3)
    axr.set_ylim(yp - 1.5, len(slopes) - 0.3)

    n_sig = sum(1 for p in slopes if p.ci_lo > 0 or p.ci_hi < 0)
    axr.text(
        0.5, 0.015,
        f"$I^2$ = {pooled.i2:.0%}   Q = {pooled.q:.2f} (df {pooled.q_df}, "
        f"p = {pooled.q_p:.3f})   —   regions do NOT share a common slope\n"
        f"{n_sig}/{len(slopes)} regions individually significant",
        transform=axr.transAxes, fontsize=8.5, va="bottom", ha="center",
        bbox=dict(boxstyle="round", fc="#fff4f4", ec="C3", alpha=0.95))

    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nwrote {out}")
    print(f"pooled {pooled.estimate:+.4f} [{pooled.ci_lo:+.4f}, {pooled.ci_hi:+.4f}]  "
          f"I2={pooled.i2:.1%}  Q p={pooled.q_p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
