"""Pool the ETAS control's mc-slope across regions (gate G1, pooled).

Each panel answers "how much forecast skill does a decade of magnitude buy the
best-fit physics baseline?" on its own catalog, network and tectonics. WHITE
alone cannot resolve the per-step increments — only its first half-decade is
significant — so the design always intended a multi-region pool.

The panels use DIFFERENT mc grids (WHITE 2.5–1.0, QTM SanJac 2.3–1.3, QTM
SaltonSea 2.5–1.7, ComCat 3.5–2.8), so per-step increments are NOT comparable
between them. The poolable quantity is the slope per decade, which is what
`pooling.block_bootstrap_slope` returns and what `random_effects_pool` combines.

Random effects, not fixed: the panels differ by construction (tectonics,
network, span, even M_TARGET), so assuming one common slope would give an
interval far too narrow for the claim. High I² is reported, not explained away —
"the information limit differs by region" is a different and more interesting
result than one universal number.

Scores are the SHAPE term (invariant 1c). The level term carries the magnitude
tail and would let a better-calibrated total rate masquerade as spatial skill.

Usage:
    python scripts/pool_etas_panels.py --background uniform
    python scripts/pool_etas_panels.py --background smoothed
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
    ("San Jacinto (WHITE)", "runs/panel_white/etas"),
    ("San Jacinto (QTM)",   "runs/panel_qtm_sanjac/etas"),
    ("Salton Sea (QTM)",    "runs/panel_qtm_saltonsea/etas"),
    ("California (ComCat)", "runs/panel_comcat/etas"),
]


def load_panel(root: Path, background: str):
    """mc -> (per-window shape score, per-window target count)."""
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
    ap.add_argument("--background", default="uniform",
                    choices=["uniform", "smoothed"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--panel", action="append", default=None,
                    help="NAME=path/to/etas, repeatable; overrides the defaults")
    args = ap.parse_args(argv)

    spec = ([tuple(p.split("=", 1)) for p in args.panel] if args.panel
            else PANELS)

    slopes, skipped = [], []
    print(f"\nPOOLED ETAS CONTROL — background={args.background}, "
          f"SHAPE term (invariant 1c)\n")
    for name, path in spec:
        root = Path(path)
        pts = load_panel(root, args.background)
        if len(pts) < 2:
            skipped.append((name, f"{len(pts)} scored point(s)"))
            continue
        mcs = sorted(pts, reverse=True)
        scores = np.vstack([pts[m][0] for m in mcs])
        tgt = pts[mcs[0]][1]
        ps = block_bootstrap_slope(name, mcs, scores, tgt,
                                   n_boot=args.n_boot, seed=0)
        slopes.append(ps)
        grid = "/".join(f"{m:g}" for m in mcs)
        print(f"  {name:22s} mc {grid:16s} n_tgt={ps.n_targets:4d}  "
              f"slope {ps.slope:+7.4f}  [{ps.ci_lo:+.4f}, {ps.ci_hi:+.4f}]"
              f"{'  *' if ps.ci_lo > 0 or ps.ci_hi < 0 else ''}")

    for name, why in skipped:
        print(f"  {name:22s} SKIPPED — {why}")
    if not slopes:
        print("\nnothing to pool")
        return 1

    print("\n  * = interval excludes zero\n")
    if len(slopes) < 2:
        print("only one usable panel; pooling needs at least two")
        return 0

    pooled = random_effects_pool(slopes)
    sig = "  *" if (pooled.ci_lo > 0 or pooled.ci_hi < 0) else ""
    print(f"  {'POOLED (DerSimonian-Laird)':22s} "
          f"{'':16s}                slope {pooled.estimate:+7.4f}  "
          f"[{pooled.ci_lo:+.4f}, {pooled.ci_hi:+.4f}]{sig}")
    print(f"\n  heterogeneity: tau^2={pooled.tau2:.5f}  I^2={pooled.i2:.1%}  "
          f"Q={pooled.q:.2f} (df={pooled.q_df}, p={pooled.q_p:.3g}), "
          f"k={len(slopes)}")
    if pooled.heterogeneous:
        print("  -> Cochran's Q is significant: the regions disagree by more than\n"
              "     sampling noise explains.")
    if pooled.i2 > 0.5:
        print("  -> I^2 above 50%: the panels are NOT measuring one common slope.\n"
              "     Report the per-region spread as the result; a single pooled\n"
              "     number would hide the more interesting claim (1k).")
    print()

    # Report which panels are individually significant, since a pooled interval
    # excluding zero on the strength of one panel is a weaker claim than one
    # supported by several.
    sig = [p.name for p in slopes if p.ci_lo > 0 or p.ci_hi < 0]
    print(f"  panels individually significant: {len(sig)}/{len(slopes)}"
          + (f"  ({', '.join(sig)})" if sig else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
