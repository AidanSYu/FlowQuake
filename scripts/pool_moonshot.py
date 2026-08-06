"""Pool the moonshot answer across regions. One number, honestly weighted.

WHY POOLING IS THE RIGHT MOVE HERE, AND WHERE IT ISN'T. Invariant 1v measured
what actually limits these intervals: WINDOWS carry 95% of the variance and
checkpoints 2.7%. Target events bind. So the only thing that narrows the answer
is more target events, and the only way to get more target events is more
regions -- not more seeds, not more GPU hours, not a denser checkpoint grid.
That is the whole argument for pooling, and it is also the argument for NOT
pretending the pooled interval is narrower than it is.

WHAT IS POOLED, AND WHY THE DIFFERENCE IS POOLED SEPARATELY. Within a region the
FlowQuake slope and the ETAS slope are computed on the SAME bootstrap window
draw, so they are paired and strongly correlated -- both fall when a resample
happens to miss the big sequences. The se of their difference is therefore NOT
recoverable from the two marginal ses; computing it that way would inflate it,
in this data by enough to matter. So the paired difference is formed inside each
region, per draw, and pooled as its own quantity. The marginal slopes are pooled
too, but only for reporting.

HETEROGENEITY IS REPORTED, NOT MINIMISED. DerSimonian-Laird random effects, with
tau^2, I^2 and Cochran's Q surfaced. Two regions is the minimum at which Q means
anything at all and it has 1 degree of freedom, so it can detect only gross
disagreement -- that limitation is printed rather than left implicit. If the
regions genuinely disagree, "the per-decade cost differs by region" is a finding,
not noise to be averaged away.

THE SPAN CAVEAT. A "per decade" slope is a line fitted across each panel's own
mc grid, and those grids differ: WHITE spans 1.5 decades, Salton Sea 0.8. Both
are extrapolated to a common per-decade unit, so the shorter lever arm is doing
more extrapolation for the same reported precision. The span is printed next to
every slope for that reason.

Usage:
    python scripts/pool_moonshot.py \
        --draws white=runs/surface_white/draws.npz \
        --draws saltonsea=runs/surface_saltonsea/draws.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import PanelSlope, random_effects_pool  # noqa: E402


def _panel(name: str, draws: np.ndarray, point: float, n_win: int,
           n_tgt: int) -> PanelSlope:
    """A PanelSlope from bootstrap draws, keeping the POINT estimate.

    The bootstrap mean is not substituted for the point estimate: block
    resampling of a ratio is mildly biased and the point estimate is the
    quantity actually being reported elsewhere. Only the se and the interval
    come from the draws.
    """
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return PanelSlope(name=name, slope=float(point),
                      se=float(np.nanstd(draws, ddof=1)),
                      ci_lo=float(lo), ci_hi=float(hi),
                      n_windows=int(n_win), n_targets=int(n_tgt))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", action="append", required=True,
                    help="name=path.npz, repeatable")
    ap.add_argument("--out", default="runs/moonshot_pooled.json")
    args = ap.parse_args(argv)

    regions = []
    for spec in args.draws:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--draws needs name=path, got {spec!r}")
        d = np.load(path)
        mcs = d["mcs"]
        regions.append({
            "name": name, "path": path,
            "sl_n": d["slope_neural"], "sl_e": d["slope_etas"],
            "pt_n": float(np.polyfit(mcs, d["neural"], 1)[0] * -1.0),
            "pt_e": float(np.polyfit(mcs, d["etas"], 1)[0] * -1.0),
            "span": float(mcs.max() - mcs.min()),
            "n_win": int(d["n_windows"]), "n_tgt": int(d["n_targets"]),
        })

    print(f"{'region':14}{'span':>7}{'windows':>9}{'targets':>9}"
          f"{'FlowQuake':>12}{'ETAS':>10}{'ETAS-FQ':>10}{'P(sign)':>9}")
    for r in regions:
        diff = r["sl_e"] - r["sl_n"]
        p_sign = float(np.mean((r["sl_e"] > 0) & (r["sl_n"] < 0)))
        r["p_sign"] = p_sign
        print(f"{r['name']:14}{r['span']:>6.1f}d{r['n_win']:>9}{r['n_tgt']:>9}"
              f"{r['pt_n']:>12.4f}{r['pt_e']:>10.4f}"
              f"{r['pt_e']-r['pt_n']:>10.4f}{p_sign:>9.4f}")

    out = {"regions": [{k: r[k] for k in
                        ("name", "span", "n_win", "n_tgt", "pt_n", "pt_e",
                         "p_sign")} for r in regions]}

    for label, key, pt in (("FlowQuake", "sl_n", "pt_n"),
                           ("ETAS", "sl_e", "pt_e"),
                           ("ETAS - FlowQuake", None, None)):
        if key is None:
            panels = [_panel(r["name"], r["sl_e"] - r["sl_n"],
                             r["pt_e"] - r["pt_n"], r["n_win"], r["n_tgt"])
                      for r in regions]
        else:
            panels = [_panel(r["name"], r[key], r[pt], r["n_win"], r["n_tgt"])
                      for r in regions]
        # random_effects_pool DROPS panels whose se is zero or non-finite, and
        # says nothing when it does. In a two-region pool that silently demotes
        # the headline back to a one-region number while still printing
        # "POOLED". Refuse instead, naming the region.
        bad = [p.name for p in panels
               if not np.isfinite(p.se) or not np.isfinite(p.slope) or p.se <= 0]
        if bad:
            raise SystemExit(
                f"{label}: region(s) {bad} have a degenerate slope or se and "
                "would be dropped from the pool without warning. Refusing to "
                "report a 'pooled' number that silently excludes a region.")
        pooled = random_effects_pool(panels)
        print(f"\n=== {label} (nats per decade of mc lowered) ===")
        print(pooled.summary())
        out[label] = {"estimate": pooled.estimate, "se": pooled.se,
                      "ci": [pooled.ci_lo, pooled.ci_hi], "tau2": pooled.tau2,
                      "i2": pooled.i2, "q": pooled.q, "q_p": pooled.q_p}

    n_reg = len(regions)
    if n_reg < 3:
        print(f"\nNOTE: {n_reg} regions. Cochran's Q has {n_reg - 1} df here, so "
              "it detects only gross\ndisagreement -- a non-significant Q is NOT "
              "evidence the regions agree, and tau^2\nis estimated too "
              "imprecisely to be read as a real between-region variance.")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
