"""Is the information floor physics, or is it just where the catalog runs out?

THE OBJECTION THIS EXISTS TO KILL. The information ladder finds that the
marginal forecast value of a magnitude band falls toward zero as events get
smaller, and reads that as a physical floor in the nucleation process. There is
an obvious deflationary alternative: catalogs go incomplete near their detection
limit, so the deepest rungs add FEWER REAL EVENTS than they should. Missing
events cannot inform anything, so incompleteness manufactures exactly the
saturation signature the ladder reports. A referee gets to raise this once, and
the answer has to be a measurement rather than an assurance.

THE TEST. Gutenberg-Richter says counts fall log-linearly with magnitude. Fit
that line on a range nobody disputes (well above the detection limit), then
extrapolate DOWN and compare to what the catalog actually holds. Incompleteness
can only ever show up as a DEFICIT against the extrapolation: a detector cannot
invent earthquakes. So a bin holding at or above its GR prediction is complete,
and the rollover magnitude is the shallowest bin that falls meaningfully short.

Note the asymmetry that makes this test safe. A ratio above 1 is not evidence of
anything wrong -- b commonly steepens toward small magnitudes, which lifts the
low bins above a line fitted higher up. Only the downside is diagnostic, which
is why the rollover threshold is one-sided.

The ladder floor is only defensible ABOVE the rollover. If they coincide, the
floor is an artifact and the result does not survive; if the catalog runs
comfortably deeper than the floor, the floor is a property of earthquakes.

Usage:
    python scripts/completeness_check.py --base configs/panel_white.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.config import Config  # noqa: E402


def rollover_magnitude(centres, counts, fit_lo, fit_hi, tol, z_crit=3.0, run=2):
    """Shallowest magnitude at which the count deficit is real and sustained.

    Returns (rollover_m, b, coef, ratios, z). rollover_m is None when nothing in
    the tested range fails, which is the outcome that clears a catalog for use.

    A NAIVE RATIO TEST DOES NOT WORK, and getting this wrong is expensive in both
    directions: a false rollover retires a usable catalog, and a missed one lets
    an incompleteness artifact be published as physics. Two corrections:

    Poisson noise. Bin counts are counts. A bin predicted at 85 that holds 68 is
    down 20%, which trips any fixed ratio, but it is 1.8 sigma and means nothing.
    Deficits are therefore also required to clear `z_crit` standard deviations of
    sqrt(pred). Real detection failure is enormous on this scale -- the WHITE
    truncation edge is -67 sigma -- so a strict threshold costs no sensitivity.

    Persistence. Detection failure is monotone: once events start being missed,
    EVERY lower bin misses more. So a genuine rollover comes with a run of
    deficient bins beneath it, while noise dips alone. Requiring `run`
    consecutive failures is what separates the two, and it is the reason this
    test can be run on sparse catalogs at all.
    """
    sel = (centres >= fit_lo) & (centres <= fit_hi) & (counts > 0)
    if sel.sum() < 3:
        raise SystemExit(
            f"GR fit range {fit_lo}-{fit_hi} holds {int(sel.sum())} usable bins; "
            "needs at least 3. Widen --fit-lo/--fit-hi.")
    coef = np.polyfit(centres[sel], np.log10(counts[sel]), 1)
    pred = 10.0 ** np.polyval(coef, centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(pred > 0, counts / pred, np.nan)
        z = np.where(pred > 0, (counts - pred) / np.sqrt(pred), np.nan)

    bad = np.isfinite(ratios) & (ratios < tol) & (z < -z_crit)

    # Walk DOWN from the fit range; bins above it are large-magnitude sampling
    # noise, not detection. A bin qualifies only if it and the `run - 1` bins
    # beneath it all fail. Near the bottom edge fewer bins exist, so the run is
    # taken over whatever remains -- the catalog's own truncation is exactly the
    # case where the run cannot be completed and must still count.
    below = np.where(centres < fit_hi)[0][::-1]
    for i in below:
        if not bad[i]:
            continue
        window = bad[max(0, i - run + 1):i + 1]
        if window.all():
            return float(centres[i]), float(-coef[0]), coef, ratios, z
    return None, float(-coef[0]), coef, ratios, z


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/panel_white.yaml")
    ap.add_argument("--fit-lo", type=float, default=1.6,
                    help="bottom of the range the GR line is fitted on. Must sit "
                         "clearly above any plausible detection limit, because "
                         "fitting through the rollover hides it.")
    ap.add_argument("--fit-hi", type=float, default=2.8)
    ap.add_argument("--bin", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=0.85,
                    help="a bin counts as incomplete below this fraction of its "
                         "GR prediction. One-sided on purpose: only deficits "
                         "diagnose incompleteness.")
    ap.add_argument("--z-crit", type=float, default=3.0,
                    help="how many Poisson sigma the deficit must clear. Sparse "
                         "bins swing wildly on counting noise alone, and a plain "
                         "ratio test reads that as detection failure.")
    ap.add_argument("--run", type=int, default=2,
                    help="consecutive deficient bins required. Incompleteness is "
                         "monotone downward, so it always comes in runs; an "
                         "isolated dip is noise.")
    ap.add_argument("--floor", type=float, default=None,
                    help="the ladder floor to clear. With this set the script "
                         "returns nonzero if the floor is not safely complete.")
    args = ap.parse_args(argv)

    cfg = Config.load(args.base)
    df = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"])
    df = df[(df["time"] >= pd.Timestamp(cfg.data.train_start))
            & (df["time"] < pd.Timestamp(cfg.data.val_start))]
    m = df["magnitude"].to_numpy()
    if m.size == 0:
        raise SystemExit(f"no events in {cfg.data.train_start}..{cfg.data.val_start}")

    lo = np.floor(m.min() / args.bin) * args.bin
    hi = np.ceil(m.max() / args.bin) * args.bin + args.bin
    edges = np.arange(lo, hi, args.bin)
    counts, _ = np.histogram(m, bins=edges)
    centres = edges[:-1] + args.bin / 2

    roll, b, coef, ratios, z = rollover_magnitude(
        centres, counts, args.fit_lo, args.fit_hi, args.tol,
        z_crit=args.z_crit, run=args.run)

    print(f"catalog   {cfg.data.catalog_path}")
    print(f"window    {cfg.data.train_start} .. {cfg.data.val_start}")
    print(f"events    {m.size:,}   m {m.min():.2f} .. {m.max():.2f}")
    print(f"GR fit    m {args.fit_lo:g}-{args.fit_hi:g}   b = {b:.3f}\n")
    print(f"{'m':>6}{'count':>9}{'GR pred':>10}{'ratio':>8}{'z':>8}")
    # The lowest bin is a boundary artifact whenever the catalog's minimum
    # magnitude falls inside it, so it is shown but never allowed to decide.
    for c_, n_, r_, z_ in zip(centres, counts, ratios, z):
        if c_ > args.fit_hi + args.bin:
            continue
        mark = ""
        if roll is not None and abs(c_ - roll) < args.bin / 2:
            mark = "  <-- rollover"
        elif c_ == centres[0]:
            mark = "  (partial bin)"
        print(f"{c_:6.2f}{n_:9,}{10**np.polyval(coef, c_):10.0f}"
              f"{r_:8.3f}{z_:8.1f}{mark}")

    if roll is None:
        print(f"\ncomplete throughout the tested range, down to m {centres[0]:.2f}")
    else:
        print(f"\nrollover at m {roll:.2f}: bins below this are missing events")

    if args.floor is not None:
        edge = roll if roll is not None else centres[0]
        ok = args.floor > edge
        print(f"\nladder floor {args.floor:g} vs completeness edge {edge:.2f}: "
              f"{'CLEAR' if ok else 'NOT CLEAR'}")
        if not ok:
            print("  The floor sits at or below the detection limit, so the "
                  "saturation it reports cannot be separated from missing "
                  "events. Raise the floor above the edge before quoting it.")
            return 1
        print("  The deepest band the ladder scores is made of events the "
              "catalog actually holds, so its near-zero gain is not a "
              "detection artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
