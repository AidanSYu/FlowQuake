"""How much of the observed saturation could be the instrument rather than the earth?

THE PROBLEM THIS ANSWERS. The information ladder finds that the marginal forecast
value of a magnitude band shrinks as events get smaller. The tilt control shows
that magnitude-dependent location error -- the kind every conventional seismic
network makes, because small events are recorded by fewer stations at lower SNR
-- STEEPENS that shrinkage. So the shape the ladder measures is contaminated by
the instrument that produced the catalog, and the contamination pushes in the
direction of the claim. That is the worst kind of confound.

It cannot be removed with network data. It CAN be bounded, and a bound is what
makes the result publishable in the meantime.

THE ARGUMENT. Tilt is measured in decades of location-error reduction per
magnitude unit: sd(m) = sd_0 * 10^(-tilt*(m - m_0)). Adding a known tilt on top
of the catalog's own unknown intrinsic tilt moves the saturation slope by a
measurable amount. That gives a response rate, in slope per decade of tilt.
Divide the whole observed slope by that rate and you get the intrinsic tilt that
would be required for ALL of the saturation to be artifact. If that number is
physically absurd, the result survives, and by how much is quantified rather than
asserted.

WHAT THIS ASSUMES, STATED PLAINLY BECAUSE IT IS THE WEAK POINT.

  1. Linearity. The slope response is extrapolated from added tilt to intrinsic
     tilt as if it were linear. Running a second, larger tilt tests this, and
     with two or more tilt levels present this script REPORTS the curvature
     instead of assuming it away.
  2. Scale. The added error is anchored at 1 km on the smallest band. If the
     catalog's intrinsic error is much larger, the response rate per decade of
     tilt is understated and the bound is optimistic.
  3. Independence. Added jitter is independent Gaussian noise; real location
     error is spatially correlated and partly systematic.

None of these is a reason to skip the bound. They are reasons to quote it as a
bound rather than a correction, which is how it is printed.

Usage:
    python scripts/artifact_bound.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Tilts a real regional network plausibly exhibits, in decades of location-error
#: reduction per magnitude unit. 0.2 is a factor 1.6 per magnitude unit and 0.4 a
#: factor 2.5; published relocation studies sit in that band. Anything far above
#: this range would be visible directly in catalogue error estimates.
PLAUSIBLE_TILTS = (0.2, 0.4)


def slopes(pattern):
    fs = sorted(glob.glob(pattern))
    return fs, np.array([json.load(open(f))["saturation_slope"] for f in fs])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="runs/panel_white/information_ladder_uniform.json")
    ap.add_argument("--tilt-glob", default="runs/panel_white/tilt_j1.0_sl{t}_s*.json",
                    help="{t} is substituted with each tilt level")
    ap.add_argument("--tilts", default="0.4,0.8")
    ap.add_argument("--out", default="runs/panel_white/artifact_bound.json")
    args = ap.parse_args(argv)

    clean = json.load(open(args.clean))["saturation_slope"]
    print(f"clean saturation slope  {clean:+.4f}   ({args.clean})\n")

    levels = []
    for t in [float(x) for x in args.tilts.split(",")]:
        fs, s = slopes(args.tilt_glob.format(t=f"{t:g}"))
        if len(s) == 0:
            print(f"tilt {t:g}: no runs found, skipped")
            continue
        levels.append((t, s))
        print(f"tilt {t:g} decades/mag  slope {s.mean():+.4f}  "
              f"sd {s.std(ddof=1) if len(s) > 1 else float('nan'):.4f}  n={len(s)}"
              f"   delta vs clean {s.mean()-clean:+.4f}")
    if not levels:
        raise SystemExit("no tilt arms found; nothing to bound")

    print(f"\n{'tilt':>6}{'slope/decade':>15}   response rate")
    rates = []
    for t, s in levels:
        r = (s.mean() - clean) / t
        rates.append(r)
        print(f"{t:>6g}{r:>15.4f}")

    # With two or more levels the linearity assumption is checkable rather than
    # assumed. A response rate that FALLS with tilt means the mechanism
    # saturates, so extrapolating the low-tilt rate down to intrinsic tilt
    # OVERSTATES the artifact -- the bound is then conservative, which is the
    # safe direction. A rising rate is the dangerous case and is called out.
    note = None
    if len(rates) >= 2:
        drift = (rates[-1] - rates[0]) / abs(rates[0]) if rates[0] else float("nan")
        print(f"\nlinearity check: response rate moves {100*drift:+.0f}% "
              f"from tilt {levels[0][0]:g} to {levels[-1][0]:g}")
        if drift < -0.15:
            note = ("response rate FALLS with tilt: the mechanism saturates, so a "
                    "bound built on the low-tilt rate is CONSERVATIVE")
        elif drift > 0.15:
            note = ("response rate RISES with tilt: extrapolation to small "
                    "intrinsic tilt UNDERSTATES the artifact. Treat the bound as "
                    "optimistic and say so.")
        else:
            note = "response rate is stable across tilt levels: linearity holds"
        print(f"  {note}")

    # The bound uses the SMALLEST-tilt rate, because intrinsic tilt is small and
    # that is the regime being extrapolated into.
    rate = rates[0]
    need = clean / rate
    print(f"\nintrinsic tilt required for the ENTIRE observed slope to be artifact")
    print(f"  {need:.2f} decades per magnitude unit")
    print(f"  = location error shrinking {10**need:,.0f}x per magnitude unit")

    shares = {}
    print("\nat plausible intrinsic tilt")
    for pl in PLAUSIBLE_TILTS:
        share = pl * rate / clean
        shares[str(pl)] = float(share)
        print(f"  tilt {pl:g}: artifact accounts for {100*share:.0f}% of the "
              f"observed slope, so {100*(1-share):.0f}% is real")

    json.dump({"clean_slope": clean,
               "levels": [{"tilt": t, "slope_mean": float(s.mean()),
                           "slope_sd": float(s.std(ddof=1)) if len(s) > 1 else None,
                           "n": len(s)} for t, s in levels],
               "response_rates": [float(r) for r in rates],
               "linearity_note": note,
               "tilt_required_for_full_artifact": float(need),
               "artifact_share": shares},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
