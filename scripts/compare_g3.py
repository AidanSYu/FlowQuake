"""Gate G3: the neural curve against its ETAS control, on the SAME frame.

The moonshot question is not "does FlowQuake beat ETAS" — that is a single
number at a single completeness. It is whether the MARGIN over the physics
baseline grows as the catalog deepens. A curve that sits a constant distance
above ETAS says the model is better; a curve that pulls away as mc drops says
the model extracts information from small events that ETAS cannot, which is the
actual claim (MOONSHOT.md invariant 3).

So this reports, per mc: the neural score, the ETAS score, and the margin — plus
the slope of each. Both arms of the neural curve are shown, because invariant 2
requires them: `matched_window` holds the observation window fixed and lets N
grow, `matched_n` holds N fixed. A margin that grows only in `matched_window`
is a sample-size effect, not an information effect.

Every point is gated on invariant 1f FIRST. `corr(n_expected, n_observed)` must
be significantly positive or the score is not interpretable — a forecaster that
emits a near-constant field can score well on shape without forecasting
anything. The attainable ceiling is inferred from the OBSERVED overdispersion,
Var(lambda) = Var(N) - E[N], so it does not depend on the model being judged.

Scores are the SHAPE term by default (invariant 1c): the level term carries the
magnitude tail, and a better-calibrated total rate must not be allowed to
masquerade as spatial skill.

Usage:
    python scripts/compare_g3.py --panel runs/panel_white --background uniform
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import block_bootstrap_slope  # noqa: E402


def _probe(w):
    """invariant 1f: does this forecast track observed counts at all?"""
    from scipy import stats
    e = np.array([x["n_expected"] for x in w], dtype=float)
    o = np.array([x["n_target_obs"] for x in w], dtype=float)
    if e.std() <= 0 or len(e) < 4:
        return float("nan"), 1.0, float("nan")
    r, p = stats.pearsonr(e, o)
    var_pred = max(o.var() - o.mean(), 0.0)      # Var(lambda) from OBSERVED counts
    ceil = float(np.sqrt(var_pred / o.var())) if o.var() > 0 else float("nan")
    return float(r), float(p), ceil


def load_neural(panel: Path, arm: str):
    """mc -> (windows, aggregate) for one neural arm."""
    out = {}
    for f in sorted(glob.glob(str(panel / f"{arm}_mc*_s*/target_process.json"))):
        m = re.search(rf"{re.escape(arm)}_mc([0-9.]+)_s", f)
        if not m:
            continue
        d = json.load(open(f))
        out[float(m.group(1))] = (d["windows"], d["aggregate"])
    return out


def load_etas(panel: Path, background: str):
    out = {}
    for f in sorted(glob.glob(str(panel / f"etas/{background}_mc*/target_process.json"))):
        m = re.search(rf"{re.escape(background)}_mc([0-9.]+)/", f)
        if not m:
            continue
        d = json.load(open(f))
        out[float(m.group(1))] = (d["windows"], d["aggregate"])
    return out


def agg_shape(w):
    s = sum(x["ll_shape"] for x in w)
    n = sum(x["n_target_obs"] for x in w)
    return s / n if n else float("nan")


def slope_of(pts, name):
    """Block-bootstrap slope over mc for a {mc: windows} mapping."""
    mcs = sorted(pts, reverse=True)
    if len(mcs) < 2:
        return None
    scores = np.vstack([np.array([x["ll_shape"] for x in pts[m]], dtype=float)
                        for m in mcs])
    tgt = np.array([x["n_target_obs"] for x in pts[mcs[0]]], dtype=float)
    return block_bootstrap_slope(name, mcs, scores, tgt, n_boot=2000, seed=0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="runs/panel_white")
    ap.add_argument("--background", default="uniform",
                    choices=["uniform", "smoothed"])
    ap.add_argument("--arms", nargs="+",
                    default=["matched_window", "matched_n"])
    args = ap.parse_args(argv)
    panel = Path(args.panel)

    etas = load_etas(panel, args.background)
    if not etas:
        print(f"no ETAS control under {panel}/etas — run scripts/etas_by_mc.py first")
        return 1

    print(f"\nGATE G3 — {panel.name}, neural vs ETAS ({args.background} background)")
    print("SHAPE term, nats per target event (invariant 1c)\n")

    margins = {}
    for arm in args.arms:
        neural = load_neural(panel, arm)
        if not neural:
            print(f"[{arm}] no scored points yet\n")
            continue
        print(f"[{arm}]")
        print(f"  {'mc':>5}{'neural':>10}{'ETAS':>10}{'margin':>10}"
              f"{'corr':>8}{'p':>10}{'share':>8}  1f")
        pts_n, pts_e = {}, {}
        for mc in sorted(neural, reverse=True):
            wn, _ = neural[mc]
            if mc not in etas:
                print(f"  {mc:>5g}  neural scored but no ETAS point at this mc")
                continue
            we, _ = etas[mc]
            sn, se = agg_shape(wn), agg_shape(we)
            r, p, ceil = _probe(wn)
            ok = "PASS" if (r > 0 and p < 0.05) else "**FAIL**"
            share = r / ceil if ceil and np.isfinite(ceil) and ceil > 0 else float("nan")
            print(f"  {mc:>5g}{sn:>10.4f}{se:>10.4f}{sn - se:>+10.4f}"
                  f"{r:>8.3f}{p:>10.1e}{share:>8.1%}  {ok}")
            pts_n[mc], pts_e[mc] = wn, we
        margins[arm] = (pts_n, pts_e)

        failed = [mc for mc in pts_n if not (_probe(pts_n[mc])[0] > 0
                                             and _probe(pts_n[mc])[1] < 0.05)]
        if failed:
            print(f"  *** invariant 1f FAILS at mc {failed} — those points are not\n"
                  f"      interpretable and no slope through them means anything.")

        sn = slope_of(pts_n, f"{arm}/neural")
        se = slope_of(pts_e, f"{arm}/etas")
        if sn and se:
            print(f"\n  {'neural slope':22s}{sn.slope:>+9.4f}  "
                  f"[{sn.ci_lo:+.4f}, {sn.ci_hi:+.4f}]")
            print(f"  {'ETAS slope':22s}{se.slope:>+9.4f}  "
                  f"[{se.ci_lo:+.4f}, {se.ci_hi:+.4f}]")
            d = sn.slope - se.slope
            sd = float(np.hypot(sn.se, se.se))
            print(f"  {'MARGIN slope':22s}{d:>+9.4f}  "
                  f"[{d - 1.96*sd:+.4f}, {d + 1.96*sd:+.4f}]"
                  "   <- does the advantage GROW as mc drops?")
            if d - 1.96 * sd > 0:
                print("     margin grows significantly: the model extracts information\n"
                      "     from small events that the physics baseline does not.")
            elif d + 1.96 * sd < 0:
                print("     margin SHRINKS significantly: ETAS closes the gap as the\n"
                      "     catalog deepens.")
            else:
                print("     margin slope is consistent with zero: the model is better by\n"
                      "     a roughly CONSTANT amount, which is a capability claim but\n"
                      "     NOT a claim about information in small events.")
        print()

    if len(margins) == 2:
        print("invariant 2 reminder: a margin that grows in matched_window but not in\n"
              "matched_n is a sample-size effect, not an information effect.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
