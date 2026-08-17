"""Where is the information floor, and is there one at all?

WHAT THIS REPLACES. The ladder's own saturation test fits a straight line to the
sequence of marginal gains and reports its slope. A positive slope means deeper
bands pay less, which is suggestive, but it answers a weaker question than the
one that matters and it answers it in the wrong units. Nobody can act on
"+0.065 nats per magnitude squared". The paper needs a magnitude, with an
interval, and an explicit contest between the two hypotheses:

    saturating   information accumulates toward a ceiling as the catalog
                 deepens. There is a smallest earthquake worth detecting.
    scale-free   information accumulates linearly without limit. Detection
                 sensitivity keeps paying forever.

THE MODEL. Write d = anchor - mc for catalog depth in magnitude units, and I(d)
for the cumulative information the ladder has accumulated by depth d. Then

    saturating   I(d) = I_inf * (1 - exp(-d / tau))
    scale-free   I(d) = c * d

tau is the e-folding scale in MAGNITUDE UNITS and is the physical quantity here:
each tau of extra depth buys a factor e less than the last. It is estimated
inside the observed range, so it needs no extrapolation. I_inf is the total
information available at infinite depth, and the fraction of it already captured
at the deepest rung is the number a detection-instrument designer actually wants.

WHY tau IS PROFILED RATHER THAN OPTIMIZED. For fixed tau, I_inf enters linearly,
so it has a closed form. Profiling over a tau grid therefore turns a
two-parameter nonlinear fit into a one-dimensional search with no starting
guess, no convergence failure, and no optimizer dependency. That matters because
this fit runs thousands of times inside the bootstrap, where a single
non-converged draw would silently corrupt an interval.

THE HONEST-EXTRAPOLATION RULE. Quoting the magnitude at which 95% of the
information is captured is fine when it lands inside the ladder and dishonest
when it does not. Every reported magnitude is therefore labelled with whether it
was observed or extrapolated, and the fraction captured at the deepest OBSERVED
rung is always reported alongside, because that one is bounded by data.

Usage:
    python scripts/floor_magnitude.py \
        --ladder runs/panel_white/information_ladder_uniform_deep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import DEFAULT_BLOCK_WINDOWS, _resample  # noqa: E402

#: Grid of e-folding scales searched, in magnitude units. The lower end is well
#: below one ladder step (a tau that short is indistinguishable from a step
#: function on this grid) and the upper end is far longer than any ladder, where
#: the saturating model becomes numerically identical to the linear one.
TAU_GRID = np.geomspace(0.02, 50.0, 900)


def fit_saturating(d, I):
    """Profile-likelihood fit of I_inf * (1 - exp(-d/tau)). Returns (I_inf, tau, rss).

    For each candidate tau the basis f = 1 - exp(-d/tau) is fixed, so the
    least-squares I_inf is <I, f> / <f, f>. Scanning tau and taking the best is
    exact on the grid and cannot fail to converge.
    """
    f = 1.0 - np.exp(-np.outer(1.0 / TAU_GRID, d))       # (n_tau, n_pts)
    ff = np.einsum("ij,ij->i", f, f)
    fi = f @ I
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(ff > 0, fi / ff, 0.0)
    rss = np.einsum("ij,ij->i", I - amp[:, None] * f, I - amp[:, None] * f)
    k = int(np.argmin(rss))
    return float(amp[k]), float(TAU_GRID[k]), float(rss[k])


def fit_linear(d, I):
    """Least-squares c*d through the origin. Returns (c, rss).

    Through the origin on purpose: I(0) = 0 holds by construction, because the
    anchor rung is the baseline every other rung is measured against. A free
    intercept would let the scale-free model buy fit with a parameter that
    describes nothing.
    """
    c = float(d @ I / (d @ d))
    r = I - c * d
    return c, float(r @ r)


def aic(rss, n, k):
    """Gaussian AIC. Only differences are used, so the constant is dropped."""
    return n * np.log(max(rss, 1e-300) / n) + 2 * k


def summarise(d, I):
    """Both fits plus the derived quantities, for one realisation of the data."""
    I_inf, tau, rss_s = fit_saturating(d, I)
    c, rss_l = fit_linear(d, I)
    n = len(d)
    d_max = float(d.max())
    return {
        "I_inf": I_inf,
        "tau": tau,
        "c": c,
        "d_aic": aic(rss_l, n, 1) - aic(rss_s, n, 2),   # >0 favours saturating
        # Bounded by data: no extrapolation is involved in this one.
        "captured_at_deepest": float(1.0 - np.exp(-d_max / tau)) if tau > 0 else np.nan,
    }


def depth_for_fraction(tau, q):
    """Depth in magnitude units at which a fraction q of I_inf is captured."""
    return -tau * np.log1p(-q)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="runs/panel_white/information_ladder_uniform.json")
    ap.add_argument("--windows", default=None,
                    help="per-window .npz; defaults to the ladder's companion file")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--fractions", default="0.90,0.95,0.99")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    meta = json.load(open(args.ladder))
    npz_path = args.windows or (args.ladder[:-5] + "_windows.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(
            f"missing {npz_path}.\nThe per-window matrix is written automatically "
            "by scripts/information_ladder.py, but only by runs made after that "
            "was added. Re-run the ladder to produce it; the summary JSON alone "
            "has already collapsed the windows and cannot support a bootstrap.")
    z = np.load(npz_path)
    ladder = z["ladder"].astype(float)
    S = z["ll_shape"].astype(float)          # (n_windows, n_rungs)
    tgt = z["n_target_obs"].astype(float)
    n_win = S.shape[0]
    if S.shape[1] != ladder.size:
        raise SystemExit(f"{npz_path} is malformed: {S.shape[1]} rung columns "
                         f"against {ladder.size} ladder entries")

    anchor = float(ladder[0])
    d = anchor - ladder[1:]                  # depth of every rung below the anchor

    def realise(idx):
        s = tgt[idx].sum()
        if s <= 0:
            return None
        pt = S[idx].sum(axis=0) / s
        return pt[1:] - pt[0]                # cumulative information vs the anchor

    I_pt = realise(np.arange(n_win))
    if I_pt is None:
        raise SystemExit("no target events in the frame")
    base = summarise(d, I_pt)

    rng = np.random.default_rng(0)
    draws = []
    for _ in range(args.n_boot):
        I_b = realise(_resample(rng, n_win, DEFAULT_BLOCK_WINDOWS))
        if I_b is not None:
            draws.append(summarise(d, I_b))

    def ci(key):
        v = np.array([x[key] for x in draws], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), v

    print(f"ladder     {args.ladder}")
    print(f"anchor     {anchor:g}   deepest rung {ladder[-1]:g}   "
          f"{len(d)} rungs   {n_win:,} windows   {int(tgt.sum()):,} targets")
    print(f"bootstrap  {len(draws):,} usable draws of {args.n_boot:,}\n")

    print("MODEL CONTEST   saturating  I_inf*(1-exp(-d/tau))   vs   scale-free  c*d")
    lo, hi, v = ci("d_aic")
    p_sat = float(np.mean(v > 0))
    print(f"  delta AIC (positive favours saturating)  {base['d_aic']:+.2f}  "
          f"[{lo:+.2f}, {hi:+.2f}]")
    print(f"  P(saturating preferred)                  {p_sat:.4f}")

    print("\nSATURATING FIT")
    for key, lab, unit in [("I_inf", "I_inf  total information available", "nats"),
                           ("tau", "tau    e-folding scale", "magnitude units")]:
        lo, hi, _ = ci(key)
        print(f"  {lab:<38} {base[key]:.4f}  [{lo:.4f}, {hi:.4f}]  {unit}")

    lo, hi, _ = ci("captured_at_deepest")
    print(f"\n  fraction of I_inf already captured at m {ladder[-1]:g}   "
          f"{base['captured_at_deepest']:.3f}  [{lo:.3f}, {hi:.3f}]")
    print("  (observed, not extrapolated: this is the number that bounds what "
          "any deeper\n   catalogue could still add)")

    print(f"\n{'capture':>9}{'magnitude':>12}{'95% CI':>24}   status")
    out_fracs = {}
    for q in [float(x) for x in args.fractions.split(",")]:
        m_pt = anchor - depth_for_fraction(base["tau"], q)
        vals = np.array([anchor - depth_for_fraction(x["tau"], q) for x in draws])
        vals = vals[np.isfinite(vals)]
        l, h = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
        obs = m_pt >= ladder[-1]
        print(f"{q:>9.0%}{m_pt:>12.2f}{f'[{l:.2f}, {h:.2f}]':>24}   "
              f"{'observed' if obs else 'EXTRAPOLATED below the ladder'}")
        out_fracs[str(q)] = {"m": m_pt, "ci": [l, h], "observed": bool(obs)}

    # --- model-free floor: where do the bands stop paying, and stay stopped? --
    # Deliberately independent of the fit above. Incompleteness and saturation
    # are both monotone, so a single band whose interval happens to cross zero
    # proves nothing; the floor is the shallowest band from which NO deeper band
    # is distinguishable from zero.
    # One resampling pass shared by every band, so the bands are PAIRED: each
    # draw perturbs the whole ladder together, exactly as the increments were
    # constructed. Resampling per band would let neighbouring bands move on
    # different window draws and break the comparison the floor rule depends on.
    pt_all = S.sum(axis=0) / tgt.sum()
    gains_pt = np.diff(pt_all)
    r2 = np.random.default_rng(1)
    gain_draws = []
    for _ in range(args.n_boot):
        idx = _resample(r2, n_win, DEFAULT_BLOCK_WINDOWS)
        s = tgt[idx].sum()
        if s > 0:
            gain_draws.append(np.diff(S[idx].sum(axis=0) / s))
    G = np.asarray(gain_draws)                       # (n_draws, n_bands)
    lo95 = np.percentile(G, 2.5, axis=0)
    inc_lo = [(float(ladder[i]), float(ladder[i + 1]),
               float(gains_pt[i]), float(lo95[i]))
              for i in range(len(ladder) - 1)]

    dead = [j for j in range(len(inc_lo))
            if all(inc_lo[k][3] <= 0 for k in range(j, len(inc_lo)))]
    emp = inc_lo[dead[0]][0] if dead else None
    print("\nMODEL-FREE FLOOR (shallowest band below which nothing is "
          "distinguishable from zero)")
    for hi_m, lo_m, g, l in inc_lo:
        flag = "  <-- floor" if emp is not None and hi_m == emp else ""
        print(f"  {hi_m:>5g} -> {lo_m:<5g}  gain {g:+.4f}   2.5% {l:+.4f}{flag}")
    if emp is None:
        print("  every band still pays at the deepest rung: NO floor reached. "
              "The ladder\n  has not gone deep enough to see one.")
    else:
        print(f"  floor at m {emp:g}")

    out = args.out or (args.ladder[:-5] + "_floor.json")
    json.dump({"ladder_file": args.ladder, "anchor": anchor,
               "deepest_rung": float(ladder[-1]),
               "jitter_km": meta.get("jitter_km"),
               "jitter_slope": meta.get("jitter_slope"),
               "n_windows": n_win, "n_targets": int(tgt.sum()),
               "n_boot_used": len(draws),
               "point": base, "p_saturating_preferred": p_sat,
               "capture_magnitudes": out_fracs,
               "empirical_floor": emp,
               "increments": [{"from": a, "to": b, "gain": g, "lo95": l}
                              for a, b, g, l in inc_lo]},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
