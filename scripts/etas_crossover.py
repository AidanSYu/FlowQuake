"""Does ETAS gain from INFORMATION or from ESTIMATION? The 2x2 that separates them.

The moonshot's central sentence is "the information is demonstrably present --
ETAS extracts it -- and the learned model fails to use it." That rests entirely
on ETAS's +0.2162 nats/decade being a measurement of information. Invariant 1x
showed it might not be: ETAS gets 520 training events at mc 2.5 and 10,601 at
mc 1.0, so its forecasts could improve purely because its PARAMETERS are better
estimated.

Refitting at a fixed event budget could not settle it -- at 520 events the
low-mc fits collapse (productivity exponent to 4e-142). This design settles it
without starving any fit, by exploiting the fact that an ETAS forecast draws on
two channels that `etas_rate_field` takes as SEPARATE arguments:

    theta   parameters (K, a, c, p, d, q, mu) estimated from the training era
    H       the conditioning history whose events drive the intensity at
            forecast time

Gain through H is INFORMATION: more small earthquakes visible in the recent past
genuinely say more about what happens next. Gain through theta is ESTIMATION: the
same model, merely fitted better. Crossing them separates the two:

              H at mc 2.5        H at mc 1.0
    th@2.5    baseline           <- pure information gain
    th@1.0    pure estimation    full published gain
              gain

Read the two off-diagonal cells against the baseline and the decomposition is
immediate. Every fit uses its full catalog, so nothing degenerates.

The rate field is analytic (no Monte-Carlo), so the only noise here is window
sampling, and the same paired block bootstrap applies.

Usage:
    python scripts/etas_crossover.py --panel runs/panel_white --mc-lo 1.0 --mc-hi 2.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.config import Config  # noqa: E402
from flowquake.etas_fit import (branching_ratio, etas_rate_field,  # noqa: E402
                                fit_etas_em)
from flowquake.pooling import DEFAULT_BLOCK_WINDOWS, _resample  # noqa: E402
from flowquake.target_process import Grid, TargetSpec, score_window  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="runs/panel_white")
    ap.add_argument("--base", default="configs/panel_white.yaml")
    ap.add_argument("--mc-hi", type=float, default=2.5)
    ap.add_argument("--mc-lo", type=float, default=1.0)
    ap.add_argument("--background", default="uniform")
    ap.add_argument("--split-background", action="store_true",
                    help="untie the background field from the parameters, "
                         "giving a 3-way information/estimation/background "
                         "split. Only meaningful for --background smoothed: "
                         "the uniform background carries no catalog "
                         "information, so its bg cell is a no-op by "
                         "construction.")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args(argv)

    cfg = Config.load(args.base)
    fr = json.load(open(f"{args.panel}/frame.json"))
    grid = Grid(**fr["grid"])
    act = np.zeros(grid.n_cells, dtype=bool)
    act[np.array(fr["active_cells"], dtype=np.int64)] = True
    spec = TargetSpec(**fr["spec"])
    spec = __import__("dataclasses").replace(spec, b_value=fr["b_value"])
    region = (grid.xmin, grid.xmin + grid.nx * grid.bin_km,
              grid.ymin, grid.ymin + grid.ny * grid.bin_km)

    df = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"]).sort_values("time")
    t0 = df["time"].iloc[0]
    tt = (df["time"] - t0).dt.total_seconds().to_numpy() / 86400.0
    xx, yy, mm = (df["x"].to_numpy(), df["y"].to_numpy(),
                  df["magnitude"].to_numpy())
    val = (pd.Timestamp(cfg.data.val_start) - t0).total_seconds() / 86400.0
    tr0 = (pd.Timestamp(cfg.data.train_start) - t0).total_seconds() / 86400.0

    # --- fit theta at each mc, on the FULL catalog at that mc (no starving) ---
    fits = {}
    for mc in (args.mc_hi, args.mc_lo):
        sel = (mm >= mc) & (tt >= tr0) & (tt < val)
        a = time.time()
        P, bg, hist = fit_etas_em(tt[sel], xx[sel], yy[sel], mm[sel], mc=mc,
                                  region=region, background=args.background)
        fits[mc] = (P, bg)
        print(f"[fit ] mc {mc:g}: {int(sel.sum()):,} events, {time.time()-a:.0f}s, "
              f"n={branching_ratio(P):.3f} K={P.K:.4f} a={P.a:.3f}", flush=True)

    # --- score the cells ------------------------------------------------------
    # A cell is (parameters, background field, conditioning history), each at
    # its own mc. The 2x2 ties the first two together because they are fitted
    # together; --split-background unties them, for the reason below.
    def score_cell(mc_p, mc_bg, mc_h):
        P = fits[mc_p][0]
        bg = fits[mc_bg][1]
        h = mm >= mc_h                      # conditioning history
        # The tail weight converts "rate of events >= mc_h" into "rate of
        # events >= m_target", so it follows the HISTORY threshold, which is
        # what the intensity actually counts -- not the fitted theta.
        sp = __import__("dataclasses").replace(spec, mc=float(mc_h))
        w_tail = sp.tail_prob(sp.m_target) if sp.tail_mode == "fixed" else 1.0
        rows = []
        a = time.time()
        for w in fr["windows"]:
            lam = etas_rate_field(P, bg, tt[h], xx[h], yy[h], mm[h], grid,
                                  float(w["start_days"]), sp.horizon_days,
                                  tail_weight=w_tail)
            obs = np.array(w["obs"], dtype=np.float64).reshape(-1, 4)
            rows.append(score_window(None, obs, grid, sp, active=act,
                                     lam_field=lam))
        tot = sum(r["n_target_obs"] for r in rows)
        print(f"[cell] P@{mc_p:g} bg@{mc_bg:g} history@{mc_h:g}: "
              f"{sum(r['ll_shape'] for r in rows) / tot:+.4f} "
              f"({time.time()-a:.0f}s)", flush=True)
        return rows

    hi, lo = args.mc_hi, args.mc_lo
    cells = {}
    if args.split_background:
        # WHY THIS MODE EXISTS. Run across two regions and two backgrounds, the
        # FULL gain is stable (+0.326 to +0.383) but the estimation channel
        # swings from -0.282 to +0.246, and both extremes are SMOOTHED
        # backgrounds. That is a clue about the design, not about seismology:
        # `fit_etas_em` returns (P, bg) and the 2x2 moves them together, so for
        # a smoothed background the "estimation" channel silently carries the
        # BACKGROUND FIELD -- a kernel density of the training catalog, which a
        # deeper catalog resolves better. Spatial information about where
        # earthquakes occur is thus scored as if it were parameter estimation.
        # Untying bg from P separates them.
        specs = [(hi, hi, hi),      # baseline
                 (hi, hi, lo),      # + deeper history        = INFORMATION
                 (lo, hi, hi),      # + refitted parameters   = ESTIMATION
                 (hi, lo, hi),      # + better background     = BACKGROUND
                 (lo, lo, lo)]      # everything deepened     = FULL
    else:
        specs = [(a_, a_, b_) for a_ in (hi, lo) for b_ in (hi, lo)]
    for s in specs:
        cells[s] = score_cell(*s)

    # --- decomposition, with a paired block bootstrap over windows -----------
    tgt = np.array([r["n_target_obs"] for r in cells[specs[0]]],
                   dtype=float)
    S = {k: np.array([r["ll_shape"] for r in v], dtype=float)
         for k, v in cells.items()}
    n_win = len(tgt)

    def agg(idx):
        s = tgt[idx].sum()
        return {k: v[idx].sum() / s for k, v in S.items()} if s > 0 else None

    pt = agg(np.arange(n_win))
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(args.n_boot):
        a_ = agg(_resample(rng, n_win, DEFAULT_BLOCK_WINDOWS))
        if a_:
            draws.append(a_)

    def ci(fn):
        v = np.array([fn(d) for d in draws])
        return np.percentile(v, 2.5), np.percentile(v, 97.5), float(np.mean(v > 0))

    BASE = specs[0]
    base = pt[BASE]
    # Keys are (parameters, background, history) in BOTH modes, so the
    # information cell -- everything held at mc_hi except the history -- is
    # identically defined either way and the two modes stay comparable. Only
    # the estimation cell differs: tied to the background in the 2x2, untied
    # when it is split out.
    named = [("FULL gain (all at mc %g)" % lo, specs[-1]),
             ("  INFORMATION (history only)", (hi, hi, lo))]
    if args.split_background:
        named += [("  ESTIMATION (parameters only)", (lo, hi, hi)),
                  ("  BACKGROUND (bg field only)", (hi, lo, hi))]
    else:
        named += [("  ESTIMATION (theta, bg tied)", (lo, lo, hi))]

    print(f"\n{'':<34}{'delta':>10}{'95% CI':>24}")
    vals = {}
    for nm, key in named:
        vals[nm.strip()] = pt[key] - base
        l, h_, _ = ci(lambda d, k=key: d[k] - d[BASE])
        print(f"{nm:<34}{pt[key]-base:>10.4f}{f'[{l:+.4f}, {h_:+.4f}]':>24}")

    k_info = (hi, hi, lo)
    k_est = (lo, hi, hi) if args.split_background else (lo, lo, hi)
    info, est = pt[k_info] - base, pt[k_est] - base
    l, h_, p = ci(lambda d: d[k_info] - d[k_est])
    print(f"{'  INFORMATION - ESTIMATION':<34}{info-est:>10.4f}"
          f"{f'[{l:+.4f}, {h_:+.4f}]':>24}   P(info>est) = {p:.4f}")

    payload = {"mc_hi": hi, "mc_lo": lo, "background": args.background,
               "split_background": bool(args.split_background),
               "cells": {"_".join(f"{x:g}" for x in k): v for k, v in pt.items()},
               "full": pt[specs[-1]] - base, "information": info,
               "estimation": est}
    if args.split_background:
        k_bg = (hi, lo, hi)
        payload["background_field"] = pt[k_bg] - base
        l, h_, p = ci(lambda d: d[k_bg] - d[k_est])
        print(f"{'  BACKGROUND - ESTIMATION':<34}{pt[k_bg]-pt[k_est]:>10.4f}"
              f"{f'[{l:+.4f}, {h_:+.4f}]':>24}   P(bg>est)   = {p:.4f}")
        payload["background_field_ci"] = [l, h_]

    suffix = "_splitbg" if args.split_background else ""
    out = f"{args.panel}/etas_crossover_{args.background}{suffix}.json"
    json.dump(payload, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
