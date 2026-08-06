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

    # --- score the 2x2 --------------------------------------------------------
    cells = {}
    for mc_th in (args.mc_hi, args.mc_lo):
        for mc_h in (args.mc_hi, args.mc_lo):
            P, bg = fits[mc_th]
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
            cells[(mc_th, mc_h)] = rows
            tot = sum(r["n_target_obs"] for r in rows)
            agg = sum(r["ll_shape"] for r in rows) / tot
            print(f"[cell] theta@{mc_th:g} history@{mc_h:g}: {agg:+.4f} "
                  f"({time.time()-a:.0f}s)", flush=True)

    # --- decomposition, with a paired block bootstrap over windows -----------
    tgt = np.array([r["n_target_obs"] for r in cells[(args.mc_hi, args.mc_hi)]],
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

    hi, lo = args.mc_hi, args.mc_lo
    base = pt[(hi, hi)]
    full = pt[(lo, lo)] - base
    info = pt[(hi, lo)] - base          # history deepened, theta held
    est = pt[(lo, hi)] - base           # theta improved, history held
    print(f"\n{'':<34}{'delta':>10}{'95% CI':>24}")
    for nm, val, fn in (
        ("FULL gain (theta+H at mc %g)" % lo, full,
         lambda d: d[(lo, lo)] - d[(hi, hi)]),
        ("  INFORMATION (H only)", info, lambda d: d[(hi, lo)] - d[(hi, hi)]),
        ("  ESTIMATION (theta only)", est, lambda d: d[(lo, hi)] - d[(hi, hi)]),
    ):
        l, h_, _ = ci(fn)
        print(f"{nm:<34}{val:>10.4f}{f'[{l:+.4f}, {h_:+.4f}]':>24}")
    l, h_, p = ci(lambda d: (d[(hi, lo)] - d[(hi, hi)]) - (d[(lo, hi)] - d[(hi, hi)]))
    print(f"{'  INFORMATION - ESTIMATION':<34}{info-est:>10.4f}"
          f"{f'[{l:+.4f}, {h_:+.4f}]':>24}   P(info>est) = {p:.4f}")

    out = f"{args.panel}/etas_crossover_{args.background}.json"
    json.dump({"mc_hi": hi, "mc_lo": lo, "background": args.background,
               "cells": {f"theta{a_}_hist{b_}": pt[(a_, b_)] for a_, b_ in pt},
               "full": full, "information": info, "estimation": est},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
