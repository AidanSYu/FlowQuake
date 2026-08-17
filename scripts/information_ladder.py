"""How small can an earthquake be and still tell you about a large one?

THE QUESTION THIS INSTRUMENT EXISTS TO ANSWER. The moonshot measured what a
decade of magnitude is worth as a single slope between two endpoints. That
number cannot distinguish the two possibilities that matter:

    saturating   the marginal value of a magnitude band falls toward zero as
                 events get smaller, so there is a size below which earthquakes
                 carry no information about large ones. That size is a physical
                 scale in the nucleation process.

    scale-free   every band pays about the same, so detection sensitivity keeps
                 buying skill with no floor in sight.

A straight line fitted between two endpoints reports the same slope either way.
Only the SHAPE of the increments separates them, so this script measures the
increments directly.

WHY ETAS IS THE INSTRUMENT AND NOT THE NEURAL MODEL. Invariant 1y established
that ETAS converts catalog depth into skill through its CONDITIONING HISTORY,
not through better-estimated parameters, and that the flexible learned model
fails to convert it at all. A model that discards the information cannot be used
to measure how much of it there is. ETAS can, and it runs on CPU.

THE DESIGN. Fit ETAS ONCE, at the shallowest threshold on the ladder. Then hold
theta AND the background field frozen and re-score the identical frame with
progressively deeper conditioning history. Every step down the ladder changes
exactly one thing: which earthquakes the intensity is allowed to see.

    anchor fit at mc_a  ->  score with history >= mc_a      baseline
                        ->  score with history >= mc_a - d  + one band
                        ->  score with history >= mc_a - 2d + two bands
                        ...

Consecutive differences are the marginal information in each band. Because the
fit never moves, none of it can be parameter estimation, which is the confound
invariant 1x raised and the 2x2 crossover had to work to rule out. Here it is
excluded by construction.

The rate field is analytic, so the only noise is window sampling and the same
paired circular block bootstrap applies. One window draw per replicate is shared
across every rung, so the increments are paired.

Usage:
    python scripts/information_ladder.py --panel runs/panel_white \
        --base configs/panel_white.yaml --anchor 2.5 --floor 1.0 --step 0.25
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.config import Config  # noqa: E402
from flowquake.etas_fit import branching_ratio, etas_rate_field, fit_etas_em  # noqa: E402
from flowquake.pooling import DEFAULT_BLOCK_WINDOWS, _resample  # noqa: E402
from flowquake.target_process import Grid, TargetSpec, score_window  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="runs/panel_white")
    ap.add_argument("--base", default="configs/panel_white.yaml")
    ap.add_argument("--anchor", type=float, default=2.5,
                    help="mc of the single ETAS fit; the top of the ladder")
    ap.add_argument("--floor", type=float, default=1.0,
                    help="deepest history threshold to test")
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--background", default="uniform")
    ap.add_argument("--fit-anchor", type=float, default=None,
                    help="mc at which ETAS is FITTED, if different from --anchor "
                         "(the ladder's top rung). Exists to test whether the "
                         "measured shape is a property of the catalogue or of one "
                         "particular fitted theta: hold the rungs fixed and vary "
                         "only which events the fit saw. Defaults to --anchor, "
                         "which is the behaviour every published run used.")
    ap.add_argument("--jitter-km", type=float, default=0.0,
                    help="Gaussian noise (sd, km) added to the epicentres of "
                         "the CONDITIONING HISTORY only, leaving targets and "
                         "the grid untouched. Sweeping this prices location "
                         "accuracy in forecast-information units, which is the "
                         "quantity dense arrays actually improve.")
    ap.add_argument("--jitter-slope", type=float, default=0.0,
                    help="decades of location-error REDUCTION per magnitude "
                         "unit above --floor, so sd(m) = jitter_km * "
                         "10^(-slope*(m - floor)). This is the confound the "
                         "uniform sweep cannot see: real networks locate small "
                         "events worse than large ones, because fewer stations "
                         "record them at lower SNR. If magnitude-dependent "
                         "error steepens the saturation slope at matched total "
                         "loss, then an apparent information floor can be "
                         "manufactured by the instrument rather than the "
                         "earthquakes. slope=0 reproduces the uniform case "
                         "BIT-IDENTICALLY (numpy draws normal(0,s) as "
                         "s*normal(0,1)), so published sweeps are unaffected.")
    ap.add_argument("--jitter-min-km", type=float, default=0.0,
                    help="lower bound on sd(m). Real networks do not locate "
                         "anything better than ~100 m however large it is; "
                         "leaving this at 0 lets the tilt drive sd to "
                         "physically absurd precision at the top of the range.")
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.base)
    fr = json.load(open(f"{args.panel}/frame.json"))
    grid = Grid(**fr["grid"])
    act = np.zeros(grid.n_cells, dtype=bool)
    act[np.array(fr["active_cells"], dtype=np.int64)] = True
    spec = dataclasses.replace(TargetSpec(**fr["spec"]), b_value=fr["b_value"])
    region = (grid.xmin, grid.xmin + grid.nx * grid.bin_km,
              grid.ymin, grid.ymin + grid.ny * grid.bin_km)

    df = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"]).sort_values("time")
    t0 = df["time"].iloc[0]
    tt = (df["time"] - t0).dt.total_seconds().to_numpy() / 86400.0
    xx, yy, mm = (df["x"].to_numpy(), df["y"].to_numpy(),
                  df["magnitude"].to_numpy())
    val = (pd.Timestamp(cfg.data.val_start) - t0).total_seconds() / 86400.0
    tr0 = (pd.Timestamp(cfg.data.train_start) - t0).total_seconds() / 86400.0

    # Perturb ONCE, before anything else, so every rung sees the same displaced
    # catalog. Jittering per-rung would let noise differ between rungs and the
    # increments would be measuring the noise draw rather than the magnitude
    # band. Targets are read from the frame and are never touched, so this
    # degrades only what the forecast conditions on, which is the channel dense
    # arrays improve.
    if args.jitter_km > 0:
        jr = np.random.default_rng(args.jitter_seed)
        # sd is anchored AT THE FLOOR and shrinks upward, so jitter_km always
        # means "the error on the smallest events the ladder scores" whatever
        # the tilt is. Anchoring at the anchor instead would silently rescale
        # the whole curve when --floor moves.
        sd = args.jitter_km * np.power(10.0, -args.jitter_slope * (mm - args.floor))
        if args.jitter_min_km > 0:
            sd = np.maximum(sd, args.jitter_min_km)
        # normal(0, s) is drawn as s * normal(0, 1), verified equal, so the
        # slope=0 path consumes the same stream and returns the same numbers as
        # the scalar-scale call this replaced.
        xx = xx + jr.normal(0.0, 1.0, size=xx.shape) * sd
        yy = yy + jr.normal(0.0, 1.0, size=yy.shape) * sd
        if args.jitter_slope:
            print(f"[jitter] magnitude-dependent: sd {args.jitter_km:g} km at "
                  f"m {args.floor:g}, falling {args.jitter_slope:g} decades per "
                  f"magnitude unit (sd {sd.min():.3f}-{sd.max():.3f} km, "
                  f"seed {args.jitter_seed})")
        else:
            print(f"[jitter] history epicentres displaced by "
                  f"{args.jitter_km:g} km sd (seed {args.jitter_seed})")

    n_steps = int(round((args.anchor - args.floor) / args.step))
    ladder = [round(args.anchor - i * args.step, 4) for i in range(n_steps + 1)]
    # Where theta is FITTED is a separate choice from where the ladder STARTS.
    # Tying them, as this script originally did, means the only way to vary the
    # fit is to vary the measurement range too, so a stable result and a
    # coincidence look identical. Decoupling them is what makes the
    # specificity control possible: same rungs, same targets, same windows,
    # several genuinely different frozen theta.
    fit_mc = args.fit_anchor if args.fit_anchor is not None else args.anchor
    print(f"ladder: {ladder}")
    print(f"theta fitted at mc {fit_mc:g}"
          f"{' (DECOUPLED from the ladder top)' if fit_mc != args.anchor else ''}"
          f", background {args.background}\n")

    sel = (mm >= fit_mc) & (tt >= tr0) & (tt < val)
    a = time.time()
    P, bg, _ = fit_etas_em(tt[sel], xx[sel], yy[sel], mm[sel], mc=fit_mc,
                           region=region, background=args.background)
    print(f"[fit ] mc {fit_mc:g}: {int(sel.sum()):,} events, "
          f"{time.time()-a:.0f}s, n={branching_ratio(P):.3f}\n")

    rungs = {}
    for mc_h in ladder:
        h = mm >= mc_h
        # The tail weight converts "rate of events >= mc_h" into "rate of events
        # >= m_target", so it follows the HISTORY threshold: it is what the
        # intensity actually counts. The fitted theta is irrelevant to it.
        sp = dataclasses.replace(spec, mc=float(mc_h))
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
        rungs[mc_h] = rows
        tot = sum(r["n_target_obs"] for r in rows)
        print(f"[rung] history >= {mc_h:>5g}  "
              f"{int(h.sum()):>8,} events  "
              f"ll_shape {sum(r['ll_shape'] for r in rows)/tot:+.4f}  "
              f"({time.time()-a:.0f}s)", flush=True)

    # --- increments, paired block bootstrap over windows ---------------------
    tgt = np.array([r["n_target_obs"] for r in rungs[ladder[0]]], dtype=float)
    S = {k: np.array([r["ll_shape"] for r in v], dtype=float)
         for k, v in rungs.items()}
    n_win = len(tgt)

    def agg(idx):
        s = tgt[idx].sum()
        return ({k: v[idx].sum() / s for k, v in S.items()}
                if s > 0 else None)

    pt = agg(np.arange(n_win))
    rng = np.random.default_rng(0)
    draws = [d for d in (agg(_resample(rng, n_win, DEFAULT_BLOCK_WINDOWS))
                         for _ in range(args.n_boot)) if d]

    def ci(fn):
        v = np.array([fn(d) for d in draws])
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
                float(np.mean(v > 0)))

    print(f"\n{'band':>16}{'marginal':>11}{'95% CI':>24}{'per decade':>13}")
    incs = []
    for i in range(1, len(ladder)):
        hi, lo = ladder[i - 1], ladder[i]
        d = pt[lo] - pt[hi]
        l, h_, p = ci(lambda x, a_=hi, b_=lo: x[b_] - x[a_])
        incs.append({"from": hi, "to": lo, "gain": d, "ci": [l, h_], "p_pos": p})
        print(f"{f'{hi:g} -> {lo:g}':>16}{d:>11.4f}"
              f"{f'[{l:+.4f}, {h_:+.4f}]':>24}{d/args.step:>13.4f}")

    total = pt[ladder[-1]] - pt[ladder[0]]
    total_lo, total_hi, total_p = ci(lambda x: x[ladder[-1]] - x[ladder[0]])
    l, h_, p = total_lo, total_hi, total_p
    print(f"{'TOTAL':>16}{total:>11.4f}{f'[{l:+.4f}, {h_:+.4f}]':>24}"
          f"{total/(args.anchor-args.floor):>13.4f}")

    # --- does the marginal value decline as events get smaller? -------------
    # A declining sequence is the signature of an information floor. Tested as
    # the slope of marginal gain against band midpoint: a POSITIVE slope means
    # gains shrink as mc falls (smaller events matter less, saturation), a
    # negative slope means they grow. Bootstrapped, so it carries an interval
    # rather than being read off three noisy numbers by eye.
    # A line through fewer than three points is not a trend. numpy will happily
    # return a slope for one or two increments, with a RankWarning that is easy
    # to miss in a long log, and the number means nothing. Refuse instead.
    MIN_INCS = 3
    t_pt = tl = th = tp = None
    if len(incs) < MIN_INCS:
        print(f"\nsaturation test SKIPPED: {len(incs)} increment(s), needs "
              f"{MIN_INCS}. Widen the ladder or shrink --step.")
    else:
        mids = np.array([(r["from"] + r["to"]) / 2 for r in incs])

        def trend(x):
            g = np.array([x[r["to"]] - x[r["from"]] for r in incs])
            return float(np.polyfit(mids, g, 1)[0])

        t_pt = trend(pt)
        tl, th, tp = ci(trend)
        print(f"\nsaturation test (slope of marginal gain vs band midpoint)")
        print(f"  slope {t_pt:+.4f}  [{tl:+.4f}, {th:+.4f}]   "
              f"P(gains shrink as mc falls) = {tp:.4f}")
        print("  positive slope => deeper bands pay LESS => approaching a floor")
        print("  interval spanning zero => no floor detected in this range")

    tag = f"_j{args.jitter_km:g}" if args.jitter_km else ""
    if args.jitter_km and args.jitter_slope:
        tag += f"_sl{args.jitter_slope:g}"
    out = (args.out
           or f"{args.panel}/information_ladder_{args.background}{tag}.json")
    json.dump({"anchor": args.anchor, "fit_anchor": fit_mc,
               "floor": args.floor, "step": args.step,
               "background": args.background, "jitter_km": args.jitter_km,
               "jitter_slope": args.jitter_slope,
               "jitter_min_km": args.jitter_min_km,
               "jitter_seed": args.jitter_seed,
               "ladder": ladder,
               "ll_shape": {str(k): pt[k] for k in ladder},
               "increments": incs, "total": total,
               # The total's interval is what decides whether a null region
               # CONTRADICTS a positive one or merely cannot see it. Printing it
               # and discarding it made that question unanswerable without a
               # re-run, which is exactly what happened to San Jacinto.
               "total_ci": [total_lo, total_hi], "p_total_pos": total_p,
               "saturation_slope": t_pt,
               "saturation_ci": ([tl, th] if t_pt is not None else None),
               "p_saturating": tp, "n_windows": n_win,
               "n_targets": int(tgt.sum())},
              open(out, "w"), indent=2)
    # The per-window matrix, always, alongside the summary. Every downstream
    # question about SHAPE -- where the floor is, whether saturating beats
    # scale-free, how much information is left below the deepest rung -- needs to
    # resample windows, and the summary JSON has already collapsed them. It is
    # ~100 kB against a ladder that costs the better part of an hour, so making
    # it optional only creates runs that have to be repeated. Column order is
    # `ladder`, row order is the frame's own window order.
    npz = out[:-5] + "_windows.npz" if out.endswith(".json") else out + "_windows.npz"
    np.savez_compressed(
        npz,
        ladder=np.array(ladder, dtype=np.float64),
        ll_shape=np.stack([S[k] for k in ladder], axis=1),
        n_target_obs=tgt,
    )
    print(f"\nwrote {out}\nwrote {npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
