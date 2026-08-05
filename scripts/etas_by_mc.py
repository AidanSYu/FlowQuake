"""ETAS re-inverted at every mc, scored on the scaling curve's own frame.

`MOONSHOT.md` invariant 3: a curve measured against a FIXED-mc ETAS is a curve
about ETAS's mc sensitivity, not about information content. ETAS must be
re-inverted at each point.

Two background modes, and the difference between them is gate G1:

  uniform    what the EarthquakeNPP benchmark ships. Reproduces the incumbent
             FlowQuake's published gains are measured against.
  smoothed   Zhuang/Helmstetter variable-bandwidth background, re-estimated
             inside the EM from background probabilities. This is the feature
             every operational ETAS has had since 2002, and the one that
             supplies +0.051 of FlowQuake's +0.060 spatial gain. If ETAS with a
             smoothed background closes the gap, §4.4's neural framing collapses
             and MUST be restated.

Runs against the SAME `frame.json` the neural curve used, through the SAME
`flowquake.target_process` scorer, so the two are directly comparable rather
than two pipelines that happen to print similar numbers.

Usage:
    python scripts/etas_by_mc.py --base configs/n1_density.yaml \
        --frame runs/scaling_curve/california/frame.json \
        --out runs/etas_by_mc/california --mc 4.0 3.0 2.5 2.0 1.5 \
        --background uniform smoothed
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.config import Config
from flowquake.proc import spawn
from flowquake.etas_fit import (
    Background, ETASParams, branching_ratio, etas_rate_field, fit_etas_em,
    simulate_etas)
from flowquake.target_process import Grid, TargetSpec, aggregate, score_window

#: Total simulated events per window the rate field is estimated from, held
#: FIXED across mc. See scripts/scaling_curve.py:sims_for_matched_resolution
#: for the derivation and the measured bias-vs-T table.
TARGET_SIM_EVENTS = 20_000

PY = sys.executable
ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": ".",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def load_frame(path: Path) -> dict:
    fr = json.load(open(path))
    if "b_value" not in fr:
        raise SystemExit(
            f"{path} predates MOONSHOT.md invariant 1c and carries no `b_value`.\n"
            f"The ETAS control would then thin its simulated catalogs with a "
            f"different magnitude tail than the neural curve did, so the two "
            f"would not be comparable — which is the entire point of this "
            f"control.\nDelete the frame and re-run scripts/scaling_curve.py to "
            f"rebuild it.")
    fr["grid"] = Grid(**fr["grid"])
    act = np.zeros(fr["grid"].n_cells, dtype=bool)
    act[np.array(fr["active_cells"], dtype=np.int64)] = True
    fr["active"] = act
    return fr


def run_one(cfg: Config, frame: dict, mc: float, background: str, out: Path,
            n_sims: int, n_iter: int, seed: int) -> dict:
    tag = f"{background}_mc{mc:g}"
    dst = out / tag
    dst.mkdir(parents=True, exist_ok=True)
    res_path = dst / "target_process.json"
    if res_path.exists():
        print(f"[skip] {tag}", flush=True)
        return json.load(open(res_path))

    spec = TargetSpec(**frame["spec"])
    spec = replace(spec, mc=float(mc), b_value=frame["b_value"])
    grid: Grid = frame["grid"]

    df = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"]).sort_values("time")
    # Use the FRAME's time origin, never this catalog's own first event. The
    # frame's window start_days are expressed against it, and two catalogs being
    # compared need not share a first event -- the surrogate null's differs from
    # the informative arm's by 1.7 hours, which would put every null-arm forecast
    # 7% of a 1-day window out of step.
    t0 = (pd.Timestamp(frame["t0"]) if "t0" in frame else df["time"].iloc[0])
    if "t0" not in frame:
        print("*** frame has no t0; falling back to this catalog's first event. "
              "Rebuild the frame — cross-arm comparisons are not safe.", flush=True)
    df["t_days"] = (df["time"] - t0).dt.total_seconds() / 86400.0
    df = df[df["magnitude"] >= mc].reset_index(drop=True)

    t = df["t_days"].to_numpy()
    x = df["x"].to_numpy(); y = df["y"].to_numpy(); m = df["magnitude"].to_numpy()

    # The fit region is the SCORING GRID, not this catalog's bounding box.
    #
    # mu is a rate DENSITY: the EM M-step sets mu = w_bg.sum() / (area * T), so
    # mu ~ 1/area. The rate field then places `mu * cell_area * horizon` in each
    # cell of the FIXED grid, so the background mass actually landed on the grid
    # is off by grid_area / region_area. Deriving the region from the catalog
    # made that ratio a function of both the arm and mc, because the box is the
    # bounding box AFTER the mc cut:
    #
    #     mc      2.5     2.0     1.5     1.0
    #     inf   1.1748  1.1204  1.0406  1.0104   <- 16% monotone swing
    #     null  1.0477  1.0386  0.9840  0.9614
    #
    # A smooth monotone mc dependence in the arm whose curve IS the result --
    # the same signature as invariants 1c, 1d and 1i. Grid.__doc__ already
    # states this rule for the scoring grid ("MUST be built once, at a reference
    # completeness... deriving the bounding box per-mc leaks a second mc
    # dependence into the metric"); the fit region simply never inherited it.
    #
    # The grid is shared between arms, fixed across mc, and equal to the area
    # actually scored, so no background mass falls outside it. Verified to
    # contain 100.0000% of both arms' events at every mc. See invariant 1p.
    region = (grid.xmin, grid.xmin + grid.nx * grid.bin_km,
              grid.ymin, grid.ymin + grid.ny * grid.bin_km)
    outside = int(((x < region[0]) | (x > region[1]) |
                   (y < region[2]) | (y > region[3])).sum())
    if outside:
        print(f"*** {tag}: {outside} of {len(x)} events lie OUTSIDE the scoring "
              f"grid and are invisible to the background normalisation. The "
              f"frame's grid does not cover this catalog.", flush=True)

    # Invert on the TRAINING era only. Using the test window would leak.
    val_days = (pd.Timestamp(cfg.data.val_start) - t0).total_seconds() / 86400.0
    fit_sel = t < val_days
    print(f"[fit ] {tag}: {int(fit_sel.sum()):,} training events, "
          f"background={background}", flush=True)
    a = time.time()
    P, bg, hist = fit_etas_em(
        t[fit_sel], x[fit_sel], y[fit_sel], m[fit_sel], mc=mc, region=region,
        background=background, n_iter=n_iter, verbose=False)
    print(f"[fit ] {tag}: {time.time()-a:.0f}s  ll={hist['best_ll']:.1f}  "
          f"n={branching_ratio(P):.3f}  mu={P.mu:.3e} K={P.K:.4f} a={P.a:.3f} "
          f"p={P.p:.3f}", flush=True)
    if branching_ratio(P) >= 1.0:
        print(f"*** WARNING {tag}: branching ratio {branching_ratio(P):.3f} >= 1 "
              f"(supercritical). The inversion is not trustworthy; usually a sign "
              f"of unmodelled short-term incompleteness at this mc.", flush=True)

    # ANALYTIC rate field — no simulation, so no Monte-Carlo resolution bias.
    #
    # A simulated field's per-cell relative variance is 1/(T*p_c) with T the
    # total simulated events, and T scales directly with the catalog rate above
    # mc. Left alone that produced +2.6 nats/decade on a surrogate null whose
    # true slope is zero; driving it down by brute force needs T ~ 20,000 per
    # window, about 165M simulations per curve point at a 1-day horizon. ETAS
    # has a closed form, so the right move is to use it rather than to
    # out-sample the bias. See flowquake.etas_fit.etas_rate_field.
    w_tail = spec.tail_prob(spec.m_target) if spec.tail_mode == "fixed" else 1.0
    print(f"[rate] {tag}: analytic field, GR thinning w={w_tail:.5f}", flush=True)

    windows = []
    for i, w in enumerate(frame["windows"]):
        start = float(w["start_days"])
        lam = etas_rate_field(P, bg, t, x, y, m, grid, start,
                              spec.horizon_days, tail_weight=w_tail)
        obs = np.array(w["obs"], dtype=np.float64).reshape(-1, 4)
        windows.append(score_window(None, obs, grid, spec,
                                    active=frame["active"], lam_field=lam))
        if (i + 1) % 200 == 0:
            print(f"    {tag}: scored {i+1}/{len(frame['windows'])}", flush=True)

    res = {"mcut": mc, "background": background, "params": asdict(P),
           "branching_ratio": branching_ratio(P), "fit_ll": hist["best_ll"],
           "n_sims": 0, "rate_field": "analytic",
           "tail_mode": spec.tail_mode,
           "aggregate": aggregate(windows), "windows": windows}
    json.dump(res, open(res_path, "w"), indent=2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", default=None, help="internal: 'background:mc'")
    ap.add_argument("--base", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mc", type=float, nargs="+", default=[2.5, 2.0, 1.5])
    ap.add_argument("--background", nargs="+", default=["uniform", "smoothed"],
                    choices=["uniform", "smoothed"])
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--n-iter", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args(argv)

    cfg = Config.load(args.base)
    frame = load_frame(Path(args.frame))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.run_one:
        bgmode, mc = args.run_one.split(":")
        run_one(cfg, frame, float(mc), bgmode, out, args.n_sims, args.n_iter,
                args.seed)
        return

    jobs = [(b, mc) for b in args.background for mc in args.mc]
    todo = [j for j in jobs
            if not (out / f"{j[0]}_mc{j[1]:g}" / "target_process.json").exists()]
    print(f"[etas] {len(todo)}/{len(jobs)} to run, concurrency={args.concurrency}",
          flush=True)

    running, queue, fin, t0 = [], list(todo), 0, time.time()
    while queue or running:
        while queue and len(running) < args.concurrency:
            b, mc = queue.pop(0)
            tag = f"{b}_mc{mc:g}"
            (out / tag).mkdir(parents=True, exist_ok=True)
            fh = open(out / tag / "run.log", "w")
            cmd = [PY, __file__, "--run-one", f"{b}:{mc}", "--base", args.base,
                   "--frame", args.frame, "--out", args.out,
                   "--n-sims", str(args.n_sims), "--n-iter", str(args.n_iter),
                   "--seed", str(args.seed)]
            running.append((tag, spawn(cmd, env=ENV, stdout=fh,
                                                  stderr=subprocess.STDOUT), fh))
            print(f"  [start] {tag}", flush=True)
        while running and all(p.poll() is None for _, p, _ in running):
            time.sleep(2.0)
        for tag, proc, fh in list(running):
            if proc.poll() is not None:
                running.remove((tag, proc, fh)); fh.close(); fin += 1
                print(f"  [done ] {tag} rc={proc.returncode} ({fin}/{len(todo)}, "
                      f"{(time.time()-t0)/60:.1f}min)", flush=True)

    rows = []
    for b in args.background:
        for mc in args.mc:
            f = out / f"{b}_mc{mc:g}" / "target_process.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            rows.append({"background": b, "mc": mc,
                         "branching_ratio": d["branching_ratio"],
                         "ll_per_target_event": d["aggregate"]["ll_per_target_event"],
                         "brier": d["aggregate"]["brier"]})
    json.dump(rows, open(out / "etas_curve.json", "w"), indent=2)

    print(f"\n{'background':12}{'mc':>6}{'n':>8}{'ll/target':>12}{'brier':>9}")
    for r in rows:
        print(f"{r['background']:12}{r['mc']:>6g}{r['branching_ratio']:>8.3f}"
              f"{r['ll_per_target_event']:>12.4f}{r['brier']:>9.4f}")
    if {"uniform", "smoothed"} <= set(args.background):
        print("\nGATE G1 — what a smoothed background alone buys ETAS:")
        for mc in args.mc:
            u = next((r for r in rows if r["background"] == "uniform" and r["mc"] == mc), None)
            s = next((r for r in rows if r["background"] == "smoothed" and r["mc"] == mc), None)
            if u and s:
                d = s["ll_per_target_event"] - u["ll_per_target_event"]
                print(f"  mc {mc:g}: {d:+.4f} nats/target event")
        print("  If this is comparable to FlowQuake's gain over uniform-background\n"
              "  ETAS, the neural framing of MANUSCRIPT.md 4.4 does not survive.")
    print(f"\nwrote {out/'etas_curve.json'}")


if __name__ == "__main__":
    main()
