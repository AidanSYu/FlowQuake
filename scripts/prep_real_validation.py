#!/usr/bin/env python
"""Stage a REAL catalog as a two-arm ground-truth validation (invariant 1g).

Builds the directory layout `scripts/validate_with_etas.py` expects, but from a
real catalog rather than a simulator:

    <out>/_cfg/informative.yaml     the catalog as it is
    <out>/_cfg/null_surrogate.yaml  same targets, small events decoupled
    <out>/informative/frame.json    windows, grid, target set, fixed GR tail
    <out>/null_surrogate/frame.json  identical target set by construction

Why real data. Three simulator-based validations were vacuous because the
generated catalogs were near-Poisson at the scored threshold -- variance/mean
1.21 against 2.0-5.4 for real regional catalogs -- so neither a neural model
nor a fitted ETAS could forecast them, and a flat null proved nothing
(MOONSHOT.md invariants 1f, 1g). A real catalog's clustering is real by
definition; the surrogate supplies the known answer the simulator was
supposed to.

The informative arm's true slope is unknown, and that is fine: the null's is
zero by construction, which is the only ground truth this test needs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.surrogate import surrogate_null  # noqa: E402
from flowquake.target_process import TargetSpec  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="reference/Datasets/WHITE/WHITE_catalog.csv")
    ap.add_argument("--base-config", default="configs/panel_white.yaml")
    ap.add_argument("--out", default="runs/real_validation")
    ap.add_argument("--m-target", type=float, default=3.0)
    ap.add_argument("--m-large", type=float, default=4.0)
    ap.add_argument("--m-split", type=float, default=None,
                    help="events below this are decoupled (default: m_target)")
    ap.add_argument("--mode", default="both",
                    choices=["time_shift", "rotate", "both"])
    ap.add_argument("--horizon", type=float, default=30.0)
    ap.add_argument("--bin-km", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)

    m_split = args.m_split if args.m_split is not None else args.m_target
    out = Path(args.out)
    (out / "_cfg").mkdir(parents=True, exist_ok=True)
    cat_dir = out / "catalogs"; cat_dir.mkdir(exist_ok=True)

    base = yaml.safe_load(open(args.base_config))
    df = pd.read_csv(args.catalog)
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True,
                                errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    print(f"[load] {args.catalog}: {len(df):,} events "
          f"{df.time.min().date()}..{df.time.max().date()}")

    arms = {
        "informative": df,
        # era_bounds keeps the per-era event counts EXACTLY equal between arms.
        # Without it the shift moved small events across the train/test boundary
        # and gave the control 26% more training data at mc 1.5.
        "null_surrogate": surrogate_null(
            df, m_split=m_split, mode=args.mode, seed=args.seed,
            era_bounds=[base["data"]["val_start"], base["data"]["test_start"]]),
    }

    # --- the checks that decide whether this staging is usable -------------
    tgt0 = df[df.magnitude >= args.m_target]
    print(f"\n[check] target set: {len(tgt0)} events M>={args.m_target:g}")
    print(f"  {'arm':16}{'n_small':>9}{'spread km':>11}{'targets':>9}"
          f"{'small/target +30d':>19}")
    ok = True
    for lbl, d in arms.items():
        sm = d[d.magnitude < m_split]
        tg = d[d.magnitude >= args.m_target]
        st, tt = sm.time.to_numpy(), tg.time.to_numpy()
        w = np.timedelta64(int(args.horizon), "D")
        coup = float(np.mean([((st > t) & (st <= t + w)).sum() for t in tt]))
        print(f"  {lbl:16}{len(sm):>9,}{np.hypot(sm.x.std(), sm.y.std()):>11.1f}"
              f"{len(tg):>9}{coup:>19.1f}")
        if len(tg) != len(tgt0):
            print(f"    *** target set differs — invariant 1 violated ***")
            ok = False

    span = (df.time.max() - df.time.min()) / np.timedelta64(int(args.horizon), "D")
    n_small = int((df.magnitude < m_split).sum())
    print(f"  {'(decoupled baseline)':16}{'':>9}{'':>11}{'':>9}{n_small/span:>19.1f}")

    # Forecastability gate. The test is the LAG-1 CORRELATION between
    # consecutive windows, not the within-window overdispersion.
    #
    # Overdispersion says excess variance EXISTS; the lag-1 correlation says it
    # is PREDICTABLE FROM THE PAST, which is what a forecaster is actually
    # asked to do. The two come apart badly. On WHITE at 30-day windows the
    # counts are 2.90x overdispersed yet the lag-1 correlation is -0.133
    # (p = 0.33): sequences begin and end inside one window, so their
    # burstiness never crosses a boundary and nothing carries over. A fitted
    # ETAS scored corr(expected, observed) = -0.084 there, and so did
    # persistence (-0.150). See MOONSHOT.md invariant 1h.
    #
    # Overdispersion is also horizon-dependent in a way that makes a fixed
    # threshold meaningless: at 1-day windows the mean is ~0.07 events, which
    # drives variance/mean toward 1 however clustered the process is.
    from scipy import stats

    t = tgt0.time
    edges = pd.date_range(t.min(), t.max() + pd.Timedelta(days=args.horizon),
                          freq=f"{int(args.horizon)}D")
    counts = np.histogram(t.astype("int64"),
                          bins=edges.astype("int64"))[0].astype(float)
    od = counts.var() / counts.mean() if counts.mean() else 0.0
    r, p = stats.pearsonr(counts[:-1], counts[1:]) if len(counts) > 4 else (0.0, 1.0)
    print(f"\n[forecastability] M>={args.m_target:g} over {args.horizon:g}-day windows: "
          f"{len(counts)} windows, {int((counts>0).sum())} non-empty, "
          f"mean {counts.mean():.3f}")
    print(f"  variance/mean      {od:>7.2f}   (diagnostic only — horizon-dependent)")
    print(f"  lag-1 correlation  {r:>7.3f}   p={p:.4g}   <- THE gate")
    if not (r > 0 and p < 0.05):
        print("  *** ABORT: no significant between-window signal. No forecaster can\n"
              "      score here, so a flat null would prove nothing (invariant 1h). ***")
        ok = False
    else:
        print("  OK — there is a between-window signal for a forecaster to find")

    if not ok:
        raise SystemExit("\nstaging failed the checks above; not writing frames")

    # --- write catalogs, configs, frames -----------------------------------
    import scripts.scaling_curve as sc
    from flowquake.target_process import aki_utsu_b

    spec = TargetSpec(m_target=args.m_target, m_large=args.m_large,
                      horizon_days=args.horizon, tail_mode="fixed")
    for lbl, d in arms.items():
        p = cat_dir / f"{lbl}.csv"
        w = d.copy()
        # write tz-NAIVE, matching the source files: pandas' read_csv(parse_dates)
        # leaves tz-aware ISO strings as objects, and build_frame then tries to
        # subtract strings
        w["time"] = pd.to_datetime(w["time"], utc=True).dt.tz_localize(None)
        w.to_csv(p, index=False)
        cfg = {**base}
        cfg["data"] = {**base["data"], "catalog_path": str(p)}
        cfg["train"] = {**base["train"], "out_dir": str(out / lbl)}
        cfg_path = out / "_cfg" / f"{lbl}.yaml"
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)

        d_out = out / lbl; d_out.mkdir(parents=True, exist_ok=True)
        c = sc.Config.load(str(cfg_path))
        if lbl == "informative":
            b = aki_utsu_b(
                pd.read_csv(p, parse_dates=["time"])
                .query("time < @c.data.val_start")["magnitude"].to_numpy(),
                float(c.data.mcut))
            fr = sc.build_frame(c, spec, bin_km=args.bin_km, out=d_out, b_mc=b)
            n_t = sum(len(w["obs"]) for w in fr["windows"])
            print(f"[frame] {lbl:16} {len(fr['windows'])} windows, {n_t} targets, "
                  f"{int(fr['active'].sum())} active cells, b={b:.3f}")
        else:
            # COPY the frame rather than rebuild it. Rebuilding gave the arms
            # different grids (854 vs 925 active cells, because rotation moves
            # the footprint) and different fixed tails (b 0.883 vs 0.960,
            # because the time shift changes which events fall in the training
            # era). Either difference means the arms are scoring different
            # regions or different magnitude tails, so any slope between them
            # is partly bookkeeping. The frame -- grid, active mask, windows,
            # target set, b -- is a property of the EXPERIMENT, not of the arm;
            # only the history each probe conditions on may differ.
            import shutil
            shutil.copy(out / "informative" / "frame.json", d_out / "frame.json")
            print(f"[frame] {lbl:16} copied from informative "
                  "(identical grid, mask, windows, targets and fixed tail)")

    print(f"\nwrote {out}")
    print("\nNEXT:\n"
          f"  python scripts/validate_with_etas.py --root {out} \\\n"
          f"      --arms informative null_surrogate --mc 2.5 2.0 1.5 1.0 \\\n"
          f"      --out {out}/etas --concurrency 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
