#!/usr/bin/env python
"""Build the causal completeness mask and write the masked catalog.

MOONSHOT.md invariant 1e. The scaling curve's x-axis is only meaningful where
the catalog is actually complete at the threshold being tested, and California
statewide is not complete at mc 1.5 -- b-value stability puts statewide Mc at
2.6. Per 1-degree cell it is 1.2-1.8 across the instrumented interior, so the
fix is to restrict the region rather than to raise the grid.

The mask restricts BOTH training and scoring, deliberately. Restricting only
the scoring would leave the model training on sub-Mc data outside the region,
where "fewer events" means "fewer detected events"; at low mc it would learn
the shape of the seismic network as though it were the shape of seismicity.

Usage:
    python scripts/build_completeness_mask.py \
        --catalog reference/Datasets/ComCat_lowmc/ComCat_lowmc_catalog.csv \
        --train-end 2011-01-01 --mc-threshold 1.5 --deg 1.0 \
        --out reference/Datasets/ComCat_masked
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.completeness import (  # noqa: E402
    build_mask, mc_by_b_stability, mc_by_max_curvature,
)
from flowquake.target_process import aki_utsu_b  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-end", default="2011-01-01",
                    help="mask is estimated from events BEFORE this date only")
    ap.add_argument("--train-start", default=None,
                    help="optional lower bound on the estimation era")
    ap.add_argument("--mc-threshold", type=float, default=1.5,
                    help="lowest mc on the scaling grid; cells worse than this "
                         "are dropped")
    ap.add_argument("--deg", type=float, default=1.0)
    ap.add_argument("--min-events", type=int, default=2000)
    ap.add_argument("--test-start", default="2014-01-01")
    ap.add_argument("--test-end", default="2020-01-17")
    ap.add_argument("--m-target", type=float, default=4.0)
    ap.add_argument("--m-large", type=float, default=5.0)
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.catalog, parse_dates=["time"]).sort_values("time")
    df = df.reset_index(drop=True)
    print(f"[load] {len(df):,} events  {df.time.min().date()} .. {df.time.max().date()}")

    est = df[df.time < pd.Timestamp(args.train_end)]
    if args.train_start:
        est = est[est.time >= pd.Timestamp(args.train_start)]

    # --- what the naive statewide numbers say, for the record ---------------
    m_all = est["magnitude"].to_numpy()
    mc_state = mc_by_b_stability(m_all, lo=0.0, hi=4.0)
    print(f"\n[statewide, estimation era {len(est):,} events]")
    print(f"  max-curvature Mc : {mc_by_max_curvature(m_all):.2f}   "
          "<- optimistic; do NOT set the grid from this")
    print(f"  b-stability   Mc : {mc_state}")
    print(f"  {'mc':>5}{'b':>8}")
    for c in (1.0, 1.5, 2.0, 2.5, 3.0):
        try:
            print(f"  {c:>5.1f}{aki_utsu_b(m_all, c):>8.3f}")
        except ValueError:
            pass

    # --- the mask -----------------------------------------------------------
    mask = build_mask(df, train_end=args.train_end,
                      mc_threshold=args.mc_threshold, deg=args.deg,
                      min_events=args.min_events)
    print(f"\n[mask] {len(mask)} of {len(mask.mc_by_cell)} evaluated cells "
          f"have training-era Mc <= {args.mc_threshold}")
    print(f"  {'lat':>6}{'lon':>7}{'Mc':>6}   status")
    for k in sorted(mask.mc_by_cell):
        v = mask.mc_by_cell[k]
        tag = "KEEP" if k in mask.cells else "drop"
        print(f"  {k[0]:>6.1f}{k[1]:>7.1f}{str(v):>6}   {tag}")

    print(f"\n[verify] pooled Mc of the accepted cells = {mask.mc_union}")
    if mask.mc_union is None or mask.mc_union > args.mc_threshold:
        print("  *** FAIL: the cells are individually complete but their "
              "MIXTURE is not.\n"
              "      Raise --mc-threshold or shrink --deg. Do not proceed.")
    else:
        print(f"  OK: <= {args.mc_threshold}, so the grid may go down to "
              f"mc {args.mc_threshold}")

    # --- cost accounting ----------------------------------------------------
    kept = mask.apply(df)
    te_all = df[(df.time >= pd.Timestamp(args.test_start)) &
                (df.time < pd.Timestamp(args.test_end))]
    te_kept = kept[(kept.time >= pd.Timestamp(args.test_start)) &
                   (kept.time < pd.Timestamp(args.test_end))]
    print(f"\n[cost] the mask is not free -- record what it removes")
    print(f"  {'':22}{'statewide':>12}{'masked':>10}{'kept':>8}")
    rows = [("all events", len(df), len(kept)),
            (f"test M>={args.m_target:g} targets",
             int((te_all.magnitude >= args.m_target).sum()),
             int((te_kept.magnitude >= args.m_target).sum())),
            (f"test M>={args.m_large:g}",
             int((te_all.magnitude >= args.m_large).sum()),
             int((te_kept.magnitude >= args.m_large).sum()))]
    for lbl, a, b in rows:
        print(f"  {lbl:22}{a:>12,}{b:>10,}{(b/a if a else 0):>7.0%}")
    for c in (3.0, 2.5, 2.0, 1.5):
        print(f"  events at mc {c:<9.1f}{int((df.magnitude>=c).sum()):>12,}"
              f"{int((kept.magnitude>=c).sum()):>10,}")

    # --- write --------------------------------------------------------------
    mask_path = out / "completeness_mask.json"
    json.dump(mask.to_dict(), open(mask_path, "w"), indent=2)
    cat_path = out / "catalog_masked.csv"
    kept.to_csv(cat_path, index=False)
    meta = {
        "source_catalog": args.catalog,
        "train_end": args.train_end,
        "train_start": args.train_start,
        "mc_threshold": args.mc_threshold,
        "deg": args.deg,
        "statewide_mc_bstability": mc_state,
        "statewide_mc_maxcurvature": float(mc_by_max_curvature(m_all)),
        "mask_mc_union": mask.mc_union,
        "n_cells": len(mask),
        "n_events_statewide": int(len(df)),
        "n_events_masked": int(len(kept)),
        "n_test_targets_statewide": rows[1][1],
        "n_test_targets_masked": rows[1][2],
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"\n[write] {mask_path}")
    print(f"[write] {cat_path}  ({len(kept):,} events)")
    print(f"[write] {out/'meta.json'}")
    print("\nPoint the run config's data.catalog_path at the masked catalog. "
          "The same mask must be used for FlowQuake, ETAS and every control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
