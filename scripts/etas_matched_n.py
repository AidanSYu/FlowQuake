"""Is the ETAS gain INFORMATION, or just better parameter estimation?

This is the experiment invariant 1j demands and the moonshot answer depends on.

THE THREAT. The headline is "a decade of magnitude is worth +0.22 nats to ETAS
and -0.71 to a learned model, so the information is demonstrably present and the
learned model fails to use it." That reading assumes ETAS's gain measures
information. There is a competing explanation:

    At mc 2.5 the WHITE inversion has only 520 training events, the likelihood
    is nearly flat along a (K, a) ridge, and the branching ratio pins at its
    barrier at EVERY mc. A 5x change in branching ratio buys 16 nats.

So a deeper catalog hands ETAS more events to ESTIMATE PARAMETERS WITH, and its
forecasts may improve for that reason alone -- nothing to do with small
earthquakes carrying information about large ones. If that is what is happening,
the moonshot's central sentence is wrong.

THE TEST. Refit ETAS at every mc from a FIXED number of training events, taking
the most recent N before the validation boundary. Estimation difficulty is then
constant along the axis and only the information content varies:

    gain survives  -> it is information. The headline stands, now defended.
    gain vanishes  -> it was estimation. The moonshot answer changes.

This is exactly invariant 2 (the matched_n arm) applied to the control, and it
is cheap: ETAS is ~200x cheaper than the neural arm and runs on CPU.

Both arms are reported, because the difference between them is the estimand:
    matched_window  every event at that mc (the published G1 setting)
    matched_n       fixed N, so estimation is held constant

Usage:
    python scripts/etas_matched_n.py --panel runs/panel_white \
        --frame runs/panel_white/frame.json --mc 2.5 2.0 1.5 1.0 --n-train 520
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.config import Config  # noqa: E402
from flowquake.etas_fit import fit_etas_em  # noqa: E402
from flowquake.pooling import DEFAULT_BLOCK_WINDOWS, _resample  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="runs/panel_white")
    ap.add_argument("--base", default="configs/panel_white.yaml")
    ap.add_argument("--mc", type=float, nargs="+", default=[2.5, 2.0, 1.5, 1.0])
    ap.add_argument("--n-train", type=int, default=0,
                    help="fixed training-event budget; 0 = the count at the "
                         "HIGHEST mc, which is the binding one")
    ap.add_argument("--background", default="uniform")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.base)
    frame = json.load(open(f"{args.panel}/frame.json"))
    g = frame["grid"]
    region = (g["xmin"], g["xmin"] + g["nx"] * g["bin_km"],
              g["ymin"], g["ymin"] + g["ny"] * g["bin_km"])

    df = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"]).sort_values("time")
    t0 = df["time"].iloc[0]
    df["t_days"] = (df["time"] - t0).dt.total_seconds() / 86400.0
    val = (pd.Timestamp(cfg.data.val_start) - t0).total_seconds() / 86400.0
    tr0 = (pd.Timestamp(cfg.data.train_start) - t0).total_seconds() / 86400.0

    # How many events does the HIGHEST mc have? That is the binding budget --
    # every other mc must be cut down to it, never up.
    counts = {}
    for mc in args.mc:
        sel = df[(df["magnitude"] >= mc) & (df["t_days"] >= tr0) & (df["t_days"] < val)]
        counts[mc] = len(sel)
    budget = args.n_train or counts[max(args.mc)]
    print(f"training-event counts by mc: "
          + "  ".join(f"{m:g}:{counts[m]}" for m in sorted(counts, reverse=True)))
    print(f"fixed budget N = {budget} (the count at mc {max(args.mc):g})\n")

    rows = []
    for mc in sorted(args.mc, reverse=True):
        sel = df[(df["magnitude"] >= mc) & (df["t_days"] >= tr0) & (df["t_days"] < val)]
        for arm in ("matched_window", "matched_n"):
            d = sel if arm == "matched_window" else sel.iloc[-budget:]
            if len(d) < 50:
                print(f"  [skip] mc {mc:g} {arm}: only {len(d)} events")
                continue
            # fit_etas_em returns (ETASParams, Background, info). The window
            # bounds matter: without them the EM treats the first event as the
            # start of time and the last as the end, which differs between arms
            # (matched_n starts later) and would confound the comparison.
            par, bg, info = fit_etas_em(
                d["t_days"].to_numpy(), d["x"].to_numpy(),
                d["y"].to_numpy(), d["magnitude"].to_numpy(),
                mc=mc, region=region, background=args.background,
                t_start=float(d["t_days"].iloc[0]), t_end=val)
            pd_ = {k: float(v) for k, v in vars(par).items()
                   if isinstance(v, (int, float))}
            rows.append({"mc": mc, "arm": arm, "n_train": int(len(d)),
                         "t_start": float(d["t_days"].iloc[0]),
                         **pd_,
                         **{f"info_{k}": (float(v[-1]) if isinstance(v, (list, tuple))
                                          and v and isinstance(v[-1], (int, float))
                                          else float(v))
                            for k, v in info.items()
                            if isinstance(v, (int, float))
                            or (isinstance(v, (list, tuple)) and v
                                and isinstance(v[-1], (int, float)))}})
            shown = "  ".join(f"{k}={pd_[k]:.4g}" for k in
                              ("K", "a", "c", "p", "d", "q", "mu") if k in pd_)
            # `info` returns per-EM-iteration HISTORIES for some keys, so take
            # the converged (last) value rather than assuming a scalar.
            def _last(*names):
                for nm in names:
                    v = info.get(nm)
                    if isinstance(v, (list, tuple)) and v:
                        return float(v[-1])
                    if isinstance(v, (int, float)):
                        return float(v)
                return float("nan")
            ll = _last("ll", "loglik", "ll_hist")
            nb = _last("n_branch", "branching", "n", "branching_ratio")
            print(f"  mc {mc:>4g}  {arm:<15} N={len(d):>6}  {shown}  "
                  f"n={nb:.4g}  ll={ll:.1f}")

    out = args.out or f"{args.panel}/etas_matched_n_{args.background}.json"
    json.dump({"budget": budget, "counts": {str(k): v for k, v in counts.items()},
               "background": args.background, "rows": rows}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print("\nNEXT: score these fits on the frame and compare the per-decade slope\n"
          "of matched_n against matched_window. If matched_n's slope collapses\n"
          "toward zero, the published ETAS gain was parameter estimation, not\n"
          "information, and the moonshot's central sentence needs rewriting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
