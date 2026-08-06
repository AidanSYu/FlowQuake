"""Honest intervals on the (mc, step) surface increments.

The surface result was reported with a standard error computed from the spread
of 23 plateau checkpoints, treated as independent samples. THEY ARE NOT. They
come from one training run and are serially correlated in training step, so that
standard error is too narrow -- the same mistake as invariant 1r (a bootstrap
named "block" that resampled single days), in a new costume.

There are two distinct sources of uncertainty here and they need different
treatment:

  WINDOWS   which aftershock sequences happen to fall in the test period.
            Target events arrive in sequences, so windows are resampled in
            CONTIGUOUS BLOCKS (pooling._resample, L = 30 days), paired across
            every mc and every checkpoint so each increment stays a
            within-window contrast.

  CHECKPOINTS  where in the plateau a given run happens to sit. Also a
            correlated series -- adjacent checkpoints share almost all their
            weights -- so this axis gets a circular block draw too rather than
            an i.i.d. one.

The headline interval resamples BOTH, because both vary between one honest
repetition of this experiment and the next. The components are reported
separately so it is visible which dominates.

What this does NOT cover: training seed. One seed was run. Seed variation is a
third source and is deliberately absent from these intervals rather than
silently folded in.

Usage:
    python scripts/surface_intervals.py --panel runs/surface_white
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import DEFAULT_BLOCK_WINDOWS, _resample  # noqa: E402

#: Block length on the CHECKPOINT axis, in grid points. The plateau grid is
#: every 500 steps, and val NLL wanders on a scale of a few thousand steps, so
#: ~3 grid points is the shortest block that carries that correlation.
DEFAULT_BLOCK_CKPTS = 3


def load_surface(panel: str, plateau_from: int):
    """-> (mcs, steps_per_mc, S[mc, ckpt, window], T[window])."""
    per_mc: dict[float, dict[int, np.ndarray]] = {}
    targets = None
    for f in glob.glob(f"{panel}/*/score_step*/target_process.json"):
        step = int(re.search(r"score_step(\d+)", f).group(1))
        if step < plateau_from:
            continue
        d = json.load(open(f))
        mc = float(d["mcut"])
        w = d["windows"]
        per_mc.setdefault(mc, {})[step] = np.array(
            [x["ll_shape"] for x in w], dtype=float)
        if targets is None:
            targets = np.array([x["n_target_obs"] for x in w], dtype=float)
    mcs = sorted(per_mc, reverse=True)
    # Every mc must contribute the SAME checkpoint steps, or the comparison
    # silently becomes "different mc at different training stages" -- which is
    # precisely the confound this whole experiment exists to remove.
    common = sorted(set.intersection(*(set(per_mc[m]) for m in mcs)))
    S = np.stack([np.stack([per_mc[m][s] for s in common]) for m in mcs])
    return mcs, common, S, targets


def bootstrap(S, T, n_boot=4000, seed=0, ci=0.95,
              block_win=DEFAULT_BLOCK_WINDOWS, block_ck=DEFAULT_BLOCK_CKPTS,
              resample_windows=True, resample_ckpts=True):
    """Plateau mean per mc, with a paired two-axis circular block bootstrap.

    Returns (point[mc], draws[boot, mc]).
    """
    n_mc, n_ck, n_win = S.shape

    def agg(w_idx, c_idx):
        # sum over windows / sum of targets -- ll_shape is a per-window TOTAL,
        # so this is nats per target event (never a target-weighted mean of
        # per-window values; that error is pinned in test_saturation_diagnostic)
        tot = T[w_idx].sum()
        if tot <= 0:
            return np.full(n_mc, np.nan)
        return (S[:, c_idx][:, :, w_idx].sum(axis=2) / tot).mean(axis=1)

    point = agg(np.arange(n_win), np.arange(n_ck))
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, n_mc))
    for b in range(n_boot):
        # ONE window draw and ONE checkpoint draw per replicate, applied to
        # every mc: that pairing is what makes each increment a within-window,
        # within-stage contrast rather than a difference of two noisy means.
        wi = _resample(rng, n_win, block_win) if resample_windows else np.arange(n_win)
        ci_ = _resample(rng, n_ck, block_ck) if resample_ckpts else np.arange(n_ck)
        draws[b] = agg(wi, ci_)
    return point, draws


def report(mcs, point, draws, label, ci=0.95):
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    print(f"\n--- {label} ---")
    print(f"{'step':>14}{'increment':>12}{'95% CI':>24}{'P(<0)':>9}")
    out = []
    for i in range(len(mcs) - 1):
        d = draws[:, i + 1] - draws[:, i]
        inc = point[i + 1] - point[i]
        lo, hi = np.nanpercentile(d, lo_q), np.nanpercentile(d, hi_q)
        p_neg = float(np.mean(d < 0))
        sig = "*" if (lo > 0 or hi < 0) else ""
        print(f"{f'{mcs[i]:g} -> {mcs[i+1]:g}':>14}{inc:>12.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}{p_neg:>9.4f} {sig}")
        out.append((mcs[i], mcs[i + 1], inc, lo, hi, p_neg))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="runs/surface_white")
    ap.add_argument("--plateau-from", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args(argv)

    mcs, steps, S, T = load_surface(args.panel, args.plateau_from)
    print(f"surface: {len(mcs)} mc x {len(steps)} checkpoints x {S.shape[2]} windows")
    print(f"plateau steps: {steps[0]}..{steps[-1]}  |  targets: {int(T.sum())}")
    print(f"block lengths: {DEFAULT_BLOCK_WINDOWS} windows, {DEFAULT_BLOCK_CKPTS} checkpoints")

    print(f"\n{'mc':>6}{'plateau mean':>15}")
    pt, _ = bootstrap(S, T, n_boot=1)
    for m, v in zip(mcs, pt):
        print(f"{m:>6g}{v:>15.4f}")

    both = report(mcs, *bootstrap(S, T, n_boot=args.n_boot), "BOTH axes (headline)")
    report(mcs, *bootstrap(S, T, n_boot=args.n_boot, resample_ckpts=False),
           "windows only (component)")
    report(mcs, *bootstrap(S, T, n_boot=args.n_boot, resample_windows=False),
           "checkpoints only (component)")

    # The published claim was an INTERSECTION hypothesis: a rise AND a fall.
    _, draws = bootstrap(S, T, n_boot=args.n_boot)
    rise = draws[:, 1] - draws[:, 0]          # 2.5 -> 2.0
    fall = draws[:, -1] - draws[:, -2]        # 1.5 -> 1.0
    print(f"\nP(rise > 0 AND fall < 0) = {float(np.mean((rise > 0) & (fall < 0))):.4f}"
          f"   [published claim: 1.0000]")
    print(f"P(monotone non-increasing across all three steps) = "
          f"{float(np.mean(np.all(np.diff(draws, axis=1) <= 0, axis=1))):.4f}")

    json.dump({"mcs": mcs, "plateau_from": args.plateau_from,
               "steps": steps, "increments": both},
              open(f"{args.panel}/intervals.json", "w"), indent=2)
    print(f"\nwrote {args.panel}/intervals.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
