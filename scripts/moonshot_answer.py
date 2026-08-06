"""The moonshot number: how much forecast skill does a decade of magnitude buy?

Answers the question MOONSHOT.md was written to earn the right to ask, using
the CORRECTED neural curve (the checkpoint-surface plateau, not the
early-stopping artefact) against the ETAS control on a bit-identical frame:
same 1,673 windows, same 132 target events, same grid, verified by hash.

The answer is not the one either branch of the original either/or anticipated.
The physics baseline GAINS a little as the catalog deepens; the flexible learned
model LOSES a lot. Both are measured on the same targets, so the divergence is
about the models, not the data.

That reframes the result. The quantity being bounded is not "the information in
small earthquakes" -- ETAS demonstrably extracts some. It is the ability of a
given model class to convert catalog depth into forecast skill, and on that axis
a high-capacity density model is not merely failing to gain, it is going
backwards.

Everything is a paired circular block bootstrap over WINDOWS (30-day blocks,
because targets arrive in aftershock sequences), with one window draw per
replicate shared by every model and every mc so each contrast stays within-window.

Usage:
    python scripts/moonshot_answer.py
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

PLATEAU_FROM = 1000


def load_neural(panel: str):
    """mc -> (n_ckpt, n_win) plateau ll_shape, plus the shared target counts."""
    per: dict[float, dict[int, np.ndarray]] = {}
    tgt = None
    for f in glob.glob(f"{panel}/*/score_step*/target_process.json"):
        step = int(re.search(r"score_step(\d+)", f).group(1))
        if step < PLATEAU_FROM:
            continue
        d = json.load(open(f))
        w = d["windows"]
        per.setdefault(float(d["mcut"]), {})[step] = np.array(
            [x["ll_shape"] for x in w], dtype=float)
        if tgt is None:
            tgt = np.array([x["n_target_obs"] for x in w], dtype=float)
    steps = sorted(set.intersection(*(set(v) for v in per.values())))
    return {m: np.stack([per[m][s] for s in steps]) for m in per}, tgt, steps


def load_etas(panel: str, bg: str):
    out = {}
    for f in glob.glob(f"{panel}/etas/{bg}_mc*/target_process.json"):
        mc = float(re.search(rf"{bg}_mc([0-9.]+)/", f).group(1))
        out[mc] = np.array([x["ll_shape"] for x in json.load(open(f))["windows"]],
                           dtype=float)
    return out


def per_decade(mcs, vals):
    """Nats per decade of mc LOWERED (positive = deeper catalog helps)."""
    return float(-np.polyfit(np.asarray(mcs, float), np.asarray(vals, float), 1)[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="runs/surface_white")
    ap.add_argument("--panel", default="runs/panel_white")
    ap.add_argument("--background", default="uniform")
    ap.add_argument("--n-boot", type=int, default=6000)
    args = ap.parse_args(argv)

    neural, tgt, steps = load_neural(args.surface)
    etas = load_etas(args.panel, args.background)
    mcs = sorted(set(neural) & set(etas), reverse=True)
    n_win = len(tgt)
    print(f"frame: {n_win} windows, {int(tgt.sum())} scored target events")
    print(f"neural plateau: {len(steps)} checkpoints, steps {steps[0]}..{steps[-1]}")
    print(f"mc grid: {mcs}   ETAS background: {args.background}\n")

    def agg(idx):
        tot = tgt[idx].sum()
        if tot <= 0:
            return None, None
        n = np.array([neural[m][:, idx].sum(axis=1).mean() / tot for m in mcs])
        e = np.array([etas[m][idx].sum() / tot for m in mcs])
        return n, e

    n0, e0 = agg(np.arange(n_win))
    print(f"{'mc':>6}{'FlowQuake':>12}{'ETAS':>10}{'margin':>10}")
    for i, m in enumerate(mcs):
        print(f"{m:>6g}{n0[i]:>12.4f}{e0[i]:>10.4f}{n0[i]-e0[i]:>10.4f}")

    rng = np.random.default_rng(0)
    bn = np.empty((args.n_boot, len(mcs)))
    be = np.empty((args.n_boot, len(mcs)))
    for b in range(args.n_boot):
        idx = _resample(rng, n_win, DEFAULT_BLOCK_WINDOWS)
        n, e = agg(idx)
        bn[b], be[b] = (n, e) if n is not None else (np.nan, np.nan)

    def ci(d):
        return np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)

    sl_n = np.array([per_decade(mcs, r) for r in bn])
    sl_e = np.array([per_decade(mcs, r) for r in be])
    pn, pe = per_decade(mcs, n0), per_decade(mcs, e0)
    print(f"\n{'':<12}{'nats per decade':>18}{'95% CI':>24}")
    for nm, p, d in (("FlowQuake", pn, sl_n), ("ETAS", pe, sl_e)):
        lo, hi = ci(d)
        print(f"{nm:<12}{p:>18.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}")
    dd = sl_e - sl_n
    lo, hi = ci(dd)
    print(f"{'ETAS - FQ':<12}{pe-pn:>18.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}")
    print(f"\nP(ETAS slope > 0)      = {float(np.mean(sl_e > 0)):.4f}")
    print(f"P(FlowQuake slope < 0) = {float(np.mean(sl_n < 0)):.4f}")
    print(f"P(they differ in SIGN) = {float(np.mean((sl_e > 0) & (sl_n < 0))):.4f}")

    # Total endpoint decline, the most robust statement available.
    tn = bn[:, -1] - bn[:, 0]
    te = be[:, -1] - be[:, 0]
    print(f"\ntotal mc {mcs[0]:g} -> {mcs[-1]:g}:")
    for nm, p, d in (("FlowQuake", n0[-1]-n0[0], tn), ("ETAS", e0[-1]-e0[0], te)):
        lo, hi = ci(d)
        print(f"  {nm:<10}{p:>10.4f}  [{lo:+.4f}, {hi:+.4f}]")

    json.dump({"mcs": mcs, "background": args.background,
               "neural": list(map(float, n0)), "etas": list(map(float, e0)),
               "slope_neural": pn, "slope_etas": pe,
               "slope_neural_ci": list(map(float, ci(sl_n))),
               "slope_etas_ci": list(map(float, ci(sl_e))),
               "p_sign_differs": float(np.mean((sl_e > 0) & (sl_n < 0))),
               "n_checkpoints": len(steps), "n_windows": n_win,
               "n_targets": int(tgt.sum())},
              open(f"{args.surface}/moonshot_answer.json", "w"), indent=2)
    print(f"\nwrote {args.surface}/moonshot_answer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
