"""Per-step increments with CIs — the summary invariant 1k actually asks for.

A single "nats per decade" slope is the wrong statistic when the curve bends,
and the ETAS probe's curve bends hard: RV9's informative arm gains +0.2174 from
mc 2.5 to 2.0, then +0.1164, then +0.0218 — roughly halving each half-decade.
A linear fit through that reports one number that describes none of the steps,
and its CI spanning zero is then read as "no effect" when the truth is "a real
effect at the top of the range that has saturated by the bottom".

So report the STEPS, each with its own paired block-bootstrap interval, using
the same resampling as `pooling.block_bootstrap_slope`: windows drawn once per
replicate and applied to every mc, so each increment stays a within-window
contrast.

Also reports, per invariant 1q, the informative arm's own curve separately from
the informative-minus-null difference — the difference validates the instrument,
the arm's own curve is the claim.

Usage:
    python scripts/saturation_diagnostic.py <run_dir> [--background uniform]
      e.g. runs/real_validation_h1/etas_gridregion
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import DEFAULT_BLOCK_WINDOWS, _resample  # noqa: E402


def load_arm(arm_dir: Path, background: str) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """mc -> (per-window shape score, per-window target count)."""
    out = {}
    for d in sorted(arm_dir.glob(f"{background}_mc*")):
        f = d / "target_process.json"
        if not f.exists():
            continue
        mc = float(d.name.replace(f"{background}_mc", ""))
        w = json.load(open(f))["windows"]
        if not w or "ll_shape" not in w[0]:
            continue
        out[mc] = (np.array([x["ll_shape"] for x in w], dtype=float),
                   np.array([x["n_target_obs"] for x in w], dtype=float))
    return out


def weighted(scores: np.ndarray, tgt: np.ndarray) -> float:
    """Nats per target event, aggregated exactly as `pooling.block_bootstrap_slope`.

    `ll_shape` is already a per-window TOTAL, so the aggregate is
    sum(scores) / sum(targets) — NOT a target-weighted mean of per-window values.
    Weighting by target count double-counts the busy windows: on RV9's mc 2.5
    informative point it returns -10.3212 where the artifact's own
    `ll_shape_per_target_event` is -5.5511.
    """
    s = tgt.sum()
    return float(scores.sum() / s) if s > 0 else float("nan")


def step_increments(pts, n_boot=2000, seed=0, ci=0.95,
                    block_len=DEFAULT_BLOCK_WINDOWS):
    """Increment between consecutive mc, each with a paired-bootstrap CI.

    The pairing matters: windows are resampled ONCE per replicate and the same
    draw is applied at both ends of every step. Resampling per-mc would break
    the within-window contrast and inflate every interval.
    """
    mcs = sorted(pts, reverse=True)
    if len(mcs) < 2:
        return mcs, []
    scores = np.vstack([pts[m][0] for m in mcs])          # (n_mc, n_win)
    tgt = pts[mcs[0]][1]
    n_win = scores.shape[1]
    rng = np.random.default_rng(seed)

    point = [weighted(scores[i], tgt) for i in range(len(mcs))]
    boots = np.empty((n_boot, len(mcs)))
    for b in range(n_boot):
        # Contiguous blocks, not i.i.d. days. The windows are consecutive 1-day
        # forecasts and aftershock sequences stay correlated for weeks, so an
        # i.i.d. draw resamples a unit far smaller than the dependence and
        # reports intervals that are too narrow. See pooling._resample.
        idx = _resample(rng, n_win, block_len)
        t = tgt[idx]
        tot = t.sum()
        if tot <= 0:
            boots[b, :] = np.nan
            continue
        # Same draw applied at every mc -- the pairing that makes each increment
        # a within-window contrast.
        boots[b, :] = scores[:, idx].sum(axis=1) / tot

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    rows = []
    for i in range(len(mcs) - 1):
        d = boots[:, i + 1] - boots[:, i]          # lower mc minus higher mc
        rows.append({
            "from_mc": mcs[i], "to_mc": mcs[i + 1],
            "increment": point[i + 1] - point[i],
            "ci_lo": float(np.nanpercentile(d, lo_q)),
            "ci_hi": float(np.nanpercentile(d, hi_q)),
            "decades": mcs[i] - mcs[i + 1],
        })
    return mcs, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--background", default="uniform")
    ap.add_argument("--arms", nargs="+", default=["informative", "null_surrogate"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    root = Path(args.run_dir)
    print(f"\nSATURATION DIAGNOSTIC — {root}")
    print("per-step increments in the SHAPE term (invariant 1c), "
          "paired block bootstrap over windows\n")

    arms = {}
    for arm in args.arms:
        pts = load_arm(root / arm, args.background)
        if len(pts) < 2:
            print(f"[{arm}] fewer than 2 scored points; skipping")
            continue
        arms[arm] = pts
        mcs, rows = step_increments(pts, n_boot=args.n_boot)
        print(f"[{arm}]  curve: " +
              "  ".join(f"mc{m:g}={weighted(*pts[m]):.4f}" for m in mcs))
        print(f"  {'step':>14}{'increment':>12}{'95% CI':>26}{'per decade':>13}{'':>4}")
        for r in rows:
            sig = "*" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else ""
            per = r["increment"] / r["decades"] if r["decades"] else float("nan")
            step = f"{r['from_mc']:g} -> {r['to_mc']:g}"
            interval = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            print(f"  {step:>14}{r['increment']:>12.4f}{interval:>26}"
                  f"{per:>13.4f} {sig}")
        print("  * = interval excludes zero\n")

    # invariant 1q: the difference is the instrument check, not the claim.
    if len(arms) >= 2:
        lead, null = args.arms[0], args.arms[1]
        if lead in arms and null in arms:
            common = sorted(set(arms[lead]) & set(arms[null]), reverse=True)
            if len(common) >= 2:
                print("[difference — instrument check only, NOT the claim (1q)]")
                for m in common:
                    d = weighted(*arms[lead][m]) - weighted(*arms[null][m])
                    print(f"    mc {m:>4g}   informative - null = {d:+.4f}")
                span = common[0] - common[-1]
                d_hi = weighted(*arms[lead][common[0]]) - weighted(*arms[null][common[0]])
                d_lo = weighted(*arms[lead][common[-1]]) - weighted(*arms[null][common[-1]])
                print(f"\n  difference grows {d_lo - d_hi:+.4f} over {span:g} decades "
                      f"({(d_lo - d_hi)/span:+.4f}/decade)")
                inc = weighted(*arms[lead][common[-1]]) - weighted(*arms[lead][common[0]])
                print(f"  informative arm alone gains {inc:+.4f} "
                      f"({inc/span:+.4f}/decade)")
                if abs(inc) > 1e-12:
                    print(f"  -> quoting the difference overstates the claim by "
                          f"{(d_lo - d_hi)/inc:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
