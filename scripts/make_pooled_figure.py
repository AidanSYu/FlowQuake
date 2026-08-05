#!/usr/bin/env python
"""THE killer figure for the multi-region design: panels + a pooled slope.

`make_moonshot_figure.py` draws one region in detail (both arms, ETAS controls,
the sign-aware interpretation). This draws all of them together and produces
the single number the paper is built around.

Why this exists. No catalog on hand gives both magnitude range and target
count: dense template-matched catalogs reach mc 0.6 but only over ~90 km and
~10 years, while ComCat reaches mc 2.5 statewide for 25 years. Each panel
therefore spans only ~1 decade, and the headline slope has to come from
combining them (see "The data plan" in MOONSHOT.md).

Two statistical choices carry that combination, and both are made in
`flowquake.pooling` rather than here:

  * WITHIN a panel, uncertainty comes from a block bootstrap over forecast
    WINDOWS, resampled once per replicate and applied to every mc. Target
    events arrive in aftershock sequences -- 87% of the ComCat mask's targets
    are one Ridgecrest sequence -- so an event-level bootstrap would report an
    interval several times too narrow.
  * ACROSS panels, DerSimonian-Laird random effects. Regions differ in
    tectonics, network, span and even M_TARGET; a fixed-effect pool would
    treat them as repeated measurements of one quantity and understate the
    uncertainty.

The metric is the SHAPE (CSEP S-test) term by default, per invariant 1c: it is
invariant to rescaling lambda, so a mis-specified magnitude tail cannot move
it. `--metric total` reproduces the tail-sensitive version for comparison.

Usage:
    python scripts/make_pooled_figure.py \
        --panel "San Jacinto (WHITE)=runs/panel_white" \
        --panel "San Jacinto (QTM)=runs/panel_qtm_sanjac" \
        --panel "Salton Sea (QTM)=runs/panel_qtm_saltonsea" \
        --panel "California (ComCat)=runs/panel_comcat" \
        --arm matched_window --out figures/moonshot_pooled.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import (  # noqa: E402
    block_bootstrap_slope, random_effects_pool,
)

METRIC_KEY = {"shape": "ll_shape", "level": "ll_level", "total": "ll"}


def load_panel(name: str, run_dir: Path, arm: str, metric: str):
    """Per-window scores at each mc for one panel.

    Returns (mcs, scores[n_mc, n_win], targets[n_win]) or None if the panel has
    fewer than two scored points, since a slope needs at least two.
    """
    curve = run_dir / "curve.json"
    if not curve.exists():
        print(f"  [skip] {name}: no curve.json in {run_dir}")
        return None
    rows = [r for r in json.load(open(curve)) if r["arm"] == arm]
    rows.sort(key=lambda r: -r["mc"])

    key = METRIC_KEY[metric]
    mcs, per_mc, targets = [], [], None
    for r in rows:
        # the scored artifact lives beside the checkpoint for that point
        hits = sorted(run_dir.glob(f"{arm}_mc{r['mc']:g}_s*/target_process.json"))
        if not hits:
            continue
        d = json.load(open(hits[0]))
        w = d["windows"]
        if key not in w[0]:
            print(f"  [skip] {name} mc={r['mc']:g}: no '{key}' — rescore with the "
                  "N-test/S-test split")
            continue
        vals = np.array([x[key] for x in w], dtype=float)
        n = np.array([x["n_target_obs"] for x in w], dtype=float)
        if targets is None:
            targets = n
        elif len(n) != len(targets) or not np.array_equal(n, targets):
            # invariant 1: every point on a curve is scored on the SAME events
            raise SystemExit(
                f"{name} mc={r['mc']:g}: target counts differ from the other "
                f"points on this curve. Invariant 1 is violated; the panel is "
                f"not comparable across mc and must be rebuilt.")
        mcs.append(float(r["mc"]))
        per_mc.append(vals)

    if len(mcs) < 2:
        print(f"  [skip] {name}: only {len(mcs)} scored point(s), need >=2")
        return None
    return mcs, np.vstack(per_mc), targets


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="NAME=RUNDIR, repeatable")
    ap.add_argument("--arm", default="matched_window",
                    choices=["matched_window", "matched_n"])
    ap.add_argument("--metric", default="shape", choices=list(METRIC_KEY))
    ap.add_argument("--out", default="figures/moonshot_pooled.png")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print(f"[load] arm={args.arm} metric={args.metric}")
    panels, raw = [], {}
    for spec in args.panel:
        if "=" not in spec:
            raise SystemExit(f"--panel needs NAME=RUNDIR, got {spec!r}")
        name, run_dir = spec.split("=", 1)
        got = load_panel(name, Path(run_dir), args.arm, args.metric)
        if got is None:
            continue
        mcs, scores, tgt = got
        p = block_bootstrap_slope(name, mcs, scores, tgt,
                                  n_boot=args.n_boot, seed=args.seed)
        panels.append(p)
        raw[name] = (mcs, scores, tgt)
        print(f"  [ok] {name}: {len(mcs)} points, {p.n_windows} windows, "
              f"{p.n_targets} targets, slope {p.slope:+.4f}")

    if not panels:
        raise SystemExit("no panel had >=2 scored points; nothing to pool")

    pooled = random_effects_pool(panels)
    print("\n" + "=" * 74)
    print(f"POOLED SLOPE — {args.metric} term, {args.arm} arm")
    print("=" * 74)
    print(pooled.summary())

    # --- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig = plt.figure(figsize=(3.1 * n + 3.4, 4.2), layout="constrained")
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.25])

    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(p.mcs, p.scores, "o-", color="#1f77b4", lw=2, ms=6)
        ax.invert_xaxis()
        ax.set_title(p.name, fontsize=9)
        ax.set_xlabel(r"$m_c$  (deeper $\rightarrow$)", fontsize=8)
        if i == 0:
            ax.set_ylabel(f"{args.metric} score per target event (nats)", fontsize=9)
        ax.grid(alpha=0.25, ls=":")
        ax.tick_params(labelsize=8)
        ax.text(0.04, 0.04, f"{p.slope:+.3f}\n[{p.ci_lo:+.3f}, {p.ci_hi:+.3f}]",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    # forest plot: per-panel slopes and the pooled estimate
    ax = fig.add_subplot(gs[0, n])
    ys = np.arange(len(panels))[::-1]
    for y, p in zip(ys, panels):
        ax.plot([p.ci_lo, p.ci_hi], [y, y], color="#444", lw=1.6)
        ax.plot([p.slope], [y], "s", color="#1f77b4", ms=6)
    ax.plot([pooled.ci_lo, pooled.ci_hi], [-1, -1], color="#d62728", lw=2.6)
    ax.plot([pooled.estimate], [-1], "D", color="#d62728", ms=8)
    ax.axvline(0.0, color="0.5", ls="--", lw=1)
    ax.set_yticks(list(ys) + [-1])
    ax.set_yticklabels([p.name for p in panels] + ["POOLED"], fontsize=8)
    ax.set_xlabel("slope (nats per decade of $m_c$)", fontsize=9)
    ax.set_title(f"$I^2$={pooled.i2:.0%}   $p_Q$={pooled.q_p:.3f}", fontsize=9)
    ax.grid(alpha=0.25, ls=":", axis="x")
    ax.tick_params(labelsize=8)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)   # constrained layout, no tight_layout

    payload = {
        "arm": args.arm, "metric": args.metric,
        "pooled": {k: getattr(pooled, k) for k in
                   ("estimate", "se", "ci_lo", "ci_hi", "tau2", "i2", "q",
                    "q_df", "q_p")},
        "panels": [{"name": p.name, "slope": p.slope, "se": p.se,
                    "ci_lo": p.ci_lo, "ci_hi": p.ci_hi, "mcs": p.mcs,
                    "scores": p.scores, "n_windows": p.n_windows,
                    "n_targets": p.n_targets} for p in panels],
    }
    json.dump(payload, open(out.with_suffix(".json"), "w"), indent=2)
    print(f"\nwrote {out}\nwrote {out.with_suffix('.json')}")

    # --- what the number means --------------------------------------------
    # Sign and heterogeneity both change the sentence. Never describe a
    # negative slope with the language of a gain.
    e, lo, hi = pooled.estimate, pooled.ci_lo, pooled.ci_hi
    print()
    if lo <= 0.0 <= hi:
        print(f"READ: pooled slope {e:+.3f} [{lo:+.3f}, {hi:+.3f}] — the interval "
              f"COVERS ZERO.\n  Across these regions, deepening the catalog does not "
              f"measurably improve\n  M>=M_TARGET forecasts. That is the LIMITS result, "
              f"and gate G3 says publish it\n  rather than hunt for a positive slope.")
    elif e > 0:
        print(f"READ: pooled slope {e:+.3f} [{lo:+.3f}, {hi:+.3f}] nats per decade, "
              f"excluding zero.\n  Small earthquakes carry forecast information about "
              f"larger ones, and the\n  amount is now measured rather than asserted.")
    else:
        print(f"READ: pooled slope {e:+.3f} [{lo:+.3f}, {hi:+.3f}] — NEGATIVE and "
              f"excluding zero.\n  Deeper catalogs make these forecasts WORSE. Before "
              f"publishing that, rule out\n  the two known ways to manufacture it: "
              f"per-cell completeness (invariant 1e)\n  and Monte-Carlo precision "
              f"matching (invariant 1d).")
    if pooled.heterogeneous:
        print(f"\n  HETEROGENEITY: I^2={pooled.i2:.0%}, Q p={pooled.q_p:.3f}. The "
              f"regions disagree by more\n  than sampling noise explains, so the pooled "
              f"number is an average over\n  genuinely different quantities. Report the "
              f"per-region slopes as the primary\n  result — 'the limit varies by "
              f"region' is a different and more interesting\n  claim than a single "
              f"universal constant, not a defect to be smoothed away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
