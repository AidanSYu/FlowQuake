"""Gate G3 figure: the neural curve against its ETAS control, same frame.

What this has to show is not "FlowQuake wins" — it does, at every completeness —
but the SHAPE of how it wins, because that is where the moonshot question lives.
The neural curve is an inverted U peaking half a decade below the reference
completeness, while the ETAS baseline rises monotonically. So the margin between
them SHRINKS as the catalog deepens: deeper catalogs buy the physics baseline
more than they buy the learned model.

Three things are drawn because each was needed to believe the result:

  left    both curves on the same frame, with the margin shaded between them
  middle  the margin alone, with its bootstrapped slope
  right   n_eff_cells — the mechanism. The neural field sharpens ~5x from mc 2.5
          to 1.5 while its accuracy peaks at 2.0, i.e. it grows confident faster
          than it grows correct. ETAS sharpens too, but its accuracy follows.

Seed-replicate points are overplotted as open markers where available, since the
single-seed-per-point design was the most obvious way this shape could have been
an artifact (it is not: spread 0.041-0.109 nats against a 0.94-nat effect).

Usage:
    python scripts/make_g3_figure.py --panel runs/panel_white --background uniform
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import block_bootstrap_slope  # noqa: E402

# Optional seed replicates produced outside the main sweep.
SEED_REPLICATES = {
    2.0: "seed2_mc2", 1.5: "seed2_mc1.5", 1.0: "seed2_mc1",
}


def _agg(w):
    n = sum(x["n_target_obs"] for x in w)
    return (sum(x["ll_shape"] for x in w) / n) if n else float("nan")


def _neff(w):
    hi = [x for x in w if x["n_target_obs"] > 0]
    return float(np.nanmedian([x["n_eff_cells"] for x in hi])) if hi else float("nan")


def load_neural(panel: Path, arm: str):
    out = {}
    for f in sorted(glob.glob(str(panel / f"{arm}_mc*_s*/target_process.json"))):
        m = re.search(rf"{re.escape(arm)}_mc([0-9.]+)_s", f)
        if m:
            out[float(m.group(1))] = json.load(open(f))["windows"]
    return out


def load_etas(panel: Path, bg: str):
    out = {}
    for f in sorted(glob.glob(str(panel / f"etas/{bg}_mc*/target_process.json"))):
        m = re.search(rf"{re.escape(bg)}_mc([0-9.]+)/", f)
        if m:
            out[float(m.group(1))] = json.load(open(f))["windows"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="runs/panel_white")
    ap.add_argument("--background", default="uniform", choices=["uniform", "smoothed"])
    ap.add_argument("--arm", default="matched_window")
    ap.add_argument("--replicate-dir", default=None,
                    help="directory holding seed-replicate runs (optional)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    panel = Path(args.panel)
    out = Path(args.out or f"figures/g3_{panel.name}_{args.arm}_{args.background}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    neural, etas = load_neural(panel, args.arm), load_etas(panel, args.background)
    mcs = sorted(set(neural) & set(etas), reverse=True)
    if len(mcs) < 2:
        print(f"need >=2 shared mc points; have {sorted(set(neural) & set(etas))}")
        return 1

    n_s = [_agg(neural[m]) for m in mcs]
    e_s = [_agg(etas[m]) for m in mcs]
    margin = [a - b for a, b in zip(n_s, e_s)]
    n_ne = [_neff(neural[m]) for m in mcs]
    e_ne = [_neff(etas[m]) for m in mcs]

    reps = {}
    if args.replicate_dir:
        rd = Path(args.replicate_dir)
        for mc, sub in SEED_REPLICATES.items():
            f = rd / sub / "target_process.json"
            if f.exists():
                reps[mc] = _agg(json.load(open(f))["windows"])

    def slope(pts_w):
        sc = np.vstack([np.array([x["ll_shape"] for x in pts_w[m]], dtype=float)
                        for m in mcs])
        tg = np.array([x["n_target_obs"] for x in pts_w[mcs[0]]], dtype=float)
        return block_bootstrap_slope("", mcs, sc, tg, n_boot=2000, seed=0)

    sn, se = slope(neural), slope(etas)
    d = sn.slope - se.slope
    sd = float(np.hypot(sn.se, se.se))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15.5, 5.0),
                                     constrained_layout=True)

    a0.fill_between(mcs, n_s, e_s, color="C0", alpha=0.12, label="margin")
    a0.plot(mcs, n_s, "o-", color="C0", lw=2.2, ms=7, label="FlowQuake", zorder=3)
    a0.plot(mcs, e_s, "s-", color="C1", lw=2.2, ms=6,
            label=f"ETAS ({args.background})", zorder=3)
    if reps:
        rm = sorted(reps, reverse=True)
        a0.plot(rm, [reps[m] for m in rm], "o", mfc="none", mec="C0", ms=10,
                mew=1.6, label="FlowQuake, 2nd seed", zorder=4)
    best = mcs[int(np.argmax(n_s))]
    a0.axvline(best, color="C0", ls=":", lw=1.2)
    a0.annotate(f"peak at $m_c$={best:g}", xy=(best, max(n_s)),
                xytext=(6, -14), textcoords="offset points",
                fontsize=8.5, color="C0")
    a0.invert_xaxis()
    a0.set_xlabel("completeness $m_c$  (deeper catalog $\\rightarrow$)")
    a0.set_ylabel("shape log-likelihood per target event (nats)")
    a0.set_title(f"{panel.name} — {args.arm}")
    a0.grid(alpha=0.3); a0.legend(fontsize=8.5, loc="best")

    a1.axhline(0, color="0.4", ls="--", lw=1)
    a1.plot(mcs, margin, "D-", color="C2", lw=2.2, ms=7)
    a1.invert_xaxis()
    a1.set_xlabel("completeness $m_c$  (deeper catalog $\\rightarrow$)")
    a1.set_ylabel("FlowQuake $-$ ETAS  (nats per target event)")
    a1.set_title("margin over the physics baseline")
    a1.grid(alpha=0.3)
    a1.text(0.5, 0.03,
            f"margin slope {d:+.3f}  [{d - 1.96*sd:+.3f}, {d + 1.96*sd:+.3f}]\n"
            + ("margin SHRINKS with depth" if d + 1.96 * sd < 0 else
               "margin grows with depth" if d - 1.96 * sd > 0 else
               "margin consistent with constant"),
            transform=a1.transAxes, ha="center", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="#f2fff2", ec="C2", alpha=0.95))

    a2.plot(mcs, n_ne, "o-", color="C0", lw=2.2, ms=7, label="FlowQuake")
    a2.plot(mcs, e_ne, "s-", color="C1", lw=2.2, ms=6, label="ETAS")
    a2.invert_xaxis(); a2.set_yscale("log")
    a2.set_xlabel("completeness $m_c$  (deeper catalog $\\rightarrow$)")
    a2.set_ylabel("$n_{\\mathrm{eff}}$ cells  (lower = sharper)")
    a2.set_title("sharpness: confidence vs correctness")
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=8.5)

    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"wrote {out}")
    print(f"  neural slope {sn.slope:+.4f} [{sn.ci_lo:+.4f}, {sn.ci_hi:+.4f}]")
    print(f"  ETAS   slope {se.slope:+.4f} [{se.ci_lo:+.4f}, {se.ci_hi:+.4f}]")
    print(f"  margin slope {d:+.4f} [{d - 1.96*sd:+.4f}, {d + 1.96*sd:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
