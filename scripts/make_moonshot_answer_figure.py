"""The killer figure: the two model classes disagree about the SIGN.

MOONSHOT.md: "If that figure does not exist, there is no paper."

What it has to show is not "FlowQuake wins" -- it does, at every completeness --
but that the two curves run in OPPOSITE DIRECTIONS on identical target events,
and that the learned model's entire advantage is spent within the measured span.

Three panels, each earning its place:

  left    both curves on one frame with the margin shaded. The crossing point is
          the operationally interesting quantity, so the margin is annotated at
          both ends rather than left to be read off.
  middle  slopes with bootstrap intervals, on a signed axis. This is the claim
          in one picture: the intervals do not overlap and do not share a sign.
  right   n_eff_cells, the mechanism. ETAS sharpens 2.37x as the catalog
          deepens while its score improves; FlowQuake gets slightly BROADER
          (0.75x) while its score falls. The learned model is not becoming
          overconfident -- its spatial resolution is SATURATED, so the extra
          events move the forecast without focusing it.

Follows the house style of make_g3_figure.py (same panel idiom, inverted mc
axis, C0/C1 series) so it sits beside the existing paper figures.

Usage:
    python scripts/make_moonshot_answer_figure.py
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moonshot_answer import PLATEAU_FROM, load_etas, load_neural  # noqa: E402


def neff(panel, bg):
    out = {}
    for f in glob.glob(f"{panel}/etas/{bg}_mc*/target_process.json"):
        mc = float(re.search(rf"{bg}_mc([0-9.]+)/", f).group(1))
        w = [x for x in json.load(open(f))["windows"] if x["n_target_obs"] > 0]
        out[mc] = float(np.nanmedian([x["n_eff_cells"] for x in w])) if w else np.nan
    return out


def neff_neural(surface):
    per = {}
    for f in glob.glob(f"{surface}/*/score_step*/target_process.json"):
        step = int(re.search(r"score_step(\d+)", f).group(1))
        if step < PLATEAU_FROM:
            continue
        d = json.load(open(f))
        w = [x for x in d["windows"] if x["n_target_obs"] > 0]
        if w:
            per.setdefault(float(d["mcut"]), []).append(
                float(np.nanmedian([x["n_eff_cells"] for x in w])))
    return {m: float(np.mean(v)) for m, v in per.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="runs/surface_white")
    ap.add_argument("--panel", default="runs/panel_white")
    ap.add_argument("--background", default="uniform")
    ap.add_argument("--out", default="figures/moonshot_answer.png")
    args = ap.parse_args(argv)

    res = json.load(open(f"{args.surface}/moonshot_answer.json"))
    mcs = res["mcs"]
    n_s, e_s = res["neural"], res["etas"]
    margin = [a - b for a, b in zip(n_s, e_s)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15.5, 5.0),
                                     constrained_layout=True)

    a0.fill_between(mcs, n_s, e_s, color="C0", alpha=0.10)
    a0.plot(mcs, n_s, "o-", color="C0", lw=2.4, ms=7.5, label="FlowQuake", zorder=3)
    a0.plot(mcs, e_s, "s-", color="C1", lw=2.4, ms=6.5,
            label=f"ETAS ({args.background})", zorder=3)
    for i in (0, len(mcs) - 1):
        a0.annotate(f"margin {margin[i]:+.2f}", xy=(mcs[i], (n_s[i] + e_s[i]) / 2),
                    xytext=(0, 0), textcoords="offset points", fontsize=8.5,
                    ha="center", color="0.25",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=.9))
    a0.invert_xaxis()
    a0.set_xlabel("completeness $m_c$   (deeper catalog $\\rightarrow$)")
    a0.set_ylabel("shape log-likelihood per target event (nats)")
    a0.set_title("identical targets, opposite directions")
    a0.grid(alpha=0.3); a0.legend(fontsize=9, loc="lower left")

    # --- the claim in one panel -------------------------------------------
    names = ["ETAS", "FlowQuake"]
    vals = [res["slope_etas"], res["slope_neural"]]
    cis = [res["slope_etas_ci"], res["slope_neural_ci"]]
    cols = ["C1", "C0"]
    y = [1, 0]
    a1.axvline(0, color="0.35", ls="--", lw=1.2, zorder=1)
    for yi, v, c, col in zip(y, vals, cis, cols):
        a1.plot([c[0], c[1]], [yi, yi], color=col, lw=3.5, solid_capstyle="round")
        a1.plot([v], [yi], "o", color=col, ms=11, zorder=3)
        a1.annotate(f"{v:+.3f}  [{c[0]:+.2f}, {c[1]:+.2f}]", xy=(v, yi),
                    xytext=(0, 15), textcoords="offset points",
                    ha="center", fontsize=9, color=col)
    a1.set_yticks(y); a1.set_yticklabels(names, fontsize=11)
    a1.set_ylim(-0.6, 1.6)
    a1.set_xlabel("nats per decade of magnitude\n(positive = deeper catalog helps)")
    a1.set_title("the intervals do not share a sign")
    a1.grid(alpha=0.3, axis="x")
    a1.text(0.5, 0.04, f"P(signs differ) = {res['p_sign_differs']:.3f}",
            transform=a1.transAxes, ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="#f2fff2", ec="C2", alpha=0.95))

    ne_n, ne_e = neff_neural(args.surface), neff(args.panel, args.background)
    mm = [m for m in mcs if m in ne_n and m in ne_e]
    a2.plot(mm, [ne_n[m] for m in mm], "o-", color="C0", lw=2.4, ms=7.5,
            label="FlowQuake")
    a2.plot(mm, [ne_e[m] for m in mm], "s-", color="C1", lw=2.4, ms=6.5,
            label="ETAS")
    a2.invert_xaxis(); a2.set_yscale("log")
    a2.set_xlabel("completeness $m_c$   (deeper catalog $\\rightarrow$)")
    a2.set_ylabel("$n_{\\mathrm{eff}}$ cells   (lower = sharper)")
    a2.set_title("mechanism: ETAS sharpens, FlowQuake saturates")
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=9)

    fig.suptitle(
        "A decade of magnitude buys the physics baseline "
        f"{res['slope_etas']:+.2f} nats and the learned model "
        f"{res['slope_neural']:+.2f} — on the same {res['n_targets']} target events",
        fontsize=12.5)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200); plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
