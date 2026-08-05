"""Build the frame for a panel without training anything.

The ETAS control (gate G1) must run on the SAME frame as that panel's neural
curve — same grid, same active mask, same windows, same target set, same fixed
GR tail — or it is not a control, it is a different experiment. `build_frame`
is normally reached only through a full `scaling_curve.py` run, which trains
first. This exposes it directly so the ETAS control can be produced while the
neural arm is still grinding, and so a frame is never re-derived a second time
by a later run (`build_frame` reloads `frame.json` if it exists).

The per-panel arguments must match `scripts/run_panels.sh` exactly. Bin size in
particular is NOT cosmetic: each value gives ~6 active cells per target event so
the panels' shape scores are comparable before pooling.

    panel              bin   m_target  m_large  mc grid
    WHITE               2.0    3.0       4.0    2.5 2.0 1.5 1.0
    QTM San Jacinto     5.0    3.0       4.0    2.3 1.8 1.3
    QTM Salton Sea      3.0    3.0       4.0    2.5 2.1 1.7
    ComCat California  15.0    4.0       5.0    3.5 3.0 2.8

`b_mc` is the MOST COMPLETE threshold (max of the grid) for the one-off b
estimate; `mc_ref` is the LEAST complete (min of the grid), the matched-
resolution reference. They are deliberately opposite ends — see invariants 1c
and 1d.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.scaling_curve as sc                      # noqa: E402
from flowquake.config import Config                     # noqa: E402
from flowquake.target_process import TargetSpec         # noqa: E402

PANELS = {
    "white":      ("configs/panel_white.yaml",         "runs/panel_white",
                   2.0, 3.0, 4.0, [2.5, 2.0, 1.5, 1.0]),
    "sanjac":     ("configs/panel_qtm_sanjac.yaml",    "runs/panel_qtm_sanjac",
                   5.0, 3.0, 4.0, [2.3, 1.8, 1.3]),
    "saltonsea":  ("configs/panel_qtm_saltonsea.yaml", "runs/panel_qtm_saltonsea",
                   3.0, 3.0, 4.0, [2.5, 2.1, 1.7]),
    "comcat":     ("configs/moonshot_lowmc.yaml",      "runs/panel_comcat",
                   15.0, 4.0, 5.0, [3.5, 3.0, 2.8]),
}


def build(name: str, horizon: float) -> dict:
    cfg_path, out_dir, bin_km, m_target, m_large, mcs = PANELS[name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(cfg_path)
    spec = TargetSpec(m_target=m_target, m_large=m_large,
                      horizon_days=horizon, tail_mode="fixed")

    frame = sc.build_frame(cfg, spec, bin_km, out, b_mc=max(mcs))

    # Matched-resolution reference is the LOWEST mc (invariant 1d).
    if frame.get("mc_ref") != min(mcs):
        frame["mc_ref"] = float(min(mcs))
        fj = json.load(open(out / "frame.json"))
        fj["mc_ref"] = frame["mc_ref"]
        json.dump(fj, open(out / "frame.json", "w"))

    g = frame["grid"]
    n_act, n_tgt = frame["n_active_cells"], frame["n_target_events"]
    cpe = n_act / max(n_tgt, 1)
    fj = json.load(open(out / "frame.json"))
    print(f"[{name:10}] b={frame['b_value']:.3f} (mc={frame['b_mc']:g})  "
          f"{frame['n_windows']} windows x {horizon:g}d  "
          f"{n_tgt} M>={m_target} targets  "
          f"grid {g.nx}x{g.ny} @ {g.bin_km:g}km -> {n_act} cells  "
          f"{cpe:.1f} cells/target  t0={fj.get('t0', 'MISSING')}")
    if cpe > 200:
        print(f"*** {name}: {cpe:.0f} cells per target event — the score will be "
              f"dominated by the water level, not skill. Coarsen bin_km.")
    elif cpe < 3:
        print(f"*** {name}: only {cpe:.1f} cells per target event — too coarse.")
    if "t0" not in fj:
        print(f"*** {name}: frame has no t0 — cross-catalog comparisons unsafe "
              f"(invariant 1n).")
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="*", default=list(PANELS),
                    choices=list(PANELS) + [[]])
    ap.add_argument("--horizon", type=float, default=1.0,
                    help="1 day; the only horizon with between-window signal "
                         "in all four panels (invariant 1h)")
    args = ap.parse_args(argv)
    for name in (args.panels or list(PANELS)):
        build(name, args.horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
