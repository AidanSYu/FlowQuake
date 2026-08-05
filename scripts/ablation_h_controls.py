"""Gate G2 — the memorization ablation with the three missing controls.

`MANUSCRIPT.md` 4.3 concludes: flexibility causes memorization, and the cure is
structural. Two things make that claim weaker than it reads:

  1. The encoder is handed ABSOLUTE x, y (`flowquake/model.py`, token dims 1-2)
     and trained for ~600+ epochs. None of the three standard fixes for
     coordinate memorization was ever run, so the supported statement is
     "absolute-coordinate conditioning plus long training memorizes" — not
     "flexibility memorizes".
  2. The reported gap is an AGGREGATE NLL. Decomposing the committed
     `runs/ablation_h/memorization_figure.json` at h=4/ckpt_last gives
     tll 3.126 -> -6.181 (9.31 nats) and sll -7.27 -> -13.47 (6.20 nats):
     **~60% of the effect is in the temporal flow head**, whose target is
     log-tau and contains no coordinates at all. Section 4.3 attributes the
     whole effect to "pinning mass on the exact training epicentres".

     The capacity story is also contradicted by the same file: the gap
     DECREASES with bottleneck width (15.50 at h=4, 14.55 at h=16, 14.06 at
     h=64), the opposite of what a flexibility-causes-memorization law predicts.

This script runs the grid that settles it, and always reports the decomposition.

Arms (`flowquake/model.py:_encoder_input`):
    full       the published setting
    safe       absolute x, y never reach the encoder
    augmented  random rotation/reflection/translation of the encoder's frame

Outcome to look for: if ANY arm keeps h>0 at or above the h=0 test NLL, the
section must be rewritten, and the encoder may be salvageable — which changes
what architecture the moonshot can use.

Usage:
    python scripts/ablation_h_controls.py --base configs/n1_density.yaml \
        --out runs/ablation_h_controls --h 0 4 16 --arms full safe augmented
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.config import Config

PY = sys.executable
ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": ".",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def plan(base: str, hs, arms, d_floors, out: Path):
    """One config per (h, arm, d_floor). h=0 has no encoder, so it needs only
    one arm — running it three times would just be three seeds of the same
    model and would misleadingly suggest the arm mattered."""
    jobs = []
    for h in hs:
        for arm in (["full"] if h == 0 else arms):
            for df in d_floors:
                tag = f"h{h}_{arm}_d{df:g}"
                cfg = Config.load(base)
                cfg.model.h_bottleneck = int(h)
                cfg.model.encoder_input = arm
                cfg.model.d_floor_km = float(df)
                cfg.train.out_dir = str(out / tag)
                cdir = out / "_cfg"; cdir.mkdir(parents=True, exist_ok=True)
                cp = str(cdir / f"{tag}.yaml"); cfg.dump(cp)
                jobs.append((tag, out / tag / "ckpt_best.pt", cp, h, arm, df))
    return jobs


def run(jobs, steps, device, concurrency):
    todo = [j for j in jobs if not j[1].exists()]
    print(f"[G2] {len(todo)}/{len(jobs)} to train, concurrency={concurrency}", flush=True)
    running, queue, fin, t0 = [], list(todo), 0, time.time()
    while queue or running:
        while queue and len(running) < concurrency:
            tag, ckpt, cp, *_ = queue.pop(0)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            fh = open(ckpt.parent / "train.log", "w")
            cmd = [PY, "-m", "flowquake.train", cp, "--out", str(ckpt.parent),
                   "--device", device]
            if steps:
                cmd += ["--steps", str(steps)]
            running.append((tag, subprocess.Popen(cmd, env=ENV, stdout=fh,
                                                  stderr=subprocess.STDOUT), fh))
            print(f"  [start] {tag}", flush=True)
        while running and all(p.poll() is None for _, p, _ in running):
            time.sleep(2.0)
        for tag, proc, fh in list(running):
            if proc.poll() is not None:
                running.remove((tag, proc, fh)); fh.close(); fin += 1
                print(f"  [done ] {tag} rc={proc.returncode} ({fin}/{len(todo)}, "
                      f"{(time.time()-t0)/60:.1f}min)", flush=True)


def evaluate(jobs, device, n_sub, ode_steps):
    """Per-arm train/test tll and sll, so the gap can be decomposed."""
    import torch
    from flowquake.data import full_sequence_batch
    from flowquake.train import load_catalog_cfg, make_model

    rows = []
    for tag, ckpt, cp, h, arm, df in jobs:
        res_path = ckpt.parent / "memorization.json"
        if res_path.exists():
            rows.append(json.load(open(res_path))); continue
        if not ckpt.exists():
            print(f"[skip] {tag}: no checkpoint"); continue
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        cfg = ck["cfg"]; cat = load_catalog_cfg(cfg)
        dev = torch.device(device)
        model = make_model(cfg, ck["stats"]).to(dev).eval()
        model.load_state_dict(ck["model"])
        row = {"tag": tag, "h": h, "arm": arm, "d_floor": df}
        for split in ("train", "test"):
            tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, split)
            if n_sub:
                idx = mask[0].nonzero(as_tuple=True)[0]
                if len(idx) > n_sub:
                    sel = np.random.default_rng(0).choice(len(idx), n_sub, replace=False)
                    keep = torch.zeros_like(mask[0]); keep[idx[torch.from_numpy(sel)]] = True
                    mask = keep.unsqueeze(0)
            with torch.no_grad():
                o = model.log_likelihood(tokens.to(dev), target.to(dev), mask.to(dev),
                                         lastk.to(dev), raw_next.to(dev), steps=ode_steps)
            row[f"{split}_tll"] = float(o["tll"].mean())
            row[f"{split}_sll"] = float(o["sll"].mean())
            row[f"{split}_nll"] = -(row[f"{split}_tll"] + row[f"{split}_sll"])
        row["gap_nll"] = row["test_nll"] - row["train_nll"]
        row["gap_from_tll"] = row["train_tll"] - row["test_tll"]
        row["gap_from_sll"] = row["train_sll"] - row["test_sll"]
        tot = abs(row["gap_from_tll"]) + abs(row["gap_from_sll"])
        row["frac_gap_temporal"] = abs(row["gap_from_tll"]) / tot if tot else None
        json.dump(row, open(res_path, "w"), indent=2)
        rows.append(row)
        print(f"  {tag}: gap {row['gap_nll']:.3f} "
              f"({row['frac_gap_temporal']:.0%} temporal)", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="runs/ablation_h_controls")
    ap.add_argument("--h", type=int, nargs="+", default=[0, 4, 16])
    ap.add_argument("--arms", nargs="+", default=["full", "safe", "augmented"],
                    choices=["full", "safe", "augmented"])
    ap.add_argument("--d-floor", type=float, nargs="+", default=[0.1])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--n-sub", type=int, default=4096)
    ap.add_argument("--ode-steps", type=int, default=64)
    ap.add_argument("--rescue-margin", type=float, default=0.10,
                    help="nats of test-NLL improvement over h=0 required "
                         "before declaring a rescue; guards against seed noise")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = plan(args.base, args.h, args.arms, args.d_floor, out)
    if not args.eval_only:
        run(jobs, args.steps, args.device, args.concurrency)
    rows = evaluate(jobs, args.device, args.n_sub, args.ode_steps)
    json.dump(rows, open(out / "controls.json", "w"), indent=2)

    base = next((r for r in rows if r["h"] == 0), None)
    print(f"\n{'arm':22}{'h':>4}{'train_nll':>11}{'test_nll':>10}{'gap':>9}"
          f"{'%temporal':>11}{'vs h=0':>9}")
    for r in sorted(rows, key=lambda r: (r["arm"], r["h"])):
        d = (r["test_nll"] - base["test_nll"]) if base else float("nan")
        print(f"{r['tag']:22}{r['h']:>4}{r['train_nll']:>11.3f}{r['test_nll']:>10.3f}"
              f"{r['gap_nll']:>9.3f}{r['frac_gap_temporal']:>10.0%}{d:>9.3f}")

    if base:
        # A rescue must clear a margin. The published effect is ~15 nats; a
        # single-seed difference of a few thousandths is seed noise, and
        # declaring a rescue on it would invert 4.3's conclusion on nothing.
        # Compare against the effect being explained, not against zero.
        mgn = args.rescue_margin
        rescued = [r for r in rows
                   if r["h"] > 0 and base["test_nll"] - r["test_nll"] >= mgn]
        near = [r for r in rows if r["h"] > 0
                and abs(r["test_nll"] - base["test_nll"]) < mgn]
        print()
        if rescued:
            print(f"*** GATE G2: control(s) RESCUE the encoder (margin {mgn} nats):")
            for r in rescued:
                print(f"      {r['tag']}: test nll {r['test_nll']:.3f} "
                      f"vs h=0 {base['test_nll']:.3f} "
                      f"({base['test_nll']-r['test_nll']:+.3f})")
            print("    MANUSCRIPT.md 4.3 must be rewritten: the failure was\n"
                  "    absolute-coordinate conditioning, not flexibility. The\n"
                  "    encoder is available to the moonshot architecture.\n"
                  "    CONFIRM ON >=3 SEEDS before acting on this.")
        elif near and not any(r["h"] > 0 and r["test_nll"] - base["test_nll"] > mgn
                              for r in rows):
            print(f"GATE G2: every h>0 arm is within {mgn} nats of h=0 — i.e. the\n"
                  "    ~15-nat published collapse did NOT reproduce here. Check the\n"
                  "    run actually trained to convergence before reading anything\n"
                  "    into it; at low step counts all arms are untrained and equal.")
        else:
            print("GATE G2: no control rescues h>0; the structural claim survives,\n"
                  "    but 4.3 must still be restated with the decomposition above —\n"
                  "    the temporal share of the gap is not about geography.")
    print(f"\nwrote {out/'controls.json'}")


if __name__ == "__main__":
    main()
