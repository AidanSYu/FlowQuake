"""The (mc, training-step) surface — kills the checkpoint-selection confound.

WHY THIS EXISTS. G3's headline is that forecast skill has an INTERIOR OPTIMUM in
mc: it rises +0.7500 from mc 2.5 to 2.0, then falls -0.6677 from 1.5 to 1.0.
Adversarial review left that claim standing but narrowed it, and one narrowing
could not be bounded numerically:

    the mc 2.5 anchor is the EARLIEST checkpoint its run ever saved -- step 200,
    with val_every=200 and warmup=400, i.e. mid-warmup at half peak learning
    rate, and val NLL rising monotonically afterwards (11.6084 -> 11.6276 ->
    12.1037 -> 13.47). Its true optimum lies inside (0, 200) and was OVERWRITTEN.

So the mc 2.5 score is a lower bound on that model's best, and the +0.7500 rise
is an UPPER BOUND on the true increment. Worse, early stopping makes "which
checkpoint gets reported" a function of mc -- and mc is the axis under study, so
the selection rule is confounded with the effect.

This script removes the rule instead of arguing about it. Every point is trained
with early stopping DISABLED for a fixed step budget, every validation
checkpoint is kept, and ll_shape is scored on a grid of them. The result is a
surface over (mc, step) from which the per-mc optimum is read off directly.

THE GRID IS DENSE EARLY ON PURPOSE. A uniform grid would step straight over
(0, 200) -- precisely the interval the experiment exists to inspect. Default is
every 50 steps to 500, then every 500 to the budget.

Usage
-----
    # train + score, GPU
    python scripts/checkpoint_surface.py --panel runs/surface_white \
        --base configs/panel_white.yaml --mc 2.5 2.0 1.5 1.0 \
        --device cuda --max-lanes 65536

    # rehearse the whole pipeline in a couple of minutes on CPU
    python scripts/checkpoint_surface.py --panel runs/surface_dry \
        --base configs/panel_white.yaml --mc 2.5 --steps 100 --val-every 25 \
        --device cpu --n-sims 20 --max-windows 6 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scaling_curve as sc  # noqa: E402
from flowquake.config import Config  # noqa: E402
from flowquake.target_process import TargetSpec  # noqa: E402


def checkpoint_grid(budget: int, val_every: int, dense_until: int = 500,
                    dense_every: int = 50, sparse_every: int = 500) -> list[int]:
    """Steps to SCORE. Dense early, sparse later.

    Saving is cheap (~90 KB a checkpoint) and scoring is not, so the two
    decisions are deliberately decoupled: train saves at every `val_every`, and
    only this subset is scored. Everything else stays on disk for a later zoom.

    Dense early because the confound being removed lives at small step counts.
    """
    want = set(range(dense_every, min(dense_until, budget) + 1, dense_every))
    want |= set(range(sparse_every, budget + 1, sparse_every))
    want.add(budget)
    # Only steps that training will actually have written.
    return sorted(s for s in want if s % val_every == 0 and s <= budget)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--mc", type=float, nargs="+", required=True)
    ap.add_argument("--arm", default="matched_window", choices=list(sc.ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-lanes", type=int, default=sc.DEFAULT_MAX_LANES)
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--sample-steps", type=int, default=16)
    ap.add_argument("--m-target", type=float, default=3.0)
    ap.add_argument("--m-large", type=float, default=4.0)
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--bin-km", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="dry-run only: truncate the frame to N windows")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny end-to-end rehearsal; NOT a scientific result")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.panel); out.mkdir(parents=True, exist_ok=True)
    grid = checkpoint_grid(args.steps, args.val_every)
    print(f"[surface] mc={args.mc} arm={args.arm} budget={args.steps} "
          f"val_every={args.val_every}")
    print(f"[surface] scoring {len(grid)} checkpoints per mc: "
          f"{grid[:6]}{' ...' if len(grid) > 6 else ''} (last {grid[-1]})")
    if args.dry_run:
        print("[surface] DRY RUN — rehearsing the pipeline, not producing a result")

    base = Config.load(args.base)
    spec = TargetSpec(m_target=args.m_target, m_large=args.m_large,
                      horizon_days=args.horizon, tail_mode="fixed")

    # ---- frame: built once, identical for every point (invariant 1) ----
    # b_mc uses the MOST COMPLETE threshold for the one-off b estimate, matching
    # scaling_curve; mc_ref is the LOWEST mc, the matched-resolution reference
    # every other point scales its n_sims up to (invariant 1d).
    frame = sc.build_frame(base, spec, args.bin_km, out, b_mc=max(args.mc))
    frame["mc_ref"] = float(min(args.mc))
    from dataclasses import replace as _replace
    spec = _replace(spec, b_value=frame["b_value"])
    if args.max_windows:
        # Prefer windows that actually CONTAIN target events. Taking the first N
        # gave 6 empty windows, so the rehearsal skipped the entire shape-scoring
        # path and reported an undefined score -- a dry run that exercises
        # nothing is worse than no dry run, because it looks like a pass.
        wins = frame["windows"]
        withtgt = [w for w in wins if len(w.get("obs", []))]
        without = [w for w in wins if not len(w.get("obs", []))]
        frame["windows"] = (withtgt[:args.max_windows]
                            + without[:max(0, args.max_windows - len(withtgt))])
        print(f"[surface] frame truncated to {len(frame['windows'])} windows "
              f"({sum(1 for w in frame['windows'] if len(w.get('obs', [])))} "
              f"with targets) (dry run)")
    print(f"[frame] {len(frame['windows'])} windows | b={frame['b_value']:.3f} "
          f"| mc_ref={frame['mc_ref']:g}")

    # ---- train, early stopping OFF, keeping every checkpoint ----
    # matched_window holds the calendar window fixed (the headline arm), so
    # train_start comes straight from the base config. matched_n would need
    # solve_train_start; it is out of scope here because the review established
    # the two arms share their mc 2.5 model bit-for-bit, so the confound this
    # experiment removes is identical in both.
    if args.arm != "matched_window":
        raise SystemExit("only matched_window is supported; matched_n needs "
                         "solve_train_start and adds no independent information "
                         "about the checkpoint confound")
    points = []
    for mc in args.mc:
        p = sc.prepare_point(args.base, mc, args.arm, args.seed,
                             base.data.train_start, out)
        if p is None:
            continue
        # prepare_point writes the base config's val_every (1000 for
        # panel_white). Without overriding it here, --val-every would only
        # change which steps this script SCORES while training saved almost
        # nothing -- the dry run caught exactly that, reporting "1 of 2 grid
        # steps were never saved". The checkpoint cadence and the scoring grid
        # have to come from the same number.
        pc = Config.load(p[2])
        pc.train.val_every = int(args.val_every)
        pc.train.patience = 10 ** 9          # belt and braces; --no-early-stop
        pc.dump(p[2])                        # is the real switch
        points.append(p)

    todo = [p for p in points if not any((p[1].parent).glob("ckpt_step*.pt"))]
    if todo:
        sc.train_many(todo, args.steps, args.device, args.concurrency,
                      train_flags=("--keep-all-ckpts", "--no-early-stop"))
    else:
        print("[surface] all points already trained")
    if args.train_only:
        return 0

    # ---- score the grid ----
    rows = []
    for tag, ckpt_best, _ in points:
        d = ckpt_best.parent
        have = {int(re.search(r"ckpt_step(\d+)\.pt", f.name).group(1)): f
                for f in d.glob("ckpt_step*.pt")}
        missing = [s for s in grid if s not in have]
        if missing:
            print(f"  [{tag}] {len(missing)} of {len(grid)} grid steps were never "
                  f"saved (training stopped early?): {missing[:8]}")
        for step in [s for s in grid if s in have]:
            ck = have[step]
            # score_point writes next to the checkpoint; give each its own dir so
            # the grid does not overwrite itself.
            work = d / f"score_step{step:06d}"
            work.mkdir(exist_ok=True)
            target = work / ck.name
            if not target.exists():
                target.write_bytes(ck.read_bytes())
            # Matched resolution (invariant 1d) is mandatory for a real result and
            # ruinous for a rehearsal: it overrode --n-sims 20 with 98,155 at
            # mc 2.5, making the dry run cost more than the thing it rehearses.
            # A dry run is checking that the PIPELINE runs, not measuring skill.
            res = sc.score_point(target, frame, spec, args.n_sims, args.device,
                                 args.sample_steps,
                                 match_precision_ref=(None if args.dry_run
                                                      else frame.get("mc_ref")),
                                 max_lanes=args.max_lanes)
            if res is None:
                print(f"  [{tag}] step {step}: scoring returned nothing")
                continue
            a = res["aggregate"]
            ll = a.get("ll_shape_per_target_event")
            if ll is None:
                # No target events in scope -> the shape term is undefined, not
                # zero. Record it and move on rather than formatting None.
                print(f"  [{tag}] step {step:6d}  ll_shape=undefined "
                      f"({a.get('n_target_events', 0)} target events in scope)",
                      flush=True)
            rows.append({"tag": tag, "mc": res["mcut"], "step": step,
                         "ll_shape_per_target_event": ll,
                         "ll_per_target_event": a.get("ll_per_target_event"),
                         "n_target_events": a.get("n_target_events"),
                         "n_sims": res["n_sims"], "max_lanes": res["max_lanes"],
                         "device": res["device"]})
            if ll is not None:
                print(f"  [{tag}] step {step:6d}  ll_shape={ll:+.4f}", flush=True)

    surf = out / ("surface_dryrun.json" if args.dry_run else "surface.json")
    json.dump({"arm": args.arm, "seed": args.seed, "steps": args.steps,
               "val_every": args.val_every, "grid": grid,
               "dry_run": bool(args.dry_run), "rows": rows},
              open(surf, "w"), indent=2)
    print(f"\nwrote {surf}  ({len(rows)} scored points)")

    # ---- read the per-mc optimum straight off the surface ----
    if rows:
        print(f"\n{'mc':>6}{'best step':>11}{'best ll_shape':>15}{'n':>5}")
        for mc in sorted({r['mc'] for r in rows}, reverse=True):
            sub = [r for r in rows if r["mc"] == mc
                   and r["ll_shape_per_target_event"] is not None]
            if not sub:
                print(f"{mc:>6g}{'--':>11}{'no scored points':>15}{0:>5}")
                continue
            b = max(sub, key=lambda r: r["ll_shape_per_target_event"])
            print(f"{mc:>6g}{b['step']:>11}"
                  f"{b['ll_shape_per_target_event']:>15.4f}{len(sub):>5}")
        print("\nThe published curve used whatever checkpoint early stopping "
              "happened to keep. These are the per-mc optima with that rule "
              "removed; recompute the increments from THESE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
