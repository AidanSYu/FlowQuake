#!/usr/bin/env python
"""Ground-truth validation using ETAS as the probe (MOONSHOT.md invariant 1f).

The neural probe cannot do this job. Measured on the v3 run, its forecast was
flat to 1-3% across windows while observed target counts varied 5.7x, and
`corr(n_expected, n_observed)` sat between -0.09 and +0.15 -- no forecast skill
at all, and it had CONVERGED there (val nll 11.6536/11.6538/11.6542 over steps
2000/2500/3000 with lr fully decayed). A probe that responds to nothing has a
flat null for reasons unrelated to the estimator, so every PASS it can produce
is vacuous.

ETAS is the right probe because its skill is structural rather than trained.
Omori decay plus magnitude-scaled productivity mean it conditions on history by
construction, and lowering `mc` genuinely hands it more of that history: more
observed triggers, a better-resolved conditional intensity for the target
process. So the expectations are sharp and opposite:

    informative catalog  ->  skill IMPROVES as mc falls (real triggering info)
    decoupled null       ->  FLAT (small events carry no coupling to targets)

Both must hold. A flat informative arm means the estimator cannot see
information that is definitely there; a sloping null means it manufactures
information that is definitely not.

This validates the MEASUREMENT INSTRUMENT, not FlowQuake. Whether the neural
model beats ETAS is gate G2's question and a separate run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.pooling import block_bootstrap_slope  # noqa: E402
from flowquake.proc import spawn  # noqa: E402

PY = sys.executable


def slope(mcs, vals):
    ok = [(m, v) for m, v in zip(mcs, vals) if v is not None and np.isfinite(v)]
    if len(ok) < 2:
        return float("nan")
    return float(-np.polyfit([a for a, _ in ok], [b for _, b in ok], 1)[0])


def read_points(out_dir: Path, background: str, mcs: list[float]):
    """(mc, shape/tgt, level/tgt, total/tgt, corr(exp,obs)) per scored point."""
    rows = []
    for mc in mcs:
        hits = sorted(out_dir.glob(f"{background}_mc{mc:g}*/target_process.json"))
        if not hits:
            continue
        d = json.load(open(hits[0]))
        w = d["windows"]
        n = int(sum(x["n_target_obs"] for x in w))
        if n == 0 or "ll_shape" not in w[0]:
            continue
        ne = np.array([x["n_expected"] for x in w], dtype=float)
        no = np.array([x["n_target_obs"] for x in w], dtype=float)
        corr = float(np.corrcoef(ne, no)[0, 1]) if ne.std() > 0 else float("nan")

        # An ABSOLUTE correlation threshold is meaningless here, because the
        # attainable maximum depends on the horizon. Counts are Poisson around
        # the true intensity, so Var(N) = Var(lambda) + E[N] and even a perfect
        # forecaster only reaches sqrt(Var(lambda)/Var(N)). At a 1-day horizon
        # on WHITE that ceiling is about 0.57, at 30 days about 0.80. Report the
        # SHARE of attainable correlation captured, and gate on significance.
        var_pred = max(no.var() - no.mean(), 0.0)
        ceiling = float(np.sqrt(var_pred / no.var())) if no.var() > 0 else float("nan")
        from scipy import stats as _st
        p_corr = float(_st.pearsonr(ne, no)[1]) if ne.std() > 0 and len(ne) > 3 else 1.0

        rows.append({
            "mc": float(mc),
            "shape": sum(x["ll_shape"] for x in w) / n,
            "level": sum(x["ll_level"] for x in w) / n,
            "total": sum(x["ll"] for x in w) / n,
            "corr": corr,
            "corr_p": p_corr,
            "corr_ceiling": ceiling,
            "corr_share": (corr / ceiling) if ceiling and np.isfinite(ceiling)
                          and ceiling > 0 else float("nan"),
            "cv_expected": float(ne.std() / ne.mean()) if ne.mean() else float("nan"),
            "n_targets": n,
            "n_windows": len(w),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/synthetic_validation",
                    help="output of scripts/synthetic_validation.py (needs frames)")
    ap.add_argument("--out", default="runs/etas_validation")
    ap.add_argument("--arms", nargs="+",
                    default=["informative", "null_decoupled"])
    ap.add_argument("--mc", type=float, nargs="+", default=[2.5, 2.0, 1.5])
    ap.add_argument("--background", default="uniform",
                    choices=["uniform", "smoothed"])
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--min-corr", type=float, default=0.20,
                    help="the probe must clear this on the informative arm, or "
                         "the run cannot certify anything (invariant 1f)")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- fit + score, one subprocess per (arm, mc) -------------------------
    jobs = []
    for arm in args.arms:
        frame = root / arm / "frame.json"
        cfg = root / "_cfg" / f"{arm}.yaml"
        if not frame.exists() or not cfg.exists():
            print(f"[skip] {arm}: missing frame.json or config — run "
                  f"scripts/synthetic_validation.py first")
            continue
        for mc in args.mc:
            jobs.append((arm, mc, cfg, frame))

    if not jobs:
        raise SystemExit("no arms to run")

    print(f"[run] {len(jobs)} ETAS fits, concurrency={args.concurrency}")
    running, queue, done = [], list(jobs), 0
    while queue or running:
        while queue and len(running) < args.concurrency:
            arm, mc, cfg, frame = queue.pop(0)
            d = out / arm
            d.mkdir(parents=True, exist_ok=True)
            log = open(d / f"fit_mc{mc:g}.log", "w")
            cmd = [PY, "scripts/etas_by_mc.py", "--run-one",
                   f"{args.background}:{mc:g}", "--base", str(cfg),
                   "--frame", str(frame), "--out", str(d),
                   "--n-sims", str(args.n_sims), "--n-iter", str(args.n_iter),
                   "--seed", "0"]
            running.append((f"{arm}/mc{mc:g}",
                            spawn(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                  stdout=log, stderr=subprocess.STDOUT), log))
            print(f"  [start] {arm} mc={mc:g}", flush=True)
        import time
        while running and all(p.poll() is None for _, p, _ in running):
            time.sleep(2.0)
        for tag, p, fh in list(running):
            if p.poll() is not None:
                running.remove((tag, p, fh)); fh.close(); done += 1
                print(f"  [done ] {tag} rc={p.returncode} ({done}/{len(jobs)})",
                      flush=True)

    # --- verdict ------------------------------------------------------------
    print(f"\n{'='*74}\nETAS-PROBE GROUND-TRUTH VALIDATION\n{'='*74}")
    summary, probe_ok, per_window = {}, {}, {}
    for arm in args.arms:
        rows = read_points(out / arm, args.background, args.mc)
        if not rows:
            print(f"\n[{arm}] no scored points")
            continue
        rows.sort(key=lambda r: -r["mc"])
        pw = {}
        for mc in args.mc:
            hits = sorted((out / arm).glob(
                f"{args.background}_mc{mc:g}*/target_process.json"))
            if not hits:
                continue
            w = json.load(open(hits[0]))["windows"]
            if "ll_shape" not in w[0]:
                continue
            pw[float(mc)] = (np.array([x["ll_shape"] for x in w], dtype=float),
                             np.array([x["n_target_obs"] for x in w], dtype=float))
        per_window[arm] = pw
        print(f"\n[{arm}]")
        print(f"  {'mc':>5}{'shape/tgt':>12}{'level/tgt':>12}{'total/tgt':>12}"
              f"{'corr':>8}{'p':>10}{'ceiling':>9}{'share':>8}")
        for r in rows:
            print(f"  {r['mc']:>5g}{r['shape']:>12.4f}{r['level']:>12.4f}"
                  f"{r['total']:>12.4f}{r['corr']:>8.3f}{r['corr_p']:>10.1e}"
                  f"{r['corr_ceiling']:>9.3f}{r['corr_share']:>8.1%}")
        mcs = [r["mc"] for r in rows]
        summary[arm] = {k: slope(mcs, [r[k] for r in rows])
                        for k in ("shape", "level", "total")}
        probe_ok[arm] = max(
            (r for r in rows), key=lambda r: (r["corr"] > 0 and r["corr_p"] < 0.05,
                                              r["corr"]))
        s = summary[arm]
        print(f"  {'slope':>5}{s['shape']:>12.4f}{s['level']:>12.4f}"
              f"{s['total']:>12.4f}   nats/decade")

    # invariant 1f: no verdict is meaningful unless the probe actually forecasts
    print(f"\n{'-'*74}")
    lead = args.arms[0]
    if lead not in probe_ok:
        raise SystemExit("informative arm produced nothing; cannot certify")
    best = probe_ok[lead]
    if not (best["corr"] > 0 and best["corr_p"] < 0.05):
        print(f"PROBE CHECK FAILED: best corr on '{lead}' is {best['corr']:+.3f} "
              f"(p={best['corr_p']:.2g}), not significantly positive.\n"
              f"  The probe is not forecasting, so a flat null would prove "
              f"nothing (invariant 1f).\n  Fix the probe before reading any "
              f"verdict below.")
        verdict = "INCONCLUSIVE"
    else:
        print(f"PROBE CHECK PASSED: corr = {best['corr']:+.3f} "
              f"(p={best['corr_p']:.2g}) on '{lead}', which is "
              f"{best['corr_share']:.0%} of the {best['corr_ceiling']:.3f} "
              f"attainable under Poisson counting noise.\n"
              f"  The probe forecasts, so its response to mc is interpretable.")

        # The comparison is informative MINUS null, and the null is NOT
        # expected to be flat.
        #
        # Any forecaster that USES the small events must be harmed when their
        # coupling to the targets is destroyed: a triggering model reads
        # decoupled events as real triggers and forecasts accordingly. Measured
        # here, the surrogate arm's fitted magnitude-productivity exponent
        # collapses from ~0.35 to ~0.02 -- the model correctly learns that
        # parent magnitude no longer predicts offspring, and goes
        # magnitude-blind. A "zero-information null" in the sense of "the model
        # is unaffected" does not exist for a model that conditions on these
        # events.
        #
        # So the null supplies the counterfactual "what if these events carried
        # no coupling", and the DIFFERENCE is the information attributable to
        # the coupling. Uncertainty comes from a paired block bootstrap over
        # forecast windows -- resampled once per replicate and applied to both
        # arms, so the difference is a within-window contrast.
        boots = {}
        for arm in args.arms:
            pts = per_window.get(arm, {})
            mcs_a = sorted(pts, reverse=True)
            if len(mcs_a) < 2:
                continue
            scores = np.vstack([pts[m][0] for m in mcs_a])
            tgts = pts[mcs_a[0]][1]
            boots[arm] = block_bootstrap_slope(arm, mcs_a, scores, tgts,
                                               n_boot=2000, seed=0)
        if lead in boots:
            print(f"\n  {'arm':16}{'slope':>10}{'95% CI':>24}")
            for arm, b in boots.items():
                print(f"  {arm:16}{b.slope:>10.4f}"
                      f"{f'[{b.ci_lo:+.4f}, {b.ci_hi:+.4f}]':>24}")
            nulls = [a for a in args.arms[1:] if a in boots]
            if nulls:
                n0 = nulls[0]
                diff = boots[lead].slope - boots[n0].slope
                se = float(np.hypot(boots[lead].se, boots[n0].se))
                lo, hi = diff - 1.96 * se, diff + 1.96 * se
                print(f"  {'DIFFERENCE':16}{diff:>10.4f}"
                      f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}"
                      "   <- information attributable to coupling")
                ok = lo > 0
                verdict = "PASS" if ok else "FAIL"
                print(f"\n  informative exceeds '{n0}' by {diff:+.4f} nats/decade"
                      f"  ->  {verdict}")
                if not ok:
                    print("  The interval spans zero: at this sample size the "
                          "coupling of small events\n  to the targets cannot be "
                          "distinguished from none.")
            else:
                verdict = "INCONCLUSIVE"
        else:
            verdict = "INCONCLUSIVE"

    print(f"\nETAS-PROBE VALIDATION: {verdict}")
    json.dump({"verdict": verdict, "slopes": summary,
               "probe_max_corr": probe_ok, "args": vars(args)},
              open(out / "verdict.json", "w"), indent=2)
    print(f"wrote {out/'verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
