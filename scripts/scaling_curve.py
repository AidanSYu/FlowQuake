"""The scaling curve — how much forecast skill lives in small earthquakes.

This is the `MOONSHOT.md` experiment. Everything else in the repository is
either a control for it or a different paper.

    "Each decade of magnitude below M_TARGET contributes X +- Y nats of forecast
     skill for M>=M_TARGET earthquakes."

Design (see `MOONSHOT.md` "Design invariants" — violating any of these voids the
result, so they are enforced here rather than left to the operator):

  * The TARGET SET never changes. Every point on the curve is scored on the same
    M>=M_TARGET events in the same test windows. `mc` changes only what the model
    SEES. Enforced by building windows/observations once, from the catalog at
    `--m-target`, and reusing them for every arm/mc/seed.
  * The METRIC is the M>=M_TARGET sub-process likelihood, never per-event `tll`.
    See `flowquake.target_process` for why. Enforced by construction — this
    script never reads `eval_test.json`.
  * The GRID is built once and persisted. Deriving it per-mc would leak a second
    mc dependence into the metric.
  * BOTH ARMS are run:
      matched_window  fixed calendar training window, N varies with mc.
                      The operationally real question. The headline.
      matched_n       fixed training-event count, window extended backwards as
                      mc rises. Proves the headline is not just sample size.
    Reporting one without the other is the trap `runs/mw_robustness.json` fell
    into. `--arms` defaults to both; running one prints a loud warning.

Gate G3 (the honest kill switch) is `--pilot`: three mc points, one region, one
seed. If the slope is flat and tight there, stop — and publish the null.

Usage
-----
    # G3 pilot, CPU-checkable first
    python scripts/scaling_curve.py --base configs/n1_density.yaml \
        --out runs/scaling_pilot --pilot --steps 200 --n-sims 20 --device cpu

    # the real sweep
    python scripts/scaling_curve.py --base configs/n1_density.yaml \
        --out runs/scaling_curve/california \
        --mc 4.0 3.5 3.0 2.5 2.0 1.5 1.0 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.config import Config
from flowquake.proc import spawn
from flowquake.target_process import (
    Grid, TargetSpec, aggregate, aki_utsu_b, score_window)

PY = sys.executable
ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."}
ARMS = ("matched_window", "matched_n")


def free_gb() -> float:
    """Free + inactive physical memory, GB. Best-effort and cross-platform."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            pg = 16384
            free = inact = 0
            for line in out.splitlines():
                if "Pages free" in line:
                    free = int(line.split()[2].rstrip("."))
                elif "Pages inactive" in line:
                    inact = int(line.split()[2].rstrip("."))
            return (free + inact) * pg / (1024 ** 3)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:                                        # noqa: BLE001
        pass
    return float("inf")


def wait_for_memory(need_gb: float, tag: str, timeout_s: float = 600.0) -> None:
    """Block until `need_gb` looks available, so the pool cannot oversubscribe.

    Added after a run at concurrency 6 with ~66k simulator lanes per worker took
    a 48 GB machine to 0 GB free and 34 GB of swap. The simulator leak that
    caused it is fixed, but a pool that never checks is a standing hazard: each
    worker holds the full catalog tensors (lastk alone is ~400 MB at mc 1.5, and
    ~2.5 GB at template-catalog scale).
    """
    t0 = time.time()
    while free_gb() < need_gb and time.time() - t0 < timeout_s:
        print(f"  [mem] holding {tag}: {free_gb():.1f} GB free, need {need_gb:.1f}",
              flush=True)
        time.sleep(15.0)


# --------------------------------------------------------------------------
# fixed experimental frame: windows, observations, grid — built ONCE
# --------------------------------------------------------------------------

def build_frame(cfg: Config, spec: TargetSpec, bin_km: float, out: Path,
                b_mc: float) -> dict:
    """Target events, forecast windows and grid. Identical for every curve point.

    Persisted to `frame.json` and reloaded on resume, so a later run cannot
    silently re-derive a different frame.
    """
    frame_path = out / "frame.json"
    if frame_path.exists():
        fr = json.load(open(frame_path))
        fr["grid"] = Grid(**fr["grid"])
        act = np.zeros(fr["grid"].n_cells, dtype=bool)
        act[np.array(fr["active_cells"], dtype=np.int64)] = True
        fr["active"] = act
        return fr

    df = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"]).sort_values("time")
    t0 = df["time"].iloc[0]
    df["t_days"] = (df["time"] - t0).dt.total_seconds() / 86400.0

    test_start = (pd.Timestamp(cfg.data.test_start) - t0).total_seconds() / 86400.0
    test_end = (pd.Timestamp(cfg.data.test_end) - t0).total_seconds() / 86400.0

    # Grid from the FULL catalog footprint (mc-independent by construction:
    # the region is a property of the network, not of the threshold we choose).
    grid = Grid.from_bounds(df["x"].to_numpy(), df["y"].to_numpy(), bin_km=bin_km)

    # Testing region: cells with >=1 event in the TRAINING era at ALL magnitudes.
    # Causal (no test-window information) and mc-independent (network footprint,
    # not our threshold). Without this the Poisson likelihood measures the
    # water-level floor rather than skill — see Grid.active_mask.
    tr = df[df["time"] < pd.Timestamp(cfg.data.val_start)]
    active = grid.active_mask(tr["x"].to_numpy(), tr["y"].to_numpy())

    # Non-overlapping horizon-length windows tiling the test period.
    n_win = int((test_end - test_start) // spec.horizon_days)
    starts = [test_start + i * spec.horizon_days for i in range(n_win)]

    tgt = df[df["magnitude"] >= spec.m_target]
    windows = []
    for s in starts:
        e = s + spec.horizon_days
        w = tgt[(tgt["t_days"] >= s) & (tgt["t_days"] < e)]
        windows.append({
            "start_days": float(s),
            "obs": np.column_stack([
                w["t_days"].to_numpy(), w["x"].to_numpy(),
                w["y"].to_numpy(), w["magnitude"].to_numpy(),
            ]).tolist(),
        })

    # Gutenberg-Richter b for the fixed magnitude tail (invariant 1c). Estimated
    # ONCE, from the training era, at the most complete threshold in the sweep,
    # and then held constant for every point. Re-estimating per mc would put a
    # second mc-dependent quantity back into the metric.
    b = aki_utsu_b(df[df["time"] < pd.Timestamp(cfg.data.val_start)]["magnitude"].to_numpy(),
                   mc=b_mc)
    spec = replace(spec, b_value=b)

    fr = {
        "catalog_path": cfg.data.catalog_path,
        # The absolute time `start_days` is measured FROM. Without it a frame is
        # not portable: any consumer that recomputes t_days from its own catalog
        # gets a different origin, and every window then refers to a different
        # absolute instant. That is not hypothetical -- the surrogate null shifts
        # the earliest small event, moving its catalog's first-event time by
        # 1.7 hours, so the null arm was being forecast 7% of a 1-day window out
        # of step with the informative arm it is compared against.
        "t0": t0.isoformat(),
        "spec": asdict(spec),
        "b_value": b,
        "b_mc": b_mc,
        "grid": asdict(grid),
        "active_cells": np.flatnonzero(active).tolist(),
        "n_active_cells": int(active.sum()),
        "n_windows": len(windows),
        "n_target_events": int(sum(len(w["obs"]) for w in windows)),
        "n_large_windows": int(sum(
            1 for w in windows
            if any(o[3] >= spec.m_large for o in w["obs"])
        )),
        "windows": windows,
    }
    json.dump(fr, open(frame_path, "w"))
    fr["grid"] = grid
    fr["active"] = active
    return fr


def solve_train_start(df: pd.DataFrame, mc: float, val_start: str, n_target: int) -> str | None:
    """Earliest train_start giving ~n_target events >= mc in [start, val_start).

    Matched-N arm: as mc rises the window must extend backwards to hold the
    training-event count fixed. Returns None if the catalog cannot supply
    n_target events at this mc — the caller must record that as a real limit of
    the curve, not silently substitute a smaller N.
    """
    sub = df[(df["magnitude"] >= mc) & (df["time"] < pd.Timestamp(val_start))]
    if len(sub) < n_target:
        return None
    # take the last n_target events; train_start is just before the earliest
    cut = sub["time"].iloc[len(sub) - n_target]
    return (cut - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# one curve point
# --------------------------------------------------------------------------

def prepare_point(base_cfg: str, mc: float, arm: str, seed: int,
                  train_start: str | None, out: Path) -> tuple[str, Path, str] | None:
    """Materialise one point's config. Returns (tag, ckpt, cfg_path) or None."""
    tag = f"{arm}_mc{mc:g}_s{seed}"
    run_dir = out / tag
    ckpt = run_dir / "ckpt_best.pt"
    if train_start is None:
        print(f"[skip] {tag}: catalog cannot supply the matched N at mc={mc}", flush=True)
        return None
    cfg = Config.load(base_cfg)
    cfg.data.mcut = float(mc)
    cfg.data.train_start = train_start
    cfg.data.aux_start = min(train_start, cfg.data.aux_start)
    cfg.train.seed = 1000 + seed
    cfg.train.out_dir = str(run_dir)
    cfg_dir = out / "_cfg"; cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = str(cfg_dir / f"{tag}.yaml")
    cfg.dump(cfg_path)
    return tag, ckpt, cfg_path


def train_many(points: list[tuple[str, Path, str]], steps: int | None, device: str,
               concurrency: int, mem_per_worker_gb: float = 3.0) -> None:
    """Train points in parallel, one THREAD per process.

    Measured on an 18-core M5 Pro at production model size (0.03M params,
    window 2048, batch 8), single-threaded processes:

        concurrency   aggregate steps/s   speedup
              1              2.89           1.00x
              2              4.90           1.69x
              4              8.75           3.02x
              6             10.87           3.75x   <- knee
              8             10.88           3.76x

    The knee is the performance-core count; beyond it work lands on efficiency
    cores and wall clock grows with no throughput gain. Intra-op threading is
    strictly worse than process parallelism here (12 threads on ONE run buys
    2.24x, six 1-thread runs buy 3.75x) because at this model size the step is
    dominated by per-op dispatch, not arithmetic. Hence OMP/MKL/VECLIB are all
    pinned to 1 below — leaving them unset makes every process fight for the
    same cores and measured 0.3x at 16-way.

    On CUDA this does not apply: use concurrency to fill the card instead.
    """
    todo = [(t, c, p) for (t, c, p) in points if not c.exists()]
    done = len(points) - len(todo)
    if done:
        print(f"[train] {done} point(s) already have checkpoints, skipping", flush=True)
    if not todo:
        return
    env = {**ENV, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    print(f"[train] {len(todo)} point(s), concurrency={concurrency}, 1 thread each",
          flush=True)
    t_start = time.time()
    running: list[tuple[str, subprocess.Popen, object]] = []
    queue, finished = list(todo), 0
    while queue or running:
        while queue and len(running) < concurrency:
            tag, ckpt, cfg_path = queue.pop(0)
            wait_for_memory(mem_per_worker_gb, tag)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            fh = open(ckpt.parent / "train.log", "w")
            cmd = [PY, "-m", "flowquake.train", cfg_path, "--out",
                   str(ckpt.parent), "--device", device]
            if steps:
                cmd += ["--steps", str(steps)]
            running.append((tag, spawn(
                cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
            print(f"  [start] {tag}", flush=True)
        # block until at least one slot frees; poll cheaply, never spin
        while running and all(p.poll() is None for _, p, _ in running):
            time.sleep(1.0)
        for tag, proc, fh in list(running):
            if proc.poll() is not None:
                running.remove((tag, proc, fh))
                fh.close()
                finished += 1
                el = time.time() - t_start
                eta = el / finished * (len(todo) - finished)
                print(f"  [done ] {tag} rc={proc.returncode} "
                      f"({finished}/{len(todo)}, {el/60:.1f}min elapsed, "
                      f"~{eta/60:.0f}min left)", flush=True)


#: Total simulated events per window the rate field must be built from.
#: Below this the shape score carries a large negative Monte-Carlo bias; see
#: `sims_for_matched_resolution`. Measured residual bias against the exact
#: density, in nats per target event:
#:
#:      T =     93   234   734  1319  5000  20000  100000
#:   bias =  -4.98 -3.02 -1.04 -0.44 -0.10  -0.03   -0.003
#:
#: 20,000 keeps it under 0.03 nats, an order of magnitude below any slope worth
#: reporting. Cost is flat in T, so this is the same total simulation work at
#: every point on the curve.
TARGET_SIM_EVENTS = 20_000


def sims_for_matched_resolution(cfg, spec: TargetSpec, cat,
                                target_events: int = TARGET_SIM_EVENTS,
                                n_min: int = 50, n_max: int = 200_000) -> int:
    """Simulations needed so the rate field is estimated from the SAME total
    number of simulated events, `target_events`, at every mc.

    THIS REPLACES an earlier version that equalised the ABSOLUTE variance of
    lambda. That was the wrong invariant and it left the dominant bias in place.

    The score is a Poisson log-likelihood, so what biases it is E[log lambda]
    against log E[lambda] -- Jensen -- and that gap is set by the RELATIVE
    variance. Writing T for the total simulated events behind the field and p_c
    for a cell's share of them, the per-cell count is ~Poisson(T p_c) and

        lambda_c = w * C_c / n_sims,  E[lambda_c] = w T p_c / n_sims
        Var(lambda_c) = w^2 T p_c / n_sims^2
        Var / E^2     = 1 / (T p_c)

    The tail weight w and the simulation count n_sims both cancel. Resolution
    depends on T ALONE, and T = n_sims * (events per simulation) grows directly
    with the catalog rate above mc -- so it varied 14-fold across a single grid
    and dragged the whole curve with it.

    Measured on the real WHITE catalog at a 30-day horizon:

        mc          2.5     2.0     1.5     1.0
        T            93     234     734    1319
        shape   -11.669 -10.822  -9.133  -7.204

    That is +3.02 nats/decade on the informative arm and +2.62 on a SURROGATE
    NULL whose true slope is zero by construction. A controlled experiment --
    same observations, same true density, only T varied -- reproduced a +4.54
    nat shift across the same T range against a measured +4.53. The entire
    slope was Monte-Carlo resolution.

    The right quantity is T per EFFECTIVE cell, and this is why the floor has
    to be high rather than merely equal. Per-cell relative variance is
    1/(T*p_c), so a sharper forecast concentrates the same T into fewer cells
    and is resolved less well. Measured at MATCHED T, the residual
    concentration-dependence of the bias:

        T          2,000    20,000    200,000
        diffuse   -0.521    -0.026     -0.001     (325 effective cells)
        sharp     -0.275    -0.021     -0.002     (104)
        very sharp -0.079    -0.003     -0.001     ( 29)

    At T = 2,000 the spread across concentrations is 0.44 nats, larger than the
    signal being measured; at T = 20,000 it is 0.023, an order of magnitude
    below it. So matching T is necessary but not sufficient -- the budget must
    also be big enough that the residual spread is negligible. `score_window`
    reports `n_eff_cells` per window so this can be checked rather than
    assumed.

    The requirement is a FLOOR on T, not exact equality. The bias curve
    flattens: -0.03 nats at T = 20,000 and -0.003 at T = 100,000, so once every
    point is on the flat part the residual differences are two orders of
    magnitude below any slope worth reporting. That matters because equality is
    not always attainable -- at a high event rate a SINGLE simulation can
    already exceed the target (at mc 1.0 on the California catalog one 90-day
    window yields ~3,700 events), and `n_sims` cannot go below 1. Overshooting
    is harmless; undershooting is not.

    Cost stays roughly flat across the curve rather than exploding at low mc:
    high-mc points run many cheap simulations, low-mc points few expensive ones.
    """
    horizon = spec.horizon_days
    span = float(cat.t_days[-1] - cat.t_days[0]) or 1.0
    n_above = int((cat.raw[:, 3].numpy() >= float(cfg.data.mcut)).sum())
    per_sim = max(n_above / span * horizon, 1e-9)   # events per simulated window
    import math
    return int(min(max(math.ceil(target_events / per_sim), n_min), n_max))


def score_point(ckpt: Path, frame: dict, spec: TargetSpec, n_sims: int,
                device: str, sample_steps: int, force: bool = False,
                match_precision_ref: float | None = None) -> dict | None:
    """Score a trained model on the fixed target process. Resumable."""
    import torch
    from flowquake.ntest import simulate_day_events, simulate_windows
    from flowquake.train import load_catalog_cfg, make_model

    score_path = ckpt.parent / "target_process.json"
    if score_path.exists() and not force:
        return json.load(open(score_path))

    dev = torch.device(device)
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg: Config = ck["cfg"]
    cat = load_catalog_cfg(cfg)
    model = make_model(cfg, ck["stats"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    model.stats.setdefault("mcut", cfg.data.mcut)

    # The spec's mc is the completeness of the catalog being scored, so the
    # fixed GR tail P(m>=m_target | m>=mc) is computed correctly per point.
    # b_value stays frozen at the frame value (invariant 1c).
    spec = replace(spec, mc=float(cfg.data.mcut))
    if match_precision_ref is not None:
        n_sims = sims_for_matched_resolution(cfg, spec, cat)
        print(f"    matched-resolution n_sims = {n_sims} at mc={cfg.data.mcut} "
              f"(T={TARGET_SIM_EVENTS:,} simulated events/window)", flush=True)
    grid: Grid = frame["grid"]
    starts = [float(w["start_days"]) for w in frame["windows"]]

    # Spend the simulation budget where it can change the answer.
    #
    # The SHAPE term is sum_c n_c log(lambda_c / Lambda). A window with no
    # observed targets contributes EXACTLY zero to it, whatever the forecast --
    # verified on real data at 0.0000000000. At a 1-day horizon 94% of windows
    # are empty, so simulating them at full resolution buys nothing.
    #
    # Those windows do contribute -Lambda to the LEVEL term, but Lambda-hat is
    # an unbiased mean at any n_sims (the N*log(Lambda) piece vanishes when
    # N = 0, so there is no Jensen term to worry about) -- only its variance
    # rises, and that averages over the 1,573 empty windows.
    #
    # So: full matched-resolution n_sims on windows that carry targets, a small
    # count on the rest. Exact for shape, unbiased for level, and 16.7x cheaper
    # at a 1-day horizon. This allocation cannot bias the primary metric,
    # because the windows it economises on contribute zero to it by identity.
    has_tgt = [len(np.array(w["obs"], dtype=np.float64).reshape(-1, 4)) > 0
               for w in frame["windows"]]
    n_lo = max(32, n_sims // 64)

    def _sim(idx_starts, ns):
        if not idx_starts:
            return {}
        ss = [starts[i] for i in idx_starts]
        if model.encoder is None:
            out = simulate_windows(model, cat, ss, ns, dev,
                                   sample_steps=sample_steps,
                                   horizon_days=spec.horizon_days)
        else:   # h>0 carries per-lane SSM state; fall back to the per-window path
            out = [simulate_day_events(model, cat, s, ns, dev,
                                       sample_steps=sample_steps,
                                       horizon_days=spec.horizon_days)
                   for s in ss]
        return dict(zip(idx_starts, out))

    hi_idx = [i for i, h in enumerate(has_tgt) if h]
    lo_idx = [i for i, h in enumerate(has_tgt) if not h]
    print(f"    {len(hi_idx)} windows with targets at n_sims={n_sims:,}; "
          f"{len(lo_idx)} empty at n_sims={n_lo:,} "
          f"(empty windows contribute 0 to the shape term by identity)",
          flush=True)
    all_sims = {**_sim(hi_idx, n_sims), **_sim(lo_idx, n_lo)}

    windows = []
    for i, w in enumerate(frame["windows"]):
        obs = np.array(w["obs"], dtype=np.float64).reshape(-1, 4)
        windows.append(score_window(all_sims[i], obs, grid, spec,
                                    active=frame["active"]))

    res = {"mcut": cfg.data.mcut, "train_start": cfg.data.train_start,
           "n_sims": n_sims, "tail_mode": spec.tail_mode, "b_value": spec.b_value,
           "tail_prob_target": spec.tail_prob(spec.m_target),
           "aggregate": aggregate(windows), "windows": windows}
    json.dump(res, open(score_path, "w"), indent=2)
    return res


# --------------------------------------------------------------------------

def score_many(ckpts: list[Path], out: Path, n_sims: int, sample_steps: int,
               device: str, concurrency: int, mem_per_worker_gb: float = 4.0) -> None:
    """Score points in parallel. Same reasoning as `train_many` — the simulator
    is an autoregressive Python loop over small tensors, so process parallelism
    beats intra-op threading."""
    env = {**ENV, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    print(f"[score] {len(ckpts)} point(s), concurrency={concurrency}", flush=True)
    t0, running, queue, fin = time.time(), [], list(ckpts), 0
    while queue or running:
        while queue and len(running) < concurrency:
            ck = queue.pop(0)
            wait_for_memory(mem_per_worker_gb, ck.parent.name)
            fh = open(ck.parent / "score.log", "w")
            cmd = [PY, __file__, "--score-one", str(ck), "--frame", str(out / "frame.json"),
                   "--n-sims", str(n_sims), "--sample-steps", str(sample_steps),
                   "--device", device]
            running.append((ck.parent.name, spawn(
                cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
            print(f"  [start] {ck.parent.name}", flush=True)
        while running and all(p.poll() is None for _, p, _ in running):
            time.sleep(1.0)
        for tag, proc, fh in list(running):
            if proc.poll() is not None:
                running.remove((tag, proc, fh)); fh.close(); fin += 1
                el = time.time() - t0
                if proc.returncode != 0:
                    print(f"  *** {tag} FAILED rc={proc.returncode}; see "
                          f"{tag}/score.log ***", flush=True)
                print(f"  [done ] {tag} rc={proc.returncode} ({fin}/{len(ckpts)}, "
                      f"~{el/fin*(len(ckpts)-fin)/60:.0f}min left)", flush=True)


def _load_frame(path: Path) -> dict:
    fr = json.load(open(path))
    fr["grid"] = Grid(**fr["grid"])
    act = np.zeros(fr["grid"].n_cells, dtype=bool)
    act[np.array(fr["active_cells"], dtype=np.int64)] = True
    fr["active"] = act
    return fr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-one", default=None,
                    help="internal: score a single checkpoint and exit")
    ap.add_argument("--frame", default=None, help="internal: frame.json for --score-one")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="parallel processes; 6 is the measured knee on an 18-core "
                         "M5 Pro (see train_many). Raise to fill a CUDA card.")
    ap.add_argument("--base", help="base config (region + splits)")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--mc", type=float, nargs="+",
                    default=[4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--m-target", type=float, default=4.0)
    ap.add_argument("--m-large", type=float, default=6.0)
    ap.add_argument("--horizon", type=float, default=30.0)
    ap.add_argument("--bin-km", type=float, default=5.0)
    ap.add_argument("--tail-mode", default="fixed", choices=["fixed", "learned"],
                    help="fixed = PRIMARY result, magnitude head out of the metric "
                         "(MOONSHOT.md invariant 1c). learned = the ablation.")
    ap.add_argument("--mem-per-worker", type=float, default=4.0,
                    help="GB that must look free before starting another "
                         "worker; guards against swapping the machine")
    ap.add_argument("--match-precision", action="store_true", default=True,
                    help="equalise the TOTAL simulated events behind the rate "
                         "field across mc. ON by default: without it the "
                         "estimator carries a +2.6 nats/decade bias on a null "
                         "whose true slope is zero (see "
                         "sims_for_matched_resolution).")
    ap.add_argument("--no-match-precision", dest="match_precision",
                    action="store_false")
    ap.add_argument("--b-mc", type=float, default=None,
                    help="completeness for the one-off b estimate (default: max --mc)")
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--sample-steps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None, help="training steps override")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pilot", action="store_true",
                    help="gate G3: 3 mc points, 1 seed, matched_window only")
    ap.add_argument("--matched-n", type=int, default=None,
                    help="training-event count for the matched_n arm "
                         "(default: the count at the highest mc, i.e. the most "
                         "restrictive point, so every mc can supply it)")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args(argv)

    # worker mode: score one checkpoint against a prebuilt frame, then exit
    if args.score_one:
        fr = _load_frame(Path(args.frame))
        sp = TargetSpec(**fr["spec"])
        score_point(Path(args.score_one), fr, sp, args.n_sims, args.device,
                    args.sample_steps,
                    match_precision_ref=(fr.get("mc_ref") if args.match_precision
                                         else None))
        return

    if not args.base or not args.out:
        raise SystemExit("--base and --out are required")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(args.base)
    spec = TargetSpec(m_target=args.m_target, m_large=args.m_large,
                      horizon_days=args.horizon, tail_mode=args.tail_mode)

    if args.pilot:
        args.mc = [2.5, 1.5, 0.5] if min(args.mc) < 1.0 else [args.mc[0], args.mc[len(args.mc)//2], args.mc[-1]]
        args.seeds = [0]
        args.arms = ["matched_window"]
        print(f"[pilot] gate G3: mc={args.mc}, 1 seed, matched_window only")
    if len(args.arms) == 1 and not args.pilot:
        print("\n*** WARNING: running one arm only. MOONSHOT.md invariant 2 requires\n"
              "*** both matched_window and matched_n. A single-arm result cannot\n"
              "*** separate information content from sample size.\n", flush=True)

    if any(m > args.m_target for m in args.mc):
        raise SystemExit(
            f"mc values {[m for m in args.mc if m > args.m_target]} exceed "
            f"--m-target {args.m_target}. The model would not see the target "
            f"events it is being scored on. Lower mc or raise --m-target."
        )

    frame = build_frame(cfg, spec, args.bin_km, out, b_mc=args.b_mc or max(args.mc))
    # Precision reference is the LOWEST mc: it has the most simulated events
    # each carrying the smallest tail weight, hence the least noisy rate
    # field. Every other point scales its n_sims UP to match it. (Note this
    # is the opposite end of the sweep from b_mc, which uses the MOST
    # COMPLETE threshold for the one-off b estimate.)
    if args.match_precision and frame.get("mc_ref") != min(args.mc):
        frame["mc_ref"] = float(min(args.mc))
        fj = json.load(open(out / "frame.json")); fj["mc_ref"] = frame["mc_ref"]
        json.dump(fj, open(out / "frame.json", "w"))
    spec = replace(spec, b_value=frame["b_value"])
    print(f"[frame] fixed GR tail: b={frame['b_value']:.3f} "
          f"(Aki-Utsu, training era, mc={frame['b_mc']:g}) | tail_mode={spec.tail_mode}",
          flush=True)
    g: Grid = frame["grid"]
    n_act, n_tgt = frame["n_active_cells"], frame["n_target_events"]
    print(f"[frame] {frame['n_windows']} windows x {spec.horizon_days:g}d | "
          f"{n_tgt} M>={spec.m_target} target events | "
          f"{frame['n_large_windows']} windows contain M>={spec.m_large} | "
          f"grid {g.nx}x{g.ny} @ {g.bin_km:g}km -> {n_act} active cells", flush=True)

    # Grid resolution must be calibrated to the target-event count, once, and
    # then held fixed. Too fine and the Poisson likelihood measures the water
    # level; too coarse and there is no spatial skill to detect. CSEP's own
    # convention (0.1 deg over RELM, ~7.7k cells) sits at ~15-40 cells per
    # target event for a decade of M>=4.
    cpe = n_act / max(n_tgt, 1)
    print(f"[frame] {cpe:.1f} active cells per target event", flush=True)
    if cpe > 200:
        print(f"*** WARNING: {cpe:.0f} cells per target event. The Poisson score "
              f"will be dominated by the {spec.floor_frac:g} water level, not by "
              f"forecast skill. Coarsen --bin-km (try "
              f"{args.bin_km * (cpe/50)**0.5:.0f}) or lengthen --horizon.", flush=True)
    elif cpe < 3:
        print(f"*** WARNING: only {cpe:.1f} cells per target event — too coarse "
              f"to resolve spatial skill. Refine --bin-km.", flush=True)
    if frame["n_large_windows"] < 5:
        print(f"*** WARNING: only {frame['n_large_windows']} windows contain an "
              f"M>={spec.m_large} event. P(M>={spec.m_large}) skill (gate G5) "
              f"will be underpowered; report the count, do not hide it.", flush=True)

    df = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"])
    n_matched = args.matched_n
    if n_matched is None and "matched_n" in args.arms:
        hi = max(args.mc)
        n_matched = int(((df["magnitude"] >= hi) &
                         (df["time"] < pd.Timestamp(cfg.data.val_start))).sum())
        print(f"[matched_n] N = {n_matched} (the count at mc={hi}, the most "
              f"restrictive point)", flush=True)

    # ---- plan every point up front, so training can be fanned out ----
    plan, points = [], []
    for arm in args.arms:
        for mc in args.mc:
            for seed in args.seeds:
                train_start = (cfg.data.train_start if arm == "matched_window"
                               else solve_train_start(df, mc, cfg.data.val_start, n_matched))
                n_train = int(((df["magnitude"] >= mc) &
                               (df["time"] >= pd.Timestamp(train_start)) &
                               (df["time"] < pd.Timestamp(cfg.data.val_start))).sum()
                              ) if train_start else 0
                p = prepare_point(args.base, mc, arm, seed, train_start, out)
                plan.append({"arm": arm, "mc": mc, "seed": seed,
                             "train_start": train_start, "n_train": n_train,
                             "ckpt": str(p[1]) if p else None})
                if p:
                    points.append(p)

    if not args.score_only:
        train_many(points, args.steps, args.device, args.concurrency,
                   mem_per_worker_gb=args.mem_per_worker)

    # ---- score (parallel; each point is an independent subprocess) ----
    to_score = [Path(r["ckpt"]) for r in plan
                if r["ckpt"] and Path(r["ckpt"]).exists()
                and not (Path(r["ckpt"]).parent / "target_process.json").exists()]
    if to_score:
        score_many(to_score, out, args.n_sims, args.sample_steps,
                   args.device, args.concurrency,
                   mem_per_worker_gb=args.mem_per_worker)

    rows = []
    for r in plan:
        agg = {}
        if r["ckpt"]:
            sp = Path(r["ckpt"]).parent / "target_process.json"
            if sp.exists():
                agg = json.load(open(sp)).get("aggregate", {})
        rows.append({**{k: r[k] for k in ("arm", "mc", "seed", "train_start", "n_train")},
                     "ll_per_target_event": agg.get("ll_per_target_event"),
                     # shape is the PRIMARY curve metric: invariant to any
                     # rescaling of lambda, so a magnitude-tail error cannot
                     # move it. level is the term that error lands in.
                     "ll_shape_per_target_event": agg.get("ll_shape_per_target_event"),
                     "ll_level_per_target_event": agg.get("ll_level_per_target_event"),
                     "ig_vs_poisson": agg.get("ig_per_target_event"),
                     "brier": agg.get("brier"),
                     "log_score_mean": agg.get("log_score_mean"),
                     "n_target_events": agg.get("n_target_events")})
    # A point that failed to score leaves no target_process.json, and an
    # earlier version simply wrote a None row and fitted the slope on whatever
    # survived. That is the dangerous failure: matched-resolution n_sims grows
    # monotonically with mc, so anything that breaks at large n_sims takes out
    # the HIGH-mc end of the grid first -- truncating the very lever arm a
    # per-decade slope is measured over, in strict mc order, while the run
    # still exits 0. Refuse to emit a curve with holes in it.
    missing = [f"{r['arm']} mc={r['mc']:g} seed={r['seed']}"
               for r in rows if r.get("ll_shape_per_target_event") is None]
    if missing:
        print(f"\n*** {len(missing)} of {len(rows)} points failed to score:")
        for m in missing:
            print(f"      {m}")
        print("    The slope's lever arm depends on WHICH points are present, and\n"
              "    failures here are ordered in mc, so a partial curve is biased\n"
              "    rather than merely noisy. Fix the failures or drop the panel.")
    json.dump(rows, open(out / "curve.json", "w"), indent=2)

    print(f"\n{'arm':16}{'mc':>5}{'N_train':>9}{'n_tgt':>7}"
          f"{'shape/tgt':>12}{'level/tgt':>12}{'total/tgt':>12}{'brier':>9}")
    for r in rows:
        fmt = lambda v: "None" if v is None else f"{v:.4f}"
        print(f"{r['arm']:16}{r['mc']:>5g}{r['n_train']:>9}"
              f"{r['n_target_events'] or 0:>7}"
              f"{fmt(r['ll_shape_per_target_event']):>12}"
              f"{fmt(r['ll_level_per_target_event']):>12}"
              f"{fmt(r['ll_per_target_event']):>12}{fmt(r['brier']):>9}")
    print(f"\nwrote {out/'curve.json'}  ({len(rows)} points)")
    print("\nNEXT: score ETAS on the SAME frame (scripts/etas_by_mc.py) — a curve "
          "without the re-inverted ETAS control is not the experiment (invariant 3).")


if __name__ == "__main__":
    main()
