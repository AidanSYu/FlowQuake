"""Generate ETAS catalog-based forecasts in the SAME pyCSEP path as FlowQuake.

For a same-days head-to-head: ETAS (the EarthquakeNPP incumbent) is run through
the identical N/S/M test harness FlowQuake uses (flowquake.csep_forecast). We
reuse the benchmark's own fitted ETAS inversion (output_data_<cat>/parameters_0.json,
Mizrahi et al. `etas` package) and, for each forecast day d, condition on the
observed catalog up to test_start+d, simulate n_sims one-day continuations, and
write them as a csep_ascii forecast CSV (CSEP_day_{d}_.csv) — byte-compatible
with flowquake.csep_forecast's --rerun scorer.

ETAS's timewindow_end equals FlowQuake's test_start (verified 2007-01-01 for
ComCat_25), so day offset d denotes the identical wall-clock forecast window in
both pipelines.

Usage (CPU only — no GPU):
    python -m flowquake.etas_csep --etas-dir reference/Experiments/ETAS/output_data_ComCat_25 \
        --days-from runs/n1_density/csep/csep_results.json \
        --n-sims 10000 --out runs/etas_csep
Then score with the shared harness:
    python -m flowquake.csep_forecast runs/n1_density/ckpt_best.pt --rerun \
        --out runs/etas_csep --days <same days>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os

# Keep each worker single-threaded so N parallel day-workers don't oversubscribe
# the BLAS thread pool (set before numpy import).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = "%Y-%m-%dT%H:%M:%S.%f"


def _load_days(spec: str):
    """Day offsets: a csep_results.json (reuse its `day` list) or 'a,b,c'."""
    if spec.endswith(".json"):
        r = json.load(open(spec))
        return [int(d["day"]) for d in r["results"]]
    return [int(x) for x in spec.split(",")]


def simulate_day(inv_template: dict, day: int, n_sims: int, m_threshold: float,
                 out_path: Path, chunksize: int = 100) -> tuple[int, int]:
    """STREAM n_sims one-day ETAS continuations for offset `day` straight to a
    csep_ascii CSV, holding only one chunk in memory at a time. Returns
    (n_events, n_nonempty_catalogs).

    Reloads the inversion with timewindow_end advanced by `day` so triggering
    conditions on observed history up to the forecast start (Mizrahi et al. 2021);
    fitted params unchanged. Memory note: the previous version accumulated every
    chunk and concatenated (peak ~ all n_sims events, ~tens of GB at n_sims=1e4,
    which OOM'd the box). simulate() already yields chunks lazily and frees each,
    so writing each chunk and discarding it bounds peak memory to ~chunksize
    catalogs regardless of n_sims.
    """
    from etas.inversion import ETASParameterCalculation
    from etas.simulation import ETASSimulation

    inv = dict(inv_template)
    base_end = dt.datetime.strptime(inv_template["timewindow_end"], "%Y-%m-%d %H:%M:%S")
    inv["timewindow_end"] = (base_end + dt.timedelta(days=day)).strftime("%Y-%m-%d %H:%M:%S")
    inv["three_dim"] = False
    inv["space_unit_in_meters"] = False

    reload = ETASParameterCalculation.load_calculation(inv)
    # FIX (trigger conditioning): load_calculation restores the CACHED source
    # events from the original inversion window (<= test_start). A genuine
    # forecast at offset `day` must condition triggering on every observed event
    # up to the advanced timewindow_end. load_calculation already re-read the full
    # fn_catalog and windowed it to [auxiliary_start, timewindow_end+day) via
    # filter_catalog, so recomputing the source set from that catalog restores the
    # post-test_start mainshocks as triggers (the fitted long-term background/
    # target map is left as-is — it is the slow-varying term, unlike triggering).
    reload.source_events = reload.prepare_source_events()
    sim = ETASSimulation(reload)
    sim.prepare()

    gen = sim.simulate(forecast_n_days=1, n_simulations=n_sims,
                       m_threshold=m_threshold, info_cols=[], chunksize=chunksize)
    n_events, cats, row = 0, set(), 0
    tmp = Path(str(out_path) + ".tmp")          # atomic: rename only on success
    with open(tmp, "w", newline="") as f:
        f.write("lon,lat,magnitude,time_string,depth,catalog_id,event_id\n")
        for df in gen:
            if df is None or not len(df):
                continue
            lon = df["longitude"].to_numpy(float)
            lat = df["latitude"].to_numpy(float)
            mag = df["magnitude"].to_numpy(float)
            tstr = pd.to_datetime(df["time"]).dt.strftime(EPOCH).to_numpy()
            cid = df["catalog_id"].to_numpy(int)
            for k in range(len(df)):
                f.write(f"{lon[k]:.5f},{lat[k]:.5f},{mag[k]:.3f},{tstr[k]},0.0,"
                        f"{cid[k]},{cid[k]}_{row}\n")
                row += 1
            cats.update(int(c) for c in cid)
            n_events += len(df)
            del df, lon, lat, mag, tstr, cid
    os.replace(tmp, out_path)
    return n_events, len(cats)


def _worker(task):
    """One forecast day end-to-end (runs in a pool worker). Returns a status str."""
    inv_template, day, n_sims, mc, out = task
    path = Path(out) / f"CSEP_day_{day}_.csv"
    if path.exists():
        return f"day {day}: exists, skip"
    t0 = dt.datetime.now()
    try:
        n_events, n_nonempty = simulate_day(inv_template, day, n_sims, mc, path)
    except Exception as e:  # one bad day shouldn't kill the sweep
        tmp = Path(str(path) + ".tmp")
        if tmp.exists():
            tmp.unlink()
        return f"day {day}: ERROR {type(e).__name__}: {e}"
    dt_s = (dt.datetime.now() - t0).total_seconds()
    return f"day {day}: {n_events} events, {n_nonempty} non-empty cats, {dt_s:.1f}s"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--etas-dir", required=True,
                    help="dir with parameters_0.json (benchmark ETAS inversion)")
    ap.add_argument("--days-from", required=True,
                    help="csep_results.json to copy day offsets from, or 'a,b,c'")
    ap.add_argument("--n-sims", type=int, default=10000)
    ap.add_argument("--mc", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog", default=None,
                    help="override fn_catalog (abs path) if the relative one fails")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel day-workers (CPU). Each is BLAS-pinned to 1 thread.")
    args = ap.parse_args(argv)

    logging.getLogger("etas").setLevel(logging.WARNING)
    etas_dir = Path(args.etas_dir)
    with open(etas_dir / "parameters_0.json") as f:
        inv_template = json.load(f)
    if args.catalog:
        inv_template["fn_catalog"] = os.path.abspath(args.catalog)
    elif not os.path.isabs(inv_template.get("fn_catalog", "")):
        # fn_catalog in the ETAS config is relative to the experiments dir
        # (reference/Experiments/ETAS), where invert/predict run — i.e.
        # etas_dir.parent, NOT etas_dir (which is output_data_<cat>).
        cand = (etas_dir.parent / inv_template["fn_catalog"]).resolve()
        if cand.exists():
            inv_template["fn_catalog"] = str(cand)
    # fn_ip / fn_src (background & source-event tables from the inversion) are
    # stored relative to the ETAS experiments dir (parent of output_data_<cat>).
    for key in ("fn_ip", "fn_src"):
        val = inv_template.get(key)
        if val and not os.path.isabs(val):
            cand = (etas_dir.parent / val).resolve()
            if cand.exists():
                inv_template[key] = str(cand)

    days = _load_days(args.days_from)
    # Absolute: under the 'spawn' start method workers inherit cwd but may run
    # before any chdir; an absolute out path is unambiguous.
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ETAS CSEP: {len(days)} days, {args.n_sims} sims/day, "
          f"{args.workers} workers -> {out_dir}", flush=True)

    tasks = [(inv_template, day, args.n_sims, args.mc, str(out_dir)) for day in days]
    done = 0
    if args.workers <= 1:
        for task in tasks:
            print(f"[{done+1}/{len(days)}] {_worker(task)}", flush=True)
            done += 1
    else:
        # 'spawn' (not Linux's default 'fork'): forking a parent that has
        # imported heavy C-extensions (shapely/pyproj/PROJ, BLAS) and then
        # respawning per task is fork-unsafe and crashes the pool
        # (BrokenProcessPool). spawn re-imports cleanly in each worker.
        # max_tasks_per_child=1: respawn each worker after every day so the
        # per-day working set (ETAS rebuilds an O(N^2) triggering/distance
        # structure over the entire history) is returned to the OS, not ratcheted.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 max_tasks_per_child=1,
                                 mp_context=ctx) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futs):
                done += 1
                print(f"[{done}/{len(days)}] {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
