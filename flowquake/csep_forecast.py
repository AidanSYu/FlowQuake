"""CSEP catalog-based forecasts for FlowQuake + the consistency tests.

Produces pyCSEP-compatible artifacts for ComCat_25 and runs the N/S/M tests
that are the EarthquakeNPP benchmark's headline evaluation:
  - N-test (number): is the forecast event count consistent with observed?
  - S-test (spatial): are observed locations consistent with the forecast's
    spatial rate density?
  - M-test (magnitude): is the observed FMD consistent with the forecast?

For each forecast day we simulate n_sims independent 1-day continuations
(flowquake.ntest.simulate_day_events), write them as one csep_ascii forecast
CSV (catalog_id = simulation index), write the observed test catalog once,
and call pyCSEP. Matches reference/Experiments/ETAS/pycsep_tests_parallel.py.

Usage:
    python -m flowquake.csep_forecast runs/final_s1555/ckpt_best.pt \
        --n-days 30 --n-sims 10000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import csep
from csep.core import regions
from csep.utils import time_utils

from .config import Config
from .data import load_catalog
from .ntest import simulate_day_events
from .train import make_model

MAX_MW = 7.65
DMW = 0.1
EPOCH = "%Y-%m-%dT%H:%M:%S.%f"


def fit_xy_to_lonlat(catalog_csv: str):
    """Cubic-polynomial inverse of the catalog's km projection (x,y)->(lon,lat).

    Fit on the full catalog (a fixed geometric transform, not seismicity, so
    no train/test leakage). Max residual ~0.36 km, far below the ~11 km CSEP
    cell — negligible for gridded spatial tests.
    """
    df = pd.read_csv(catalog_csv)
    x, y = df.x.to_numpy(), df.y.to_numpy()

    def design(x, y):
        cols = []
        for i in range(4):
            for j in range(4 - i):
                cols.append((x ** i) * (y ** j))
        return np.stack(cols, 1)

    A = design(x, y)
    c_lon, *_ = np.linalg.lstsq(A, df.longitude.to_numpy(), rcond=None)
    c_lat, *_ = np.linalg.lstsq(A, df.latitude.to_numpy(), rcond=None)

    def xy2lonlat(xk, yk):
        B = design(np.asarray(xk, float), np.asarray(yk, float))
        return B @ c_lon, B @ c_lat

    return xy2lonlat


def write_forecast_csv(events, xy2lonlat, day_start: dt.datetime, path: Path) -> int:
    """events: list (len n_sims) of (n_i,4) [t_days_abs, x_km, y_km, mag].
    t_days_abs is days since catalog t0; convert to wall-clock via day_start
    and the within-day fraction. Returns number of non-empty catalogs.
    """
    rows = []
    n_nonempty = 0
    for cid, ev in enumerate(events):
        if len(ev) == 0:
            continue
        n_nonempty += 1
        lon, lat = xy2lonlat(ev[:, 1], ev[:, 2])
        frac = ev[:, 0] - np.floor(ev[:, 0])  # within-day fraction
        for k in range(len(ev)):
            t = day_start + dt.timedelta(days=float(frac[k]))
            rows.append((lon[k], lat[k], float(ev[k, 3]),
                         t.strftime(EPOCH), 0.0, cid, f"{cid}_{k}"))
    with open(path, "w", newline="") as f:
        f.write("lon,lat,magnitude,time_string,depth,catalog_id,event_id\n")
        for r in rows:
            f.write(f"{r[0]:.5f},{r[1]:.5f},{r[2]:.3f},{r[3]},{r[4]},{r[5]},{r[6]}\n")
    return n_nonempty


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n-days", type=int, default=30)
    ap.add_argument("--days", type=int, nargs="+", default=None)
    ap.add_argument("--n-sims", type=int, default=10000)
    ap.add_argument("--sample-steps", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["cfg"]
    device = torch.device(args.device)
    cat = load_catalog(
        cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
        cfg.data.train_start, cfg.data.val_start, cfg.data.test_start, cfg.data.test_end,
    )
    model = make_model(cfg, ckpt["stats"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    model.stats.setdefault("mcut", cfg.data.mcut)

    xy2lonlat = fit_xy_to_lonlat(cfg.data.catalog_path)
    region = regions.california_relm_region()
    mag_bins = regions.magnitude_bins(cfg.data.mcut, MAX_MW, DMW)
    smr = regions.create_space_magnitude_region(region, mag_bins)

    t0 = cat.times[0].to_pydatetime()
    test_start_days = float(
        (pd.Timestamp(cfg.data.test_start) - cat.times[0]).total_seconds() / 86400.0
    )
    n_test_days = int(
        (pd.Timestamp(cfg.data.test_end) - pd.Timestamp(cfg.data.test_start)).days
    )
    days = args.days or list(np.linspace(0, n_test_days - 1, args.n_days, dtype=int))

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "csep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Observed test catalog (csep_ascii), written once.
    obs_path = out_dir / "observed.csv"
    dfc = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"])
    dfc = dfc[(dfc.magnitude >= cfg.data.mcut) &
              (dfc.time >= cfg.data.test_start) & (dfc.time < cfg.data.test_end)]
    with open(obs_path, "w", newline="") as f:
        f.write("lon,lat,magnitude,time_string,depth,catalog_id,event_id\n")
        for r in dfc.itertuples():
            f.write(f"{r.longitude:.5f},{r.latitude:.5f},{r.magnitude:.3f},"
                    f"{r.time.strftime(EPOCH)},0.0,-1,{r.id}\n")
    observed_all = csep.load_catalog(str(obs_path))

    from csep.core.catalog_evaluations import number_test, spatial_test, magnitude_test

    results = []
    for day in days:
        start_days = test_start_days + int(day)
        events = simulate_day_events(model, cat, start_days, args.n_sims, device,
                                     args.sample_steps)
        day_start = t0 + dt.timedelta(days=float(start_days))
        fc_path = out_dir / f"CSEP_day_{int(day)}_.csv"
        n_nonempty = write_forecast_csv(events, xy2lonlat, day_start, fc_path)
        sim_counts = np.array([len(e) for e in events])
        n_obs = int(((cat.t_days >= start_days) & (cat.t_days < start_days + 1.0)).sum())

        fc_start = time_utils.strptime_to_utc_datetime(
            day_start.strftime("%Y-%m-%d %H:%M:%S"))
        fc_end = fc_start + dt.timedelta(days=1)
        rec = {"day": int(day), "n_obs": n_obs, "n_nonempty": int(n_nonempty),
               "sim_mean": float(sim_counts.mean())}

        if n_nonempty == 0:
            rec["skipped"] = "no simulated events"
            results.append(rec)
            print(json.dumps(rec)); continue

        forecast = csep.load_catalog_forecast(
            str(fc_path), start_time=fc_start, end_time=fc_end,
            region=smr, filter_spatial=True, apply_filters=True, n_cat=args.n_sims,
        )
        forecast.filters = [
            f"origin_time >= {forecast.start_epoch}",
            f"origin_time < {forecast.end_epoch}",
            f"magnitude >= {forecast.min_magnitude}",
        ]
        forecast.get_expected_rates(verbose=False)

        obs = observed_all.filter_spatial(forecast.region)
        obs = obs.filter(f"magnitude >= {cfg.data.mcut}")
        obs = obs.filter(forecast.filters)

        rec["n_obs_csep"] = int(obs.event_count)

        def _q(x):  # pyCSEP returns scalar or (lower, upper); make JSON-safe
            if isinstance(x, (list, tuple, np.ndarray)):
                return [float(v) for v in x]
            return float(x)

        try:
            nt = number_test(forecast, obs, verbose=False)
            rec["N"] = {"quantile": _q(nt.quantile)}
        except Exception as e:
            rec["N_err"] = str(e)
        if obs.event_count > 0:
            try:
                st = spatial_test(forecast, obs, verbose=False)
                rec["S"] = {"quantile": _q(st.quantile),
                            "observed": float(st.observed_statistic)}
            except Exception as e:
                rec["S_err"] = str(e)
            try:
                mt = magnitude_test(forecast, obs, verbose=False)
                rec["M"] = {"quantile": _q(mt.quantile),
                            "observed": float(mt.observed_statistic)}
            except Exception as e:
                rec["M_err"] = str(e)
        results.append(rec)
        print(json.dumps(rec))

    summary = csep_summary(results)
    out = {"ckpt": args.ckpt, "n_sims": args.n_sims, "n_days": len(days),
           "summary": summary, "results": results}
    with open(out_dir / "csep_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'csep_results.json'}")


def csep_summary(results):
    """Pass rates at 95% for each test. A day passes if its quantile lies in
    [0.025, 0.975]; for tests reporting both tails (lower, upper), require the
    interval to sit inside that band (consistent with ETAS's reporting)."""
    def passes(q):
        if q is None:
            return None
        if isinstance(q, (list, tuple)):
            return bool(min(q) >= 0.025 and max(q) <= 0.975)
        return bool(0.025 <= q <= 0.975)
    out = {}
    for key in ["N", "S", "M"]:
        flags = [passes(r[key]["quantile"]) for r in results if key in r]
        flags = [f for f in flags if f is not None]
        out[key] = {"n_eval": len(flags), "n_pass": int(sum(flags)),
                    "pass_rate": (sum(flags) / len(flags)) if flags else None}
    return out


if __name__ == "__main__":
    main()
