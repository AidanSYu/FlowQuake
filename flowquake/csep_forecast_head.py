"""CSEP catalog-based forecast with the FULL-HISTORY neural-ETAS spatial head.

The head is spatial-only, so event counts/times/magnitudes come from the same
validated production simulator (flowquake.ntest.simulate_day_events); only event
LOCATIONS are drawn from the head's per-day spatial density on the CSEP grid
(frozen at the forecast-day-start history — the standard 1-day CSEP convention,
identical to how the ETAS gridded forecast is built). We then run the same
pyCSEP N/S/M tests on the same days at a matched simulation budget, so the S-test
is a like-for-like comparison against the production kernel-mixture head and ETAS.

Usage:
    python -m flowquake.csep_forecast_head runs/n1_density/ckpt_best.pt \
        --head ComCat_25 --n-days 100 --n-sims 1000
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
from csep.core.catalog_evaluations import number_test, spatial_test, magnitude_test

from .config import Config
from .csep_forecast import csep_summary, fit_xy_to_lonlat
from .ntest import simulate_day_events
from .neural_etas_forecast import head_features_at, load_head
from .train import load_catalog_cfg, make_model

MAX_MW, DMW, EPOCH = 7.65, 0.1, "%Y-%m-%dT%H:%M:%S.%f"


@torch.no_grad()
def head_grid_logdensity(model, P, hist, t_now, prev_idx, cell_lat, cell_lon,
                         device, chunk=256):
    """log head spatial density at each CSEP cell center (radians), chunked."""
    out = np.empty(len(cell_lat), dtype=np.float64)
    for i0 in range(0, len(cell_lat), chunk):
        sl = slice(i0, i0 + chunk)
        feats = head_features_at(P, hist, t_now, prev_idx, cell_lat[sl], cell_lon[sl], device=device)
        out[sl] = model(**feats).cpu().numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--head", default="ComCat_25", help="neural-ETAS head region name")
    ap.add_argument("--head-seed", type=int, default=0)
    ap.add_argument("--n-days", type=int, default=100)
    ap.add_argument("--days", type=int, nargs="+", default=None)
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--sample-steps", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["cfg"]
    cat = load_catalog_cfg(cfg)
    model = make_model(cfg, ckpt["stats"]).to(device).eval()
    model.load_state_dict(ckpt["model"]); model.stats.setdefault("mcut", cfg.data.mcut)
    xy2lonlat = fit_xy_to_lonlat(cfg.data.catalog_path)

    head, P = load_head(args.head, seed=args.head_seed, device=device)

    region = regions.california_relm_region()
    mag_bins = regions.magnitude_bins(cfg.data.mcut, MAX_MW, DMW)
    smr = regions.create_space_magnitude_region(region, mag_bins)
    mids = region.midpoints()                    # (n_cell, 2) lon,lat
    cell_lon = np.radians(mids[:, 0]); cell_lat = np.radians(mids[:, 1])
    dh = region.dh

    # frozen history in the head's lat/lon frame (catalog is time-sorted)
    dfc_all = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    all_lat = np.radians(dfc_all["latitude"].to_numpy())
    all_lon = np.radians(dfc_all["longitude"].to_numpy())
    all_t = cat.t_days
    all_m = dfc_all["magnitude"].to_numpy()

    t0 = cat.times[0].to_pydatetime()
    test_start_days = float((pd.Timestamp(cfg.data.test_start) - cat.times[0]).total_seconds() / 86400.0)
    n_test_days = int((pd.Timestamp(cfg.data.test_end) - pd.Timestamp(cfg.data.test_start)).days)
    days = args.days or list(np.linspace(0, n_test_days - 1, args.n_days, dtype=int))

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "csep_head"
    out_dir.mkdir(parents=True, exist_ok=True)

    # observed catalog once
    obs_path = out_dir / "observed.csv"
    dfo = dfc_all[(dfc_all.magnitude >= cfg.data.mcut) &
                  (dfc_all.time >= cfg.data.test_start) & (dfc_all.time < cfg.data.test_end)]
    with open(obs_path, "w", newline="") as f:
        f.write("lon,lat,magnitude,time_string,depth,catalog_id,event_id\n")
        for r in dfo.itertuples():
            f.write(f"{r.longitude:.5f},{r.latitude:.5f},{r.magnitude:.3f},"
                    f"{r.time.strftime(EPOCH)},0.0,-1,{r.id}\n")

    results = []
    for day in days:
        start_days = test_start_days + int(day)
        day_start = t0 + dt.timedelta(days=float(start_days))
        fc_path = out_dir / f"CSEP_day_{int(day)}_.csv"
        n_obs = int(((cat.t_days >= start_days) & (cat.t_days < start_days + 1.0)).sum())

        # counts/times/mags from the production simulator (head unchanged there)
        events = simulate_day_events(model, cat, start_days, args.n_sims, device, args.sample_steps)

        # head spatial field for this day (frozen at history before day start)
        n_hist = int(np.searchsorted(all_t, start_days, side="left"))
        hist = dict(lat=all_lat[:n_hist], lon=all_lon[:n_hist], t_days=all_t[:n_hist], mag=all_m[:n_hist])
        logd = head_grid_logdensity(head, P, hist, start_days, n_hist - 1, cell_lat, cell_lon, device)
        p = np.exp(logd - logd.max()); p /= p.sum()

        # write head-located catalogs: keep each sim's times+mags, redraw locations
        rng = np.random.default_rng(1000 + int(day))
        rows = []; n_nonempty = 0
        for cid, ev in enumerate(events):
            if len(ev) == 0:
                continue
            n_nonempty += 1
            cells = rng.choice(len(p), size=len(ev), p=p)
            jit = (rng.random((len(ev), 2)) - 0.5) * dh
            lon = mids[cells, 0] + jit[:, 0]; lat = mids[cells, 1] + jit[:, 1]
            frac = ev[:, 0] - np.floor(ev[:, 0])
            for k in range(len(ev)):
                t = day_start + dt.timedelta(days=float(frac[k]))
                rows.append((lon[k], lat[k], float(ev[k, 3]), t.strftime(EPOCH), 0.0, cid, f"{cid}_{k}"))
        with open(fc_path, "w", newline="") as f:
            f.write("lon,lat,magnitude,time_string,depth,catalog_id,event_id\n")
            for r in rows:
                f.write(f"{r[0]:.5f},{r[1]:.5f},{r[2]:.3f},{r[3]},{r[4]},{r[5]},{r[6]}\n")

        rec = {"day": int(day), "n_obs": n_obs, "n_nonempty": int(n_nonempty),
               "sim_mean": float(np.mean([len(e) for e in events]))}
        if n_nonempty == 0:
            rec["skipped"] = "no simulated events"; results.append(rec); print(json.dumps(rec)); continue

        fc_start = time_utils.strptime_to_utc_datetime(day_start.strftime("%Y-%m-%d %H:%M:%S"))
        fc_end = fc_start + dt.timedelta(days=1)
        forecast = csep.load_catalog_forecast(str(fc_path), start_time=fc_start, end_time=fc_end,
                                              region=smr, filter_spatial=True, apply_filters=True, n_cat=args.n_sims)
        forecast.filters = [f"origin_time >= {forecast.start_epoch}",
                            f"origin_time < {forecast.end_epoch}",
                            f"magnitude >= {forecast.min_magnitude}"]
        forecast.get_expected_rates(verbose=False)
        obs = csep.load_catalog(str(obs_path))
        obs = obs.filter_spatial(forecast.region).filter(f"magnitude >= {cfg.data.mcut}").filter(forecast.filters)
        rec["n_obs_csep"] = int(obs.event_count)

        def _q(x):
            return [float(v) for v in x] if isinstance(x, (list, tuple, np.ndarray)) else float(x)
        try:
            rec["N"] = {"quantile": _q(number_test(forecast, obs, verbose=False).quantile)}
        except Exception as e:
            rec["N_err"] = str(e)
        if obs.event_count > 0:
            try:
                st = spatial_test(forecast, obs, verbose=False)
                rec["S"] = {"quantile": _q(st.quantile), "observed": float(st.observed_statistic)}
            except Exception as e:
                rec["S_err"] = str(e)
            try:
                mt = magnitude_test(forecast, obs, verbose=False)
                rec["M"] = {"quantile": _q(mt.quantile), "observed": float(mt.observed_statistic)}
            except Exception as e:
                rec["M_err"] = str(e)
        results.append(rec); print(json.dumps(rec), flush=True)

    summary = csep_summary(results)
    out = {"ckpt": args.ckpt, "head": args.head, "n_sims": args.n_sims,
           "n_days": len(days), "summary": summary, "results": results,
           "note": "spatial locations from full-history neural-ETAS head; counts/times/mags from production simulator"}
    json.dump(out, open(out_dir / "csep_results.json", "w"), indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'csep_results.json'}")


if __name__ == "__main__":
    main()
