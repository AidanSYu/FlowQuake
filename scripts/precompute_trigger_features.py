"""Precompute full-history trigger features for the neural-ETAS spatial head.

For every target event (time >= timewindow_start) of an inverted-ETAS catalog:
  * trig_num  = sum_{j<i} w_ij K_ij(s_i)      (frozen ETAS params, ALL priors)
  * trig_den  = sum_{j<i} w_ij Z_j
  * kde_k(s_i) = (1/n_i) sum_{j<i} Gaussian_bw_k(r_ij)   causal multi-scale
        seismicity density at the target location (bw in KDE_BWS, km);
        each Gaussian integrates to 1 over R^2, so the mixture is a valid
        (slightly mass-leaking at region edges, i.e. conservative) density.
  * a NEAR SET of parents selected TARGET-LOCATION-INDEPENDENTLY:
        top NEAR_W by ETAS weight w_ij  +  NEAR_P nearest to the location of
        event i-1 (sequence locality; causal). Stored per parent: index, r^2 to
        the target, dt (days). The trainer recomputes their base contribution
        live and learns modulations; far-field = (trig - near_base) stays
        frozen. Selection never looks at s_i, so the learned density remains
        properly normalized.

Verifies the frozen sums reproduce the package's SLL on test rows before
writing runs/<name>_trigfeat.npz.

Run (CPU, ~30-60 min): python scripts/precompute_trigger_features.py ComCat_25
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KDE_BWS = [1.5, 6.0, 25.0, 100.0]
NEAR_W, NEAR_P = 256, 128
BLOCK = 128


def round_half_up(x, delta=0.1):
    return np.floor(x / delta + 0.5) * delta


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="ComCat_25")
    ap.add_argument("--catalog", default=None,
                    help="alternate catalog csv (e.g. the 2020-2026 extended one); "
                         "magnitudes get delta_m rounding + mc filter like the package")
    ap.add_argument("--targets-from", default=None, help="override target window start")
    ap.add_argument("--out", default=None, help="output stem (default <name>)")
    args = ap.parse_args()
    name = args.name
    out_dir = Path(f"reference/Experiments/ETAS/output_data_{name}")
    meta = json.loads((out_dir / "parameters_0.json").read_text())
    p = meta["final_parameters"]
    mu = 10.0 ** p["log10_mu"]
    k0 = 10.0 ** p["log10_k0"]
    c = 10.0 ** p["log10_c"]
    tau = 10.0 ** p["log10_tau"]
    d = 10.0 ** p["log10_d"]
    a, omega, gamma, rho = p["a"], p["omega"], p["gamma"], p["rho"]
    mc, area, R = meta["mc"], meta["area"], meta["earth_radius"]
    train_start = pd.Timestamp(args.targets_from or meta["timewindow_start"])

    if args.catalog:
        cat = pd.read_csv(args.catalog, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
        cat["magnitude"] = round_half_up(cat["magnitude"].to_numpy())
        cat = cat[cat["magnitude"] >= mc].reset_index(drop=True)
        cat["SLL"] = np.nan  # no reference on alternate catalogs
    else:
        cat = pd.read_csv(out_dir / "augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    t_days = (cat["time"] - cat["time"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
    lat = np.radians(cat["latitude"].to_numpy())
    lon = np.radians(cat["longitude"].to_numpy())
    m = cat["magnitude"].to_numpy()
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)

    targets = np.flatnonzero((cat["time"] >= train_start).to_numpy())
    event_index = cat["index_from_zero"].to_numpy() if "index_from_zero" in cat else np.arange(len(cat))
    n_t, n_near = len(targets), NEAR_W + NEAR_P
    print(f"{name}: {len(cat)} events, {n_t} targets from {cat['time'].iloc[targets[0]]}")

    prod = k0 * np.exp(a * (m - mc))
    dm = d * np.exp(gamma * (m - mc))
    zint = np.pi / (rho * np.power(dm, rho))
    inv2bw2 = np.array([1.0 / (2.0 * bw * bw) for bw in KDE_BWS])
    kde_norm = np.array([1.0 / (2.0 * np.pi * bw * bw) for bw in KDE_BWS])

    trig_num = np.zeros(n_t)
    trig_den = np.zeros(n_t)
    kde = np.zeros((n_t, len(KDE_BWS)))
    near_idx = np.full((n_t, n_near), -1, dtype=np.int32)
    near_r2 = np.zeros((n_t, n_near), dtype=np.float32)
    near_dt = np.zeros((n_t, n_near), dtype=np.float32)
    near_base_num = np.zeros(n_t)
    near_base_den = np.zeros(n_t)

    t0 = time.time()
    for b0 in range(0, n_t, BLOCK):
        idx = targets[b0:b0 + BLOCK]
        hi = idx[-1]
        dlat = lat[idx, None] - lat[None, :hi]
        dlon = lon[idx, None] - lon[None, :hi]
        h = np.sin(dlat / 2.0) ** 2 + cos_lat[idx, None] * cos_lat[None, :hi] * np.sin(dlon / 2.0) ** 2
        r2 = np.square(2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))
        dt = np.maximum(t_days[idx, None] - t_days[None, :hi], 0.0)
        mask = np.arange(hi)[None, :] < idx[:, None]
        w = np.where(mask, prod[None, :hi] * np.exp(-dt / tau) / np.power(dt + c, 1.0 + omega), 0.0)
        kmat = np.power(r2 + dm[None, :hi], -(1.0 + rho))
        sl = slice(b0, b0 + len(idx))
        trig_num[sl] = mu * 0.0 + (w * kmat).sum(axis=1)
        trig_den[sl] = (w * zint[None, :hi]).sum(axis=1)
        n_prior = idx.astype(np.float64)  # index == number of prior events
        for k in range(len(KDE_BWS)):
            g = kde_norm[k] * np.exp(-r2 * inv2bw2[k])
            kde[sl, k] = np.where(mask, g, 0.0).sum(axis=1) / np.maximum(n_prior, 1.0)

        # near-set selection (target-location-independent)
        w_masked = np.where(mask, w, -1.0)
        prev = idx - 1  # most recent event before target
        dlat_p = lat[prev, None] - lat[None, :hi]
        dlon_p = lon[prev, None] - lon[None, :hi]
        hp = np.sin(dlat_p / 2.0) ** 2 + cos_lat[prev, None] * cos_lat[None, :hi] * np.sin(dlon_p / 2.0) ** 2
        r2_prev = np.where(mask, np.square(2.0 * R * np.arcsin(np.sqrt(np.clip(hp, 0.0, 1.0)))), np.inf)
        for row, i_ev in enumerate(idx):
            n_avail = i_ev
            kw = min(NEAR_W, n_avail)
            top_w = np.argpartition(w_masked[row, :i_ev], -kw)[-kw:] if n_avail > kw else np.arange(n_avail)
            kp = min(NEAR_P, n_avail)
            top_p = np.argpartition(r2_prev[row, :i_ev], kp - 1)[:kp] if n_avail > kp else np.arange(n_avail)
            sel = np.unique(np.concatenate([top_w, top_p]))[:n_near]
            g = b0 + row
            near_idx[g, :len(sel)] = sel
            near_r2[g, :len(sel)] = r2[row, sel]
            near_dt[g, :len(sel)] = dt[row, sel]
            near_base_num[g] = (w[row, sel] * kmat[row, sel]).sum()
            near_base_den[g] = (w[row, sel] * zint[sel]).sum()
        if (b0 // BLOCK) % 50 == 0:
            el = time.time() - t0
            print(f"  {b0}/{n_t}  ({el:.0f}s, eta {el / max(b0, 1) * (n_t - b0):.0f}s)", flush=True)

    # verify on test rows: frozen sums must reproduce the package SLL
    sll_ref = cat["SLL"].to_numpy()[targets]
    fin = np.isfinite(sll_ref)
    sll_ours = np.log(mu + trig_num) - np.log(mu * area + trig_den)
    if fin.any():
        err = np.abs(sll_ours[fin] - sll_ref[fin]).max()
        print(f"verify vs package SLL on {fin.sum()} test rows: max_abs_err {err:.3e}")
        assert err < 1e-6, "frozen sums do not reproduce ETAS -- abort"
    else:
        print(f"no reference SLL (alternate catalog); mean frozen-ETAS sll {sll_ours.mean():.4f}")

    Path("runs").mkdir(exist_ok=True)
    np.savez_compressed(
        f"runs/{args.out or name}_trigfeat.npz",
        targets=targets.astype(np.int64), trig_num=trig_num, trig_den=trig_den,
        far_num=trig_num - near_base_num, far_den=trig_den - near_base_den,
        kde=kde, kde_bws=np.array(KDE_BWS),
        near_idx=near_idx, near_r2=near_r2, near_dt=near_dt,
        mag=m.astype(np.float32), t_days=t_days,
        time_iso=cat["time"].astype(str).to_numpy(),
        event_index=event_index.astype(np.int64),
        params=np.array([mu, k0, c, tau, d, a, omega, gamma, rho, mc, area]),
    )
    print(f"wrote runs/{name}_trigfeat.npz  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
