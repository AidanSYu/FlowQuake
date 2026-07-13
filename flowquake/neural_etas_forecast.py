"""Full-history neural-ETAS head as a spatial FORECAST field, for CSEP.

The head is a spatial density conditioned on history; the benchmark's CSEP
S-test needs the forecast's expected spatial rate on a grid. Because the head's
near set and far field are selected target-location-INDEPENDENTLY (by ETAS
weight and the previous event's location, never by the query point), at a fixed
forecast time every query location s shares the same near set / far priors, and
only the per-prior distances r_j(s) and the KDE vary with s. So the head's
spatial density can be evaluated on the whole CSEP grid in one vectorized pass
per forecast day — the same cost as an ETAS gridded forecast.

This module provides `head_density_at(...)`, which evaluates the trained head's
log spatial density at arbitrary query points given a frozen history, and a
`--validate` mode that reconstructs the precomputed trigfeat for real events and
confirms the field evaluator reproduces the head's per-event SLL exactly.

Run:  python -m flowquake.neural_etas_forecast --validate ComCat_25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .neural_etas import NeuralETASSpatialHead

KDE_BWS = [1.5, 6.0, 25.0, 100.0]
NEAR_W, NEAR_P = 256, 128
PARAM_NAMES = ["mu", "k0", "c", "tau", "d", "a", "omega", "gamma", "rho", "mc", "area"]


def _haversine2_km(lat1, lon1, lat2, lon2, R):
    """Squared great-circle distance (km^2). lat/lon in radians. Broadcasts."""
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return np.square(2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def head_features_at(P, hist, t_now, prev_idx, q_lat, q_lon, device="cpu"):
    """Compute the head's per-query features at frozen history `hist` (dict with
    lat/lon rad, t_days, mag arrays for events strictly before t_now) evaluated
    at query points (q_lat, q_lon, radians). Returns tensors matching the head
    forward() signature. Selection (near set) is location-independent.
    """
    mu, k0, c, tau, d, a, omega, gamma, rho, mc, area, R = (
        P["mu"], P["k0"], P["c"], P["tau"], P["d"], P["a"], P["omega"],
        P["gamma"], P["rho"], P["mc"], P["area"], P["R"])
    hm, ht = hist["mag"], hist["t_days"]
    hlat, hlon = hist["lat"], hist["lon"]
    n_prior = len(hm)
    dt = np.maximum(t_now - ht, 0.0)
    w = k0 * np.exp(a * (hm - mc)) * np.exp(-dt / tau) * np.power(dt + c, -(1.0 + omega))
    dm = d * np.exp(gamma * (hm - mc))
    zint = np.pi / (rho * np.power(dm, rho))

    # near set: top-NEAR_W by ETAS weight + NEAR_P nearest to the previous event
    kw = min(NEAR_W, n_prior)
    top_w = np.argpartition(w, -kw)[-kw:] if n_prior > kw else np.arange(n_prior)
    r2_prev = _haversine2_km(hlat[prev_idx], hlon[prev_idx], hlat, hlon, R)
    kp = min(NEAR_P, n_prior)
    top_p = np.argpartition(r2_prev, kp - 1)[:kp] if n_prior > kp else np.arange(n_prior)
    near = np.unique(np.concatenate([top_w, top_p]))
    far_mask = np.ones(n_prior, bool); far_mask[near] = False

    Q = len(q_lat)
    # distances query x prior (km^2)
    r2 = _haversine2_km(q_lat[:, None], q_lon[:, None], hlat[None, :], hlon[None, :], R)
    # far field summed over far priors (scalar den; per-query num)
    far_num = (w[far_mask][None, :] * np.power(r2[:, far_mask] + dm[far_mask][None, :], -(1.0 + rho))).sum(1)
    far_den = float((w[far_mask] * zint[far_mask]).sum())
    # KDE background per query
    kde = np.zeros((Q, len(KDE_BWS)))
    for k, bw in enumerate(KDE_BWS):
        g = (1.0 / (2.0 * np.pi * bw * bw)) * np.exp(-r2 / (2.0 * bw * bw))
        kde[:, k] = g.sum(1) / max(n_prior, 1)
    # near-set tensors (query x n_near)
    nn = len(near)
    r2n = r2[:, near]
    dtn = np.broadcast_to(dt[near][None, :], (Q, nn)).copy()
    mgn = np.broadcast_to(hm[near][None, :], (Q, nn)).copy()
    valid = np.ones((Q, nn), np.float32)
    t = lambda arr: torch.as_tensor(arr, dtype=torch.float32, device=device)
    return dict(far_num=t(far_num), far_den=torch.full((Q,), far_den, device=device),
                kde=t(kde), r2=t(r2n), dt=t(np.maximum(dtn, 0.0)), mag=t(mgn), valid=t(valid))


def load_head(name, seed=0, device="cpu"):
    ck = torch.load(f"runs/neural_etas/{name}/head_full_s{seed}.pt", map_location="cpu", weights_only=False)
    model = NeuralETASSpatialHead(ck["params"], n_kde=ck["n_kde"], use_mlp=ck["use_mlp"]).to(device)
    model.load_state_dict(ck["state"]); model.eval()
    P = dict(zip(PARAM_NAMES, [float(ck["params"][k]) for k in PARAM_NAMES]))
    P["R"] = 6378.1  # benchmark earth_radius (parameters_0.json), constant across regions
    return model, P


def validate(name):
    """Reconstruct the field evaluator at a handful of real events' own history
    states and confirm it matches the precomputed trigfeat + head SLL."""
    out_dir = Path(f"reference/Experiments/ETAS/output_data_{name}")
    meta = json.loads((out_dir / "parameters_0.json").read_text())
    P = {**dict(zip(["mu", "k0", "c", "tau", "d"], [10.0 ** meta["final_parameters"][f"log10_{x}"]
                                                    for x in ["mu", "k0", "c", "tau", "d"]])),
         "a": meta["final_parameters"]["a"], "omega": meta["final_parameters"]["omega"],
         "gamma": meta["final_parameters"]["gamma"], "rho": meta["final_parameters"]["rho"],
         "mc": meta["mc"], "area": meta["area"], "R": meta["earth_radius"]}
    model, _ = load_head(name)
    z = np.load(f"runs/{name}_trigfeat.npz", allow_pickle=True)
    cat = pd.read_csv(out_dir / "augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    lat = np.radians(cat["latitude"].to_numpy()); lon = np.radians(cat["longitude"].to_numpy())
    tday = (cat["time"] - cat["time"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
    mag = cat["magnitude"].to_numpy()
    targets = z["targets"]
    rng = np.random.default_rng(0)
    pick = targets[rng.choice(len(targets), 40, replace=False)]
    errs = []
    for i in pick:
        hist = dict(lat=lat[:i], lon=lon[:i], t_days=tday[:i], mag=mag[:i])
        feats = head_features_at(P, hist, tday[i], i - 1, lat[i:i+1], lon[i:i+1])
        with torch.no_grad():
            f_field = model(**feats).item()
        # head SLL from the trigfeat pipeline for this event
        j = int(np.flatnonzero(targets == i)[0])
        errs.append(abs(f_field - _head_sll_from_trigfeat(model, z, j)))
    errs = np.array(errs)
    print(f"{name}: field-vs-trigfeat head SLL max_abs_err {errs.max():.3e} mean {errs.mean():.3e} over {len(pick)} events")
    print("MATCH" if errs.max() < 1e-3 else "MISMATCH -- investigate")


def _head_sll_from_trigfeat(model, z, j):
    """Evaluate the head on precomputed trigfeat row j (the training path)."""
    near_idx = z["near_idx"][j]
    valid = (near_idx >= 0).astype(np.float32)
    mag_all = z["mag"]
    near_m = mag_all[np.maximum(near_idx, 0)] * (near_idx >= 0) + z["params"][9] * (near_idx < 0)
    t = lambda a: torch.as_tensor(np.asarray(a)[None], dtype=torch.float32)
    with torch.no_grad():
        return model(far_num=t(z["far_num"][j]), far_den=t(z["far_den"][j]), kde=t(z["kde"][j]),
                     r2=t(z["near_r2"][j]), dt=t(np.maximum(z["near_dt"][j], 0.0)),
                     mag=t(near_m), valid=t(valid)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", metavar="NAME", default=None)
    args = ap.parse_args()
    if args.validate:
        validate(args.validate)


if __name__ == "__main__":
    main()
