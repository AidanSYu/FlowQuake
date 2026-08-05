"""Read the trained model as an INSTRUMENT, not a scoreboard.

The model has already learned three physically-interpretable objects that no
script in this repository has ever plotted, compared, or falsified:

  theta, rho   `KernelMixtureHead._params` learns a per-component elliptical
               triggering kernel with axes (d*rho, d/rho) at azimuth theta
               (`flowquake/heads.py:82-86`; the code comment says
               "fault-strike elongation"). Nobody has asked whether the theta
               the model puts around an M>=6 parent agrees with that event's
               actual rupture azimuth. If it does, the claim is:

                 "a marked point process given only hypocentres, times and
                  magnitudes recovers mainshock rupture azimuth to within X
                  degrees within N hours — before any finite-fault inversion
                  exists"

               which has direct operational value, because USGS/INGV currently
               WAIT for a rupture model to make an aftershock forecast
               anisotropic. Compare against GCMT/SCSN nodal-plane strikes or
               SRCMOD long-axis azimuths.

  beta         `GRMagnitudeHead.beta` (`flowquake/heads.py:167-168`) is a
               history-conditioned Gutenberg-Richter b-value with an exact
               likelihood and CSEP M-test validation. The b-value-as-stress-meter
               debate is one of the most contested in the field and this is a
               data-driven b(t | recent seismicity) that has never been plotted.

  bg weight    the last two mixture components are [uniform, smoothed-map]; their
               posterior share is P(this event is background | history) — the
               probabilistic-declustering product agencies actually use ETAS for.

None of this needs new data or a GPU. It converts a benchmark model into a
measuring instrument, which is the difference between a JGR paper and a claim
about the Earth (MOONSHOT.md, "venue ladder").

Usage:
    python scripts/readouts.py runs/n1_density/ckpt_best.pt \
        --out runs/readouts --parent-mag 5.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowquake.config import Config
from flowquake.data import LAST_K, full_sequence_batch
from flowquake.train import load_catalog_cfg, make_model

# elapsed times after a mainshock at which to read the kernel, in days
ELAPSED_DAYS = [1 / 24, 6 / 24, 1.0, 7.0, 30.0]


def theta_to_strike_deg(theta_rad: np.ndarray) -> np.ndarray:
    """Kernel azimuth -> geographic strike, degrees clockwise from North.

    EarthquakeNPP's frame is (x, y) = (NORTHING, EASTING) — verified against the
    shipped catalog to 5e-12 km, see scripts/build_comcat_lowmc.py. So the head's
    theta = atan2(dy, dx) (flowquake/heads.py:85) is already measured from North
    towards East, i.e. it IS the geographic azimuth; no 90-degree conversion.
    Folded to [0, 180) because a strike and its reciprocal are the same line.

    (An earlier version applied the usual east-north convention and would have
    reported every rupture azimuth rotated by 90 degrees.)
    """
    return np.degrees(theta_rad) % 180.0


def circ_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest angle between two axial directions (mod 180), in degrees."""
    d = np.abs((a - b) % 180.0)
    return np.minimum(d, 180.0 - d)


@torch.no_grad()
def kernel_readout(model, cat, device, parent_mag: float, out: Path) -> list[dict]:
    """theta / rho / d / q of the kernel component anchored AT each large parent,
    read at several elapsed times after it."""
    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "test")
    idx_test = np.flatnonzero(mask[0].numpy())
    raw = cat.raw.numpy()          # [log_tau, x, y, mag]
    t_days = cat.t_days

    parents = np.flatnonzero((raw[:, 3] >= parent_mag))
    parents = parents[(parents > LAST_K) & (parents < len(raw) - 2)]
    print(f"[theta] {len(parents)} parents with M>={parent_mag}", flush=True)

    rows = []
    for p in parents:
        for dt_target in ELAPSED_DAYS:
            # first test-window position at least dt_target after the parent
            want = t_days[p] + dt_target
            cand = idx_test[(t_days[idx_test] >= want)]
            if len(cand) == 0:
                continue
            i = int(cand[0])
            if t_days[i] - t_days[p] > 3.0 * dt_target + 1.0:
                continue        # no event close enough to that elapsed time
            cond = model._cond(tokens[:, :i + 1].to(device),
                               torch.zeros(1, i + 1, dtype=torch.bool).index_fill_(
                                   1, torch.tensor([i]), True).to(device))
            comp_xy, comp_feats = model._comp_inputs(lastk[0, i].unsqueeze(0).to(device))
            log_w, d, q, rho, theta = model.head_s._params(cond, comp_feats)
            # locate the component anchored at this parent (match by position)
            px, py = raw[p, 1], raw[p, 2]
            dist = np.hypot(comp_xy[0, :, 0].cpu().numpy() - px,
                            comp_xy[0, :, 1].cpu().numpy() - py)
            j = int(np.argmin(dist))
            if dist[j] > 0.5:      # parent not represented among components
                continue
            rows.append({
                "parent_idx": int(p), "parent_mag": float(raw[p, 3]),
                "parent_x": float(px), "parent_y": float(py),
                "target_idx": i, "elapsed_days": float(t_days[i] - t_days[p]),
                "elapsed_bin_days": dt_target, "component": j,
                "theta_rad": float(theta[0, j]),
                "strike_deg": float(theta_to_strike_deg(np.array([float(theta[0, j])]))[0]),
                "rho": float(rho[0, j]), "d_km": float(d[0, j]), "q": float(q[0, j]),
                "log_weight": float(log_w[0, j]),
            })
    json.dump(rows, open(out / "kernel_theta.json", "w"), indent=2)
    return rows


@torch.no_grad()
def beta_readout(model, cat, device, out: Path, chunk: int = 2048) -> dict:
    """History-conditioned Gutenberg-Richter b through the test window."""
    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "test")
    idx = np.flatnonzero(mask[0].numpy())
    betas, times = [], []
    for c0 in range(0, len(idx), chunk):
        sel = idx[c0:c0 + chunk]
        mk = torch.zeros(1, tokens.shape[1], dtype=torch.bool)
        mk[0, torch.from_numpy(sel)] = True
        cond = model._cond(tokens.to(device), mk.to(device))
        b = model.head_m.beta(cond).cpu().numpy() / math.log(10.0)   # beta -> b
        betas.append(b); times.append(cat.t_days[sel])
    b = np.concatenate(betas); t = np.concatenate(times)
    res = {"n": int(len(b)), "b_mean": float(b.mean()), "b_std": float(b.std()),
           "b_p05": float(np.percentile(b, 5)), "b_p95": float(np.percentile(b, 95)),
           "t_days": t.tolist(), "b_value": b.tolist()}
    json.dump(res, open(out / "beta_trajectory.json", "w"), indent=2)
    print(f"[beta ] b = {res['b_mean']:.3f} +- {res['b_std']:.3f} "
          f"(5-95%: {res['b_p05']:.3f}-{res['b_p95']:.3f})", flush=True)
    return res


@torch.no_grad()
def background_readout(model, cat, device, out: Path, chunk: int = 2048) -> dict:
    """P(background | history) per test event: the last two mixture components
    are [uniform, smoothed map], so their posterior share is the probabilistic
    declustering an agency would use for PSHA background rates."""
    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "test")
    idx = np.flatnonzero(mask[0].numpy())
    pbg, times = [], []
    for c0 in range(0, len(idx), chunk):
        sel = idx[c0:c0 + chunk]
        mk = torch.zeros(1, tokens.shape[1], dtype=torch.bool)
        mk[0, torch.from_numpy(sel)] = True
        cond = model._cond(tokens.to(device), mk.to(device))
        _, comp_feats = model._comp_inputs(lastk[0, sel].to(device))
        log_w, *_ = model.head_s._params(cond, comp_feats)
        w = log_w.exp()
        pbg.append(w[:, -2:].sum(-1).cpu().numpy())
        times.append(cat.t_days[sel])
    p = np.concatenate(pbg); t = np.concatenate(times)
    res = {"n": int(len(p)), "mean_p_background": float(p.mean()),
           "implied_branching_ratio": float(1.0 - p.mean()),
           "t_days": t.tolist(), "p_background": p.tolist()}
    json.dump(res, open(out / "background_probability.json", "w"), indent=2)
    print(f"[bg   ] mean P(background) = {res['mean_p_background']:.4f} "
          f"-> implied branching ratio {res['implied_branching_ratio']:.4f}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="runs/readouts")
    ap.add_argument("--parent-mag", type=float, default=5.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strikes", default=None,
                    help="optional CSV with columns parent_idx,strike_deg "
                         "(GCMT/SCSN nodal plane or SRCMOD long axis) to score "
                         "the azimuth recovery against")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = ck["cfg"]
    cat = load_catalog_cfg(cfg)
    model = make_model(cfg, ck["stats"]).to(dev).eval()
    model.load_state_dict(ck["model"])

    rows = kernel_readout(model, cat, dev, args.parent_mag, out)
    beta_readout(model, cat, dev, out)
    background_readout(model, cat, dev, out)

    if rows:
        rho = np.array([r["rho"] for r in rows])
        print(f"[theta] {len(rows)} readings | aspect ratio rho: "
              f"mean {rho.mean():.3f}, p95 {np.percentile(rho, 95):.3f}")
        if rho.max() < 1.05:
            print("       NOTE: kernels are essentially isotropic (rho ~ 1), so\n"
                  "       theta carries no information. Report that as the finding —\n"
                  "       the model did not learn rupture-scale anisotropy — rather\n"
                  "       than correlating a meaningless angle against strike.")
        for e in ELAPSED_DAYS:
            sub = [r for r in rows if r["elapsed_bin_days"] == e]
            if sub:
                rr = np.array([r["rho"] for r in sub])
                print(f"       t+{e*24:>5.1f}h: n={len(sub):>4} rho={rr.mean():.3f}")

    if args.strikes and rows:
        import pandas as pd
        ref = pd.read_csv(args.strikes).set_index("parent_idx")["strike_deg"].to_dict()
        scored = [(r, ref[r["parent_idx"]]) for r in rows if r["parent_idx"] in ref]
        if scored:
            print(f"\n[strike] {len(scored)} readings matched to a reference strike")
            for e in ELAPSED_DAYS:
                sub = [(r, s) for r, s in scored if r["elapsed_bin_days"] == e]
                if not sub:
                    continue
                err = circ_diff_deg(np.array([r["strike_deg"] for r, _ in sub]),
                                    0.0) * 0 + np.array(
                    [circ_diff_deg(np.array([r["strike_deg"]]), s)[0] for r, s in sub])
                print(f"   t+{e*24:>5.1f}h  n={len(sub):>3}  median |dstrike| = "
                      f"{np.median(err):5.1f} deg   within 20 deg: "
                      f"{100*np.mean(err <= 20):4.0f}%")
            print("   A random axial guess gives a median error of 45 deg; that is\n"
                  "   the null this must beat before any claim is made.")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
