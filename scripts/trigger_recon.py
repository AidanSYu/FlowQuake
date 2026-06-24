"""Confirm the ceiling: can an ETAS-style spatial density with FULL triggering
history (not just last-64) reach ETAS's sll -8.69 on the test set?

For each test event, gather ALL prior events within 50 km, weight each by
productivity 10^(a*(m-mc)) * Omori (1+dt/c)^-p, place a mag-scaled power-law
spatial kernel, add an adaptive background, normalise to a proper density, and
maximise mean log f over the test events (a few free params). If this reaches
~ -8.69 it proves the gap is triggering coverage and gives the kernel to add.

Run (CPU): python scripts/trigger_recon.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")
SECONDS_PER_DAY = 86400.0
MC = 2.5
RMAX = 50.0
MAXNB = 256


def adaptive_bg(x, y, xmin, ymin, xmax, ymax, k=4, smin=0.75, floor=0.03, bin_km=2.0):
    nx = int(np.ceil((xmax - xmin) / bin_km)); ny = int(np.ceil((ymax - ymin) / bin_km))
    pts = np.column_stack([x, y]); tree = cKDTree(pts)
    dk, _ = tree.query(pts, k=k + 1); sig = np.clip(dk[:, -1], smin, 60.0)
    gx = np.clip(((x - xmin) / bin_km).astype(int), 0, nx - 1)
    gy = np.clip(((y - ymin) / bin_km).astype(int), 0, ny - 1)
    edges = np.logspace(np.log10(sig.min()), np.log10(sig.max() + 1e-6), 13)
    grid = np.zeros((nx, ny))
    for b in range(12):
        sel = (sig >= edges[b]) & (sig < edges[b + 1] if b < 11 else sig <= edges[b + 1])
        if sel.any():
            sub = np.zeros((nx, ny)); np.add.at(sub, (gx[sel], gy[sel]), 1.0)
            grid += gaussian_filter(sub, sigma=np.sqrt(edges[b] * edges[b + 1]) / bin_km)
    area = (xmax - xmin) * (ymax - ymin)
    grid = grid / (grid.sum() * bin_km ** 2 + 1e-12)
    grid = (1 - floor) * grid + floor / area
    return np.log(grid).astype(np.float64), nx, ny, bin_km


def main():
    fq = pd.read_csv("runs/n1_density/per_event_test.csv", parse_dates=["time"]).rename(columns={"sll": "fq_sll"})
    full = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    fx, fy = full["x"].to_numpy(), full["y"].to_numpy()
    fm = full["magnitude"].to_numpy()
    ft = (full["time"] - full["time"].iloc[0]).dt.total_seconds().to_numpy() / SECONDS_PER_DAY
    test_set = set(fq["time"]); is_test = full["time"].isin(test_set).to_numpy()
    idxs = np.flatnonzero(is_test)

    xmin, ymin = fx.min() - 10, fy.min() - 10
    xmax, ymax = fx.max() + 10, fy.max() + 10
    # background fit on pre-test events only (causal)
    pretest = ft < ft[idxs[0]]
    bg_log, nx, ny, bin_km = adaptive_bg(fx[pretest], fy[pretest], xmin, ymin, xmax, ymax)

    def bg_at(xx, yy):
        ix = np.clip(((xx - xmin) / bin_km).astype(int), 0, nx - 1)
        iy = np.clip(((yy - ymin) / bin_km).astype(int), 0, ny - 1)
        return bg_log[ix, iy]

    tree = cKDTree(np.column_stack([fx, fy]))
    dist, nbr = tree.query(np.column_stack([fx[idxs], fy[idxs]]), k=400)

    # ragged -> padded arrays of prior neighbours within RMAX
    R = np.zeros((len(idxs), MAXNB)); DT = np.zeros_like(R); MG = np.zeros_like(R)
    VALID = np.zeros((len(idxs), MAXNB), bool)
    for j, i in enumerate(idxs):
        cand = nbr[j]; d = dist[j]
        keep = (cand < i) & (d < RMAX)
        cc = cand[keep][:MAXNB]; dd = d[keep][:MAXNB]
        n = len(cc)
        R[j, :n] = dd; DT[j, :n] = ft[i] - ft[cc]; MG[j, :n] = fm[cc]; VALID[j, :n] = True
    bgL = bg_at(fx[idxs], fy[idxs])

    ebg = np.exp(bgL)

    def negll_on(theta, Rs, DTs, MGs, Vs, bgs):
        la, lc, lp, ld, lq, lgam, lwbg = theta
        a, c, p = np.exp(la), np.exp(lc), 1.0 + np.exp(lp)
        d0, q = np.exp(ld), 1.05 + np.exp(lq)
        gam, wbg = np.exp(lgam), np.exp(lwbg)
        dj2 = (d0 ** 2) * np.power(10.0, gam * (MGs - MC))          # mag-scaled zone
        psi = np.power(10.0, a * (MGs - MC)) * np.power(1.0 + DTs / c, -p)
        psi = np.where(Vs, psi, 0.0)
        fr = (q - 1.0) / (np.pi * dj2) * np.power(1.0 + Rs ** 2 / dj2, -q)
        trig = (psi * fr).sum(1)
        norm = psi.sum(1) + wbg
        dens = (trig + wbg * bgs) / np.clip(norm, 1e-12, None)
        return -np.log(np.clip(dens, 1e-300, None)).mean()

    negll = lambda th: negll_on(th, R, DT, MG, VALID, ebg)
    # a few hand-set ETAS-like params first (instant) -> always get a number
    #          [a,    c,     p,    d0,  q,    gam,  wbg]
    presets = {
        "etas-typical": [0.6, 0.01, 1.15, 1.0, 1.5, 0.5, 1.0],
        "tight-d":      [0.6, 0.01, 1.15, 0.4, 1.4, 0.4, 0.5],
        "broad":        [0.8, 0.05, 1.20, 2.0, 1.6, 0.6, 2.0],
    }
    for name, ps in presets.items():
        th = np.log([ps[0], ps[1], ps[2] - 1.0, ps[3], ps[4] - 1.05, ps[5], ps[6]])
        print(f"  preset {name:>14}: sll {-negll(th):.4f}", flush=True)

    # light refine on a 3000-event subsample (fast)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(idxs), min(3000, len(idxs)), replace=False)
    th0 = np.log([0.6, 0.01, 0.15, 0.6, 0.35, 0.4, 0.5])
    print(f"refining on {len(sub)} events ...", flush=True)
    res = minimize(lambda th: negll_on(th, R[sub], DT[sub], MG[sub], VALID[sub], ebg[sub]),
                   th0, method="Powell",
                   options={"maxiter": 400, "xtol": 1e-2, "ftol": 1e-3})
    la, lc, lp, ld, lq, lgam, lwbg = res.x
    print("fitted: a=%.3f c=%.4f p=%.3f d0=%.3f q=%.3f gam=%.3f wbg=%.3f" % (
        np.exp(la), np.exp(lc), 1 + np.exp(lp), np.exp(ld), 1.05 + np.exp(lq),
        np.exp(lgam), np.exp(lwbg)))
    sll = -negll(res.x)
    print(f"\nETAS-recon (full history) sll: {sll:.4f}")
    print(f"FQ committed              sll: {fq.fq_sll.mean():.4f}")
    print(f"ETAS published            sll: -8.6898")
    print(f"\n=> ceiling check: recon {'BEATS' if sll > -8.6898 else 'trails'} ETAS; "
          f"gain over FQ = {sll - fq.fq_sll.mean():+.4f}")


if __name__ == "__main__":
    main()
