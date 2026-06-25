"""True ceiling with a PROPER normalizer (ETAS-style full-history sum).

trigger_recon normalised only over each event's nearest neighbours -> not a
valid density. A real model must use f(s) = [A*mu(s) + sum_j phi_j K_j(s)] /
[A + sum_j phi_j], where the denominator sums phi over ALL prior events. This
computes that exact density (numerator via the target's prior neighbours,
denominator via all prior events) on a test subsample, fits the few kernel
params, and reports the achievable sll. This is the honest go/no-go for a
neural-ETAS spatial head.

Run (CPU): python scripts/proper_ceiling.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scripts.trigger_recon import adaptive_bg
ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")
SECONDS_PER_DAY = 86400.0
MC = 2.5
RMAX = 50.0
MAXNB = 256


def main():
    fq = pd.read_csv("runs/n1_density/per_event_test.csv", parse_dates=["time"]).rename(columns={"sll": "fq_sll"})
    full = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    fx, fy = full["x"].to_numpy(), full["y"].to_numpy()
    fm = full["magnitude"].to_numpy()
    ft = (full["time"] - full["time"].iloc[0]).dt.total_seconds().to_numpy() / SECONDS_PER_DAY
    is_test = full["time"].isin(set(fq["time"])).to_numpy()
    idxs = np.flatnonzero(is_test)

    xmin, ymin, xmax, ymax = fx.min() - 10, fy.min() - 10, fx.max() + 10, fy.max() + 10
    pretest = ft < ft[idxs[0]]
    bg_log, nx, ny, bin_km = adaptive_bg(fx[pretest], fy[pretest], xmin, ymin, xmax, ymax)
    bg_at = lambda xx, yy: bg_log[np.clip(((xx-xmin)/bin_km).astype(int),0,nx-1),
                                   np.clip(((yy-ymin)/bin_km).astype(int),0,ny-1)]

    # subsample test events for the fit (denominator is O(N) per event)
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(idxs), 4000, replace=False))
    sidx = idxs[sub]

    # numerator neighbours: target's nearest PRIOR events within RMAX
    tree = cKDTree(np.column_stack([fx, fy]))
    dist, nbr = tree.query(np.column_stack([fx[sidx], fy[sidx]]), k=400)
    R = np.zeros((len(sidx), MAXNB)); DT = np.zeros_like(R); MG = np.zeros_like(R); V = np.zeros((len(sidx), MAXNB), bool)
    for j, i in enumerate(sidx):
        cand, d = nbr[j], dist[j]
        keep = (cand < i) & (d < RMAX); cc = cand[keep][:MAXNB]
        n = len(cc); R[j,:n]=d[keep][:MAXNB]; DT[j,:n]=ft[i]-ft[cc]; MG[j,:n]=fm[cc]; V[j,:n]=True
    bgL = bg_at(fx[sidx], fy[sidx])
    ebg = np.exp(bgL)

    # denominator: sum over ALL prior events of phi_j  (chunked, exact)
    def full_denom(a, c, p):
        Z = np.empty(len(sidx))
        for j, I in enumerate(sidx):
            dtj = ft[I] - ft[:I]
            Z[j] = (np.power(10.0, a*(fm[:I]-MC)) * np.power(1.0+dtj/c, -p)).sum()
        return Z

    def negll(theta, with_full=True):
        a, lc, lp, ld, lq, lgam, lwbg = theta
        c, p = np.exp(lc), 1.0+np.exp(lp); d0, q = np.exp(ld), 1.05+np.exp(lq)
        gam, wbg = np.exp(lgam), np.exp(lwbg)
        dj2 = (d0**2)*np.power(10.0, gam*(MG-MC))
        psi = np.power(10.0, a*(MG-MC))*np.power(1.0+DT/c, -p); psi = np.where(V, psi, 0.0)
        fr = (q-1.0)/(np.pi*dj2)*np.power(1.0+R**2/dj2, -q)
        num = (psi*fr).sum(1) + wbg*ebg
        den = (full_denom(a, c, p) if with_full else psi.sum(1)) + wbg
        return -np.log(np.clip(num/np.clip(den,1e-12,None), 1e-300, None)).mean()

    # presets first (with full denominator)
    print("with PROPER full-history normaliser:")
    for name, ps in {"etas-typical":[0.6,0.01,1.15,1.0,1.5,0.5,1.0],
                     "tight-d":[0.6,0.01,1.15,0.4,1.4,0.4,0.5],
                     "broad":[0.8,0.05,1.20,2.0,1.6,0.6,2.0]}.items():
        th=[ps[0],np.log(ps[1]),np.log(ps[2]-1),np.log(ps[3]),np.log(ps[4]-1.05),np.log(ps[5]),np.log(ps[6])]
        print(f"  {name:>14}: sll {-negll(th):.4f}", flush=True)
    print(f"\nFQ committed sll -9.054 | n1_near sll -9.015 | ETAS sll -8.6898")


if __name__ == "__main__":
    main()
