"""Honest spatial ceiling: FIT a proper-normalised full-history triggering
density and see if it can reach/beat ETAS's -8.6898.

Denominator (the ETAS normaliser) uses fixed ETAS-style temporal triggering
weights phi_j = 10^(a*(m_j-mc)) * (1+dt/c)^-p over ALL prior events; the
spatial kernel (d0, q, gamma) and background weight A are fitted. If the best
fit lands near -8.69, matching ETAS is feasible (-> build the full-history head
and net a total-NLL win via the +0.05 temporal margin). If it stalls well
short, spatial matching is not worth it and we pivot.

Run (CPU): python scripts/proper_fit.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scripts.trigger_recon import adaptive_bg
ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")
SECONDS_PER_DAY = 86400.0; MC = 2.5; RMAX = 50.0; MAXNB = 256
A_PROD, C_OM, P_OM = 0.8, 0.01, 1.15   # fixed ETAS-style temporal triggering


def main():
    fq = pd.read_csv("runs/n1_density/per_event_test.csv", parse_dates=["time"]).rename(columns={"sll":"fq_sll"})
    full = pd.read_csv(ETAS_DIR/"augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    fx, fy, fm = full["x"].to_numpy(), full["y"].to_numpy(), full["magnitude"].to_numpy()
    ft = (full["time"]-full["time"].iloc[0]).dt.total_seconds().to_numpy()/SECONDS_PER_DAY
    idxs = np.flatnonzero(full["time"].isin(set(fq["time"])).to_numpy())
    xmin,ymin,xmax,ymax = fx.min()-10,fy.min()-10,fx.max()+10,fy.max()+10
    bg_log,nx,ny,bk = adaptive_bg(fx[ft<ft[idxs[0]]], fy[ft<ft[idxs[0]]], xmin,ymin,xmax,ymax)
    bg_at = lambda xx,yy: bg_log[np.clip(((xx-xmin)/bk).astype(int),0,nx-1), np.clip(((yy-ymin)/bk).astype(int),0,ny-1)]

    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(idxs), 6000, replace=False)); sidx = idxs[sub]
    P_all = np.power(10.0, A_PROD*(fm-MC))

    print("precomputing proper denominators (all prior events) ...", flush=True)
    Den0 = np.empty(len(sidx))
    for j,I in enumerate(sidx):
        Den0[j] = (P_all[:I]*np.power(1.0+(ft[I]-ft[:I])/C_OM, -P_OM)).sum()

    print("precomputing numerator neighbours ...", flush=True)
    tree = cKDTree(np.column_stack([fx,fy]))
    dist,nbr = tree.query(np.column_stack([fx[sidx],fy[sidx]]), k=400)
    R=np.zeros((len(sidx),MAXNB)); PHI=np.zeros_like(R); MG=np.zeros_like(R); V=np.zeros((len(sidx),MAXNB),bool)
    for j,I in enumerate(sidx):
        cand,d = nbr[j],dist[j]; keep=(cand<I)&(d<RMAX); cc=cand[keep][:MAXNB]; n=len(cc)
        R[j,:n]=d[keep][:MAXNB]; MG[j,:n]=fm[cc]; V[j,:n]=True
        PHI[j,:n]=P_all[cc]*np.power(1.0+(ft[I]-ft[cc])/C_OM,-P_OM)   # same phi as denom
    ebg = np.exp(bg_at(fx[sidx],fy[sidx]))

    def negll(theta):
        ld,lq,lgam,lA = theta
        d0,q,gam,A = np.exp(ld),1.05+np.exp(lq),np.exp(lgam),np.exp(lA)
        dj2=(d0**2)*np.power(10.0,gam*(MG-MC))
        K=(q-1.0)/(np.pi*dj2)*np.power(1.0+R**2/dj2,-q)
        num=(PHI*np.where(V,K,0.0)).sum(1)+A*ebg
        den=Den0+A
        return -np.log(np.clip(num/np.clip(den,1e-12,None),1e-300,None)).mean()

    print("fitting spatial kernel (d0,q,gamma,A) ...", flush=True)
    best=None
    for d0,q,gam,A in [(1.0,1.5,0.5,1.0),(0.5,1.4,0.4,0.5),(2.0,1.6,0.3,2.0)]:
        th0=[np.log(d0),np.log(q-1.05),np.log(gam),np.log(A)]
        r=minimize(negll,th0,method="Nelder-Mead",options={"maxiter":1500,"xatol":1e-3,"fatol":1e-4})
        if best is None or r.fun<best.fun: best=r
    ld,lq,lgam,lA=best.x
    print("fitted: d0=%.3f q=%.3f gam=%.3f A=%.3f"%(np.exp(ld),1.05+np.exp(lq),np.exp(lgam),np.exp(lA)))
    sll=-best.fun
    print(f"\nproper-normalised FIT sll: {sll:.4f}")
    print(f"FQ committed -9.054 | n1_near -9.015 | ETAS -8.6898")
    print(f"=> spatial match {'FEASIBLE (>= ETAS)' if sll>=-8.69 else 'within '+format(-8.6898-sll,'+.3f')+' of ETAS'}")


if __name__ == "__main__":
    main()
