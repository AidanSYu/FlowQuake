"""Decisive go/no-go: can a properly-normalised neural-ETAS spatial head reach
ETAS's sll -8.6898 on the ComCat test set?

Spatial density of event i given history (events j<i), evaluated at the event:
    f(s_i) = [ A*mu(s_i) + sum_{j<i} phi_j K_j(s_i) ] / [ A + sum_{j<i} phi_j ]
  phi_j = 10^(a*(m_j-mc)) * sum_m w_m exp(-(t_i-t_j)/tau_m)   (productivity x learned Omori)
  K_j   = (q-1)/(pi d_j^2) (1 + r^2/d_j^2)^-q,  d_j^2 = d0^2 10^(gamma*(m_j-mc))  (mag-scaled)
  mu    = adaptive (variable-bandwidth) smoothed-seismicity background, train-only fit

Denominator sums over ALL prior events (proper normaliser) via a stable
running scan S_m[i]; numerator over the event's prior neighbours within RMAX.
Learns w_m (Omori as a positive mixture of fixed exponentials), d0, q, gamma, A.
Trained on train+val target events, early-stopped on val, evaluated on test.

Run: python scripts/neural_etas_spatial.py [--alpha 0.8] [--device cuda]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.spatial import cKDTree
from scripts.trigger_recon import adaptive_bg

ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")
SECONDS_PER_DAY = 86400.0; MC = 2.5; RMAX = 50.0; MAXNB = 256
TAUS = np.array([0.003, 0.03, 0.3, 3.0, 30.0, 300.0, 3000.0])  # Omori timescales (days)


def stable_scan(t, P, taus):
    """S_m[i] = sum_{j<i} P_j exp(-(t_i - t_j)/tau_m), per timescale (N, M)."""
    N = len(t); M = len(taus); S = np.zeros((N, M))
    cur = np.zeros(M)
    for i in range(1, N):
        dt = t[i] - t[i - 1]
        cur = np.exp(-dt / taus) * (cur + P[i - 1])
        S[i] = cur
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.8)  # productivity 10^(a*(m-mc))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    dev = torch.device(args.device)

    full = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    fx, fy, fm = full["x"].to_numpy(), full["y"].to_numpy(), full["magnitude"].to_numpy()
    ft = (full["time"] - full["time"].iloc[0]).dt.total_seconds().to_numpy() / SECONDS_PER_DAY
    N = len(full)
    val_start = pd.Timestamp("1998-01-01"); test_start = pd.Timestamp("2007-01-01"); test_end = pd.Timestamp("2020-01-17")
    tt = full["time"]
    is_train = ((tt >= "1981-01-01") & (tt < val_start)).to_numpy()
    is_val = ((tt >= val_start) & (tt < test_start)).to_numpy()
    is_test = ((tt >= test_start) & (tt < test_end)).to_numpy()

    # adaptive background, fit train-only (< val_start), looked up at each event
    xmin, ymin, xmax, ymax = fx.min()-10, fy.min()-10, fx.max()+10, fy.max()+10
    trn = (tt < val_start).to_numpy()
    bg_log, nx, ny, bk = adaptive_bg(fx[trn], fy[trn], xmin, ymin, xmax, ymax)
    bgL = bg_log[np.clip(((fx-xmin)/bk).astype(int),0,nx-1), np.clip(((fy-ymin)/bk).astype(int),0,ny-1)]

    print("denominator scan ...", flush=True)
    P = np.power(10.0, args.alpha * (fm - MC))
    S = stable_scan(ft, P, TAUS)                       # (N, M)

    print("numerator neighbours ...", flush=True)
    tree = cKDTree(np.column_stack([fx, fy]))
    R = np.zeros((N, MAXNB), np.float32); DT = np.zeros_like(R); MGN = np.zeros_like(R)
    V = np.zeros((N, MAXNB), bool)
    for s in range(0, N, 8192):
        e = min(s + 8192, N)
        dist, nbr = tree.query(np.column_stack([fx[s:e], fy[s:e]]), k=400)
        for r, i in enumerate(range(s, e)):
            cand, d = nbr[r], dist[r]
            keep = (cand < i) & (d < RMAX); cc = cand[keep][:MAXNB]; n = len(cc)
            R[i, :n] = d[keep][:MAXNB]; DT[i, :n] = ft[i] - ft[cc]; MGN[i, :n] = fm[cc]; V[i, :n] = True

    # to torch
    taus = torch.tensor(TAUS, device=dev, dtype=torch.float32)
    St = torch.tensor(S, device=dev, dtype=torch.float32)
    Rt, DTt, MGt, Vt = (torch.tensor(a, device=dev) for a in (R, DT, MGN, V))
    bgt = torch.tensor(np.exp(bgL), device=dev, dtype=torch.float32)
    Pn = torch.pow(torch.tensor(10.0, device=dev), args.alpha * (MGt - MC))  # neighbour productivity

    # params
    w_raw = torch.zeros(len(TAUS), device=dev, requires_grad=True)
    log_d0 = torch.tensor(np.log(1.0), device=dev, requires_grad=True)
    log_qm = torch.tensor(np.log(0.5), device=dev, requires_grad=True)   # q = 1.05 + softplus
    log_gam = torch.tensor(np.log(0.4), device=dev, requires_grad=True)
    log_A = torch.tensor(np.log(1.0), device=dev, requires_grad=True)
    params = [w_raw, log_d0, log_qm, log_gam, log_A]
    opt = torch.optim.Adam(params, lr=0.05)

    idx_train = np.flatnonzero(is_train | is_val)   # fit on train+val
    idx_test = np.flatnonzero(is_test)

    def sll_on(idx, train=True):
        ii = torch.tensor(idx, device=dev)
        w = torch.nn.functional.softplus(w_raw)
        d0 = torch.exp(log_d0); q = 1.05 + torch.nn.functional.softplus(log_qm)
        gam = torch.exp(log_gam); A = torch.exp(log_A)
        # neighbour temporal weight: 10^(a dm) * sum_m w_m exp(-dt/tau_m)
        decay = torch.exp(-DTt[ii].unsqueeze(-1) / taus)          # (B,K,M)
        om = (decay * w).sum(-1)                                   # (B,K)
        phi = Pn[ii] * om
        dj2 = d0**2 * torch.pow(torch.tensor(10.0, device=dev), gam * (MGt[ii] - MC))
        K = (q - 1.0) / (np.pi * dj2) * torch.pow(1.0 + Rt[ii]**2 / dj2, -q)
        num = (phi * torch.where(Vt[ii], K, torch.zeros_like(K))).sum(-1) + A * bgt[ii]
        den = (St[ii] * w).sum(-1) + A
        return torch.log(torch.clamp(num / torch.clamp(den, min=1e-12), min=1e-30))

    best_val = -1e9; best_state = None; bad = 0
    rng = np.random.default_rng(0)
    for step in range(1, args.steps + 1):
        b = rng.choice(idx_train, 8192, replace=False)
        loss = -sll_on(b).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                vsll = sll_on(np.flatnonzero(is_val)).mean().item()
                tsll = sll_on(idx_test).mean().item()
            print(f"step {step:>5}  train_sll {-loss.item():.4f}  val {vsll:.4f}  test {tsll:.4f}", flush=True)
            if vsll > best_val:
                best_val = vsll; bad = 0
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad >= 8:
                    print("early stop"); break

    for p, b in zip(params, best_state):
        p.data.copy_(b)
    with torch.no_grad():
        test_sll = sll_on(idx_test).mean().item()
    w = torch.nn.functional.softplus(w_raw).detach().cpu().numpy()
    print(f"\nalpha={args.alpha}  d0={torch.exp(log_d0).item():.3f}  "
          f"q={1.05+torch.nn.functional.softplus(log_qm).item():.3f}  "
          f"gam={torch.exp(log_gam).item():.3f}  A={torch.exp(log_A).item():.3f}")
    print("omori w_m:", np.round(w, 3), "taus:", TAUS)
    print(f"\nneural-ETAS spatial TEST sll: {test_sll:.4f}")
    print(f"FQ committed -9.054 | n1_near -9.015 | ETAS -8.6898")
    print(f"=> {'MATCHES/BEATS ETAS' if test_sll>=-8.69 else 'within '+format(-8.6898-test_sll,'+.3f')+' of ETAS'}")
    if test_sll >= -8.74:
        print("   (with tll +0.053 margin this flips TOTAL nll vs ETAS 7.2554)")


if __name__ == "__main__":
    main()
