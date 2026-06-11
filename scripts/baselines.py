"""Trivial CPU baselines on the val split — calibrate what signal exists.

1. Temporal: unconditional lognormal on tau, and AR(1) lognormal
   (log tau_i ~ N(a*log tau_{i-1} + b, s^2)). Reported as tll (log 1/day).
2. Spatial: KDE forecast — isotropic Gaussian mixture over the last K event
   locations (+ uniform background over the catalog bounding box), bandwidth
   and background weight fit on the train split. Reported as sll (log 1/km^2).

If the KDE sll lands far above the NN's -13.8, the spatial signal is there
and a structured mixture head is the justified pivot.
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from flowquake.config import Config
from flowquake.data import load_catalog

cfg = Config.load("configs/comcat25.yaml")
cat = load_catalog(cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
                   cfg.data.train_start, cfg.data.val_start, cfg.data.test_start,
                   cfg.data.test_end)

t = cat.t_days
x = cat.raw[:, 1].numpy().astype(np.float64)
y = cat.raw[:, 2].numpy().astype(np.float64)
log_tau = cat.raw[:, 0].numpy().astype(np.float64)

E = cat.n_events
val_idx = np.flatnonzero(cat.target_val)
train_idx = np.flatnonzero(np.arange(E) < val_idx.min())
train_idx = train_idx[1:]  # event 0 has filler log_tau

# --- temporal -----------------------------------------------------------
lt_tr = log_tau[train_idx]
mu0, s0 = lt_tr.mean(), lt_tr.std()
lt_val = log_tau[val_idx]
tll_uncond = (-0.5 * ((lt_val - mu0) / s0) ** 2 - np.log(s0) -
              0.5 * np.log(2 * np.pi) - lt_val).mean()

prev_tr, cur_tr = log_tau[train_idx - 1], log_tau[train_idx]
a, b = np.polyfit(prev_tr, cur_tr, 1)
res = cur_tr - (a * prev_tr + b)
s1 = res.std()
prev_v = log_tau[val_idx - 1]
res_v = lt_val - (a * prev_v + b)
tll_ar1 = (-0.5 * (res_v / s1) ** 2 - np.log(s1) -
           0.5 * np.log(2 * np.pi) - lt_val).mean()

print(f"tll  unconditional-lognormal: {tll_uncond:8.4f}")
print(f"tll  AR(1)-lognormal:         {tll_ar1:8.4f}")
print(f"tll  references: Poisson 0.5126 | ETAS 1.4343 | NN best ~0.91")

# --- spatial: KDE over last-K locations ---------------------------------
AREA = (x.max() - x.min()) * (y.max() - y.min())  # crude bounding box, km^2
K = 32

def kde_sll(idx, sigma, w_bg):
    # mixture: w_bg uniform + (1-w_bg) * mean of K Gaussians at last-K coords
    tgt_x, tgt_y = x[idx], y[idx]
    comps = np.zeros((len(idx), K))
    for k in range(1, K + 1):
        dx = tgt_x - x[idx - k]
        dy = tgt_y - y[idx - k]
        comps[:, k - 1] = -0.5 * (dx**2 + dy**2) / sigma**2 - np.log(
            2 * np.pi * sigma**2)
    m = comps.max(axis=1)
    mix = np.exp(m) * np.exp(comps - m[:, None]).mean(axis=1)
    dens = w_bg / AREA + (1 - w_bg) * mix
    return np.log(dens).mean()

# fit on a train subsample
fit_idx = train_idx[train_idx > K][-20000:]
best = (None, None, -np.inf)
for sigma in [1, 2, 3, 5, 8, 12, 20, 35]:
    for w_bg in [0.05, 0.1, 0.2, 0.35, 0.5]:
        v = kde_sll(fit_idx, sigma, w_bg)
        if v > best[2]:
            best = (sigma, w_bg, v)
sigma, w_bg, fit_v = best
val_v = kde_sll(val_idx, sigma, w_bg)
print(f"\nsll  last-{K} KDE (sigma={sigma} km, bg={w_bg}): train {fit_v:8.4f} | val {val_v:8.4f}")
print(f"sll  references: Poisson -13.7745 | ETAS -8.6898 | NN stuck -13.8")

# magnitude baseline: GR / exponential fit
m_all = cat.raw[:, 3].numpy().astype(np.float64)
beta = 1.0 / (m_all[train_idx] - 2.5 + 0.05).mean()
mv = m_all[val_idx] - 2.5 + 0.05
mll_exp = (np.log(beta) - beta * mv).mean()
print(f"\nmll  exponential(GR beta={beta:.3f}): val {mll_exp:8.4f} | NN ~-0.50")
