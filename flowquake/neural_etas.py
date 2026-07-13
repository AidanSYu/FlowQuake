"""Neural-ETAS spatial head: a strict superset of the benchmark ETAS spatial
density.

    f(s_i | H_i) = [ bg(s_i) + alpha * far_num_i + sum_{j in near} w'_j K'_j(s_i) ]
                   / [ mu' A + alpha * far_den_i + sum_{j in near} w'_j Z'_j ]

  * far_num/far_den: frozen full-history ETAS sums minus the near set
    (precomputed; see scripts/precompute_trigger_features.py).
  * near set: per-parent neural modulations of the ETAS weight and kernel
    shape, driven only by (m_j, dt_ij) -- never by the target location, so
    the density stays properly normalized (Z' has closed form).
  * bg: learnable mixture of the uniform background and causal multi-scale
    KDE maps of prior seismicity (ETAS's background is uniform).

With the KDE gate closed: modulations = 0, alpha = 1, bg = uniform mu, so the
head reproduces the benchmark ETAS spatial likelihood up to float precision.
The default training init opens a small KDE gate (~5%) to keep background-map
gradients alive; reported gains are always measured against the package ETAS
scores, not against this near-ETAS initialization.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralETASSpatialHead(nn.Module):
    def __init__(self, params: dict, n_kde: int, hidden: int = 32, use_mlp: bool = True,
                 kde_gate_init: float = -2.94, refit_globals: bool = False):
        super().__init__()
        for k, v in params.items():   # mu,k0,c,tau,d,a,omega,gamma,rho,mc,area
            self.register_buffer(k, torch.tensor(float(v)), persistent=False)
        self.use_mlp = use_mlp
        # optional flETAS-style classical control: refit the global kernel
        # parameters by SGD (log-offsets from the inversion, init 0) with the
        # learned background but NO neural modulation. NOTE: this reweights only
        # the NEAR-set kernels exactly; the far field keeps inversion shapes
        # scaled by alpha (a conservative lower bound on a full refit).
        self.refit_globals = refit_globals
        self.g_off = nn.Parameter(torch.zeros(4)) if refit_globals else None  # [a, log d, rho, log c]
        # background = (1-sigmoid(gate)) * uniform + sigmoid(gate) * softmax-mix
        # of KDE maps. gate init -2.94 -> 5% KDE mass: near-ETAS at init while
        # keeping usable gradients (a softmax with a +16 "exact" logit starves
        # the KDE components -- observed: weights frozen at [1,0,...]).
        self.kde_gate = nn.Parameter(torch.tensor(kde_gate_init))
        self.kde_logits = nn.Parameter(torch.zeros(n_kde))
        self.log_mu_adj = nn.Parameter(torch.zeros(()))
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 3))
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, far_num, far_den, kde, r2, dt, mag, valid):
        """All (B,) or (B,P)/(B,K). Returns (B,) log f(s_i)."""
        feats = torch.stack([(mag - self.mc) / 2.0,
                             (torch.log(dt + 1e-3) - 2.0) / 3.0], dim=-1)
        if self.use_mlp:
            mods = self.mlp(feats)
            dlogw = mods[..., 0].clamp(-4.0, 4.0)
            dlogd = mods[..., 1].clamp(-3.0, 3.0)
            drho = mods[..., 2].clamp(-1.5, 1.5)
        elif self.refit_globals:
            da, dld, drho_g, dlc = (o.clamp(-1.5, 1.5) for o in self.g_off)
            # exact refits of a and c on the near set; d/rho offsets applied below
            dlogw = (da * (mag - self.mc)
                     + (1.0 + self.omega) * (torch.log(dt + self.c)
                                             - torch.log(dt + self.c * torch.exp(dlc))))
            dlogd = dld.expand_as(mag)
            drho = drho_g.expand_as(mag)
        else:
            dlogw = dlogd = drho = torch.zeros_like(mag)
        w = (self.k0 * torch.exp(self.a * (mag - self.mc))
             * torch.exp(-dt / self.tau)
             * torch.pow(dt + self.c, -(1.0 + self.omega))
             * torch.exp(dlogw)) * valid
        dmj = self.d * torch.exp(self.gamma * (mag - self.mc)) * torch.exp(dlogd)
        rhoj = (self.rho * torch.exp(drho)).clamp(0.05, 5.0)
        kj = torch.pow(r2 + dmj, -(1.0 + rhoj))
        zj = math.pi / (rhoj * torch.pow(dmj, rhoj))
        near_num = (w * kj).sum(-1)
        near_den = (w * zj).sum(-1)

        gate = torch.sigmoid(self.kde_gate)
        kde_w = F.softmax(self.kde_logits, dim=0)
        mu_eff = self.mu * torch.exp(self.log_mu_adj)
        bg_num = mu_eff * self.area * ((1.0 - gate) / self.area + gate * (kde_w * kde).sum(-1))
        alpha = torch.exp(self.log_alpha)
        num = bg_num + alpha * far_num + near_num
        den = mu_eff * self.area + alpha * far_den + near_den
        return torch.log(num.clamp_min(1e-30)) - torch.log(den.clamp_min(1e-30))
