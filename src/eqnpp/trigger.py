"""Learned triggering-kernel spatial head — the mechanism built to beat ETAS on SLL.

Why summary-state heads lose: ETAS's spatial intensity is a *sum of kernels centered on
every past epicenter* (aftershocks cluster near recent events).  An autoregressive encoder
that compresses the whole catalog into one vector h throws away the precise recent
locations that predict the next one.  Empirically a 16-component MDN on h gets SLL ~ -11.5
on SCEDC_30 vs ETAS's -7.6.

This head keeps the recent epicenters explicit.  Given the K most recent events (their
encoder states h_j, standardized locations z_j, and ages a_j), it models

    p(z | history) = sum_b pi_b  N(z; mu_b, Sigma_b)        # learned background field
                   + sum_j pi_j  N(z; z_j + d_j, Sigma_j)   # learned triggering kernels

where every kernel's offset d_j, anisotropic covariance Sigma_j (Cholesky), and weight
pi_j are predicted from that event's state h_j and log-age a_j — a learned, anisotropic,
recency/productivity-weighted version of ETAS's spatial triggering term.  Exact log-density
(logsumexp over background + K kernels); samples by picking a component.

Operates in standardized z-space; the model adds the constant standardizer log-Jacobian to
report physical density (1/km^2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)
NEG_INF = -1e9


def _mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


@dataclass
class SpatialContext:
    """Everything a spatial head may need for a set of M scored positions.

    Simple heads (gaussian/mdn/realnvp/fm) use only `h_self`.  The triggering head also
    uses the recent-event window.
      h_self        [M, H]        encoder state h_{p-1} that conditions target p
      recent_h      [M, K, H]     states of the K most recent events (events p-K..p-1)
      recent_z      [M, K, 2]     their standardized locations
      recent_logage [M, K]        log(days since each recent event, +eps)
      mask          [M, K] bool   True where a recent slot is a real event
    """
    h_self: torch.Tensor
    recent_h: Optional[torch.Tensor] = None
    recent_z: Optional[torch.Tensor] = None
    recent_logage: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


def _full_gauss_logpdf(z, mu, l11, l22, l21):
    """log N(z; mu, LL^T) for lower-tri Cholesky L=[[l11,0],[l21,l22]]. Broadcasts over
    trailing component axis. z,mu: [...,2]; l*: [...]."""
    d = z - mu
    u0 = d[..., 0] / l11
    u1 = (d[..., 1] - l21 * u0) / l22
    quad = u0 ** 2 + u1 ** 2
    logdet = 2 * (torch.log(l11) + torch.log(l22))
    return -0.5 * (quad + logdet + 2 * LOG_2PI)


class TriggeringKernelHead(nn.Module):
    def __init__(self, cond_dim: int, n_bg: int = 8, hidden: int = 128,
                 max_log_scale: float = 3.0, min_scale: float = 1e-3):
        super().__init__()
        self.H = cond_dim
        self.n_bg = n_bg
        self.max_log_scale = max_log_scale
        self.min_scale = min_scale
        # background field: MDN over z conditioned on h_self -> per-comp (logit, mean(2), chol(3))
        self.bg_net = _mlp([cond_dim, hidden, hidden, n_bg * 6])
        # per recent-event kernel params from (h_j, logage_j): (logit, offset(2), chol(3)) = 6
        self.kern_net = _mlp([cond_dim + 1, hidden, hidden, 6])

    # ---------- parameter producers ----------
    def _bg_params(self, h_self):
        B = h_self.shape[0]
        out = self.bg_net(h_self).view(B, self.n_bg, 6)
        logit = out[..., 0]                                    # [B,nb]
        mu = out[..., 1:3]                                     # [B,nb,2]
        l11 = self._scale(out[..., 3]); l22 = self._scale(out[..., 4]); l21 = out[..., 5]
        return logit, mu, l11, l22, l21

    def _kern_params(self, recent_h, recent_z, recent_logage):
        inp = torch.cat([recent_h, recent_logage.unsqueeze(-1)], dim=-1)  # [M,K,H+1]
        out = self.kern_net(inp)                               # [M,K,6]
        logit = out[..., 0]
        offset = torch.tanh(out[..., 1:3]) * 1.0               # bounded offset in z-units
        mu = recent_z + offset
        l11 = self._scale(out[..., 3]); l22 = self._scale(out[..., 4]); l21 = out[..., 5]
        return logit, mu, l11, l22, l21

    def _scale(self, raw):
        return F.softplus(raw.clamp(max=self.max_log_scale)) + self.min_scale

    # ---------- core: all component log-weights and log-densities ----------
    def _components(self, sc: SpatialContext):
        h_self = sc.h_self
        bg_logit, bg_mu, bl11, bl22, bl21 = self._bg_params(h_self)         # [M,nb],...
        k_logit, k_mu, kl11, kl22, kl21 = self._kern_params(sc.recent_h, sc.recent_z,
                                                            sc.recent_logage)  # [M,K],...
        k_logit = k_logit.masked_fill(~sc.mask, NEG_INF)
        logits = torch.cat([bg_logit, k_logit], dim=-1)                     # [M, nb+K]
        log_w = torch.log_softmax(logits, dim=-1)
        return (log_w, bg_mu, bl11, bl22, bl21, k_mu, kl11, kl22, kl21)

    def log_prob(self, z, sc: SpatialContext):
        log_w, bg_mu, bl11, bl22, bl21, k_mu, kl11, kl22, kl21 = self._components(sc)
        zb = z.unsqueeze(1)                                                  # [M,1,2]
        bg_lp = _full_gauss_logpdf(zb, bg_mu, bl11, bl22, bl21)              # [M,nb]
        k_lp = _full_gauss_logpdf(zb, k_mu, kl11, kl22, kl21)               # [M,K]
        comp_lp = torch.cat([bg_lp, k_lp], dim=-1)                           # [M,nb+K]
        return torch.logsumexp(log_w + comp_lp, dim=-1)

    def train_loss(self, z, sc: SpatialContext):
        return -self.log_prob(z, sc)

    @torch.no_grad()
    def sample(self, sc: SpatialContext, n: int = 1):
        log_w, bg_mu, bl11, bl22, bl21, k_mu, kl11, kl22, kl21 = self._components(sc)
        mu = torch.cat([bg_mu, k_mu], dim=1)                                # [M, C, 2]
        l11 = torch.cat([bl11, kl11], dim=1); l22 = torch.cat([bl22, kl22], dim=1)
        l21 = torch.cat([bl21, kl21], dim=1)
        w = torch.exp(log_w)                                                # [M,C]
        M = w.shape[0]
        idx = torch.multinomial(w, n, replacement=True)                     # [M,n]
        mi = torch.arange(M, device=w.device).unsqueeze(1).expand(-1, n)
        mu_s = mu[mi, idx]; l11s = l11[mi, idx]; l22s = l22[mi, idx]; l21s = l21[mi, idx]
        eps = torch.randn(M, n, 2, device=w.device)
        x = mu_s[..., 0] + l11s * eps[..., 0]
        y = mu_s[..., 1] + l21s * eps[..., 0] + l22s * eps[..., 1]
        return torch.stack([x, y], dim=-1)
