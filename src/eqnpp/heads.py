"""Temporal and magnitude (mark) density heads.

Temporal head: a mixture of log-normals over the inter-event time Δt (Shchur et al.,
"Intensity-Free Learning of Temporal Point Processes").  For an inter-event-time density
f(τ|h), the per-event temporal log-likelihood is exactly

    TLL = log f(τ_i | h_i) = log λ(t_i) - ∫_0^{τ_i} λ(t) dt,

which is the benchmark's TLL definition — so no numerical survival integral is needed and
the metric is directly comparable to ETAS (ComCat_25: TLL = 1.434).

Magnitude head: Gutenberg-Richter is an exponential law for (m - Mc); we model it with a
small (optionally conditional) exponential so Track-B catalogs carry magnitudes for the
pyCSEP magnitude test.  Not scored in Track-A TLL/SLL.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

LOG_2PI = math.log(2 * math.pi)


def _mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class LogNormalMixtureTime(nn.Module):
    """f(τ | h) = sum_k w_k LogNormal(τ; μ_k, σ_k).  TLL = log f(τ)."""

    def __init__(self, cond_dim: int, n_comp: int = 16, hidden: int = 128, t_floor: float = 1e-6):
        super().__init__()
        self.k = n_comp
        self.t_floor = t_floor
        self.net = _mlp([cond_dim, hidden, hidden, 3 * n_comp])

    def _params(self, c):
        out = self.net(c).view(c.shape[0], self.k, 3)
        log_w = torch.log_softmax(out[..., 0], dim=-1)     # [B,K]
        mu = out[..., 1]                                   # mean of log τ
        log_sigma = out[..., 2].clamp(-7, 4)
        return log_w, mu, log_sigma

    def log_prob(self, tau, c):
        tau = tau.clamp_min(self.t_floor)
        log_w, mu, log_sigma = self._params(c)
        log_tau = torch.log(tau).unsqueeze(-1)             # [B,1]
        sigma = torch.exp(log_sigma)
        comp = (-log_tau - log_sigma - 0.5 * LOG_2PI
                - 0.5 * ((log_tau - mu) / sigma) ** 2)      # [B,K] lognormal logpdf
        return torch.logsumexp(log_w + comp, dim=-1)        # [B]

    def train_loss(self, tau, c):
        return -self.log_prob(tau, c)

    @torch.no_grad()
    def sample(self, c, n: int = 1):
        log_w, mu, log_sigma = self._params(c)
        w = torch.exp(log_w)
        idx = torch.multinomial(w, n, replacement=True)     # [B,n]
        B = c.shape[0]
        bi = torch.arange(B, device=c.device).unsqueeze(1).expand(-1, n)
        mu_s, sig_s = mu[bi, idx], torch.exp(log_sigma[bi, idx])
        log_tau = mu_s + sig_s * torch.randn(B, n, device=c.device)
        return torch.exp(log_tau).clamp_min(self.t_floor)   # [B,n] days


class GutenbergRichterHead(nn.Module):
    """(m - Mc) ~ Exponential(beta), beta = b*ln10.  Optionally conditioned on history.

    Returns log-density of m and can sample magnitudes for Track-B simulation.
    """

    def __init__(self, cond_dim: int = 0, conditional: bool = False, hidden: int = 64):
        super().__init__()
        self.conditional = conditional
        if conditional:
            self.net = _mlp([cond_dim, hidden, 1])
        else:
            # global log-beta parameter; init near b=1 -> beta=ln10
            self.log_beta = nn.Parameter(torch.tensor(math.log(math.log(10.0))))

    def _beta(self, c):
        if self.conditional:
            return torch.nn.functional.softplus(self.net(c).squeeze(-1)) + 1e-3
        return torch.exp(self.log_beta).expand(c.shape[0])

    def log_prob(self, dm, c):
        # dm = m - Mc >= 0 ; Exponential: log beta - beta*dm
        beta = self._beta(c)
        dm = dm.clamp_min(0.0)
        return torch.log(beta) - beta * dm

    def train_loss(self, dm, c):
        return -self.log_prob(dm, c)

    @torch.no_grad()
    def sample(self, c, n: int = 1):
        beta = self._beta(c).unsqueeze(-1)                  # [B,1]
        u = torch.rand(c.shape[0], n, device=c.device).clamp_(1e-9, 1 - 1e-9)
        return -torch.log1p(-u) / beta                      # [B,n] = m - Mc
