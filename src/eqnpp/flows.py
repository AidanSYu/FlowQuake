"""Spatial density heads over (x, y) — the lever that wins the SLL battle vs ETAS.

All heads operate in *standardized* coordinate space z (the model adds the constant
standardizer log-Jacobian to report physical density in 1/km^2).  Every head exposes:

    log_prob(z, c) -> [B]        exact spatial log-density (the eval metric)
    sample(c, n)    -> [B(,n),2] draw locations (for Track-B catalog simulation)
    train_loss(z,c) -> [B]       per-sample objective to MINIMIZE

For likelihood heads train_loss = -log_prob.  For the flow-matching CNF, train_loss is
the conditional-flow-matching velocity-regression loss (the SEED's mechanism), while
log_prob is obtained by integrating the instantaneous change-of-variables.

Ablation ladder (same encoder, swap the head) isolates the density lever:
    Gaussian  ->  GaussianMixture (MDN)  ->  RealNVP  ->  FlowMatchingCNF
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)


def _mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def std_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    # z: [..., d] -> [...]
    d = z.shape[-1]
    return -0.5 * (z.pow(2).sum(-1) + d * LOG_2PI)


# --------------------------------------------------------------------------------------
# 1. Diagonal Gaussian (reproduces the reference dummy's spatial head)
# --------------------------------------------------------------------------------------


class GaussianHead(nn.Module):
    def __init__(self, cond_dim: int, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.net = _mlp([cond_dim, hidden, hidden, 2 * dim])

    def _params(self, c):
        out = self.net(c)
        mu, log_std = out.chunk(2, dim=-1)
        return mu, log_std.clamp(-7, 7)

    def log_prob(self, z, c):
        mu, log_std = self._params(c)
        var = torch.exp(2 * log_std)
        return (-0.5 * ((z - mu) ** 2 / var + 2 * log_std + LOG_2PI)).sum(-1)

    def train_loss(self, z, c):
        return -self.log_prob(z, c)

    @torch.no_grad()
    def sample(self, c, n: int = 1):
        mu, log_std = self._params(c)
        std = torch.exp(log_std)
        eps = torch.randn(c.shape[0], n, self.dim, device=c.device)
        return mu.unsqueeze(1) + std.unsqueeze(1) * eps


# --------------------------------------------------------------------------------------
# 2. Gaussian mixture (MDN) — full 2x2 covariance per component via Cholesky
# --------------------------------------------------------------------------------------


class GaussianMixtureHead(nn.Module):
    def __init__(self, cond_dim: int, dim: int = 2, n_comp: int = 16, hidden: int = 128):
        super().__init__()
        assert dim == 2, "full-covariance MDN implemented for 2D"
        self.dim = dim
        self.k = n_comp
        # per component: logit(1) + mean(2) + L params (l11,l22 via softplus, l21) = 1+2+3
        self.net = _mlp([cond_dim, hidden, hidden, n_comp * 6])

    def _params(self, c):
        B = c.shape[0]
        out = self.net(c).view(B, self.k, 6)
        logit = out[..., 0]
        mu = out[..., 1:3]
        l11 = F.softplus(out[..., 3]) + 1e-4
        l22 = F.softplus(out[..., 4]) + 1e-4
        l21 = out[..., 5]
        return logit, mu, l11, l22, l21

    def log_prob(self, z, c):
        logit, mu, l11, l22, l21 = self._params(c)
        log_w = torch.log_softmax(logit, dim=-1)             # [B,K]
        d = z.unsqueeze(1) - mu                              # [B,K,2]
        # precision = (L L^T)^{-1}; we have L lower-tri (Cholesky of covariance).
        # Solve L u = d  -> u; quadratic = ||u||^2 ; logdet(cov) = 2 log(l11 l22)
        u0 = d[..., 0] / l11
        u1 = (d[..., 1] - l21 * u0) / l22
        quad = u0 ** 2 + u1 ** 2
        logdet = 2 * (torch.log(l11) + torch.log(l22))
        comp_logprob = -0.5 * (quad + logdet + self.dim * LOG_2PI)
        return torch.logsumexp(log_w + comp_logprob, dim=-1)

    def train_loss(self, z, c):
        return -self.log_prob(z, c)

    @torch.no_grad()
    def sample(self, c, n: int = 1):
        logit, mu, l11, l22, l21 = self._params(c)
        w = torch.softmax(logit, dim=-1)
        idx = torch.multinomial(w, n, replacement=True)       # [B,n]
        B = c.shape[0]
        bi = torch.arange(B, device=c.device).unsqueeze(1).expand(-1, n)
        mu_s = mu[bi, idx]                                    # [B,n,2]
        l11s, l22s, l21s = l11[bi, idx], l22[bi, idx], l21[bi, idx]
        eps = torch.randn(B, n, 2, device=c.device)
        x = mu_s[..., 0] + l11s * eps[..., 0]
        y = mu_s[..., 1] + l21s * eps[..., 0] + l22s * eps[..., 1]
        return torch.stack([x, y], dim=-1)


# --------------------------------------------------------------------------------------
# 3. Conditional RealNVP (affine coupling) — exact likelihood normalizing flow
# --------------------------------------------------------------------------------------


class _AffineCoupling(nn.Module):
    def __init__(self, cond_dim, dim=2, hidden=128, transform_second=True):
        super().__init__()
        self.transform_second = transform_second  # which coordinate is transformed
        self.net = _mlp([1 + cond_dim, hidden, hidden, 2])  # -> log_s, t

    def forward(self, z, c):
        # z: [B,2]; transform one coord conditioned on the other + c
        a, b = (z[..., 0:1], z[..., 1:2]) if self.transform_second else (z[..., 1:2], z[..., 0:1])
        log_s, t = self.net(torch.cat([a, c], -1)).chunk(2, -1)
        log_s = torch.tanh(log_s) * 2.0
        b2 = b * torch.exp(log_s) + t
        out = torch.cat([a, b2], -1) if self.transform_second else torch.cat([b2, a], -1)
        return out, log_s.squeeze(-1)

    def inverse(self, z, c):
        a, b2 = (z[..., 0:1], z[..., 1:2]) if self.transform_second else (z[..., 1:2], z[..., 0:1])
        log_s, t = self.net(torch.cat([a, c], -1)).chunk(2, -1)
        log_s = torch.tanh(log_s) * 2.0
        b = (b2 - t) * torch.exp(-log_s)
        out = torch.cat([a, b], -1) if self.transform_second else torch.cat([b, a], -1)
        return out


class RealNVPHead(nn.Module):
    def __init__(self, cond_dim: int, dim: int = 2, n_layers: int = 8, hidden: int = 128):
        super().__init__()
        assert dim == 2
        self.dim = dim
        self.layers = nn.ModuleList([
            _AffineCoupling(cond_dim, dim, hidden, transform_second=(i % 2 == 0))
            for i in range(n_layers)
        ])

    def log_prob(self, z, c):
        ldj = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            z, ls = layer(z, c)
            ldj = ldj + ls
        return std_normal_logprob(z) + ldj  # forward maps data->base; base logprob + ldj

    def train_loss(self, z, c):
        return -self.log_prob(z, c)

    @torch.no_grad()
    def sample(self, c, n: int = 1):
        B = c.shape[0]
        z = torch.randn(B, n, self.dim, device=c.device)
        cc = c.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)
        z = z.reshape(B * n, self.dim)
        for layer in reversed(self.layers):
            z = layer.inverse(z, cc)
        return z.reshape(B, n, self.dim)


# --------------------------------------------------------------------------------------
# 4. Flow-matching conditional CNF — the SEED's headline mechanism
# --------------------------------------------------------------------------------------


class _VelocityField(nn.Module):
    """v_theta(t, z, c) -> R^dim.  t in [0,1] fed as a scalar feature."""

    def __init__(self, cond_dim, dim=2, hidden=128):
        super().__init__()
        self.net = _mlp([dim + 1 + cond_dim, hidden, hidden, hidden, dim])

    def forward(self, t, z, c):
        if t.dim() == 0:
            t = t.expand(z.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([z, t, c], dim=-1))


class FlowMatchingCNF(nn.Module):
    """Conditional flow matching (Lipman et al. 2023, OT path) for the spatial density.

    Training: regress v_theta to the straight-line target velocity (z1 - z0).
    Density:  integrate dz/dt=v with the trace term backward from data (t=1) to base (t=0).
    Sampling: integrate dz/dt=v forward from base (t=0) to data (t=1).
    """

    def __init__(self, cond_dim: int, dim: int = 2, hidden: int = 128,
                 sigma_min: float = 1e-4, n_steps: int = 40):
        super().__init__()
        self.dim = dim
        self.sigma_min = sigma_min
        self.n_steps = n_steps
        self.v = _VelocityField(cond_dim, dim, hidden)

    def train_loss(self, z1, c):
        B = z1.shape[0]
        t = torch.rand(B, device=z1.device)
        z0 = torch.randn_like(z1)
        zt = (1 - (1 - self.sigma_min) * t).unsqueeze(-1) * z0 + t.unsqueeze(-1) * z1
        target = z1 - (1 - self.sigma_min) * z0
        pred = self.v(t, zt, c)
        return ((pred - target) ** 2).sum(-1)  # [B]

    # --- ODE integrators (fixed-step RK4; no torchdiffeq dependency) ---

    def _divergence(self, t, z, c):
        # exact 2D divergence = dvx/dzx + dvy/dzy
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            v = self.v(t, z, c)
            div = 0.0
            for i in range(self.dim):
                gi = torch.autograd.grad(v[:, i].sum(), z, create_graph=False, retain_graph=True)[0]
                div = div + gi[:, i]
        return v.detach(), div.detach()

    def log_prob(self, z1, c):
        """Integrate (z, logdet) from t=1 -> t=0.  log p1(x) = logN(z0) + D, where
        D = ∫_1^0 div v dt accumulated while stepping t downward."""
        B = z1.shape[0]
        z = z1
        D = torch.zeros(B, device=z1.device)
        steps = self.n_steps
        ts = torch.linspace(1.0, 0.0, steps + 1, device=z1.device)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            h = t1 - t0  # negative
            # RK4 on augmented state s=(z, a) with ds/dt = (v, div v)
            v1, d1 = self._divergence(t0, z, c)
            v2, d2 = self._divergence(t0 + 0.5 * h, z + 0.5 * h * v1, c)
            v3, d3 = self._divergence(t0 + 0.5 * h, z + 0.5 * h * v2, c)
            v4, d4 = self._divergence(t0 + h, z + h * v3, c)
            z = z + (h / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
            D = D + (h / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
        z0 = z
        return std_normal_logprob(z0) + D

    @torch.no_grad()
    def sample(self, c, n: int = 1):
        B = c.shape[0]
        cc = c.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)
        z = torch.randn(B * n, self.dim, device=c.device)
        steps = self.n_steps
        ts = torch.linspace(0.0, 1.0, steps + 1, device=c.device)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            h = t1 - t0
            v1 = self.v(t0, z, cc)
            v2 = self.v(t0 + 0.5 * h, z + 0.5 * h * v1, cc)
            v3 = self.v(t0 + 0.5 * h, z + 0.5 * h * v2, cc)
            v4 = self.v(t0 + h, z + h * v3, cc)
            z = z + (h / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        return z.reshape(B, n, self.dim)


def build_spatial_head(kind: str, cond_dim: int, **kw) -> nn.Module:
    kind = kind.lower()
    if kind == "gaussian":
        return GaussianHead(cond_dim, **kw)
    if kind in ("mdn", "gmm", "mixture"):
        return GaussianMixtureHead(cond_dim, **kw)
    if kind in ("realnvp", "nvp", "flow"):
        return RealNVPHead(cond_dim, **kw)
    if kind in ("fm", "cnf", "flowmatching", "flow_matching"):
        return FlowMatchingCNF(cond_dim, **kw)
    raise ValueError(f"unknown spatial head {kind!r}")
