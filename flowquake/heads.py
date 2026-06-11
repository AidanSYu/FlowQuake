"""Structured closed-form heads: kernel-mixture spatial, conditional-GR mag.

The spatial head is a Hawkes-style triggering kernel: a Gaussian mixture whose
components sit AT the last K observed event locations (so at eval time the
mass moves with the data — nothing about geography is baked into weights),
plus a uniform background over the region. Per-component mixture logits and
bandwidths come from a small MLP fed [h, component recency, magnitude,
distance-to-current] — i.e. the inputs of an ETAS kernel. Log-likelihood is
closed form in km units: sll = log f(x, y) in log(1/km^2).

The magnitude head is a conditional Gutenberg-Richter exponential with
rate beta(h): mll = log beta - beta * (m - mc), in log(1/mag-unit).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


class KernelMixtureHead(nn.Module):
    """Mixture of ETAS-style power-law kernels at recent event locations.

    Per-component radial density (per km^2), the form ETAS uses for
    aftershock spatial decay (heavy-tailed, unlike a Gaussian):
        f(r) = (q - 1) / (pi d^2) * (1 + r^2 / d^2)^(-q)
    with per-component scale d and exponent q from a small MLP fed
    [cond, log_dt, mag, log_dist]. Background = mixture of a uniform
    component and (optionally) a train-era seismicity KDE map.
    """

    def __init__(
        self,
        cond_dim: int,
        n_comp: int,
        hidden: int = 64,
        d_floor_km: float = 0.25,
        d_init_km: float = 2.5,
        q_floor: float = 1.15,
        q_init: float = 1.8,
        bg_frac_init: float = 0.35,
    ):
        super().__init__()
        self.n_comp = n_comp
        self.d_floor = d_floor_km
        self.q_floor = q_floor
        self.h_proj = nn.Linear(cond_dim, hidden)
        # per-component scorer -> (logit, d_raw, q_raw)
        self.comp_mlp = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.SiLU(), nn.Linear(hidden, 3)
        )
        self.bg_logit = nn.Linear(cond_dim, 2)  # [uniform, kde-map]
        with torch.no_grad():
            self.comp_mlp[-1].weight.mul_(0.1)
            self.comp_mlp[-1].bias[:] = torch.tensor([
                0.0,
                math.log(math.exp(d_init_km - d_floor_km) - 1.0),
                math.log(math.exp(q_init - q_floor) - 1.0),
            ])
            self.bg_logit.weight.mul_(0.0)
            # each bg part starts at bg_frac/2 vs n_comp equal components
            self.bg_logit.bias.fill_(
                math.log(bg_frac_init / 2 / (1 - bg_frac_init)) + math.log(n_comp)
            )

    def _params(self, cond: torch.Tensor, comp_feats: torch.Tensor):
        """Returns log-weights (B, K+2) [comps..., uniform, kde], d (B,K), q (B,K)."""
        hp = F.silu(self.h_proj(cond)).unsqueeze(1).expand(-1, self.n_comp, -1)
        out = self.comp_mlp(torch.cat([hp, comp_feats], dim=-1))  # (B, K, 3)
        logits = torch.cat([out[..., 0], self.bg_logit(cond)], dim=-1)
        log_w = F.log_softmax(logits, dim=-1)
        d = self.d_floor + F.softplus(out[..., 1])
        q = self.q_floor + F.softplus(out[..., 2])
        return log_w, d, q

    def log_prob(self, s_km: torch.Tensor, comp_xy: torch.Tensor,
                 comp_feats: torch.Tensor, cond: torch.Tensor, bg_area: float,
                 log_kde_at_s: torch.Tensor | None = None):
        """s_km: (B, 2); comp_xy: (B, K, 2) centers (km); returns (B,) log f."""
        log_w, d, q = self._params(cond, comp_feats)
        r2 = (s_km.unsqueeze(1) - comp_xy).pow(2).sum(-1)          # (B, K)
        log_comp = (torch.log(q - 1.0) - math.log(math.pi) - 2.0 * torch.log(d)
                    - q * torch.log1p(r2 / d.pow(2)))
        log_uni = torch.full_like(log_w[:, :1], -math.log(bg_area))
        if log_kde_at_s is None:
            log_kde = torch.full_like(log_w[:, :1], -1e30)  # disabled
        else:
            log_kde = log_kde_at_s.unsqueeze(-1)
        return torch.logsumexp(
            log_w + torch.cat([log_comp, log_uni, log_kde], dim=-1), dim=-1
        )

    @torch.no_grad()
    def sample(self, comp_xy: torch.Tensor, comp_feats: torch.Tensor,
               cond: torch.Tensor, bg_box: tuple[float, float, float, float],
               kde_sampler=None):
        """Returns (B, 2) km. kde_sampler(n) -> (n, 2) km draws from the map."""
        log_w, d, q = self._params(cond, comp_feats)
        w = log_w.exp()
        if kde_sampler is None:
            w[:, -1] = 0.0  # no kde map available
        choice = torch.multinomial(w, 1).squeeze(-1)               # (B,)
        idx = choice.clamp(max=self.n_comp - 1)
        centers = comp_xy.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        dk = d.gather(1, idx.view(-1, 1)).squeeze(1)
        qk = q.gather(1, idx.view(-1, 1)).squeeze(1)
        # inverse-CDF radial draw: F(r) = 1 - (1 + r^2/d^2)^(1-q)
        u = torch.rand_like(dk).clamp(1e-9, 1 - 1e-9)
        r = dk * torch.sqrt(torch.pow(1.0 - u, 1.0 / (1.0 - qk)) - 1.0)
        ang = 2.0 * math.pi * torch.rand_like(dk)
        out = centers + torch.stack([r * ang.cos(), r * ang.sin()], dim=-1)

        xmin, ymin, xmax, ymax = bg_box
        uu = torch.rand(out.shape, device=out.device)
        uni_pt = torch.stack(
            [xmin + uu[:, 0] * (xmax - xmin), ymin + uu[:, 1] * (ymax - ymin)], dim=-1
        )
        out = torch.where((choice == self.n_comp).unsqueeze(-1), uni_pt, out)
        if kde_sampler is not None:
            is_kde = choice == self.n_comp + 1
            if is_kde.any():
                out[is_kde] = kde_sampler(int(is_kde.sum()))
        return out


class GRMagnitudeHead(nn.Module):
    """m - mc ~ Exponential(beta(cond)); closed-form, GR-faithful."""

    def __init__(self, cond_dim: int, beta_init: float = 2.0):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 1)
        with torch.no_grad():
            self.lin.weight.mul_(0.0)
            self.lin.bias.fill_(math.log(math.exp(beta_init) - 1.0))

    def beta(self, cond: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.lin(cond)).squeeze(-1) + 1e-3

    def log_prob(self, m: torch.Tensor, cond: torch.Tensor, mc: float) -> torch.Tensor:
        """m: (B,) raw magnitudes >= mc. Half-bin shift handles discretization."""
        beta = self.beta(cond)
        dm = torch.clamp(m - mc, min=0.0) + 0.005
        return torch.log(beta) - beta * dm

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, mc: float) -> torch.Tensor:
        beta = self.beta(cond)
        u = torch.rand_like(beta).clamp_min(1e-12)
        return mc - torch.log(u) / beta
