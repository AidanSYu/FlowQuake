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
    def __init__(
        self,
        cond_dim: int,
        n_comp: int,
        hidden: int = 64,
        sigma_floor_km: float = 0.3,
        sigma_init_km: float = 3.0,
        bg_frac_init: float = 0.35,
    ):
        super().__init__()
        self.n_comp = n_comp
        self.sigma_floor = sigma_floor_km
        self.h_proj = nn.Linear(cond_dim, hidden)
        # per-component scorer: [h_proj, log_dt_n, mag_n, log_dist_n] -> (logit, log_sigma_raw)
        self.comp_mlp = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.SiLU(), nn.Linear(hidden, 2)
        )
        self.bg_logit = nn.Linear(cond_dim, 1)
        # init: sigma ~ sigma_init, background fraction ~ bg_frac_init
        with torch.no_grad():
            self.comp_mlp[-1].weight.mul_(0.1)
            self.comp_mlp[-1].bias[:] = torch.tensor(
                [0.0, math.log(math.exp(sigma_init_km - sigma_floor_km) - 1.0)]
            )
            self.bg_logit.weight.mul_(0.0)
            # bg competes against n_comp equal components: logit = log(f/(1-f)) + log K
            self.bg_logit.bias.fill_(
                math.log(bg_frac_init / (1 - bg_frac_init)) + math.log(n_comp)
            )

    def _params(self, cond: torch.Tensor, comp_feats: torch.Tensor):
        """cond: (B, C); comp_feats: (B, K, 3) normalized [log_dt, mag, log_dist].

        Returns mixture log-weights (B, K+1) (last = background) and sigmas (B, K).
        """
        hp = F.silu(self.h_proj(cond)).unsqueeze(1).expand(-1, self.n_comp, -1)
        out = self.comp_mlp(torch.cat([hp, comp_feats], dim=-1))  # (B, K, 2)
        logits = torch.cat([out[..., 0], self.bg_logit(cond)], dim=-1)  # (B, K+1)
        log_w = F.log_softmax(logits, dim=-1)
        sigma = self.sigma_floor + F.softplus(out[..., 1])
        return log_w, sigma

    def log_prob(self, s_km: torch.Tensor, comp_xy: torch.Tensor,
                 comp_feats: torch.Tensor, cond: torch.Tensor, bg_area: float):
        """s_km: (B, 2) target; comp_xy: (B, K, 2) component centers (km)."""
        log_w, sigma = self._params(cond, comp_feats)
        d2 = (s_km.unsqueeze(1) - comp_xy).pow(2).sum(-1)          # (B, K)
        log_comp = -0.5 * d2 / sigma.pow(2) - 2.0 * torch.log(sigma) - LOG_2PI
        log_bg = torch.full_like(log_w[:, :1], -math.log(bg_area))
        return torch.logsumexp(log_w + torch.cat([log_comp, log_bg], dim=-1), dim=-1)

    @torch.no_grad()
    def sample(self, comp_xy: torch.Tensor, comp_feats: torch.Tensor,
               cond: torch.Tensor, bg_box: tuple[float, float, float, float]):
        """Returns (B, 2) km. bg_box = (xmin, ymin, xmax, ymax)."""
        log_w, sigma = self._params(cond, comp_feats)
        choice = torch.multinomial(log_w.exp(), 1).squeeze(-1)      # (B,)
        is_bg = choice == self.n_comp
        idx = choice.clamp(max=self.n_comp - 1)
        centers = comp_xy.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        sig = sigma.gather(1, idx.view(-1, 1)).squeeze(1)
        out = centers + sig.unsqueeze(-1) * torch.randn_like(centers)
        xmin, ymin, xmax, ymax = bg_box
        u = torch.rand(out.shape, device=out.device)
        bg_pt = torch.stack(
            [xmin + u[:, 0] * (xmax - xmin), ymin + u[:, 1] * (ymax - ymin)], dim=-1
        )
        return torch.where(is_bg.unsqueeze(-1), bg_pt, out)


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
