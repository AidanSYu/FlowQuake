"""Conditional flow-matching (rectified flow) density heads.

Training: simulation-free FM regression. z_t = (1-t) z0 + t u with
z0 ~ N(0, I); the velocity target is (u - z0).

Likelihood: exact continuous-normalizing-flow log-density by integrating the
ODE backward from the datum with the exact divergence (dimension is 1-2, so
the full Jacobian trace is cheap via torch.func.jacrev).

Sampling: forward ODE from Gaussian noise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


class TimeEmbed(nn.Module):
    """Fourier features of the flow time t in [0, 1]."""

    def __init__(self, n_freq: int = 4):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi)
        self.dim = 2 * n_freq + 1

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,) -> (..., dim)
        ang = t.unsqueeze(-1) * self.freqs
        return torch.cat([t.unsqueeze(-1), ang.sin(), ang.cos()], dim=-1)


class CondFlow(nn.Module):
    """Conditional rectified-flow density model for a d-dim target."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.temb = TimeEmbed()
        d_in = dim + self.temb.dim + cond_dim
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers += [nn.Linear(d_in if i == 0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)
        # Near-zero initial velocity field keeps early training stable.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def velocity(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z: (B, d); t: (B,) or scalar tensor; cond: (B, c)."""
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        return self.net(torch.cat([z, self.temb(t), cond], dim=-1))

    def fm_loss(self, u: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Mean rectified-flow matching loss. u: (B, d), cond: (B, c)."""
        B = u.shape[0]
        t = torch.rand(B, device=u.device, dtype=u.dtype)
        z0 = torch.randn_like(u)
        zt = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * u
        v = self.velocity(zt, t, cond)
        return F.mse_loss(v, u - z0)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 32) -> torch.Tensor:
        """Forward RK4 integration from N(0, I). cond: (B, c) -> (B, d)."""
        z = torch.randn(cond.shape[0], self.dim, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / steps
        for i in range(steps):
            t0 = torch.tensor(i * dt, device=z.device, dtype=z.dtype)
            k1 = self.velocity(z, t0, cond)
            k2 = self.velocity(z + 0.5 * dt * k1, t0 + 0.5 * dt, cond)
            k3 = self.velocity(z + 0.5 * dt * k2, t0 + 0.5 * dt, cond)
            k4 = self.velocity(z + dt * k3, t0 + dt, cond)
            z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    def _vel_and_div(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """Velocity and exact divergence via per-sample Jacobian trace."""

        def vel_single(z_i, t_i, c_i):
            return self.net(torch.cat([z_i, self.temb(t_i), c_i], dim=-1))

        jac_fn = torch.func.jacrev(vel_single, argnums=0)

        def both(z_i, t_i, c_i):
            return vel_single(z_i, t_i, c_i), torch.diagonal(jac_fn(z_i, t_i, c_i)).sum()

        if t.dim() == 0:
            t = t.expand(z.shape[0])
        return torch.func.vmap(both)(z, t, cond)

    @torch.no_grad()
    def log_prob(self, u: torch.Tensor, cond: torch.Tensor, steps: int = 64) -> torch.Tensor:
        """Exact log-density of u under the flow. Returns (B,)."""
        z = u.float()
        cond = cond.float()
        logdet = torch.zeros(z.shape[0], device=z.device)
        dt = 1.0 / steps
        # Integrate backward t: 1 -> 0, accumulating logdet = int_0^1 div dt;
        # then log p_1(u) = log N(z(0)) - logdet.
        for i in range(steps):
            t1 = 1.0 - i * dt

            def f(z_, l_, t_):
                t_t = torch.tensor(t_, device=z_.device, dtype=z_.dtype)
                v, div = self._vel_and_div(z_, t_t, cond)
                return -v, div  # moving backward in t

            k1z, k1l = f(z, logdet, t1)
            k2z, k2l = f(z + 0.5 * dt * k1z, logdet + 0.5 * dt * k1l, t1 - 0.5 * dt)
            k3z, k3l = f(z + 0.5 * dt * k2z, logdet + 0.5 * dt * k2l, t1 - 0.5 * dt)
            k4z, k4l = f(z + dt * k3z, logdet + dt * k3l, t1 - dt)
            z = z + dt / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)
            logdet = logdet + dt / 6.0 * (k1l + 2 * k2l + 2 * k3l + k4l)

        log_p0 = -0.5 * (z.pow(2).sum(-1) + self.dim * LOG_2PI)
        return log_p0 - logdet
