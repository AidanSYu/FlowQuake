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
    """Conditional rectified-flow density model for a d-dim target.

    sigma_min > 0 uses the path z_t = (1 - (1 - sigma_min) t) z0 + t u, so the
    modeled density is the data convolved with N(0, sigma_min^2 I): a KDE-style
    bandwidth floor that prevents collapse onto (discretized) training targets.
    """

    def __init__(self, dim: int, cond_dim: int, hidden: int = 256, n_layers: int = 3,
                 sigma_min: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.sigma_min = sigma_min
        self.temb = TimeEmbed()
        d_in = dim + self.temb.dim + cond_dim
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers += [nn.Linear(d_in if i == 0 else hidden, hidden), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)
        # Near-zero initial velocity field keeps early training stable.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        #: Set by `enable_compiled_net()`, and held inside a LIST on purpose.
        #: `torch.compile` returns an `OptimizedModule`, which is an nn.Module, so
        #: assigning it to an attribute makes nn.Module register it as a CHILD --
        #: every weight then appears twice in state_dict and the checkpoint format
        #: silently changes. Verified: plain assignment altered state_dict keys.
        #: A plain list is not registered, so the compiled wrapper stays invisible
        #: to serialisation.
        self._compiled_box: list = []

    def enable_compiled_net(self, mode: str = "reduce-overhead") -> bool:
        """Opt in to a `torch.compile`d velocity network. Returns success.

        OFF BY DEFAULT, and deliberately so. After the buffer/embedding rewrite
        the remaining sampler cost is `linear` (~30%) and `silu` (~24%) -- fusion
        targets, and on a GPU the 64 velocity evaluations per `sample_next` are
        launch-bound rather than compute-bound, which is what "reduce-overhead"
        (CUDA graphs) exists for. But that is a claim about a device this has not
        been measured on, and compilation also costs seconds per distinct batch
        shape, so enabling it blindly could easily be a net loss.

        It is opt-in for a correctness reason too: unlike the buffer rewrite,
        compilation may reassociate floating-point work, so results are NOT
        guaranteed bit-identical. A curve must therefore be produced entirely
        with it on or entirely off -- the same whole-curve rule invariant 1t
        applies to device choice.
        """
        try:
            self._compiled_box = [torch.compile(self.net, mode=mode, dynamic=False)]
            return True
        except Exception:                                        # noqa: BLE001
            self._compiled_box = []
            return False

    def velocity(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z: (B, d); t: (B,) or scalar tensor; cond: (B, c)."""
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        return self.net(torch.cat([z, self.temb(t), cond], dim=-1))

    def fm_loss(self, u: torch.Tensor, cond: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """Mean conditional flow-matching loss. u: (B, d), cond: (B, c).

        `weights` (B,) reweights the per-event contribution. It exists so the
        training objective can be tilted toward the magnitudes the model is
        actually scored on; see FlowQuakeTPP.fm_losses. With weights=None the
        result is bit-identical to the unweighted mean, because mse over (B, d)
        averaged whole equals the per-row mean averaged again.
        """
        B = u.shape[0]
        s = self.sigma_min
        t = torch.rand(B, device=u.device, dtype=u.dtype)
        z0 = torch.randn_like(u)
        zt = (1.0 - (1.0 - s) * t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * u
        v = self.velocity(zt, t, cond)
        tgt = u - (1.0 - s) * z0
        if weights is None:
            return F.mse_loss(v, tgt)
        return (F.mse_loss(v, tgt, reduction="none").mean(dim=-1) * weights).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 32) -> torch.Tensor:
        """Forward RK4 integration from N(0, I). cond: (B, c) -> (B, d).

        Numerically identical to the obvious loop, but avoids two costs that a
        CPU profile showed dominating the sampler (which is 98.5% of scoring):

        `torch.cat` was 17.5% of runtime and `sin`/`cos` 11%. Both are pure
        overhead here, because of two facts about this particular integration:

          * the RK4 sub-step time is a SCALAR, identical for every lane, and its
            ~3*steps distinct values are known before the loop starts -- so the
            Fourier time embedding was being recomputed across all B lanes 4*steps
            times to produce `temb.dim` distinct numbers;
          * `cond` never changes during the solve -- so `cat` was re-copying the
            widest block of the network input on all 4*steps velocity calls.

        So the embedding is computed once per distinct time and broadcast, and
        the network input is written into ONE preallocated buffer whose `cond`
        block is filled a single time. The buffer is contiguous and row-major
        with the same column order the `cat` produced, so `self.net` sees exactly
        the same bytes and results stay bit-identical -- pinned by
        tests/test_flow_sample_equivalence.py.
        """
        B, d = cond.shape[0], self.dim
        te_dim, c = self.temb.dim, cond.shape[1]
        z = torch.randn(B, d, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / steps

        buf = torch.empty(B, d + te_dim + c, device=cond.device, dtype=cond.dtype)
        buf[:, d + te_dim:] = cond                    # constant for the whole solve

        net = self._compiled_box[0] if self._compiled_box else self.net

        def vel(zc: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
            buf[:, :d] = zc
            buf[:, d:d + te_dim] = te                 # broadcasts (te_dim,) -> (B, te_dim)
            return net(buf)

        for i in range(steps):
            # Scalar times -> (te_dim,) embeddings, computed once each.
            t_a = torch.tensor(i * dt, device=z.device, dtype=z.dtype)
            te_a = self.temb(t_a)
            te_b = self.temb(t_a + 0.5 * dt)
            te_c = self.temb(t_a + dt)
            k1 = vel(z, te_a)
            k2 = vel(z + 0.5 * dt * k1, te_b)
            k3 = vel(z + 0.5 * dt * k2, te_b)
            k4 = vel(z + dt * k3, te_c)
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
