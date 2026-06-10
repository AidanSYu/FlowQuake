"""FlowQuake: selective-SSM whole-catalog encoder + flow-matching marked TPP.

Factorization per event (chain rule):
    f(tau, x, y, m | H) = f_t(tau | h) * f_s(x, y | tau, h) * f_m(m | tau, x, y, h)

All three heads are conditional rectified flows over *normalized* targets;
log-likelihoods are corrected back to physical units here:
    tll: log f_t in log(1/day)   (log-tau transform + standardization Jacobian)
    sll: log f_s in log(1/km^2)
    mll: log f_m in log(1/mag-unit)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import RECENCY_LAGS, TAU_FLOOR_DAYS
from .flow import CondFlow
from .ssm import SSMEncoder

D_IN = 4 + len(RECENCY_LAGS)  # [log_tau, x, y, m] + recency order statistics


class FlowQuakeTPP(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 64,
        n_heads: int = 8,
        expand: int = 2,
        chunk: int = 64,
        flow_hidden: int = 256,
        flow_layers: int = 3,
        stats: dict | None = None,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
        sigma_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dropout: float = 0.0,
        mag_dequant: float = 0.0,  # raw magnitude units (catalog grid ~0.01)
        input_noise: float = 0.0,  # train-time token jitter (normalized units)
    ):
        super().__init__()
        self.encoder = SSMEncoder(
            d_in=D_IN, d_model=d_model, n_layers=n_layers, d_state=d_state,
            n_heads=n_heads, expand=expand, chunk=chunk,
        )
        st, ss, sm = sigma_min
        # Heads condition on [h, current clean token] — the token carries the
        # previous event's absolute location and the Hawkes-style recency
        # statistics directly, so the heads need not re-derive them from h.
        # Spatial head models the OFFSET from the previous event (ETAS-style
        # triggering kernel).
        self.head_t = CondFlow(1, cond_dim=d_model + D_IN, hidden=flow_hidden,
                               n_layers=flow_layers, sigma_min=st, dropout=dropout)
        self.head_s = CondFlow(2, cond_dim=d_model + D_IN + 1, hidden=flow_hidden,
                               n_layers=flow_layers, sigma_min=ss, dropout=dropout)
        self.head_m = CondFlow(1, cond_dim=d_model + D_IN + 3, hidden=flow_hidden,
                               n_layers=flow_layers, sigma_min=sm, dropout=dropout)
        self.h_drop = nn.Dropout(dropout)
        self.mag_dequant = mag_dequant
        self.input_noise = input_noise
        self.loss_weights = loss_weights
        self.stats = dict(stats or {})

    # --- training ---------------------------------------------------------

    def fm_losses(self, tokens: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """tokens/target: (B, W, 4); mask: (B, W) bool over prediction positions."""
        enc_in = tokens
        if self.training and self.input_noise > 0:
            # Fresh history jitter every visit: the encoder state cannot
            # become a unique fingerprint of catalog position, which is the
            # memorization channel for the heads.
            enc_in = tokens + self.input_noise * torch.randn_like(tokens)
        h = self.encoder(enc_in)
        tok = tokens[mask]           # clean current token (K, D_IN)
        cond = torch.cat([self.h_drop(h[mask]), tok], dim=-1)
        s_last = tok[:, 1:3]         # absolute prev location (normalized)
        tgt = target[mask]           # (K, 4) normalized [log_tau, x, y, m]
        u_t = tgt[:, 0:1]
        u_s = tgt[:, 1:3] - s_last   # spatial offset target
        u_m = tgt[:, 3:4]
        if self.training and self.mag_dequant > 0:
            jitter = (torch.rand_like(u_m) - 0.5) * self.mag_dequant / self.stats["mag_std"]
            u_m = u_m + jitter
        loss_t = self.head_t.fm_loss(u_t, cond)
        loss_s = self.head_s.fm_loss(u_s, torch.cat([cond, u_t], dim=-1))
        loss_m = self.head_m.fm_loss(u_m, torch.cat([cond, u_t, u_s], dim=-1))
        wt, ws, wm = self.loss_weights
        total = wt * loss_t + ws * loss_s + wm * loss_m
        return total, {"loss_t": loss_t.item(), "loss_s": loss_s.item(), "loss_m": loss_m.item()}

    # --- evaluation -------------------------------------------------------

    @torch.no_grad()
    def encode_full(self, tokens: torch.Tensor, segment: int = 16384) -> torch.Tensor:
        """Whole-catalog encoding with carried streaming state. tokens: (1, E, 4)."""
        E = tokens.shape[1]
        outs = []
        layer_states = None
        for s in range(0, E, segment):
            h, layer_states = self.encoder.prefill(tokens[:, s : s + segment], layer_states)
            outs.append(h)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def log_likelihood(
        self,
        tokens: torch.Tensor,   # (1, E, 4)
        target: torch.Tensor,   # (1, E, 4)
        mask: torch.Tensor,     # (1, E)
        steps: int = 64,
        event_chunk: int = 4096,
        segment: int = 16384,
    ) -> dict[str, torch.Tensor]:
        """Per-event tll/sll/mll (physical units) at masked positions."""
        st = self.stats
        h_all = self.encode_full(tokens, segment=segment)
        cond_all = torch.cat([h_all[mask], tokens[mask]], dim=-1)
        s_last_all = tokens[mask][:, 1:3]
        tgt = target[mask]
        tll, sll, mll = [], [], []
        for i in range(0, cond_all.shape[0], event_chunk):
            cs = cond_all[i : i + event_chunk]
            ts = tgt[i : i + event_chunk]
            sl = s_last_all[i : i + event_chunk]
            u_t, u_m = ts[:, 0:1], ts[:, 3:4]
            u_s = ts[:, 1:3] - sl  # offset target; translation has unit Jacobian
            lp_t = self.head_t.log_prob(u_t, cs, steps=steps)
            lp_s = self.head_s.log_prob(u_s, torch.cat([cs, u_t], dim=-1), steps=steps)
            lp_m = self.head_m.log_prob(u_m, torch.cat([cs, u_t, u_s], dim=-1), steps=steps)
            # Jacobian corrections to physical units.
            log_tau = ts[:, 0] * st["log_tau_std"] + st["log_tau_mean"]
            tll.append(lp_t - math.log(st["log_tau_std"]) - log_tau)
            sll.append(lp_s - math.log(st["x_std"] * st["y_std"]))
            mll.append(lp_m - math.log(st["mag_std"]))
        return {
            "tll": torch.cat(tll),
            "sll": torch.cat(sll),
            "mll": torch.cat(mll),
        }

    # --- simulation -------------------------------------------------------

    @torch.no_grad()
    def sample_next(self, h: torch.Tensor, token_last: torch.Tensor, steps: int = 24):
        """Sample (tau_days, x_km, y_km, mag) for the next event.

        h: (B, d_model); token_last: (B, D_IN) current clean token (its [1:3]
        slice is the previous event's normalized absolute location).
        """
        st = self.stats
        cond = torch.cat([h, token_last], dim=-1)
        s_last = token_last[:, 1:3]
        u_t = self.head_t.sample(cond, steps=steps)
        u_s = self.head_s.sample(torch.cat([cond, u_t], dim=-1), steps=steps)
        u_m = self.head_m.sample(torch.cat([cond, u_t, u_s], dim=-1), steps=steps)
        abs_s = u_s + s_last
        tau = torch.exp(u_t[:, 0] * st["log_tau_std"] + st["log_tau_mean"])
        x = abs_s[:, 0] * st["x_std"] + st["x_mean"]
        y = abs_s[:, 1] * st["y_std"] + st["y_mean"]
        m = u_m[:, 0] * st["mag_std"] + st["mag_mean"]
        return tau, x, y, m, (u_t, abs_s, u_m)

    def build_token(self, tau_days, x, y, m, t_next, t_buf):
        """Physical next-event attributes -> normalized token (B, D_IN).

        t_next: (B,) absolute event time in days; t_buf: (B, >=max_lag) recent
        event times, most recent first (NOT including this event).
        """
        st = self.stats
        core = torch.stack(
            [
                (torch.log(torch.clamp(tau_days, min=TAU_FLOOR_DAYS)) - st["log_tau_mean"])
                / st["log_tau_std"],
                (x - st["x_mean"]) / st["x_std"],
                (y - st["y_mean"]) / st["y_std"],
                (m - st["mag_mean"]) / st["mag_std"],
            ],
            dim=-1,
        )
        rec_mean = torch.as_tensor(st["rec_mean"], device=core.device, dtype=core.dtype)
        rec_std = torch.as_tensor(st["rec_std"], device=core.device, dtype=core.dtype)
        recs = []
        for j, k in enumerate(RECENCY_LAGS):
            dtk = torch.clamp(t_next - t_buf[:, k - 1], min=TAU_FLOOR_DAYS)
            recs.append((torch.log(dtk) - rec_mean[j].double()) / rec_std[j].double())
        return torch.cat([core, torch.stack(recs, dim=-1).to(core.dtype)], dim=-1)
