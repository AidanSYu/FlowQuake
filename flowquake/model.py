"""FlowQuake: selective-SSM whole-catalog encoder + structured marked TPP.

Heads (chain rule f = f_t * f_s * f_m):
  time      — conditional rectified flow on normalized log-tau (exact ODE LL);
              tll in log(1/day) after Jacobian corrections.
  space     — closed-form kernel mixture anchored at the last K observed event
              locations (Hawkes-style triggering kernel + uniform background);
              sll directly in log(1/km^2).
  magnitude — conditional Gutenberg-Richter exponential beta(h);
              mll directly in log(1/mag-unit).

The spatial/magnitude heads were pivoted from flows after val-LL diagnostics:
flows anchored structure in their weights (memorizing train geography),
while mixture components anchored to observed recent events move with the
data at evaluation time. The flow stays where it beats its baselines (time).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import LAST_K, MIX_K, RECENCY_LAGS, TAU_FLOOR_DAYS, TOKEN_DIM
from .flow import CondFlow
from .heads import GRMagnitudeHead, KernelMixtureHead
from .ssm import SSMEncoder

D_IN = TOKEN_DIM

# Token dims safe for head conditioning: everything except absolute x, y.
# (log_tau, mag, and the per-lag relational block are translation-invariant
# statistics that cannot fingerprint a specific catalog position-era.)
SAFE_TOKEN_DIMS = [0, 3] + list(range(4, TOKEN_DIM))


class FlowQuakeTPP(nn.Module):
    """h_bottleneck:
        0  -> heads see relational token features only ("neural ETAS";
              memorization through conditioning structurally impossible)
        k  -> additionally a k-dim noisy linear readout of the SSM state
              (the whole-catalog channel, too narrow to memorize events)
    """

    def __init__(
        self,
        d_model: int = 96,
        n_layers: int = 4,
        d_state: int = 32,
        n_heads: int = 6,
        expand: int = 2,
        chunk: int = 64,
        flow_hidden: int = 96,
        flow_layers: int = 3,
        stats: dict | None = None,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
        sigma_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dropout: float = 0.0,
        mix_hidden: int = 64,
        mag_dequant: float = 0.0,   # kept for config compat (GR head absorbs it)
        input_noise: float = 0.0,   # train-time token jitter (normalized units)
        h_bottleneck: int = 0,
        h_noise: float = 0.1,       # train-time noise on the bottleneck channel
    ):
        super().__init__()
        self.h_bottleneck = h_bottleneck
        self.h_noise = h_noise
        if h_bottleneck > 0:
            self.encoder = SSMEncoder(
                d_in=D_IN, d_model=d_model, n_layers=n_layers, d_state=d_state,
                n_heads=n_heads, expand=expand, chunk=chunk,
            )
            self.h_proj = nn.Linear(d_model, h_bottleneck)
        else:
            self.encoder = None
        cond_dim = len(SAFE_TOKEN_DIMS) + h_bottleneck
        self.head_t = CondFlow(1, cond_dim=cond_dim, hidden=flow_hidden,
                               n_layers=flow_layers, sigma_min=sigma_min[0],
                               dropout=dropout)
        self.head_s = KernelMixtureHead(cond_dim, n_comp=MIX_K, hidden=mix_hidden)
        self.head_m = GRMagnitudeHead(cond_dim)
        self.input_noise = input_noise
        self.loss_weights = loss_weights
        self.stats = dict(stats or {})

    # --- shared plumbing ----------------------------------------------------

    def _comp_inputs(self, lastk_sel: torch.Tensor):
        """lastk_sel: (B, K, 4) raw [x, y, log_dt, m] -> (centers km, feats)."""
        st = self.stats
        comp_xy = lastk_sel[..., :2]
        cur = comp_xy[:, 0:1, :]  # component 0 is the current (last) event
        dist = (comp_xy - cur).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        feats = torch.stack(
            [
                (lastk_sel[..., 2] - st["log_tau_mean"]) / st["log_tau_std"],
                (lastk_sel[..., 3] - st["mag_mean"]) / st["mag_std"],
                torch.log(dist + 0.1) / 3.0,
            ],
            dim=-1,
        )
        return comp_xy, feats

    # --- background fault-map lookup ---------------------------------------

    def _log_kde(self, s_km: torch.Tensor) -> torch.Tensor | None:
        """Train-era seismicity log-density (per km^2) at s_km (B, 2)."""
        st = self.stats
        if "kde_log_grid" not in st:
            return None
        if getattr(self, "_kde_t", None) is None or self._kde_t.device != s_km.device:
            self._kde_t = torch.as_tensor(st["kde_log_grid"], device=s_km.device)
        g = self._kde_t
        ix = ((s_km[:, 0] - st["bg_xmin"]) / st["kde_bin"]).long().clamp(0, g.shape[0] - 1)
        iy = ((s_km[:, 1] - st["bg_ymin"]) / st["kde_bin"]).long().clamp(0, g.shape[1] - 1)
        return g[ix, iy]

    def _kde_sampler(self, device):
        st = self.stats
        if "kde_log_grid" not in st:
            return None
        if getattr(self, "_kde_probs", None) is None or self._kde_probs.device != device:
            g = torch.as_tensor(st["kde_log_grid"], device=device).double()
            self._kde_probs = (g - g.max()).exp().flatten()
            self._kde_shape = st["kde_log_grid"].shape

        def sampler(n):
            flat = torch.multinomial(self._kde_probs, n, replacement=True)
            nx, ny = self._kde_shape
            ix = (flat // ny).float()
            iy = (flat % ny).float()
            u = torch.rand(n, 2, device=device)
            x = st["bg_xmin"] + (ix + u[:, 0]) * st["kde_bin"]
            y = st["bg_ymin"] + (iy + u[:, 1]) * st["kde_bin"]
            return torch.stack([x, y], dim=-1)

        return sampler

    # --- shared conditioning ----------------------------------------------

    def _cond(self, tokens, mask):
        """Build head conditioning rows for masked positions."""
        tok_safe = tokens[mask][:, SAFE_TOKEN_DIMS]
        if self.h_bottleneck == 0:
            return tok_safe
        enc_in = tokens
        if self.training and self.input_noise > 0:
            enc_in = tokens + self.input_noise * torch.randn_like(tokens)
        h = self.h_proj(self.encoder(enc_in)[mask])
        if self.training and self.h_noise > 0:
            h = h + self.h_noise * torch.randn_like(h)
        return torch.cat([tok_safe, h], dim=-1)

    # --- training -------------------------------------------------------

    def fm_losses(self, tokens, target, mask, lastk, raw_next):
        """tokens (B,W,D_IN); target (B,W,4) normalized; mask (B,W) bool;
        lastk (B,W,K,4) raw; raw_next (B,W,3) raw (x,y,mag) of next event."""
        st = self.stats
        cond = self._cond(tokens, mask)

        u_t = target[mask][:, 0:1]
        loss_t = self.head_t.fm_loss(u_t, cond)

        comp_xy, comp_feats = self._comp_inputs(lastk[mask])
        s_tgt = raw_next[mask][:, :2]
        sll = self.head_s.log_prob(s_tgt, comp_xy, comp_feats, cond, st["bg_area"],
                                   log_kde_at_s=self._log_kde(s_tgt))
        loss_s = -sll.mean()

        m_tgt = raw_next[mask][:, 2]
        mll = self.head_m.log_prob(m_tgt, cond, st["mcut"])
        loss_m = -mll.mean()

        wt, ws, wm = self.loss_weights
        total = wt * loss_t + ws * loss_s + wm * loss_m
        return total, {"loss_t": loss_t.item(), "loss_s": loss_s.item(),
                       "loss_m": loss_m.item()}

    # --- evaluation -------------------------------------------------------

    @torch.no_grad()
    def encode_full(self, tokens: torch.Tensor, segment: int = 16384) -> torch.Tensor:
        E = tokens.shape[1]
        outs = []
        layer_states = None
        for s in range(0, E, segment):
            h, layer_states = self.encoder.prefill(tokens[:, s : s + segment], layer_states)
            outs.append(h)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def _cond_eval(self, tokens, mask, segment):
        tok_safe = tokens[mask][:, SAFE_TOKEN_DIMS]
        if self.h_bottleneck == 0:
            return tok_safe
        h_all = self.encode_full(tokens, segment=segment)
        return torch.cat([tok_safe, self.h_proj(h_all[mask])], dim=-1)

    @torch.no_grad()
    def log_likelihood(self, tokens, target, mask, lastk, raw_next,
                       steps: int = 64, event_chunk: int = 4096,
                       segment: int = 16384) -> dict[str, torch.Tensor]:
        """Per-event tll/sll/mll in physical units at masked positions."""
        st = self.stats
        cond_all = self._cond_eval(tokens, mask, segment)
        tgt = target[mask]
        lk = lastk[mask]
        rn = raw_next[mask]
        tll, sll, mll = [], [], []
        for i in range(0, cond_all.shape[0], event_chunk):
            cs = cond_all[i : i + event_chunk]
            u_t = tgt[i : i + event_chunk, 0:1]
            lp_t = self.head_t.log_prob(u_t, cs, steps=steps)
            log_tau = tgt[i : i + event_chunk, 0] * st["log_tau_std"] + st["log_tau_mean"]
            tll.append(lp_t - math.log(st["log_tau_std"]) - log_tau)

            comp_xy, comp_feats = self._comp_inputs(lk[i : i + event_chunk])
            s_tgt = rn[i : i + event_chunk, :2]
            sll.append(self.head_s.log_prob(s_tgt, comp_xy, comp_feats, cs,
                                            st["bg_area"],
                                            log_kde_at_s=self._log_kde(s_tgt)))
            mll.append(self.head_m.log_prob(rn[i : i + event_chunk, 2], cs, st["mcut"]))
        return {"tll": torch.cat(tll), "sll": torch.cat(sll), "mll": torch.cat(mll)}

    # --- simulation -------------------------------------------------------

    @torch.no_grad()
    def sample_next(self, h: torch.Tensor | None, token_last: torch.Tensor,
                    lastk_lane: torch.Tensor, steps: int = 24):
        """h (B,d_model) or None when h_bottleneck==0; token_last (B,D_IN);
        lastk_lane (B,K,4) raw components. Returns (tau_days, x_km, y_km, mag)."""
        st = self.stats
        cond = token_last[:, SAFE_TOKEN_DIMS]
        if self.h_bottleneck > 0:
            cond = torch.cat([cond, self.h_proj(h)], dim=-1)
        u_t = self.head_t.sample(cond, steps=steps)
        tau = torch.exp(u_t[:, 0] * st["log_tau_std"] + st["log_tau_mean"])
        comp_xy, comp_feats = self._comp_inputs(lastk_lane)
        s = self.head_s.sample(comp_xy, comp_feats, cond,
                               (st["bg_xmin"], st["bg_ymin"], st["bg_xmax"], st["bg_ymax"]),
                               kde_sampler=self._kde_sampler(cond.device))
        m = self.head_m.sample(cond, st["mcut"]).clamp(max=8.5)
        return tau, s[:, 0], s[:, 1], m

    def build_token(self, tau_days, x, y, m, t_next, bufs):
        """Physical next-event attributes -> normalized token (B, D_IN).

        bufs = (t_buf, x_buf, y_buf, m_buf), recent-first raw history per lane.
        """
        st = self.stats
        t_buf, x_buf, y_buf, m_buf = bufs
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
            recs.append(torch.log(dtk).to(core.dtype))
            recs.append(x - x_buf[:, k - 1])
            recs.append(y - y_buf[:, k - 1])
            recs.append(m_buf[:, k - 1])
        rec = (torch.stack(recs, dim=-1) - rec_mean) / rec_std
        return torch.cat([core, rec], dim=-1)

    @staticmethod
    def lastk_from_bufs(t_last, bufs, static_big=None):
        """Mixture components (B, MIX_K, 4) from lane history buffers (which
        hold the last events most-recent-first INCLUDING the current last
        event) plus a static big-trigger block for the simulation horizon."""
        t_buf, x_buf, y_buf, m_buf = bufs
        K = LAST_K
        log_dt = torch.log(
            torch.clamp(t_last.unsqueeze(1) - t_buf[:, :K], min=TAU_FLOOR_DAYS)
        ).to(x_buf.dtype)
        recent = torch.stack(
            [x_buf[:, :K], y_buf[:, :K], log_dt, m_buf[:, :K]], dim=-1
        )
        if static_big is None:
            return recent
        return torch.cat([recent, static_big], dim=1)
