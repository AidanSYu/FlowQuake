"""EarthquakeNPP model: encoder + temporal/spatial/(magnitude) density heads.

Convention: feature row i describes event i (its Δt from i-1, its x, y, magnitude).  The
encoder hidden state after consuming events 0..i, i.e. hidden[i], conditions the
prediction of event i+1.  So the context for a target event at position p is hidden[p-1].

Metrics (mean over scored events):
    TLL = log f(Δt_p | hidden[p-1])              temporal log-density (1/day)
    SLL = log p(x_p, y_p | hidden[p-1])          spatial log-density  (1/km^2)
both directly comparable to the benchmark's ETAS numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ssm import build_encoder
from .flows import build_spatial_head
from .heads import LogNormalMixtureTime, GutenbergRichterHead
from .trigger import TriggeringKernelHead, SpatialContext


@dataclass
class ModelConfig:
    encoder: str = "ssm"          # ssm | gru
    spatial: str = "trigger"      # gaussian | mdn | realnvp | fm | trigger
    d_model: int = 96
    n_layers: int = 3
    d_state: int = 8
    n_time_comp: int = 16
    n_spatial_comp: int = 16      # for mdn
    n_flow_layers: int = 8        # for realnvp
    fm_hidden: int = 128
    fm_steps: int = 40
    n_recent: int = 64            # for trigger: # of recent epicenters used as kernels
    n_bg: int = 8                 # for trigger: # of background-field components
    use_magnitude: bool = False
    d_in: int = 4


class EarthquakeNPP(nn.Module):
    def __init__(self, cfg: ModelConfig, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg.encoder, d_in=cfg.d_in, d_model=cfg.d_model,
                                     n_layers=cfg.n_layers, d_state=cfg.d_state)
        H = cfg.d_model
        self.time_head = LogNormalMixtureTime(H, n_comp=cfg.n_time_comp)
        if cfg.spatial == "mdn":
            self.spatial_head = build_spatial_head("mdn", H, n_comp=cfg.n_spatial_comp)
        elif cfg.spatial == "realnvp":
            self.spatial_head = build_spatial_head("realnvp", H, n_layers=cfg.n_flow_layers)
        elif cfg.spatial == "fm":
            self.spatial_head = build_spatial_head("fm", H, hidden=cfg.fm_hidden, n_steps=cfg.fm_steps)
        elif cfg.spatial == "trigger":
            self.spatial_head = TriggeringKernelHead(H, n_bg=cfg.n_bg)
        else:
            self.spatial_head = build_spatial_head("gaussian", H)
        self.mag_head = GutenbergRichterHead(H, conditional=False) if cfg.use_magnitude else None

        self.register_buffer("sp_mean", mean.clone().float())
        self.register_buffer("sp_std", std.clone().float())

    # ---- spatial unit handling ----
    @property
    def sp_log_det(self) -> torch.Tensor:
        return -torch.log(self.sp_std).sum()

    @property
    def needs_recent(self) -> bool:
        return isinstance(self.spatial_head, TriggeringKernelHead)

    def _z(self, xy: torch.Tensor) -> torch.Tensor:
        return (xy - self.sp_mean) / self.sp_std

    def standardize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        return self._z(xy)

    # ---- encoding ----
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.encoder(feats)

    @torch.no_grad()
    def encode_long(self, feats: torch.Tensor, chunk: int = 8192, left_ctx: int = 4096) -> torch.Tensor:
        """Causal per-event hidden states for one long sequence [N, d_in] -> [N, H].

        Slides over the sequence in chunks, each prepended with `left_ctx` preceding
        events as warm-up (discarded from the output).  Because the SSM's memory decays,
        a few thousand events of left context is effectively full history — far beyond the
        32-event window of the reference dummy, while keeping memory bounded.
        """
        N = feats.shape[0]
        H = self.cfg.d_model
        out = feats.new_zeros(N, H)
        pos = 0
        while pos < N:
            end = min(pos + chunk, N)
            start = max(0, pos - left_ctx)
            seg = feats[start:end].unsqueeze(0)               # [1, L, d_in]
            hid = self.encoder(seg)[0]                        # [L, H]
            out[pos:end] = hid[pos - start:end - start]
            pos = end
        return out

    # ---- per-event densities ----
    # Temporal / magnitude heads condition only on h_self.  Spatial heads receive a
    # SpatialContext; the triggering head uses the recent-event window, others use h_self.
    def temporal_logprob(self, c, dt):
        return self.time_head.log_prob(dt, c)

    def spatial_logprob(self, sc: SpatialContext, xy):
        z = self._z(xy)
        if self.needs_recent:
            lp = self.spatial_head.log_prob(z, sc)
        else:
            lp = self.spatial_head.log_prob(z, sc.h_self)
        return lp + self.sp_log_det

    def magnitude_logprob(self, c, dm):
        return self.mag_head.log_prob(dm, c)

    # ---- training losses (per sample, to minimize) ----
    def temporal_loss(self, c, dt):
        return self.time_head.train_loss(dt, c)

    def spatial_loss(self, sc: SpatialContext, xy):
        # const sp_log_det is irrelevant to optimization; FM head uses velocity MSE
        z = self._z(xy)
        if self.needs_recent:
            return self.spatial_head.train_loss(z, sc)
        return self.spatial_head.train_loss(z, sc.h_self)

    def magnitude_loss(self, c, dm):
        return self.mag_head.train_loss(dm, c)
