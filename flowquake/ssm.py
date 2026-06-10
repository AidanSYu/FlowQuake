"""Selective state-space (Mamba-2 style SSD) encoder in pure PyTorch.

No custom CUDA kernels (runs on Windows): the selective scan is computed with
the chunked block decomposition of the SSD form — exact, parallel within
chunks, sequential across chunks in fp32.

Per head h with scalar decay: state H_t = a_t * H_{t-1} + dt_t * B_t x_t^T,
y_t = C_t . H_t + D x_t, where a_t = exp(-dt_t * A_h) depends on the input
through dt_t (selectivity).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def segsum_decay(log_a: torch.Tensor) -> torch.Tensor:
    """Lower-triangular pairwise decay matrix within a chunk.

    log_a: (..., Q) per-step log decays (<= 0).
    Returns (..., Q, Q) with M[t, s] = exp(sum_{r=s+1..t} log_a_r) for s <= t,
    and 0 for s > t.
    """
    Q = log_a.shape[-1]
    cs = torch.cumsum(log_a, dim=-1)
    # sum_{r=s+1..t} = cs[t] - cs[s]
    diff = cs.unsqueeze(-1) - cs.unsqueeze(-2)  # (..., t, s)
    mask = torch.tril(torch.ones(Q, Q, dtype=torch.bool, device=log_a.device))
    return torch.where(mask, diff, torch.full_like(diff, -torch.inf)).exp()


def selective_scan_chunked(
    x: torch.Tensor,      # (B, L, H, P) inputs per head
    dt: torch.Tensor,     # (B, L, H) positive step sizes
    A: torch.Tensor,      # (H,) positive decay rates
    Bm: torch.Tensor,     # (B, L, N) input projection (shared across heads)
    Cm: torch.Tensor,     # (B, L, N) output projection (shared across heads)
    chunk: int = 64,
    init_state: torch.Tensor | None = None,  # (B, H, N, P)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact selective scan via chunked SSD decomposition (fp32).

    Returns (y, final_state): y (B, L, H, P), final_state (B, H, N, P).
    """
    Bsz, L, H, P = x.shape
    N = Bm.shape[-1]
    orig_dtype = x.dtype
    x, dt, A, Bm, Cm = (t.float() for t in (x, dt, A, Bm, Cm))

    pad = (-L) % chunk
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        Bm = F.pad(Bm, (0, 0, 0, pad))
        Cm = F.pad(Cm, (0, 0, 0, pad))
    Lp = L + pad
    nc = Lp // chunk

    xb = x.view(Bsz, nc, chunk, H, P)
    dtb = dt.view(Bsz, nc, chunk, H)
    Bb = Bm.view(Bsz, nc, chunk, N)
    Cb = Cm.view(Bsz, nc, chunk, N)

    log_a = -dtb * A  # (B, nc, Q, H), <= 0
    log_a = log_a.permute(0, 1, 3, 2)  # (B, nc, H, Q)
    cs = torch.cumsum(log_a, dim=-1)   # within-chunk cumulative decay

    # Intra-chunk (diagonal block): Y[t] = sum_{s<=t} decay(t,s) * (C_t.B_s) * dt_s * x_s
    M = segsum_decay(log_a)                                # (B, nc, H, Q, Q)
    G = torch.einsum("bctn,bcsn->bcts", Cb, Bb)            # (B, nc, Q, Q)
    W = M * G.unsqueeze(2)                                 # (B, nc, H, t, s)
    y_intra = torch.einsum("bchts,bcsh,bcshp->bcthp", W, dtb, xb)

    # Chunk summary states: S_c = sum_s decay(end, s) * dt_s * B_s x_s^T  (N, P)
    decay_to_end = (cs[..., -1:] - cs).exp()               # (B, nc, H, Q)
    S = torch.einsum("bchs,bcsh,bcsn,bcshp->bchnp", decay_to_end, dtb, Bb, xb)

    # Sequential inter-chunk recurrence on summaries.
    chunk_decay = cs[..., -1].exp()                        # (B, nc, H)
    states = []  # state entering each chunk
    s_run = (
        init_state.float()
        if init_state is not None
        else torch.zeros(Bsz, H, N, P, dtype=torch.float32, device=x.device)
    )
    for c in range(nc):
        states.append(s_run)
        s_run = chunk_decay[:, c, :, None, None] * s_run + S[:, c]
    states = torch.stack(states, dim=1)                    # (B, nc, H, N, P)

    # Inter-chunk contribution: Y[t] += C_t . (decay(t, start) * state_in)
    y_inter = torch.einsum("bcht,bctn,bchnp->bcthp", cs.exp(), Cb, states)

    y = (y_intra + y_inter).reshape(Bsz, Lp, H, P)[:, :L]
    return y.to(orig_dtype), s_run


def selective_scan_ref(x, dt, A, Bm, Cm, init_state=None):
    """Naive O(L) recurrence in fp64 — reference for tests."""
    Bsz, L, H, P = x.shape
    N = Bm.shape[-1]
    x, dt, A, Bm, Cm = (t.double() for t in (x, dt, A, Bm, Cm))
    state = (
        init_state.double()
        if init_state is not None
        else torch.zeros(Bsz, H, N, P, dtype=torch.float64, device=x.device)
    )
    ys = []
    for t in range(L):
        a = torch.exp(-dt[:, t] * A)                       # (B, H)
        upd = torch.einsum("bn,bhp->bhnp", Bm[:, t], dt[:, t, :, None] * x[:, t])
        state = a[:, :, None, None] * state + upd
        ys.append(torch.einsum("bn,bhnp->bhp", Cm[:, t], state))
    return torch.stack(ys, dim=1), state


class SSDBlock(nn.Module):
    """One Mamba-2 style block: in_proj -> causal conv -> selective scan -> gated norm -> out_proj."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        n_heads: int = 8,
        expand: int = 2,
        d_conv: int = 4,
        chunk: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        A_min: float = 1.0,
        A_max: float = 16.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        assert self.d_inner % n_heads == 0
        self.n_heads = n_heads
        self.d_head = self.d_inner // n_heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.chunk = chunk

        d_xbc = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, self.d_inner + d_xbc + n_heads)
        self.conv = nn.Conv1d(d_xbc, d_xbc, d_conv, groups=d_xbc, padding=d_conv - 1)

        # dt bias: softplus(bias) log-uniform in [dt_min, dt_max]
        dt_init = torch.exp(
            torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.dt_bias = nn.Parameter(dt_init + torch.log(-torch.expm1(-dt_init)))
        self.A_log = nn.Parameter(
            torch.log(A_min + (A_max - A_min) * torch.rand(n_heads))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.norm = nn.RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _split(self, zxbcdt: torch.Tensor):
        return torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads],
            dim=-1,
        )

    def forward(
        self,
        u: torch.Tensor,
        init_state: torch.Tensor | None = None,
        conv_init: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        """u: (B, L, d_model). Optionally seed scan/conv state (for streaming)."""
        B, L, _ = u.shape
        z, xbc, dt_raw = self._split(self.in_proj(u))

        xbc_t = xbc.transpose(1, 2)  # (B, C, L)
        if conv_init is not None:
            k = conv_init.shape[2]
            xbc_t = torch.cat([conv_init, xbc_t], dim=2)
            conv_out = self.conv(xbc_t)[..., k:k + L]
        else:
            conv_out = self.conv(xbc_t)[..., :L]
        conv_cache = xbc_t[..., -(self.d_conv - 1):] if return_state else None
        xbc = F.silu(conv_out).transpose(1, 2)

        x, Bm, Cm = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.view(B, L, self.n_heads, self.d_head)
        dt = F.softplus(dt_raw + self.dt_bias)
        A = torch.exp(self.A_log)

        y, state = selective_scan_chunked(
            x, dt, A, Bm, Cm, chunk=self.chunk, init_state=init_state
        )
        y = y + self.D.view(self.n_heads, self.d_head) * x
        y = y.reshape(B, L, self.d_inner)
        y = self.norm(y * F.silu(z))
        out = self.out_proj(y)
        if return_state:
            return out, (state, conv_cache)
        return out

    def step(self, u: torch.Tensor, state: torch.Tensor, conv_cache: torch.Tensor):
        """Single-event update for autoregressive simulation.

        u: (B, d_model); state: (B, H, N, P); conv_cache: (B, d_xbc, d_conv-1).
        Returns (out (B, d_model), new_state, new_conv_cache).
        """
        z, xbc, dt_raw = self._split(self.in_proj(u))
        window = torch.cat([conv_cache, xbc.unsqueeze(-1)], dim=2)  # (B, C, d_conv)
        conv_w = self.conv.weight.squeeze(1)                        # (C, d_conv)
        conv_out = (window * conv_w).sum(-1) + self.conv.bias
        new_cache = window[..., 1:]
        xbc = F.silu(conv_out)

        x, Bm, Cm = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.view(-1, self.n_heads, self.d_head)
        dt = F.softplus(dt_raw + self.dt_bias)
        A = torch.exp(self.A_log)

        a = torch.exp(-dt * A)                                      # (B, H)
        upd = torch.einsum("bn,bhp->bhnp", Bm, dt.unsqueeze(-1) * x)
        state = a[:, :, None, None] * state + upd
        y = torch.einsum("bn,bhnp->bhp", Cm, state)
        y = y + self.D.view(self.n_heads, self.d_head) * x
        y = y.reshape(u.shape[0], self.d_inner)
        y = self.norm(y * F.silu(z))
        return self.out_proj(y), state, new_cache


class SSMEncoder(nn.Module):
    """Stack of pre-norm residual SSD blocks: the whole-catalog event encoder."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 64,
        n_heads: int = 8,
        expand: int = 2,
        chunk: int = 64,
    ):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(n_layers))
        self.blocks = nn.ModuleList(
            SSDBlock(d_model, d_state=d_state, n_heads=n_heads, expand=expand, chunk=chunk)
            for _ in range(n_layers)
        )
        self.norm_f = nn.RMSNorm(d_model)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, L, d_in) -> h: (B, L, d_model), causal."""
        h = self.embed(feats)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.norm_f(h)

    def init_stream(self, batch: int, device, dtype=torch.float32):
        """Zeroed per-layer (scan state, conv cache) for streaming/simulation."""
        states = []
        for b in self.blocks:
            s = torch.zeros(batch, b.n_heads, b.d_state, b.d_head, device=device)
            c = torch.zeros(batch, b.d_inner + 2 * b.d_state, b.d_conv - 1,
                            device=device, dtype=dtype)
            states.append((s, c))
        return states

    def prefill(self, feats: torch.Tensor, layer_states: list | None = None):
        """Run a (segment of a) sequence, carrying per-layer streaming state."""
        h = self.embed(feats)
        new_states = []
        for i, (norm, block) in enumerate(zip(self.norms, self.blocks)):
            init = layer_states[i] if layer_states is not None else (None, None)
            out, (s, c) = block(
                norm(h), init_state=init[0], conv_init=init[1], return_state=True
            )
            h = h + out
            new_states.append((s, c))
        return self.norm_f(h), new_states

    def step(self, feat: torch.Tensor, layer_states: list):
        """feat: (B, d_in). Advances one event; returns (h (B, d_model), new states)."""
        h = self.embed(feat)
        new_states = []
        for norm, block, (s, c) in zip(self.norms, self.blocks, layer_states):
            out, s2, c2 = block.step(norm(h), s, c)
            h = h + out
            new_states.append((s2, c2))
        return self.norm_f(h), new_states
