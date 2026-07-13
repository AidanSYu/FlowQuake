"""Pure-PyTorch selective SSM (Mamba / S6) whole-catalog encoder.

No custom CUDA kernels (no mamba_ssm / causal-conv1d / triton) so it runs on
Windows-native PyTorch. The sequential linear recurrence

    h_t = a_t ⊙ h_{t-1} + b_t

is evaluated with a Hillis–Steele parallel associative scan (O(L log L), fully
vectorized over the time axis), which is fast enough to push the *entire* 92k-event
catalog through in a single forward pass (B=1) — giving every test event its full
real history, the way ETAS sees it.

A naive sequential scan is kept for unit-testing the parallel one.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Associative scans for the first-order linear recurrence h_t = a_t h_{t-1} + b_t
#   a, b : [B, L, D, N]  (a_t in (0,1], from exp(Δ·A) with A<0)
#   returns h : [B, L, D, N]   with h_{-1} = 0
# --------------------------------------------------------------------------------------


def selective_scan_sequential(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, L, D, N = a.shape
    h = torch.zeros(B, D, N, dtype=a.dtype, device=a.device)
    out = []
    for t in range(L):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def selective_scan_parallel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive associative scan via combine((A1,B1),(A2,B2)) = (A1*A2, A2*B1+B2)."""
    A = a
    Bx = b
    L = a.shape[1]
    d = 1
    while d < L:
        A_prev = torch.cat([torch.ones_like(A[:, :d]), A[:, : L - d]], dim=1)
        B_prev = torch.cat([torch.zeros_like(Bx[:, :d]), Bx[:, : L - d]], dim=1)
        Bx = A * B_prev + Bx      # uses current A as the "later" decay A2
        A = A_prev * A
        d *= 2
    return Bx


# --------------------------------------------------------------------------------------
# Selective SSM (S6) core
# --------------------------------------------------------------------------------------


class SelectiveSSM(nn.Module):
    """Input-dependent (selective) diagonal SSM, the S6 core of Mamba.

    in/out: [B, L, d_inner].  State dim N per channel. Δ, B, C are functions of the input.
    """

    def __init__(self, d_inner: int, d_state: int = 8, dt_rank: int | None = None,
                 use_parallel_scan: bool = True):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)
        self.use_parallel_scan = use_parallel_scan

        # x -> (Δ_lowrank, B, C)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # A (diagonal, negative): parameterized as -exp(A_log), shape [d_inner, d_state]
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # initialize dt_proj bias so softplus(bias) ~ dt in [1e-3, 1e-1] (Mamba init)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        dt = dt.clamp_min(1e-4)
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log)  # [D, N], negative

        proj = self.x_proj(x)  # [B, L, dt_rank + 2N]
        dt_lr, Bc, Cc = torch.split(proj, [self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(dt_lr))  # [B, L, D] > 0

        # discretize (ZOH for A, Euler for B): dA=exp(Δ A), dB≈Δ·B
        dA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, D, N))       # [B,L,D,N] in (0,1]
        dBx = (delta.unsqueeze(-1) * Bc.unsqueeze(2)) * x.unsqueeze(-1)  # [B,L,D,N]

        scan = selective_scan_parallel if self.use_parallel_scan else selective_scan_sequential
        h = scan(dA, dBx)  # [B, L, D, N]

        y = torch.einsum("bldn,bln->bld", h, Cc) + self.D.view(1, 1, D) * x
        return y


class MambaBlock(nn.Module):
    """One Mamba block: RMSNorm -> in_proj -> causal depthwise conv -> SiLU -> SSM -> gate -> out_proj, with residual."""

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 2,
                 use_parallel_scan: bool = True):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else _RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
                                padding=d_conv - 1, bias=True)
        self.ssm = SelectiveSSM(d_inner, d_state=d_state, use_parallel_scan=use_parallel_scan)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.norm(x)
        xz = self.in_proj(h)
        xin, z = xz.chunk(2, dim=-1)               # [B,L,d_inner] each
        # causal depthwise conv
        xc = self.conv1d(xin.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xc = F.silu(xc)
        y = self.ssm(xc)
        y = y * F.silu(z)                          # gate
        return x + self.out_proj(y)


class _RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


# --------------------------------------------------------------------------------------
# Encoders (return per-step causal hidden states [B, L, d_model])
# --------------------------------------------------------------------------------------


class SelectiveSSMEncoder(nn.Module):
    """Whole-catalog selective-SSM encoder.  state after consuming event i conditions event i+1."""

    def __init__(self, d_in: int = 4, d_model: int = 96, n_layers: int = 3,
                 d_state: int = 8, d_conv: int = 4, expand: int = 2,
                 use_parallel_scan: bool = True):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                       use_parallel_scan=use_parallel_scan)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else _RMSNorm(d_model)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.embed(feats)
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


class GRUEncoder(nn.Module):
    """GRU baseline encoder (the reference dummy's encoder), for the encoder-lever ablation."""

    def __init__(self, d_in: int = 4, d_model: int = 96, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRU(d_in, d_model, num_layers=n_layers, batch_first=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(feats)  # [B, L, d_model], causal by construction
        return out


def build_encoder(kind: str, d_in: int = 4, d_model: int = 96, n_layers: int = 3,
                  **kw) -> nn.Module:
    kind = kind.lower()
    if kind in ("ssm", "mamba", "selective"):
        return SelectiveSSMEncoder(d_in=d_in, d_model=d_model, n_layers=n_layers, **kw)
    if kind == "gru":
        nl = kw.pop("gru_layers", 2)
        return GRUEncoder(d_in=d_in, d_model=d_model, n_layers=nl)
    raise ValueError(f"unknown encoder kind {kind!r}")
