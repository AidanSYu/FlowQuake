"""A small REAL model plus catalog, for tests that need genuine randomness.

The mock model in `test_ntest_time_precision.py` emits a constant tau, which is
right for testing time arithmetic and useless here: a deterministic model would
satisfy "same seed reproduces" trivially and could not detect a broken seed. So
this builds an actual `FlowQuakeTPP` with random weights and drives the real
sampling path (flow ODE, spatial head, magnitude head).

`h_bottleneck=0` matches every reported config, and `simulate_windows` requires
it -- the batched path raises NotImplementedError when an encoder is present
because it carries no per-lane SSM state.
"""
from __future__ import annotations

import numpy as np
import torch

from flowquake.data import BIG_M, LAST_K, RECENCY_LAGS, TOKEN_DIM
from flowquake.model import FlowQuakeTPP

N_EVENTS = 64
LAST_EVENT_DAY = 100.0
#: Windows open just after the final catalog event. A start equal to the last
#: event would exclude it (n_hist uses searchsorted "left"), leaving t_last a
#: whole event early and making the first-step rejection sampler demand an
#: impossible tau -- the failure mode already documented in
#: test_ntest_time_precision.py.
WINDOW_STARTS = [LAST_EVENT_DAY + 1e-6, LAST_EVENT_DAY + 1.0]

STATS = {
    "log_tau_mean": -1.0, "log_tau_std": 1.0,
    "x_mean": 0.0, "x_std": 10.0, "y_mean": 0.0, "y_std": 10.0,
    "mag_mean": 3.0, "mag_std": 0.5, "mcut": 2.0, "bg_area": 1e4,
    "bg_xmin": -50.0, "bg_xmax": 50.0, "bg_ymin": -50.0, "bg_ymax": 50.0,
    "rec_mean": [0.0] * (4 * len(RECENCY_LAGS)),
    "rec_std": [1.0] * (4 * len(RECENCY_LAGS)),
}


class TinyCatalog:
    """Minimal stand-in for flowquake.data.Catalog (the fields ntest reads)."""

    def __init__(self, t_days, raw):
        self.t_days = np.asarray(t_days, dtype=np.float64)
        self.raw = raw
        self.feats = torch.zeros(len(t_days), TOKEN_DIM)
        self.lastk = torch.zeros(len(t_days), LAST_K + BIG_M, 4)
        self.stats = STATS


def make_tiny_model_and_catalog(seed: int = 0):
    """Return (model, catalog, window_starts) ready for `simulate_windows`."""
    torch.manual_seed(seed)
    model = FlowQuakeTPP(
        d_model=16, n_layers=1, d_state=8, n_heads=2, flow_hidden=16,
        mix_hidden=16, flow_layers=2, h_bottleneck=0, stats=dict(STATS),
    ).eval()

    t = np.linspace(0.0, LAST_EVENT_DAY, N_EVENTS)
    rng = np.random.default_rng(seed)
    raw = torch.zeros(N_EVENTS, 4)
    raw[:, 0] = torch.as_tensor(t, dtype=torch.float32)
    raw[:, 1] = torch.as_tensor(rng.uniform(-20, 20, N_EVENTS), dtype=torch.float32)
    raw[:, 2] = torch.as_tensor(rng.uniform(-20, 20, N_EVENTS), dtype=torch.float32)
    raw[:, 3] = torch.as_tensor(rng.uniform(2.0, 4.0, N_EVENTS), dtype=torch.float32)

    return model, TinyCatalog(t, raw), list(WINDOW_STARTS)
