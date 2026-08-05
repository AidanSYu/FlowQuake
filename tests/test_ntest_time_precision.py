"""Absolute event times must be carried in float64 through the simulator.

The catalog's time axis is days since the catalog epoch, so by the test era the
values are 3000-6600. float32 has a 21-42 second quantum there, which is
2400x coarser than the TAU_FLOOR_DAYS=1e-7 (~9 ms) floor the rest of the code
carefully clamps to. Two things break when `t_last` is float32:

  * `lastk_from_bufs` is handed `t_last` (float32) and differences it against
    `t_buf` (float64). At the first sampling step of a window both hold the SAME
    event, so the lag must be exactly zero -- but float32 rounding makes it
    +-21 s at random, and log(21 s) vs log(9 ms) is a 7.8-log-unit swing in a
    feature normalized by log_tau_std ~ 2.8. That is ~2.8 sigma of pure noise
    injected into the single most informative short-term-triggering input, on
    the one step that decides whether the window produces any events at all.

  * `t_next = t_last + tau` silently discards any tau below half a quantum, so
    an event drawn 10 s after its predecessor advances simulated time by
    exactly zero.

The accumulated-drift version of this concern was measured and REFUTED (3e-6
relative over 200 steps, non-monotone in mc -- the sum is dominated by long
gaps and to-nearest rounding cancels). What survives is the first-step lag
corruption above, which is noise rather than bias: it cannot manufacture a
slope, but it degrades exactly the signal the history-conditioning test is
trying to detect.
"""
import numpy as np
import torch

from flowquake.data import BIG_M, LAST_K
from flowquake.ntest import simulate_day_events

TOK = 8          # mocked token width; the real TOKEN_DIM is irrelevant here
TAU = 0.01       # days between simulated events (864 s -- well above the
                 # float32 quantum, so this test is about accuracy, not stalls)

# A day number that float32 cannot represent: float32(4000.3) = 4000.2998046875,
# i.e. 16.9 s low. Any float32 arithmetic on it shows up immediately.
LAST_EVENT_DAY = 4000.3

# The window must open just AFTER the last catalog event: n_hist is
# searchsorted(t_days, day_start, "left"), so a day_start equal to the last
# event excludes it and leaves t_last a whole event too early -- the first-step
# rejection sampler would then demand tau >= 1.0 and reject every draw.
DAY_START = LAST_EVENT_DAY + 1e-6


class _Cat:
    """Minimal stand-in for flowquake.data.Catalog."""

    def __init__(self, t_days):
        e = len(t_days)
        self.t_days = np.asarray(t_days, dtype=np.float64)
        self.feats = torch.zeros(e, TOK)
        self.lastk = torch.zeros(e, LAST_K + BIG_M, 4)
        self.raw = torch.zeros(e, 4)


class _MockModel:
    """Emits a constant tau so the expected event times are exact by hand."""

    encoder = None

    def __init__(self):
        self.stats = {"mcut": 0.0}
        self.first_step_lag = None

    def lastk_from_bufs(self, t_last, bufs, static_big=None):
        if self.first_step_lag is None:
            # t_last and t_buf[:, 0] are the SAME catalog event at this point.
            self.first_step_lag = (t_last - bufs[0][:, 0]).clone()
        return torch.zeros(len(t_last), 1, 4)

    def sample_next(self, h_cur, tok_last, lastk_lane, steps=16):
        n = tok_last.shape[0]
        f = lambda v: torch.full((n,), float(v))
        return f(TAU), f(0.0), f(0.0), f(1.0)

    def build_token(self, tau, x, y, m, t_next, bufs):
        return torch.zeros(tau.shape[0], TOK)


def _run(n_sims=4):
    cat = _Cat([LAST_EVENT_DAY - 1.0, LAST_EVENT_DAY])
    model = _MockModel()
    sims = simulate_day_events(
        model, cat, DAY_START, n_sims, torch.device("cpu"),
        sample_steps=1, horizon_days=1.0)
    return model, sims


def test_first_step_lag_is_exactly_zero():
    """The lag handed to the encoder on step 1 must be 0, not float32 noise."""
    model, _ = _run()
    lag = model.first_step_lag
    assert lag is not None, "lastk_from_bufs was never called"
    # float32 t_last gives |lag| up to 2.4e-4 days (21 s); float64 gives exactly 0.
    assert torch.all(lag == 0.0), (
        f"first-step lag should be identically zero, got max |lag| = "
        f"{lag.abs().max().item():.3e} days "
        f"({lag.abs().max().item() * 86400:.1f} s) -- t_last is not float64")


def test_simulated_times_are_exact_at_test_era_day_numbers():
    """t_next = t_last + tau must not be quantised to the float32 grid."""
    _, sims = _run()
    assert all(len(s) for s in sims), "no events simulated"
    # Step against the float64 value of the float32 tau the model actually
    # emitted. tau is legitimately float32 -- it is a small relative quantity
    # and TAU_FLOOR_DAYS clamps in that space -- so folding its 1e-7 relative
    # error into `expected` isolates what this test is really pinning: the
    # precision of the ABSOLUTE time axis, not of the increment.
    step = float(np.float32(TAU))
    for s in sims:
        t = np.asarray(s)[:, 0]
        expected = LAST_EVENT_DAY + step * np.arange(1, len(t) + 1)
        # float32 t_last is off by ~1e-4 days (8.6 s) on the first event alone,
        # 5 orders of magnitude above this bound; float64 leaves only its own
        # ~1e-10 accumulation over the ~100 events in the window.
        assert np.allclose(t, expected, rtol=0, atol=1e-9), (
            f"max time error {np.abs(t - expected).max():.3e} days "
            f"({np.abs(t - expected).max() * 86400:.4f} s)")


def test_time_axis_dtype_is_float64():
    """Pin the dtype directly so a future edit cannot silently downcast."""
    _, sims = _run(n_sims=2)
    for s in sims:
        assert np.asarray(s)[:, 0].dtype == np.float64
