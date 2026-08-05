"""Slope uncertainty within a panel, and pooling slopes across panels.

Needed because the moonshot went multi-region (see MOONSHOT.md, "The data
plan"). No single catalog gives both magnitude range and target count: dense
template-matched catalogs reach mc 0.6 over ~90 km and 13 years, while ComCat
reaches mc 2.5 over the whole state for 25 years. Each panel therefore carries
only ~1-1.5 decades, and the headline number has to combine them.

Two things have to be right, and both are easy to get wrong in the flattering
direction.

WITHIN a panel, target events are not independent. They arrive in aftershock
sequences: inside the ComCat completeness mask 87% of the M>=4 targets were the
2019 Ridgecrest sequence and a single 30-day window held 85% of them. An
event-level bootstrap would treat that as ~126 independent observations and
report a confidence interval several times too narrow. `block_bootstrap_slope`
resamples WINDOWS, and resamples the same windows at every mc so the pairing
across the x-axis survives -- the slope is a within-window contrast, and
breaking the pairing would inflate the variance instead.

ACROSS panels, regions are not replicates. They differ in tectonics, network
geometry, catalog span and M_TARGET. A fixed-effect pool assumes one true slope
measured repeatedly and would understate the uncertainty; `random_effects_pool`
(DerSimonian-Laird) estimates the between-region variance and widens the
interval accordingly. If the regions genuinely disagree, that shows up as tau
and I-squared rather than being averaged away -- and heterogeneity is itself a
result worth reporting, since a limit that varies by region is a different
claim from a universal one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["slope_per_decade", "block_bootstrap_slope", "PanelSlope",
           "random_effects_pool", "PooledSlope"]


def slope_per_decade(mcs: np.ndarray, scores: np.ndarray) -> float:
    """Nats gained per decade of mc LOWERED.

    `mc` runs downward along the figure's x-axis, so the reported slope negates
    the ordinary least-squares gradient: positive means skill improves as the
    catalog goes deeper.
    """
    mcs = np.asarray(mcs, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(mcs) & np.isfinite(scores)
    if ok.sum() < 2:
        return float("nan")
    return float(-np.polyfit(mcs[ok], scores[ok], 1)[0])


@dataclass
class PanelSlope:
    """One region's slope with a block-bootstrap interval."""
    name: str
    slope: float
    se: float
    ci_lo: float
    ci_hi: float
    n_windows: int
    n_targets: int
    mcs: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def variance(self) -> float:
        return float(self.se ** 2)


#: Default moving-block length, in forecast windows. The windows are
#: CONSECUTIVE 1-day forecasts, so a block of 1 window is a single day — far
#: shorter than the dependence it is supposed to absorb. Aftershock sequences
#: decay by Omori and stay correlated for weeks, so an i.i.d. resample of days
#: reports intervals that are too narrow. 30 days is the shortest block that
#: covers the bulk of a typical sequence's window-to-window correlation here.
DEFAULT_BLOCK_WINDOWS = 30


def _resample(rng, n_win: int, block_len: int) -> np.ndarray:
    """Index draw for one replicate — CIRCULAR moving block.

    `block_len=1` is the i.i.d. bootstrap over windows — kept only to reproduce
    numbers computed before the moving-block fix, never as a default.

    Circular (wrap-around) rather than plain moving-block, because the plain
    version degenerates. With starts restricted to [0, n - L] and L >= n there is
    exactly one legal start, so every replicate returns the identity permutation,
    the resampling variance is zero and the interval collapses onto the point
    estimate. `tests/test_pooling.py::test_flat_panel_interval_covers_zero` caught
    exactly that on a short panel. Wrapping keeps n legal starts at any L, so the
    draw stays a genuine resample.

    `block_len` is also capped at n/4 so a replicate always contains at least
    four blocks; with one or two blocks the "bootstrap" is little more than a
    random rotation of the original series.
    """
    if block_len <= 1:
        return rng.integers(0, n_win, n_win)
    block_len = max(1, min(int(block_len), max(1, n_win // 4)))
    n_blocks = int(np.ceil(n_win / block_len))
    starts = rng.integers(0, n_win, n_blocks)               # any start, wraps
    idx = (starts[:, None] + np.arange(block_len)[None, :]) % n_win
    return idx.ravel()[:n_win]


def block_bootstrap_slope(
    name: str,
    mcs: list[float],
    window_scores: np.ndarray,
    window_targets: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    block_len: int = DEFAULT_BLOCK_WINDOWS,
) -> PanelSlope:
    """Slope and CI, resampling CONTIGUOUS BLOCKS of forecast windows.

    HISTORY, because this was wrong and the wrongness was invisible. The
    function was named "block bootstrap" and its docstring claimed to handle the
    fact that target events arrive in aftershock sequences — but it drew
    `rng.integers(0, n_win, n_win)`, an i.i.d. resample of INDIVIDUAL windows.
    The windows are consecutive 1-day forecasts (verified: every start_days
    difference is exactly 1.0), so the resampled unit was one day while the unit
    of dependence is a sequence lasting weeks. Every interval this module
    produced was therefore too narrow, on both the G1 pooled slopes and the G3
    increments. An adversarial review caught it; nothing in the test suite did,
    because an i.i.d. bootstrap is perfectly well-behaved — just answering a
    different question.

    `block_len` windows are now drawn as contiguous runs, which carries the
    within-sequence correlation into the replicate. Set `block_len=1` to
    reproduce the old (too narrow) numbers.

    `window_scores` is (n_mc, n_windows): the per-window score (use the SHAPE
    term, invariant 1c) for each point on the grid. `window_targets` is
    (n_windows,), the target-event count per window, shared across mc by
    invariant 1.

    Windows are resampled ONCE per replicate and applied to every mc, which
    keeps each replicate an honest re-run of the same experiment. Resampling
    independently per mc would destroy the pairing that makes the slope a
    within-window contrast and would inflate the interval rather than correct
    it.

    The per-mc aggregate is the target-weighted mean, matching the
    nats-per-target-event convention used everywhere else, so a replicate that
    happens to draw the big sequence twice is weighted as such.
    """
    scores = np.asarray(window_scores, dtype=float)
    tgt = np.asarray(window_targets, dtype=float)
    n_mc, n_win = scores.shape
    if len(tgt) != n_win:
        raise ValueError(f"window_targets has {len(tgt)}, expected {n_win}")
    if len(mcs) != n_mc:
        raise ValueError(f"mcs has {len(mcs)}, expected {n_mc}")

    def agg(idx):
        w = tgt[idx]
        tot = w.sum()
        if tot <= 0:
            return np.full(n_mc, np.nan)
        return scores[:, idx].sum(axis=1) / tot

    point = slope_per_decade(mcs, agg(np.arange(n_win)))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _resample(rng, n_win, block_len)
        draws[i] = slope_per_decade(mcs, agg(idx))
    draws = draws[np.isfinite(draws)]
    if len(draws) < 2:
        return PanelSlope(name, point, float("nan"), float("nan"),
                          float("nan"), n_win, int(tgt.sum()),
                          list(mcs), list(agg(np.arange(n_win))))

    a = (1.0 - ci) / 2.0
    return PanelSlope(
        name=name, slope=point, se=float(draws.std(ddof=1)),
        ci_lo=float(np.quantile(draws, a)), ci_hi=float(np.quantile(draws, 1 - a)),
        n_windows=n_win, n_targets=int(tgt.sum()),
        mcs=list(map(float, mcs)), scores=[float(v) for v in agg(np.arange(n_win))],
    )


@dataclass
class PooledSlope:
    estimate: float
    se: float
    ci_lo: float
    ci_hi: float
    tau2: float
    i2: float
    q: float
    q_df: int
    q_p: float
    panels: list[PanelSlope] = field(default_factory=list)

    @property
    def heterogeneous(self) -> bool:
        """Regions disagree by more than sampling noise explains."""
        return self.q_p < 0.10

    def summary(self) -> str:
        out = [f"{'panel':18}{'slope':>9}{'95% CI':>22}{'windows':>9}{'targets':>9}"]
        for p in self.panels:
            out.append(f"{p.name:18}{p.slope:>9.4f}"
                       f"{f'[{p.ci_lo:+.4f}, {p.ci_hi:+.4f}]':>22}"
                       f"{p.n_windows:>9}{p.n_targets:>9}")
        out.append(f"{'POOLED (RE)':18}{self.estimate:>9.4f}"
                   f"{f'[{self.ci_lo:+.4f}, {self.ci_hi:+.4f}]':>22}")
        out.append(f"  tau^2={self.tau2:.5f}  I^2={self.i2:.0%}  "
                   f"Q={self.q:.2f} (df={self.q_df}, p={self.q_p:.3f})"
                   f"{'  <- REGIONS DISAGREE' if self.heterogeneous else ''}")
        return "\n".join(out)


def random_effects_pool(panels: list[PanelSlope], ci: float = 0.95) -> PooledSlope:
    """DerSimonian-Laird random-effects pool across regions.

    Fixed-effect pooling would assume every region measures the same underlying
    slope and that the only scatter is sampling noise. That is false here by
    construction -- the panels differ in tectonics, network, span and even
    M_TARGET -- and assuming it would produce an interval far too narrow for
    the claim being made.

    Reports tau^2 (between-region variance), I^2 (share of total variance that
    is between-region rather than within), and Cochran's Q. High heterogeneity
    is NOT a failure to explain away: "the information limit differs by region"
    is a different and more interesting claim than a single universal number,
    and it must be reported as such rather than buried under an average.
    """
    from scipy import stats

    good = [p for p in panels if math.isfinite(p.slope) and math.isfinite(p.se)
            and p.se > 0]
    if len(good) == 0:
        raise ValueError("no usable panels")
    y = np.array([p.slope for p in good], dtype=float)
    v = np.array([p.variance for p in good], dtype=float)

    if len(good) == 1:
        z = stats.norm.ppf(1 - (1 - ci) / 2)
        se = float(np.sqrt(v[0]))
        return PooledSlope(float(y[0]), se, float(y[0] - z * se),
                           float(y[0] + z * se), 0.0, 0.0, 0.0, 0, 1.0, panels)

    w = 1.0 / v
    fixed = float((w * y).sum() / w.sum())
    q = float((w * (y - fixed) ** 2).sum())
    df = len(good) - 1
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0

    wr = 1.0 / (v + tau2)
    est = float((wr * y).sum() / wr.sum())
    se = float(math.sqrt(1.0 / wr.sum()))
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0

    return PooledSlope(
        estimate=est, se=se, ci_lo=est - z * se, ci_hi=est + z * se,
        tau2=float(tau2), i2=float(i2), q=q, q_df=df,
        q_p=float(1.0 - stats.chi2.cdf(q, df)) if df > 0 else 1.0,
        panels=panels,
    )
