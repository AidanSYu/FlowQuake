"""Shared statistics for paired earthquake-forecast comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GainSummary:
    """Mean paired gain with ordinary SE and autocorrelation-aware CI."""

    n: int
    mean: float
    stderr: float
    ci_low: float
    ci_high: float
    decision: str

    def asdict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "stderr": self.stderr,
            "ci": [self.ci_low, self.ci_high],
            "decision": self.decision,
        }


def ci_decision(ci_low: float, ci_high: float) -> str:
    """Classify a two-sided CI against zero."""

    if ci_low > 0.0:
        return "win"
    if ci_high < 0.0:
        return "loss"
    return "tie"


def stationary_block_bootstrap_ci(
    values,
    *,
    mean_block: int = 50,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Stationary block-bootstrap CI for the mean of a serially ordered series.

    Blocks have geometric lengths with expected size ``mean_block`` and wrap
    around the series. This is designed for paired per-event gains where nearby
    events are correlated by aftershock sequences.
    """

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or np.allclose(x, x[0]):
        m = float(x.mean())
        return m, m

    p_new = 1.0 / max(float(mean_block), 1.0)
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        pos = 0
        total = 0.0
        while pos < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(p_new))
            take = min(length, n - pos)
            idx = (start + np.arange(take)) % n
            total += float(x[idx].sum())
            pos += take
        means[b] = total / n
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def _bootstrap_means(
    values,
    *,
    mean_block: int = 50,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Stationary block-bootstrap distribution of the mean (helper)."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    p_new = 1.0 / max(float(mean_block), 1.0)
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        pos = 0
        total = 0.0
        while pos < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(p_new))
            take = min(length, n - pos)
            idx = (start + np.arange(take)) % n
            total += float(x[idx].sum())
            pos += take
        means[b] = total / n
    return means


def block_bootstrap_pvalue(
    values,
    *,
    mean_block: int = 50,
    n_boot: int = 4000,
    seed: int = 0,
) -> float:
    """Two-sided bootstrap p-value for H0: mean == 0 (add-one smoothed)."""

    means = _bootstrap_means(values, mean_block=mean_block, n_boot=n_boot, seed=seed)
    b = len(means)
    p_lo = (np.sum(means <= 0.0) + 1.0) / (b + 1.0)
    p_hi = (np.sum(means >= 0.0) + 1.0) / (b + 1.0)
    return float(min(1.0, 2.0 * min(p_lo, p_hi)))


def holm_bonferroni(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a dict of {label: p}."""

    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (label, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adjusted[label] = float(min(1.0, running))
    return {k: adjusted[k] for k in pvals}


def tost_equivalence(
    values,
    margin: float,
    *,
    mean_block: int = 50,
    n_boot: int = 4000,
    seed: int = 0,
) -> dict:
    """Bootstrap TOST: equivalent to zero at ``margin`` if the 90% CI of the
    mean lies within (-margin, +margin) (two one-sided 5% tests)."""

    means = _bootstrap_means(values, mean_block=mean_block, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(means, [5.0, 95.0])
    return {
        "margin": float(margin),
        "ci90": [float(lo), float(hi)],
        "equivalent": bool(lo > -margin and hi < margin),
    }


def paired_gain_summary(
    values,
    *,
    mean_block: int = 50,
    n_boot: int = 2000,
    seed: int = 0,
) -> GainSummary:
    """Summarize a paired gain series in nats/event."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return GainSummary(0, float("nan"), float("nan"), float("nan"), float("nan"), "missing")
    mean = float(x.mean())
    stderr = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = stationary_block_bootstrap_ci(
        x, mean_block=mean_block, n_boot=n_boot, seed=seed
    )
    return GainSummary(n, mean, stderr, lo, hi, ci_decision(lo, hi))
