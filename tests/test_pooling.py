"""Tests for slope uncertainty and cross-region pooling.

Both failure modes these guard against produce intervals that are too NARROW,
which is the direction that turns a null result into a claimed discovery.
"""

import numpy as np
import pytest

from flowquake.pooling import (
    PanelSlope, block_bootstrap_slope, random_effects_pool, slope_per_decade,
)


def make_panel(mcs, true_slope, n_win=50, noise=0.05, seed=0,
               concentrated=False, base=-7.0):
    """Per-window shape scores whose target-weighted slope is `true_slope`.

    `concentrated` reproduces the real hazard: one window holding most of the
    target events, as the 2019 Ridgecrest sequence does inside the ComCat mask.
    """
    r = np.random.default_rng(seed)
    if concentrated:
        tgt = np.ones(n_win)
        tgt[0] = 10.0 * n_win
    else:
        tgt = r.integers(1, 6, n_win).astype(float)
    per_mc = base + np.array([-true_slope * (m - mcs[0]) for m in mcs])
    scores = np.empty((len(mcs), n_win))
    for i, mu in enumerate(per_mc):
        scores[i] = (mu + r.normal(0, noise, n_win)) * tgt
    return scores, tgt


def test_slope_sign_convention_is_gain_per_decade_lowered():
    """Positive slope must mean 'better as the catalog goes deeper'."""
    mcs = [2.5, 2.0, 1.5, 1.0]
    improving = [-7.0, -6.8, -6.6, -6.4]      # score rises as mc falls
    assert slope_per_decade(mcs, improving) > 0
    assert slope_per_decade(mcs, improving[::-1]) < 0


def test_block_bootstrap_recovers_a_planted_slope():
    mcs = [2.5, 2.0, 1.5, 1.0]
    scores, tgt = make_panel(mcs, 0.30, seed=1)
    p = block_bootstrap_slope("t", mcs, scores, tgt, n_boot=500, seed=0)
    assert p.slope == pytest.approx(0.30, abs=0.05)
    assert p.ci_lo < 0.30 < p.ci_hi


def test_flat_panel_interval_covers_zero():
    mcs = [2.5, 2.0, 1.5, 1.0]
    scores, tgt = make_panel(mcs, 0.0, seed=2)
    p = block_bootstrap_slope("flat", mcs, scores, tgt, n_boot=500, seed=0)
    assert p.ci_lo < 0.0 < p.ci_hi


def test_concentrated_targets_widen_the_interval():
    """THE load-bearing test for within-panel uncertainty.

    When one window holds most of the target events, the effective sample size
    is close to one sequence, and the interval must reflect that. An
    event-level bootstrap would report roughly the same width in both cases --
    which is exactly how a Ridgecrest-dominated panel gets mistaken for 126
    independent observations.
    """
    mcs = [2.5, 2.0, 1.5, 1.0]
    spread, t1 = make_panel(mcs, 0.30, seed=3, concentrated=False)
    conc, t2 = make_panel(mcs, 0.30, seed=3, concentrated=True)
    a = block_bootstrap_slope("spread", mcs, spread, t1, n_boot=800, seed=0)
    b = block_bootstrap_slope("conc", mcs, conc, t2, n_boot=800, seed=0)
    assert b.se > 2 * a.se, (a.se, b.se)


def test_bootstrap_pairs_windows_across_mc():
    """Windows must be resampled once per replicate, not per mc.

    With a common per-window offset shared across every mc, the slope is
    unaffected by which windows are drawn, so a correctly paired bootstrap
    gives a very tight interval. Resampling independently per mc would let the
    offsets disagree between points and inflate the spread.
    """
    mcs = [2.5, 2.0, 1.5, 1.0]
    r = np.random.default_rng(0)
    n_win = 40
    tgt = np.ones(n_win)
    offset = r.normal(0, 3.0, n_win)              # huge shared window effect
    per_mc = np.array([-0.30 * (m - mcs[0]) for m in mcs])
    scores = np.stack([(mu + offset) * tgt for mu in per_mc])
    p = block_bootstrap_slope("paired", mcs, scores, tgt, n_boot=400, seed=0)
    assert p.se < 1e-6, p.se
    assert p.slope == pytest.approx(0.30, abs=1e-6)


# --- cross-region pooling ---------------------------------------------------

def _panel(name, slope, se):
    return PanelSlope(name, slope, se, slope - 2 * se, slope + 2 * se, 40, 100)


def test_pool_of_agreeing_panels_is_tighter_than_any_one():
    ps = [_panel("a", 0.30, 0.05), _panel("b", 0.32, 0.05), _panel("c", 0.28, 0.05)]
    pooled = random_effects_pool(ps)
    assert pooled.estimate == pytest.approx(0.30, abs=0.02)
    assert pooled.se < min(p.se for p in ps)
    assert pooled.tau2 == pytest.approx(0.0, abs=1e-4)
    assert not pooled.heterogeneous


def test_disagreeing_panels_are_flagged_not_averaged_away():
    """Heterogeneity is a result, not a nuisance to smooth over."""
    ps = [_panel("a", 0.10, 0.02), _panel("b", 0.90, 0.02), _panel("c", 0.50, 0.02)]
    pooled = random_effects_pool(ps)
    assert pooled.tau2 > 0.0
    assert pooled.i2 > 0.9
    assert pooled.heterogeneous
    # and the interval must be wider than a fixed-effect pool would give
    fixed_se = (sum(1.0 / p.variance for p in ps)) ** -0.5
    assert pooled.se > fixed_se


def test_random_effects_never_narrower_than_fixed_effect():
    for slopes in ([0.3, 0.3, 0.3], [0.1, 0.5, 0.9], [0.2, 0.25, 0.6]):
        ps = [_panel(str(i), s, 0.04) for i, s in enumerate(slopes)]
        pooled = random_effects_pool(ps)
        fixed_se = (sum(1.0 / p.variance for p in ps)) ** -0.5
        assert pooled.se >= fixed_se - 1e-12


def test_single_panel_pool_returns_that_panel():
    pooled = random_effects_pool([_panel("only", 0.25, 0.06)])
    assert pooled.estimate == pytest.approx(0.25)
    assert pooled.tau2 == 0.0


def test_pool_rejects_empty_input():
    with pytest.raises(ValueError):
        random_effects_pool([])


def test_summary_renders():
    ps = [_panel("white", 0.30, 0.05), _panel("qtm", 0.22, 0.08)]
    s = random_effects_pool(ps).summary()
    assert "POOLED (RE)" in s and "white" in s and "I^2" in s
