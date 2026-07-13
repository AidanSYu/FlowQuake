import numpy as np

from flowquake.stats import ci_decision, paired_gain_summary, stationary_block_bootstrap_ci


def test_stationary_block_bootstrap_constant_series_is_exact():
    lo, hi = stationary_block_bootstrap_ci(np.full(200, 0.07), n_boot=100)
    assert np.isclose(lo, 0.07)
    assert np.isclose(hi, 0.07)


def test_paired_gain_summary_classifies_clear_win_and_loss():
    win = paired_gain_summary(np.linspace(0.01, 0.05, 200), n_boot=200, seed=1)
    loss = paired_gain_summary(np.linspace(-0.05, -0.01, 200), n_boot=200, seed=2)
    assert win.decision == "win"
    assert loss.decision == "loss"


def test_ci_decision_tie_when_interval_crosses_zero():
    assert ci_decision(-0.01, 0.02) == "tie"
