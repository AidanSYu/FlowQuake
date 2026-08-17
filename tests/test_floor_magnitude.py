"""The floor estimator, which produces the paper's headline number.

WHAT IT HAS TO GET RIGHT. The result is "there is a smallest earthquake worth
detecting, and it is at magnitude X". Two ways to be wrong, and the second is
much worse than the first.

Reporting a floor that is not there. If information really accumulates without
limit, an estimator that always prefers the saturating model would invent a
detection limit and tell the entire fibre-optic community to stop building
sensitivity. So the contest must be losable: on genuinely scale-free data the
saturating model must NOT win.

Reporting the wrong floor. tau is the e-folding scale in magnitude units and
every quoted magnitude descends from it, so an estimator that recovers tau
sloppily moves the headline.

These tests work on synthetic ladders where the truth is known by construction.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.floor_magnitude import (  # noqa: E402
    aic, depth_for_fraction, fit_linear, fit_saturating, summarise,
)

DEPTHS = np.arange(0.25, 1.76, 0.25)          # a 2.5 -> 0.75 ladder, step 0.25


def saturating(d, I_inf, tau):
    return I_inf * (1.0 - np.exp(-d / tau))


def test_saturating_fit_recovers_its_own_parameters():
    for I_inf, tau in [(0.40, 0.70), (1.00, 0.25), (0.15, 1.50)]:
        got_I, got_tau, rss = fit_saturating(DEPTHS, saturating(DEPTHS, I_inf, tau))
        assert got_I == pytest.approx(I_inf, rel=0.02)
        assert got_tau == pytest.approx(tau, rel=0.05)
        assert rss < 1e-6


def test_linear_fit_recovers_its_slope_and_passes_through_the_origin():
    c, rss = fit_linear(DEPTHS, 0.23 * DEPTHS)
    assert c == pytest.approx(0.23, rel=1e-9)
    assert rss < 1e-20


def test_scale_free_truth_does_not_get_called_saturating():
    """The contest has to be losable, or the floor is an artifact of the method.

    Perfectly linear accumulation. The saturating model can mimic it with a long
    tau, but it spends an extra parameter to do so, and AIC must charge it.
    """
    out = summarise(DEPTHS, 0.23 * DEPTHS)
    assert out["d_aic"] < 0


def test_saturating_truth_is_detected():
    out = summarise(DEPTHS, saturating(DEPTHS, 0.40, 0.55))
    assert out["d_aic"] > 0
    assert out["tau"] == pytest.approx(0.55, rel=0.05)


def test_a_hard_floor_is_read_as_a_short_tau():
    """Information that stops dead partway down must not look scale-free."""
    I = np.where(DEPTHS <= 1.0, 0.35 * DEPTHS, 0.35)
    out = summarise(DEPTHS, I)
    assert out["d_aic"] > 0
    assert out["tau"] < 1.0


def test_capture_fraction_at_the_deepest_rung_is_consistent_with_tau():
    I_inf, tau = 0.40, 0.70
    out = summarise(DEPTHS, saturating(DEPTHS, I_inf, tau))
    assert out["captured_at_deepest"] == pytest.approx(
        1.0 - np.exp(-DEPTHS.max() / out["tau"]), rel=1e-9)


def test_depth_for_fraction_inverts_the_model():
    tau = 0.63
    for q in (0.5, 0.9, 0.95, 0.99):
        d = depth_for_fraction(tau, q)
        assert 1.0 - np.exp(-d / tau) == pytest.approx(q, rel=1e-9)
        assert d > 0


def test_deeper_capture_requires_smaller_magnitude():
    """Monotonicity. 99% capture cannot sit above 90% capture."""
    tau = 0.5
    ds = [depth_for_fraction(tau, q) for q in (0.90, 0.95, 0.99)]
    assert ds == sorted(ds)


def test_aic_charges_for_the_extra_parameter():
    """Equal fit, more parameters, worse score. Otherwise the contest is rigged."""
    assert aic(0.01, 7, 2) > aic(0.01, 7, 1)


def test_a_short_ladder_cannot_manufacture_a_floor_from_two_points():
    """Two points fit a straight line exactly, so neither model can be preferred
    on fit alone and AIC must fall back on the parameter count.

    This is the degenerate case a referee will probe: the answer has to be
    'scale-free', because that is the model that explains it with less.
    """
    d = np.array([0.25, 0.50])
    out = summarise(d, 0.23 * d)
    assert out["d_aic"] < 0
