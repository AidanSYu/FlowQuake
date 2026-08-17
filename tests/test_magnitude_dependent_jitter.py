"""Magnitude-dependent location error, the confound the uniform sweep cannot see.

WHY THIS EXISTS. The information ladder reports that the marginal forecast value
of a magnitude band falls to zero below about M1.25, and reads that as a floor in
the earthquake process. The uniform jitter sweep shows that location error of
equal size on every event changes the LEVEL of the curve without changing its
SHAPE, which is reassuring but answers the wrong question. Real networks do not
make uniform errors: small events are recorded by fewer stations at lower SNR, so
they are located worse. That is error correlated with exactly the axis the ladder
measures along, and it could manufacture a floor out of a scale-free truth.

The tilt is sd(m) = jitter_km * 10^(-slope * (m - floor)).

Two properties have to hold or the control is not usable. slope = 0 must
reproduce the uniform path EXACTLY, because every jitter number already quoted
was produced under it and a control that silently moves the baseline proves
nothing. And the tilt must run in the physically correct direction, giving small
events MORE displacement, not less.
"""
import numpy as np
import pytest


def sd_of(mag, jitter_km, slope, floor, min_km=0.0):
    """The scale rule under test, kept identical to information_ladder.py."""
    sd = jitter_km * np.power(10.0, -slope * (np.asarray(mag, float) - floor))
    return np.maximum(sd, min_km) if min_km > 0 else sd


def test_zero_slope_draws_are_bit_identical_to_the_scalar_scale_call():
    """s * normal(0,1) must equal normal(0,s), or published sweeps shift.

    The script was changed from `jr.normal(0.0, jitter_km, size=n)` to
    `jr.normal(0.0, 1.0, size=n) * sd`. If numpy did not draw scaled normals by
    transforming standard ones, that edit would silently invalidate every jitter
    result already in runs/panel_white.
    """
    n, km = 4096, 1.7
    old = np.random.default_rng(0).normal(0.0, km, size=n)
    new = np.random.default_rng(0).normal(0.0, 1.0, size=n) * sd_of(
        np.full(n, 2.0), km, 0.0, 1.0)
    assert np.array_equal(old, new)


def test_zero_slope_is_uniform_whatever_the_magnitudes():
    m = np.array([0.6, 1.0, 2.5, 5.35])
    assert np.allclose(sd_of(m, 1.0, 0.0, 0.75), 1.0)


def test_small_events_are_displaced_more_than_large_ones():
    """The whole point. A tilt that helped small events would be backwards."""
    m = np.array([0.75, 1.5, 2.5, 4.0])
    sd = sd_of(m, 1.0, 0.4, 0.75)
    assert np.all(np.diff(sd) < 0)
    assert sd[0] == pytest.approx(1.0)          # anchored at the floor


def test_slope_is_decades_per_magnitude_unit():
    """One magnitude unit up must divide sd by exactly 10^slope."""
    slope = 0.4
    sd = sd_of([1.0, 2.0, 3.0], 1.0, slope, 1.0)
    assert sd[0] / sd[1] == pytest.approx(10.0 ** slope, rel=1e-9)
    assert sd[1] / sd[2] == pytest.approx(10.0 ** slope, rel=1e-9)


def test_anchor_is_the_floor_not_the_catalog_minimum():
    """jitter_km must keep meaning 'error on the smallest band SCORED'.

    If sd were anchored anywhere else, moving --floor would rescale the entire
    curve and the deep ladder would not be comparable to the shallow one.
    """
    for floor in (0.75, 1.0, 1.5):
        assert sd_of([floor], 2.0, 0.4, floor)[0] == pytest.approx(2.0)


def test_min_km_clamps_the_top_of_the_range_only():
    """Without a clamp the tilt claims metre-scale accuracy on large events."""
    m = np.array([0.75, 2.0, 5.0])
    free = sd_of(m, 1.0, 0.4, 0.75)
    held = sd_of(m, 1.0, 0.4, 0.75, min_km=0.1)
    assert free[-1] < 0.1 and held[-1] == pytest.approx(0.1)
    assert held[0] == pytest.approx(free[0])    # the floor end is untouched


def test_the_ladder_exposes_the_knobs():
    """A control nobody can invoke is not a control."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from scripts.information_ladder import main
    with pytest.raises(SystemExit):
        main(["--help"])
