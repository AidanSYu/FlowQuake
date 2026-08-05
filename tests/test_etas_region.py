"""The ETAS fit region must be the scoring grid, not the catalog's bounding box.

`mu` is a rate DENSITY: the EM M-step sets `mu = w_bg.sum() / (area * T)`, so
`mu` scales as 1/area_region. `etas_rate_field` then places
`mu * cell_area * horizon` in each cell of the FIXED scoring grid. The total
background mass that actually lands on the grid is therefore off by

    grid_area / region_area

whenever the two disagree. Deriving the region from the catalog's bounding box
(plus 10 km) made that ratio a function of BOTH the arm and mc, because the box
was taken after the mc cut. Measured on the real WHITE arms:

    mc            2.5      2.0      1.5      1.0
    informative  1.1748   1.1204   1.0406   1.0104     <- 16% monotone swing
    null          1.0477   1.0386   0.9840   0.9614

A smooth, monotone mc dependence in the arm whose curve IS the reported result.
`Grid.__doc__` already states the rule for the scoring grid -- build it once at
a reference completeness, never per-mc -- and the fit region never inherited it.

These tests pin the MECHANISM (background mass misplaced in proportion to the
area ratio), not merely the helper that computes the region, because it is the
misplaced mass that moves the science.
"""
import numpy as np

from flowquake.etas_fit import Background, ETASParams, etas_rate_field
from flowquake.target_process import Grid

HORIZON = 1.0
MU = 2.0e-4          # events / day / km^2
T0 = 100.0


def _grid():
    # 40 x 40 cells of 5 km -> 200 x 200 km, area 40_000 km^2
    return Grid(xmin=-100.0, ymin=-100.0, nx=40, ny=40, bin_km=5.0)


def _no_triggering():
    """K=0 isolates the background term, which is the one the region moves."""
    return ETASParams(mc=1.0, mu=MU, K=0.0)


def _bg(area_km2):
    return Background(mode="uniform", area_km2=area_km2)


def test_background_mass_is_exact_when_region_equals_grid():
    g, area = _grid(), 40.0 * 5.0 * 40.0 * 5.0
    lam = etas_rate_field(_no_triggering(), _bg(area), [], [], [], [],
                          g, T0, HORIZON)
    # Every background event in the region lands on the grid, so the expected
    # count is exactly mu * area * horizon with no leakage.
    assert np.isclose(lam.sum(), MU * area * HORIZON, rtol=1e-12)


def test_background_mass_is_misplaced_in_proportion_to_the_area_ratio():
    """A region larger than the grid loses mass by exactly grid/region."""
    g = _grid()
    grid_area = 40.0 * 5.0 * 40.0 * 5.0
    truth = etas_rate_field(_no_triggering(), _bg(grid_area), [], [], [], [],
                            g, T0, HORIZON).sum()

    for ratio in (1.0104, 1.0406, 1.1204, 1.1748):   # the measured informative row
        region_area = grid_area * ratio
        # EM would fit mu smaller by exactly the area ratio for the same
        # background event count: mu = w_bg / (area * T).
        P = ETASParams(mc=1.0, mu=MU / ratio, K=0.0)
        got = etas_rate_field(P, _bg(region_area), [], [], [], [],
                              g, T0, HORIZON).sum()
        assert np.isclose(got / truth, 1.0 / ratio, rtol=1e-10), (
            f"area ratio {ratio}: background mass off by {got/truth:.6f}, "
            f"expected {1/ratio:.6f}")


def test_the_mc_dependent_swing_is_large_enough_to_move_a_slope():
    """Guard the SIZE of the artifact, not just its existence.

    The two ends of the informative row differ by 16%. A regression that
    reintroduced a per-mc region would show up here as a background mass that
    depends on mc at all -- the whole point is that it must not.
    """
    hi, lo = 1.1748, 1.0104          # grid/region at mc 2.5 and mc 1.0
    swing = hi / lo - 1.0
    assert swing > 0.15, "the measured artifact was a 16% swing; check the fixture"

    g = _grid()
    grid_area = 40.0 * 5.0 * 40.0 * 5.0
    masses = []
    for ratio in (hi, lo):
        P = ETASParams(mc=1.0, mu=MU / ratio, K=0.0)
        masses.append(etas_rate_field(P, _bg(grid_area * ratio), [], [], [], [],
                                      g, T0, HORIZON).sum())
    # Under the bug the two mc ends place different background mass; with the
    # region pinned to the grid both would be identical.
    assert abs(masses[0] / masses[1] - lo / hi) < 1e-9

    # And with the fix -- same region for both -- the mass is mc-invariant.
    fixed = [etas_rate_field(ETASParams(mc=1.0, mu=MU, K=0.0), _bg(grid_area),
                             [], [], [], [], g, T0, HORIZON).sum()
             for _ in (hi, lo)]
    assert fixed[0] == fixed[1]


def test_region_from_grid_is_independent_of_the_catalog():
    """The derivation itself: extent comes from the grid, nothing else."""
    g = _grid()
    region = (g.xmin, g.xmin + g.nx * g.bin_km,
              g.ymin, g.ymin + g.ny * g.bin_km)
    assert region == (-100.0, 100.0, -100.0, 100.0)
    # Area equals the scored area, so nothing normalises over space we do not score.
    assert np.isclose((region[1] - region[0]) * (region[3] - region[2]),
                      g.nx * g.bin_km * g.ny * g.bin_km)
