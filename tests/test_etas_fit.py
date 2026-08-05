"""Tests for the self-contained ETAS (scaling-curve control + gate G1).

Covers the properties everything downstream depends on: exactly normalised
kernels (so the log-likelihood is a proper point-process likelihood and
comparable with FlowQuake's), a tractable pair builder, parameter recovery on
data with known truth, and a simulator whose output feeds the SAME
`flowquake.target_process` scorer the neural model does.
"""

import math

import numpy as np
import pytest

from flowquake.etas_fit import (
    Background, ETASParams, branching_ratio, fit_etas_em, log_likelihood,
    omori, omori_int, parent_pairs, productivity, radius_for_tol,
    simulate_etas, spatial,
)

REGION = (-200.0, 200.0, -200.0, 200.0)
AREA = 400.0 ** 2
TRUE = ETASParams(mu=8e-6, K=0.35, a=0.5, c=0.02, p=1.2, d=1.5, gamma=0.6,
                  q=1.8, mc=2.5)


def test_omori_is_a_normalised_density():
    from scipy import integrate
    val, _ = integrate.quad(lambda s: omori(np.array([s]), TRUE)[0], 0, np.inf,
                            limit=400)
    assert val == pytest.approx(1.0, abs=1e-8)


def test_omori_int_matches_quadrature():
    from scipy import integrate
    for dt in (0.1, 1.0, 100.0, 3650.0):
        num, _ = integrate.quad(lambda s: omori(np.array([s]), TRUE)[0], 0, dt,
                                limit=400)
        assert omori_int(np.array([dt]), TRUE)[0] == pytest.approx(num, abs=1e-8)


def test_spatial_is_a_normalised_density():
    from scipy import integrate
    for dm in (0.0, 2.0):
        val, _ = integrate.quad(
            lambda r: 2 * math.pi * r * spatial(np.array([r * r]),
                                                np.array([TRUE.mc + dm]), TRUE)[0],
            0, np.inf, limit=400)
        assert val == pytest.approx(1.0, abs=1e-6)


def test_radius_for_tol_matches_the_exact_tail():
    for dm in (0.0, 1.5, 3.0):
        R = radius_for_tol(TRUE.mc + dm, TRUE, tol=0.01)
        d2 = TRUE.d ** 2 * 10.0 ** (TRUE.gamma * dm)
        assert (1.0 + R * R / d2) ** (1.0 - TRUE.q) == pytest.approx(0.01, rel=1e-9)


def test_radius_scales_with_parent_magnitude():
    """A single global cutoff cannot serve both ends of the magnitude range.

    d_j = d * 10^(gamma*dm/2), so the radius ratio over 3 magnitude units is
    exactly 10^(gamma*3/2) ~ 7.9x here. That spread is why the cutoff must be
    per-parent rather than global."""
    small = radius_for_tol(TRUE.mc, TRUE, 0.01)
    big = radius_for_tol(TRUE.mc + 3.0, TRUE, 0.01)
    assert big / small == pytest.approx(10.0 ** (TRUE.gamma * 3.0 / 2.0), rel=1e-9)
    assert big / small > 5.0


def test_parent_pairs_is_subquadratic_and_causal():
    rng = np.random.default_rng(0)
    n = 20000
    t = np.sort(rng.uniform(0, 3650, n))
    x, y = rng.uniform(-200, 200, n), rng.uniform(-200, 200, n)
    m = TRUE.mc + rng.exponential(0.434, n)
    total = 0
    for ci, pj in parent_pairs(t, x, y, m, TRUE):
        assert np.all(pj < ci), "a parent must precede its child"
        assert np.all(t[ci] - t[pj] > 0)
        total += len(ci)
    # the O(E^2) version produced ~n^2/2 = 2e8 here; anything near that is a bug
    assert total < n * 400


def test_simulator_output_matches_scorer_contract():
    """Columns must be [t, x, y, m], in-window and in-region — the format
    flowquake.target_process consumes for BOTH models."""
    bg = Background(mode="uniform", area_km2=AREA)
    sims = simulate_etas(TRUE, bg, [], [], [], [], 100.0, 30.0, REGION,
                         n_sims=20, seed=1)
    assert len(sims) == 20
    for ev in sims:
        assert ev.ndim == 2 and ev.shape[1] == 4
        if len(ev):
            assert np.all(ev[:, 0] >= 100.0) and np.all(ev[:, 0] < 130.0)
            assert np.all(np.abs(ev[:, 1]) <= 200.0)
            assert np.all(ev[:, 3] >= TRUE.mc)


def test_simulated_background_rate_is_unbiased():
    """With K=0 there are no aftershocks, so the count is pure background."""
    P0 = ETASParams(mu=1e-4, K=0.0, mc=2.5)
    bg = Background(mode="uniform", area_km2=AREA)
    sims = simulate_etas(P0, bg, [], [], [], [], 0.0, 50.0, REGION, n_sims=300,
                         seed=2)
    expected = P0.mu * AREA * 50.0
    got = float(np.mean([len(e) for e in sims]))
    assert got == pytest.approx(expected, rel=0.1)


def test_history_drives_aftershocks():
    """A large recent event must raise the rate by its ANALYTIC expectation.

    The previous assertion was `loud > 3 * quiet`, and it passed against a
    simulator that double-thinned the first generation. That bug made the
    expected first-generation count sum_i w_i^2 instead of sum_i w_i, which is
    not a uniform error: it INFLATES parents with w_i > 1 and SUPPRESSES those
    with w_i < 1. This fixture has a single M6 with w = 8.1, so the bug pushed
    its contribution to ~65 and the loose ratio test sailed through. Real
    catalogs are dominated by small parents with w << 1, where the same bug
    crushes triggering -- so the old assertion was not merely weak, it was
    passing *because* of the defect, and in the direction that matters least.

    Assert against the closed form instead, which is checked independently by
    `test_simulator_matches_analytic_first_generation`.
    """
    from flowquake.etas_fit import omori_int, productivity

    bg = Background(mode="uniform", area_km2=AREA)
    common = dict(t0=100.0, horizon_days=10.0, region=REGION, n_sims=400, seed=5)
    quiet = simulate_etas(TRUE, bg, [], [], [], [], **common)
    loud = simulate_etas(TRUE, bg, [99.9], [0.0], [0.0], [6.0], **common)
    mq = np.mean([len(e) for e in quiet])
    ml = np.mean([len(e) for e in loud])

    # expected DIRECT offspring of the M6 inside the window
    w = float((productivity(np.array([6.0]), TRUE)
               * (omori_int(np.array([10.1]), TRUE)
                  - omori_int(np.array([0.1]), TRUE)))[0])
    assert w > 1.0, "fixture should have a substantial trigger"
    # the increment must be at least the direct term, and no more than the
    # direct term amplified by the cascade
    assert ml - mq >= 0.8 * w, (mq, ml, w)
    assert ml - mq <= 3.0 * w, (mq, ml, w)


@pytest.mark.slow
def test_em_recovers_known_parameters():
    """Identifiable parameters within 25%; (d, q) are excluded because they are
    near-degenerate in this kernel — see fit_etas_em's closing note."""
    bg = Background(mode="uniform", area_km2=AREA)
    cat = simulate_etas(TRUE, bg, [], [], [], [], 0.0, 3650.0, REGION, n_sims=1,
                        seed=3, max_events=60000)[0]
    t, x, y, m = cat.T
    P, _, hist = fit_etas_em(t, x, y, m, mc=2.5, region=REGION,
                            background="uniform", n_iter=4)
    for k in ("mu", "K", "a", "p", "gamma"):
        assert getattr(P, k) == pytest.approx(getattr(TRUE, k), rel=0.25), k
    assert hist["best_ll"] == max(hist["ll"])
    assert branching_ratio(P) == pytest.approx(branching_ratio(TRUE), rel=0.25)


@pytest.mark.slow
def test_em_stays_subcritical_under_misspecification():
    """A mis-specified region must not produce a runaway inversion.

    Regression: with the region set ~56x too large, mu collapses to ~1e-11 and
    the unconstrained EM attributed everything to triggering, reaching a
    branching ratio of 469.9 (K = 321.9). An ETAS with n >= 1 is non-stationary
    and its simulations do not terminate, so it is useless as a forecast at any
    likelihood. The M-step barrier bounds it.
    """
    bg = Background(mode="uniform", area_km2=AREA)
    cat = simulate_etas(TRUE, bg, [], [], [], [], 0.0, 3650.0, REGION, n_sims=1,
                        seed=3, max_events=60000)[0]
    t, x, y, m = cat.T
    for region in (REGION, (-1500.0, 1500.0, -1500.0, 1500.0)):
        P, _, _ = fit_etas_em(t, x, y, m, mc=2.5, region=region,
                              background="uniform", n_iter=3)
        assert branching_ratio(P) < 1.0


def test_projection_matches_earthquakenpp_convention():
    """EarthquakeNPP stores (x, y) = (NORTHING, EASTING). Reproducing its
    catalog requires that swap; without it the frame is mirrored and every
    downstream km coordinate is wrong. Skipped when the clone is absent."""
    import sys
    from pathlib import Path
    ref = Path("reference/Datasets/ComCat/ComCat_catalog.csv")
    if not ref.exists():
        pytest.skip("benchmark clone not present")
    import pandas as pd
    sys.path.insert(0, "scripts")
    from build_comcat_lowmc import azimuthal_equidistant
    d = pd.read_csv(ref)
    x, y = azimuthal_equidistant(d.latitude.to_numpy(), d.longitude.to_numpy(),
                                 d.latitude.mean(), d.longitude.mean())
    assert np.max(np.hypot(x - d.x.to_numpy(), y - d.y.to_numpy())) < 1e-6


# ---------------------------------------------------------------------------
# Simulator vs the closed-form intensity.
#
# The bug this guards against was silent and severe: the first generation of
# aftershocks from OBSERVED history was double-thinned. The caller drew
# k ~ Poisson(w_i) and then passed each parent repeated k times into
# `offspring`, which drew its own Poisson -- so the expected first-generation
# count was sum_i w_i^2 instead of sum_i w_i. With w_i < 1 that suppresses
# history-driven triggering quadratically (measured 0.102 against an analytic
# 0.277), leaving the constant background to dominate. The forecast came out
# nearly constant and corr(n_expected, n_observed) was about zero on real data,
# which reads as "ETAS has no skill here" rather than "the simulator is wrong".
# ---------------------------------------------------------------------------

def _history(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    return (np.sort(rng.uniform(0, 2000, n)), rng.normal(0, 25, n),
            rng.normal(0, 25, n), 1.0 + rng.exponential(0.43, n))


def _params():
    from flowquake.etas_fit import Background, ETASParams
    P = ETASParams(mu=2e-6, K=0.3, a=0.6, c=0.05, p=1.15, d=1.0,
                   gamma=0.5, q=1.7, mc=1.0)
    bg = Background(mode="uniform", area_km2=40000.0, grid=None,
                    xmin=-100, ymin=-100, bin_km=5.0)
    return P, bg


def test_simulator_matches_analytic_first_generation():
    """THE regression. The simulator must EXCEED the first-generation closed
    form (it adds later generations) but stay within a sane factor of it."""
    from flowquake.etas_fit import etas_rate_field, simulate_etas
    from flowquake.target_process import Grid, TargetSpec, rate_field

    P, bg = _params()
    t, x, y, m = _history()
    grid = Grid.from_bounds(np.array([-100.0, 100.0]), np.array([-100.0, 100.0]),
                            bin_km=5.0)
    spec = TargetSpec(m_target=1.0, m_large=4.0, horizon_days=1.0,
                      tail_mode="fixed", mc=1.0, b_value=1.0)

    analytic = etas_rate_field(P, bg, t, x, y, m, grid, 2001.0, 1.0).sum()
    sims = simulate_etas(P, bg, t, x, y, m, 2001.0, 1.0, (-100, 100, -100, 100),
                         n_sims=20_000, b_value=1.0, seed=1)
    simulated = rate_field(sims, grid, spec).sum()

    assert analytic > 0
    ratio = simulated / analytic
    assert 1.0 <= ratio <= 1.6, (
        f"simulator/analytic = {ratio:.3f}. Below 1 means the history-driven "
        f"first generation is being thinned twice; far above 1.6 means the "
        f"cascade is running away.")


def test_analytic_field_scales_linearly_with_productivity():
    """Sanity on the closed form: doubling K doubles the triggered part."""
    from dataclasses import replace

    from flowquake.etas_fit import etas_rate_field
    from flowquake.target_process import Grid

    P, bg = _params()
    t, x, y, m = _history()
    grid = Grid.from_bounds(np.array([-100.0, 100.0]), np.array([-100.0, 100.0]),
                            bin_km=5.0)
    base_bg = etas_rate_field(replace(P, K=0.0), bg, t, x, y, m, grid, 2001.0, 1.0).sum()
    a = etas_rate_field(P, bg, t, x, y, m, grid, 2001.0, 1.0).sum() - base_bg
    b = etas_rate_field(replace(P, K=P.K * 2), bg, t, x, y, m, grid,
                        2001.0, 1.0).sum() - base_bg
    assert b == pytest.approx(2 * a, rel=1e-9)


def test_analytic_field_responds_to_recent_history():
    """A large recent event must raise the forecast; the whole point of ETAS.

    If this fails the control has no conditioning and cannot serve as the
    guaranteed-skill probe (MOONSHOT.md invariant 1f).
    """
    from flowquake.etas_fit import etas_rate_field
    from flowquake.target_process import Grid

    P, bg = _params()
    t, x, y, m = _history()
    grid = Grid.from_bounds(np.array([-100.0, 100.0]), np.array([-100.0, 100.0]),
                            bin_km=5.0)
    quiet = etas_rate_field(P, bg, t, x, y, m, grid, 2001.0, 1.0).sum()

    t2 = np.append(t, 2000.9)          # M5 one hour before the window opens
    x2, y2 = np.append(x, 0.0), np.append(y, 0.0)
    m2 = np.append(m, 5.0)
    loud = etas_rate_field(P, bg, t2, x2, y2, m2, grid, 2001.0, 1.0).sum()
    assert loud > 3 * quiet, (quiet, loud)


def test_smoothed_background_redistributes_but_does_not_rescale():
    """Gate G1 compares uniform against smoothed backgrounds, so the two must
    differ in SHAPE only.

    Two traps here, both silent. The smoothed background lives on its own raster
    with a different origin and cell size, so it has to be resampled onto the
    scoring grid rather than assumed aligned -- an earlier version fell back to
    a uniform field whenever the shapes differed, which would have reported
    "smoothed" results that were actually uniform. And it must be normalised
    against the SCORING grid, not the background's own area, or the two modes
    differ in total rate as well as in distribution and G1 reads both at once.
    """
    from flowquake.etas_fit import Background, ETASParams, etas_rate_field
    from flowquake.target_process import Grid

    P = ETASParams(mu=2e-6, K=0.0, mc=1.0)          # K=0 -> background only
    t, x, y, m = _history(400, seed=3)
    grid = Grid.from_bounds(np.array([-100.0, 100.0]), np.array([-100.0, 100.0]),
                            bin_km=5.0)
    uni = Background(mode="uniform", area_km2=40000.0, grid=None,
                     xmin=-100, ymin=-100, bin_km=5.0)
    # deliberately misaligned raster: different origin AND cell size
    g = np.zeros((30, 30)); g[10:20, 10:20] = 1.0
    smo = Background(mode="smoothed", area_km2=40000.0, grid=g,
                     xmin=-90.0, ymin=-90.0, bin_km=6.0)

    lu = etas_rate_field(P, uni, t, x, y, m, grid, 2001.0, 1.0)
    ls = etas_rate_field(P, smo, t, x, y, m, grid, 2001.0, 1.0)

    assert ls.sum() == pytest.approx(lu.sum(), rel=1e-9), "level must not change"
    assert (ls > 0).sum() < (lu > 0).sum() / 4, "shape must actually concentrate"


def test_analytic_field_integrates_the_kernel_over_each_cell():
    """Midpoint quadrature is not good enough when d < bin_km, and it isn't.

    The spatial scale fitted on WHITE is d = 1.0 km against 2 km scoring cells,
    so the kernel varies by a factor of several INSIDE one cell. Evaluating the
    density at the cell centre and multiplying by area overestimates the
    parent's own cell by 1.94x and underestimates its neighbours by 0.84x. The
    field is then over-concentrated exactly where a target is most likely to
    fall -- and that lands in the SHAPE term, the primary metric, not in the
    level term where it could be argued away.
    """
    from flowquake.etas_fit import (
        Background, ETASParams, etas_rate_field, omori_int, productivity, spatial)
    from flowquake.target_process import Grid

    P = ETASParams(mu=0.0, K=0.6, a=0.4, c=0.25, p=1.14, d=1.0, gamma=0.0,
                   q=1.53, mc=2.0)
    grid = Grid(xmin=-50.0, ymin=-50.0, nx=50, ny=50, bin_km=2.0)
    bg = Background(mode="uniform", area_km2=1e4, grid=None,
                    xmin=-50.0, ymin=-50.0, bin_km=2.0)

    # one parent sitting exactly on a cell centre: the worst case for midpoint
    px = grid.xmin + 10.5 * grid.bin_km
    py = grid.ymin + 10.5 * grid.bin_km
    t, x, y, m = np.array([0.0]), np.array([px]), np.array([py]), np.array([P.mc])
    lam = etas_rate_field(P, bg, t, x, y, m, grid, 0.5, 1.0, tail_weight=1.0)

    # reference: fine sub-grid quadrature of the same kernel over the same grid
    n = 12
    off = (np.arange(n) + 0.5) / n * grid.bin_km - grid.bin_km / 2
    gx = (np.arange(grid.nx) + 0.5) * grid.bin_km + grid.xmin
    gy = (np.arange(grid.ny) + 0.5) * grid.bin_km + grid.ymin
    cx, cy = np.repeat(gx, grid.ny), np.tile(gy, grid.nx)
    ox, oy = np.meshgrid(off, off, indexing="ij")
    r2 = ((cx[:, None] + ox.ravel()[None, :] - px) ** 2 +
          (cy[:, None] + oy.ravel()[None, :] - py) ** 2)
    ref_shape = (spatial(r2, np.full(r2.shape, P.mc), P).mean(axis=1)
                 * grid.cell_area_km2)
    w = float((productivity(m, P) * (omori_int(np.array([1.5]), P)
                                     - omori_int(np.array([0.5]), P)))[0])

    assert lam.sum() == pytest.approx(ref_shape.sum() * w, rel=0.01)
    # and the SHAPE must match cell by cell, which is what the metric reads
    pa = lam / lam.sum()
    pb = ref_shape / ref_shape.sum()
    tv = 0.5 * np.abs(pa - pb).sum()
    assert tv < 0.02, f"spatial total-variation distance {tv:.4f} vs fine quadrature"
