"""Tests for the completeness mask (`MOONSHOT.md` invariant 1).

The properties here are the ones that decide whether a point on the scaling
curve measures information or measures detection. `test_b_stability_finds_
planted_mc` is the load-bearing one: if the estimator cannot recover a known
Mc from a catalog we truncated ourselves, it cannot be trusted to tell us
which grid points are legal on the real one.
"""

import json

import numpy as np
import pandas as pd
import pytest

from flowquake.completeness import (
    CompletenessMask, build_mask, mc_by_b_stability, mc_by_max_curvature,
    mc_map, shi_bolt_sigma,
)


def gr_sample(n, b=1.0, m_floor=0.0, seed=0):
    r = np.random.default_rng(seed)
    return m_floor + r.exponential(1.0 / (b * np.log(10.0)), n)


def detected_catalog(n, mu, sigma=0.3, b=1.0, seed=0):
    """A GR catalog thinned by a cumulative-normal detection curve.

    Ogata & Katsura (1993). This, not a sharp truncation, is what real network
    incompleteness looks like, and the difference is the whole point:

    Sharp truncation does NOT bias b. A mixture of GR laws that share a b is
    still exponential with that same b -- only the amplitude steps at each
    cutoff -- so an estimator sees a perfectly stable b right down to the
    lowest cutoff. Testing against sharp truncation would therefore certify an
    estimator that is blind to the actual failure.

    Gradual detection is different. Partial detection near the roll-off keeps
    some small events and drops others, flattening the low end of the FMD and
    dragging apparent b DOWN. That is the signature measured on the real
    catalog (b climbing 0.74 -> 0.98 as the threshold rose from 1.0 to 3.0),
    and it is what these tests reproduce.

    `mu` is the 50%-detection magnitude; usable completeness is near
    `mu + 1.88*sigma` (97% detection).
    """
    from scipy.stats import norm
    r = np.random.default_rng(seed)
    m = -1.0 + r.exponential(1.0 / (b * np.log(10.0)), n)
    return m[r.random(n) < norm.cdf((m - mu) / sigma)]


# --- Mc estimation ---------------------------------------------------------

def test_b_stability_finds_planted_mc():
    """Recover a known completeness level, and never be OPTIMISTIC about it.

    The two directions are not symmetric. Overestimating Mc costs range on the
    x-axis; underestimating it licenses grid points below completeness, where
    the curve measures detection and the headline number is meaningless. So
    the tolerance is deliberately lopsided. Measured seed-to-seed scatter is
    about +/-0.2 at these sample sizes.
    """
    sigma = 0.3
    for mu in (1.0, 1.5):
        mc_true = mu + 1.88 * sigma          # 97% detection
        for seed in range(3):
            got = mc_by_b_stability(
                detected_catalog(2_000_000, mu, sigma, seed=seed),
                lo=0.0, hi=4.0)
            assert got is not None
            assert got >= mc_true - 0.25, (
                f"OPTIMISTIC: planted Mc~{mc_true:.2f} (mu={mu}, seed={seed}), "
                f"got {got} -- would license points below completeness")
            assert got <= mc_true + 0.7, (
                f"needlessly conservative: planted ~{mc_true:.2f}, got {got}")


def test_b_drifts_downward_below_completeness():
    """The diagnostic itself: b must fall as the threshold drops below Mc.

    This is the pattern that exposed the statewide Mc problem. If it ever
    stops holding, reading a b-vs-threshold table means nothing.
    """
    from flowquake.target_process import aki_utsu_b
    m = detected_catalog(4_000_000, 1.5, 0.3, seed=4)
    bs = [aki_utsu_b(m, c) for c in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)]
    assert all(x < y for x, y in zip(bs, bs[1:])), bs
    assert bs[0] < 0.7 and bs[-1] > 0.95


def test_b_stability_returns_none_when_never_complete():
    """A catalog whose b never settles must be excluded, not given a number."""
    r = np.random.default_rng(0)
    # magnitudes with no GR structure at all: b drifts everywhere
    m = r.uniform(0.0, 3.0, 50_000)
    assert mc_by_b_stability(m, lo=0.0, hi=2.5, dm_ref=0.5) is None


def test_max_curvature_is_optimistic_relative_to_b_stability():
    """Documents WHY the grid is not set from MAXC.

    On the real catalog MAXC returned 1.30 where b-stability returned 2.6, and
    the 1.30 is what originally licensed a grid running down to mc 1.5. MAXC
    reads the mode of the histogram, which sits at the PEAK of the detection
    roll-off -- around 50% detection, not the ~97% completeness needs.
    """
    for mu in (1.0, 1.5, 2.0):
        m = detected_catalog(4_000_000, mu, 0.3, seed=int(mu * 7))
        maxc = mc_by_max_curvature(m)
        bstab = mc_by_b_stability(m, lo=0.0, hi=4.0)
        assert bstab is not None
        assert maxc < bstab, f"mu={mu}: MAXC {maxc:.2f} vs b-stability {bstab}"


def test_pooling_unequal_cells_raises_apparent_mc():
    """The mixture artifact that made the statewide number unusable.

    Pooling a well-instrumented region with a poorly-instrumented one gives an
    aggregate whose completeness is worse than the good region's, so a single
    statewide Mc understates how far down the good cells can actually be
    trusted. That is the entire argument for masking by cell rather than
    raising the grid.

    The bad region needs COMPARABLE event counts to dominate the pooled FMD,
    which is the real situation: statewide, the cells that fail the mask hold
    roughly 430k of the 658k events. With counts matched this reproduces the
    measured statewide value almost exactly (pooled 2.6 against a good-cell
    1.4), which is why the reproduction is worth locking in.
    """
    good = detected_catalog(2_000_000, 1.0, 0.3, seed=1)
    bad = detected_catalog(20_000_000, 2.0, 0.3, seed=2)
    assert 0.5 < len(bad) / len(good) < 2.0, "counts must be comparable"
    mc_good = mc_by_b_stability(good, lo=0.0, hi=4.0)
    mc_pooled = mc_by_b_stability(np.concatenate([good, bad]), lo=0.0, hi=4.0)
    assert mc_good is not None and mc_pooled is not None
    assert mc_pooled > mc_good + 0.5, (
        f"pooled {mc_pooled} should be much worse than good-cell {mc_good}")


def test_shi_bolt_sigma_shrinks_with_sample_size():
    m_small = gr_sample(500, seed=0)
    m_big = gr_sample(50_000, seed=0)
    assert shi_bolt_sigma(m_big, 1.0) < shi_bolt_sigma(m_small, 1.0)


# --- mask construction -----------------------------------------------------

def _two_region_catalog(seed=0):
    """One complete cell at Mc 1.0, one incomplete cell at Mc 2.5.

    Both span the full time range so the causal cut keeps both in the training
    era; the only thing separating them is completeness.
    """
    r = np.random.default_rng(seed)
    rows = []
    for lat, lon, mu, n in ((34.5, -118.5, 0.5, 1_000_000),
                            (40.5, -124.5, 2.0, 8_000_000)):
        m = detected_catalog(n, mu, 0.3, seed=seed + int(lat))
        rows.append(pd.DataFrame({
            "time": pd.Timestamp("1995-01-01") + pd.to_timedelta(
                r.uniform(0, 9000, len(m)), unit="D"),
            "latitude": lat + r.uniform(-0.4, 0.4, len(m)),
            "longitude": lon + r.uniform(-0.4, 0.4, len(m)),
            "magnitude": m,
        }))
    return pd.concat(rows, ignore_index=True)


def test_mask_keeps_complete_cell_and_drops_incomplete_one():
    df = _two_region_catalog()
    mask = build_mask(df, train_end="2020-01-01", mc_threshold=1.5, deg=1.0)
    assert (34.0, -119.0) in mask.cells
    assert (40.0, -125.0) not in mask.cells
    assert len(mask) == 1


def test_mask_union_mc_is_reported_and_respects_threshold():
    df = _two_region_catalog()
    mask = build_mask(df, train_end="2020-01-01", mc_threshold=1.5, deg=1.0)
    assert mask.mc_union is not None
    assert mask.mc_union <= mask.mc_threshold


def test_mask_is_causal():
    """Events after train_end must not influence which cells are selected.

    The failure this guards against is subtle and would not crash: pick the
    testing region using the test period, and the region silently concentrates
    on wherever the test earthquakes happened.
    """
    df = _two_region_catalog()
    m1 = build_mask(df, train_end="2011-01-01", mc_threshold=1.5, deg=1.0)

    # a flood of well-detected events in the REJECTED cell, but only after
    # train_end: a non-causal mask would now accept that cell
    r = np.random.default_rng(7)
    m = detected_catalog(1_000_000, 0.5, 0.3, seed=11)
    n = len(m)
    intruder = pd.DataFrame({
        "time": pd.Timestamp("2015-01-01") + pd.to_timedelta(
            r.uniform(0, 1000, n), unit="D"),
        "latitude": 40.5 + r.uniform(-0.4, 0.4, n),
        "longitude": -124.5 + r.uniform(-0.4, 0.4, n),
        "magnitude": m,
    })
    m2 = build_mask(pd.concat([df, intruder], ignore_index=True),
                    train_end="2011-01-01", mc_threshold=1.5, deg=1.0)
    assert m1.cells == m2.cells


def test_mask_apply_selects_only_accepted_cells():
    df = _two_region_catalog()
    mask = build_mask(df, train_end="2020-01-01", mc_threshold=1.5, deg=1.0)
    kept = mask.apply(df)
    assert len(kept) > 0
    assert kept["latitude"].between(34.0, 35.0).all()


def test_mask_roundtrips_through_json():
    """The mask is serialised and reused by every arm; it must survive a trip
    to disk exactly, or two models can end up scoring different regions."""
    df = _two_region_catalog()
    mask = build_mask(df, train_end="2020-01-01", mc_threshold=1.5, deg=1.0)
    back = CompletenessMask.from_dict(json.loads(json.dumps(mask.to_dict())))
    assert back.cells == mask.cells
    assert back.deg == mask.deg
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    assert np.array_equal(back.contains(lat, lon), mask.contains(lat, lon))


def test_build_mask_refuses_when_nothing_qualifies():
    df = _two_region_catalog()
    with pytest.raises(ValueError, match="no cell"):
        build_mask(df, train_end="2020-01-01", mc_threshold=0.1, deg=1.0)


def test_mc_map_skips_undersampled_cells():
    df = _two_region_catalog()
    got = mc_map(df, deg=1.0, min_events=100_000)
    assert got == {}


# ---------------------------------------------------------------------------
# MBS vs the single-reference rule, and the safety margin.
#
# These lock in the reasons the default estimator changed. The single-reference
# first-crossing rule (Cao & Gao 2002) fires on noise: across nested estimation
# eras for one real 1-degree cell it returned 1.3, 2.1, 1.1, 1.8 -- impossible
# as physics, since shrinking to a more recent era can only improve
# completeness. Since these estimates decide which cells enter the testing
# region, an unstable estimator makes the target set arbitrary.
# ---------------------------------------------------------------------------

def test_mbs_is_less_scattered_than_single_reference():
    """Both are noisy; MBS must be the tighter of the two on ground truth."""
    scatter = {}
    for method in ("mbs", "single"):
        errs = []
        for mu in (0.5, 1.0, 1.5):
            truth = mu + 1.88 * 0.3
            for seed in range(4):
                v = mc_by_b_stability(
                    detected_catalog(2_000_000, mu, 0.3, seed=seed),
                    lo=0.0, hi=4.0, method=method)
                if v is not None:
                    errs.append(v - truth)
        scatter[method] = float(np.std(errs))
    assert scatter["mbs"] <= scatter["single"], scatter


def test_both_estimators_are_optimistic_which_is_why_a_margin_exists():
    """The bias runs in the dangerous direction, so document it as a test.

    If this ever flips sign, MC_SAFETY_MARGIN is miscalibrated and the mask
    will start admitting cells it should reject.
    """
    from flowquake.completeness import MC_SAFETY_MARGIN
    errs = []
    for mu in (0.5, 1.0, 1.5):
        truth = mu + 1.88 * 0.3
        for seed in range(4):
            v = mc_by_b_stability(detected_catalog(2_000_000, mu, 0.3, seed=seed),
                                  lo=0.0, hi=4.0)
            if v is not None:
                errs.append(v - truth)
    bias, sd = float(np.mean(errs)), float(np.std(errs))
    assert bias < 0.0, f"expected optimistic bias, got {bias:+.3f}"
    assert MC_SAFETY_MARGIN >= abs(bias), (
        f"margin {MC_SAFETY_MARGIN} does not cover measured bias {bias:+.3f}")
    assert MC_SAFETY_MARGIN <= abs(bias) + 2 * sd, "margin is needlessly wide"


def test_margin_blocks_a_marginal_cell():
    """A cell whose Mc sits just under the threshold must still be rejected."""
    df = _two_region_catalog()
    mc = mc_map(df[df.time < pd.Timestamp("2020-01-01")], deg=1.0,
                min_events=2000)
    good = mc[(34.0, -119.0)]
    assert good is not None
    # a threshold the cell clears outright, and one it only clears without margin
    build_mask(df, train_end="2020-01-01", mc_threshold=good + 0.4, deg=1.0)
    with pytest.raises(ValueError, match="no cell"):
        build_mask(df, train_end="2020-01-01", mc_threshold=good + 0.1, deg=1.0)


def test_mask_reports_worst_cell_not_only_the_union():
    """The union can look better than its worst member; the worst one binds.

    Pooling lets abundant events from good cells swamp a sparse incomplete one,
    so `mc_union` alone can certify a grid the cells cannot individually
    support. The score is evaluated cell by cell, so `mc_worst_cell` is the
    constraint that actually matters.
    """
    df = _two_region_catalog()
    mask = build_mask(df, train_end="2020-01-01", mc_threshold=2.6, deg=1.0)
    assert mask.mc_worst_cell is not None
    assert mask.mc_worst_cell == max(mask.mc_by_cell[k] for k in mask.cells)
    assert "mc_worst_cell" in mask.to_dict()
