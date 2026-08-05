"""Tests for surrogate nulls (MOONSHOT.md invariant 1g).

A surrogate is only a null if it removes the COUPLING and nothing else. Every
marginal it accidentally changes becomes a confound -- that is exactly how the
first null control failed, by spreading small events over twice the footprint
of the real ones and so measuring noise-damage rather than no-information.
"""

import numpy as np
import pandas as pd
import pytest

from flowquake.surrogate import (
    circular_time_shift, rotate_about_centroid, surrogate_null,
)


@pytest.fixture
def catalog():
    """Clustered small events around a few target events."""
    r = np.random.default_rng(0)
    rows = []
    for k in range(12):
        t0 = 500.0 * k + r.uniform(0, 100)
        cx, cy = r.uniform(-50, 50), r.uniform(-50, 50)
        rows.append((t0, cx, cy, 4.0 + r.exponential(0.4)))       # target
        n = r.integers(80, 160)
        rows.append(None)
        for _ in range(n):                                        # aftershocks
            rows.append((t0 + r.exponential(20.0), cx + r.normal(0, 5),
                         cy + r.normal(0, 5), 0.5 + r.exponential(0.4)))
    rows = [x for x in rows if x is not None]
    d = pd.DataFrame(rows, columns=["t_days", "x", "y", "magnitude"])
    d["time"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(d["t_days"], unit="D")
    return d.sort_values("time").reset_index(drop=True)


def test_targets_are_untouched(catalog):
    """Invariant 1: every arm must be scored on the SAME target events."""
    s = surrogate_null(catalog, m_split=3.0, seed=1)
    a = catalog[catalog.magnitude >= 3.0].sort_values("time").reset_index(drop=True)
    b = s[s.magnitude >= 3.0].sort_values("time").reset_index(drop=True)
    assert len(a) == len(b)
    assert np.allclose(a.magnitude.to_numpy(), b.magnitude.to_numpy())
    assert np.allclose(a.x.to_numpy(), b.x.to_numpy())
    assert (a.time.to_numpy() == b.time.to_numpy()).all()


@pytest.mark.parametrize("mode", ["time_shift", "rotate", "both"])
def test_marginals_are_preserved(mode, catalog):
    """Count, magnitude distribution and spatial SPREAD must not move.

    The footprint check is the one that matters: the original null control
    doubled it (231 km against 121 km) and that alone cost 0.47 nats/decade.
    """
    s = surrogate_null(catalog, m_split=3.0, mode=mode, seed=2)
    a = catalog[catalog.magnitude < 3.0]
    b = s[s.magnitude < 3.0]
    assert len(a) == len(b)
    assert np.allclose(np.sort(a.magnitude.to_numpy()),
                       np.sort(b.magnitude.to_numpy()))
    # A rigid rotation preserves pairwise distances and TOTAL spread, but it
    # redistributes variance between the x and y axes for an anisotropic cloud,
    # so per-axis std is not the invariant to assert. Total spread is.
    spread = lambda d: float(np.hypot(d.x.std(), d.y.std()))
    assert spread(b) == pytest.approx(spread(a), rel=0.05)


def test_time_shift_destroys_temporal_coupling(catalog):
    """The point of the exercise: small events must stop predicting targets."""
    def coupling(d):
        """Mean count of small events in the 30 days AFTER each target.

        The fixture is aftershock-only, so this is where its coupling lives.
        Measuring the 30 days BEFORE would measure foreshocks, which the
        fixture does not have -- randomising would then RAISE the count toward
        the mean density and look like coupling had increased.
        """
        tg = d.loc[d.magnitude >= 3.0, "time"].to_numpy()
        sm = d.loc[d.magnitude < 3.0, "time"].to_numpy()
        w = np.timedelta64(30, "D")
        return float(np.mean([((sm > t) & (sm <= t + w)).sum() for t in tg]))

    before = coupling(catalog)
    after = coupling(surrogate_null(catalog, m_split=3.0,
                                    mode="time_shift", seed=3))
    assert before > 5.0, "fixture is not actually coupled"
    assert after < 0.5 * before, (before, after)


def test_rotation_preserves_pairwise_distances(catalog):
    x = catalog["x"].to_numpy()[:200]
    y = catalog["y"].to_numpy()[:200]
    nx, ny = rotate_about_centroid(x, y, 0.7)
    d0 = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    d1 = np.hypot(nx[:, None] - nx[None, :], ny[:, None] - ny[None, :])
    assert np.allclose(d0, d1)


def test_circular_shift_stays_in_span_and_is_a_bijection():
    """A plain offset would push events off the end and thin late windows,
    which is itself an mc-dependent artifact since thinning scales with count."""
    t = np.linspace(0.0, 100.0, 501)[:-1]   # half-open: 100.0 IS 0.0 after wrap
    s = circular_time_shift(t, 37.0, 0.0, 100.0)
    assert s.min() >= 0.0 and s.max() < 100.0 + 1e-9
    assert len(np.unique(np.round(s, 6))) == len(np.unique(np.round(t, 6)))


def test_shift_is_never_trivially_small(catalog):
    """A few-day shift would leave the triggering relationship intact.

    Measured PER EVENT with identity preserved. Comparing sorted times would
    not work: a circular shift maps the set of times onto nearly itself, so
    the sorted sequence barely moves however large the shift is.
    """
    d = catalog.copy()
    d["_id"] = np.arange(len(d))
    span_ns = float(d.time.astype("int64").max() - d.time.astype("int64").min())
    for seed in range(8):
        s = surrogate_null(d, m_split=3.0, mode="time_shift", seed=seed)
        a = d[d.magnitude < 3.0].set_index("_id").time.astype("int64")
        b = s[s.magnitude < 3.0].set_index("_id").time.astype("int64")
        disp = np.mod((b - a.reindex(b.index)).to_numpy(), span_ns) / span_ns
        # one constant shift for every event
        assert np.ptp(disp) < 1e-6, f"seed {seed}: shift is not constant"
        assert 0.15 < float(disp[0]) < 0.85, (
            f"seed {seed}: shift {disp[0]:.3f} of span is too close to 0 or 1")


def test_rejects_unknown_mode(catalog):
    with pytest.raises(ValueError):
        surrogate_null(catalog, m_split=3.0, mode="scramble")


def test_rotation_requires_projected_coordinates(catalog):
    with pytest.raises(ValueError, match="x/y"):
        surrogate_null(catalog.drop(columns=["x", "y"]), m_split=3.0,
                       mode="rotate")


def test_era_bounds_preserve_per_era_counts(catalog):
    """Each era must keep exactly the small-event count it started with.

    Real small events cluster in time; spreading them uniformly across the whole
    span moves some over the train/test boundary. Measured on WHITE without
    era_bounds, the surrogate arm had 1,604 training events at mc 2.0 against
    the informative arm's 1,383 -- a 26% data advantage for the arm that is
    meant to be the control, which confounds any comparison of the two.
    """
    bounds = [pd.Timestamp("2003-01-01"), pd.Timestamp("2007-01-01")]
    s = surrogate_null(catalog, m_split=3.0, mode="time_shift", seed=4,
                       era_bounds=bounds)

    def per_era(d):
        sm = d[d.magnitude < 3.0]
        edges = [d.time.min()] + bounds + [d.time.max() + pd.Timedelta("1s")]
        return [int(((sm.time >= a) & (sm.time < b)).sum())
                for a, b in zip(edges, edges[1:])]

    assert per_era(s) == per_era(catalog), (per_era(catalog), per_era(s))


def test_without_era_bounds_counts_can_move(catalog):
    """Documents why the parameter exists: the unbounded shift redistributes."""
    s = surrogate_null(catalog, m_split=3.0, mode="time_shift", seed=4)
    half = catalog.time.min() + (catalog.time.max() - catalog.time.min()) / 2
    n0 = int((catalog[catalog.magnitude < 3.0].time < half).sum())
    n1 = int((s[s.magnitude < 3.0].time < half).sum())
    assert n0 != n1, "fixture should be temporally clustered enough to shift"
