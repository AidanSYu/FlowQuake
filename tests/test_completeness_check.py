"""The completeness detector, which decides whether the information floor is real.

WHY THE STAKES ARE HIGH IN BOTH DIRECTIONS. The ladder reports that magnitude
bands stop paying below about M1.25. If the catalog is going incomplete there,
that is an artifact and the headline is wrong. So a MISSED rollover publishes a
detection limit as physics. But a FALSE rollover is nearly as bad: it retires a
usable catalog, and catalogs deep enough to see the floor at all are scarce.

The first version of this detector fired on any bin below 85% of its
Gutenberg-Richter prediction, and it called a rollover at M2.45 in the QTM San
Jacinto catalog off a bin holding 68 events against 85 predicted. That is 1.8
sigma of ordinary counting noise, and the bin directly beneath it was ABOVE
prediction. Acting on it would have thrown away the one candidate replication
site. These tests exist so that failure cannot come back.

Two corrections are under test. Deficits must clear `z_crit` Poisson sigma, and
they must persist across `run` consecutive bins, because incompleteness is
monotone downward while noise dips alone.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.completeness_check import rollover_magnitude  # noqa: E402


def gr_counts(centres, b=1.0, a=5.0):
    """Noise-free Gutenberg-Richter counts: the complete-catalog ideal."""
    return np.round(10.0 ** (a - b * centres)).astype(int)


CENTRES = np.arange(0.55, 3.55, 0.1)


def test_a_complete_catalog_reports_no_rollover():
    counts = gr_counts(CENTRES)
    roll, b, _, _, _ = rollover_magnitude(CENTRES, counts, 1.6, 2.8, 0.85)
    assert roll is None
    assert b == pytest.approx(1.0, abs=0.02)


def test_a_real_truncation_is_caught():
    """Everything below M1.2 removed. That must be found, and at the edge."""
    counts = gr_counts(CENTRES)
    counts[CENTRES < 1.2] = 0
    roll, _, _, _, _ = rollover_magnitude(CENTRES, counts, 1.6, 2.8, 0.85)
    assert roll is not None and roll == pytest.approx(1.15, abs=0.06)


def test_a_gradual_rollover_is_caught():
    """Detection usually fades rather than cliffs. Still has to trip."""
    counts = gr_counts(CENTRES).astype(float)
    fade = np.clip((CENTRES - 0.9) / 0.5, 0.0, 1.0)      # full by M1.4
    roll, _, _, _, _ = rollover_magnitude(
        CENTRES, np.round(counts * fade).astype(int), 1.6, 2.8, 0.85)
    assert roll is not None and 1.0 <= roll <= 1.4


def test_an_isolated_sparse_dip_is_not_a_rollover():
    """The exact San Jacinto false positive: 68 seen where 85 predicted.

    Down 20%, which trips a bare ratio test, but 1.8 sigma and unsupported by
    the bin below. Calling this incomplete cost a replication site.
    """
    counts = gr_counts(CENTRES)
    i = int(np.argmin(np.abs(CENTRES - 2.45)))
    counts = counts.copy()
    counts[i] = int(round(counts[i] * 0.80))
    roll, _, _, _, _ = rollover_magnitude(CENTRES, counts, 1.6, 2.8, 0.85)
    assert roll is None


def test_persistence_alone_decides_between_two_identical_deficits():
    """Same deficit, same counting weight, one isolated and one sustained.

    Everything except persistence is held fixed: a dense catalog so the Poisson
    gate is cleared either way, the same 20% shortfall, the same bin. Only the
    run below it differs, so a difference in verdict can come from nothing else.
    """
    base = gr_counts(CENTRES, a=7.0).astype(float)
    at = np.abs(CENTRES - 1.45) < 0.05

    one = base.copy()
    one[at] *= 0.80
    many = base.copy()
    many[CENTRES <= 1.45] *= 0.80

    r_one = rollover_magnitude(
        CENTRES, np.round(one).astype(int), 1.6, 2.8, 0.85)[0]
    r_many = rollover_magnitude(
        CENTRES, np.round(many).astype(int), 1.6, 2.8, 0.85)[0]
    assert r_one is None
    assert r_many == pytest.approx(1.45, abs=0.06)


def test_a_rollover_reaching_into_the_fit_range_is_hidden():
    """A precondition of the method, documented rather than papered over.

    The GR line is fitted on data. Depress the fit range itself and the line
    follows, the deficit vanishes against it, and the detector returns clean.
    That is why --fit-lo must sit clearly above any plausible detection limit,
    and why a suspicious result gets refitted higher before it is believed. It
    is exactly how the San Jacinto verdict was checked.
    """
    counts = gr_counts(CENTRES, a=7.0).astype(float)
    counts[CENTRES <= 2.45] *= 0.80          # swallows most of the 1.6-2.8 fit
    swallowed = rollover_magnitude(
        CENTRES, np.round(counts).astype(int), 1.6, 2.8, 0.85)[0]
    assert swallowed is None

    # Refit entirely above the affected range and the same data gives it up.
    recovered = rollover_magnitude(
        CENTRES, np.round(counts).astype(int), 2.6, 3.4, 0.85)[0]
    assert recovered == pytest.approx(2.45, abs=0.06)


def test_poisson_gate_scales_with_count_not_ratio():
    """A 20% deficit is decisive in a dense bin and meaningless in a sparse one.

    Ratio alone cannot tell those apart, which is the whole bug. Same relative
    deficit, same persistence, opposite verdicts purely from counting weight.
    """
    dense = gr_counts(CENTRES, a=7.0)         # ~1e5 at the bottom
    sparse = gr_counts(CENTRES, a=2.6)        # ~tens at the bottom
    out = []
    for c in (dense, sparse):
        c = c.astype(float)
        c[CENTRES <= 1.45] *= 0.80
        out.append(rollover_magnitude(
            CENTRES, np.round(c).astype(int), 1.6, 2.8, 0.85)[0])
    assert out[0] is not None      # dense: 20% short is overwhelming
    assert out[1] is None          # sparse: 20% short is noise


def test_excess_counts_never_trigger_a_rollover():
    """b steepens toward small magnitudes, lifting low bins above the fit line.

    That is normal and must stay invisible to a one-sided detector: a network
    cannot invent earthquakes, so only deficits diagnose anything.
    """
    counts = gr_counts(CENTRES).astype(float)
    counts[CENTRES < 1.6] *= 1.35
    roll, _, _, _, _ = rollover_magnitude(
        CENTRES, np.round(counts).astype(int), 1.6, 2.8, 0.85)
    assert roll is None


def test_fit_range_with_too_few_bins_is_refused():
    """Silently fitting GR through two points would be worse than stopping."""
    counts = gr_counts(CENTRES)
    with pytest.raises(SystemExit):
        rollover_magnitude(CENTRES, counts, 2.75, 2.85, 0.85)


def test_bins_above_the_fit_range_cannot_trigger():
    """Large-magnitude bins are sparse and noisy, and are not about detection."""
    counts = gr_counts(CENTRES).astype(float)
    counts[CENTRES > 3.0] *= 0.2
    roll, _, _, _, _ = rollover_magnitude(
        CENTRES, np.round(counts).astype(int), 1.6, 2.8, 0.85)
    assert roll is None
