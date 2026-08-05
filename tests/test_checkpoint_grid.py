"""The scoring grid must be dense where the confound lives.

The (mc, step) surface exists because the published mc 2.5 anchor was step 200 --
mid-warmup, with its true optimum inside (0, 200) and overwritten. A uniform grid
would step straight over that interval and the experiment would cost ~25 GPU-hours
to answer a different question, so the density near zero is not a preference.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from checkpoint_surface import checkpoint_grid


def test_dense_below_500_sparse_after():
    g = checkpoint_grid(budget=12000, val_every=50)
    early = [s for s in g if s <= 500]
    assert early == [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], early
    late = [s for s in g if s > 500]
    assert late[:3] == [1000, 1500, 2000], late[:3]
    assert g[-1] == 12000


def test_resolves_the_interval_that_motivated_the_experiment():
    """At least two grid points strictly inside (0, 200)."""
    g = checkpoint_grid(budget=12000, val_every=50)
    inside = [s for s in g if 0 < s < 200]
    assert len(inside) >= 2, f"cannot resolve the (0,200) optimum: {inside}"


def test_only_steps_training_actually_saves():
    """Training writes at multiples of val_every; the grid must not ask for others."""
    for ve in (25, 50, 200):
        g = checkpoint_grid(budget=2000, val_every=ve)
        assert all(s % ve == 0 for s in g), (ve, g)
        assert all(s <= 2000 for s in g)


def test_coarse_val_every_cannot_resolve_the_confound():
    """A regression guard: val_every=200 reproduces the original blind spot.

    This is the setting that created the problem. If someone runs the surface
    with it, the grid genuinely cannot see inside (0, 200) -- the test documents
    that rather than pretending the grid fixes a cadence it cannot.
    """
    g = checkpoint_grid(budget=12000, val_every=200)
    assert [s for s in g if 0 < s < 200] == []
