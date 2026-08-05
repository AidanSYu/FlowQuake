"""A seeded run must be reproducible, so optimisations can be proved harmless.

Production scoring is UNSEEDED on purpose: the estimator has real Monte-Carlo
spread (sd 0.036 nats at mc 1.0, invariant 1s) and freezing one draw would
understate it. The cost of that choice is that an unseeded run cannot tell
"this refactor changed the answer" apart from "this is a different draw" -- so
every performance change would be unverifiable, which is exactly the situation
in which a subtly wrong optimisation gets committed and never noticed.

`simulate_windows(seed=...)` closes that gap. These tests pin three properties:

  1. same seed  -> byte-identical output   (the regression baseline)
  2. diff seed  -> different output        (proves 1 isn't passing trivially,
                                            e.g. because the model emits nothing)
  3. chunking   -> lanes stay independent  (the disaster case, below)

(3) is the one worth spelling out. `simulate_windows` recurses to bound memory,
and the natural way to write seeding -- seed at the top of the function -- would
re-seed every recursive chunk with the same value and hand each chunk IDENTICAL
random draws. Nothing would crash. The counts would look plausible. But the
effective sample size would silently collapse to a single chunk while every
downstream interval carried on as though it had n_sims independent lanes. So the
seed is applied once at entry and the recursion inherits the stream, and this
test fails if anyone ever "tidies" that into a per-call seed.
"""
import numpy as np
import pytest
import torch

from flowquake.ntest import simulate_windows

from helpers_sampler import make_tiny_model_and_catalog


def _flat(out):
    """Concatenate a nested simulate_windows result into one comparable array."""
    rows = [e for win in out for sim in win for e in np.atleast_2d(sim)
            if np.size(e)]
    return np.concatenate([np.atleast_2d(r) for r in rows]) if rows \
        else np.zeros((0, 4))


@pytest.fixture(scope="module")
def fixture():
    return make_tiny_model_and_catalog()


def test_same_seed_is_bit_identical(fixture):
    model, cat, starts = fixture
    kw = dict(sample_steps=4, horizon_days=1.0, max_lanes=4096)
    a = _flat(simulate_windows(model, cat, starts, 64, "cpu", seed=1234, **kw))
    b = _flat(simulate_windows(model, cat, starts, 64, "cpu", seed=1234, **kw))
    assert a.shape == b.shape, f"shapes differ: {a.shape} vs {b.shape}"
    assert np.array_equal(a, b), "same seed did not reproduce bit-for-bit"


def test_different_seed_differs(fixture):
    """Guards against test 1 passing because nothing is ever simulated."""
    model, cat, starts = fixture
    kw = dict(sample_steps=4, horizon_days=1.0, max_lanes=4096)
    a = _flat(simulate_windows(model, cat, starts, 64, "cpu", seed=1, **kw))
    b = _flat(simulate_windows(model, cat, starts, 64, "cpu", seed=2, **kw))
    assert a.size > 0, "fixture simulated no events; test 1 would be vacuous"
    assert not (a.shape == b.shape and np.array_equal(a, b)), \
        "different seeds produced identical output"


def test_chunking_does_not_duplicate_draws(fixture):
    """The disaster case: chunks must NOT be carbon copies of each other.

    Force the simulation-splitting branch (a single window exceeding the lane
    budget), then check the per-chunk event counts are not all identical. If the
    seed were re-applied per recursive call every chunk would be the same draw,
    and this is the cheapest observable consequence.
    """
    model, cat, starts = fixture
    n_sims, chunk = 256, 64
    out = simulate_windows(model, cat, starts[:1], n_sims, "cpu", seed=7,
                           sample_steps=4, horizon_days=1.0, max_lanes=chunk)
    per_sim = [len(np.atleast_2d(s)) if np.size(s) else 0 for s in out[0]]
    assert len(per_sim) == n_sims

    groups = [tuple(per_sim[i:i + chunk]) for i in range(0, n_sims, chunk)]
    assert len(groups) > 1, "did not exercise the chunking branch"
    assert len(set(groups)) > 1, (
        "every chunk produced an identical draw -- the seed is being re-applied "
        "inside the recursion, collapsing the effective sample size")


def test_unseeded_runs_still_vary(fixture):
    """The production path must stay stochastic; seeding is opt-in only."""
    model, cat, starts = fixture
    kw = dict(sample_steps=4, horizon_days=1.0, max_lanes=4096)
    torch.manual_seed(0)
    a = _flat(simulate_windows(model, cat, starts, 64, "cpu", **kw))
    b = _flat(simulate_windows(model, cat, starts, 64, "cpu", **kw))
    assert not (a.shape == b.shape and np.array_equal(a, b)), \
        "unseeded runs were identical -- scoring MC noise would be invisible"
