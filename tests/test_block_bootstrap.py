"""The bootstrap must resample CONTIGUOUS BLOCKS, not individual days.

`block_bootstrap_slope` was named a block bootstrap and its docstring claimed to
handle the fact that target events arrive in aftershock sequences. It drew
`rng.integers(0, n_win, n_win)` — an i.i.d. resample of individual forecast
windows. The windows are consecutive 1-day forecasts (every `start_days`
difference in the committed frames is exactly 1.0) and an aftershock sequence
stays correlated for weeks, so the unit being resampled was ~30x smaller than
the unit of dependence and every reported interval was too narrow. That applied
to the G1 pooled slopes and the G3 increments alike.

Nothing in the suite caught it, because an i.i.d. bootstrap is perfectly
well-behaved — it just answers a different question. So these tests pin the
PROPERTY that distinguishes the two: a block draw must preserve serial structure
that an i.i.d. draw destroys.
"""
import numpy as np

from flowquake.pooling import DEFAULT_BLOCK_WINDOWS, _resample


def test_block_draw_returns_the_right_length():
    rng = np.random.default_rng(0)
    for n, L in ((1673, 30), (100, 7), (50, 50), (10, 90)):
        idx = _resample(rng, n, L)
        assert len(idx) == n
        assert idx.min() >= 0 and idx.max() < n


def test_block_len_1_is_the_old_iid_behaviour():
    """Kept only to reproduce pre-fix numbers; must never be the default."""
    a = _resample(np.random.default_rng(7), 500, 1)
    b = np.random.default_rng(7).integers(0, 500, 500)
    assert np.array_equal(a, b)
    assert DEFAULT_BLOCK_WINDOWS > 1, "the default must not be the i.i.d. draw"


def test_blocks_are_contiguous_runs_modulo_wraparound():
    """Within a block, indices advance by exactly 1 — CIRCULARLY.

    The draw wraps (see `_resample`), so a run starting near the end continues
    at index 0 and its raw diff is -(n-1) there. Checking `diff == 1` outright
    passes or fails depending on the seed: across 20 seeds, 20 runs wrap, but a
    single seed has a ~36% chance of showing none. Compare modulo n instead, and
    sweep seeds so the test cannot be lucky.
    """
    n, L = 1673, 30
    for seed in range(12):
        idx = _resample(np.random.default_rng(seed), n, L)
        n_full = len(idx) // L
        for k in range(n_full):
            run = idx[k * L:(k + 1) * L]
            step = (np.diff(run)) % n
            assert np.all(step == 1), (
                f"seed {seed} run {k} is not contiguous mod {n}: {run[:5]}...")


def test_short_panels_still_get_a_real_resample():
    """The degeneracy that the pooling suite caught.

    A plain moving block with starts in [0, n-L] has exactly one legal start
    when L >= n, so every replicate is the identity, resampling variance is zero
    and the interval collapses onto the point estimate. Wrapping plus the n/4 cap
    must keep the draw genuinely random at any panel length.
    """
    for n in (8, 20, 40, 120):
        draws = {tuple(_resample(np.random.default_rng(s), n, 30)) for s in range(25)}
        assert len(draws) > 1, f"n={n}: every replicate identical — no resampling"
        # and it must not be the identity permutation every time
        ident = tuple(range(n))
        assert not all(d == ident for d in draws), f"n={n}: draws are the identity"


def test_block_draw_preserves_serial_correlation_that_iid_destroys():
    """The property that actually matters.

    Build a series with strong serial correlation. A contiguous-block resample
    must retain most of the lag-1 correlation; an i.i.d. resample must destroy
    it. This is the difference that makes the interval honest.
    """
    n = 1673
    rng = np.random.default_rng(0)
    # AR(1) with a long memory, standing in for an aftershock sequence.
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + rng.normal()

    def lag1(v):
        v = v - v.mean()
        return float((v[:-1] * v[1:]).sum() / (v * v).sum())

    truth = lag1(x)
    assert truth > 0.8, "fixture is not strongly correlated"

    blk = np.mean([lag1(x[_resample(np.random.default_rng(s), n, 30)])
                   for s in range(20)])
    iid = np.mean([lag1(x[_resample(np.random.default_rng(s), n, 1)])
                   for s in range(20)])

    assert iid < 0.15, f"i.i.d. draw should destroy serial correlation, got {iid:.3f}"
    assert blk > 0.6, f"block draw should retain it, got {blk:.3f}"


def test_longer_blocks_give_wider_intervals_on_correlated_data():
    """Sanity: the fix must WIDEN intervals, never narrow them, when data is
    serially correlated. If a change to _resample ever narrows them, it is
    reintroducing the bug under a new name."""
    from flowquake.pooling import block_bootstrap_slope
    n = 1673
    rng = np.random.default_rng(1)
    base = np.zeros(n)
    for i in range(1, n):
        base[i] = 0.95 * base[i - 1] + rng.normal()
    tgt = rng.integers(0, 2, n).astype(float)
    tgt[0] = 1.0
    scores = np.vstack([base, base + 0.3 * tgt])

    widths = {}
    for L in (1, 30):
        p = block_bootstrap_slope("t", [2.0, 1.0], scores, tgt,
                                  n_boot=400, seed=0, block_len=L)
        widths[L] = p.ci_hi - p.ci_lo
    assert widths[30] > widths[1], (
        f"block bootstrap must not be narrower than i.i.d. on correlated data: "
        f"L=30 width {widths[30]:.4f} vs L=1 {widths[1]:.4f}")
