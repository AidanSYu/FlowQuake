"""Tests for the mc-invariant target-process metric (`MOONSHOT.md` invariant 1b).

The property these lock in is the one the whole scaling curve rests on: the
score must depend ONLY on the M>=m_target sub-process, so that a model trained
at mc 0.5 and a model trained at mc 2.5 are scored on the same physical
quantity. If `test_invariant_to_subthreshold_events` ever fails, the x-axis of
the killer figure is measuring bookkeeping, not information.
"""

from dataclasses import replace

import numpy as np
import pytest

from flowquake.target_process import (
    Grid, TargetSpec, aggregate, p_at_least_one, poisson_ll,
    rate_field, reliability, score_window,
)


@pytest.fixture
def grid():
    rng = np.random.default_rng(0)
    return Grid.from_bounds(rng.uniform(-100, 100, 2000),
                            rng.uniform(-100, 100, 2000), bin_km=10.0)


@pytest.fixture
def spec():
    # 'learned' for the legacy tests: they assert on sampled magnitudes.
    # The fixed-tail behaviour has its own tests below.
    return TargetSpec(m_target=4.0, m_large=6.0, horizon_days=30.0,
                      tail_mode='learned', mc=2.5, b_value=1.0)


def _sims(n_mean, clustered, seed, n_sims=200, m_min=4.0):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n_sims):
        k = r.poisson(n_mean)
        if k == 0:
            out.append(np.zeros((0, 4)))
            continue
        if clustered:
            xx, yy = r.normal(0, 15, k), r.normal(0, 15, k)
        else:
            xx, yy = r.uniform(-100, 100, k), r.uniform(-100, 100, k)
        out.append(np.column_stack(
            [r.random(k) * 30, xx, yy, m_min + r.exponential(0.45, k)]))
    return out


def _obs(seed=0, n=8):
    r = np.random.default_rng(seed)
    return np.column_stack([r.random(n) * 30, r.normal(0, 15, n),
                            r.normal(0, 15, n), 4.0 + r.exponential(0.45, n)])


def test_invariant_to_subthreshold_events(grid, spec):
    """THE load-bearing property. Adding sub-threshold events must not move the
    score by even one ulp — that is what makes points at different mc
    comparable."""
    base = _sims(8, True, 2)
    obs = _obs()
    s0 = score_window(base, obs, grid, spec)

    r = np.random.default_rng(99)
    noisy = []
    for ev in base:
        chatter = np.column_stack([
            r.random(400) * 30, r.uniform(-100, 100, 400), r.uniform(-100, 100, 400),
            np.clip(0.5 + r.exponential(0.45, 400), None, spec.m_target - 1e-3),
        ])
        noisy.append(np.vstack([ev, chatter]) if len(ev) else chatter)
    s1 = score_window(noisy, obs, grid, spec)

    assert s1["ll"] == pytest.approx(s0["ll"], abs=0.0)
    assert s1["ll_level"] == pytest.approx(s0["ll_level"], abs=0.0)
    assert s1["ll_shape"] == pytest.approx(s0["ll_shape"], abs=0.0)
    assert s1["n_expected"] == pytest.approx(s0["n_expected"], abs=0.0)
    assert s1["p_large"] == pytest.approx(s0["p_large"], abs=0.0)


def test_sharper_forecast_scores_better(grid, spec):
    """A forecast concentrated where events actually occur must win."""
    obs = _obs()
    good = score_window(_sims(8, True, 2), obs, grid, spec)
    bad = score_window(_sims(8, False, 3), obs, grid, spec)
    assert good["ll"] > bad["ll"]
    assert aggregate([good], reference=[bad])["ig_per_target_event"] > 0


def test_active_mask_restricts_scoring(grid, spec):
    """The testing-region mask must change the score and exclude outside events."""
    r = np.random.default_rng(5)
    active = grid.active_mask(r.normal(0, 20, 500), r.normal(0, 20, 500))
    assert 0 < active.sum() < grid.n_cells
    obs = _obs()
    full = score_window(_sims(8, True, 2), obs, grid, spec)
    masked = score_window(_sims(8, True, 2), obs, grid, spec, active=active)
    assert masked["ll"] != full["ll"]
    assert masked["n_target_obs"] <= full["n_target_obs"]


def test_rate_field_conserves_expected_count(grid, spec):
    sims = _sims(8, True, 11)
    lam = rate_field(sims, grid, spec)
    direct = np.mean([float((e[:, 3] >= spec.m_target).sum()) if len(e) else 0.0
                      for e in sims])
    assert lam.sum() == pytest.approx(direct, rel=1e-12)


def test_p_large_matches_definition(grid, spec):
    sims = _sims(8, True, 12)
    manual = np.mean([1.0 if len(e) and np.any(e[:, 3] >= spec.m_large) else 0.0
                      for e in sims])
    assert p_at_least_one(sims, spec) == pytest.approx(manual)


def test_poisson_ll_maximised_at_truth():
    """Sanity: the Poisson log-likelihood peaks when lambda equals the counts."""
    obs = np.array([0.0, 2.0, 1.0, 5.0])
    best = poisson_ll(np.maximum(obs, 1e-9), obs)
    for scale in (0.5, 0.75, 1.5, 2.0):
        assert poisson_ll(np.maximum(obs * scale, 1e-9), obs) < best


def test_reliability_bins_are_wellformed():
    w = [{"p_large": p, "obs_large": p > 0.5} for p in np.linspace(0, 1, 40)]
    for b in reliability(w, n_bins=5):
        assert b["n"] > 0
        assert 0.0 <= b["mean_forecast"] <= 1.0
        assert 0.0 <= b["observed_freq"] <= 1.0


def test_aggregate_rejects_mismatched_reference(grid, spec):
    obs = _obs()
    a = score_window(_sims(8, True, 2), obs, grid, spec)
    with pytest.raises(ValueError):
        aggregate([a, a], reference=[a])


# --- fixed magnitude tail (MOONSHOT.md invariant 1c) -----------------------
# These cover the PRIMARY scoring path. The property that matters is that the
# learned magnitude head cannot influence the score at all: if it can, the
# scaling curve is partly measuring GR-head calibration, whose difficulty is
# itself mc-dependent, and the headline slope is contaminated.

@pytest.fixture
def fixed_spec():
    return TargetSpec(m_target=4.0, m_large=6.0, horizon_days=30.0,
                      tail_mode="fixed", mc=2.5, b_value=1.0)


def test_fixed_tail_ignores_sampled_magnitudes(grid, fixed_spec):
    """THE property. Replacing every simulated magnitude must not move the score."""
    base = _sims(8, True, 2)
    obs = _obs()
    s0 = score_window(base, obs, grid, fixed_spec)

    r = np.random.default_rng(4)
    for mag in (0.0, 9.9, None):
        mangled = []
        for ev in base:
            e = ev.copy()
            if len(e):
                e[:, 3] = r.uniform(0, 10, len(e)) if mag is None else mag
            mangled.append(e)
        s1 = score_window(mangled, obs, grid, fixed_spec)
        assert s1["ll"] == pytest.approx(s0["ll"], abs=0.0)
        assert s1["p_large"] == pytest.approx(s0["p_large"], abs=0.0)


def test_fixed_tail_still_rewards_sharper_forecasts(grid, fixed_spec):
    """Removing the magnitude head must not remove rate/location skill."""
    obs = _obs()
    good = score_window(_sims(8, True, 2), obs, grid, fixed_spec)
    bad = score_window(_sims(8, False, 3), obs, grid, fixed_spec)
    assert good["ll"] > bad["ll"]


def test_tail_prob_is_gutenberg_richter(fixed_spec):
    assert fixed_spec.tail_prob(2.5) == pytest.approx(1.0)
    assert fixed_spec.tail_prob(3.5) == pytest.approx(0.1)
    assert fixed_spec.tail_prob(4.5) == pytest.approx(0.01)


def test_fixed_tail_p_large_matches_analytic_thinning(grid, fixed_spec):
    sims = _sims(8, True, 12)
    p = fixed_spec.tail_prob(fixed_spec.m_large)
    manual = np.mean([
        1.0 - (1.0 - p) ** (0 if len(e) == 0 else int(grid.inside(e[:, 1], e[:, 2]).sum()))
        for e in sims])
    assert p_at_least_one(sims, fixed_spec, grid) == pytest.approx(manual)


def test_aki_utsu_recovers_planted_b():
    from flowquake.target_process import aki_utsu_b
    for b_true in (0.8, 1.0, 1.2):
        r = np.random.default_rng(int(b_true * 100))
        m = 2.5 + r.exponential(1.0 / (b_true * np.log(10.0)), 200_000)
        assert aki_utsu_b(m, mc=2.5, dm=0.0) == pytest.approx(b_true, rel=0.02)


def test_aki_utsu_refuses_tiny_samples():
    from flowquake.target_process import aki_utsu_b
    with pytest.raises(ValueError):
        aki_utsu_b(np.array([2.6, 2.7, 3.1]), mc=2.5)


def test_batched_windows_match_sequential_in_distribution():
    """simulate_windows batches all forecast windows into the lane dimension
    (3.9x measured at 52 windows). Lanes are independent, so only scheduling
    changes — the event-count and magnitude distributions must agree."""
    torch = pytest.importorskip("torch")
    from flowquake.data import RECENCY_LAGS, TOKEN_DIM
    from flowquake.model import FlowQuakeTPP
    from flowquake.ntest import simulate_day_events, simulate_windows

    n = 4000
    r = np.random.default_rng(0)
    t = np.sort(r.uniform(0, 2000, n))

    class Cat:
        t_days = t
        feats = torch.randn(n, TOKEN_DIM) * 0.1
        raw = torch.stack([torch.zeros(n),
                           torch.tensor(r.uniform(-50, 50, n), dtype=torch.float32),
                           torch.tensor(r.uniform(-50, 50, n), dtype=torch.float32),
                           torch.tensor(2.5 + r.exponential(0.4, n), dtype=torch.float32)], 1)
        lastk = torch.randn(n, 80, 4).abs()

    stats = {"log_tau_mean": 0.0, "log_tau_std": 1.0, "x_mean": 0.0, "x_std": 1.0,
             "y_mean": 0.0, "y_std": 1.0, "mag_mean": 3.0, "mag_std": 1.0,
             "mcut": 2.5, "bg_area": 1e4,
             "bg_xmin": -50.0, "bg_xmax": 50.0, "bg_ymin": -50.0, "bg_ymax": 50.0,
             "rec_mean": [0.0] * (4 * len(RECENCY_LAGS)),
             "rec_std": [1.0] * (4 * len(RECENCY_LAGS))}
    m = FlowQuakeTPP(d_model=16, n_layers=1, d_state=8, n_heads=2, flow_hidden=16,
                     mix_hidden=16, flow_layers=2, h_bottleneck=0, stats=stats,
                     mix_k=80).eval()
    starts = [float(t[500]), float(t[1500]), float(t[2500])]
    dev = torch.device("cpu")
    torch.manual_seed(0)
    seq = [simulate_day_events(m, Cat, s, 60, dev, sample_steps=2, horizon_days=10.0)
           for s in starts]
    torch.manual_seed(0)
    bat = simulate_windows(m, Cat, starts, 60, dev, sample_steps=2, horizon_days=10.0)

    assert len(bat) == len(seq)
    for w in range(len(starts)):
        assert len(bat[w]) == len(seq[w])
        ns = np.mean([len(e) for e in seq[w]])
        nb = np.mean([len(e) for e in bat[w]])
        assert abs(ns - nb) <= 0.35 * max(ns, 1.0), f"window {w}: {ns} vs {nb}"
        for ev in bat[w]:
            if len(ev):
                assert np.all(ev[:, 0] >= starts[w])
                assert np.all(ev[:, 0] < starts[w] + 10.0)


def test_matched_resolution_scales_sims_inversely_to_event_rate():
    """MOONSHOT.md invariant 1d, as CORRECTED.

    This replaces `test_matched_precision_scales_sims_to_equalise_variance`,
    which asserted the opposite direction and passed against an implementation
    that equalised the ABSOLUTE variance of lambda. That was the wrong
    invariant: a Poisson log-likelihood is biased by the gap between E[log lam]
    and log E[lam], which is governed by the RELATIVE variance 1/(T*p_c) with T
    the total simulated events. Both the tail weight and n_sims cancel out of
    that expression, so the only thing to equalise is T -- and T = n_sims x
    (events per simulation), which means n_sims must scale INVERSELY with the
    catalog's event rate, not upward with mc.

    The old test passing is why the bias survived: it locked in a direction
    derived from the wrong quantity.
    """
    import sys
    import types

    import torch
    sys.path.insert(0, "scripts")
    from scaling_curve import TARGET_SIM_EVENTS, sims_for_matched_resolution

    counts = {2.5: 4702, 2.0: 14789, 1.5: 47276, 1.0: 148907}
    mags, prev = [], 0
    for mc in (2.5, 2.0, 1.5, 1.0):
        mags += [mc + 0.01] * (counts[mc] - prev)
        prev = counts[mc]
    arr = np.array(mags, dtype=np.float32)
    raw = torch.from_numpy(np.stack([np.zeros_like(arr)] * 3 + [arr], axis=1))

    span = 3650.0
    cat = types.SimpleNamespace(t_days=np.array([0.0, span]), raw=raw)
    spec = TargetSpec(m_target=3.0, m_large=4.5, horizon_days=90.0,
                      tail_mode="fixed", b_value=0.869)

    def sims(mc):
        cfg = types.SimpleNamespace(data=types.SimpleNamespace(mcut=mc))
        return sims_for_matched_resolution(cfg, replace(spec, mc=mc), cat)

    # sparser catalog -> MORE simulations, because each one yields fewer events.
    # Non-strict at the bottom: mc 1.5 and 1.0 both need fewer than n_min
    # simulations to clear the target, so both sit on the floor.
    s = [sims(mc) for mc in (2.5, 2.0, 1.5, 1.0)]
    assert all(a >= b for a, b in zip(s, s[1:])), s
    assert s[0] > s[-1], s

    # and T = n_sims x events-per-sim clears the target at EVERY point.
    # A floor, not equality: at mc 1.0 one simulation already yields ~3,700
    # events, so n_sims hits its lower bound and T overshoots. Harmless -- the
    # bias curve is flat up there (-0.03 nats at 20k, -0.003 at 100k).
    Ts = {}
    for mc in (2.5, 2.0, 1.5, 1.0):
        per_sim = counts[mc] / span * spec.horizon_days
        Ts[mc] = sims(mc) * per_sim
        assert Ts[mc] >= TARGET_SIM_EVENTS * 0.98, (mc, Ts)
    # the points that are not floor-clamped land close to the target
    assert Ts[2.5] == pytest.approx(TARGET_SIM_EVENTS, rel=0.05), Ts


@pytest.mark.slow
def test_simulate_windows_memory_is_bounded_at_high_lane_counts():
    """Regression: peak memory must not scale with (lanes x steps).

    The batched simulator originally appended a FULL lane-width array per step.
    With matched-precision n_sims the lane count reaches ~66k (52 windows x 1277
    sims) and the horizon cap is 18,000 steps, so those records ran to tens of
    GB and pushed a 48 GB machine into 34 GB of swap. Records now hold only live
    entries, and `max_lanes` splits the window list.
    """
    import resource
    import torch
    from flowquake.data import RECENCY_LAGS, TOKEN_DIM
    from flowquake.model import FlowQuakeTPP
    from flowquake.ntest import simulate_windows

    n = 3000
    r = np.random.default_rng(0)
    t = np.sort(r.uniform(0, 2000, n))

    class Cat:
        t_days = t
        feats = torch.randn(n, TOKEN_DIM) * 0.1
        raw = torch.stack([torch.zeros(n),
                           torch.tensor(r.uniform(-50, 50, n), dtype=torch.float32),
                           torch.tensor(r.uniform(-50, 50, n), dtype=torch.float32),
                           torch.tensor(2.5 + r.exponential(0.4, n), dtype=torch.float32)], 1)
        lastk = torch.randn(n, 80, 4).abs()

    stats = {"log_tau_mean": 0.0, "log_tau_std": 1.0, "x_mean": 0.0, "x_std": 1.0,
             "y_mean": 0.0, "y_std": 1.0, "mag_mean": 3.0, "mag_std": 1.0,
             "mcut": 2.5, "bg_area": 1e4,
             "bg_xmin": -50.0, "bg_xmax": 50.0, "bg_ymin": -50.0, "bg_ymax": 50.0,
             "rec_mean": [0.0] * (4 * len(RECENCY_LAGS)),
             "rec_std": [1.0] * (4 * len(RECENCY_LAGS))}
    m = FlowQuakeTPP(d_model=16, n_layers=1, d_state=8, n_heads=2, flow_hidden=16,
                     mix_hidden=16, flow_layers=2, h_bottleneck=0, stats=stats,
                     mix_k=80).eval()

    starts = [float(t[i]) for i in range(200, 1400, 60)]     # 20 windows
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS
    out = simulate_windows(m, Cat, starts, 400, torch.device("cpu"),
                           sample_steps=2, horizon_days=30.0, max_lanes=4096)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grew_gb = max(after - before, 0) / (1024 ** 3)

    assert len(out) == len(starts) and all(len(w) == 400 for w in out)
    assert grew_gb < 2.0, f"peak grew {grew_gb:.2f} GB; the record leak is back"


# ---------------------------------------------------------------------------
# Magnitude quantum / b-value.
#
# `test_aki_utsu_recovers_planted_b` above passes dm=0.0 explicitly, so it only
# ever proved the estimator is right when TOLD the quantum. Every real caller
# used the default. On continuous synthetic magnitudes that default returned
# b=0.869 against a true 0.993-1.002, and because the fixed tail extrapolates
# over m_target - mc decades the error came out as an mc-DEPENDENT over-forecast
# (1.16x at mc 2.5 rising to 1.83x at mc 1.0) — a fake slope in the level term.
# These tests exercise the default path.
# ---------------------------------------------------------------------------

def test_magnitude_quantum_detects_binning():
    from flowquake.target_process import magnitude_quantum
    r = np.random.default_rng(0)
    cont = 2.5 + r.exponential(0.434, 50_000)
    assert magnitude_quantum(cont) == 0.0
    assert magnitude_quantum(np.round(cont, 1)) == pytest.approx(0.1)
    assert magnitude_quantum(np.round(cont, 2)) == pytest.approx(0.01)


def test_aki_utsu_default_dm_unbiased_on_continuous_magnitudes():
    """The regression: default dm must not assume 0.1 binning."""
    from flowquake.target_process import aki_utsu_b
    for b_true in (0.8, 1.0, 1.2):
        r = np.random.default_rng(int(b_true * 100))
        m = 2.5 + r.exponential(1.0 / (b_true * np.log(10.0)), 200_000)
        assert aki_utsu_b(m, mc=2.5) == pytest.approx(b_true, rel=0.02)


def test_aki_utsu_default_dm_unbiased_on_binned_magnitudes():
    """Bin BEFORE thresholding, the way a reported catalog is actually built.

    Rounding a distribution that is truncated at exactly mc leaves the lowest
    bin half-width, which inflates the mean excess by dm/2 and drives b to
    1/(ln10*(0.4343+0.05)) = 0.896. That is an artifact of the generator, not
    of the estimator: real catalogs round an unbounded magnitude and threshold
    afterwards, so every bin including the lowest is full width.
    """
    from flowquake.target_process import aki_utsu_b
    r = np.random.default_rng(3)
    m = np.round(1.0 + r.exponential(1.0 / np.log(10.0), 2_000_000), 1)
    assert aki_utsu_b(m[m >= 2.5], mc=2.5) == pytest.approx(1.0, rel=0.03)


# ---------------------------------------------------------------------------
# Level / shape decomposition (CSEP N-test / S-test).
# ---------------------------------------------------------------------------

def test_ll_parts_sum_to_total():
    from scipy.special import gammaln
    from flowquake.target_process import poisson_ll, poisson_ll_parts
    r = np.random.default_rng(1)
    lam = r.gamma(2.0, 1.0, 500)
    n = r.poisson(lam)
    ll, level, shape = poisson_ll_parts(lam, n)
    assert ll == pytest.approx(poisson_ll(lam, n), rel=1e-12)
    assert ll == pytest.approx(level + shape - gammaln(n + 1.0).sum(), rel=1e-12)


def test_shape_term_is_invariant_to_rescaling_lambda():
    """The property that makes shape immune to magnitude-tail error.

    A mis-specified b rescales lambda by a constant factor that depends on mc.
    That moves `level` and must leave `shape` untouched, otherwise the primary
    curve metric inherits the tail bug all over again.
    """
    from flowquake.target_process import poisson_ll_parts
    r = np.random.default_rng(2)
    lam = r.gamma(2.0, 1.0, 500)
    n = r.poisson(lam)
    _, base_level, base_shape = poisson_ll_parts(lam, n)
    for c in (0.2, 1.83, 12.0):
        _, level, shape = poisson_ll_parts(lam * c, n)
        assert shape == pytest.approx(base_shape, abs=1e-9)
        assert abs(level - base_level) > 1e-6


def test_shape_still_rewards_a_sharper_forecast():
    """Invariance to scale must not have cost it discrimination."""
    from flowquake.target_process import poisson_ll_parts
    r = np.random.default_rng(5)
    lam = r.gamma(2.0, 1.0, 500)
    n = r.poisson(lam)
    flat = np.full_like(lam, lam.mean())
    assert poisson_ll_parts(lam, n)[2] > poisson_ll_parts(flat, n)[2]


# ---------------------------------------------------------------------------
# Forecastability of the target series.
#
# Three validation runs were spent before anyone checked whether the synthetic
# catalog was forecastable AT ALL. It was not: variance/mean of the M-target
# window counts was 1.21 against 2.0-5.4 for real regional catalogs, so the
# optimal forecast really was near-constant and every arm came back flat for
# reasons unrelated to the estimator. Both probes agreed -- neural corr +0.154,
# fitted ETAS corr -0.073.
# ---------------------------------------------------------------------------

def test_overdispersion_separates_clustered_from_poisson():
    """The one-line check that would have saved three runs."""
    r = np.random.default_rng(0)

    def var_over_mean(counts):
        c = np.asarray(counts, dtype=float)
        return float(c.var() / c.mean())

    poisson = r.poisson(3.0, 2000)
    assert var_over_mean(poisson) == pytest.approx(1.0, abs=0.15)

    # a clustered series: Poisson rate itself varies window to window
    rate = r.gamma(1.0, 3.0, 2000)
    clustered = r.poisson(rate)
    assert var_over_mean(clustered) > 2.0


def test_flat_forecast_scores_no_better_than_constant_on_poisson_targets():
    """Why a near-Poisson validation catalog cannot certify the estimator.

    When the target series carries no predictable structure, a history-aware
    forecast and a constant one score the same. A validation built on such a
    catalog therefore cannot distinguish a working estimator from a broken
    one, whatever its null arm does.

    The claim is about the EXPECTATION, not one realisation: on any single
    draw a random "informed" forecast can align with Poisson noise by chance
    and score better. Averaged over draws it cannot.
    """
    from flowquake.target_process import poisson_ll_parts
    r = np.random.default_rng(1)
    n_cells = 200
    truth = np.full(n_cells, 0.5)           # homogeneous: nothing to predict
    flat = truth.copy()

    d_flat, d_wig = [], []
    for _ in range(400):
        obs = r.poisson(truth)
        wiggly = truth * np.exp(r.normal(0, 0.4, n_cells))   # structure, but wrong
        d_flat.append(poisson_ll_parts(flat, obs)[2])
        d_wig.append(poisson_ll_parts(wiggly, obs)[2])
    assert np.mean(d_wig) < np.mean(d_flat), (np.mean(d_wig), np.mean(d_flat))


# ---------------------------------------------------------------------------
# Monte-Carlo RESOLUTION of the rate field.
#
# The largest bias found in this pipeline, and it survived an earlier fix that
# equalised the wrong quantity. The score is a Poisson log-likelihood, so the
# Jensen gap between E[log lambda] and log E[lambda] is governed by the
# RELATIVE variance, which for a field built from T total simulated events is
# 1/(T*p_c) -- independent of both the tail weight and the simulation count.
# T grows directly with the catalog rate above mc, so a flat n_sims made it
# vary 14-fold across one grid (93 -> 1319) and produced +3.02 nats/decade on
# the informative arm and +2.62 on a SURROGATE NULL whose true slope is zero.
# ---------------------------------------------------------------------------

def _shape_at_T(T, p, obs, rng, floor_frac=0.001, reps=30):
    from flowquake.target_process import poisson_ll_parts
    n_obs = obs.sum()
    out = []
    for _ in range(reps):
        c = rng.multinomial(T, p).astype(float)
        lam = c / T * n_obs
        lam = (1 - floor_frac) * lam + floor_frac * lam.sum() / len(lam)
        out.append(poisson_ll_parts(lam, obs)[2] / n_obs)
    return float(np.mean(out))


def test_shape_score_is_biased_by_simulation_count():
    """The defect itself: same truth, same observations, only T differs."""
    from flowquake.target_process import poisson_ll_parts
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(400) * 0.3)
    obs = rng.multinomial(120, p).astype(float)
    truth = poisson_ll_parts(p * 120, obs)[2] / 120

    low = _shape_at_T(100, p, obs, rng)
    high = _shape_at_T(20_000, p, obs, rng)
    assert low < high, "more simulated events must resolve the field better"
    assert truth - low > 1.0, (
        f"expected a large bias at T=100, got {truth - low:.3f}")
    assert abs(truth - high) < 0.15, (
        f"T=20000 should be nearly unbiased, got {truth - high:.3f}")


def test_matched_T_removes_the_slope_across_mc():
    """THE regression. Two 'mc points' whose only difference is event rate.

    Unmatched, the sparser point scores far worse purely from resolution and a
    fake slope appears. Matched on T, the difference collapses.
    """
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(400) * 0.3)
    obs = rng.multinomial(120, p).astype(float)

    # events per simulated window at two thresholds, a 14x spread as measured
    per_sim = {"high_mc": 0.9, "low_mc": 13.2}
    n_sims_flat = 100
    unmatched = {k: _shape_at_T(max(int(v * n_sims_flat), 1), p, obs, rng)
                 for k, v in per_sim.items()}
    T = 20_000
    matched = {k: _shape_at_T(T, p, obs, rng) for k in per_sim}

    gap_unmatched = unmatched["low_mc"] - unmatched["high_mc"]
    gap_matched = matched["low_mc"] - matched["high_mc"]
    assert gap_unmatched > 2.0, f"expected a large fake gap, got {gap_unmatched:.3f}"
    assert abs(gap_matched) < 0.1, f"matched gap should vanish, got {gap_matched:.3f}"


def test_sims_for_matched_resolution_holds_T_constant():
    """The helper must return n_sims inversely proportional to the event rate."""
    import types
    from scripts.scaling_curve import sims_for_matched_resolution, TARGET_SIM_EVENTS

    class _Col:                       # mimics a torch column: .numpy()
        def __init__(self, a): self._a = a
        def numpy(self): return self._a

    class _Raw:                       # mimics a torch tensor: raw[:, 3].numpy()
        def __init__(self, n): self._n = n
        def __getitem__(self, key): return _Col(np.full(self._n, 5.0))

    class _Cat:
        def __init__(self, n, span=1000.0):
            self.t_days = np.array([0.0, span])
            self.raw = _Raw(n)

    spec = TargetSpec(m_target=4.0, m_large=6.0, horizon_days=1.0,
                      tail_mode="fixed", mc=2.5, b_value=1.0)
    got = {}
    for n_events in (1_000, 14_000):
        cfg = types.SimpleNamespace(data=types.SimpleNamespace(mcut=1.0))
        n = sims_for_matched_resolution(cfg, spec, _Cat(n_events))
        per_sim = n_events / 1000.0 * spec.horizon_days
        got[n_events] = n * per_sim              # this is T
    # T must be the same at both rates, and equal to the target
    assert got[1_000] == pytest.approx(got[14_000], rel=0.02), got
    assert got[1_000] == pytest.approx(TARGET_SIM_EVENTS, rel=0.02), got


def test_empty_windows_contribute_exactly_zero_to_shape():
    """Licenses the budget split in scripts/scaling_curve.py:score_point.

    shape = sum_c n_c log(lambda_c / Lambda). With no observed targets every
    n_c is zero, so the window contributes EXACTLY zero regardless of the
    forecast. At a 1-day horizon 94% of windows are empty, so simulating them
    at full resolution buys nothing -- and economising there cannot bias the
    primary metric, because the contribution is zero by identity rather than by
    approximation.

    The level term is different: an empty window still contributes -Lambda. But
    Lambda-hat is an unbiased mean at any n_sims, since the N*log(Lambda) piece
    vanishes when N = 0 and there is no Jensen term.
    """
    from flowquake.target_process import poisson_ll_parts
    rng = np.random.default_rng(0)
    obs = np.zeros(300)
    for scale in (0.5, 1.0, 50.0):
        lam = rng.gamma(2.0, scale, 300)
        ll, level, shape = poisson_ll_parts(lam, obs)
        assert shape == 0.0, f"empty window must give shape 0, got {shape}"
        assert level == pytest.approx(-lam.sum(), rel=1e-12)


def test_level_of_an_empty_window_is_unbiased_in_n_sims():
    """So a coarse simulation budget on empty windows costs variance, not bias."""
    from flowquake.target_process import poisson_ll_parts
    rng = np.random.default_rng(1)
    truth = rng.gamma(2.0, 0.05, 200)
    obs = np.zeros(200)
    exact = poisson_ll_parts(truth, obs)[1]
    for T in (200, 20_000):
        est = [poisson_ll_parts(rng.poisson(truth * T) / T, obs)[1]
               for _ in range(400)]
        assert np.mean(est) == pytest.approx(exact, rel=0.02), (T, np.mean(est))


def test_simulate_windows_splits_sims_when_one_window_exceeds_the_budget():
    """Regression: a single window with n_sims > max_lanes must not allocate
    them all at once.

    The window-splitting path computes `per = max_lanes // n_sims` and, when
    that is zero, the old code fell back to `per = 1` and simulated the whole
    window anyway -- so the lane budget silently did nothing. Matched-resolution
    n_sims reaches ~98,500 for one window at a 1-day horizon and mc 2.5, which
    is the same unbounded working set that drove a 48 GB machine into 34 GB of
    swap. Correct behaviour is to split the SIMULATIONS and concatenate.
    """
    torch = pytest.importorskip("torch")
    from flowquake.data import RECENCY_LAGS, TOKEN_DIM
    from flowquake.model import FlowQuakeTPP
    from flowquake.ntest import simulate_windows

    n = 2000
    r = np.random.default_rng(0)
    tt = np.sort(r.uniform(0, 2000, n))

    class Cat:
        t_days = tt
        feats = torch.randn(n, TOKEN_DIM) * 0.1
        raw = torch.stack([torch.zeros(n),
                           torch.tensor(r.uniform(-50, 50, n), dtype=torch.float32),
                           torch.tensor(r.uniform(-50, 50, n), dtype=torch.float32),
                           torch.tensor(2.5 + r.exponential(0.4, n), dtype=torch.float32)], 1)
        lastk = torch.randn(n, 80, 4).abs()

    stats = {"log_tau_mean": 0.0, "log_tau_std": 1.0, "x_mean": 0.0, "x_std": 1.0,
             "y_mean": 0.0, "y_std": 1.0, "mag_mean": 3.0, "mag_std": 1.0,
             "mcut": 2.5, "bg_area": 1e4,
             "bg_xmin": -50.0, "bg_xmax": 50.0, "bg_ymin": -50.0, "bg_ymax": 50.0,
             "rec_mean": [0.0] * (4 * len(RECENCY_LAGS)),
             "rec_std": [1.0] * (4 * len(RECENCY_LAGS))}
    m = FlowQuakeTPP(d_model=16, n_layers=1, d_state=8, n_heads=2, flow_hidden=16,
                     mix_hidden=16, flow_layers=2, h_bottleneck=0, stats=stats,
                     mix_k=80).eval()

    starts = [float(tt[1500])]
    n_sims, max_lanes = 4096, 512
    sims = simulate_windows(m, Cat, starts, n_sims, torch.device("cpu"),
                            sample_steps=2, horizon_days=1.0, max_lanes=max_lanes)
    assert len(sims) == 1
    assert len(sims[0]) == n_sims, (
        f"got {len(sims[0])} simulations, expected {n_sims} — the split must "
        "concatenate, not truncate")


def test_shape_is_finite_when_unoccupied_cells_have_zero_rate():
    """0 * log(0) must be treated as 0, not nan.

    numpy evaluates `0 * -inf` as nan, so summing the shape term over ALL cells
    makes the score depend on whether an unoccupied cell happens to have zero
    rate. Production waters the field first, but a nan here would be
    indistinguishable from a missing curve point, and the identity
    "unoccupied cells contribute nothing" should hold structurally rather than
    by the caller remembering to water.
    """
    from flowquake.target_process import poisson_ll_parts
    lam = np.array([0.0, 0.0, 2.0, 1.0])
    obs = np.array([0.0, 0.0, 3.0, 1.0])
    ll, level, shape = poisson_ll_parts(lam, obs)
    assert np.isfinite(shape), shape
    assert np.isfinite(level)
    # matches the hand computation over occupied cells only
    L = lam.sum()
    assert shape == pytest.approx(3 * np.log(2.0 / L) + 1 * np.log(1.0 / L))


def test_frame_records_its_time_origin():
    """A frame's `start_days` is meaningless without the origin it is measured from.

    Two catalogs being compared need not share a first event. The surrogate null
    shifts the earliest small event, moving its catalog's first-event time by
    1.7 hours relative to the informative arm -- so a consumer that recomputed
    t_days from its own catalog put every null-arm forecast 7% of a 1-day window
    out of step with the arm it is compared against, while the informative arm
    stayed correctly aligned. An asymmetric misalignment between arms biases the
    comparison, not just the noise.
    """
    import tempfile
    from pathlib import Path

    import scripts.scaling_curve as sc

    cfg = sc.Config.load("configs/panel_white.yaml")
    spec = TargetSpec(m_target=3.0, m_large=4.0, horizon_days=1.0,
                      tail_mode="fixed")
    with tempfile.TemporaryDirectory() as td:
        fr = sc.build_frame(cfg, spec, bin_km=5.0, out=Path(td), b_mc=2.5)
        assert "t0" in fr, "frame must record the absolute origin of start_days"
        import pandas as pd
        t0 = pd.Timestamp(fr["t0"])
        df = pd.read_csv(cfg.data.catalog_path, parse_dates=["time"])
        assert t0 == df["time"].min()
        # and start_days must be consistent with it
        first = min(w["start_days"] for w in fr["windows"])
        recovered = t0 + pd.Timedelta(days=first)
        assert recovered >= pd.Timestamp(cfg.data.test_start) - pd.Timedelta(days=1)
