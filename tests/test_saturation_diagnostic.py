"""Per-step increments must aggregate exactly as the committed artifacts do.

`ll_shape` is a per-window TOTAL, so nats-per-target-event is

    sum(scores) / sum(targets)

NOT a target-weighted mean of per-window values. I wrote the weighted form
first, and it is the dangerous kind of wrong: it runs, it is monotone, it
preserves the sign of every increment, and it reports -10.3212 where the
artifact's own `ll_shape_per_target_event` is -5.5511. Nothing about the output
looks broken -- it simply answers a different question, and the resulting
"overstatement factor" came out 4.0x instead of 6.0x.

These tests pin the aggregation against a hand-computable case and against the
pairing property that makes each increment a within-window contrast.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "saturation_diagnostic",
    Path(__file__).resolve().parent.parent / "scripts" / "saturation_diagnostic.py")
sat = importlib.util.module_from_spec(_spec)
sys.modules["saturation_diagnostic"] = sat
_spec.loader.exec_module(sat)


def test_aggregate_is_sum_over_sum_not_a_weighted_mean():
    # Two windows: one carries 1 target and scores -1, the other carries 9
    # targets and scores -18 (i.e. -2 per target).
    scores = np.array([-1.0, -18.0])
    tgt = np.array([1.0, 9.0])

    # correct: total nats / total targets
    assert sat.weighted(scores, tgt) == (-19.0) / 10.0 == -1.9

    # the wrong form would weight per-window values by target count
    wrong = (scores * tgt).sum() / tgt.sum()
    assert wrong == (-1.0 * 1 + -18.0 * 9) / 10.0
    assert not np.isclose(wrong, -1.9), "fixture no longer separates the two forms"


def test_aggregate_matches_a_committed_artifact():
    """The check that actually caught the bug: agree with the artifact's own key."""
    import json
    f = (Path(__file__).resolve().parent.parent / "runs/real_validation_h1"
         / "etas_subgrid/informative/uniform_mc2.5/target_process.json")
    if not f.exists():                      # artifact not present in a fresh clone
        return
    j = json.load(open(f))
    w = j["windows"]
    s = np.array([x["ll_shape"] for x in w], dtype=float)
    t = np.array([x["n_target_obs"] for x in w], dtype=float)
    assert np.isclose(sat.weighted(s, t),
                      j["aggregate"]["ll_shape_per_target_event"], rtol=1e-12)


def test_zero_targets_gives_nan_not_a_divide_by_zero():
    assert np.isnan(sat.weighted(np.array([-1.0, -2.0]), np.array([0.0, 0.0])))


def test_increments_are_paired_across_mc():
    """One window draw per replicate, applied at every mc.

    Build two mc levels whose per-window scores differ by exactly a constant.
    Then every bootstrap replicate must see exactly that constant, so the
    increment's CI collapses to a point. Independent resampling per mc would
    smear it.
    """
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(-5.0, 2.0, n)
    tgt = rng.integers(0, 3, n).astype(float)
    tgt[0] = 1.0                                   # guarantee a nonzero total
    delta = 0.25
    pts = {2.5: (base, tgt), 2.0: (base + delta * tgt, tgt)}

    mcs, rows = sat.step_increments(pts, n_boot=200, seed=1)
    assert mcs == [2.5, 2.0]
    r = rows[0]
    # sum((base + delta*tgt)) / sum(tgt) - sum(base)/sum(tgt) == delta exactly
    assert np.isclose(r["increment"], delta, rtol=1e-12)
    assert np.isclose(r["ci_lo"], delta, atol=1e-9)
    assert np.isclose(r["ci_hi"], delta, atol=1e-9)


def test_increment_sign_is_lower_mc_minus_higher_mc():
    """A curve that improves as mc drops must report POSITIVE increments."""
    n = 50
    tgt = np.ones(n)
    pts = {2.5: (np.full(n, -6.0), tgt), 2.0: (np.full(n, -5.0), tgt)}
    _, rows = sat.step_increments(pts, n_boot=50, seed=0)
    assert rows[0]["from_mc"] == 2.5 and rows[0]["to_mc"] == 2.0
    assert np.isclose(rows[0]["increment"], +1.0)
    assert np.isclose(rows[0]["decades"], 0.5)
