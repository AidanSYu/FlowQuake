"""Pooling the moonshot answer across regions.

The one thing that can silently go wrong here is UNPAIRING. Within a region the
FlowQuake and ETAS slopes come from the same bootstrap window draw, so they rise
and fall together; the difference is far better determined than either slope.
Forming the difference from the two marginal standard errors instead of per-draw
would inflate its interval and could turn a decisive contrast into an
inconclusive one. Nothing about the output shape reveals which was done, so it
is tested directly.
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "pool_moonshot", ROOT / "scripts" / "pool_moonshot.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


def _write(path, sl_n, sl_e, mcs=(2.5, 2.0, 1.5, 1.0), neural=None, etas=None,
           n_win=1000, n_tgt=100):
    mcs = np.asarray(mcs, dtype=float)
    if neural is None:
        neural = -4.0 - 0.5 * (2.5 - mcs)      # falls as mc drops
    if etas is None:
        etas = -5.0 + 0.2 * (2.5 - mcs)        # rises as mc drops
    np.savez(path, slope_neural=np.asarray(sl_n, float),
             slope_etas=np.asarray(sl_e, float), mcs=mcs,
             neural=np.asarray(neural, float), etas=np.asarray(etas, float),
             n_windows=np.int64(n_win), n_targets=np.int64(n_tgt))


def test_difference_is_pooled_paired_not_from_marginals(tmp_path):
    """Correlated draws must yield a difference far tighter than the marginals."""
    rng = np.random.default_rng(0)
    common = rng.normal(0, 1.0, 4000)          # the shared window-draw effect
    sl_n = -0.7 + common                       # both slopes move together...
    sl_e = 0.2 + common + rng.normal(0, 0.01, 4000)   # ...difference is stable
    _write(tmp_path / "a.npz", sl_n, sl_e)

    out = tmp_path / "o.json"
    pm.main(["--draws", f"a={tmp_path/'a.npz'}", "--out", str(out)])
    got = json.loads(out.read_text())

    se_n = got["FlowQuake"]["se"]
    se_e = got["ETAS"]["se"]
    se_d = got["ETAS - FlowQuake"]["se"]
    naive = np.hypot(se_n, se_e)
    # The paired se is ~0.01; the naive combination is ~1.4. Two orders apart.
    assert se_d < 0.1 * naive, (se_d, naive)
    assert se_d == pytest.approx(0.01, abs=0.005)


def test_point_estimates_follow_the_sign_convention(tmp_path):
    """Positive = a deeper catalog HELPS. Neural falls, ETAS rises.

    The point estimates come from the `neural`/`etas` curves, NOT from the
    draws, so the draws here only have to be non-degenerate.
    """
    rng = np.random.default_rng(7)
    _write(tmp_path / "a.npz", rng.normal(-0.7, 0.1, 500),
           rng.normal(0.2, 0.1, 500))
    out = tmp_path / "o.json"
    pm.main(["--draws", f"a={tmp_path/'a.npz'}", "--out", str(out)])
    r = json.loads(out.read_text())["regions"][0]
    assert r["pt_n"] < 0          # -0.5 per decade by construction
    assert r["pt_e"] > 0          # +0.2 per decade by construction
    assert r["pt_n"] == pytest.approx(-0.5, abs=1e-9)
    assert r["pt_e"] == pytest.approx(+0.2, abs=1e-9)
    assert r["span"] == pytest.approx(1.5)


def test_two_regions_pool_between_their_slopes(tmp_path):
    """A pooled estimate must lie between the regions, not outside them."""
    rng = np.random.default_rng(1)
    _write(tmp_path / "a.npz", rng.normal(-0.8, 0.1, 3000),
           rng.normal(0.3, 0.1, 3000))
    _write(tmp_path / "b.npz", rng.normal(-0.4, 0.1, 3000),
           rng.normal(0.1, 0.1, 3000),
           mcs=(2.5, 2.1, 1.7),
           neural=[-4.0, -4.2, -4.4], etas=[-5.0, -4.95, -4.9])
    out = tmp_path / "o.json"
    pm.main(["--draws", f"a={tmp_path/'a.npz'}",
             "--draws", f"b={tmp_path/'b.npz'}", "--out", str(out)])
    got = json.loads(out.read_text())
    pts = [r["pt_n"] for r in got["regions"]]
    est = got["FlowQuake"]["estimate"]
    assert min(pts) <= est <= max(pts)
    assert got["FlowQuake"]["q_p"] <= 1.0
    assert len(got["regions"]) == 2
    # The two panels have different mc spans; both must be recorded, because a
    # per-decade slope read off 0.8 decades extrapolates further than one read
    # off 1.5 and the reader has to be able to see that.
    assert {round(r["span"], 1) for r in got["regions"]} == {1.5, 0.8}


def test_missing_path_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        pm.main(["--draws", "noequalssign", "--out", str(tmp_path / "o.json")])


def test_degenerate_region_is_refused_not_silently_dropped(tmp_path):
    """A zero-variance region must abort the pool, naming itself.

    `random_effects_pool` filters such panels out without comment, so a
    two-region pool would keep printing "POOLED" while reporting one region.
    That is the failure this guard exists to prevent.
    """
    rng = np.random.default_rng(3)
    _write(tmp_path / "good.npz", rng.normal(-0.8, 0.1, 500),
           rng.normal(0.3, 0.1, 500))
    _write(tmp_path / "flat.npz", np.full(500, -0.4), np.full(500, 0.1))
    with pytest.raises(SystemExit) as e:
        pm.main(["--draws", f"good={tmp_path/'good.npz'}",
                 "--draws", f"flat={tmp_path/'flat.npz'}",
                 "--out", str(tmp_path / "o.json")])
    assert "flat" in str(e.value)
    assert "good" not in str(e.value)
