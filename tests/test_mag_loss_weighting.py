"""GR-tilting of the training objective.

The hypothesis this machinery exists to test: the model degrades on deeper
catalogs because the catalog is Gutenberg-Richter distributed, so lowering mc
adds events in geometrically increasing numbers at the small end and the loss
quietly becomes a loss about micro-seismicity, while scoring stays on rare large
events. Weighting each event by 10^(gamma*(m-mc)) undoes that drift, and
gamma = b makes every magnitude decade contribute equally.

Two things have to hold or the experiment is worthless. gamma = 0 must reproduce
the untilted objective EXACTLY, since every published number was produced under
it. And the weights must actually favour large events, in the right direction,
by the right factor.
"""
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowquake.flow import CondFlow  # noqa: E402


def _flow(seed=0):
    torch.manual_seed(seed)
    return CondFlow(1, cond_dim=5, hidden=16, n_layers=2, sigma_min=0.02)


def test_unit_weights_match_the_unweighted_loss():
    """weights=ones must equal weights=None, or gamma=0 is not a no-op."""
    f = _flow()
    u = torch.randn(64, 1)
    cond = torch.randn(64, 5)
    torch.manual_seed(99)
    plain = f.fm_loss(u, cond)
    torch.manual_seed(99)
    ones = f.fm_loss(u, cond, weights=torch.ones(64))
    assert torch.allclose(plain, ones, atol=1e-6), (plain.item(), ones.item())


def test_weights_change_the_loss_when_not_uniform():
    """A non-uniform weighting must actually move the number."""
    f = _flow()
    u = torch.randn(64, 1)
    cond = torch.randn(64, 5)
    w = torch.linspace(0.1, 2.0, 64)
    w = w / w.mean()
    torch.manual_seed(7)
    plain = f.fm_loss(u, cond)
    torch.manual_seed(7)
    tilted = f.fm_loss(u, cond, weights=w)
    assert not torch.allclose(plain, tilted, atol=1e-6)


def test_gr_weights_favour_large_events_by_the_right_factor():
    """w = 10^(gamma (m - mc)), normalised to mean 1.

    At gamma = b the ratio between two events one magnitude unit apart must be
    exactly 10^b, because that is what cancels the GR frequency decline.
    """
    mc, gamma = 1.0, 0.95
    m = torch.tensor([1.0, 2.0, 3.0])
    w = torch.pow(10.0, gamma * (m - mc))
    w = w / w.mean()
    assert w[1] / w[0] == pytest.approx(10.0 ** gamma, rel=1e-6)
    assert w[2] / w[1] == pytest.approx(10.0 ** gamma, rel=1e-6)
    assert w.mean() == pytest.approx(1.0, rel=1e-6)
    assert w[0] < w[1] < w[2]          # larger events weigh more, not less


def test_zero_gamma_gives_exactly_uniform_weights():
    m = torch.tensor([0.5, 1.0, 4.0, 6.0])
    w = torch.pow(10.0, 0.0 * (m - 1.0))
    assert torch.allclose(w / w.mean(), torch.ones(4))


def test_model_carries_the_gamma_attribute():
    """The tilt has to survive config -> make_model, or runs silently untilt."""
    import dataclasses as dc

    from flowquake.config import Config
    from flowquake.train import make_model

    cfg = Config.load(str(ROOT / "configs" / "panel_white.yaml"))
    assert cfg.model.mag_loss_gamma == 0.0            # published default
    assert make_model(cfg, None).mag_loss_gamma == 0.0
    cfg.model = dc.replace(cfg.model, mag_loss_gamma=0.95)
    assert make_model(cfg, None).mag_loss_gamma == pytest.approx(0.95)
