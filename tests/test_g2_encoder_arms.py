"""Gate G2: the three missing controls on the memorization ablation.

MANUSCRIPT.md 4.3 concludes "flexibility causes memorization, the cure is
structural". But the encoder was handed ABSOLUTE x, y and trained ~600+ epochs,
and none of the standard fixes for coordinate memorization was tried. These
tests assert the arms do what they claim, so the rerun is meaningful:

  full       absolute coordinates reach the encoder (the published setting)
  safe       they cannot, by construction
  augmented  they are present but carry no stable signal across passes
"""

import torch

from flowquake.data import RECENCY_LAGS, TOKEN_DIM
from flowquake.model import SAFE_TOKEN_DIMS, FlowQuakeTPP

STATS = {"log_tau_mean": 0.0, "log_tau_std": 1.0, "x_mean": 0.0, "x_std": 1.0,
         "y_mean": 0.0, "y_std": 1.0, "mag_mean": 3.0, "mag_std": 1.0,
         "mcut": 2.5, "bg_area": 1e5, "rec_mean": [0.0] * (4 * len(RECENCY_LAGS)),
         "rec_std": [1.0] * (4 * len(RECENCY_LAGS))}


def _model(mode, h=4):
    return FlowQuakeTPP(d_model=16, n_layers=1, d_state=8, n_heads=2,
                        flow_hidden=16, mix_hidden=16, flow_layers=2,
                        h_bottleneck=h, stats=STATS, encoder_input=mode)


def test_safe_arm_drops_absolute_coordinates():
    """Shifting the frame must leave a `safe` encoder's input untouched."""
    m = _model("safe").eval()
    t = torch.randn(2, 32, TOKEN_DIM)
    shifted = t.clone()
    shifted[..., 1] += 500.0     # absolute x
    shifted[..., 2] -= 300.0     # absolute y
    assert torch.equal(m._encoder_input(t), m._encoder_input(shifted))
    assert m._encoder_input(t).shape[-1] == len(SAFE_TOKEN_DIMS)


def test_full_arm_keeps_absolute_coordinates():
    """The published setting must be sensitive to the frame — that is the point."""
    m = _model("full").eval()
    t = torch.randn(2, 32, TOKEN_DIM)
    shifted = t.clone()
    shifted[..., 1] += 500.0
    assert not torch.equal(m._encoder_input(t), m._encoder_input(shifted))
    assert m._encoder_input(t).shape[-1] == TOKEN_DIM


def test_augmented_arm_varies_across_passes_but_only_in_training():
    m = _model("augmented")
    t = torch.randn(2, 32, TOKEN_DIM)
    m.train()
    torch.manual_seed(0); a = m._encoder_input(t)
    torch.manual_seed(1); b = m._encoder_input(t)
    assert not torch.allclose(a, b), "augmentation must resample each pass"
    m.eval()
    assert torch.equal(m._encoder_input(t), t), "no augmentation at eval time"


def test_augmentation_preserves_relational_geometry():
    """A rigid transform must not change inter-event distances — otherwise it is
    corrupting the signal rather than removing an absolute reference."""
    m = _model("augmented").train()
    t = torch.zeros(1, 8, TOKEN_DIM)
    t[..., 1] = torch.tensor([0.0, 3.0, 6.0, 9.0, 1.0, 2.0, 3.0, 4.0])
    t[..., 2] = torch.tensor([0.0, 4.0, 8.0, 12.0, 1.0, 2.0, 3.0, 4.0])
    out = m._encoder_input(t)
    d0 = torch.cdist(t[0, :, 1:3], t[0, :, 1:3])
    d1 = torch.cdist(out[0, :, 1:3], out[0, :, 1:3])
    assert torch.allclose(d0, d1, atol=1e-4)


def test_all_arms_train_and_evaluate():
    """Every arm must survive a forward/backward and the eval path, since the
    eval path builds encoder input separately (model.encode_full)."""
    B, W = 2, 40
    for mode in ("full", "safe", "augmented"):
        m = _model(mode)
        tokens = torch.randn(B, W, TOKEN_DIM)
        target = torch.randn(B, W, 4)
        mask = torch.zeros(B, W, dtype=torch.bool); mask[:, 10:] = True
        lastk = torch.randn(B, W, m.head_s.n_comp, 4).abs()
        raw_next = torch.randn(B, W, 3); raw_next[..., 2] = 3.0
        out = m.fm_losses(tokens, target, mask, lastk, raw_next)
        loss = out[0] if isinstance(out, tuple) else out["loss"]
        loss.backward()
        assert torch.isfinite(loss), mode
        m.eval()
        with torch.no_grad():
            h = m.encode_full(tokens, segment=16)
        assert h.shape[:2] == (B, W), mode
