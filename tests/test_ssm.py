import torch

from flowquake.ssm import (
    SSDBlock,
    SSMEncoder,
    selective_scan_chunked,
    selective_scan_ref,
)


def _rand_inputs(B=2, L=200, H=3, P=5, N=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, H, P, generator=g)
    dt = torch.rand(B, L, H, generator=g) * 0.1 + 1e-3
    A = torch.rand(H, generator=g) * 10 + 0.5
    Bm = torch.randn(B, L, N, generator=g)
    Cm = torch.randn(B, L, N, generator=g)
    return x, dt, A, Bm, Cm


def test_chunked_scan_matches_naive():
    x, dt, A, Bm, Cm = _rand_inputs(L=200)  # not a multiple of chunk
    y, s = selective_scan_chunked(x, dt, A, Bm, Cm, chunk=64)
    y_ref, s_ref = selective_scan_ref(x, dt, A, Bm, Cm)
    assert torch.allclose(y, y_ref.float(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(s, s_ref.float(), atol=1e-4, rtol=1e-4)


def test_scan_init_state_continuation():
    """Scanning [first half] then [second half with carried state] == full scan."""
    x, dt, A, Bm, Cm = _rand_inputs(L=128)
    y_full, s_full = selective_scan_chunked(x, dt, A, Bm, Cm, chunk=32)
    y1, s1 = selective_scan_chunked(x[:, :64], dt[:, :64], A, Bm[:, :64], Cm[:, :64], chunk=32)
    y2, s2 = selective_scan_chunked(
        x[:, 64:], dt[:, 64:], A, Bm[:, 64:], Cm[:, 64:], chunk=32, init_state=s1
    )
    assert torch.allclose(torch.cat([y1, y2], dim=1), y_full, atol=1e-4)
    assert torch.allclose(s2, s_full, atol=1e-4)


def test_block_causality():
    torch.manual_seed(0)
    block = SSDBlock(d_model=32, d_state=8, n_heads=4, chunk=16).eval()
    u = torch.randn(1, 100, 32)
    y1 = block(u)
    u2 = u.clone()
    u2[:, 60:] += 100.0  # perturb the future
    y2 = block(u2)
    assert torch.allclose(y1[:, :60], y2[:, :60], atol=1e-5)
    assert not torch.allclose(y1[:, 60:], y2[:, 60:], atol=1e-2)


def test_encoder_prefill_matches_forward():
    torch.manual_seed(0)
    enc = SSMEncoder(d_in=4, d_model=32, n_layers=2, d_state=8, n_heads=4, chunk=16).eval()
    feats = torch.randn(1, 90, 4)
    h_fwd = enc(feats)
    h1, st = enc.prefill(feats[:, :50])
    h2, _ = enc.prefill(feats[:, 50:], st)
    h_seg = torch.cat([h1, h2], dim=1)
    assert torch.allclose(h_fwd, h_seg, atol=1e-4)


def test_encoder_step_matches_forward():
    torch.manual_seed(0)
    enc = SSMEncoder(d_in=4, d_model=32, n_layers=2, d_state=8, n_heads=4, chunk=16).eval()
    feats = torch.randn(1, 40, 4)
    h_fwd = enc(feats)
    _, st = enc.prefill(feats[:, :30])
    h = None
    for i in range(30, 40):
        h, st = enc.step(feats[:, i], st)
    assert torch.allclose(h_fwd[:, -1], h, atol=1e-4)
