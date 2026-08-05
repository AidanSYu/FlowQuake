"""The optimised RK4 sampler must be BIT-IDENTICAL to the obvious loop.

`CondFlow.sample` was rewritten for speed: the Fourier time embedding is
computed once per distinct sub-step time instead of across all lanes on every
velocity call, and the network input is written into one preallocated buffer
instead of being rebuilt by `torch.cat`. A CPU profile put `cat` at 17.5% of
runtime and `sin`/`cos` at 11%, inside a sampler that is 98.5% of scoring cost.

Neither change is an approximation, so "close enough" is the wrong bar. This
test keeps a literal transcription of the ORIGINAL loop and asserts the two
agree exactly, because the failure that matters here is silent: a wrong column
order or a stale buffer row would still return plausible samples and would only
show up as a slightly different forecast, indistinguishable from Monte-Carlo
noise in an unseeded run.
"""
import torch

from flowquake.flow import CondFlow


def _reference_sample(flow: CondFlow, cond: torch.Tensor, steps: int) -> torch.Tensor:
    """The pre-optimisation implementation, verbatim.

    Kept here rather than in the module so the optimised version has something
    independent to be checked against.
    """
    z = torch.randn(cond.shape[0], flow.dim, device=cond.device, dtype=cond.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t0 = torch.tensor(i * dt, device=z.device, dtype=z.dtype)
        k1 = flow.velocity(z, t0, cond)
        k2 = flow.velocity(z + 0.5 * dt * k1, t0 + 0.5 * dt, cond)
        k3 = flow.velocity(z + 0.5 * dt * k2, t0 + 0.5 * dt, cond)
        k4 = flow.velocity(z + dt * k3, t0 + dt, cond)
        z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def _flow(seed=0, dim=2, cond_dim=5, hidden=32, n_layers=3):
    torch.manual_seed(seed)
    f = CondFlow(dim=dim, cond_dim=cond_dim, hidden=hidden, n_layers=n_layers).eval()
    # The last layer is zero-initialised by design, which would make the whole
    # velocity field zero and the comparison vacuous. Give it real weights.
    with torch.no_grad():
        f.net[-1].weight.normal_(0, 0.3)
        f.net[-1].bias.normal_(0, 0.1)
    return f


@torch.no_grad()
def test_optimised_sample_is_bit_identical():
    for steps in (4, 16):
        for B in (1, 7, 64):
            f = _flow()
            cond = torch.randn(B, 5)
            torch.manual_seed(99)
            got = f.sample(cond, steps=steps)
            torch.manual_seed(99)
            want = _reference_sample(f, cond, steps=steps)
            assert torch.equal(got, want), (
                f"steps={steps} B={B}: optimised sample diverged from reference; "
                f"max|diff| = {(got - want).abs().max().item():.3e}")


@torch.no_grad()
def test_reference_is_not_trivially_zero():
    """Guards the comparison above from passing on an all-zero velocity field."""
    f = _flow()
    cond = torch.randn(16, 5)
    torch.manual_seed(3)
    out = f.sample(cond, steps=8)
    torch.manual_seed(3)
    z0 = torch.randn(16, f.dim)
    assert not torch.allclose(out, z0), "flow did not move the sample at all"


@torch.no_grad()
def test_buffer_is_not_reused_across_substeps():
    """Each RK4 stage must see its OWN z, not a leftover from the previous one.

    A preallocated buffer invites exactly this bug: forget to rewrite the z
    block and every stage silently evaluates the velocity at the same point,
    turning RK4 into a much worse integrator that still returns numbers.
    Detected by checking the four stages genuinely differ.
    """
    f = _flow()
    cond = torch.randn(8, 5)
    dt = 1.0 / 8
    z = torch.randn(8, f.dim)
    t0 = torch.tensor(0.0)
    k1 = f.velocity(z, t0, cond)
    k2 = f.velocity(z + 0.5 * dt * k1, t0 + 0.5 * dt, cond)
    assert not torch.equal(k1, k2), "RK4 stages are identical -- z is not being updated"


@torch.no_grad()
def test_compiled_net_never_enters_state_dict():
    """A compiled wrapper must not change the checkpoint format.

    `torch.compile` returns an `OptimizedModule`, which IS an nn.Module, so the
    obvious `self._compiled_net = torch.compile(...)` makes nn.Module register
    it as a child: every weight then appears a second time under a
    `_compiled_net.*` prefix. Caught in exactly that form -- the first version of
    this change altered state_dict keys -- and it is the dangerous kind of bug,
    because it silently rewrites every checkpoint saved afterwards while the
    model keeps working.
    """
    f = _flow()
    before = set(f.state_dict())
    if not f.enable_compiled_net():
        import pytest
        pytest.skip("torch.compile unavailable in this environment")
    assert set(f.state_dict()) == before, (
        "enabling the compiled net changed state_dict keys: "
        f"{sorted(set(f.state_dict()) - before)}")


@torch.no_grad()
def test_compile_is_off_by_default():
    """Nothing should silently opt a run into non-bit-identical arithmetic."""
    f = _flow()
    assert not f._compiled_box, "compiled net must be opt-in"
