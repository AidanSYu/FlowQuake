import torch

from flowquake.heads import GRMagnitudeHead, KernelMixtureHead


def test_kernel_mixture_normalizes():
    """Power-law mixture + uniform bg must integrate to ~1 over the bg box."""
    torch.manual_seed(0)
    K, C = 4, 8
    head = KernelMixtureHead(cond_dim=C, n_comp=K)
    comp_xy = torch.tensor([[[0.0, 0], [5, 5], [-4, 3], [2, -6]]])
    comp_feats = torch.randn(1, K, 3)
    cond = torch.randn(1, C)
    half = 300.0  # heavy tails: q_init 1.8, d 2.5 -> <0.1% mass beyond 300 km
    area = (2 * half) ** 2

    n = 601
    g = torch.linspace(-half, half, n)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    lp = head.log_prob(
        pts, comp_xy.expand(len(pts), -1, -1), comp_feats.expand(len(pts), -1, -1),
        cond.expand(len(pts), -1), bg_area=area,
    )
    cell = (2 * half / (n - 1)) ** 2
    integral = lp.exp().sum() * cell
    assert abs(integral.item() - 1.0) < 0.03, integral


def test_kernel_mixture_rewards_proximity():
    """Density at a component center must beat far-away density."""
    torch.manual_seed(0)
    K, C = 4, 8
    head = KernelMixtureHead(cond_dim=C, n_comp=K)
    comp_xy = torch.zeros(2, K, 2)
    comp_feats = torch.zeros(2, K, 3)
    cond = torch.zeros(2, C)
    s = torch.tensor([[0.0, 0.0], [150.0, 150.0]])
    lp = head.log_prob(s, comp_xy, comp_feats, cond, bg_area=1e6)
    assert lp[0] > lp[1] + 5


def test_gr_head_normalizes_and_fits():
    torch.manual_seed(0)
    head = GRMagnitudeHead(cond_dim=4, beta_init=2.0)
    cond = torch.zeros(1, 4)
    m = torch.linspace(2.5, 12.0, 4001)
    lp = head.log_prob(m, cond.expand(len(m), -1), mc=2.5)
    integral = lp.exp().sum() * (m[1] - m[0])
    assert abs(integral.item() - 1.0) < 0.01, integral
    s = head.sample(cond.expand(10000, -1), mc=2.5)
    assert abs(s.mean().item() - (2.5 + 0.5)) < 0.05  # E[m] = mc + 1/beta
