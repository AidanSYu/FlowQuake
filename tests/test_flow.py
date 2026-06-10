import math

import torch

from flowquake.flow import CondFlow


def _train_flow(flow, sampler, cond_fn, steps=400, batch=512, lr=1e-2):
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    for _ in range(steps):
        u = sampler(batch)
        loss = flow.fm_loss(u, cond_fn(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return flow


def test_log_prob_matches_gaussian_1d():
    """Flow trained on N(mu, sigma) should give near-analytic log-density."""
    torch.manual_seed(0)
    mu, sigma = 1.5, 0.7
    flow = CondFlow(dim=1, cond_dim=1, hidden=64, n_layers=2)
    sampler = lambda b: mu + sigma * torch.randn(b, 1)
    cond_fn = lambda b: torch.zeros(b, 1)
    _train_flow(flow, sampler, cond_fn)

    u = torch.linspace(mu - sigma, mu + sigma, 9).unsqueeze(-1)
    lp = flow.log_prob(u, torch.zeros(9, 1), steps=64)
    analytic = -0.5 * (((u[:, 0] - mu) / sigma) ** 2) - math.log(sigma) \
        - 0.5 * math.log(2 * math.pi)
    assert (lp - analytic).abs().max() < 0.15, (lp, analytic)


def test_log_prob_matches_gaussian_2d_conditional():
    """2D conditional flow: mean depends on the condition."""
    torch.manual_seed(0)
    flow = CondFlow(dim=2, cond_dim=1, hidden=64, n_layers=2)

    def sampler_with_cond(b):
        c = torch.randn(b, 1)
        u = torch.cat([c + 0.3 * torch.randn(b, 1), -c + 0.3 * torch.randn(b, 1)], dim=-1)
        return u, c

    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    for _ in range(500):
        u, c = sampler_with_cond(512)
        loss = flow.fm_loss(u, c)
        opt.zero_grad()
        loss.backward()
        opt.step()

    c = torch.tensor([[0.8], [-0.5]])
    u = torch.cat([c, -c], dim=-1)  # at the conditional mean
    lp = flow.log_prob(u, c, steps=64)
    analytic = 2 * (-math.log(0.3) - 0.5 * math.log(2 * math.pi))
    assert (lp - analytic).abs().max() < 0.3, (lp, analytic)


def test_sample_moments():
    torch.manual_seed(0)
    mu, sigma = -0.8, 1.3
    flow = CondFlow(dim=1, cond_dim=1, hidden=64, n_layers=2)
    _train_flow(flow, lambda b: mu + sigma * torch.randn(b, 1),
                lambda b: torch.zeros(b, 1), steps=800)
    s = flow.sample(torch.zeros(20000, 1), steps=32)
    assert abs(s.mean().item() - mu) < 0.1
    assert abs(s.std().item() - sigma) < 0.1


def test_density_integrates_to_one():
    """Untrained (random-ish) flow must still be a normalized density."""
    torch.manual_seed(3)
    flow = CondFlow(dim=1, cond_dim=1, hidden=32, n_layers=2)
    # give the velocity net nonzero weights
    for p in flow.parameters():
        torch.nn.init.normal_(p, std=0.3)
    grid = torch.linspace(-8, 8, 801).unsqueeze(-1)
    lp = flow.log_prob(grid, torch.zeros(801, 1), steps=64)
    integral = torch.trapezoid(lp.exp(), grid[:, 0])
    assert abs(integral.item() - 1.0) < 0.02, integral
