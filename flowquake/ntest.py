"""Temporal N-test: daily forecast count distributions via simulation.

For each forecast day (offset from the test-window start), the model is
conditioned on the full observed catalog up to the day start, then S
independent continuations are sampled for a 1-day horizon. The observed
count is scored against the simulated count distribution with the CSEP
quantiles:
    delta_1 = P(N_sim >= N_obs)   (catastrophic underprediction if ~0)
    delta_2 = P(N_sim <= N_obs)   (overprediction if ~0)

Usage:
    python -m flowquake.ntest runs/comcat25/ckpt_best.pt --days 0 7 30 90 --n-sims 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import load_catalog
from .train import make_model

MAX_EVENTS_PER_DAY = 200
MAX_REJECTION_ROUNDS = 200


@torch.no_grad()
def simulate_daily_counts(
    model, cat, day_start_days: float, n_sims: int, device, sample_steps: int = 16
) -> np.ndarray:
    """Sample next-day event counts from the model.

    day_start_days: forecast day start, in catalog days (same clock as cat.t_days).
    """
    day_end = day_start_days + 1.0
    n_hist = int(np.searchsorted(cat.t_days, day_start_days, side="left"))
    tokens = cat.feats[:n_hist].unsqueeze(0).to(device)

    h_last, states = None, None
    seg = 16384
    for s in range(0, n_hist, seg):
        h, states = model.encoder.prefill(tokens[:, s : s + seg], states)
    h_last = h[:, -1]

    # Broadcast streaming state across simulation lanes.
    states = [(st.expand(n_sims, -1, -1, -1).contiguous(),
               cc.expand(n_sims, -1, -1).contiguous()) for st, cc in states]
    h_cur = h_last.expand(n_sims, -1).contiguous()
    t_last = torch.full((n_sims,), float(cat.t_days[n_hist - 1]), device=device)
    tok_last = tokens[0, -1].expand(n_sims, -1).contiguous()
    # Recent event times (most recent first) for recency features.
    max_lag = 64
    hist_times = cat.t_days[max(0, n_hist - max_lag): n_hist][::-1].copy()
    t_buf = torch.as_tensor(hist_times, device=device, dtype=torch.float64)
    if len(t_buf) < max_lag:
        t_buf = torch.cat([t_buf, t_buf[-1:].expand(max_lag - len(t_buf))])
    t_buf = t_buf.unsqueeze(0).expand(n_sims, -1).contiguous()

    counts = torch.zeros(n_sims, dtype=torch.long, device=device)
    active = torch.ones(n_sims, dtype=torch.bool, device=device)
    first = True

    for _ in range(MAX_EVENTS_PER_DAY):
        if not active.any():
            break
        tau, x, y, m, _ = model.sample_next(h_cur, tok_last, steps=sample_steps)
        if first:
            # The first continuation event must land after the day start:
            # rejection-sample the truncated conditional (we observed no
            # event between the last catalog event and the day start).
            need = active & (t_last + tau < day_start_days)
            for _r in range(MAX_REJECTION_ROUNDS):
                if not need.any():
                    break
                tau2, x2, y2, m2, _ = model.sample_next(h_cur, tok_last, steps=sample_steps)
                take = need & (t_last + tau2 >= day_start_days)
                tau = torch.where(take, tau2, tau)
                x = torch.where(take, x2, x)
                y = torch.where(take, y2, y)
                m = torch.where(take, m2, m)
                need = need & ~take
            active = active & ~need  # lanes that never accepted: no event today
            first = False

        t_next = t_last + tau
        live = active & (t_next < day_end)
        counts += live.long()
        active = live
        if not active.any():
            break

        # Append the sampled event and advance the encoder one step.
        m = torch.clamp(m, min=model.stats["mcut"])
        feat = model.build_token(tau, x, y, m, t_next.double(), t_buf).to(h_cur.dtype)
        h_new, new_states = model.encoder.step(feat, states)
        upd = active.view(-1, 1)
        h_cur = torch.where(upd, h_new, h_cur)
        tok_last = torch.where(upd, feat, tok_last)
        t_buf = torch.where(
            upd, torch.cat([t_next.double().unsqueeze(1), t_buf[:, :-1]], dim=1), t_buf
        )
        states = [
            (torch.where(active.view(-1, 1, 1, 1), s2, s1),
             torch.where(active.view(-1, 1, 1), c2, c1))
            for (s1, c1), (s2, c2) in zip(states, new_states)
        ]
        t_last = torch.where(active, t_next, t_last)

    return counts.cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--days", type=int, nargs="+", default=None,
                    help="forecast day offsets from test start")
    ap.add_argument("--n-days", type=int, default=20,
                    help="if --days not given: evenly spaced days across the test window")
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["cfg"]
    device = torch.device(args.device)

    cat = load_catalog(
        cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
        cfg.data.train_start, cfg.data.val_start, cfg.data.test_start,
        cfg.data.test_end,
    )
    model = make_model(cfg, ckpt["stats"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    model.stats.setdefault("mcut", cfg.data.mcut)

    test_start_days = float(
        (pd.Timestamp(cfg.data.test_start) - cat.times[0]).total_seconds() / 86400.0
    )
    n_test_days = int(
        (pd.Timestamp(cfg.data.test_end) - pd.Timestamp(cfg.data.test_start)).days
    )
    days = args.days or list(np.linspace(0, n_test_days - 1, args.n_days, dtype=int))

    results = []
    for d in days:
        start = test_start_days + d
        sims = simulate_daily_counts(model, cat, start, args.n_sims, device)
        n_obs = int(((cat.t_days >= start) & (cat.t_days < start + 1.0)).sum())
        delta1 = float((sims >= n_obs).mean())
        delta2 = float((sims <= n_obs).mean())
        rec = {
            "day": int(d), "n_obs": n_obs,
            "sim_mean": float(sims.mean()), "sim_median": float(np.median(sims)),
            "delta1": delta1, "delta2": delta2,
            "pass_95": bool(min(delta1, delta2) >= 0.025),
        }
        results.append(rec)
        print(json.dumps(rec))

    n_pass = sum(r["pass_95"] for r in results)
    summary = {
        "n_days": len(results), "n_pass_95": n_pass,
        "pass_rate": n_pass / len(results),
        "n_sims": args.n_sims, "results": results,
    }
    out = Path(args.ckpt).parent / "ntest.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nN-test: {n_pass}/{len(results)} days consistent at 95% | wrote {out}")


if __name__ == "__main__":
    main()
