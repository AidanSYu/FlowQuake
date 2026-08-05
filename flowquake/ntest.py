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
from .data import BIG_M, BIG_MAG_MIN, LAST_K, TAU_FLOOR_DAYS
from .train import load_catalog_cfg, make_model

MAX_EVENTS_PER_DAY = 200
MAX_REJECTION_ROUNDS = 200


@torch.no_grad()
def simulate_day_events(
    model, cat, day_start_days: float, n_sims: int, device, sample_steps: int = 16,
    horizon_days: float = 1.0, max_events: int | None = None,
) -> list[np.ndarray]:
    """Sample n_sims independent continuations over `horizon_days`, returning
    per-simulation event lists. Each entry is an (n_i, 4) array of columns
    [t_days, x_km, y_km, magnitude]. The catalog-based CSEP forecast is exactly
    this set of simulated catalogs; simulate_daily_counts() is the count-only
    wrapper used by the temporal N-test.

    `horizon_days` defaults to 1.0 (the CSEP daily convention, unchanged). Longer
    horizons are used by `flowquake.target_process` for the M>=M_TARGET
    sub-process likelihood and for P(M>=6 within 30 days). The per-lane event cap
    scales with the horizon unless `max_events` is given explicitly.

    NOTE on long horizons: the BIG_M long-lived trigger components are refreshed
    whenever a simulated event exceeds `data.BIG_MAG_MIN`, so a large event
    occurring inside the forecast window does start triggering. The trailing
    2-year window itself is held fixed, which is exact to within the horizon
    (30 days out of 730) and is stated in the Methods.
    """
    day_end = day_start_days + float(horizon_days)
    n_hist = int(np.searchsorted(cat.t_days, day_start_days, side="left"))
    tokens = cat.feats[:n_hist].unsqueeze(0).to(device)

    if model.encoder is not None:
        states = None
        seg = 16384
        for s in range(0, n_hist, seg):
            h, states = model.encoder.prefill(
                model._encoder_input(tokens[:, s : s + seg]), states)
        # Broadcast streaming state across simulation lanes.
        states = [(st.expand(n_sims, -1, -1, -1).contiguous(),
                   cc.expand(n_sims, -1, -1).contiguous()) for st, cc in states]
        h_cur = h[:, -1].expand(n_sims, -1).contiguous()
    else:
        states, h_cur = None, None
    # float64, NOT the default float32. This is an ABSOLUTE time: days since the
    # catalog epoch, so 3000-6600 in the test era, where float32's quantum is
    # 21-42 s -- 2400x coarser than the TAU_FLOOR_DAYS=1e-7 (~9 ms) floor
    # everything else clamps to. Two things broke. `t_next = t_last + tau`
    # silently discarded any tau below half a quantum, so an event drawn 10 s
    # after its predecessor advanced simulated time by exactly zero. And
    # `lastk_from_bufs` differences this against the float64 `t_buf`: on the
    # first step of a window both hold the SAME event, so the lag must be
    # identically zero, but rounding made it +-21 s at random -- and log(21 s)
    # vs log(9 ms) is a 7.8-log-unit swing in a feature normalized by
    # log_tau_std ~ 2.8, i.e. ~2.8 sigma of noise in the single most informative
    # short-term-triggering input, on the step that decides whether the window
    # produces any events at all. Rule: difference in the wide type, then narrow.
    t_last = torch.full((n_sims,), float(cat.t_days[n_hist - 1]), device=device,
                        dtype=torch.float64)
    tok_last = tokens[0, -1].expand(n_sims, -1).contiguous()
    # Recent-event history buffers (most recent first, raw units).
    max_lag = 64
    lo = max(0, n_hist - max_lag)

    def _mk_buf(arr, dtype):
        b = torch.as_tensor(arr[lo:n_hist][::-1].copy(), device=device, dtype=dtype)
        if len(b) < max_lag:
            b = torch.cat([b, b[-1:].expand(max_lag - len(b))])
        return b.unsqueeze(0).expand(n_sims, -1).contiguous()

    t_buf = _mk_buf(cat.t_days, torch.float64)
    x_buf = _mk_buf(cat.raw[:, 1].numpy(), torch.float32)
    y_buf = _mk_buf(cat.raw[:, 2].numpy(), torch.float32)
    m_buf = _mk_buf(cat.raw[:, 3].numpy(), torch.float32)

    active = torch.ones(n_sims, dtype=torch.bool, device=device)
    first = True
    rec_t, rec_x, rec_y, rec_m, rec_live = [], [], [], [], []  # per-step records

    # Long-lived trigger block: [BIG_M big triggers] + [optional near tier].
    # 1-day horizon (the CSEP convention): held static, exactly as before.
    #
    # This block is unchanged, but the bit-identity this comment used to claim
    # for committed N-test/CSEP artifacts NO LONGER HOLDS: widening `t_last` to
    # float64 above changes the sampled stream. Measured, paired same-seed over
    # 30 windows: total expected count −1.005% and mean spatial TV 0.105, against
    # a Monte-Carlo noise floor (same code, different seed) of −0.713% and 0.853.
    # So the artifacts move by less than their own sampling scatter — the
    # substance stands, the bytes do not. See MOONSHOT.md invariant 1o and the
    # provenance caveat on results/CLAIMS.md family 3.
    # Longer horizons: the block's elapsed times advance with simulated time and
    # large simulated events are inserted, so an M6 inside the window triggers.
    big_block = cat.lastk[n_hist - 1, LAST_K:]
    dynamic_big = horizon_days > 1.0
    if dynamic_big:
        t_ref = float(cat.t_days[n_hist - 1])
        _e = lambda a: a.to(device).unsqueeze(0).expand(n_sims, -1).contiguous()
        big_x, big_y = _e(big_block[:, 0]), _e(big_block[:, 1])
        big_m = _e(big_block[:, 3])
        # recover absolute trigger times from the frozen log elapsed time
        big_t = _e((t_ref - torch.exp(big_block[:, 2].double())))
        static_big = None
    else:
        static_big = big_block.unsqueeze(0).expand(n_sims, -1, -1).to(device).contiguous()

    cap = int(max_events) if max_events else int(MAX_EVENTS_PER_DAY * max(1.0, horizon_days))
    lane_rows = torch.arange(n_sims, device=device)

    for _ in range(cap):
        if not active.any():
            break
        bufs = (t_buf, x_buf, y_buf, m_buf)
        if dynamic_big:
            bdt = torch.log(
                torch.clamp(t_last.double().unsqueeze(1) - big_t, min=TAU_FLOOR_DAYS)
            ).to(big_x.dtype)
            static_big = torch.stack([big_x, big_y, bdt, big_m], dim=-1)
        lastk_lane = model.lastk_from_bufs(t_last.double(), bufs, static_big)
        tau, x, y, m = model.sample_next(h_cur, tok_last, lastk_lane, steps=sample_steps)
        if first:
            # The first continuation event must land after the day start:
            # rejection-sample the truncated conditional (we observed no
            # event between the last catalog event and the day start).
            need = active & (t_last + tau < day_start_days)
            for _r in range(MAX_REJECTION_ROUNDS):
                if not need.any():
                    break
                tau2, x2, y2, m2 = model.sample_next(h_cur, tok_last, lastk_lane, steps=sample_steps)
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
        m = torch.clamp(m, min=model.stats["mcut"])
        # Record this step's emitted events (lanes that are live).
        rec_t.append(t_next.detach().cpu().numpy().copy())
        rec_x.append(x.detach().cpu().numpy().copy())
        rec_y.append(y.detach().cpu().numpy().copy())
        rec_m.append(m.detach().cpu().numpy().copy())
        rec_live.append(live.detach().cpu().numpy().copy())
        active = live
        if not active.any():
            break

        # Append the sampled event and advance the encoder one step.
        feat = model.build_token(
            tau, x, y, m, t_next.double(), (t_buf, x_buf, y_buf, m_buf)
        ).to(tok_last.dtype)
        upd = active.view(-1, 1)
        if model.encoder is not None:
            h_new, new_states = model.encoder.step(
                model._encoder_input(feat.unsqueeze(1)).squeeze(1), states)
            h_cur = torch.where(upd, h_new, h_cur)
            states = [
                (torch.where(active.view(-1, 1, 1, 1), s2, s1),
                 torch.where(active.view(-1, 1, 1), c2, c1))
                for (s1, c1), (s2, c2) in zip(states, new_states)
            ]
        tok_last = torch.where(upd, feat, tok_last)

        def _push(buf, val):
            return torch.where(
                upd, torch.cat([val.unsqueeze(1).to(buf.dtype), buf[:, :-1]], dim=1), buf
            )

        t_buf = _push(t_buf, t_next.double())
        x_buf = _push(x_buf, x)
        y_buf = _push(y_buf, y)
        m_buf = _push(m_buf, m)
        t_last = torch.where(active, t_next, t_last)

        if dynamic_big:
            # A simulated event large enough to be a long-lived trigger evicts
            # the weakest entry of the BIG_M block (same "largest in window"
            # rule as data.big_trigger_matrix).
            sel = active & (m >= BIG_MAG_MIN)
            if sel.any():
                j = big_m[:, :BIG_M].argmin(dim=1)
                r, jj = lane_rows[sel], j[sel]
                big_x[r, jj] = x[sel]
                big_y[r, jj] = y[sel]
                big_t[r, jj] = t_next[sel].double()
                big_m[r, jj] = m[sel]

    # Assemble per-simulation event lists from the step records.
    events = [[] for _ in range(n_sims)]
    for st_t, st_x, st_y, st_m, st_live in zip(rec_t, rec_x, rec_y, rec_m, rec_live):
        idx = np.flatnonzero(st_live)
        for i in idx:
            events[i].append((st_t[i], st_x[i], st_y[i], st_m[i]))
    return [np.array(e, dtype=np.float64).reshape(-1, 4) for e in events]


@torch.no_grad()
def simulate_daily_counts(
    model, cat, day_start_days: float, n_sims: int, device, sample_steps: int = 16
) -> np.ndarray:
    """Next-day event counts per simulation (temporal N-test)."""
    events = simulate_day_events(model, cat, day_start_days, n_sims, device, sample_steps)
    return np.array([len(e) for e in events], dtype=np.int64)


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

    cat = load_catalog_cfg(cfg)
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


@torch.no_grad()
def simulate_windows(
    model, cat, starts, n_sims: int, device, sample_steps: int = 16,
    horizon_days: float = 30.0, max_events: int | None = None,
    compact_every: int = 64, compact_below: float = 0.5,
    max_lanes: int = 16384,
):
    """Simulate MANY forecast windows in one batched autoregressive pass.

    `simulate_day_events` handles one window at a time, which is what the CSEP
    daily protocol needs. The scaling curve instead scores ~52 windows per
    model, and doing them sequentially wastes almost all the available
    parallelism: the inner loop is a Python `for` over sampling steps issuing
    tiny tensor ops, so the cost is dominated by per-step dispatch, not by lane
    count. Measured on the synthetic validation, the mc 1.0 point (148k events,
    ~745 simulated events per 90-day window) was by far the slowest job in the
    sweep for exactly this reason.

    Every lane is independent, so windows can simply share the lane dimension:
    (n_windows * n_sims) lanes advance together and the Python loop runs ONCE
    instead of n_windows times. Each lane carries its own history buffers, its
    own start time and its own horizon end, so the result is identical in
    distribution to looping — only the scheduling changes.

    Returns a list of lists: `out[w][s]` is the (n_i, 4) array for window w,
    simulation s, matching `simulate_day_events`'s per-window contract.
    """
    starts = [float(s) for s in starts]
    # Hard lane budget. Peak memory scales with (windows x sims), and with
    # matched-precision n_sims that product reaches ~66k on a real sweep, which
    # drove a 48 GB machine into 34 GB of swap. Split the window list so the
    # working set stays bounded; the results are identical because lanes are
    # independent.
    if len(starts) * n_sims > max_lanes:
        per = max_lanes // max(n_sims, 1)
        if per >= 1:
            # split across WINDOWS
            out = []
            for i in range(0, len(starts), per):
                out.extend(simulate_windows(
                    model, cat, starts[i:i + per], n_sims, device,
                    sample_steps=sample_steps, horizon_days=horizon_days,
                    max_events=max_events, compact_every=compact_every,
                    compact_below=compact_below, max_lanes=max_lanes))
            return out
        # A SINGLE window already exceeds the budget, which happens once
        # matched-resolution n_sims gets large on a sparse catalog: at a 1-day
        # horizon and mc 2.5, T = 20,000 needs ~98,500 simulations for one
        # window. Splitting by window cannot help there -- the old code fell
        # back to `per = 1` and allocated all of them anyway, reintroducing
        # exactly the unbounded working set that put a 48 GB machine into 34 GB
        # of swap. Split the SIMULATIONS instead and concatenate; lanes are
        # independent, so the result is identical.
        out = [[] for _ in starts]
        done = 0
        while done < n_sims:
            take = min(max_lanes, n_sims - done)
            part = simulate_windows(
                model, cat, starts, take, device,
                sample_steps=sample_steps, horizon_days=horizon_days,
                max_events=max_events, compact_every=compact_every,
                compact_below=compact_below, max_lanes=max_lanes)
            for w in range(len(starts)):
                out[w].extend(part[w])
            done += take
        return out

    W, S = len(starts), n_sims
    L = W * S
    n_hist = [int(np.searchsorted(cat.t_days, s, side="left")) for s in starts]
    max_lag = LAST_K

    def lane_buf(arr, dtype):
        """(L, max_lag) most-recent-first history per lane."""
        rows = []
        for nh in n_hist:
            lo = max(0, nh - max_lag)
            b = np.asarray(arr[lo:nh][::-1], dtype=np.float64).copy()
            if len(b) < max_lag:
                b = np.concatenate([b, np.full(max_lag - len(b), b[-1] if len(b) else 0.0)])
            rows.append(b)
        t = torch.as_tensor(np.stack(rows), device=device, dtype=dtype)
        return t.repeat_interleave(S, dim=0).contiguous()

    t_buf = lane_buf(cat.t_days, torch.float64)
    x_buf = lane_buf(cat.raw[:, 1].numpy(), torch.float32)
    y_buf = lane_buf(cat.raw[:, 2].numpy(), torch.float32)
    m_buf = lane_buf(cat.raw[:, 3].numpy(), torch.float32)

    start_t = torch.tensor(starts, device=device, dtype=torch.float64).repeat_interleave(S)
    end_t = start_t + float(horizon_days)
    # float64 for the same reason as the single-window simulator above: this is
    # an absolute day number, and every quantity it is differenced against
    # (t_buf, start_t, end_t, t_ref) is already float64. It was the lone
    # narrow one.
    t_last = torch.tensor([float(cat.t_days[nh - 1]) for nh in n_hist],
                          device=device, dtype=torch.float64).repeat_interleave(S)
    tok_last = cat.feats[[nh - 1 for nh in n_hist]].to(device).repeat_interleave(S, dim=0)

    big = torch.stack([cat.lastk[nh - 1, LAST_K:] for nh in n_hist]).to(device)
    big = big.repeat_interleave(S, dim=0).contiguous()
    big_x, big_y, big_m = big[..., 0].clone(), big[..., 1].clone(), big[..., 3].clone()
    t_ref = torch.tensor([float(cat.t_days[nh - 1]) for nh in n_hist],
                         device=device, dtype=torch.float64).repeat_interleave(S)
    big_t = (t_ref.unsqueeze(1) - torch.exp(big[..., 2].double())).contiguous()

    if model.encoder is not None:
        raise NotImplementedError(
            "batched windows do not carry per-lane SSM state; every reported "
            "config runs h_bottleneck=0 so the encoder is absent. Use "
            "simulate_day_events for h>0.")
    h_cur = None

    active = torch.ones(L, dtype=torch.bool, device=device)
    first = True
    rec_t, rec_x, rec_y, rec_m, rec_id = [], [], [], [], []
    cap = int(max_events) if max_events else int(MAX_EVENTS_PER_DAY * max(1.0, horizon_days))
    # `lane_id` maps rows of the (possibly compacted) working tensors back to
    # their original lane, so records stay attributable after compaction.
    lane_id = torch.arange(L, device=device)
    rows_idx = torch.arange(L, device=device)

    for _step in range(cap):
        if not active.any():
            break
        # Lane compaction. Event counts per lane are heavy-tailed (Omori), so a
        # handful of lanes keep the loop alive long after most have run past the
        # horizon, and without compaction every iteration still pays for the full
        # L rows. Dropping finished lanes makes the tail O(surviving).
        #
        # Measured, honestly: 1.13x at mc 1.0 (148k events, ~789 events/sim —
        # the case that actually hurts) and a slight REGRESSION at mc 2.5, where
        # lanes finish together and the gather is not repaid. A modest win
        # confined to the heavy-tail regime, not a general speedup. Set
        # compact_every=0 to disable.
        if compact_every and _step and _step % compact_every == 0:
            frac = float(active.float().mean())
            if frac < compact_below and int(active.sum()) > 0:
                keep = active.nonzero(as_tuple=True)[0]
                lane_id = lane_id[keep]
                t_buf, x_buf = t_buf[keep], x_buf[keep]
                y_buf, m_buf = y_buf[keep], m_buf[keep]
                big_x, big_y = big_x[keep], big_y[keep]
                big_m, big_t = big_m[keep], big_t[keep]
                start_t, end_t = start_t[keep], end_t[keep]
                t_last, tok_last = t_last[keep], tok_last[keep]
                active = active[keep]
                rows_idx = torch.arange(len(keep), device=device)
        bdt = torch.log(torch.clamp(t_last.double().unsqueeze(1) - big_t,
                                    min=TAU_FLOOR_DAYS)).to(big_x.dtype)
        static_big = torch.stack([big_x, big_y, bdt, big_m], dim=-1)
        lastk_lane = model.lastk_from_bufs(t_last.double(), (t_buf, x_buf, y_buf, m_buf),
                                           static_big)
        tau, x, y, m = model.sample_next(h_cur, tok_last, lastk_lane, steps=sample_steps)
        if first:
            need = active & (t_last.double() + tau.double() < start_t)
            for _r in range(MAX_REJECTION_ROUNDS):
                if not need.any():
                    break
                tau2, x2, y2, m2 = model.sample_next(h_cur, tok_last, lastk_lane,
                                                     steps=sample_steps)
                take = need & (t_last.double() + tau2.double() >= start_t)
                tau = torch.where(take, tau2, tau); x = torch.where(take, x2, x)
                y = torch.where(take, y2, y); m = torch.where(take, m2, m)
                need = need & ~take
            active = active & ~need
            first = False

        t_next = t_last + tau
        live = active & (t_next.double() < end_t)
        m = torch.clamp(m, min=model.stats["mcut"])
        # Record ONLY the live entries. Appending the full L-wide vector every
        # step is what made this explode: with matched-precision n_sims the lane
        # count reaches ~66k (52 windows x 1277 sims) and the cap is 18,000
        # steps, so full-width records are tens of GB and drove the machine into
        # 34 GB of swap. Storing just the survivors bounds this by the number of
        # events actually simulated, which is what gets returned anyway.
        liv = live.nonzero(as_tuple=True)[0]
        if len(liv):
            rec_id.append(lane_id[liv].cpu().numpy())
            rec_t.append(t_next[liv].detach().cpu().numpy())
            rec_x.append(x[liv].detach().cpu().numpy())
            rec_y.append(y[liv].detach().cpu().numpy())
            rec_m.append(m[liv].detach().cpu().numpy())
        active = live
        if not active.any():
            break

        feat = model.build_token(tau, x, y, m, t_next.double(),
                                 (t_buf, x_buf, y_buf, m_buf)).to(tok_last.dtype)
        upd = active.view(-1, 1)
        tok_last = torch.where(upd, feat, tok_last)

        def _push(buf, val):
            return torch.where(upd, torch.cat(
                [val.unsqueeze(1).to(buf.dtype), buf[:, :-1]], dim=1), buf)

        t_buf = _push(t_buf, t_next.double()); x_buf = _push(x_buf, x)
        y_buf = _push(y_buf, y); m_buf = _push(m_buf, m)
        t_last = torch.where(active, t_next, t_last)

        sel = active & (m >= BIG_MAG_MIN)
        if sel.any():
            j = big_m.argmin(dim=1)
            r, jj = rows_idx[sel], j[sel]
            big_x[r, jj] = x[sel]; big_y[r, jj] = y[sel]
            big_t[r, jj] = t_next[sel].double(); big_m[r, jj] = m[sel]

    out = [[[] for _ in range(S)] for _ in range(W)]
    for st_t, st_x, st_y, st_m, st_id in zip(rec_t, rec_x, rec_y, rec_m, rec_id):
        for r in range(len(st_id)):
            i = int(st_id[r])          # original lane, survives compaction
            out[i // S][i % S].append((st_t[r], st_x[r], st_y[r], st_m[r]))
    return [[np.array(e, dtype=np.float64).reshape(-1, 4) for e in win] for win in out]
