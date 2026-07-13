"""Train the neural-ETAS spatial head on precomputed full-history features.

Splits follow the benchmark: targets before val_start train, [val_start,
test_start) validate (early stopping), >= test_start scored at the end against
the package's per-event ETAS SLL (paired stationary-block-bootstrap). The
reported grids include ablations and multiple seeds; do not describe these runs
as a test-scored-once protocol.

Two configurations, reported separately (the ablation):
  --no-mlp : learnable background mixture + scalars only
  (default): + per-parent near-set modulations

Run (CPU ok): python scripts/train_neural_etas.py ComCat_25
              python scripts/train_neural_etas.py ComCat_25 --no-mlp
"""
import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.neural_etas import NeuralETASSpatialHead
from flowquake.stats import paired_gain_summary

PARAM_NAMES = ["mu", "k0", "c", "tau", "d", "a", "omega", "gamma", "rho", "mc", "area"]


def load(name: str, device: str):
    z = np.load(f"runs/{name}_trigfeat.npz", allow_pickle=True)
    params = dict(zip(PARAM_NAMES, z["params"]))
    tgt = z["targets"]
    mag_all = z["mag"]
    near_idx = z["near_idx"]
    valid = (near_idx >= 0).astype(np.float32)
    near_m = mag_all[np.maximum(near_idx, 0)] * (near_idx >= 0) + params["mc"] * (near_idx < 0)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
    data = {
        "far_num": t(z["far_num"]), "far_den": t(z["far_den"]), "kde": t(z["kde"]),
        "r2": t(z["near_r2"]), "dt": t(np.maximum(z["near_dt"], 0.0)),
        "mag": t(near_m), "valid": t(valid),
    }
    times = pd.to_datetime(z["time_iso"])[tgt]
    if "event_index" in z:
        event_index = np.asarray(z["event_index"])[tgt]
    else:
        event_index = tgt
    etas_sll = np.log(params["mu"] + z["trig_num"]) - np.log(params["mu"] * params["area"] + z["trig_den"])
    return params, data, times, etas_sll, mag_all[tgt], len(tgt), event_index


def batch(data, idx):
    return {k: v[idx] for k, v in data.items()}


def mean_ll(model, data, idx, bs=8192):
    out = []
    with torch.no_grad():
        for i0 in range(0, len(idx), bs):
            out.append(model(**batch(data, idx[i0:i0 + bs])))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--refit-globals", action="store_true",
                    help="classical control: SGD-refit global kernel params + learned bg, no MLP")
    ap.add_argument("--val-start", default="1998-01-01")
    ap.add_argument("--test-start", default="2007-01-01")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    params, data, times, etas_sll, tgt_mag, n, event_index = load(args.name, args.device)
    tr = torch.as_tensor(np.flatnonzero(times < args.val_start), device=args.device)
    va = torch.as_tensor(np.flatnonzero((times >= args.val_start) & (times < args.test_start)), device=args.device)
    te = torch.as_tensor(np.flatnonzero(times >= args.test_start), device=args.device)
    print(f"{args.name}: train {len(tr)}  val {len(va)}  test {len(te)}  (mlp={'off' if args.no_mlp else 'on'})")

    use_mlp = not (args.no_mlp or args.refit_globals)
    model = NeuralETASSpatialHead(params, n_kde=data["kde"].shape[1], use_mlp=use_mlp,
                                  refit_globals=args.refit_globals).to(args.device)

    exact_model = NeuralETASSpatialHead(
        params, n_kde=data["kde"].shape[1], use_mlp=False, kde_gate_init=-30.0
    ).to(args.device)
    exact_test = mean_ll(exact_model, data, te)
    te_np = te.cpu().numpy()
    exact_err = (exact_test.cpu().numpy() - etas_sll[te_np])
    print(f"gate-closed ETAS reproduction max_abs_err {abs(exact_err).max():.3e}")
    assert abs(exact_err).max() < 2e-5, "gate-closed head does not reproduce ETAS"

    init_test = mean_ll(model, data, te).mean().item()
    etas_test = float(etas_sll[te.cpu().numpy()].mean())
    print(f"init test sll {init_test:.4f}   ETAS {etas_test:.4f}   (near-match; 5% KDE gate at init)")
    assert abs(init_test - etas_test) < 0.05, "init too far from ETAS"

    groups = [
        {"params": [model.kde_gate, model.kde_logits, model.log_mu_adj, model.log_alpha], "lr": 1e-2},
        {"params": model.mlp.parameters(), "lr": 2e-3, "weight_decay": 1e-5},
    ]
    if model.g_off is not None:
        groups.append({"params": [model.g_off], "lr": 5e-3})
    opt = torch.optim.Adam(groups)
    best_val, best_state, bad = -1e18, None, 0
    for ep in range(args.epochs):
        perm = tr[torch.randperm(len(tr), device=args.device)]
        tot = 0.0
        model.train()
        for i0 in range(0, len(perm), args.bs):
            idx = perm[i0:i0 + args.bs]
            ll = model(**batch(data, idx))
            loss = -ll.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += -loss.item() * len(idx)
        model.eval()
        val = mean_ll(model, data, va).mean().item()
        marker = ""
        if val > best_val + 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            bad += 1
        print(f"ep {ep:3d}  train {tot / len(tr):+.4f}  val {val:+.4f}{marker}", flush=True)
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    tag0 = "refit_globals" if args.refit_globals else ("bg_only" if args.no_mlp else "full")
    ckpt_dir = Path(f"runs/neural_etas/{args.name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "params": params,
                "n_kde": data["kde"].shape[1], "use_mlp": use_mlp,
                "refit_globals": bool(args.refit_globals)},
               ckpt_dir / f"head_{tag0}_s{args.seed}.pt")
    sll_neural = mean_ll(model, data, te).cpu().numpy()
    sll_etas = etas_sll[te_np]
    g = paired_gain_summary(sll_neural - sll_etas, seed=args.seed + 1).asdict()
    tag = tag0 + f"_s{args.seed}"
    out_dir = Path(f"runs/neural_etas/{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index_from_zero": event_index[te_np],
                  "time": times[te_np], "magnitude": tgt_mag[te_np],
                  "sll_neural": sll_neural, "sll_etas": sll_etas}).to_csv(
        out_dir / f"per_event_{tag}.csv", index=False)
    gate = float(torch.sigmoid(model.kde_gate))
    kde_w = (gate * torch.softmax(model.kde_logits, 0)).detach().cpu().numpy().round(5).tolist()
    bg_w = [round(1.0 - gate, 5)] + kde_w
    summary = {
        "test_sll_neural": float(sll_neural.mean()), "test_sll_etas": float(sll_etas.mean()),
        "dS_mean": round(g["mean"], 4), "dS_ci": [round(g["ci"][0], 4), round(g["ci"][1], 4)],
        "decision": g["decision"], "n_test": int(len(te)),
        "val_sll": best_val, "bg_weights[unif,kde...]": bg_w,
        "alpha_far": float(torch.exp(model.log_alpha)), "mu_adj": float(torch.exp(model.log_mu_adj)),
        "config": tag, "seed": args.seed,
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
