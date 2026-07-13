"""Does the ETAS-beating spatial head TRANSFER across tectonic regimes?

The head's learned triggering modulation g(m_j, dt) is translation-invariant
(depends only on parent magnitude and elapsed time, never on location), so in
principle a head trained in one region should improve ETAS in another. We test
it: take a head trained on a SOURCE region and apply its learned weights to a
TARGET region's precomputed features (which encode the TARGET's own ETAS
inversion frozen sums and the TARGET's causal KDE background). Because the head's
buffers (mu, k0, ..., area) are persistent=False, load_state_dict transfers only
the learned tensors (kde_gate, kde_logits, log_mu_adj, log_alpha, mlp), leaving
the target's ETAS-init buffers intact.

Two modes per (source -> target):
  * zero-shot : apply ALL source learned weights to the target.
  * few-shot  : transfer the source MLP (the genuinely neural, translation-
                invariant part) FROZEN, and re-fit only the 4 background/scale
                scalars on the target's val split (the spatial analogue of the
                temporal story's per-region normalization).

dS is the paired per-event spatial gain over the TARGET's region-fitted ETAS on
the target TEST split, with a stationary block-bootstrap CI.

Run: python scripts/transfer_neural_etas.py --source ComCat_25 \
        --targets Italy_25 Japan_25 Chile_25 Greece_25 Iran_25
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.neural_etas import NeuralETASSpatialHead
from flowquake.stats import paired_gain_summary
from scripts.train_neural_etas import load, mean_ll, batch

SPLITS = {"ComCat_25": ("1998-01-01", "2007-01-01"),
          "Italy_25": ("2009-01-01", "2011-01-01"),
          "Japan_25": ("2009-01-01", "2011-01-01"),
          "Chile_25": ("2009-01-01", "2011-01-01"),
          "Greece_25": ("2009-01-01", "2011-01-01"),
          "Iran_25": ("2009-01-01", "2011-01-01")}


def load_source_state(src_ckpt):
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    return ck["state"], ck["n_kde"], ck["use_mlp"]


def build_target_head(tgt_feat, n_kde, use_mlp, device="cpu"):
    params, data, times, etas_sll, tgt_mag, n, _ = load(tgt_feat, device)
    model = NeuralETASSpatialHead(params, n_kde=n_kde, use_mlp=use_mlp)
    return model, params, data, times, etas_sll, tgt_mag, n


def fewshot_recalibrate(model, data, va_idx, epochs=80, lr=1e-2):
    """Re-fit only the 4 background/scale scalars on target val; MLP frozen."""
    for p in model.mlp.parameters():
        p.requires_grad_(False)
    if model.g_off is not None:
        model.g_off.requires_grad_(False)
    opt = torch.optim.Adam([model.kde_gate, model.kde_logits,
                            model.log_mu_adj, model.log_alpha], lr=lr)
    best, best_state = -1e18, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        ll = model(**batch(data, va_idx)).mean()
        (-ll).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = model(**batch(data, va_idx)).mean().item()
        if v > best + 1e-6:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def score(model, data, te_idx, etas_sll, seed):
    model.eval()
    sll = mean_ll(model, data, te_idx).numpy()
    et = etas_sll[te_idx.numpy()]
    g = paired_gain_summary(sll - et, seed=seed).asdict()
    return {"n": int(len(te_idx)), "sll_head": float(sll.mean()), "sll_etas": float(et.mean()),
            "dS": round(g["mean"], 4), "ci": [round(g["ci"][0], 4), round(g["ci"][1], 4)],
            "decision": g["decision"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ComCat_25")
    ap.add_argument("--source-seed", type=int, default=0)
    ap.add_argument("--targets", nargs="+",
                    default=["Italy_25", "Japan_25", "Chile_25", "Greece_25", "Iran_25"])
    args = ap.parse_args()

    src_ckpt = f"runs/neural_etas/{args.source}/head_full_s{args.source_seed}.pt"
    state, n_kde, use_mlp = load_source_state(src_ckpt)
    print(f"source head: {src_ckpt} (n_kde={n_kde}, mlp={use_mlp})\n")

    results = {"source": args.source, "targets": {}}
    for i, tgt in enumerate(args.targets):
        val_start, test_start = SPLITS[tgt]
        model, params, data, times, etas_sll, tgt_mag, n = build_target_head(tgt, n_kde, use_mlp)
        te = torch.as_tensor(np.flatnonzero(times >= test_start))
        va = torch.as_tensor(np.flatnonzero((times >= val_start) & (times < test_start)))

        # in-region ceiling (target's own trained head) for reference
        own = None
        own_ckpt = f"runs/neural_etas/{tgt}/head_full_s0.pt"
        if Path(own_ckpt).exists():
            own = json.load(open(f"runs/neural_etas/{tgt}/summary_full_s0.json")).get("dS_mean")

        # zero-shot: load all source learned weights onto target-init head
        model.load_state_dict(state, strict=False)
        zs = score(model, data, te, etas_sll, seed=700 + i)

        # few-shot: reload source weights, recalibrate the 4 scalars on target val
        model.load_state_dict(state, strict=False)
        model = fewshot_recalibrate(model, data, va)
        fs = score(model, data, te, etas_sll, seed=800 + i)

        results["targets"][tgt] = {"own_native_dS": own, "zero_shot": zs, "few_shot": fs}
        print(f"{tgt:10} own_dS {own}  |  zero-shot dS {zs['dS']:+.4f} {zs['ci']} {zs['decision']}"
              f"  |  few-shot dS {fs['dS']:+.4f} {fs['ci']} {fs['decision']}")

    Path("runs/neural_etas").mkdir(parents=True, exist_ok=True)
    out = f"runs/neural_etas/transfer_from_{args.source}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
