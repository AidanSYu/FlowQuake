"""Re-evaluate the memorization ablation checkpoints (no training) — §4.3.

For each h in {0,4,16,64}, load BOTH checkpoints saved by scripts/ablation_h.py:
  - ckpt_best : the early-stopped (best held-out) checkpoint
  - ckpt_last : the converged / over-trained checkpoint
and measure per-event log-likelihood on a held-in TRAIN subsample and on the
full TEST set, using the same metric as the paper. Writes
runs/ablation_h/memorization_figure.json so §4.3 is reproducible from committed
artifacts (ablation_h.py only saves the ckpt_best summary).

CPU by default so it never contends with a GPU job already running on the
shared card. Run: python scripts/memorization_eval.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")
from flowquake.config import Config
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.train import make_model

H_GRID = [0, 4, 16, 64]
SUB = 4096
ODE_STEPS = 64
OUT = Path("runs/ablation_h")


@torch.no_grad()
def eval_split(model, cat, split, device, n_sub=None, seed=0):
    """Per-event mean tll/sll/mll/nll on a split (optional train subsample)."""
    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, split)
    if n_sub:
        idx = mask[0].nonzero(as_tuple=True)[0]
        if len(idx) > n_sub:
            sel = np.random.default_rng(seed).choice(len(idx), n_sub, replace=False)
            keep = torch.zeros_like(mask[0])
            keep[idx[torch.from_numpy(sel)]] = True
            mask = keep.unsqueeze(0)
    out = model.log_likelihood(tokens.to(device), target.to(device), mask.to(device),
                               lastk.to(device), raw_next.to(device), steps=ODE_STEPS)
    tll, sll, mll = (out[k].mean().item() for k in ("tll", "sll", "mll"))
    return {"tll": tll, "sll": sll, "mll": mll, "nll": -(tll + sll)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; safe alongside a GPU job) or cuda")
    args = ap.parse_args()
    device = args.device

    base = yaml.safe_load(open("configs/comcat25.yaml"))
    cat = load_catalog(base["data"]["catalog_path"], base["data"]["mcut"],
                       base["data"]["aux_start"], base["data"]["train_start"],
                       base["data"]["val_start"], base["data"]["test_start"],
                       base["data"]["test_end"])

    rows = []
    for h in H_GRID:
        cfg_path = OUT / f"h{h}.yaml"
        for which in ("best", "last"):
            ckpt_path = OUT / f"h{h}" / f"ckpt_{which}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = make_model(Config.load(cfg_path), ckpt["stats"]).to(device).eval()
            model.load_state_dict(ckpt["model"])
            tr = eval_split(model, cat, "train", device, n_sub=SUB)
            te = eval_split(model, cat, "test", device)
            row = {"h": h, "ckpt": which, "step": int(ckpt.get("step", -1)),
                   "train": tr, "test": te, "gap_nll": te["nll"] - tr["nll"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
            json.dump(rows, open(OUT / "memorization_figure.json", "w"), indent=2)

    print(f"\n{'h':>3} {'ckpt':>5} {'step':>6} {'train_nll':>10} {'test_nll':>10} {'gap':>8}")
    for r in rows:
        print(f"{r['h']:>3} {r['ckpt']:>5} {r['step']:>6} {r['train']['nll']:>10.3f} "
              f"{r['test']['nll']:>10.3f} {r['gap_nll']:>8.3f}")


if __name__ == "__main__":
    main()
