"""M3 — the memorization ablation (the paper's mechanism figure).

For h_bottleneck in {0, 4, 16, 64}: train the canonical model, then measure
per-event LL on a held-in TRAIN subsample and on the TEST set. The thesis:
as the whole-catalog channel widens, TRAIN LL improves while TEST LL degrades
(the encoder state becomes an era-fingerprint the heads memorize) — so the
train/test gap grows with h. h=0 ("neural ETAS") generalizes; flexibility
backfires.

Run (GPU): python scripts/ablation_h.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")
from flowquake.config import Config
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.train import main as train_main, make_model

H_GRID = [0, 4, 16, 64]
SEED = 1553
SUB = 4096
ODE_STEPS = 64
OUT = Path("runs/ablation_h")
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def eval_split(model, cat, split, device, n_sub=None, seed=0):
    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, split)
    if n_sub:
        idx = mask[0].nonzero(as_tuple=True)[0]
        if len(idx) > n_sub:
            sel = np.random.default_rng(seed).choice(len(idx), n_sub, replace=False)
            keep = torch.zeros_like(mask[0]); keep[idx[torch.from_numpy(sel)]] = True
            mask = keep.unsqueeze(0)
    out = model.log_likelihood(tokens.to(device), target.to(device), mask.to(device),
                               lastk.to(device), raw_next.to(device), steps=ODE_STEPS)
    tll, sll, mll = (out[k].mean().item() for k in ("tll", "sll", "mll"))
    return {"tll": tll, "sll": sll, "mll": mll, "nll": -(tll + sll)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = yaml.safe_load(open("configs/comcat25.yaml"))
    rows = []
    for h in H_GRID:
        cfg = dict(base)
        cfg["model"] = {**base["model"], "h_bottleneck": h}
        cfg["train"] = {**base["train"], "seed": SEED, "out_dir": f"runs/ablation_h/h{h}"}
        cfg_path = OUT / f"h{h}.yaml"
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)

        print(f"\n===== h_bottleneck={h} =====", flush=True)
        train_main([str(cfg_path), "--out", cfg["train"]["out_dir"]])

        ckpt = torch.load(Path(cfg["train"]["out_dir"]) / "ckpt_best.pt",
                          map_location="cpu", weights_only=False)
        model = make_model(Config.load(cfg_path), ckpt["stats"]).to(device).eval()
        model.load_state_dict(ckpt["model"])
        cat = load_catalog(base["data"]["catalog_path"], base["data"]["mcut"],
                           base["data"]["aux_start"], base["data"]["train_start"],
                           base["data"]["val_start"], base["data"]["test_start"],
                           base["data"]["test_end"])
        tr = eval_split(model, cat, "train", device, n_sub=SUB)
        te = eval_split(model, cat, "test", device)
        row = {"h": h, "train": tr, "test": te,
               "gap_nll": te["nll"] - tr["nll"], "step": ckpt["step"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump(rows, open(OUT / "ablation_h.json", "w"), indent=2)

    print("\n=== memorization ablation ===")
    print(f"{'h':>4} {'train_nll':>10} {'test_nll':>10} {'gap':>8} {'best_step':>10}")
    for r in rows:
        print(f"{r['h']:>4} {r['train']['nll']:>10.4f} {r['test']['nll']:>10.4f} "
              f"{r['gap_nll']:>8.4f} {r['step']:>10}")


if __name__ == "__main__":
    main()
