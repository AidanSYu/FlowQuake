"""Probe: is eval-time conditioning OOD vs training-time conditioning?

Compares per-event LL at the SAME masked positions under:
  A) full-catalog streaming h (the eval path: encode_full, clean tokens)
  B) training-style h: 2048-window forward, zero initial state, clean tokens
  C) full-catalog h with training-style input noise (0.1) on encoder input
  D) untrained (fresh-init) model, full path  -> the "do-nothing" baseline

Run: python scripts/probe_h_mismatch.py runs/comcat25/ckpt_last.pt
"""

import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from flowquake.config import Config
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.train import make_model

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/comcat25/ckpt_last.pt"
device = "cuda"
N_SUB = 256
W = 2048
STEPS = 32

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg: Config = ckpt["cfg"]
cat = load_catalog(cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
                   cfg.data.train_start, cfg.data.val_start, cfg.data.test_start,
                   cfg.data.test_end)
model = make_model(cfg, ckpt["stats"]).to(device).eval()
model.load_state_dict(ckpt["model"])
print(f"ckpt step {ckpt['step']}  (input_noise={model.input_noise})")

torch.manual_seed(0)
fresh = make_model(cfg, cat.stats).to(device).eval()  # untrained reference

st = model.stats
feats = cat.feats.to(device)


def head_ll(model, h, tok, tgt, steps=STEPS):
    """Replicates model.log_likelihood given precomputed h rows."""
    cond = torch.cat([h, tok], dim=-1)
    u_t, u_m = tgt[:, 0:1], tgt[:, 3:4]
    u_s = tgt[:, 1:3] - tok[:, 1:3]
    lp_t = model.head_t.log_prob(u_t, cond, steps=steps)
    lp_s = model.head_s.log_prob(u_s, torch.cat([cond, u_t], dim=-1), steps=steps)
    log_tau = tgt[:, 0] * st["log_tau_std"] + st["log_tau_mean"]
    tll = lp_t - math.log(st["log_tau_std"]) - log_tau
    sll = lp_s - math.log(st["x_std"] * st["y_std"])
    return tll, sll


rng = np.random.default_rng(0)
for split in ["train", "val"]:
    tokens, target, mask = full_sequence_batch(cat, split)
    tokens, target = tokens.to(device), target.to(device)
    idx = mask[0].nonzero(as_tuple=True)[0]
    idx = idx[idx >= W]  # need a full window of history
    sel = torch.from_numpy(rng.choice(idx.numpy(), N_SUB, replace=False)).long()
    sel, _ = torch.sort(sel)
    tok = feats[sel]
    tgt = target[0, sel]

    with torch.no_grad():
        # A) full-catalog streaming h
        h_full_all = model.encode_full(tokens)
        h_A = h_full_all[0, sel]
        # B) training-style windowed h (zero init state, window 2048)
        wins = torch.stack([feats[i - W + 1: i + 1] for i in sel.tolist()])
        h_B = []
        for c in range(0, N_SUB, 32):
            h_B.append(model.encoder(wins[c:c + 32])[:, -1])
        h_B = torch.cat(h_B)
        # C) full-catalog h with train-style input noise
        noisy = tokens + 0.1 * torch.randn_like(tokens)
        h_C = model.encode_full(noisy)[0, sel]
        # D) untrained model, full path
        h_D = fresh.encode_full(tokens)[0, sel]

        tllA, sllA = head_ll(model, h_A, tok, tgt)
        tllB, sllB = head_ll(model, h_B, tok, tgt)
        tllC, sllC = head_ll(model, h_C, tok, tgt)
        tllD, sllD = head_ll(fresh, h_D, tok, tgt)

    dh = (h_A - h_B).abs().mean().item()
    print(f"\n[{split}] n={N_SUB}  |h_A|={h_A.abs().mean():.3f} "
          f"|h_B|={h_B.abs().mean():.3f}  mean|h_A-h_B|={dh:.4f}")
    for name, tll, sll in [("A full-clean ", tllA, sllA),
                           ("B window2048 ", tllB, sllB),
                           ("C full-noised", tllC, sllC),
                           ("D untrained  ", tllD, sllD)]:
        print(f"  {name}: tll {tll.mean():8.3f} (med {tll.median():8.3f})   "
              f"sll {sll.mean():8.3f} (med {sll.median():8.3f})")
