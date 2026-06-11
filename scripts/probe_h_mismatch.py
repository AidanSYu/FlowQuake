"""Probe: is eval-time conditioning OOD vs training-time conditioning?

Per-event LL at the SAME masked positions under:
  A) full-catalog streaming h (the eval path: encode_full, clean tokens)
  B) training-style h: 2048-window forward, zero initial state, clean tokens
  C) full-catalog h with training-style input noise (0.1) on encoder input
  D) untrained (fresh-init) model, full path  -> the "do-nothing" baseline

If B >> A on val, eval conditioning is OOD vs training (state mismatch).
If A ~ B both flat vs D, the heads aren't using h at all.

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
N_SUB = 512
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
fresh = make_model(cfg, cat.stats).to(device).eval()

st = model.stats
feats = cat.feats.to(device)


def head_ll(mdl, h, tok, tgt, lk, rn, steps=STEPS):
    cond = torch.cat([h, tok], dim=-1)
    lp_t = mdl.head_t.log_prob(tgt[:, 0:1], cond, steps=steps)
    log_tau = tgt[:, 0] * st["log_tau_std"] + st["log_tau_mean"]
    tll = lp_t - math.log(st["log_tau_std"]) - log_tau
    comp_xy, comp_feats = mdl._comp_inputs(lk)
    sll = mdl.head_s.log_prob(rn[:, :2], comp_xy, comp_feats, cond, st["bg_area"])
    mll = mdl.head_m.log_prob(rn[:, 2], cond, st["mcut"])
    return tll, sll, mll


rng = np.random.default_rng(0)
for split in ["train", "val"]:
    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, split)
    tokens, target = tokens.to(device), target.to(device)
    idx = mask[0].nonzero(as_tuple=True)[0]
    idx = idx[idx >= W]
    sel = torch.from_numpy(rng.choice(idx.numpy(), N_SUB, replace=False)).long()
    sel, _ = torch.sort(sel)
    tok = feats[sel]
    tgt = target[0, sel]
    lk = lastk[0, sel].to(device)
    rn = raw_next[0, sel].to(device)

    with torch.no_grad():
        h_A = model.encode_full(tokens)[0, sel]
        wins = torch.stack([feats[i - W + 1: i + 1] for i in sel.tolist()])
        h_B = torch.cat([model.encoder(wins[c:c + 32])[:, -1]
                         for c in range(0, N_SUB, 32)])
        h_C = model.encode_full(tokens + 0.1 * torch.randn_like(tokens))[0, sel]
        h_D = fresh.encode_full(tokens)[0, sel]

        rows = [("A full-clean ", *head_ll(model, h_A, tok, tgt, lk, rn)),
                ("B window2048 ", *head_ll(model, h_B, tok, tgt, lk, rn)),
                ("C full-noised", *head_ll(model, h_C, tok, tgt, lk, rn)),
                ("D untrained  ", *head_ll(fresh, h_D, tok, tgt, lk, rn))]

    dh = (h_A - h_B).abs().mean().item()
    print(f"\n[{split}] n={N_SUB}  |h_A|={h_A.abs().mean():.3f} "
          f"|h_B|={h_B.abs().mean():.3f}  mean|h_A-h_B|={dh:.4f}")
    for name, tll, sll, mll in rows:
        print(f"  {name}: tll {tll.mean():8.3f}  sll {sll.mean():8.3f} "
              f"(med {sll.median():8.3f})  mll {mll.mean():7.3f}")
