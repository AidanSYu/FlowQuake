"""Diagnose val-LL collapse: ODE step sensitivity, train vs val, z0 norms."""

import sys
import torch
import numpy as np

sys.path.insert(0, ".")
from flowquake.config import Config
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.train import make_model

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/comcat25/ckpt_last.pt"
device = "cuda"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg: Config = ckpt["cfg"]
cat = load_catalog(cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
                   cfg.data.train_start, cfg.data.val_start, cfg.data.test_start,
                   cfg.data.test_end)
model = make_model(cfg, ckpt["stats"]).to(device).eval()
model.load_state_dict(ckpt["model"])
print(f"ckpt step {ckpt['step']}")

n_sub = 512
rng = np.random.default_rng(0)

for split in ["train", "val"]:
    tokens, target, mask = full_sequence_batch(cat, split)
    idx = mask[0].nonzero(as_tuple=True)[0]
    sel = rng.choice(len(idx), n_sub, replace=False)
    keep = torch.zeros_like(mask[0])
    keep[idx[torch.from_numpy(sel)]] = True
    mask_s = keep.unsqueeze(0)
    for steps in [32, 128, 512]:
        out = model.log_likelihood(tokens.to(device), target.to(device),
                                   mask_s.to(device), steps=steps)
        print(f"{split:5s} steps={steps:4d}  tll {out['tll'].mean():9.3f}  "
              f"sll {out['sll'].mean():9.3f}  mll {out['mll'].mean():9.3f}  "
              f"tll_med {out['tll'].median():8.3f}")

# z0 norm distribution for the time head at 512 steps on val
tokens, target, mask = full_sequence_batch(cat, "val")
idx = mask[0].nonzero(as_tuple=True)[0][:256]
keep = torch.zeros_like(mask[0]); keep[idx] = True
h = model.encode_full(tokens.to(device))[keep.unsqueeze(0).to(device)]
tgt = target[keep.unsqueeze(0)].to(device)

flow = model.head_t
z = tgt[:, 0:1].float()
cond = h.float()
dt = 1.0 / 512
with torch.no_grad():
    for i in range(512):
        t1 = 1.0 - i * dt
        v, _ = flow._vel_and_div(z, torch.tensor(t1, device=device), cond)
        z = z - dt * v  # euler backward for speed
print("z0 abs: mean", z.abs().mean().item(), "max", z.abs().max().item())
vn = flow.velocity(tgt[:, 0:1].float(), torch.tensor(1.0, device=device), cond)
print("|v(u,1)|: mean", vn.abs().mean().item(), "max", vn.abs().max().item())
