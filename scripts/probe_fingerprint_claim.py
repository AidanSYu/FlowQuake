"""Verify reviewer claim: input_noise=0.1 on encoder input does NOT blur the
positional fingerprint in h.

Measures, on a 2048-event training window with the trained checkpoint:
  1. mean L2 |h_noised - h_clean| per position
  2. typical inter-position L2 |h_i - h_j| (random pairs) and mean |h|
  3. top-1 re-identification: nearest neighbour of h under noise draw 1
     among all 2048 positions' h under INDEPENDENT noise draw 2
  4. per-layer/head memory lengths 1/(mean_dt * A) from the checkpoint

Run: python scripts/probe_fingerprint_claim.py runs/comcat25/ckpt_last.pt
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from flowquake.config import Config
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.train import make_model

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/comcat25/ckpt_last.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
W = 2048
NOISE = 0.1

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg: Config = ckpt["cfg"]
cat = load_catalog(cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
                   cfg.data.train_start, cfg.data.val_start, cfg.data.test_start,
                   cfg.data.test_end)
model = make_model(cfg, ckpt["stats"]).to(device).eval()
model.load_state_dict(ckpt["model"])
print(f"ckpt step {ckpt['step']}  input_noise={model.input_noise}  device={device}")

tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "train")
idx = mask[0].nonzero(as_tuple=True)[0]
start = int(idx[len(idx) // 2].item())  # a window in the middle of train
win = tokens[:, start : start + W].to(device)
print(f"window [{start}, {start + W})  token dim {win.shape[-1]}")

torch.manual_seed(123)
with torch.no_grad():
    h_clean = model.encoder(win)[0]                                   # (W, d)
    h_n1 = model.encoder(win + NOISE * torch.randn_like(win))[0]
    h_n2 = model.encoder(win + NOISE * torch.randn_like(win))[0]

d_noise1 = (h_n1 - h_clean).norm(dim=-1)
d_noise12 = (h_n1 - h_n2).norm(dim=-1)
hnorm = h_clean.norm(dim=-1)

# typical inter-position distance: random pairs + adjacent positions
g = torch.Generator(device="cpu").manual_seed(0)
ii = torch.randint(0, W, (20000,), generator=g)
jj = torch.randint(0, W, (20000,), generator=g)
keep = ii != jj
pair_d = (h_clean[ii[keep]] - h_clean[jj[keep]]).norm(dim=-1)
adj_d = (h_clean[1:] - h_clean[:-1]).norm(dim=-1)

print(f"|h| mean                       {hnorm.mean():.2f}")
print(f"|h_noised - h_clean| mean      {d_noise1.mean():.2f}")
print(f"|h_n1 - h_n2| (same pos) mean  {d_noise12.mean():.2f}")
print(f"inter-position |dh| rand pairs {pair_d.mean():.2f} (median {pair_d.median():.2f})")
print(f"adjacent-position |dh| mean    {adj_d.mean():.2f} (median {adj_d.median():.2f})")

# top-1 re-identification: for each position i, nearest h_n2_j to h_n1_i
D = torch.cdist(h_n1, h_n2)            # (W, W)
top1 = (D.argmin(dim=1) == torch.arange(W, device=device)).float().mean()
# also vs clean gallery
Dc = torch.cdist(h_n1, h_clean)
top1c = (Dc.argmin(dim=1) == torch.arange(W, device=device)).float().mean()
print(f"top-1 re-ID noised-vs-noised   {top1.item() * 100:.2f}%  ({W} candidates)")
print(f"top-1 re-ID noised-vs-clean    {top1c.item() * 100:.2f}%  ({W} candidates)")

# nearest WRONG neighbour margin: is the noise ball smaller than half the
# nearest inter-position gap?
D2 = torch.cdist(h_clean, h_clean) + torch.eye(W, device=device) * 1e9
nn_gap = D2.min(dim=1).values
print(f"nearest-neighbour gap in clean h: mean {nn_gap.mean():.2f}, "
      f"p5 {nn_gap.quantile(0.05):.2f}, min {nn_gap.min():.2f}")
frac_sep = (nn_gap > 2 * d_noise1).float().mean()
print(f"fraction of positions with nn_gap > 2*noise displacement: "
      f"{frac_sep.item() * 100:.1f}%")

# memory lengths 1/(mean dt * A) per layer/head, dt from this window forward
print("\nper-layer memory lengths (events), 1/(mean_dt * A):")
with torch.no_grad():
    h = model.encoder.embed(win)
    for li, (norm, block) in enumerate(zip(model.encoder.norms, model.encoder.blocks)):
        u = norm(h)
        _, _, dt_raw = block._split(block.in_proj(u))
        dt = F.softplus(dt_raw + block.dt_bias)         # (1, L, H)
        A = torch.exp(block.A_log)                      # (H,)
        mem = 1.0 / (dt.mean(dim=(0, 1)) * A)
        print(f"  layer {li}: " + " ".join(f"{m:.1f}" for m in mem.tolist()))
        h = h + block(u)
