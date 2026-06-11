"""Cold-start (training-like) vs warm (eval-like) encoder state diagnostic.

For a saved checkpoint:
1. Dump the learned per-layer decay spectrum: A = exp(A_log), dt at the
   softplus bias point, implied memory length 1/(dt*A) vs the 2048 window.
2. Compare h at the same val positions computed (a) warm via encode_full from
   event 0 (what validate()/evaluate use) vs (b) cold-start 2048-token windows
   ending at the position (what training exposes).
3. Compare val tll/sll/mll under both conditionings.
"""

import sys
import torch
import torch.nn.functional as Fn
import numpy as np

sys.path.insert(0, ".")
from flowquake.config import Config
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.train import make_model

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/comcat25/ckpt_last.pt"
device = "cpu"
N_SUB = 192
ODE_STEPS = 32
WIN = 2048

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg: Config = ckpt["cfg"]
cat = load_catalog(cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
                   cfg.data.train_start, cfg.data.val_start, cfg.data.test_start,
                   cfg.data.test_end)
model = make_model(cfg, ckpt["stats"]).to(device).eval()
model.load_state_dict(ckpt["model"])
print(f"ckpt {ckpt_path} step {ckpt['step']} val {ckpt.get('val')}")

# --- 1. decay spectrum -----------------------------------------------------
print("\n--- learned decay spectrum (per layer) ---")
for li, blk in enumerate(model.encoder.blocks):
    A = torch.exp(blk.A_log.detach())
    dt0 = Fn.softplus(blk.dt_bias.detach())  # dt at zero pre-activation
    mem = 1.0 / (dt0 * A)                    # events to decay by 1/e
    print(f"layer {li}: A {A.numpy().round(3)}  dt_bias {dt0.numpy().round(4)}  "
          f"mem_len {mem.numpy().round(0)}  (window={WIN})")

# --- 2. warm vs cold h -----------------------------------------------------
tokens, target, mask = full_sequence_batch(cat, "val")
tokens, target = tokens.to(device), target.to(device)
idx = mask[0].nonzero(as_tuple=True)[0]
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(idx), N_SUB, replace=False))
pos = idx[torch.from_numpy(sel)]            # positions, all >= WIN into catalog

with torch.no_grad():
    h_warm_all = model.encode_full(tokens)              # (1, E, d)
    h_warm = h_warm_all[0, pos]                         # (N, d)

    # cold-start: window of WIN tokens ending at pos (pos is last index)
    wins = torch.stack([tokens[0, p - WIN + 1: p + 1] for p in pos])  # (N, WIN, 32)
    h_cold = []
    for i in range(0, N_SUB, 32):
        h_cold.append(model.encoder(wins[i:i + 32])[:, -1])
    h_cold = torch.cat(h_cold)                          # (N, d)

    # also state norms: final scan-state magnitude warm vs cold (layer 0..L)
    _, st_warm = model.encoder.prefill(tokens)
    _, st_cold = model.encoder.prefill(tokens[:, -WIN:])
    for li, ((sw, _), (sc, _)) in enumerate(zip(st_warm, st_cold)):
        print(f"layer {li}: |state| warm {sw.abs().mean():.4f}  "
              f"cold-{WIN} {sc.abs().mean():.4f}  ratio {sw.abs().mean()/sc.abs().mean():.2f}")

d = (h_warm - h_cold).norm(dim=-1)
print(f"\n|h_warm - h_cold| mean {d.mean():.4f}  |h_warm| mean {h_warm.norm(dim=-1).mean():.4f}  "
      f"|h_cold| mean {h_cold.norm(dim=-1).mean():.4f}")
cos = Fn.cosine_similarity(h_warm, h_cold, dim=-1)
print(f"cosine(h_warm, h_cold): mean {cos.mean():.4f}  min {cos.min():.4f}")

# --- 3. LL under both conditionings ----------------------------------------
import math
st = model.stats
tok = tokens[0, pos]
tgt = target[0, pos]
u_t, u_m = tgt[:, 0:1], tgt[:, 3:4]
u_s = tgt[:, 1:3] - tok[:, 1:3]
log_tau = tgt[:, 0] * st["log_tau_std"] + st["log_tau_mean"]

for name, h in [("warm", h_warm), ("cold", h_cold)]:
    with torch.no_grad():
        cond = torch.cat([h, tok], dim=-1)
        lp_t = model.head_t.log_prob(u_t, cond, steps=ODE_STEPS)
        lp_s = model.head_s.log_prob(u_s, torch.cat([cond, u_t], -1), steps=ODE_STEPS)
        lp_m = model.head_m.log_prob(u_m, torch.cat([cond, u_t, u_s], -1), steps=ODE_STEPS)
    tll = (lp_t - math.log(st["log_tau_std"]) - log_tau).mean().item()
    sll = (lp_s - math.log(st["x_std"] * st["y_std"])).mean().item()
    mll = (lp_m - math.log(st["mag_std"])).mean().item()
    print(f"{name}: tll {tll:8.3f}  sll {sll:8.3f}  mll {mll:8.3f}")
