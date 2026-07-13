"""Training / evaluation loops for EarthquakeNPP models."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .data import build_catalog_tensors, CatalogTensors, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from .model import EarthquakeNPP, ModelConfig
from .trigger import SpatialContext


@dataclass
class TrainConfig:
    dataset: str = "ComCat_25"
    crop_len: int = 512
    batch: int = 32
    warmup: int = 64           # min history (within crop) before a position is scored
    steps: int = 3000
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    val_every: int = 250
    eval_chunk: int = 8192
    eval_left_ctx: int = 4096
    eval_batch: int = 4096
    patience: int = 6          # early stop after this many non-improving validations
    seed: int = 0
    device: str = "cuda"
    spatial_weight: float = 1.0
    log: bool = True


# --------------------------------------------------------------------------------------
# crop sampling
# --------------------------------------------------------------------------------------


class CatalogTrainer:
    def __init__(self, cat: CatalogTensors, tcfg: TrainConfig, ctx_len: int, mean, std):
        self.cat = cat
        self.tcfg = tcfg
        self.ctx_len = ctx_len                       # K: history before a scored position
        self.feats = cat.features()                 # [N,4]
        self.dt = cat.dt
        self.t = cat.t_days
        self.xy = torch.stack([cat.x, cat.y], -1)   # [N,2]
        self.z = (self.xy - mean) / std             # standardized locations [N,2]
        self.dm = (cat.mag - cat.cfg.mc).clamp_min(0.0)
        self.split = cat.split
        train_idx = torch.nonzero(self.split == SPLIT_TRAIN, as_tuple=False).squeeze(-1)
        self.train_lo = int(train_idx.min())
        self.train_hi = int(train_idx.max())
        self.N = cat.n

    def sample_starts(self, B: int) -> torch.Tensor:
        Lc, wu = self.tcfg.crop_len, self.tcfg.warmup
        lo = max(0, self.train_lo - Lc + wu + 1)
        hi = min(self.N - Lc, self.train_hi - wu)
        hi = max(hi, lo)
        return torch.randint(lo, hi + 1, (B,), device=self.feats.device)

    def crop_batch(self, B: int):
        Lc = self.tcfg.crop_len
        starts = self.sample_starts(B)               # [B]
        ar = torch.arange(Lc, device=self.feats.device)
        gidx = starts.unsqueeze(1) + ar.unsqueeze(0)  # [B,Lc] global indices
        feats = self.feats[gidx]                      # [B,Lc,4]
        return feats, gidx, starts


def _unfold(x, K):
    """[B, Lc, ...] -> windows [B, Lc-K+1, K, ...] where window w covers [w, w+K-1]."""
    # x: [B, Lc, D] or [B, Lc]
    if x.dim() == 2:
        return x.unfold(1, K, 1)                                 # [B, Lc-K+1, K]
    D = x.shape[-1]
    return x.unfold(1, K, 1).permute(0, 1, 3, 2)                 # [B, Lc-K+1, K, D]


def _gather_train(model: EarthquakeNPP, hidden, gidx, trainer: CatalogTrainer):
    """Build (SpatialContext, targets) for scored positions in a crop batch.

    Scored positions are local p in [K, Lc-1] (so every scored event has K real
    predecessors inside the crop) that fall in the TRAIN split.  context for target p is
    h_{p-1}; the recent window covers events p-K..p-1.
    """
    K = trainer.ctx_len
    B, Lc, H = hidden.shape
    # recent windows aligned to target positions p=K..Lc-1  (window start w=p-K=0..Lc-1-K)
    rec_h = _unfold(hidden, K)[:, : Lc - K]                      # [B, Lc-K, K, H]
    rec_z = _unfold(_window_src(trainer.z, gidx), K)[:, : Lc - K]   # [B, Lc-K, K, 2]
    rec_t = _unfold(_window_src(trainer.t, gidx), K)[:, : Lc - K]   # [B, Lc-K, K]
    h_self = rec_h[:, :, -1, :]                                  # [B, Lc-K, H] = h_{p-1}
    t_self = rec_t[:, :, -1:]                                    # [B, Lc-K, 1]
    rec_logage = torch.log((t_self - rec_t).clamp_min(0.0) + 1e-3)

    gp = gidx[:, K:Lc]                                           # [B, Lc-K] global target idx
    is_train = (trainer.split[gp] == SPLIT_TRAIN)               # [B, Lc-K]
    flat = is_train.reshape(-1)
    sc = SpatialContext(
        h_self=h_self.reshape(-1, H)[flat],
        recent_h=rec_h.reshape(-1, K, H)[flat] if model.needs_recent else None,
        recent_z=rec_z.reshape(-1, K, 2)[flat] if model.needs_recent else None,
        recent_logage=rec_logage.reshape(-1, K)[flat] if model.needs_recent else None,
        mask=torch.ones(int(flat.sum()), K, dtype=torch.bool, device=hidden.device)
        if model.needs_recent else None,
    )
    tdt = trainer.dt[gp].reshape(-1)[flat]
    txy = trainer.xy[gp].reshape(-1, 2)[flat]
    tdm = trainer.dm[gp].reshape(-1)[flat]
    return sc, tdt, txy, tdm


def _window_src(arr, gidx):
    """Map a global per-event array [N] or [N,D] onto crop layout [B,Lc(,D)] via gidx."""
    return arr[gidx]


def train(model: EarthquakeNPP, cat: CatalogTensors, tcfg: TrainConfig):
    dev = tcfg.device
    torch.manual_seed(tcfg.seed)
    model.to(dev)
    K = model.cfg.n_recent if model.needs_recent else tcfg.warmup
    trainer = CatalogTrainer(cat, tcfg, ctx_len=K, mean=model.sp_mean, std=model.sp_std)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.steps)

    best_val = -float("inf")
    best_state = None
    bad = 0
    t0 = time.time()
    for step in range(1, tcfg.steps + 1):
        model.train()
        feats, gidx, _ = trainer.crop_batch(tcfg.batch)
        hidden = model.encode(feats)
        sc, tdt, txy, tdm = _gather_train(model, hidden, gidx, trainer)
        if sc.h_self.shape[0] == 0:
            continue
        l_t = model.temporal_loss(sc.h_self, tdt).mean()
        l_s = model.spatial_loss(sc, txy).mean()
        loss = l_t + tcfg.spatial_weight * l_s
        if model.mag_head is not None:
            loss = loss + model.magnitude_loss(sc.h_self, tdm).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        sched.step()

        if step % tcfg.val_every == 0 or step == tcfg.steps:
            vt, vs, _ = evaluate(model, cat, SPLIT_VAL, tcfg)
            score = vt + vs
            if tcfg.log:
                print(f"step {step:5d} | train l_t {l_t.item():+.3f} l_s {l_s.item():+.3f} "
                      f"| val TLL {vt:+.4f} SLL {vs:+.4f} sum {score:+.4f} "
                      f"| {time.time()-t0:.0f}s", flush=True)
            if score > best_val:
                best_val = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= tcfg.patience:
                    if tcfg.log:
                        print(f"early stop at step {step} (no val improvement in {bad} checks)")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def evaluate(model: EarthquakeNPP, cat: CatalogTensors, split: int, tcfg: TrainConfig):
    """Full-history evaluation: encode the whole catalog (chunked) and score the target split.

    Every scored event sees its full real history (via encode_long); the triggering head
    additionally gets the K most recent epicenters explicitly.
    """
    dev = tcfg.device
    model.eval()
    K = model.cfg.n_recent if model.needs_recent else 1
    feats = cat.features().to(dev)
    hidden = model.encode_long(feats, chunk=tcfg.eval_chunk, left_ctx=tcfg.eval_left_ctx)
    idx = cat.split_indices(split).to(dev)          # positions p (>=1) in split
    idx = idx[idx >= K]                              # need K real predecessors
    if idx.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    dt = cat.dt.to(dev)
    xy = torch.stack([cat.x, cat.y], -1).to(dev)
    z = (xy - model.sp_mean) / model.sp_std
    t = cat.t_days.to(dev)
    dm = (cat.mag - cat.cfg.mc).clamp_min(0.0).to(dev)
    ar = torch.arange(K, device=dev)
    tll_sum = sll_sum = mll_sum = 0.0
    n = 0
    eb = tcfg.eval_batch
    for s in range(0, idx.numel(), eb):
        p = idx[s:s + eb]                           # [b]
        h_self = hidden[p - 1]
        if model.needs_recent:
            ridx = p.unsqueeze(1) - K + ar.unsqueeze(0)   # [b,K] = p-K..p-1
            t_self = t[p - 1].unsqueeze(1)
            sc = SpatialContext(
                h_self=h_self,
                recent_h=hidden[ridx],
                recent_z=z[ridx],
                recent_logage=torch.log((t_self - t[ridx]).clamp_min(0.0) + 1e-3),
                mask=torch.ones(p.numel(), K, dtype=torch.bool, device=dev),
            )
        else:
            sc = SpatialContext(h_self=h_self)
        tll = model.temporal_logprob(h_self, dt[p])
        sll = model.spatial_logprob(sc, xy[p])
        tll_sum += tll.sum().item()
        sll_sum += sll.sum().item()
        if model.mag_head is not None:
            mll_sum += model.magnitude_logprob(h_self, dm[p]).sum().item()
        n += p.numel()
    return tll_sum / n, sll_sum / n, (mll_sum / n if model.mag_head is not None else float("nan"))
