"""Training: chunked crops over the whole catalog, early stop on val NLL.

Usage:
    python -m flowquake.train configs/comcat25.yaml [--steps N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import CropDataset, full_sequence_batch, load_catalog
from .model import FlowQuakeTPP


def make_model(cfg: Config, stats: dict) -> FlowQuakeTPP:
    m = cfg.model
    return FlowQuakeTPP(
        d_model=m.d_model, n_layers=m.n_layers, d_state=m.d_state,
        n_heads=m.n_heads, expand=m.expand, chunk=m.chunk,
        flow_hidden=m.flow_hidden, flow_layers=m.flow_layers,
        stats=stats, loss_weights=tuple(m.loss_weights),
        sigma_min=tuple(getattr(m, "sigma_min", (0.0, 0.0, 0.0))),
        dropout=getattr(m, "dropout", 0.0),
        mag_dequant=getattr(m, "mag_dequant", 0.0),
        input_noise=getattr(m, "input_noise", 0.0),
    )


@torch.no_grad()
def validate(model, cat, device, n_events: int, ode_steps: int, seed: int = 0):
    """Subsampled val-set NLL (per-event, physical units)."""
    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "val")
    idx = mask[0].nonzero(as_tuple=True)[0]
    if n_events and len(idx) > n_events:
        sel = torch.from_numpy(
            np.random.default_rng(seed).choice(len(idx), n_events, replace=False)
        )
        keep = torch.zeros_like(mask[0])
        keep[idx[sel]] = True
        mask = keep.unsqueeze(0)
    out = model.log_likelihood(
        tokens.to(device), target.to(device), mask.to(device),
        lastk.to(device), raw_next.to(device), steps=ode_steps
    )
    tll = out["tll"].mean().item()
    sll = out["sll"].mean().item()
    mll = out["mll"].mean().item()
    return {"tll": tll, "sll": sll, "mll": mll, "nll": -(tll + sll)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-after", action="store_true",
                    help="run the full test evaluation on ckpt_best when done")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.steps is not None:
        cfg.train.steps = args.steps
    if args.out is not None:
        cfg.train.out_dir = args.out
    tc = cfg.train

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    device = torch.device(args.device)

    out_dir = Path(tc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "config.yaml")

    cat = load_catalog(
        cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
        cfg.data.train_start, cfg.data.val_start, cfg.data.test_start,
        cfg.data.test_end,
    )
    print(f"catalog: {cat.n_events} events | train targets {cat.target_train.sum()} "
          f"| val {cat.target_val.sum()} | test {cat.target_test.sum()}")

    ds = CropDataset(cat, window=tc.window, burn_in=tc.burn_in,
                     n_crops=tc.batch_size * 1_000_000, seed=tc.seed)
    dl = DataLoader(ds, batch_size=tc.batch_size, num_workers=0)

    model = make_model(cfg, cat.stats).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    def lr_at(step):
        if step < tc.warmup:
            return step / max(tc.warmup, 1)
        p = (step - tc.warmup) / max(tc.steps - tc.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    metrics_f = open(out_dir / "metrics.jsonl", "a")
    best_nll = float("inf")
    bad_checks = 0
    t0 = time.time()
    it = iter(dl)

    for step in range(1, tc.steps + 1):
        batch = [t.to(device) for t in next(it)]
        loss, parts = model.fm_losses(*batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        sched.step()

        if step % 100 == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), **parts,
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec))
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()

        if step % tc.val_every == 0 or step == tc.steps:
            model.eval()
            v = validate(model, cat, device, tc.val_events, tc.val_ode_steps)
            model.train()
            rec = {"step": step, "val": v, "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec))
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            torch.save({"cfg": cfg, "model": model.state_dict(), "stats": cat.stats,
                        "step": step, "val": v}, out_dir / "ckpt_last.pt")
            if v["nll"] < best_nll:
                best_nll = v["nll"]
                bad_checks = 0
                torch.save({"cfg": cfg, "model": model.state_dict(), "stats": cat.stats,
                            "step": step, "val": v}, out_dir / "ckpt_best.pt")
                print(f"  new best val nll {best_nll:.4f}")
            else:
                bad_checks += 1
                if bad_checks >= tc.patience:
                    print(f"early stop at step {step} (best val nll {best_nll:.4f})")
                    break

    metrics_f.close()
    print(f"done in {time.time() - t0:.0f}s; best val nll {best_nll:.4f}")

    if args.eval_after and (out_dir / "ckpt_best.pt").exists():
        from . import evaluate as eval_mod
        eval_mod.main([str(out_dir / "ckpt_best.pt"), "--steps", "64",
                       "--device", args.device])


if __name__ == "__main__":
    main()
