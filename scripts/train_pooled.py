"""Pooled multi-region pre-training -- the "foundation model" for FlowQuake.

Trains ONE set of weights on several regions at once. The model operates in
NORMALIZED space and carries a single `stats` dict used only to (un)normalize;
the learned weights are region-agnostic (this is exactly why zero-/few-shot
transfer works -- see scripts/transfer_eval.py). So pooling is faithful iff:
  * each minibatch is region-homogeneous (one region's normalized crops), and
  * model.stats is swapped to that region's stats for the step (and the cached
    KDE grid is invalidated so the spatial head reads the right background map).

This gives a leave-one-region-out protocol: pre-train on N-1 regions here, then
  zero-shot  -> scripts/transfer_eval.py --ckpt <out>/ckpt_best.pt --catalog <heldout>
  few-shot   -> python -m flowquake.train <heldout_cfg> --init-from <out>/ckpt_best.pt --steps 2000

Region sampling is temperature-weighted (w_r proportional to N_train_r ** alpha)
so small regions are not drowned by large ones. Early stopping uses the pooled
validation NLL of the TRAINING regions only -- the held-out region is never seen.

Run:
  python scripts/train_pooled.py --configs configs/japan_n1.yaml configs/chile_n1.yaml \
      configs/greece_n1.yaml --out runs/pool_loo_iran --steps 30000 --alpha 0.5
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from flowquake.config import Config
from flowquake.data import CropDataset, full_sequence_batch
from flowquake.train import load_catalog_cfg, make_model


def set_region(model, stats):
    """Point the model at a region's normalization context for one step."""
    model.stats = stats
    model._kde_t = None       # invalidate cached background-KDE grid/sampler
    model._kde_probs = None


@torch.no_grad()
def validate_pooled(model, cats, stats, device, n_events, ode_steps, seed=0):
    """Event-count-weighted mean val NLL across the TRAINING regions."""
    tot_n, tot_nll = 0, 0.0
    per = {}
    for name, cat in cats.items():
        set_region(model, stats[name])
        tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "val")
        idx = mask[0].nonzero(as_tuple=True)[0]
        if n_events and len(idx) > n_events:
            sel = torch.from_numpy(np.random.default_rng(seed).choice(len(idx), n_events, replace=False))
            keep = torch.zeros_like(mask[0]); keep[idx[sel]] = True
            mask = keep.unsqueeze(0)
        out = model.log_likelihood(tokens.to(device), target.to(device), mask.to(device),
                                   lastk.to(device), raw_next.to(device), steps=ode_steps)
        n = out["tll"].numel()
        nll = -(out["tll"].mean().item() + out["sll"].mean().item())
        per[name] = {"nll": round(nll, 4), "tll": round(out["tll"].mean().item(), 4),
                     "sll": round(out["sll"].mean().item(), 4), "n": int(n)}
        tot_n += n; tot_nll += nll * n
    return tot_nll / max(tot_n, 1), per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="one yaml per training region")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--alpha", type=float, default=0.5, help="region sampling temperature exponent")
    ap.add_argument("--seed", type=int, default=1555)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    base = Config.load(args.configs[0])    # model/train hyperparams come from the first region
    tc = base.train
    cfgs = {Path(c).stem: Config.load(c) for c in args.configs}

    cats, stats, loaders = {}, {}, {}
    for name, cfg in cfgs.items():
        cat = load_catalog_cfg(cfg)
        cats[name] = cat; stats[name] = cat.stats
        ds = CropDataset(cat, window=tc.window, burn_in=tc.burn_in,
                         n_crops=tc.batch_size * 1_000_000,
                         seed=tc.seed + sum(ord(ch) for ch in name))  # deterministic per-region offset
        loaders[name] = iter(DataLoader(ds, batch_size=tc.batch_size, num_workers=0))
        print(f"  {name}: {cat.n_events} events | train {cat.target_train.sum()} "
              f"val {cat.target_val.sum()}")

    names = list(cats)
    w = np.array([float(cats[n].target_train.sum()) ** args.alpha for n in names])
    w /= w.sum()
    print(f"region sampling weights: {dict(zip(names, np.round(w, 3)))}")

    # all regions must yield identical model architecture (mix_k, cond_dim)
    model = make_model(base, stats[names[0]]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"pooled model: {n_params/1e6:.2f}M params over {len(names)} regions")

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    def lr_at(step):
        if step < tc.warmup:
            return step / max(tc.warmup, 1)
        p = (step - tc.warmup) / max(args.steps - tc.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    rng = np.random.default_rng(args.seed)
    metrics_f = open(out_dir / "metrics.jsonl", "a")
    best_nll, bad, t0 = float("inf"), 0, time.time()
    base.dump(out_dir / "config.yaml")

    for step in range(1, args.steps + 1):
        name = names[rng.choice(len(names), p=w)]
        batch = [t.to(device) for t in next(loaders[name])]
        set_region(model, stats[name])
        loss, parts = model.fm_losses(*batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step(); sched.step()

        if step % 100 == 0 or step == 1:
            rec = {"step": step, "region": name, "loss": round(loss.item(), 4), **parts,
                   "lr": round(sched.get_last_lr()[0], 6), "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec)); metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()

        if step % tc.val_every == 0 or step == args.steps:
            model.eval()
            nll, per = validate_pooled(model, cats, stats, device, tc.val_events, tc.val_ode_steps)
            model.train()
            rec = {"step": step, "pooled_val_nll": round(nll, 4), "per": per,
                   "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec)); metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
            torch.save({"cfg": base, "model": model.state_dict(), "stats": stats[names[0]],
                        "step": step, "pooled_val_nll": nll, "regions": names},
                       out_dir / "ckpt_last.pt")
            if nll < best_nll:
                best_nll, bad = nll, 0
                torch.save({"cfg": base, "model": model.state_dict(), "stats": stats[names[0]],
                            "step": step, "pooled_val_nll": nll, "regions": names},
                           out_dir / "ckpt_best.pt")
                print(f"  new best pooled val nll {best_nll:.4f}")
            else:
                bad += 1
                if bad >= tc.patience:
                    print(f"early stop at step {step} (best pooled val nll {best_nll:.4f})"); break

    metrics_f.close()
    print(f"done in {time.time()-t0:.0f}s; best pooled val nll {best_nll:.4f}; ckpt in {out_dir}")


if __name__ == "__main__":
    main()
