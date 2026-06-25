"""Zero-shot cross-region transfer: apply a California-trained FlowQuake to a
DIFFERENT tectonic regime (Japan) WITHOUT retraining, and measure how much of
its skill survives. ETAS must be refit per region; FlowQuake's heads are
observation-anchored / translation-invariant, so in principle one model
transfers. This is the de-risk for that claim.

Model WEIGHTS stay California-trained. Input standardisation + background map
are recomputed on the target catalog (data preprocessing, not model fitting —
strictly less info than ETAS, which refits all parameters). The temporal flow
operates in normalised space, so we feed target-normalised tokens and
un-normalise tll with the target's stats.

Baselines on the target: homogeneous Poisson (temporal) and uniform-in-region
(spatial), so we can report "fraction of a naive baseline's deficit closed".

Run: python scripts/transfer_eval.py --ckpt runs/n1_density/ckpt_best.pt \
        --catalog reference/Datasets/Japan_Deprecated/Japan_catalog.csv \
        --mcut 3.0 --test-start 2011-01-01 --test-end 2020-01-01 --device cuda
"""
import argparse, json, math
from pathlib import Path
import numpy as np, torch
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.train import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/n1_density/ckpt_best.pt")
    ap.add_argument("--catalog", default="reference/Datasets/Japan_Deprecated/Japan_catalog.csv")
    ap.add_argument("--mcut", type=float, default=3.0)
    ap.add_argument("--aux-start", default="1990-01-01")
    ap.add_argument("--train-start", default="1990-06-01")
    ap.add_argument("--val-start", default="2009-01-01")
    ap.add_argument("--test-start", default="2011-01-01")
    ap.add_argument("--test-end", default="2020-01-01")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    n_near = getattr(cfg.data, "n_near", 0)
    adaptive_bg = getattr(cfg.data, "adaptive_bg", False)

    # target catalog tensors + TARGET stats (preprocessing only)
    cat = load_catalog(args.catalog, args.mcut, args.aux_start, args.train_start,
                       args.val_start, args.test_start, args.test_end,
                       n_near=n_near, near_rmax_km=getattr(cfg.data, "near_rmax_km", 30.0),
                       adaptive_bg=adaptive_bg)
    print(f"target: {cat.n_events} events | test targets {cat.target_test.sum()}")

    model = make_model(cfg, cat.stats).to(dev).eval()   # build with TARGET stats
    model.load_state_dict(ckpt["model"])                # California-trained WEIGHTS
    model.stats = cat.stats                             # un-normalise in target units

    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "test")
    with torch.no_grad():
        out = model.log_likelihood(tokens.to(dev), target.to(dev), mask.to(dev),
                                   lastk.to(dev), raw_next.to(dev), steps=args.steps)
    tll = out["tll"].mean().item(); sll = out["sll"].mean().item(); mll = out["mll"].mean().item()

    # target baselines
    t_days = cat.t_days
    test_mask = cat.target_test
    period = t_days[test_mask].max() - t_days[test_mask].min()
    lam = test_mask.sum() / period                       # events/day
    pois_tll = math.log(lam) - 1.0                        # E[log lam - lam*tau], exp gaps
    area = cat.stats["bg_area"]
    unif_sll = -math.log(area)

    print(f"\n=== zero-shot California -> {Path(args.catalog).stem} ===")
    print(f"{'':16}{'tll':>9}{'sll':>10}{'nll':>9}")
    print(f"{'Poisson/uniform':16}{pois_tll:>9.4f}{unif_sll:>10.4f}{-(pois_tll+unif_sll):>9.4f}")
    print(f"{'FlowQuake(CA)':16}{tll:>9.4f}{sll:>10.4f}{-(tll+sll):>9.4f}")
    print(f"\ntemporal: FQ beats Poisson by {tll - pois_tll:+.4f} nats/event"
          f"  ({'TRANSFERS' if tll > pois_tll + 0.1 else 'weak'})")
    print(f"spatial : FQ beats uniform by {sll - unif_sll:+.4f} nats/event")
    # name outputs by TARGET catalog so opposite-direction runs don't clobber
    tag = Path(args.catalog).stem.replace("_catalog", "")
    json.dump({"tll": tll, "sll": sll, "mll": mll, "pois_tll": pois_tll,
               "unif_sll": unif_sll, "n_test": int(test_mask.sum()),
               "ckpt": args.ckpt},
              open(f"runs/transfer_{tag}.json", "w"), indent=2)
    # back-compat alias used by the figure script for the CA->Japan run
    if tag == "Japan":
        import shutil
        shutil.copy(f"runs/transfer_{tag}.json", "runs/transfer_japan.json")

    # per-event scores for the paired vs-ETAS comparison (align on time)
    import pandas as pd
    mask_np = np.zeros(cat.n_events, dtype=bool)
    nxt = mask[0].cpu().numpy(); mask_np[1:] = nxt[:-1]
    pe = pd.DataFrame({"time": cat.times[mask_np],
                       "tll": out["tll"].cpu().numpy(),
                       "sll": out["sll"].cpu().numpy()})
    pe.to_csv(Path("runs") / f"transfer_{tag}_per_event.csv", index=False)
    if tag == "Japan":
        pe.to_csv(Path("runs") / "transfer_japan_per_event.csv", index=False)


if __name__ == "__main__":
    main()
