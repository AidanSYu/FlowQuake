"""Test-window evaluation against the EarthquakeNPP ETAS baseline.

Computes per-event tll/sll/mll over the test targets, compares with the
published ETAS scores (ll_scores.json) and, when available, the per-event
ETAS TLL/SLL from augmented_catalog.csv for a paired comparison.

Usage:
    python -m flowquake.evaluate runs/comcat25/ckpt_best.pt [--steps 96]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import full_sequence_batch
from .train import load_catalog_cfg, make_model

ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")


def paired_vs_etas(cat, mask_np, tll, sll, etas_csv: Path):
    """Merge per-event scores with ETAS per-event TLL/SLL by timestamp."""
    aug = pd.read_csv(etas_csv, parse_dates=["time"])
    aug = aug.dropna(subset=["TLL", "SLL"])
    ours = pd.DataFrame({
        "time": cat.times[mask_np],
        "tll": tll,
        "sll": sll,
    })
    merged = ours.merge(aug[["time", "TLL", "SLL"]], on="time", how="inner")
    if len(merged) == 0:
        return None
    d_t = merged["tll"] - merged["TLL"]
    d_s = merged["sll"] - merged["SLL"]
    d_j = d_t + d_s
    out = {"n_matched": int(len(merged))}
    for name, d in [("temporal", d_t), ("spatial", d_s), ("joint", d_j)]:
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[name] = {
            "mean_gain": float(d.mean()),
            "stderr": float(se),
            "win_rate": float((d > 0).mean()),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--steps", type=int, default=96, help="ODE steps for exact LL")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--etas-dir", default=str(ETAS_DIR))
    # frozen-checkpoint forward-window scoring: override the eval catalog and
    # test window WITHOUT touching the checkpoint's stats or train/val splits.
    ap.add_argument("--catalog", default=None, help="override catalog_path")
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--tag", default=None, help="output suffix instead of split name")
    ap.add_argument("--etas-aug", default=None,
                    help="per-event ETAS csv (time,TLL,SLL) for pairing")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["cfg"]
    if args.catalog:
        cfg.data.catalog_path = args.catalog
    if args.test_start:
        cfg.data.test_start = args.test_start
    if args.test_end:
        cfg.data.test_end = args.test_end
    device = torch.device(args.device)

    cat = load_catalog_cfg(cfg)
    model = make_model(cfg, ckpt["stats"]).to(device).eval()
    model.load_state_dict(ckpt["model"])

    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, args.split)
    out = model.log_likelihood(
        tokens.to(device), target.to(device), mask.to(device),
        lastk.to(device), raw_next.to(device), steps=args.steps
    )
    tll = out["tll"].cpu().numpy()
    sll = out["sll"].cpu().numpy()
    mll = out["mll"].cpu().numpy()

    res = {
        "split": args.split,
        "n_events": int(len(tll)),
        "ode_steps": args.steps,
        "tll": float(tll.mean()),
        "sll": float(sll.mean()),
        "mll": float(mll.mean()),
        "nll": float(-(tll.mean() + sll.mean())),
    }

    etas_dir = Path(args.etas_dir)
    ll_file = etas_dir / "ll_scores.json"
    if ll_file.exists() and args.split == "test":
        with open(ll_file) as f:
            base = json.load(f)
        res["baselines"] = base
        res["beats_ETAS_nll"] = bool(res["nll"] < base["ETAS"]["nll"])

    # mask positions select events whose *next* event is the target; the
    # target events themselves are positions shifted by +1.
    mask_np = np.zeros(cat.n_events, dtype=bool)
    nxt = mask[0].cpu().numpy()
    mask_np[1:] = nxt[:-1]
    aug_csv = Path(args.etas_aug) if args.etas_aug else etas_dir / "augmented_catalog.csv"
    if aug_csv.exists() and (args.split == "test" or args.etas_aug):
        paired = paired_vs_etas(cat, mask_np, tll, sll, aug_csv)
        if paired:
            res["paired_vs_ETAS"] = paired

    tag = args.tag or args.split
    out_path = Path(args.ckpt).parent / f"eval_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    per_event = pd.DataFrame({
        "time": cat.times[mask_np], "tll": tll, "sll": sll, "mll": mll,
    })
    per_event.to_csv(Path(args.ckpt).parent / f"per_event_{tag}.csv", index=False)

    print(json.dumps(res, indent=2))
    if "baselines" in res:
        e = res["baselines"]["ETAS"]
        print(f"\n=== ComCat_25 {args.split} ===")
        print(f"{'':12} {'tll':>9} {'sll':>10} {'nll':>9}")
        print(f"{'ETAS':12} {e['tll']:9.4f} {e['sll']:10.4f} {e['nll']:9.4f}")
        print(f"{'FlowQuake':12} {res['tll']:9.4f} {res['sll']:10.4f} {res['nll']:9.4f}")


if __name__ == "__main__":
    main()
