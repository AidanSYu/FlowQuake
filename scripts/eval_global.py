"""Evaluate ONE pooled global model on each region's held-out test set.

The checkpoint is pre-trained on all listed regions' training windows, so this
is not leave-one-region-out transfer. The deployment claim tested here is
"one shared set of weights, no per-region inversion and no per-region weight
training after pooling", paired against each region's own ETAS inversion.

Run: python scripts/eval_global.py --ckpt runs/pool_global/ckpt_best.pt
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.config import Config
from flowquake.data import full_sequence_batch
from flowquake.stats import paired_gain_summary
from flowquake.train import make_model, load_catalog_cfg

ER = Path("reference/Experiments/ETAS")
REGIONS = {
    "California": ("configs/n1_density.yaml", "output_data_ComCat_25"),
    "Italy":      ("configs/italy_n1.yaml",   "output_data_Italy_25"),
    "Japan":      ("configs/japan_n1.yaml",   "output_data_Japan_25"),
    "Chile":      ("configs/chile_n1.yaml",   "output_data_Chile_25"),
    "Greece":     ("configs/greece_n1.yaml",  "output_data_Greece_25"),
    "Iran":       ("configs/iran_n1.yaml",    "output_data_Iran_25"),
}


def etas_pe(d):
    f = ER / d / "augmented_catalog.csv"
    return pd.read_csv(f, parse_dates=["time"]).dropna(subset=["TLL", "SLL"])[["time", "TLL", "SLL"]] if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/pool_global/ckpt_best.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/global_eval.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gcfg = ckpt["cfg"]

    print(f"{'region':11}{'global tll':>14}{'sll':>9}{'dTemp':>9}{'dT 95% CI':>21}{'dTot':>9}  vs region-ETAS")
    res = {}
    for reg, (cfgp, ed) in REGIONS.items():
        cfg = Config.load(cfgp)
        cat = load_catalog_cfg(cfg)
        model = make_model(gcfg, cat.stats).to(dev).eval()
        model.load_state_dict(ckpt["model"])
        model.stats = cat.stats
        tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "test")
        with torch.no_grad():
            out = model.log_likelihood(tokens.to(dev), target.to(dev), mask.to(dev),
                                       lastk.to(dev), raw_next.to(dev), steps=64)
        tll = out["tll"].cpu().numpy(); sll = out["sll"].cpu().numpy()
        # align to test event times (next-event positions)
        m = np.zeros(cat.n_events, dtype=bool); nx = mask[0].cpu().numpy(); m[1:] = nx[:-1]
        pe = pd.DataFrame({"time": cat.times[m], "tll": tll, "sll": sll})
        epe = etas_pe(ed)
        rec = {"tll": float(tll.mean()), "sll": float(sll.mean())}
        if epe is not None:
            mg = pe.merge(epe, on="time", how="inner")
            dT = mg["tll"] - mg["TLL"]; dTot = (mg["tll"] + mg["sll"]) - (mg["TLL"] + mg["SLL"])
            zt = dT.mean() / (dT.std(ddof=1) / np.sqrt(len(mg)))
            seed = sum(ord(ch) for ch in reg) % 10000
            gt = paired_gain_summary(dT, seed=seed)
            gj = paired_gain_summary(dTot, seed=seed + 1)
            rec.update(
                dT=gt.mean, dT_se=gt.stderr, dT_ci=gt.asdict()["ci"],
                dT_decision=gt.decision, dTot=gj.mean,
                dTot_ci=gj.asdict()["ci"], dTot_decision=gj.decision,
                z_t=float(zt), n=len(mg),
            )
            ci = f"[{gt.ci_low:+.3f},{gt.ci_high:+.3f}]"
            print(f"{reg:11}{tll.mean():>14.4f}{sll.mean():>9.4f}{gt.mean:>+9.3f}{ci:>21}{gj.mean:>+9.3f}  {gt.decision.upper()}")
        res[reg] = rec
        pe.to_csv(f"runs/global_{reg}_per_event.csv", index=False)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  (global = one pooled model, no per-region weight fit, vs each region's own ETAS)")


if __name__ == "__main__":
    main()
