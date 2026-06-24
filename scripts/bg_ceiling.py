"""Estimate the retraining ceiling for a better background.

The eval-time grid swap is capped by FROZEN mixture weights. To estimate what
retraining could reach, floor-mix a reliable adaptive background into the
existing density with weight w:  f_new = (1-w) f_model + w f_adaptive_bg.
This is exactly what the model would learn if it routed weight w to a good
background component. Sweep w to bound the achievable sll.

Run (CPU): python scripts/bg_ceiling.py
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
from flowquake.config import Config
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.model import SAFE_TOKEN_DIMS
from flowquake.train import make_model
from scripts.bg_experiment import adaptive_kde

ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")
CKPT = "runs/n1_density/ckpt_best.pt"


def main():
    torch.set_grad_enabled(False)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["cfg"]
    cat = load_catalog(cfg.data.catalog_path, cfg.data.mcut, cfg.data.aux_start,
                       cfg.data.train_start, cfg.data.val_start,
                       cfg.data.test_start, cfg.data.test_end)
    model = make_model(cfg, ckpt["stats"]).eval()
    model.load_state_dict(ckpt["model"])
    stats = model.stats
    tokens, target, mask, lastk, raw_next = full_sequence_batch(cat, "test")

    x, y = cat.raw[:, 1].numpy(), cat.raw[:, 2].numpy()
    pretest = np.asarray(cat.times < cfg.data.test_start)
    adapt = adaptive_kde(x[pretest], y[pretest], stats, k=4, sigma_min=0.75, floor=0.03)

    cond = tokens[0][mask[0]][:, SAFE_TOKEN_DIMS]
    comp_xy, comp_feats = model._comp_inputs(lastk[0][mask[0]])
    s_tgt = raw_next[0][mask[0]][:, :2]

    # baseline model sll (committed grid) and adaptive bg log-density at events
    sll_model = []
    for i in range(0, cond.shape[0], 8192):
        sl = slice(i, i + 8192)
        sll_model.append(model.head_s.log_prob(
            s_tgt[sl], comp_xy[sl], comp_feats[sl], cond[sl], stats["bg_area"],
            log_kde_at_s=model._log_kde(s_tgt[sl])))
    sll_model = torch.cat(sll_model).numpy()

    model.stats["kde_log_grid"] = adapt
    model._kde_t = None
    bg_log = model._log_kde(s_tgt).numpy()  # adaptive bg log-density at each event

    aug = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"]).dropna(subset=["SLL"])
    mask_np = np.zeros(cat.n_events, dtype=bool); mask_np[1:] = mask[0].numpy()[:-1]
    ev = pd.DataFrame({"time": cat.times[mask_np]})
    etas = ev.merge(aug[["time", "SLL"]], on="time", how="inner")["SLL"].to_numpy()

    print(f"{'w_bg':>6} {'sll':>9} {'gapETAS':>8} {'win%':>6} {'<-14':>6} {'<-12':>6}")
    print(f"{'0.000':>6} {sll_model.mean():>9.4f} {(etas-sll_model).mean():>8.4f} "
          f"{100*(sll_model>etas).mean():>5.1f} {(sll_model<-14).sum():>6} {(sll_model<-12).sum():>6}")
    for w in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        mixed = np.logaddexp(np.log(1 - w) + sll_model, np.log(w) + bg_log)
        print(f"{w:>6.3f} {mixed.mean():>9.4f} {(etas-mixed).mean():>8.4f} "
              f"{100*(mixed>etas).mean():>5.1f} {(mixed<-14).sum():>6} {(mixed<-12).sum():>6}")
    print(f"\nETAS sll -8.6898  (FQ wins if sll > -8.6898)")


if __name__ == "__main__":
    main()
