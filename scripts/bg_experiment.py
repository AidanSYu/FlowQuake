"""Test better spatial-background fields by re-scoring sll only (CPU, no ODE).

The spatial gap is a thin floor problem: ~11% of test events fall in holes of
the train-era fixed-bandwidth KDE and score sll < -14. ETAS's adaptive
background never collapses there. This swaps candidate background grids into the
existing ckpt_best and recomputes the closed-form spatial sll for all test
events (no temporal ODE, no GPU), reporting mean sll + paired gap vs ETAS.

Candidates: the baseline grid; adaptive variable-bandwidth smoothed seismicity
(Helmstetter et al. 2007 — k-th-NN bandwidth, the standard background method,
not ETAS-specific); plus floor / event-window variations.

Run (CPU): python scripts/bg_experiment.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from flowquake.config import Config
from flowquake.data import full_sequence_batch, load_catalog
from flowquake.train import make_model

ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")
CKPT = "runs/n1_density/ckpt_best.pt"


def grid_axes(stats, bin_km=2.0):
    nx = int(np.ceil((stats["bg_xmax"] - stats["bg_xmin"]) / bin_km))
    ny = int(np.ceil((stats["bg_ymax"] - stats["bg_ymin"]) / bin_km))
    return nx, ny


def to_cells(x, y, stats, nx, ny, bin_km=2.0):
    gx = np.clip(((x - stats["bg_xmin"]) / bin_km).astype(int), 0, nx - 1)
    gy = np.clip(((y - stats["bg_ymin"]) / bin_km).astype(int), 0, ny - 1)
    return gx, gy


def fixed_kde(x, y, stats, sigma_km=4.0, floor=0.02, bin_km=2.0):
    """The baseline-style fixed-bandwidth KDE (replicates data.py at sigma)."""
    nx, ny = grid_axes(stats, bin_km)
    gx, gy = to_cells(x, y, stats, nx, ny, bin_km)
    grid = np.zeros((nx, ny))
    np.add.at(grid, (gx, gy), 1.0)
    grid = gaussian_filter(grid, sigma=sigma_km / bin_km)
    grid = grid / (grid.sum() * bin_km ** 2 + 1e-12)
    grid = (1 - floor) * grid + floor / stats["bg_area"]
    return np.log(grid).astype(np.float32)


def adaptive_kde(x, y, stats, k=6, sigma_min=1.0, sigma_max=60.0, floor=0.02,
                 n_buckets=12, bin_km=2.0):
    """Helmstetter-style variable-bandwidth smoothed seismicity.

    Each event smoothed with sigma_i = clamp(distance to its k-th nearest
    neighbour). Dense (fault) regions stay sharp; isolated events spread wide,
    so off-fault density never collapses to the floor.
    """
    nx, ny = grid_axes(stats, bin_km)
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    dk, _ = tree.query(pts, k=min(k + 1, len(pts)))
    sig = np.clip(dk[:, -1], sigma_min, sigma_max)
    gx, gy = to_cells(x, y, stats, nx, ny, bin_km)

    edges = np.logspace(np.log10(sig.min()), np.log10(sig.max() + 1e-6),
                        n_buckets + 1)
    grid = np.zeros((nx, ny))
    for b in range(n_buckets):
        sel = (sig >= edges[b]) & (sig <= edges[b + 1] if b == n_buckets - 1
                                   else sig < edges[b + 1])
        if not sel.any():
            continue
        sub = np.zeros((nx, ny))
        np.add.at(sub, (gx[sel], gy[sel]), 1.0)
        s_mid = np.sqrt(edges[b] * edges[b + 1])
        grid += gaussian_filter(sub, sigma=s_mid / bin_km)
    grid = grid / (grid.sum() * bin_km ** 2 + 1e-12)
    grid = (1 - floor) * grid + floor / stats["bg_area"]
    return np.log(grid).astype(np.float32)


def score_sll(model, tokens, mask, lastk, raw_next, grid, stats):
    """Closed-form spatial sll for all masked test events under a bg grid."""
    from flowquake.model import SAFE_TOKEN_DIMS
    model.stats["kde_log_grid"] = grid
    model._kde_t = None  # bust the cached tensor
    cond = tokens[0][mask[0]][:, SAFE_TOKEN_DIMS]
    comp_xy, comp_feats = model._comp_inputs(lastk[0][mask[0]])
    s_tgt = raw_next[0][mask[0]][:, :2]
    out = []
    for i in range(0, cond.shape[0], 8192):
        sl = slice(i, i + 8192)
        lk = model._log_kde(s_tgt[sl])
        out.append(model.head_s.log_prob(
            s_tgt[sl], comp_xy[sl], comp_feats[sl], cond[sl],
            stats["bg_area"], log_kde_at_s=lk))
    return torch.cat(out).numpy()


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

    # event sets for the background field
    times = cat.times
    x, y = cat.raw[:, 1].numpy(), cat.raw[:, 2].numpy()
    fit_train = np.asarray(times < cfg.data.val_start)     # baseline window
    fit_pretest = np.asarray(times < cfg.data.test_start)  # +val, still causal

    # ETAS per-event SLL for paired gap
    aug = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"])
    aug = aug.dropna(subset=["SLL"])
    mask_np = np.zeros(cat.n_events, dtype=bool)
    nxt = mask[0].numpy()
    mask_np[1:] = nxt[:-1]
    ev_times = cat.times[mask_np]

    candidates = {
        "baseline (committed grid)": np.asarray(stats["kde_log_grid"]),
        "fixed s8 floor2 train": fixed_kde(x[fit_train], y[fit_train], stats, 8, 0.02),
        "fixed s8 floor5 pretest": fixed_kde(x[fit_pretest], y[fit_pretest], stats, 8, 0.05),
        "adapt k6 smin1 floor2 train": adaptive_kde(x[fit_train], y[fit_train], stats, 6, 1, floor=0.02),
        "adapt k6 smin1 floor2 pretest": adaptive_kde(x[fit_pretest], y[fit_pretest], stats, 6, 1, floor=0.02),
        "adapt k10 smin1.5 floor2 pretest": adaptive_kde(x[fit_pretest], y[fit_pretest], stats, 10, 1.5, floor=0.02),
        "adapt k6 smin1 floor5 pretest": adaptive_kde(x[fit_pretest], y[fit_pretest], stats, 6, 1, floor=0.05),
        "adapt k4 smin0.75 floor3 pretest": adaptive_kde(x[fit_pretest], y[fit_pretest], stats, 4, 0.75, floor=0.03),
    }

    print(f"{'background':>38} {'sll':>9} {'gapETAS':>8} {'win%':>6} "
          f"{'<-14':>6} {'<-12':>6}")
    base_grid = np.asarray(stats["kde_log_grid"]).copy()
    rows = {}
    for name, grid in candidates.items():
        sll = score_sll(model, tokens, mask, lastk, raw_next, grid, stats)
        ours = pd.DataFrame({"time": ev_times, "sll": sll})
        mg = ours.merge(aug[["time", "SLL"]], on="time", how="inner")
        gap = (mg["SLL"] - mg["sll"]).mean()  # +ve = ETAS wins
        win = (mg["sll"] > mg["SLL"]).mean()
        rows[name] = float(sll.mean())
        print(f"{name:>38} {sll.mean():>9.4f} {gap:>8.4f} {100*win:>5.1f} "
              f"{(sll<-14).sum():>6} {(sll<-12).sum():>6}")
    model.stats["kde_log_grid"] = base_grid
    print(f"\nETAS sll -8.6898  (FQ wins if sll > -8.6898)")
    json.dump(rows, open("runs/n1_density/bg_experiment.json", "w"), indent=2)


if __name__ == "__main__":
    main()
