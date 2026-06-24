"""Where does the ComCat spatial gap (FQ sll -9.05 vs ETAS -8.69) live?

Pairs FQ per-event sll with ETAS per-event SLL and decomposes the gap by
physically meaningful regimes, so the next spatial improvement is aimed, not
guessed. Two axes:

  1. ETAS's OWN background/triggered split, from the inversion columns:
     bg_frac = lambd_star / lambd  (lambd_star = background intensity mu(x,y),
     lambd = total). High bg_frac -> ETAS thinks this event is background.
  2. Geometry reconstructed from the catalog: distance to the nearest of the
     last-K events, time since the last event, magnitude of the nearest recent
     large event.

For each bin reports n, mean per-event gap (ETAS - FQ; +ve = ETAS wins), and
TOTAL nats = mean_gap * n / N_test (its contribution to the -0.364 headline).

Run (CPU): python scripts/spatial_gap_decomp.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
ETAS_DIR = ROOT / "reference/Experiments/ETAS/output_data_ComCat_25"
FQ_CSV = ROOT / "runs/n1_density/per_event_test.csv"
LASTK = 64
BIG_MAG_MIN = 4.5
BIG_WINDOW_DAYS = 730.0
SECONDS_PER_DAY = 86400.0


def main():
    fq = pd.read_csv(FQ_CSV, parse_dates=["time"])
    aug = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"])
    aug = aug.dropna(subset=["SLL"]).copy()

    m = fq.merge(
        aug[["time", "x", "y", "magnitude", "SLL", "lambd", "lambd_star"]],
        on="time", how="inner",
    )
    m = m.rename(columns={"sll": "fq_sll", "SLL": "etas_sll"})
    N = len(m)
    print(f"matched {N} test events")
    print(f"FQ sll  {m.fq_sll.mean():.4f}   ETAS sll {m.etas_sll.mean():.4f}   "
          f"gap {m.etas_sll.mean() - m.fq_sll.mean():+.4f}")

    m["gap"] = m["etas_sll"] - m["fq_sll"]          # +ve = ETAS wins
    m["bg_frac"] = (m["lambd_star"] / m["lambd"]).clip(0, 1)

    # --- reconstruct geometry over the FULL catalog (need pre-test history) ---
    full = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"])
    full = full.sort_values("time").reset_index(drop=True)
    fx, fy = full["x"].to_numpy(), full["y"].to_numpy()
    fmag = full["magnitude"].to_numpy()
    ft = (full["time"] - full["time"].iloc[0]).dt.total_seconds().to_numpy() / SECONDS_PER_DAY

    test_pos = full["time"].isin(set(m["time"])).to_numpy()
    idxs = np.flatnonzero(test_pos)

    nn_dist = np.empty(len(idxs))     # dist to nearest of last-K
    last_dist = np.empty(len(idxs))   # dist to immediately preceding event
    dt_last = np.empty(len(idxs))     # days since preceding event
    nn_big_dist = np.full(len(idxs), np.nan)  # dist to nearest recent M>=4.5
    for j, i in enumerate(idxs):
        lo = max(0, i - LASTK)
        px, py = fx[lo:i], fy[lo:i]
        d = np.sqrt((px - fx[i]) ** 2 + (py - fy[i]) ** 2)
        nn_dist[j] = d.min() if len(d) else np.nan
        last_dist[j] = np.sqrt((fx[i-1]-fx[i])**2 + (fy[i-1]-fy[i])**2) if i > 0 else np.nan
        dt_last[j] = ft[i] - ft[i-1] if i > 0 else np.nan
        bw = (ft[i] - ft[:i] < BIG_WINDOW_DAYS) & (fmag[:i] >= BIG_MAG_MIN)
        if bw.any():
            db = np.sqrt((fx[:i][bw]-fx[i])**2 + (fy[:i][bw]-fy[i])**2)
            nn_big_dist[j] = db.min()

    g = pd.DataFrame({"time": full["time"].to_numpy()[idxs]})
    g["nn_dist"], g["last_dist"], g["dt_last"], g["nn_big_dist"] = (
        nn_dist, last_dist, dt_last, nn_big_dist)
    m = m.merge(g, on="time", how="left")

    def report(title, col, edges, labels):
        print(f"\n=== gap by {title} ===")
        print(f"{'bin':>22} {'n':>6} {'mean_gap':>9} {'tot_nats':>9} "
              f"{'fq_sll':>8} {'etas':>8}")
        b = pd.cut(m[col], edges, labels=labels, include_lowest=True)
        for lab in labels:
            s = m[b == lab]
            if not len(s):
                continue
            print(f"{lab:>22} {len(s):>6} {s.gap.mean():>9.4f} "
                  f"{s.gap.sum()/N:>9.4f} {s.fq_sll.mean():>8.3f} "
                  f"{s.etas_sll.mean():>8.3f}")

    report("ETAS background fraction", "bg_frac",
           [-0.01, 0.05, 0.2, 0.5, 0.8, 1.01],
           ["<.05 (triggered)", ".05-.2", ".2-.5", ".5-.8", ">.8 (background)"])
    report("dist to nearest last-K (km)", "nn_dist",
           [-0.01, 0.5, 2, 5, 15, 50, 1e9],
           ["<0.5", "0.5-2", "2-5", "5-15", "15-50", ">50"])
    report("dist to preceding event (km)", "last_dist",
           [-0.01, 0.5, 2, 5, 15, 50, 1e9],
           ["<0.5", "0.5-2", "2-5", "5-15", "15-50", ">50"])
    report("days since last event", "dt_last",
           [-1e-9, 1e-3, 1e-2, 0.1, 1, 10, 1e9],
           ["<1e-3", "1e-3-1e-2", "0.01-0.1", "0.1-1", "1-10", ">10"])
    report("dist to recent M>=4.5 (km)", "nn_big_dist",
           [-0.01, 2, 5, 15, 50, 1e9],
           ["<2", "2-5", "5-15", "15-50", ">50"])

    # worst single events (largest ETAS-FQ gap) — what is FQ missing badly?
    print("\n=== 12 worst events (ETAS - FQ) ===")
    w = m.nlargest(12, "gap")[
        ["time", "magnitude", "gap", "fq_sll", "etas_sll", "bg_frac",
         "nn_dist", "dt_last", "nn_big_dist"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(w.to_string(index=False))

    out = ROOT / "runs/n1_density/spatial_gap_decomp.json"
    summ = {
        "n": N, "fq_sll": float(m.fq_sll.mean()),
        "etas_sll": float(m.etas_sll.mean()), "gap": float(m.gap.mean()),
        "frac_events_etas_wins": float((m.gap > 0).mean()),
        "gap_from_background_bg>0.5": float(m[m.bg_frac > 0.5].gap.sum() / N),
        "gap_from_triggered_bg<0.5": float(m[m.bg_frac <= 0.5].gap.sum() / N),
    }
    json.dump(summ, open(out, "w"), indent=2)
    print(f"\nwrote {out}\n{json.dumps(summ, indent=2)}")


if __name__ == "__main__":
    main()
