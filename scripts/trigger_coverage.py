"""Background problem or triggering-coverage problem?

For each test event find the nearest PRIOR event in the whole catalog (not just
FQ's last-64 window). If the catastrophically-scored events are spatially close
to an older/moderate prior event (small dist, large rank-back), then ETAS wins
by integrating triggering over all history while FQ mis-files them as background
-> the fix is expanding the triggering component set, not the background.

Run (CPU): python scripts/trigger_coverage.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ETAS_DIR = Path("reference/Experiments/ETAS/output_data_ComCat_25")
SECONDS_PER_DAY = 86400.0
LASTK = 64


def main():
    fq = pd.read_csv("runs/n1_density/per_event_test.csv", parse_dates=["time"]).rename(columns={"sll": "fq_sll"})
    full = pd.read_csv(ETAS_DIR / "augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    fx, fy = full["x"].to_numpy(), full["y"].to_numpy()
    fm = full["magnitude"].to_numpy()
    ft = (full["time"] - full["time"].iloc[0]).dt.total_seconds().to_numpy() / SECONDS_PER_DAY
    is_test = full["time"].isin(set(fq["time"])).to_numpy()
    idxs = np.flatnonzero(is_test)

    tree = cKDTree(np.column_stack([fx, fy]))
    K = 400
    dist, nbr = tree.query(np.column_stack([fx[idxs], fy[idxs]]), k=K)

    nn_all = np.full(len(idxs), np.nan)     # nearest prior event distance (all history)
    nn_rank = np.full(len(idxs), -1)        # how many events back it is
    nn_dt = np.full(len(idxs), np.nan)      # days since it
    nn_mag = np.full(len(idxs), np.nan)
    in_lastk = np.zeros(len(idxs), bool)    # is that nn within FQ's last-64?
    for j, i in enumerate(idxs):
        cand = nbr[j]
        prior = cand[(cand < i)]            # earlier in time (sorted by time)
        if len(prior) == 0:
            continue
        # nbr is sorted by distance; first prior is nearest prior spatial neighbour
        p = prior[0]
        nn_all[j] = np.hypot(fx[p] - fx[i], fy[p] - fy[i])
        nn_rank[j] = i - p
        nn_dt[j] = ft[i] - ft[p]
        nn_mag[j] = fm[p]
        in_lastk[j] = (i - p) <= LASTK

    g = pd.DataFrame({"time": full["time"].to_numpy()[idxs], "nn_all": nn_all,
                      "nn_rank": nn_rank, "nn_dt": nn_dt, "nn_mag": nn_mag,
                      "in_lastk": in_lastk})
    aug = full[["time", "SLL"]]
    m = fq.merge(g, on="time").merge(aug, on="time").rename(columns={"SLL": "etas_sll"})
    m["gap"] = m["etas_sll"] - m["fq_sll"]
    N = len(m)
    print(f"matched {N}; mean gap {m.gap.mean():+.4f}\n")

    # of the events with a CLOSE prior neighbour, how many are outside last-64?
    close = m[m.nn_all < 5]
    print(f"events with a prior neighbour < 5km: {len(close)} ({100*len(close)/N:.0f}%)")
    print(f"  of those, nearest prior is OUTSIDE FQ last-64: "
          f"{(~close.in_lastk).sum()} ({100*(~close.in_lastk).mean():.0f}%)")
    print(f"  their mean gap: {close.gap.mean():+.4f}  "
          f"(outside-64 subset: {close[~close.in_lastk].gap.mean():+.4f})\n")

    def report(title, col, edges, labels):
        print(f"=== gap by {title} ===")
        print(f"{'bin':>16} {'n':>6} {'mean_gap':>9} {'tot_nats':>9} {'fq':>8} {'etas':>8} {'%out64':>7}")
        b = pd.cut(m[col], edges, labels=labels, include_lowest=True)
        for lab in labels:
            s = m[b == lab]
            if len(s):
                print(f"{lab:>16} {len(s):>6} {s.gap.mean():>9.4f} {s.gap.sum()/N:>9.4f} "
                      f"{s.fq_sll.mean():>8.3f} {s.etas_sll.mean():>8.3f} {100*(~s.in_lastk).mean():>6.0f}%")
        print()

    report("nearest prior nbr dist (km, ALL history)", "nn_all",
           [-.01, 0.5, 2, 5, 15, 50, 1e9], ["<0.5", "0.5-2", "2-5", "5-15", "15-50", ">50"])
    report("rank-back of nearest prior nbr", "nn_rank",
           [0, 64, 256, 1024, 4096, 1e9], ["1-64", "64-256", "256-1k", "1k-4k", ">4k"])
    report("days since nearest prior nbr", "nn_dt",
           [-1e-9, 0.1, 1, 10, 100, 1e9], ["<0.1", "0.1-1", "1-10", "10-100", ">100"])

    # worst events: is the nearest prior neighbour close but old?
    print("=== 12 worst events: nearest prior neighbour ===")
    w = m.nlargest(12, "gap")[["magnitude", "gap", "fq_sll", "etas_sll", "nn_all", "nn_rank", "nn_dt", "nn_mag", "in_lastk"]]
    with pd.option_context("display.width", 200):
        print(w.to_string(index=False))


if __name__ == "__main__":
    main()
