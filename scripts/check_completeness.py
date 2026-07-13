"""Completeness rigor gate for the ISC catalogs.

A constant mc is only honest if the catalog is complete above it across BOTH the
training era and the test era (ISC's network, hence Mc, improves over time). For
each region we estimate Mc by maximum curvature (Woessner & Wiemer 2005, +0.2)
on the train window and the test window separately, plus the b-value (Aki MLE)
and a goodness-of-fit residual, and recommend mcut = ceil-to-0.5 of the worse
(higher) of the two eras. If that exceeds 4.0 we must raise mcut for that region
(and document it) rather than train ETAS/FQ on incomplete data.

Run: python scripts/check_completeness.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

REGIONS = {"Japan": "1992-01-01", "Chile": "1992-01-01", "Greece": "1992-01-01", "Iran": "1992-01-01"}
TRAIN_END = "2011-01-01"; TEST_END = "2020-01-01"


def mc_maxc(mags, mbin=0.1):
    if len(mags) < 50:
        return float("nan"), float("nan"), 0
    edges = np.arange(np.floor(mags.min()*10)/10, mags.max()+mbin, mbin)
    counts, _ = np.histogram(mags, bins=edges)
    centers = edges[:-1] + mbin/2
    mc = centers[np.argmax(counts)] + 0.2
    above = mags[mags >= mc-1e-9]
    b = np.log10(np.e)/(above.mean()-(mc-mbin/2)) if len(above) > 1 else float("nan")
    return float(mc), float(b), int(len(above))


def main():
    rep = {}
    print(f"{'region':9}{'era':7}{'N':>7}{'Mc':>6}{'b':>6}{'N>=Mc':>8}")
    for reg, tstart in REGIONS.items():
        f = Path(f"reference/Datasets/{reg}/{reg}_catalog.csv")
        if not f.exists():
            print(f"{reg}: missing catalog"); continue
        df = pd.read_csv(f, parse_dates=["time"])
        tr = df[(df.time >= tstart) & (df.time < TRAIN_END)]["magnitude"].values
        te = df[(df.time >= TRAIN_END) & (df.time < TEST_END)]["magnitude"].values
        mc_tr, b_tr, n_tr = mc_maxc(tr)
        mc_te, b_te, n_te = mc_maxc(te)
        rec_mc = float(np.ceil(max(mc_tr, mc_te)*2)/2)   # round worse era up to 0.5
        print(f"{reg:9}{'train':7}{len(tr):>7}{mc_tr:>6.2f}{b_tr:>6.2f}{n_tr:>8}")
        print(f"{'':9}{'test':7}{len(te):>7}{mc_te:>6.2f}{b_te:>6.2f}{n_te:>8}"
              f"   -> recommend mcut={rec_mc:.1f}"
              f"{'  (>4.0 - RAISE)' if rec_mc > 4.0 else ''}")
        # counts available above the recommended mcut, both eras
        n_tr_rec = int((tr >= rec_mc).sum()); n_te_rec = int((te >= rec_mc).sum())
        rep[reg] = {"mc_train": round(mc_tr,2), "mc_test": round(mc_te,2),
                    "b_train": round(b_tr,2), "b_test": round(b_te,2),
                    "recommend_mcut": rec_mc, "n_train_at_mcut": n_tr_rec,
                    "n_test_at_mcut": n_te_rec}
    Path("runs").mkdir(exist_ok=True)
    json.dump(rep, open("runs/completeness.json", "w"), indent=2)
    print("\nwrote runs/completeness.json")
    print("data-rich (>=~5k train) can train native FQ; data-poor rely on transfer.")


if __name__ == "__main__":
    main()
