"""Head-to-head: FlowQuake transfer variants vs Japan-fitted ETAS on the Japan
test window (2011-2020, incl. Tohoku). Aggregate scoreboard + paired per-event
gains (mean, stderr, win-rate) merged on event time, exactly like the ComCat
evaluator. nll = -(tll+sll), matching EarthquakeNPP (magnitude reported apart).

Run after invert_etas.py + predict_etas.py Japan_25 finish.
Run: python scripts/japan_compare.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ETAS = Path("reference/Experiments/ETAS/output_data_Japan_25")
VARIANTS = {
    "FlowQuake CA zero-shot": "runs/transfer_japan_per_event.csv",
    "FlowQuake CA few-shot(2k)": "runs/japan_fewshot/per_event_test.csv",
    "FlowQuake Japan-native": "runs/japan_n1/per_event_test.csv",
}


def main():
    aug_f = ETAS / "augmented_catalog.csv"
    ll_f = ETAS / "ll_scores.json"
    if not aug_f.exists():
        print(f"ETAS output not ready yet: {aug_f} missing"); return
    aug = pd.read_csv(aug_f, parse_dates=["time"]).dropna(subset=["TLL", "SLL"])
    etas_t, etas_s = aug["TLL"].mean(), aug["SLL"].mean()
    print(f"ETAS Japan per-event set: {len(aug)} events")
    if ll_f.exists():
        ll = json.load(open(ll_f)); print("ll_scores.json:", ll.get("ETAS", ll))

    print(f"\n{'model':28}{'tll':>9}{'sll':>10}{'nll':>9}   vs ETAS (paired, matched events)")
    print(f"{'ETAS (Japan-fit)':28}{etas_t:>9.4f}{etas_s:>10.4f}{-(etas_t+etas_s):>9.4f}")
    for name, path in VARIANTS.items():
        p = Path(path)
        if not p.exists():
            print(f"{name:28}  (missing {path})"); continue
        fq = pd.read_csv(p, parse_dates=["time"])
        m = fq.merge(aug[["time", "TLL", "SLL"]], on="time", how="inner")
        if not len(m):
            print(f"{name:28}  (no matched events)"); continue
        t, s = m["tll"].mean(), m["sll"].mean()
        dt, ds, dj = m["tll"]-m["TLL"], m["sll"]-m["SLL"], (m["tll"]+m["sll"])-(m["TLL"]+m["SLL"])
        def stat(d): return d.mean(), d.std(ddof=1)/np.sqrt(len(d)), (d > 0).mean()
        gt, et, wt = stat(dt); gs, es, ws = stat(ds); gj, ej, wj = stat(dj)
        print(f"{name:28}{t:>9.4f}{s:>10.4f}{-(t+s):>9.4f}   n={len(m)}")
        print(f"{'  d temporal':28}{gt:>+9.4f}  +-{et:.4f}  win {wt:5.1%}")
        print(f"{'  d spatial':28}{gs:>+9.4f}  +-{es:.4f}  win {ws:5.1%}")
        print(f"{'  d TOTAL':28}{gj:>+9.4f}  +-{ej:.4f}  win {wj:5.1%}  "
              f"{'<<< BEATS ETAS' if gj > 0 else ''}")


if __name__ == "__main__":
    main()
