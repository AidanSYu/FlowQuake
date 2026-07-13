"""Out-of-sample skill-over-time decomposition.

The per-event log-likelihoods are already filtration-respecting: every forecast
conditions only on prior events (the SSM and last-K features are causal), and
both models are evaluated causally inside the held-out decade. Because the same
test windows informed development/model selection, this script should be cited
as a stability decomposition, not as a registered prospective forecast: it shows
whether the win is stable through the whole out-of-sample period or an average
driven by one burst (e.g. an aftershock sequence).

Per region and per time-bin we report, paired on matched events:
  dT  = mean(FQ.tll - ETAS.TLL)   temporal information gain (nats/event)
  dS  = mean(FQ.sll - ETAS.SLL)   spatial
  dTot= dT + dS
with a stationary block-bootstrap CI on the overall gain (blocks preserve the
aftershock autocorrelation that a naive per-event bootstrap would break).

Run: python scripts/prospective_eval.py            # all regions, best deployable FQ
     python scripts/prospective_eval.py --bin 180D
"""
import argparse, json
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.stats import stationary_block_bootstrap_ci

ER = Path("reference/Experiments/ETAS")
REGIONS = {
    "California": dict(etas="output_data_ComCat_25", regime="transform",  fq="runs/n1_density/per_event_test.csv"),
    "Japan":      dict(etas="output_data_Japan_25",  regime="subduction", fq="runs/japan_n1/per_event_test.csv"),
    "Chile":      dict(etas="output_data_Chile_25",  regime="subduction", fq="runs/chile_n1/per_event_test.csv"),
    "Greece":     dict(etas="output_data_Greece_25", regime="extension",  fq="runs/greece_n1/per_event_test.csv",
                       fq_transfer="runs/greece_fewshot/per_event_test.csv"),
    "Iran":       dict(etas="output_data_Iran_25",   regime="collision",  fq="runs/iran_n1/per_event_test.csv",
                       fq_transfer="runs/iran_fewshot/per_event_test.csv"),
}


def etas_pe(d):
    f = ER / d / "augmented_catalog.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, parse_dates=["time"]).dropna(subset=["TLL", "SLL"])[["time", "TLL", "SLL"]]


def analyze(fq_csv, epe, bin_):
    if not Path(fq_csv).exists() or epe is None:
        return None
    m = pd.read_csv(fq_csv, parse_dates=["time"]).merge(epe, on="time", how="inner")
    if not len(m):
        return None
    m = m.sort_values("time").reset_index(drop=True)
    m["dT"] = m["tll"] - m["TLL"]; m["dS"] = m["sll"] - m["SLL"]; m["dTot"] = m["dT"] + m["dS"]
    g = m.set_index("time").groupby(pd.Grouper(freq=bin_))
    series = [{"t": str(t.date()), "n": int(len(grp)),
               "dT": float(grp["dT"].mean()), "dS": float(grp["dS"].mean()),
               "dTot": float(grp["dTot"].mean())}
              for t, grp in g if len(grp)]
    loT, hiT = stationary_block_bootstrap_ci(m["dT"].values)
    loTot, hiTot = stationary_block_bootstrap_ci(m["dTot"].values, seed=1)
    frac_pos = float(np.mean([s["dT"] > 0 for s in series]))
    return {"n": int(len(m)),
            "dT": float(m["dT"].mean()), "dT_ci": [loT, hiT],
            "dS": float(m["dS"].mean()),
            "dTot": float(m["dTot"].mean()), "dTot_ci": [loTot, hiTot],
            "bins_dT_positive_frac": frac_pos, "series": series}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="90D", help="pandas offset for time bins (e.g. 90D, 180D, Q)")
    ap.add_argument("--out", default="runs/prospective.json")
    args = ap.parse_args()

    res = {}
    print(f"{'region':12}{'model':14}{'n':>7}{'dT':>9}{'dT 95% CI':>20}{'dTot':>9}{'%bins+':>8}")
    for reg, c in REGIONS.items():
        epe = etas_pe(c["etas"])
        res[reg] = {"regime": c["regime"]}
        for label, key in [("native", "fq"), ("transfer", "fq_transfer")]:
            if key not in c:
                continue
            a = analyze(c[key], epe, args.bin)
            if a is None:
                continue
            res[reg][label] = a
            ci = f"[{a['dT_ci'][0]:+.3f},{a['dT_ci'][1]:+.3f}]"
            sig = "*" if a["dT_ci"][0] > 0 else (" " if a["dT_ci"][1] > 0 else "x")
            print(f"{reg:12}{label:14}{a['n']:>7}{a['dT']:>+9.3f}{ci:>20}{a['dTot']:>+9.3f}"
                  f"{a['bins_dT_positive_frac']*100:>7.0f}% {sig}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  (* = temporal gain CI strictly > 0)")


if __name__ == "__main__":
    main()
