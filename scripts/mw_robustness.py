"""Mw-homogenization robustness + the data-efficiency mechanism behind the
temporal win.

A reviewer concern is that FlowQuake's temporal advantage could be an artifact of
mixed magnitude types (ML, Md, Mw). We test this with full re-runs (re-invert
ETAS + retrain FlowQuake) on Mw-homogenized and density-matched catalogs, and
report the paired per-event temporal gain dT = tll_FQ - TLL_ETAS with a
stationary block-bootstrap CI.

Findings (see runs/mw_robustness.json):
  * California (ComCat). ML approx Mw in California (Clinton et al. 2006; Bakun
    1984) and NCSN Md is calibrated to ML (Oppenheimer et al. 1992), so ComCat
    magnitudes are already Mw-equivalent -- the treatment used by EarthquakeNPP,
    UCERF3 and USGS operational ETAS. The production model (trained on the dense
    mc 2.5 catalog) BEATS ETAS on the M>=3.0 subset, where magnitudes are
    unambiguously Mw -> the headline win is Mw-robust.
  * The advantage is data-efficiency, not magnitude: a model RE-TRAINED on only
    the sparse M>=3.0 data LOSES on those same events, while the dense-trained
    production model WINS on them. FlowQuake exploits the dense low-magnitude
    record that ETAS's parametric form underuses.
  * Italy (INGV). ML << Mw and the catalog is Md-dominated, so Mw homogenization
    (Gasperini et al. 2013) is mandatory and COMPRESSES the catalog ~2x above
    completeness; the win erodes -- a density effect (a density-matched ML
    control at mc 2.8 also only ties), not a magnitude-scale artifact.

Run: python scripts/mw_robustness.py   -> runs/mw_robustness.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.stats import paired_gain_summary

ER = Path("reference/Experiments/ETAS")


def etas_pe(dirname):
    p = ER / dirname / "augmented_catalog.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["time"]).dropna(subset=["TLL", "SLL"])[
        ["time", "magnitude", "TLL", "SLL"]
    ]


def train_count(catalog, mc, train_start, test_start):
    d = pd.read_csv(catalog, parse_dates=["time"])
    return int(((d.magnitude >= mc) & (d.time >= train_start) & (d.time < test_start)).sum())


def compare(fq_csv, etas_dir, seed, mmin=None, train=None):
    epe = etas_pe(etas_dir)
    if epe is None or not Path(fq_csv).exists():
        return None
    fq = pd.read_csv(fq_csv, parse_dates=["time"])
    m = fq.merge(epe, on="time", how="inner").sort_values("time").reset_index(drop=True)
    if mmin is not None:
        m = m[m["magnitude"] >= mmin]
    if not len(m):
        return None
    out = {"n": int(len(m)), "fq_tll": float(m.tll.mean()), "etas_tll": float(m.TLL.mean())}
    for key, col in [("dT", m.tll - m.TLL), ("dS", m.sll - m.SLL),
                     ("dTot", (m.tll + m.sll) - (m.TLL + m.SLL))]:
        g = paired_gain_summary(col.values, seed=seed).asdict()
        out[key] = {"mean": round(g["mean"], 4), "ci": [round(g["ci"][0], 4), round(g["ci"][1], 4)],
                    "decision": g["decision"]}
    if train:
        out["train_events"] = train_count(*train)
    return out


def main():
    CC = "reference/Datasets/ComCat/ComCat_catalog.csv"
    IT = "reference/Datasets/Italy/Italy_catalog.csv"
    IM = "reference/Datasets/Italy_Mw/Italy_Mw_catalog.csv"
    res = {
        "california": {
            "comcat_mc25_headline": compare("runs/n1_density/per_event_test.csv", "output_data_ComCat_25",
                                            seed=1, train=(CC, 2.5, "1981-01-01", "2007-01-01")),
            "comcat_mc25_production_on_Mge3": compare("runs/n1_density/per_event_test.csv", "output_data_ComCat_25",
                                                      seed=2, mmin=3.0),
            "comcat_mc30_retrained": compare("runs/comcat_mc30/per_event_test.csv", "output_data_ComCat_30",
                                             seed=3, train=(CC, 3.0, "1981-01-01", "2007-01-01")),
        },
        "italy": {
            "ml_mc25_headline": compare("runs/italy_n1/per_event_test.csv", "output_data_Italy_25",
                                        seed=4, train=(IT, 2.5, "1994-01-01", "2011-01-01")),
            "ml_mc28_density_control": compare("runs/italy_mc28_n1/per_event_test.csv", "output_data_Italy_28",
                                               seed=5, train=(IT, 2.8, "1994-01-01", "2011-01-01")),
            "mw_uniform_mc26": compare("runs/italy_mw_n1/per_event_test.csv", "output_data_Italy_Mw_25",
                                       seed=6, train=(IM, 2.6, "1994-01-01", "2011-01-01")),
        },
        "interpretation": (
            "California ML~Mw so ComCat is already Mw-equivalent; the production model beats ETAS on the "
            "unambiguously-Mw M>=3 subset (Mw-robust). The mc3.0-retrained model loses on the same events, "
            "showing the edge is data-efficiency (needs the dense low-mag record), not magnitude. Italy's "
            "Md-dominated catalog must be Mw-homogenized, which halves it above completeness; the win erodes "
            "as a density effect (density-matched ML mc2.8 control also only ties)."
        ),
    }
    Path("runs").mkdir(exist_ok=True)
    json.dump(res, open("runs/mw_robustness.json", "w"), indent=2)
    print("wrote runs/mw_robustness.json\n")
    for region, blk in [("CALIFORNIA", res["california"]), ("ITALY", res["italy"])]:
        print(region)
        for name, r in blk.items():
            if r is None:
                print(f"  {name}: (missing)"); continue
            tr = f" train={r.get('train_events','-')}" if 'train_events' in r else ""
            print(f"  {name:32} n={r['n']:>6}{tr}  dT={r['dT']['mean']:+.4f} "
                  f"CI[{r['dT']['ci'][0]:+.4f},{r['dT']['ci'][1]:+.4f}] {r['dT']['decision']}")


if __name__ == "__main__":
    main()
