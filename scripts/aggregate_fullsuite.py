"""Regenerate runs/fullsuite_summary.json — the §4.1 headline table source.

3-seed means (and std) of per-event tll/sll/nll for the production N1 model on
each EarthquakeNPP California catalog, plus the ETAS baseline read straight from
each run's eval_test.json. Makes the headline table reproducible from committed
per-seed runs (previously fullsuite_summary.json was hand-assembled).

The seed-run directories are listed explicitly because the ComCat runs predate
the DS_n1[_sSEED] naming convention used for the other four datasets.

Run: python scripts/aggregate_fullsuite.py
"""

import json
from pathlib import Path

import numpy as np

RUNS = Path("runs")

# dataset -> the three seed run directories (each holds eval_test.json)
SEED_DIRS = {
    "ComCat_25": ["n1_density", "n1_s1553", "n1_s1554"],
    "WHITE_06": ["WHITE_06_n1", "WHITE_06_n1_s1554", "WHITE_06_n1_s1555"],
    "SanJac_10": ["SanJac_10_n1", "SanJac_10_n1_s1554", "SanJac_10_n1_s1555"],
    "SaltonSea_10": ["SaltonSea_10_n1", "SaltonSea_10_n1_s1554", "SaltonSea_10_n1_s1555"],
    "SCEDC_20": ["SCEDC_20_n1", "SCEDC_20_n1_s1554", "SCEDC_20_n1_s1555"],
}


def main():
    summary = {}
    for ds, dirs in SEED_DIRS.items():
        evals = [json.load(open(RUNS / d / "eval_test.json")) for d in dirs]
        tll = np.array([e["tll"] for e in evals])
        sll = np.array([e["sll"] for e in evals])
        nll = np.array([e["nll"] for e in evals])
        etas = evals[0]["baselines"]["ETAS"]  # same baseline across seeds
        summary[ds] = {
            "tll": float(tll.mean()), "tll_sd": float(tll.std(ddof=1)),
            "sll": float(sll.mean()), "sll_sd": float(sll.std(ddof=1)),
            "nll": float(nll.mean()), "nll_sd": float(nll.std(ddof=1)),
            "etas_tll": etas["tll"], "etas_sll": etas["sll"], "etas_nll": etas["nll"],
            "n": len(evals), "seed_dirs": dirs,
        }
        d = summary[ds]
        print(f"{ds:>13}: FQ tll {d['tll']:.4f}+-{d['tll_sd']:.4f}  "
              f"ETAS {d['etas_tll']:.4f}  d{d['tll']-d['etas_tll']:+.4f}")
    json.dump(summary, open(RUNS / "fullsuite_summary.json", "w"), indent=2)
    print(f"\nwrote {RUNS / 'fullsuite_summary.json'}")


if __name__ == "__main__":
    main()
