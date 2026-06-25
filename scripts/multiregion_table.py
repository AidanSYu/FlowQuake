"""Master cross-regime generalization table: FlowQuake (native / CA-zero-shot /
CA-few-shot) vs region-fitted ETAS, across all regions. nll = -(tll+sll),
matching EarthquakeNPP (magnitude apart). Paired per-event vs ETAS where the
ETAS baseline exists; tolerant of pieces still computing.

Regions: ComCat (California, transform) is the transfer SOURCE; Japan & Chile
(subduction), Greece (extension), Iran (collision) are targets.

Run: python scripts/multiregion_table.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ETAS_ROOT = Path("reference/Experiments/ETAS")
# region -> (etas_output_dir, native_run_dir, transfer_tag, regime)
REGIONS = {
    "California": ("output_data_ComCat_25", "n1_density", None, "transform (SOURCE)"),
    "Japan":      ("output_data_Japan_25",  "japan_n1",   "Japan",  "subduction"),
    "Chile":      ("output_data_Chile_25",  "chile_n1",   "Chile",  "subduction"),
    "Greece":     ("output_data_Greece_25", "greece_n1",  "Greece", "extension"),
    "Iran":       ("output_data_Iran_25",   "iran_n1",    "Iran",   "collision"),
}


def etas_scores(d):
    f = ETAS_ROOT / d / "ll_scores.json"
    if not f.exists():
        return None
    j = json.load(open(f)); return j["ETAS"]


def etas_perevent(d):
    f = ETAS_ROOT / d / "augmented_catalog.csv"
    if not f.exists():
        return None
    a = pd.read_csv(f, parse_dates=["time"]).dropna(subset=["TLL", "SLL"])
    return a[["time", "TLL", "SLL"]]


def fq_eval(run):
    f = Path("runs") / run / "eval_test.json"
    if not f.exists():
        return None
    r = json.load(open(f)); return r["tll"], r["sll"]


def paired(per_event_csv, etas_pe):
    p = Path(per_event_csv)
    if not p.exists() or etas_pe is None:
        return None
    fq = pd.read_csv(p, parse_dates=["time"])
    m = fq.merge(etas_pe, on="time", how="inner")
    if not len(m):
        return None
    dt = m["tll"] - m["TLL"]; ds = m["sll"] - m["SLL"]
    return dict(n=len(m), tll=m["tll"].mean(), sll=m["sll"].mean(),
                dT=dt.mean(), dT_se=dt.std(ddof=1)/np.sqrt(len(m)),
                dS=ds.mean(), dTot=(dt+ds).mean())


def main():
    print(f"{'region':12}{'regime':20}{'model':16}{'tll':>8}{'sll':>9}{'nll':>8}"
          f"{'dTemp':>9}{'dTotal':>9}")
    master = {}
    for reg, (ed, run, tag, regime) in REGIONS.items():
        es = etas_scores(ed); epe = etas_perevent(ed)
        master[reg] = {"regime": regime}
        if es:
            print(f"{reg:12}{regime:20}{'ETAS(native)':16}{es['tll']:>8.3f}{es['sll']:>9.3f}"
                  f"{es['nll']:>8.3f}{'':>9}{'':>9}")
            master[reg]["ETAS"] = es
        else:
            print(f"{reg:12}{regime:20}{'ETAS(native)':16}{'  ...fitting...':>34}")
        # native FQ
        nv = fq_eval(run)
        if nv:
            t, s = nv; pr = paired(f"runs/{run}/per_event_test.csv", epe)
            dT = f"{pr['dT']:>+9.3f}" if pr else f"{'':>9}"; dTot = f"{pr['dTot']:>+9.3f}" if pr else f"{'':>9}"
            print(f"{'':12}{'':20}{'FQ native':16}{t:>8.3f}{s:>9.3f}{-(t+s):>8.3f}{dT}{dTot}")
            master[reg]["native"] = dict(tll=t, sll=s, paired=pr)
        # zero-shot (CA -> region)
        if tag:
            pr = paired(f"runs/transfer_{tag}_per_event.csv", epe)
            zj = Path(f"runs/transfer_{tag}.json")
            if zj.exists():
                z = json.load(open(zj)); t, s = z["tll"], z["sll"]
                dT = f"{pr['dT']:>+9.3f}" if pr else f"{'':>9}"; dTot = f"{pr['dTot']:>+9.3f}" if pr else f"{'':>9}"
                print(f"{'':12}{'':20}{'FQ CA zero-shot':16}{t:>8.3f}{s:>9.3f}{-(t+s):>8.3f}{dT}{dTot}")
                master[reg]["zeroshot"] = dict(tll=t, sll=s, paired=pr)
        print()
    json.dump(master, open("runs/multiregion_master.json", "w"), indent=2, default=float)
    print("wrote runs/multiregion_master.json")


if __name__ == "__main__":
    main()
