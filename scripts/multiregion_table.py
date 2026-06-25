"""Master cross-regime generalization table: FlowQuake (native / transfer
zero-shot / few-shot) vs region-fitted ETAS, across regions spanning 4 tectonic
regimes. nll = -(tll+sll). Paired per-event vs ETAS where the baseline exists.
Transfer is within a completeness regime (mc2.5: CA<->Japan; mc4.0: Chile->rest).

Run: python scripts/multiregion_table.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ER = Path("reference/Experiments/ETAS")
# region -> dict(etas_dir, native_run, regime, mc, src=transfer source label)
REGIONS = {
    "California": dict(etas="output_data_ComCat_25", run="n1_density", regime="transform", mc=2.5, src=None),
    "Japan":      dict(etas="output_data_Japan_25",  run="japan_n1",  regime="subduction", mc=2.5, src="CA"),
    "Chile":      dict(etas="output_data_Chile_25",  run="chile_n1",  regime="subduction", mc=4.0, src=None),
    "Greece":     dict(etas="output_data_Greece_25", run="greece_n1", regime="extension",  mc=4.0, src="Chile"),
    "Iran":       dict(etas="output_data_Iran_25",   run="iran_n1",   regime="collision",  mc=4.0, src="Chile"),
}
# per-event score files for transfer (zero-shot) and few-shot, by region
ZS = {"Japan": "runs/transfer_Japan_per_event.csv", "Greece": "runs/transfer_Greece_per_event.csv",
      "Iran": "runs/transfer_Iran_per_event.csv"}
FS = {"Japan": "runs/japan_fewshot/per_event_test.csv", "Greece": "runs/greece_fewshot/per_event_test.csv",
      "Iran": "runs/iran_fewshot/per_event_test.csv"}


def etas_pe(d):
    f = ER / d / "augmented_catalog.csv"
    return pd.read_csv(f, parse_dates=["time"]).dropna(subset=["TLL", "SLL"])[["time", "TLL", "SLL"]] if f.exists() else None


def etas_agg(d):
    f = ER / d / "ll_scores.json"
    return json.load(open(f))["ETAS"] if f.exists() else None


def paired(csv, epe):
    if not Path(csv).exists() or epe is None:
        return None
    m = pd.read_csv(csv, parse_dates=["time"]).merge(epe, on="time", how="inner")
    if not len(m):
        return None
    dt, ds = m["tll"] - m["TLL"], m["sll"] - m["SLL"]
    return dict(n=len(m), tll=m["tll"].mean(), sll=m["sll"].mean(),
                dT=float(dt.mean()), dTse=float(dt.std(ddof=1)/np.sqrt(len(m))),
                dS=float(ds.mean()), dTot=float((dt+ds).mean()))


def row(label, tll, sll, pr):
    dT = f"{pr['dT']:>+8.3f}" if pr else " " * 8
    dTot = f"{pr['dTot']:>+8.3f}" if pr else " " * 8
    win = f"{'WIN' if pr and pr['dT'] > 0 else ''}" if pr else ""
    print(f"{'':12}{'':12}{label:18}{tll:>8.3f}{sll:>9.3f}{-(tll+sll):>8.3f}{dT}{dTot}  {win}")


def main():
    print(f"{'region':12}{'regime':12}{'model':18}{'tll':>8}{'sll':>9}{'nll':>8}{'dTemp':>8}{'dTot':>8}  (vs ETAS)")
    master = {}
    for reg, c in REGIONS.items():
        epe = etas_pe(c["etas"]); eag = etas_agg(c["etas"])
        print(f"\n{reg:12}{c['regime']:12}{'(mc %.1f)'%c['mc']:18}")
        master[reg] = dict(regime=c["regime"], mc=c["mc"])
        if eag:
            print(f"{'':12}{'':12}{'ETAS native':18}{eag['tll']:>8.3f}{eag['sll']:>9.3f}{eag['nll']:>8.3f}")
            master[reg]["ETAS"] = eag
        else:
            print(f"{'':12}{'':12}{'ETAS native':18}{'  ...fitting...':>25}")
        nf = Path("runs") / c["run"] / "eval_test.json"
        if nf.exists():
            r = json.load(open(nf)); pr = paired(f"runs/{c['run']}/per_event_test.csv", epe)
            row("FQ native", r["tll"], r["sll"], pr); master[reg]["native"] = dict(tll=r["tll"], sll=r["sll"], paired=pr)
        if reg in ZS and Path(ZS[reg]).exists():
            z = json.load(open(f"runs/transfer_{reg}.json")); pr = paired(ZS[reg], epe)
            row(f"FQ {c['src']}->zs", z["tll"], z["sll"], pr); master[reg]["zeroshot"] = dict(src=c["src"], tll=z["tll"], sll=z["sll"], paired=pr)
        if reg in FS and Path(FS[reg]).exists():
            fr = json.load(open(Path(FS[reg]).parent / "eval_test.json")); pr = paired(FS[reg], epe)
            row(f"FQ {c['src']}->few", fr["tll"], fr["sll"], pr); master[reg]["fewshot"] = dict(tll=fr["tll"], sll=fr["sll"], paired=pr)
    json.dump(master, open("runs/multiregion_master.json", "w"), indent=2, default=float)
    print("\nwrote runs/multiregion_master.json")


if __name__ == "__main__":
    main()
