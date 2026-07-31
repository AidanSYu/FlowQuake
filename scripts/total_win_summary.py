"""Paired TOTAL-likelihood comparison vs ETAS on ComCat_25: FlowQuake temporal
head (flow) + neural-ETAS spatial head, per event, block-bootstrap CIs.

Windows:
  test    : 2007-2020 benchmark test window
  forward : 2020-2026 out-of-time/pseudo-prospective window (frozen scoring)

Run: python scripts/total_win_summary.py -> runs/total_win.json
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.stats import paired_gain_summary, block_bootstrap_pvalue

ETAS_AUG = "reference/Experiments/ETAS/output_data_ComCat_25/augmented_catalog.csv"


def add_time_rank(df):
    out = df.sort_values("time").copy()
    out["_time_rank"] = out.groupby("time").cumcount()
    return out


def head_seed_csvs(head: str) -> list[Path]:
    """Every training seed of the neural-ETAS spatial head, sorted by seed.

    This used to be a hardcoded `per_event_full_s0.csv`, which made the headline
    total-likelihood win a single training seed of a stochastically trained head.
    """
    d = Path("runs/neural_etas") / head
    out = [(int(p.stem.rsplit("_s", 1)[-1]), p) for p in sorted(d.glob("per_event_full_s*.csv"))
           if p.stem.rsplit("_s", 1)[-1].isdigit()]
    return [p for _, p in sorted(out)]


def window(tag, fq_csv, head_csv, etas_csv, etas_cols=("TLL", "SLL"), seed=11):
    # head_csv may be one path or a list of per-seed paths. A list is averaged
    # over seeds; the output always records how many seeds went in.
    heads = [head_csv] if isinstance(head_csv, (str, Path)) else list(head_csv)
    if not heads:
        return {"missing": [f"runs/neural_etas/*/per_event_full_s*.csv for {tag}"]}
    if not all(Path(f).exists() for f in (fq_csv, etas_csv, *heads)):
        return {"missing": [str(f) for f in (fq_csv, etas_csv, *heads) if not Path(f).exists()]}
    fq = pd.read_csv(fq_csv, parse_dates=["time"])[["time", "tll"]]
    per_seed = [pd.read_csv(f, parse_dates=["time"])[["time", "sll_neural"]] for f in heads]
    if len(per_seed) == 1:
        hd = per_seed[0]
    else:
        # Average sll_neural per (time, duplicate-rank) across seeds, keeping only
        # events every seed scored, so the pairing below is unchanged.
        stacked = pd.concat([add_time_rank(h) for h in per_seed])
        grp = stacked.groupby(["time", "_time_rank"], as_index=False)
        hd = grp.agg(sll_neural=("sll_neural", "mean"), _n=("sll_neural", "size"))
        hd = hd[hd["_n"] == len(per_seed)][["time", "sll_neural"]]
    et = pd.read_csv(etas_csv, parse_dates=["time"]).dropna(subset=list(etas_cols))
    et = et[["time", *etas_cols]].rename(columns={etas_cols[0]: "TLL", etas_cols[1]: "SLL"})
    m = add_time_rank(fq).merge(add_time_rank(hd), on=["time", "_time_rank"])
    m = m.merge(add_time_rank(et), on=["time", "_time_rank"]).sort_values("time")
    out = {"n": int(len(m)),
           "n_temporal_scored": int(len(fq)), "n_head_scored": int(len(hd)),
           "n_etas_scored": int(len(et)),
           "coverage_vs_etas": round(float(len(m) / len(et)), 4) if len(et) else None,
           "pairing_key": "time+duplicate_rank",
           "head_seed_files": [Path(f).name for f in heads],
           "n_head_seeds": len(heads),
           "single_seed_warning": len(heads) < 2,
           "fq_tll": float(m.tll.mean()), "fq_sll": float(m.sll_neural.mean()),
           "fq_nll": float(-(m.tll.mean() + m.sll_neural.mean())),
           "etas_tll": float(m.TLL.mean()), "etas_sll": float(m.SLL.mean()),
           "etas_nll": float(-(m.TLL.mean() + m.SLL.mean()))}
    for key, col in [("dT", m.tll - m.TLL), ("dS", m.sll_neural - m.SLL),
                     ("dTot", (m.tll + m.sll_neural) - (m.TLL + m.SLL))]:
        g = paired_gain_summary(col.to_numpy(), seed=seed).asdict()
        out[key] = {"mean": round(g["mean"], 4),
                    "ci": [round(g["ci"][0], 4), round(g["ci"][1], 4)],
                    "decision": g["decision"],
                    "win_rate": round(float((col.to_numpy() > 0.0).mean()), 4),
                    "p_boot": round(block_bootstrap_pvalue(col.to_numpy(), seed=seed + 1), 5)}
        seed += 2
    return out


def main():
    res = {
        "notes": [
            "forward_2020_2026 is a retrospective out-of-time/pseudo-prospective replication, not a registered prospective forecast.",
            "non-California/Italy multi-region totals are summarized in runs/stats_hardening.json with pairing coverage.",
        ],
        "test_2007_2020": window(
            "test", "runs/n1_density/per_event_test.csv",
            head_seed_csvs("ComCat_25"), ETAS_AUG),
        "forward_2020_2026": window(
            "forward", "runs/n1_density/per_event_forward.csv",
            "runs/neural_etas/ComCat_25/per_event_forward_full.csv",
            "runs/forward_etas/per_event.csv", seed=31),
    }
    json.dump(res, open("runs/total_win.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
