"""Multiple-comparison and equivalence hardening for the cross-regime claims.

Two upgrades a hostile referee would demand:
  1. Holm-Bonferroni across the per-region temporal-gain claims (the family =
     one headline dT claim per region), using stationary-block-bootstrap
     p-values that preserve aftershock clustering.
  2. TOST equivalence tests for every "ties ETAS" claim: a tie is only
     claimable if the 90% CI of the mean gain lies within (-margin, +margin).
     Margins 0.05 and 0.10 nats/event are both reported. Positive effects
     below the 0.05 margin are still statistical wins when their CI is > 0,
     but are flagged as small.

Reads the same per-event files as scripts/multiregion_table.py.
Run: python scripts/stats_hardening.py   -> runs/stats_hardening.json
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.stats import (block_bootstrap_pvalue, holm_bonferroni,
                             paired_gain_summary, tost_equivalence)

ER = Path("reference/Experiments/ETAS")
ETAS_DIR = {
    "California": "output_data_ComCat_25", "Italy": "output_data_Italy_25",
    "Japan": "output_data_Japan_25", "Chile": "output_data_Chile_25",
    "Greece": "output_data_Greece_25", "Iran": "output_data_Iran_25",
}
# One pre-stated model choice per region. Use native models where the target
# region has enough native training support; use few-shot only for the sparse
# Greece/Iran cases where the native model is data-starved.
HEADLINE = {
    "California": "runs/n1_density/per_event_test.csv",
    "Italy": "runs/italy_n1/per_event_test.csv",
    "Japan": "runs/japan_n1/per_event_test.csv",
    "Chile": "runs/chile_n1/per_event_test.csv",
    "Greece": "runs/greece_fewshot/per_event_test.csv",
    "Iran": "runs/iran_fewshot/per_event_test.csv",
}


def etas_scores(region: str) -> pd.DataFrame:
    epe = pd.read_csv(ER / ETAS_DIR[region] / "augmented_catalog.csv", parse_dates=["time"])
    return epe.dropna(subset=["TLL", "SLL"])[["index_from_zero", "time", "TLL", "SLL"]]


def add_time_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Stable duplicate-safe timestamp key for old per-event artifacts.

    New artifacts carry ``index_from_zero``. Older FlowQuake/head CSVs have only
    timestamps, and Japan contains one duplicate timestamp; joining on
    (time, duplicate-rank) avoids the cartesian mispair caused by a raw
    timestamp-only merge.
    """

    out = df.sort_values("time").copy()
    out["_time_rank"] = out.groupby("time").cumcount()
    return out


def pair_with_etas(region: str, model: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    epe = etas_scores(region)
    if "index_from_zero" in model.columns:
        m = model[["index_from_zero", *cols]].merge(epe, on="index_from_zero", how="inner")
        key = "index_from_zero"
    else:
        m = add_time_rank(model[["time", *cols]]).merge(
            add_time_rank(epe), on=["time", "_time_rank"], how="inner"
        )
        key = "time+duplicate_rank"
    m = m.sort_values("index_from_zero")
    meta = {
        "n_etas_scored": int(len(epe)),
        "n_model_scored": int(len(model)),
        "n_matched": int(len(m)),
        "coverage_vs_etas": round(float(len(m) / len(epe)), 4) if len(epe) else None,
        "pairing_key": key,
    }
    return m, meta


def pair_total(region: str, fq: pd.DataFrame, head: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    epe = etas_scores(region)
    if "index_from_zero" in fq.columns and "index_from_zero" in head.columns:
        m = (fq[["index_from_zero", "tll"]]
             .merge(head[["index_from_zero", "sll_neural"]], on="index_from_zero", how="inner")
             .merge(epe, on="index_from_zero", how="inner"))
        key = "index_from_zero"
    else:
        fqk = add_time_rank(fq[["time", "tll"]])
        hdk = add_time_rank(head[["time", "sll_neural"]])
        etk = add_time_rank(epe)
        m = fqk.merge(hdk, on=["time", "_time_rank"], how="inner")
        m = m.merge(etk, on=["time", "_time_rank"], how="inner")
        key = "time+duplicate_rank"
    m = m.sort_values("index_from_zero")
    meta = {
        "n_etas_scored": int(len(epe)),
        "n_temporal_scored": int(len(fq)),
        "n_head_scored": int(len(head)),
        "n_matched": int(len(m)),
        "coverage_vs_etas": round(float(len(m) / len(epe)), 4) if len(epe) else None,
        "pairing_key": key,
    }
    return m, meta


def gains(region: str, csv: str):
    model = pd.read_csv(csv, parse_dates=["time"])
    m, meta = pair_with_etas(region, model, ["tll", "sll"])
    return ((m["tll"] - m["TLL"]).to_numpy(),
            ((m["tll"] + m["sll"]) - (m["TLL"] + m["SLL"])).to_numpy(),
            meta)


def main():
    res = {"per_region": {}, "family_dT_holm": None, "notes": []}
    pvals = {}
    for i, (region, csv) in enumerate(HEADLINE.items()):
        if not Path(csv).exists():
            res["per_region"][region] = {"missing": csv}
            continue
        dT, dTot, pair_meta = gains(region, csv)
        s = paired_gain_summary(dT, seed=100 + i).asdict()
        p = block_bootstrap_pvalue(dT, seed=200 + i)
        entry = {
            "n": s["n"], "pairing": pair_meta,
            "temporal_variant": "fewshot" if "fewshot" in csv else "native",
            "dT_mean": round(s["mean"], 4),
            "dT_ci": [round(s["ci"][0], 4), round(s["ci"][1], 4)],
            "dT_p_boot": round(p, 5), "dT_decision_raw": s["decision"],
            "dT_abs_below_0.05": bool(abs(s["mean"]) < 0.05),
        }
        for margin in (0.05, 0.10):
            entry[f"dT_tost_{margin}"] = tost_equivalence(dT, margin, seed=300 + i)
        st = paired_gain_summary(dTot, seed=400 + i).asdict()
        entry["dTot_mean"] = round(st["mean"], 4)
        entry["dTot_ci"] = [round(st["ci"][0], 4), round(st["ci"][1], 4)]
        entry["dTot_decision"] = st["decision"]
        if region == "Italy":
            for margin in (0.05, 0.10):
                entry[f"dTot_tost_{margin}"] = tost_equivalence(dTot, margin, seed=500 + i)
        res["per_region"][region] = entry
        pvals[region] = p
    if pvals:
        adj = holm_bonferroni(pvals)
        res["family_dT_holm"] = {
            r: {"p_raw": round(pvals[r], 5), "p_holm": round(adj[r], 5),
                "significant_05": bool(adj[r] < 0.05)} for r in pvals
        }
    # --- second family: TOTAL likelihood with the neural-ETAS spatial head ---
    HEAD_COMBOS = {
        "California": ("runs/n1_density/per_event_test.csv", "ComCat_25"),
        "Italy": ("runs/italy_n1/per_event_test.csv", "Italy_25"),
        "Japan": ("runs/japan_n1/per_event_test.csv", "Japan_25"),
        "Chile": ("runs/chile_n1/per_event_test.csv", "Chile_25"),
        "Greece": ("runs/greece_fewshot/per_event_test.csv", "Greece_25"),
        "Iran": ("runs/iran_fewshot/per_event_test.csv", "Iran_25"),
    }
    ETAS_DIR["California"] = "output_data_ComCat_25"
    tot_p, tot_rows = {}, {}
    for i, (region, (fq_csv, head)) in enumerate(HEAD_COMBOS.items()):
        head_csv = f"runs/neural_etas/{head}/per_event_full_s0.csv"
        if not (Path(fq_csv).exists() and Path(head_csv).exists()):
            continue
        fq = pd.read_csv(fq_csv, parse_dates=["time"])
        hd = pd.read_csv(head_csv, parse_dates=["time"])
        m, pair_meta = pair_total(region, fq, hd)
        d = (m["tll"] + m["sll_neural"] - m["TLL"] - m["SLL"]).to_numpy()
        s = paired_gain_summary(d, seed=600 + i).asdict()
        tot_p[region] = block_bootstrap_pvalue(d, seed=700 + i)
        tot_rows[region] = {"n": s["n"], "dTot_mean": round(s["mean"], 4),
                            "dTot_ci": [round(s["ci"][0], 4), round(s["ci"][1], 4)],
                            "decision": s["decision"], "p_boot": round(tot_p[region], 5),
                            "dTot_abs_below_0.05": bool(abs(s["mean"]) < 0.05),
                            "temporal_variant": "fewshot" if "fewshot" in fq_csv else "native",
                            "pairing": pair_meta}
    if tot_p:
        adj = holm_bonferroni(tot_p)
        for r in tot_rows:
            tot_rows[r]["p_holm"] = round(adj[r], 5)
            tot_rows[r]["significant_05_holm"] = bool(adj[r] < 0.05)
        res["total_with_head_family"] = tot_rows

    res["notes"].append("family = one headline dT claim per region; Holm step-down; native for California/Italy/Japan/Chile, few-shot only for sparse Greece/Iran")
    res["notes"].append("tie claims require TOST equivalence, not just a CI crossing 0; positive effects below 0.05 nats/event are small wins, not large-margin wins")
    res["notes"].append("total_with_head_family: flow temporal (native; fewshot for Greece/Iran) + "
                        "region-trained neural-ETAS spatial head vs region-fitted ETAS; "
                        "the head requires the region's ETAS inversion (upgrade, not inversion-free); "
                        "non-California/Italy totals are paired on the intersection of temporal/head/ETAS-scored events and report coverage")
    Path("runs").mkdir(exist_ok=True)
    json.dump(res, open("runs/stats_hardening.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
