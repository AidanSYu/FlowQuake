"""Score a TRAINED neural-ETAS head on another precomputed feature file
(e.g. the 2020-2026 forward window) -- frozen weights, out-of-time scoring.

Run: python scripts/score_neural_etas.py runs/neural_etas/ComCat_25/head_full_s0.pt \
         ComCat_25_forward --out runs/neural_etas/ComCat_25/per_event_forward_full.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.neural_etas import NeuralETASSpatialHead
from flowquake.stats import paired_gain_summary
from scripts.train_neural_etas import PARAM_NAMES, load, mean_ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("feat", help="feature stem, e.g. ComCat_25_forward")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    params, data, times, etas_sll, tgt_mag, n, event_index = load(args.feat, "cpu")
    model = NeuralETASSpatialHead(
        ck["params"], n_kde=ck["n_kde"], use_mlp=ck.get("use_mlp", True),
        refit_globals=ck.get("refit_globals", False),
    )
    model.load_state_dict(ck["state"])
    model.eval()

    idx = torch.arange(n)
    sll_neural = mean_ll(model, data, idx).numpy()
    g = paired_gain_summary(sll_neural - etas_sll, seed=7).asdict()
    pd.DataFrame({"index_from_zero": event_index,
                  "time": times, "magnitude": tgt_mag,
                  "sll_neural": sll_neural, "sll_etas": etas_sll}).to_csv(args.out, index=False)
    summary = {"n": int(n), "sll_neural": float(sll_neural.mean()),
               "sll_etas_frozen": float(etas_sll.mean()),
               "dS_mean": round(g["mean"], 4),
               "dS_ci": [round(g["ci"][0], 4), round(g["ci"][1], 4)],
               "dS_win_rate": round(float((sll_neural - etas_sll > 0.0).mean()), 4),
               "decision": g["decision"], "ckpt": args.ckpt, "feat": args.feat}
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
