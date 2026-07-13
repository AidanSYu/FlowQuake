"""Train an EarthquakeNPP model and report test TLL/SLL vs the ETAS incumbent.

Examples
--------
# headline kill: selective-SSM whole-catalog encoder + flow-matching spatial head
python train.py --dataset ComCat_25 --encoder ssm --spatial fm --steps 4000

# encoder-lever ablation (same flow head, swap encoder)
python train.py --dataset ComCat_25 --encoder gru --spatial fm
# density-lever ablation (same SSM, swap head)
python train.py --dataset ComCat_25 --encoder ssm --spatial gaussian
"""
import argparse
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eqnpp.data import build_catalog_tensors
from eqnpp.model import EarthquakeNPP, ModelConfig
from eqnpp.train_lib import TrainConfig, train, evaluate
from eqnpp.data import SPLIT_TEST, SPLIT_VAL
from eqnpp import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ComCat_25")
    ap.add_argument("--encoder", default="ssm", choices=["ssm", "gru"])
    ap.add_argument("--spatial", default="trigger",
                    choices=["gaussian", "mdn", "realnvp", "fm", "trigger"])
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--d_state", type=int, default=8)
    ap.add_argument("--n_recent", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--fm_steps", type=int, default=40)
    ap.add_argument("--use_magnitude", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    print(f"[eqnpp] dataset={args.dataset} encoder={args.encoder} spatial={args.spatial} "
          f"device={args.device}", flush=True)

    cat = build_catalog_tensors(args.dataset, device=args.device)
    mcfg = ModelConfig(encoder=args.encoder, spatial=args.spatial, d_model=args.d_model,
                       n_layers=args.n_layers, d_state=args.d_state, fm_steps=args.fm_steps,
                       n_recent=args.n_recent, use_magnitude=args.use_magnitude)
    model = EarthquakeNPP(mcfg, cat.standardizer.mean, cat.standardizer.std).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eqnpp] model params: {n_params:,}", flush=True)

    tcfg = TrainConfig(dataset=args.dataset, steps=args.steps, batch=args.batch,
                       crop_len=args.crop_len, lr=args.lr, seed=args.seed, device=args.device)

    t0 = time.time()
    model, best_val = train(model, cat, tcfg)
    train_time = time.time() - t0

    tll, sll, mll = evaluate(model, cat, SPLIT_TEST, tcfg)
    print("\n" + metrics.summarize(args.dataset, tll, sll))
    if args.use_magnitude:
        print(f"  magnitude LL = {mll:+.4f}")

    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    tag = args.tag or f"{args.dataset}_{args.encoder}_{args.spatial}_s{args.seed}"
    result = dict(dataset=args.dataset, encoder=args.encoder, spatial=args.spatial,
                  d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state,
                  fm_steps=args.fm_steps, steps=args.steps, seed=args.seed,
                  n_params=n_params, train_time_s=round(train_time, 1),
                  test_tll=tll, test_sll=sll, test_mll=mll,
                  etas_tll=metrics.ETAS.get(args.dataset, [None, None])[0],
                  etas_sll=metrics.ETAS.get(args.dataset, [None, None])[1],
                  sll_minus_etas=(sll - metrics.ETAS[args.dataset][1]) if args.dataset in metrics.ETAS else None)
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    torch.save({"model": model.state_dict(), "cfg": vars(args)}, out / f"{tag}.pt")
    print(f"\n[eqnpp] saved runs/{tag}.json and .pt  (train {train_time:.0f}s)")


if __name__ == "__main__":
    main()
