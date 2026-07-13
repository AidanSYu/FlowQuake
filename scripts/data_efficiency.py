"""Data-efficiency curve -- the operational money figure.

Question a forecaster actually faces: "I only have K years of catalog -- train
from scratch, or transfer a pre-trained model?" We vary the TRAINING HISTORY
LENGTH (truncate train_start within [.., val_start); you cannot randomly drop
events from a point process without breaking triggering) on a data-rich region
and, at each size, compare three deployable options on the FIXED 2011-2020 test:
  * native   : FlowQuake trained from scratch on [train_start, val_start)
  * zero-shot : the leave-one-region-out pooled model (never saw this region)
                applied directly (region stats only -- no weight training)
  * few-shot  : that pooled model gently fine-tuned (low LR) on the same window
The expected story: native collapses as history shrinks; zero-shot is a robust
floor needing no region training; few-shot is best-of-both -- exactly where ETAS,
inverted per region, also struggles.

Run: python scripts/data_efficiency.py --region chile \
        --pool-ckpt runs/pool_loo_chile/ckpt_best.pt --base configs/chile_n1.yaml
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.config import Config

PY = sys.executable
ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."}
# valid truncations: train window is [start, val_start); keep starts < val_start
DEFAULT_STARTS = ["1992-01-01", "2000-01-01", "2004-01-01", "2006-01-01", "2007-01-01", "2008-01-01"]


def train(cfg_path, out, steps, device, init=None):
    cmd = [PY, "-m", "flowquake.train", cfg_path, "--out", out, "--steps", str(steps),
           "--eval-after", "--device", device]
    if init:
        cmd += ["--init-from", init]
    subprocess.run(cmd, env=ENV, check=False)
    f = Path(out) / "eval_test.json"
    return json.load(open(f)) if f.exists() else None


def zero_shot(ckpt, region_cap, base, start, device):
    cmd = [PY, "scripts/transfer_eval.py", "--ckpt", ckpt,
           "--catalog", f"reference/Datasets/{region_cap}/{region_cap}_catalog.csv",
           "--mcut", str(base.data.mcut), "--aux-start", start, "--train-start", start,
           "--val-start", base.data.val_start, "--test-start", base.data.test_start,
           "--test-end", base.data.test_end, "--device", device]
    subprocess.run(cmd, env=ENV, check=False)
    f = Path(f"runs/transfer_{region_cap}.json")
    return json.load(open(f)) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--pool-ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--starts", nargs="+", default=DEFAULT_STARTS)
    ap.add_argument("--native-steps", type=int, default=6000)
    ap.add_argument("--few-steps", type=int, default=1500)
    ap.add_argument("--few-lr", type=float, default=5e-5)   # gentle: avoid catastrophic forgetting
    ap.add_argument("--device", default="cuda", help="passed through to train/eval commands")
    args = ap.parse_args()
    cap = args.region.capitalize()

    base = Config.load(args.base)
    cat = pd.read_csv(base.data.catalog_path, parse_dates=["time"])
    tmp = Path("runs/_de_cfgs"); tmp.mkdir(parents=True, exist_ok=True)
    results = []
    for start in args.starts:
        if start >= base.data.val_start:
            print(f"skip {start} (>= val_start {base.data.val_start})"); continue
        n_train = int(((cat.time >= start) & (cat.time < base.data.val_start) &
                       (cat.magnitude >= base.data.mcut)).sum())
        # native config
        c = Config.load(args.base); c.data.train_start = start; c.data.aux_start = start
        nat_cfg = str(tmp / f"{args.region}_{start}_nat.yaml"); c.dump(nat_cfg)
        # few-shot config: gentle LR + short warmup
        c2 = Config.load(args.base); c2.data.train_start = start; c2.data.aux_start = start
        c2.train.lr = args.few_lr; c2.train.warmup = 100
        few_cfg = str(tmp / f"{args.region}_{start}_few.yaml"); c2.dump(few_cfg)

        print(f"\n===== {cap} start={start} train_N={n_train} =====", flush=True)
        nat = train(nat_cfg, f"runs/de_{args.region}_{start}_native", args.native_steps, args.device)
        zs = zero_shot(args.pool_ckpt, cap, base, start, args.device)
        few = train(few_cfg, f"runs/de_{args.region}_{start}_few", args.few_steps, args.device, init=args.pool_ckpt)
        results.append({"start": start, "n_train": n_train,
                        "native_tll": nat["tll"] if nat else None,
                        "zeroshot_tll": zs["tll"] if zs else None,
                        "few_tll": few["tll"] if few else None})
        print(f"  native={results[-1]['native_tll']} zero={results[-1]['zeroshot_tll']} few={results[-1]['few_tll']}", flush=True)

    out = f"runs/data_efficiency_{args.region}.json"
    json.dump(results, open(out, "w"), indent=2)
    def fmt(v): return f"{v:>11.4f}" if isinstance(v, (int, float)) else f"{'--':>11}"
    print(f"\n{'start':12}{'n_train':>9}{'native':>11}{'zero-shot':>11}{'few-shot':>11}")
    for r in results:
        print(f"{r['start']:12}{r['n_train']:>9}{fmt(r['native_tll'])}{fmt(r['zeroshot_tll'])}{fmt(r['few_tll'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
