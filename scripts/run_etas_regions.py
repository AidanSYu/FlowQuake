"""Phase B: re-fit region-ETAS baselines on the authoritative ISC catalogs.

Per region: invert_etas.py <Cfg> (single-threaded EM, ~3-4h for the big regions)
then predict_etas.py <Cfg> (multiprocessing pool, fast) -> augmented_catalog.csv
+ ll_scores.json. Inversions run with bounded concurrency (default 2) to respect
memory; the predict step (which itself spawns ~30 workers) runs one at a time.

Both scripts resolve config/<Cfg>.json and write output_data_<Cfg>/ relative to
reference/Experiments/ETAS, so everything runs with that as cwd.

Run: python scripts/run_etas_regions.py Japan_25 Chile_25 Greece_25 Iran_25 [--invert-jobs 2]
"""
import argparse, subprocess, sys, time
from pathlib import Path

ETAS_DIR = Path("reference/Experiments/ETAS")
LOG_DIR = Path("runs"); LOG_DIR.mkdir(exist_ok=True)


def run_stage(stage, cfgs, max_par):
    """Run `python <stage>.py <cfg>` for each cfg, at most max_par at a time."""
    pending = list(cfgs); active = []   # list of (cfg, Popen, logfile)
    while pending or active:
        while pending and len(active) < max_par:
            cfg = pending.pop(0)
            lf = open(LOG_DIR / f"etas_{stage}_{cfg}.log", "w")
            p = subprocess.Popen([sys.executable, f"{stage}.py", cfg],
                                 cwd=str(ETAS_DIR), stdout=lf,
                                 stderr=subprocess.STDOUT)
            active.append((cfg, p, lf))
            print(f"[{time.strftime('%H:%M:%S')}] {stage} START {cfg}  (pid {p.pid})", flush=True)
        time.sleep(15)
        still = []
        for cfg, p, lf in active:
            if p.poll() is None:
                still.append((cfg, p, lf))
            else:
                lf.close()
                tag = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
                print(f"[{time.strftime('%H:%M:%S')}] {stage} {tag} {cfg}", flush=True)
        active = still


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="ETAS config stems, e.g. Japan_25 Chile_25")
    ap.add_argument("--invert-jobs", type=int, default=2)
    ap.add_argument("--skip-invert", action="store_true")
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_invert:
        print(f"=== INVERT ({len(args.configs)} regions, {args.invert_jobs} parallel) ===", flush=True)
        run_stage("invert_etas", args.configs, args.invert_jobs)
    if not args.skip_predict:
        print("=== PREDICT (sequential; each uses a worker pool) ===", flush=True)
        run_stage("predict_etas", args.configs, 1)
    print(f"=== ETAS regions done in {(time.time()-t0)/3600:.1f}h ===", flush=True)


if __name__ == "__main__":
    main()
