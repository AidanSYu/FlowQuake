"""Ground-truth validation of the scaling-curve estimator.

`MOONSHOT.md` claims we can MEASURE how much forecast skill about large
earthquakes lives in small ones. Before spending money measuring it on real
data, we must show the estimator recovers a **planted** answer — and, just as
importantly, that it reports *no* signal when there is none to find.

Two catalogs, generated as a matched pair:

  A "informative"  A subcritical ETAS branching process down to `m_floor`.
                   Small events are genuine members of the cascade: an active
                   M1.5 swarm really does raise the rate of the next M>=3. So a
                   model that sees lower mc has real extra information, and the
                   curve SHOULD slope.

  B "null"         Identical at m >= `m_split` — the same events, the same
                   M>=M_TARGET target set, event for event. Below `m_split` the
                   events are REPLACED by a homogeneous Poisson process in
                   space and time with the same count. The small events are
                   therefore pure noise, carrying zero information about the
                   targets, and the curve MUST be flat.

The estimator passes only if it shows a slope on A and none on B. A slope on B
means the harness manufactures signal from event density alone — which would
invalidate every number the real experiment could produce.

This is the cheap, decisive precondition for gate G3.

Usage:
    python scripts/synthetic_validation.py --out runs/synthetic_validation \
        --steps 3000 --n-sims 200 --concurrency 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY = sys.executable
T0 = pd.Timestamp("1971-01-01")


def branching_catalog(years: float, m_floor: float, b: float, br: float,
                      bg_per_day: float, seed: int, region_km: float = 400.0,
                      r_max_km: float = 300.0) -> pd.DataFrame:
    """Subcritical space-time ETAS-like branching process.

    IMPORTANT — the clustering you get at the SCORED threshold is not the
    clustering of the catalog. For productivity `10^(a(m-m0))` and GR `b`, a
    parent at the target threshold yields on the order of

        n_sub ~ br * 10^(-(b - a) * (M_TARGET - m0))

    offspring that also clear that threshold. With this function's a = 0.5,
    b = 1.0 and M_TARGET - m0 = 2.5 that is ~0.025: the target sub-process is
    POISSON even though the full catalog's branching ratio is 0.9 and the
    catalog looks properly clustered. Measured consequence: variance/mean of
    the M>=3 window counts was 1.21 against 2.0-5.4 for real regional
    catalogs, and neither a trained neural model (corr +0.15) nor a fitted
    ETAS (corr -0.07) could forecast it -- correctly, because there was
    nothing there.

    Raising `br` does not fix it. At a = 0.5, b = 1.0 the mean branching ratio
    is 2*br, so br = 0.75 is already SUPERCRITICAL and stays finite only
    because of the productivity clip below. Clustering at the scored threshold
    needs a ~ b (the self-similar case), which also requires truncating
    magnitudes to keep the branching integral finite.

    Use this for controlled experiments where the generating parameters must
    be known. For validating the estimator, prefer a real catalog with a
    surrogate null (MOONSHOT.md invariant 1g) -- its clustering is real by
    definition.

    Omori in time, power-law in space, Gutenberg-Richter magnitudes, productivity
    scaling with parent magnitude. Generation-by-generation and vectorised, so a
    half-million events take seconds.

    `region_km` and `r_max_km` are not cosmetic. The spatial kernel
    r ~ sqrt(Pareto) has tail index ~1.4, so offsets compound across generations
    and an unbounded process wanders off the planet — measured 73,908 km of span
    before these were added, which NaNs the spatial head at step 1. Real
    catalogs are bounded by a network polygon (EarthquakeNPP applies the RELM
    polygon to ComCat), so clipping the offset and dropping out-of-region
    children is the faithful choice, not a hack.
    """
    rng = np.random.default_rng(seed)
    T = years * 365.25
    beta = b * np.log(10.0)

    def mags(n):
        return m_floor + rng.exponential(1.0 / beta, n)

    n_bg = rng.poisson(bg_per_day * T)
    cur = (np.sort(rng.uniform(0, T, n_bg)),
           np.clip(rng.normal(0, 120, n_bg), -region_km, region_km),
           np.clip(rng.normal(0, 120, n_bg), -region_km, region_km), mags(n_bg))
    gens, total = [cur], n_bg
    for _ in range(20):
        ti, xi, yi, mi = cur
        if len(ti) == 0 or total > 2_000_000:
            break
        lam = np.clip(br * 10.0 ** (0.5 * (mi - m_floor)), 0.0, 40.0)
        k = rng.poisson(lam)
        if k.sum() == 0:
            break
        par = np.repeat(np.arange(len(ti)), k)
        n = len(par)
        dt = 0.01 * ((1.0 - rng.random(n)) ** (-1.0 / 0.15) - 1.0)
        tn = ti[par] + dt
        r = np.minimum(
            0.5 * 10.0 ** (0.45 * (mi[par] - m_floor)) *
            np.sqrt((1.0 - rng.random(n)) ** (-1.0 / 0.7) - 1.0), r_max_km)
        th = rng.uniform(0, 2 * np.pi, n)
        xn, yn = xi[par] + r * np.cos(th), yi[par] + r * np.sin(th)
        # in-region survivors only: the network-polygon filter a real catalog applies
        keep = (tn < T) & (np.abs(xn) <= region_km) & (np.abs(yn) <= region_km)
        cur = (tn[keep], xn[keep], yn[keep], mags(int(keep.sum())))
        total += len(cur[0])
        gens.append(cur)

    t = np.concatenate([g[0] for g in gens])
    x = np.concatenate([g[1] for g in gens])
    y = np.concatenate([g[2] for g in gens])
    m = np.concatenate([g[3] for g in gens])
    o = np.argsort(t)
    return pd.DataFrame({"t_days": t[o], "x": x[o], "y": y[o], "magnitude": m[o]})


def make_uniform_null(df: pd.DataFrame, m_split: float, seed: int) -> pd.DataFrame:
    """Keep m >= m_split exactly; replace everything below with uniform noise.

    This is a NEGATIVE-information control, not a zero-information one, and the
    distinction was originally missed here. The sub-split events of a branching
    catalog are aftershocks: they sit on the same structures as the targets.
    Measured on the v2 catalogs, small-event spatial spread was 121 km on the
    informative arm against 119 km for the targets, but 231 km here — uniform
    over the bounding box, nearly 2x too wide. A model trained on this has its
    spatial density dragged toward uniform, away from where the targets live,
    so the arm loses skill as mc drops (measured -0.47 nats/decade in the shape
    term) even though the added events are independent of the targets.

    Retained deliberately: "does the pipeline degrade gracefully when fed
    actively misleading events?" is a real question. It is just not the null
    that certifies the estimator. Use `make_decoupled_null` for that.
    """
    rng = np.random.default_rng(seed)
    keep = df[df["magnitude"] >= m_split]
    small = df[df["magnitude"] < m_split]
    n = len(small)
    noise = pd.DataFrame({
        "t_days": rng.uniform(df["t_days"].min(), df["t_days"].max(), n),
        "x": rng.uniform(df["x"].min(), df["x"].max(), n),
        "y": rng.uniform(df["y"].min(), df["y"].max(), n),
        # resample the observed sub-split magnitudes: identical marginal
        "magnitude": rng.permutation(small["magnitude"].to_numpy()),
    })
    out = pd.concat([keep, noise], ignore_index=True).sort_values("t_days")
    return out.reset_index(drop=True)


def make_decoupled_null(df: pd.DataFrame, m_split: float, gen_kwargs: dict,
                        seed: int) -> pd.DataFrame:
    """Arm A's targets grafted onto an INDEPENDENT realisation's small events.

    The correct zero-coupling control. The donor catalog is drawn from the same
    generator with the same parameters and a different seed, so its sub-split
    population matches arm A's in count, b-value, Omori clustering AND spatial
    footprint — the background law is a fixed N(0, 120 km) field, so every
    realisation has the same marginal density. What it cannot have is any
    dependence on arm A's particular target events, because it never saw them.

    So the only thing that differs from arm A is the coupling between the small
    events and the target set, which is exactly the quantity the scaling curve
    claims to measure. The counts are matched by resampling the donor pool to
    arm A's size, keeping the count-vs-mc profile identical across arms.

    Expected slope: at or slightly above zero. Slightly above is legitimate and
    not a failure — decoupled small events still trace the background field,
    which is genuine information about where targets occur. A NEGATIVE slope
    here means the pipeline is being harmed by well-distributed events, and a
    large positive one means the estimator is inventing skill.
    """
    rng = np.random.default_rng(seed)
    keep = df[df["magnitude"] >= m_split]
    n = int((df["magnitude"] < m_split).sum())

    donor = branching_catalog(seed=seed, **gen_kwargs)
    pool = donor[donor["magnitude"] < m_split]
    if len(pool) == 0:
        raise ValueError("donor catalog has no sub-split events")
    idx = rng.choice(len(pool), size=n, replace=len(pool) < n)
    small = pool.iloc[np.sort(idx)].reset_index(drop=True)
    # clip to arm A's span so the two arms occupy the same observation window
    lo, hi = float(df["t_days"].min()), float(df["t_days"].max())
    small = small[(small["t_days"] >= lo) & (small["t_days"] <= hi)]

    out = pd.concat([keep, small], ignore_index=True).sort_values("t_days")
    return out.reset_index(drop=True)


def write_catalog(df: pd.DataFrame, path: Path) -> None:
    d = df.copy()
    d["time"] = T0 + pd.to_timedelta(d["t_days"], unit="D")
    d["latitude"] = 35.0 + d["y"] / 111.0
    d["longitude"] = -119.0 + d["x"] / (111.0 * np.cos(np.radians(35.0)))
    d["id"] = np.arange(len(d))
    path.parent.mkdir(parents=True, exist_ok=True)
    d[["time", "x", "y", "magnitude", "latitude", "longitude", "id"]].to_csv(
        path, index=False)


def write_config(catalog: Path, out_dir: Path, path: Path, mcut: float) -> None:
    cfg = {
        "data": {"catalog_path": str(catalog), "mcut": mcut,
                 "aux_start": "1971-01-01", "train_start": "1981-01-01",
                 "val_start": "1998-01-01", "test_start": "2007-01-01",
                 "test_end": "2020-01-17"},
        "model": {"d_model": 96, "n_layers": 4, "d_state": 32, "n_heads": 6,
                  "expand": 2, "chunk": 64, "flow_hidden": 96, "mix_hidden": 64,
                  "flow_layers": 3, "loss_weights": [1.0, 1.0, 0.5],
                  "sigma_min": [0.02, 0.01, 0.05], "dropout": 0.1,
                  "input_noise": 0.1, "h_bottleneck": 0,
                  "spatial_density_feat": True, "d_floor_km": 0.1},
        "train": {"window": 2048, "burn_in": 256, "batch_size": 8,
                  "steps": 3000, "lr": 3e-4, "weight_decay": 0.03,
                  "warmup": 200, "grad_clip": 1.0, "seed": 1555,
                  "val_every": 500, "val_events": 2048, "val_ode_steps": 16,
                  "patience": 8, "out_dir": str(out_dir)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)


def slope(mcs: list[float], lls: list[float]) -> float:
    """Nats per decade of magnitude. x is mc, so a NEGATIVE mc step is a decade
    downward; we report skill gained per decade of extra small events."""
    ok = [(m, l) for m, l in zip(mcs, lls) if l is not None]
    if len(ok) < 2:
        return float("nan")
    m = np.array([a for a, _ in ok]); l = np.array([b for _, b in ok])
    return float(-np.polyfit(m, l, 1)[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/synthetic_validation")
    ap.add_argument("--years", type=float, default=49.0)
    ap.add_argument("--m-floor", type=float, default=0.5)
    ap.add_argument("--m-split", type=float, default=2.5,
                    help="events below this are randomised in the null catalog")
    ap.add_argument("--m-target", type=float, default=3.0)
    ap.add_argument("--m-large", type=float, default=4.5)
    ap.add_argument("--mc", type=float, nargs="+", default=[2.5, 2.0, 1.5, 1.0])
    ap.add_argument("--b", type=float, default=1.0)
    # Defaults chosen to match the CLUSTERING of real catalogs, not for speed.
    # The original 0.45 / 6.0 produced a background-dominated process whose
    # M>=3 window counts were essentially Poisson (variance/mean = 1.2 against
    # 2.0-5.4 for the real regional catalogs, 38.9 for ComCat M>=4). Only ~17%
    # of its variance was predictable even in principle, so NO probe could
    # forecast it -- neural (corr +0.15) and a fitted ETAS (corr -0.07) both
    # returned essentially constant forecasts, which was the correct answer.
    # A validation catalog with nothing to forecast cannot validate anything.
    # 0.75 / 0.10 gives variance/mean = 2.44, between QTM SanJac and WHITE.
    ap.add_argument("--branching", type=float, default=0.75)
    ap.add_argument("--bg-per-day", type=float, default=0.10)
    ap.add_argument("--min-overdispersion", type=float, default=1.8,
                    help="refuse to run if the target series is less clustered "
                         "than this (variance/mean); real regional catalogs "
                         "measure 2.0-5.4")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--sample-steps", type=int, default=8)
    ap.add_argument("--horizon", type=float, default=90.0)
    ap.add_argument("--bin-km", type=float, default=15.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--mem-per-worker", type=float, default=4.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cat_dir = out / "catalogs"

    gen_kwargs = dict(years=args.years, m_floor=args.m_floor, b=args.b,
                      br=args.branching, bg_per_day=args.bg_per_day)
    paths = {k: cat_dir / f"{k}.csv"
             for k in ("informative", "null_decoupled", "null_uniform")}
    if not all(p.exists() for p in paths.values()):
        print("[gen] branching catalog ...", flush=True)
        A = branching_catalog(seed=args.seed, **gen_kwargs)
        write_catalog(A, paths["informative"])
        write_catalog(make_decoupled_null(A, args.m_split, gen_kwargs,
                                          args.seed + 101),
                      paths["null_decoupled"])
        write_catalog(make_uniform_null(A, args.m_split, args.seed + 1),
                      paths["null_uniform"])
    cats = {k: pd.read_csv(p) for k, p in paths.items()}

    for lbl, d in cats.items():
        counts = {f"mc{c:g}": int((d.magnitude >= c).sum()) for c in args.mc}
        print(f"   {lbl:16} n={len(d):>7,} {counts} | M>={args.m_target:g}: "
              f"{int((d.magnitude >= args.m_target).sum())} | "
              f"M>={args.m_large:g}: {int((d.magnitude >= args.m_large).sum())}")

    # The comparison is only valid if every arm carries the SAME target set and
    # the small events occupy the same footprint (the v2 failure was footprint).
    ref_n = int((cats["informative"].magnitude >= args.m_target).sum())
    ok_tgt = all(int((d.magnitude >= args.m_target).sum()) == ref_n
                 for d in cats.values())
    print(f"\n[gen] target sets identical across arms: {ok_tgt}"
          f"{'' if ok_tgt else '   *** MISMATCH — the comparison is invalid ***'}")
    print(f"  {'arm':16}{'small n':>9}{'sd_x km':>10}{'sd_y km':>10}")
    tg = cats["informative"][cats["informative"].magnitude >= args.m_target]
    for lbl, d in cats.items():
        s = d[d.magnitude < args.m_split]
        print(f"  {lbl:16}{len(s):>9,}{s.x.std():>10.1f}{s.y.std():>10.1f}")
    print(f"  {'(targets)':16}{len(tg):>9,}{tg.x.std():>10.1f}{tg.y.std():>10.1f}"
          "   <- decoupled null must match this, uniform null will not")

    # --- clustering gate ----------------------------------------------------
    # A ground-truth validation is only meaningful if the target process is
    # actually forecastable. Check that BEFORE spending hours on it. Poisson
    # gives variance/mean = 1; everything above that is the part a forecaster
    # could in principle capture.
    tgt_t = cats["informative"].loc[
        cats["informative"].magnitude >= args.m_target, "t_days"].to_numpy()
    edges = np.arange(tgt_t.min(), tgt_t.max() + args.horizon, args.horizon)
    counts = np.histogram(tgt_t, bins=edges)[0].astype(float)
    od = float(counts.var() / counts.mean()) if counts.mean() > 0 else 0.0
    frac = max(0.0, counts.var() - counts.mean()) / counts.var() if counts.var() else 0.0
    print(f"\n[clustering] M>={args.m_target:g} counts over {args.horizon:g}-day "
          f"windows: mean={counts.mean():.2f}, variance/mean={od:.2f}x "
          f"({frac:.0%} of variance predictable in principle)")
    print("  reference — real catalogs: QTM SanJac 2.03x, WHITE 2.90x, "
          "QTM SaltonSea 5.36x, ComCat M>=4 38.92x")
    if od < args.min_overdispersion:
        raise SystemExit(
            f"\n*** ABORT: variance/mean {od:.2f} < {args.min_overdispersion} ***\n"
            f"  This catalog is close to Poisson, so there is almost nothing for a\n"
            f"  forecaster to predict and every arm will come back flat REGARDLESS\n"
            f"  of whether the estimator works. That is not a validation.\n"
            f"  Raise --branching (try 0.75-0.80) and lower --bg-per-day\n"
            f"  (try 0.05-0.10), or lower --min-overdispersion only if you can say\n"
            f"  why a near-Poisson target is the right test.")

    results = {}
    for lbl, cat in paths.items():
        cdir = out / lbl
        cfg_path = out / "_cfg" / f"{lbl}.yaml"
        write_config(cat, cdir, cfg_path, mcut=max(args.mc))
        cmd = [PY, "scripts/scaling_curve.py", "--base", str(cfg_path),
               "--out", str(cdir), "--mc", *[f"{m:g}" for m in args.mc],
               "--arms", "matched_window", "matched_n", "--seeds", "0",
               "--m-target", str(args.m_target), "--m-large", str(args.m_large),
               "--horizon", str(args.horizon), "--bin-km", str(args.bin_km),
               "--n-sims", str(args.n_sims), "--sample-steps", str(args.sample_steps),
               "--steps", str(args.steps), "--device", args.device,
               "--concurrency", str(args.concurrency), "--tail-mode", "fixed",
               "--mem-per-worker", str(args.mem_per_worker)]
        print(f"\n{'='*70}\n[curve] {lbl}\n{'='*70}", flush=True)
        subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
        cj = cdir / "curve.json"
        results[lbl] = json.load(open(cj)) if cj.exists() else []

    # ---- verdict ----------------------------------------------------------
    # PRIMARY metric is the SHAPE (CSEP S-test) term. The total likelihood also
    # carries the level (N-test) term, and level absorbs magnitude-tail error
    # with mc-dependent weight: the fixed tail extrapolates over m_target - mc
    # decades, so a b error of db shifts the forecast count by 10**(db*(m_target
    # - mc)). Shape is invariant to lam -> c*lam and so cannot move for that
    # reason. Level is still reported, because a level slope is diagnostic of
    # exactly that tail problem and should not be hidden.
    print(f"\n{'='*70}\nGROUND-TRUTH VERDICT\n{'='*70}")
    arms = ["informative", "null_decoupled", "null_uniform"]
    keys = [("shape", "ll_shape_per_target_event"),
            ("level", "ll_level_per_target_event"),
            ("total", "ll_per_target_event")]
    summary = {}
    for arm in ("matched_window", "matched_n"):
        summary[arm] = {}
        print(f"\n[{arm}]")
        print(f"  {'catalog':16}{'mc':>6}{'shape/tgt':>12}{'level/tgt':>12}{'total/tgt':>12}")
        for lbl in arms:
            rows = [r for r in results.get(lbl, []) if r["arm"] == arm]
            rows.sort(key=lambda r: -r["mc"])
            if not rows:
                continue
            for r in rows:
                cells = "".join(
                    f"{'None' if r.get(k) is None else round(r[k], 4):>12}"
                    for _, k in keys)
                print(f"  {lbl:16}{r['mc']:>6g}{cells}")
            summary[arm][lbl] = {
                name: slope([r["mc"] for r in rows], [r.get(k) for r in rows])
                for name, k in keys}
            s = summary[arm][lbl]
            print(f"  {lbl:16}{'slope':>6}"
                  f"{s['shape']:>12.4f}{s['level']:>12.4f}{s['total']:>12.4f}"
                  "   nats/decade")

    ok = []
    for arm in ("matched_window", "matched_n"):
        if not {"informative", "null_decoupled"} <= summary[arm].keys():
            ok.append(False); continue
        si = summary[arm]["informative"]["shape"]
        sn = summary[arm]["null_decoupled"]["shape"]
        # The decoupled null must not manufacture skill, and must not destroy it
        # either: |slope| small. The informative arm must clear it by 2x.
        null_flat = abs(sn) < 0.10
        passed = (si > 0) and (si > 2 * abs(sn)) and null_flat
        ok.append(passed)
        print(f"\n[{arm}] SHAPE: informative {si:+.4f} vs decoupled null {sn:+.4f}"
              f"  (null flat: {null_flat}) -> {'PASS' if passed else 'FAIL'}")
        if "null_uniform" in summary[arm]:
            print(f"           uniform null {summary[arm]['null_uniform']['shape']:+.4f}"
                  "  (negative-information control; a negative slope is expected here)")
    verdict = "PASS" if all(ok) else "FAIL"
    print(f"\nESTIMATOR VALIDATION: {verdict}")
    if verdict == "FAIL":
        print("  A slope on the decoupled null, or none on the informative catalog,\n"
              "  means the harness cannot measure what MOONSHOT.md claims. Do NOT\n"
              "  proceed to gate G3 on real data until this passes.")
    json.dump({"summary": summary, "verdict": verdict, "args": vars(args)},
              open(out / "verdict.json", "w"), indent=2)
    print(f"\nwrote {out/'verdict.json'}")


if __name__ == "__main__":
    main()
