"""Score a FROZEN ComCat-family ETAS inversion on the 2020-2026 forward window.

This is a retrospective out-of-time / pseudo-prospective replication: the
parameters were fitted on data through 2007 and published with the benchmark,
and nothing here is refit, but the 2020-2026 events already existed when this
project scored the window. Do not describe this as a registered prospective
forecast.

Uses the reimplementation verified to reproduce the package per-event SLL to
1.8e-9 (scripts/etas_sll_repro.py), extended with the temporal term:
    TLL_i = log lambda*(t_i) - int_{t_{i-1}}^{t_i} lambda*(s) ds
where the time-decay integral H(T) = int_0^T e^{-t/tau}(t+c)^{-(1+omega)} dt is
computed by cumulative quadrature on a log mesh and interpolated (same scheme
as the package, denser mesh).

Validation mode re-scores the ORIGINAL 2007-2020 test window and compares
per-event TLL/SLL to the package's augmented_catalog.csv. SLL matches to
~1e-9; TLL matches to ~1e-5 except the first target, where this implementation
uses a consistent window-start anchor for both background and triggering terms.

Run:  python scripts/etas_forward_eval.py --validate     (must pass first)
      python scripts/etas_forward_eval.py                (forward window)
      python scripts/etas_forward_eval.py --name ComCat_25_refit2020
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

ETAS_ROOT = Path("reference/Experiments/ETAS")
EXT = Path("reference/Datasets/ComCat_extended/ComCat_extended_catalog.csv")
BLOCK = 128


def round_half_up(x, delta=0.1):
    return np.floor(x / delta + 0.5) * delta


def output_dir(name: str) -> Path:
    return ETAS_ROOT / f"output_data_{name}"


def load_params(name: str):
    meta = json.loads((output_dir(name) / "parameters_0.json").read_text())
    p = meta["final_parameters"]
    return dict(mu=10.0 ** p["log10_mu"], k0=10.0 ** p["log10_k0"], c=10.0 ** p["log10_c"],
                tau=10.0 ** p["log10_tau"], d=10.0 ** p["log10_d"], a=p["a"], omega=p["omega"],
                gamma=p["gamma"], rho=p["rho"], mc=meta["mc"], area=meta["area"],
                R=meta["earth_radius"], aux=pd.Timestamp(meta["auxiliary_start"]),
                fit_start=meta.get("timewindow_start"), fit_end=meta.get("timewindow_end"),
                name=name)


def build_H(P, t_max):
    """Cumulative integral of the time decay on a log mesh."""
    mesh = np.concatenate([[0.0], np.logspace(-8, np.log10(t_max * 1.05), 6000)])
    f = lambda t: np.exp(-t / P["tau"]) * (t + P["c"]) ** (-(1.0 + P["omega"]))
    vals = np.zeros(len(mesh))
    for i in range(1, len(mesh)):
        vals[i] = vals[i - 1] + quad(f, mesh[i - 1], mesh[i], limit=200)[0]
    return lambda T: np.interp(np.maximum(T, 0.0), mesh, vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ComCat_25",
                    help="ETAS output_data_<name> stem to use for parameters")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--start", default="2020-01-17")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--out", default=None,
                    help="output directory for forward scores; default runs/forward_etas[_<name>]")
    args = ap.parse_args()
    P = load_params(args.name)

    if args.validate:
        cat = pd.read_csv(output_dir(args.name) / "augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
        t_start, t_end = pd.Timestamp("2007-01-01"), pd.Timestamp("2020-01-17")
    else:
        cat = pd.read_csv(EXT, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
        cat["magnitude"] = round_half_up(cat["magnitude"].to_numpy())
        cat = cat[cat["magnitude"] >= P["mc"]].reset_index(drop=True)
        t_start, t_end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    t_days = (cat["time"] - P["aux"]).dt.total_seconds().to_numpy() / 86400.0
    lat = np.radians(cat["latitude"].to_numpy())
    lon = np.radians(cat["longitude"].to_numpy())
    m = cat["magnitude"].to_numpy()
    cos_lat = np.cos(lat)
    targets = np.flatnonzero(((cat["time"] >= t_start) & (cat["time"] < t_end)).to_numpy())
    n_t = len(targets)
    print(f"{len(cat)} events, {n_t} targets in [{t_start.date()} .. {t_end.date()})")

    prod = P["k0"] * np.exp(P["a"] * (m - P["mc"]))
    dm = P["d"] * np.exp(P["gamma"] * (m - P["mc"]))
    zint = np.pi / (P["rho"] * np.power(dm, P["rho"]))
    H = build_H(P, t_days[targets[-1]] + 1.0)

    lam = np.empty(n_t); lam_star = np.empty(n_t); cum = np.empty(n_t)
    for b0 in range(0, n_t, BLOCK):
        idx = targets[b0:b0 + BLOCK]
        hi = idx[-1]
        dlat = lat[idx, None] - lat[None, :hi]
        dlon = lon[idx, None] - lon[None, :hi]
        h = np.sin(dlat / 2.0) ** 2 + cos_lat[idx, None] * cos_lat[None, :hi] * np.sin(dlon / 2.0) ** 2
        r2 = np.square(2.0 * P["R"] * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))
        dt = np.maximum(t_days[idx, None] - t_days[None, :hi], 0.0)
        mask = np.arange(hi)[None, :] < idx[:, None]
        w = np.where(mask, prod[None, :hi] * np.exp(-dt / P["tau"]) * np.power(dt + P["c"], -(1.0 + P["omega"])), 0.0)
        sl = slice(b0, b0 + len(idx))
        lam[sl] = P["mu"] + (w * np.power(r2 + dm[None, :hi], -(1.0 + P["rho"]))).sum(axis=1)
        lam_star[sl] = P["mu"] * P["area"] + (w * zint[None, :hi]).sum(axis=1)
        cum[sl] = (P["mu"] * P["area"] * t_days[idx]
                   + (np.where(mask, prod[None, :hi] * zint[None, :hi] * H(dt), 0.0)).sum(axis=1))
        if (b0 // BLOCK) % 25 == 0:
            print(f"  {b0}/{n_t}", flush=True)

    # anchor: Lambda from 0 to window start
    i0 = targets[0]
    t0 = (t_start - P["aux"]).total_seconds() / 86400.0
    anchor = (P["mu"] * P["area"] * t0
              + (prod[:i0] * zint[:i0] * H(t0 - t_days[:i0])).sum())
    int_lam = np.ediff1d(cum, to_begin=cum[0] - anchor)
    TLL = np.log(lam_star) - int_lam
    SLL = np.log(lam) - np.log(lam_star)

    if args.validate:
        ref_T = cat["TLL"].to_numpy()[targets]
        ref_S = cat["SLL"].to_numpy()[targets]
        eT = np.abs(TLL - ref_T); eS = np.abs(SLL - ref_S)
        print(f"TLL  max_abs_err {eT.max():.3e}  mean {eT.mean():.3e}")
        print(f"SLL  max_abs_err {eS.max():.3e}  mean {eS.mean():.3e}")
        print(f"mean TLL ours {TLL.mean():.4f} ref {ref_T.mean():.4f} | SLL ours {SLL.mean():.4f} ref {ref_S.mean():.4f}")
        return

    default_out = "runs/forward_etas" if args.name == "ComCat_25" else f"runs/forward_etas_{args.name}"
    out = Path(args.out or default_out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": cat["time"].iloc[targets].values, "magnitude": m[targets],
                  "TLL": TLL, "SLL": SLL}).to_csv(out / "per_event.csv", index=False)
    summary = {"window": [str(t_start), str(t_end)], "n": int(n_t),
               "tll": float(TLL.mean()), "sll": float(SLL.mean()),
               "nll": float(-(TLL.mean() + SLL.mean())),
               "etas_name": args.name,
               "fit_window": [P.get("fit_start"), P.get("fit_end")],
               "params_frozen_from": f"{args.name} inversion"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
