"""Stage-0 harness check: reproduce ETAS's per-event lambd / lambd_star / SLL
on ComCat_25 EXACTLY from parameters_0.json, outside the `etas` package.

Why: the neural-ETAS spatial head is initialized at the inverted ETAS
parameters so that at init it IS ETAS; every nat gained in training is then a
real modeling gain. That guarantee is only as good as this reproduction. We
recompute, for every test-window event,
    lambda(t_i, s_i) = mu + sum_{j<i} k0 e^{a(m_j-mc)} (r_ij^2 + d e^{g(m_j-mc)})^{-(1+rho)}
                             e^{-dt_ij/tau} (dt_ij + c)^{-(1+omega)}
    lambda*(t_i)     = mu A + sum_{j<i} k0 e^{a(m_j-mc)} pi/(rho (d e^{g(m_j-mc)})^rho)
                             e^{-dt_ij/tau} (dt_ij + c)^{-(1+omega)}
with haversine distances (R=6378.1) over ALL prior events (aux included, no
truncation), and compare against the package's stored columns.

Run (CPU, ~minutes): python scripts/etas_sll_repro.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reference/Experiments/ETAS/output_data_ComCat_25")
BLOCK = 256


def main() -> None:
    meta = json.loads((OUT / "parameters_0.json").read_text())
    p = meta["final_parameters"]
    mu = 10.0 ** p["log10_mu"]
    k0 = 10.0 ** p["log10_k0"]
    c = 10.0 ** p["log10_c"]
    tau = 10.0 ** p["log10_tau"]
    d = 10.0 ** p["log10_d"]
    a, omega, gamma, rho = p["a"], p["omega"], p["gamma"], p["rho"]
    mc, area, R = meta["mc"], meta["area"], meta["earth_radius"]

    cat = pd.read_csv(OUT / "augmented_catalog.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    assert (cat["index_from_zero"].to_numpy() == np.arange(len(cat))).all(), "index_from_zero mismatch"
    t_days = (cat["time"] - cat["time"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
    lat = np.radians(cat["latitude"].to_numpy())
    lon = np.radians(cat["longitude"].to_numpy())
    m = cat["magnitude"].to_numpy()

    test = np.flatnonzero(np.isfinite(cat["SLL"].to_numpy()))
    print(f"catalog {len(cat)} events, {len(test)} test targets "
          f"({cat['time'].iloc[test[0]]} .. {cat['time'].iloc[test[-1]]})")

    prod = k0 * np.exp(a * (m - mc))               # aftershock_number(m_j)
    dm = d * np.exp(gamma * (m - mc))              # aftershock_zone(m_j), km^2
    zint = np.pi / (rho * np.power(dm, rho))       # space_integral(m_j)

    lam = np.empty(len(test))
    lam_star = np.empty(len(test))
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    for b0 in range(0, len(test), BLOCK):
        idx = test[b0:b0 + BLOCK]
        hi = idx[-1]                              # all sources for this block are j < hi
        # pairwise haversine^2 target-block x sources[:hi]
        dlat = lat[idx, None] - lat[None, :hi]
        dlon = lon[idx, None] - lon[None, :hi]
        h = np.sin(dlat / 2.0) ** 2 + cos_lat[idx, None] * cos_lat[None, :hi] * np.sin(dlon / 2.0) ** 2
        r2 = np.square(2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))
        dt = t_days[idx, None] - t_days[None, :hi]
        mask = np.arange(hi)[None, :] < idx[:, None]
        w = prod[None, :hi] * np.exp(-dt / tau) / np.power(dt + c, 1.0 + omega)
        w = np.where(mask, w, 0.0)
        lam[b0:b0 + len(idx)] = mu + (w * np.power(r2 + dm[None, :hi], -(1.0 + rho))).sum(axis=1)
        lam_star[b0:b0 + len(idx)] = mu * area + (w * zint[None, :hi]).sum(axis=1)
        if (b0 // BLOCK) % 20 == 0:
            print(f"  block {b0}/{len(test)}", flush=True)

    sll = np.log(lam) - np.log(lam_star)
    ref_lam = cat["lambd"].to_numpy()[test]
    ref_star = cat["lambd_star"].to_numpy()[test]
    ref_sll = cat["SLL"].to_numpy()[test]

    def report(name, ours, ref):
        rel = np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-300)
        print(f"{name:10} max_rel_err {rel.max():.3e}   mean_rel_err {rel.mean():.3e}")

    report("lambd", lam, ref_lam)
    report("lambd*", lam_star, ref_star)
    dabs = np.abs(sll - ref_sll)
    print(f"SLL        max_abs_err {dabs.max():.3e}   mean_abs_err {dabs.mean():.3e}")
    print(f"mean SLL   ours {sll.mean():.6f}   reference {ref_sll.mean():.6f}   (target -8.6898)")
    ok = dabs.max() < 1e-4
    print("EXACT MATCH" if ok else "MISMATCH -- investigate before building the head")
    json.dump({"max_abs_sll_err": float(dabs.max()), "mean_sll_ours": float(sll.mean()),
               "mean_sll_ref": float(ref_sll.mean()), "n_test": int(len(test)), "match": bool(ok)},
              open("runs/etas_sll_repro.json", "w"), indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
