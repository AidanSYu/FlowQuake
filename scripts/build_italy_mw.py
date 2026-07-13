"""Build an Mw-homogenized Italy catalog for the magnitude-robustness re-run.

Reads the re-fetched INGV catalog (which carries a `magtype` column: ML, Md, Mw,
mb, ...) and converts every magnitude to moment magnitude Mw via PUBLISHED,
type-specific empirical relations (see CONVERSIONS below; coefficients are filled
from the cited sources). Events already reported as Mw are kept as-is. The output
is a standard EarthquakeNPP-format catalog (id,time,longitude,latitude,magnitude,
x,y,magtype) plus a shape file and a meta sidecar with the recomputed Mc/b-value.

This is the input for a self-consistent Mw re-run: ETAS is inverted and FlowQuake
is trained on this homogenized catalog, and the temporal gain dT = tll_FQ - TLL_ETAS
is compared to the as-reported Italy result (+0.071) to confirm the dense-catalog
temporal win is not an artifact of magnitude-type heterogeneity.

Run: python scripts/build_italy_mw.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("reference/Datasets")
from plot_utils import azimuthal_equidistant_projection  # noqa

RAW = Path("reference/Datasets/Italy_mw_raw/Italy_mw_raw_catalog.csv")
OUT_DIR = Path("reference/Datasets/Italy_Mw")
BBOX = (36.0, 47.0, 6.0, 19.0)  # minlat, maxlat, minlon, maxlon (matches Italy_25)

# --- PUBLISHED conversion relations (Gasperini, Lolli & Vannucci 2013, BSSA) ---
# General orthogonal (chi-square) regression, form Mw = b*M + a, per magnitude
# TYPE (ML, Md). We apply a SINGLE, TIME-CONSISTENT relation per type across the
# whole catalog (the modern BSI/ISIDe-era relations, which also natively cover the
# 2009-2020 validation/test window). This is deliberate: a forecasting model reads
# the catalog as a continuous time series, so a magnitude scale that is consistent
# in time is required. The per-EPOCH HORUS/CPTI15 recipe (different CSTI/CSI/BSI
# coefficients before 1997 / 1997-2002 / >=2003) is correct for a static parametric
# catalogue but injects artificial time discontinuities (e.g. the same Md=3.0 ->
# Mw 3.38 in 2002 but 2.90 in 2003), which corrupt a temporal NPP's learned
# triggering features. The reported magtype still selects ML vs Md (Italy flips
# Md->ML ~2005-04-16); each type uses one fixed relation, so the Mw scale is
# monotonic and time-consistent.
CITATION = ("Gasperini, Lolli & Vannucci (2013), BSSA 103(4), 2227-2246, "
            "doi:10.1785/0120120356 (BSI/ISIDe-era GOR relations, applied "
            "time-uniformly per magnitude type for temporal consistency)")

# (slope b, intercept a) per magnitude type -- single relation, all years
_REL = {"ML": (1.0, 0.08), "MD": (1.456, -1.472)}  # Gasperini 2013 BSI [2.22],[2.24]


def to_mw(mag, magtype, year=None):
    t = (magtype or "?").upper()
    if t.startswith("MW") or t in ("MWW", "MWC", "MWB", "MWR"):
        return mag                                   # already Mw
    kind = "MD" if t.startswith("MD") else "ML"      # bare/ML/mb-rare -> ML branch
    b, a = _REL[kind]
    return b * mag + a
# -----------------------------------------------------------------------------


def estimate_mc(mags, mbin=0.1):
    mags = np.asarray(mags, float)
    edges = np.arange(np.floor(mags.min() * 10) / 10, mags.max() + mbin, mbin)
    counts, _ = np.histogram(mags, bins=edges)
    centers = edges[:-1] + mbin / 2
    mc_maxc = centers[np.argmax(counts)] + 0.2
    above = mags[mags >= mc_maxc - 1e-9]
    b = np.log10(np.e) / (above.mean() - (mc_maxc - mbin / 2)) if len(above) > 1 else float("nan")
    return float(mc_maxc), float(b), int(len(above))


def main():
    if CITATION is None:
        sys.exit("Refusing to run: conversion coefficients/CITATION not filled in yet.")
    if not RAW.exists():
        sys.exit(f"Re-fetched catalog not found: {RAW} (run build_region.py Italy_mw_raw first)")
    df = pd.read_csv(RAW, parse_dates=["time"])
    if "magtype" not in df.columns:
        sys.exit("Re-fetched catalog has no magtype column.")
    df = df.dropna(subset=["magnitude", "latitude", "longitude", "time"]).reset_index(drop=True)
    mix_before = df["magtype"].fillna("?").str.upper().value_counts().head(10).to_dict()

    df["mw"] = [to_mw(m, t, y) for m, t, y in
                zip(df["magnitude"].to_numpy(float), df["magtype"].fillna("?"), df["time"].dt.year.to_numpy())]
    df = df.dropna(subset=["mw"]).sort_values("time").reset_index(drop=True)

    lat0, lon0 = df["latitude"].mean(), df["longitude"].mean()
    x, y = azimuthal_equidistant_projection(df["latitude"].values, df["longitude"].values, lat0, lon0)
    out = pd.DataFrame({"id": range(len(df)), "time": df["time"],
                        "longitude": df["longitude"], "latitude": df["latitude"],
                        "magnitude": df["mw"].round(3), "x": x, "y": y,
                        "magtype": "Mw(" + df["magtype"].fillna("?").str.upper() + ")"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "Italy_Mw_catalog.csv", index=False)
    corners = np.array([[BBOX[1], BBOX[2]], [BBOX[1], BBOX[3]], [BBOX[0], BBOX[3]], [BBOX[0], BBOX[2]]], float)
    np.save(OUT_DIR / "Italy_Mw_shape.npy", corners)

    mc, b, n_above = estimate_mc(out["magnitude"].values)
    meta = {"name": "Italy_Mw", "source": "INGV (re-fetched), homogenized to Mw",
            "conversion": CITATION, "n_events": int(len(out)),
            "magtype_mix_before": {k: int(v) for k, v in mix_before.items()},
            "Mc_maxc": round(mc, 2), "b_value": round(b, 3), "n_above_Mc": n_above,
            "counts_by_mc": {str(m): int((out.magnitude >= m).sum()) for m in [2.5, 3.0, 3.5, 4.0]}}
    (OUT_DIR / "Italy_Mw_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Italy_Mw: {len(out)} events  Mc(maxc)={mc:.2f}  b={b:.2f}")
    print(f"  magtype mix (before): {mix_before}")
    print(f"  conversion: {CITATION}")
    print(f"  counts: {meta['counts_by_mc']}")
    print(f"  wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
