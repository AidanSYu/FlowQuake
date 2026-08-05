"""Build a LOW-COMPLETENESS California catalog for the scaling curve.

The benchmark ships ComCat at mc 2.5. `MOONSHOT.md` needs the x-axis to run well
below that — the whole claim is about what the small events buy — so this pulls
the same region with the same recipe at a much lower magnitude floor.

Recipe is the EarthquakeNPP construction (reference/Datasets/ComCat/README.md),
matched so the resulting catalog is interchangeable with the benchmark's above
mc 2.5, but **self-contained**: the azimuthal-equidistant projection is
implemented here rather than imported from `reference/Datasets/plot_utils.py`,
so this runs on a bare box with no benchmark clone. `--verify-against` checks
the projection reproduces the benchmark's stored x/y when that clone IS present.

    USGS ComCat query, bbox 30.79..44.45 lat / -130..-110 lon, eventtype=earthquake
    -> RELM/CSEP California polygon filter
    -> magnitude floor
    -> duplicate location/time jitter (seed 42)
    -> azimuthal-equidistant projection about the catalog centre

**Completeness is not constant in time**, and this is the single most important
caveat for the curve. California is complete to ~2.5 back to 1971 but only to
~1.0-1.5 after network upgrades around 2000. `--completeness-report` estimates
Mc per era by maximum curvature so the curve's low-mc points can be restricted
to eras where that mc is actually meaningful. A point at mc 1.0 trained on an
era that is only complete to 2.0 is measuring missing data, not information.

Usage:
    python scripts/build_comcat_lowmc.py --min-mag 1.0 --completeness-report
"""
from __future__ import annotations

import argparse
import io
import json
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MAX_LAT, MIN_LAT, MAX_LON, MIN_LON = 44.45, 30.79, -110.0, -130.0
USGS = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
        "&starttime={s}&endtime={e}&minlatitude={mila}&maxlatitude={mala}"
        "&minlongitude={milo}&maxlongitude={malo}&minmagnitude={mm}"
        "&eventtype=earthquake&orderby=time-asc")
R_EARTH_KM = 6371.0


def azimuthal_equidistant(lat, lon, lat0, lon0):
    """Azimuthal-equidistant projection about (lat0, lon0), returning (x, y) km
    in EarthquakeNPP's convention: **x is northing and y is easting**.

    Same projection the benchmark applies; implemented here so the script has no
    dependency on the benchmark clone. Verified against the shipped x/y by
    `--verify-against` (matches to 5e-12 km).
    """
    p, l = np.radians(lat), np.radians(lon)
    p0, l0 = math_radians(lat0), math_radians(lon0)
    cos_c = np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(l - l0)
    cos_c = np.clip(cos_c, -1.0, 1.0)
    c = np.arccos(cos_c)
    k = np.where(np.abs(c) < 1e-12, 1.0, c / np.where(np.sin(c) == 0, 1.0, np.sin(c)))
    east = R_EARTH_KM * k * np.cos(p) * np.sin(l - l0)
    north = R_EARTH_KM * k * (np.cos(p0) * np.sin(p)
                              - np.sin(p0) * np.cos(p) * np.cos(l - l0))
    # EarthquakeNPP's convention is (x, y) = (NORTHING, EASTING), not the usual
    # (easting, northing). Verified by solving for the transform that reproduces
    # the shipped ComCat_catalog.csv: with the swap and centre = catalog mean
    # lat/lon it matches to 5.1e-12 km; without it, 1,926 km. Getting this
    # backwards silently puts a new catalog in a mirrored frame.
    return north, east


def math_radians(v):
    return np.radians(v)


def fetch(start: str, end: str, min_mag: float, chunk_days: float = 30.0,
          retries: int = 3, cache: Path | None = None,
          min_window_days: float = 1.0 / 48.0) -> pd.DataFrame:
    """Download in adaptive windows, checkpointing as it goes.

    USGS returns at most 20,000 rows and answers an over-large query with
    **HTTP 400**, not with a truncated body. The first version treated 400 as a
    transient error and retried the same doomed query, then died — which is what
    happened at 2019-06-24, the window containing the Ridgecrest M7.1 sequence
    whose M>=1.0 aftershocks run to tens of thousands in days. 298 of ~305
    chunks were already fetched and all of them were lost, because nothing was
    written until the end.

    So: a 400 (or a full 20k body) halves the window and retries, recursively,
    down to `min_window_days`; and every completed chunk is appended to `cache`
    so a late failure costs one window, not the whole download. Dense windows
    recover to the base width afterwards, so one aftershock sequence does not
    force sub-hour queries across three decades.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    base = float(chunk_days)
    cur, end_dt = datetime.fromisoformat(start), datetime.fromisoformat(end)
    out, wrote_header = [], False
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)

    def get(a: datetime, b: datetime) -> pd.DataFrame | None:
        """None means 'too big, split me'."""
        url = USGS.format(s=a.strftime("%Y-%m-%dT%H:%M:%S"),
                          e=b.strftime("%Y-%m-%dT%H:%M:%S"),
                          mila=MIN_LAT, mala=MAX_LAT, milo=MIN_LON, malo=MAX_LON,
                          mm=min_mag)
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, context=ctx, timeout=300) as r:
                    txt = r.read().decode()
                d = pd.read_csv(io.StringIO(txt)) if txt.strip() else pd.DataFrame()
                return None if len(d) >= 19500 else d
            except urllib.error.HTTPError as exc:
                if exc.code == 400:
                    return None            # over the 20k cap: split
                if attempt == retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))
            except Exception as exc:                       # noqa: BLE001
                if attempt == retries - 1:
                    raise
                print(f"    retry {attempt+1} ({exc})", flush=True)
                time.sleep(5 * (attempt + 1))
        return pd.DataFrame()

    def pull(a: datetime, b: datetime, depth: int = 0):
        d = get(a, b)
        if d is None:
            span = (b - a).total_seconds() / 86400.0
            if span <= min_window_days:
                print(f"  ! {a} .. {b} still over the cap at the minimum window; "
                      f"skipping (raise --min-mag for this era)", flush=True)
                return
            mid = a + (b - a) / 2
            print(f"  {a.date()} span {span:.3f}d over cap -> splitting", flush=True)
            pull(a, mid, depth + 1); pull(mid, b, depth + 1)
            return
        nonlocal wrote_header
        if len(d):
            out.append(d)
            if cache is not None:
                d.to_csv(cache, mode="a", header=not wrote_header and
                         not cache.exists(), index=False)
                wrote_header = True
        print(f"  {a.date()} -> {b.date()}: {len(d):>6} rows"
              f"{'' if depth == 0 else f'  (split depth {depth})'}", flush=True)

    while cur < end_dt:
        nxt = min(cur + timedelta(days=base), end_dt)
        pull(cur, nxt)
        cur = nxt
    if not out:
        raise SystemExit("no data returned")
    return pd.concat(out, ignore_index=True)


def relm_filter(df: pd.DataFrame, shape_path: Path | None) -> pd.DataFrame:
    """RELM/CSEP California polygon. Falls back to the bbox when the polygon
    file is absent, and says so — the polygon changes the count materially."""
    if shape_path is None or not shape_path.exists():
        print("  ! california_shape.npy not found: using the bounding box only.\n"
              "    This is NOT the benchmark region; do not mix the two in one "
              "comparison.", flush=True)
        return df
    from matplotlib.path import Path as MplPath
    verts = np.load(shape_path)          # [lat, lon]
    poly = MplPath(np.column_stack([verts[:, 1], verts[:, 0]]))
    keep = poly.contains_points(np.column_stack([df["longitude"], df["latitude"]]))
    print(f"  RELM polygon: {int(keep.sum()):,}/{len(df):,} retained", flush=True)
    return df[keep].reset_index(drop=True)


def jitter_duplicates(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """The benchmark's duplicate handling: identical (time, lat, lon) rows get a
    tiny perturbation so the point process stays simple (no coincident points)."""
    rng = np.random.default_rng(seed)
    # pandas >= 3 refuses to write datetime64[ns] into a datetime64[us] column,
    # so normalise the resolution before adding sub-second jitter.
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).astype("datetime64[ns]")
    dup = df.duplicated(subset=["time", "latitude", "longitude"], keep=False)
    n = int(dup.sum())
    if n:
        df.loc[dup, "latitude"] = df.loc[dup, "latitude"] + rng.uniform(-1e-4, 1e-4, n)
        df.loc[dup, "longitude"] = df.loc[dup, "longitude"] + rng.uniform(-1e-4, 1e-4, n)
        df.loc[dup, "time"] = df.loc[dup, "time"] + pd.to_timedelta(
            rng.uniform(0, 1e-3, n), unit="s")
        print(f"  jittered {n:,} duplicate rows", flush=True)
    return df


def maxcurv_mc(mags: np.ndarray, dm: float = 0.1, correction: float = 0.2) -> float:
    """Maximum-curvature Mc with the standard +0.2 correction (Woessner & Wiemer)."""
    if len(mags) < 100:
        return float("nan")
    bins = np.arange(np.floor(mags.min() * 10) / 10,
                     np.ceil(mags.max() * 10) / 10 + dm, dm)
    hist, edges = np.histogram(mags, bins=bins)
    return float(edges[int(np.argmax(hist))] + correction)


def completeness_report(df: pd.DataFrame, era_years: int = 5) -> list[dict]:
    """Mc per era. The curve's low-mc points are only meaningful in eras whose
    Mc is at or below them."""
    df = df.copy()
    df["era"] = df["time"].dt.year // era_years * era_years
    rows = []
    for era, g in df.groupby("era"):
        rows.append({"era": f"{int(era)}-{int(era)+era_years-1}",
                     "n": int(len(g)), "mc_maxcurv": maxcurv_mc(g["magnitude"].to_numpy())})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mag", type=float, default=1.0)
    ap.add_argument("--start", default="1971-01-01")
    ap.add_argument("--end", default="2020-01-17")
    ap.add_argument("--out", default="reference/Datasets/ComCat_lowmc")
    ap.add_argument("--shape", default="reference/Datasets/ComCat/california_shape.npy")
    ap.add_argument("--chunk-days", type=float, default=30.0)
    ap.add_argument("--completeness-report", action="store_true")
    ap.add_argument("--verify-against", default=None,
                    help="benchmark ComCat_catalog.csv; checks the projection")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"raw_M{args.min_mag:g}.csv"
    if raw_path.exists():
        print(f"[cache] {raw_path}")
        raw = pd.read_csv(raw_path, parse_dates=["time"])
    else:
        print(f"[fetch] USGS ComCat M>={args.min_mag} {args.start}..{args.end}")
        raw = fetch(args.start, args.end, args.min_mag, args.chunk_days,
                    cache=out / f'_partial_M{args.min_mag:g}.csv')
        raw["time"] = pd.to_datetime(raw["time"], format="ISO8601", utc=True).dt.tz_localize(None)
        raw.to_csv(raw_path, index=False)
    print(f"[fetch] {len(raw):,} events")

    df = raw.dropna(subset=["latitude", "longitude", "mag"]).copy()
    df = df.rename(columns={"mag": "magnitude"})
    df = df.sort_values("time").reset_index(drop=True)
    df = relm_filter(df, Path(args.shape))
    df = jitter_duplicates(df).sort_values("time").reset_index(drop=True)

    lat0, lon0 = float(df["latitude"].mean()), float(df["longitude"].mean())
    df["x"], df["y"] = azimuthal_equidistant(df["latitude"].to_numpy(),
                                             df["longitude"].to_numpy(), lat0, lon0)
    df["id"] = np.arange(len(df))

    meta = {"min_mag": args.min_mag, "start": args.start, "end": args.end,
            "n_events": int(len(df)), "proj_center": [lat0, lon0],
            "bbox": [MIN_LAT, MAX_LAT, MIN_LON, MAX_LON],
            "relm_polygon": Path(args.shape).exists()}

    if args.verify_against and Path(args.verify_against).exists():
        ref = pd.read_csv(args.verify_against, parse_dates=["time"])
        c0 = [float(ref["latitude"].mean()), float(ref["longitude"].mean())]
        rx, ry = azimuthal_equidistant(ref["latitude"].to_numpy(),
                                       ref["longitude"].to_numpy(), c0[0], c0[1])
        err = float(np.max(np.hypot(rx - ref["x"].to_numpy(), ry - ref["y"].to_numpy())))
        meta["projection_max_abs_err_km"] = err
        print(f"[verify] projection reproduces the benchmark x/y to {err:.3e} km")
        if err > 1e-3:
            print("*** WARNING: projection mismatch. Do not mix this catalog with "
                  "the benchmark's in one comparison until resolved.")

    cat_path = out / "ComCat_lowmc_catalog.csv"
    df[["time", "latitude", "longitude", "magnitude", "x", "y", "id"]].to_csv(
        cat_path, index=False)
    print(f"[write] {cat_path}  ({len(df):,} events)")

    for c in (1.0, 1.5, 2.0, 2.5, 3.0):
        print(f"   mc {c}: {int((df['magnitude'] >= c).sum()):,}")

    if args.completeness_report:
        rows = completeness_report(df)
        meta["completeness_by_era"] = rows
        print(f"\n{'era':14}{'n':>10}{'Mc(maxcurv)':>13}")
        for r in rows:
            print(f"{r['era']:14}{r['n']:>10,}{r['mc_maxcurv']:>13.2f}")
        print("\nUse this to bound the curve: a point at mc X is only meaningful\n"
              "in eras whose Mc <= X. MOONSHOT.md invariant 1 requires the TARGET\n"
              "set to stay fixed, so restrict the TRAINING window, never the targets.")

    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"[write] {out/'meta.json'}")


if __name__ == "__main__":
    main()
