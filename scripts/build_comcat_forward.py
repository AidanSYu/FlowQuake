"""Extend the EarthquakeNPP ComCat catalog forward for out-of-time evaluation.

Replicates the benchmark's exact construction recipe (reference/Datasets/ComCat/
README.md) for the window AFTER the benchmark's test_nll_end (2020-01-17):
  * USGS ComCat download, bbox 30.79..44.45 lat / -130..-110 lon, M>=2.0,
    eventtype=earthquake (same query as plot_utils.download_USGS_data).
  * RELM/CSEP California polygon filter (california_shape.npy, [lat,lon] verts).
  * Mcut 2.5.
  * Duplicate-location/time jitter (same ranges, seed 42).
  * Azimuthal-equidistant projection about the ORIGINAL catalog's center
    (mean lat/lon of the 1971-2020 processed catalog), verified against the
    stored x,y before use.

Outputs:
  reference/Datasets/ComCat_forward/ComCat_forward_catalog.csv   (forward only)
  reference/Datasets/ComCat_extended/ComCat_extended_catalog.csv (1971->end, for
      history-conditioned scoring; identical rows to the original through
      2020-01-17, forward rows appended with continuing ids)
  reference/Datasets/ComCat_forward/ComCat_forward_meta.json

The frozen production checkpoint + the frozen ComCat_25 ETAS inversion are then
scored on the forward window. This is a retrospective out-of-time replication,
not a registered prospective forecast.

Run (CPU/network): python scripts/build_comcat_forward.py
"""
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
from matplotlib.path import Path as MplPath

sys.path.append("reference/Datasets")
from plot_utils import azimuthal_equidistant_projection  # noqa: E402

FORWARD_START = "2020-01-17 00:00:00"   # == benchmark test_nll_end (exclusive there)
FORWARD_END = "2026-07-01 00:00:00"
MAX_LAT, MIN_LAT, MAX_LON, MIN_LON = 44.45, 30.79, -110.0, -130.0
DL_MAG, MCUT = 2.0, 2.5
ORIG = Path("reference/Datasets/ComCat/ComCat_catalog.csv")
SHAPE = Path("reference/Datasets/ComCat/california_shape.npy")
OUT_FWD = Path("reference/Datasets/ComCat_forward")
OUT_EXT = Path("reference/Datasets/ComCat_extended")

URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
       "starttime={s}&endtime={e}&maxlatitude={a}&minlatitude={b}"
       "&maxlongitude={c}&minlongitude={d}&minmagnitude={m}"
       "&eventtype=earthquake&orderby=time-asc")
_CTX = ssl.create_default_context()


def fetch_window(s: datetime, e: datetime, depth: int = 0) -> pd.DataFrame:
    """One USGS CSV window; retry, then recursively halve on persistent errors
    or suspiciously large responses (the API caps at 20000 rows)."""
    url = URL.format(s=s.strftime("%Y-%m-%dT%H:%M:%S"), e=e.strftime("%Y-%m-%dT%H:%M:%S"),
                     a=MAX_LAT, b=MIN_LAT, c=MAX_LON, d=MIN_LON, m=DL_MAG)
    splittable = (e - s) > timedelta(days=2)
    last = None
    for attempt in range(4):
        try:
            txt = urllib.request.urlopen(url, timeout=240, context=_CTX).read().decode("utf-8", "replace")
            df = pd.read_csv(io.StringIO(txt))
            if len(df) >= 19000 and splittable:
                break  # near the row cap -> split below
            return df
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(4 * (attempt + 1))
    if splittable:
        mid = s + (e - s) / 2
        print(f"    split {s.date()}..{e.date()} ({type(last).__name__ if last else 'row-cap'})", flush=True)
        return pd.concat([fetch_window(s, mid, depth + 1), fetch_window(mid, e, depth + 1)],
                         ignore_index=True)
    raise RuntimeError(f"window {s}..{e} failed permanently: {last}")


def main() -> None:
    orig = pd.read_csv(ORIG, parse_dates=["time"])
    lat0, lon0 = orig["latitude"].mean(), orig["longitude"].mean()
    # Verify we recovered the projection center: reproject stored lat/lon.
    vx, vy = azimuthal_equidistant_projection(orig["latitude"].values, orig["longitude"].values, lat0, lon0)
    err = float(np.max(np.hypot(vx - orig["x"].values, vy - orig["y"].values)))
    print(f"projection center ({lat0:.6f}, {lon0:.6f})  max reprojection error {err:.6f} km")
    if err > 0.05:
        raise SystemExit("projection center mismatch -- do NOT proceed")

    frames = []
    cur, end = datetime.fromisoformat(FORWARD_START), datetime.fromisoformat(FORWARD_END)
    while cur < end:
        ce = min(cur + timedelta(days=92), end)
        t0 = time.time()
        df = fetch_window(cur, ce)
        frames.append(df)
        print(f"  {cur.date()}..{ce.date()}  {len(df):5d}  {time.time()-t0:5.1f}s", flush=True)
        cur = ce
    raw = pd.concat(frames, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None)
    raw = raw.rename(columns={"mag": "magnitude"})
    raw = raw.dropna(subset=["time", "latitude", "longitude", "magnitude"])
    raw = raw.drop_duplicates(subset="id").sort_values("time").reset_index(drop=True)
    n_raw = len(raw)

    # RELM polygon filter -- polygon vertices are [lat, lon] (README convention).
    poly = MplPath(np.load(SHAPE))
    inside = poly.contains_points(np.column_stack([raw["latitude"], raw["longitude"]]), radius=1e-9)
    cat = raw[inside].copy()
    n_poly = len(cat)

    # Strict forward window + magnitude cut.
    cat = cat[(cat["time"] >= FORWARD_START) & (cat["time"] < FORWARD_END)]
    cat = cat[cat["magnitude"] >= MCUT].copy()
    n_mag = len(cat)

    # Duplicate jitter, same procedure/ranges as the README (seed 42).
    rng = np.random.default_rng(42)
    sj, tj = 0.005, 0.1 / 86400.0
    while True:
        dup = cat.duplicated(subset=["longitude", "latitude"], keep=False)
        if not dup.any():
            break
        k = int(dup.sum())
        cat.loc[dup, "longitude"] += rng.uniform(-sj / 2, sj / 2, k)
        cat.loc[dup, "latitude"] += rng.uniform(-sj / 2, sj / 2, k)
        print(f"  jittered {k} duplicate locations")
    while True:
        dup = cat.duplicated(subset=["time"], keep=False)
        if not dup.any():
            break
        k = int(dup.sum())
        cat.loc[dup, "time"] += pd.to_timedelta(rng.uniform(-tj / 2, tj / 2, k), unit="s")
        print(f"  jittered {k} duplicate times")
    cat = cat.sort_values("time").reset_index(drop=True)

    x, y = azimuthal_equidistant_projection(cat["latitude"].values, cat["longitude"].values, lat0, lon0)
    first_id = int(orig["id"].max()) + 1
    fwd = pd.DataFrame({"id": np.arange(first_id, first_id + len(cat)),
                        "time": cat["time"], "longitude": cat["longitude"],
                        "latitude": cat["latitude"], "magnitude": cat["magnitude"],
                        "x": x, "y": y,
                        "magtype": cat.get("magType", pd.Series([""] * len(cat))).fillna("").str.upper().values})

    OUT_FWD.mkdir(parents=True, exist_ok=True)
    OUT_EXT.mkdir(parents=True, exist_ok=True)
    fwd.to_csv(OUT_FWD / "ComCat_forward_catalog.csv", index=False)
    ext = pd.concat([orig, fwd.drop(columns=["magtype"])], ignore_index=True)
    ext.to_csv(OUT_EXT / "ComCat_extended_catalog.csv", index=False)

    mt = fwd["magtype"].value_counts().head(8).to_dict()
    meta = {"forward_start": FORWARD_START, "forward_end": FORWARD_END,
            "projection_center": [float(lat0), float(lon0)],
            "reprojection_err_km": err, "mcut": MCUT,
            "n_raw_download": int(n_raw), "n_in_polygon": int(n_poly),
            "n_final_forward": int(len(fwd)), "n_extended_total": int(len(ext)),
            "magtype_mix": {k: int(v) for k, v in mt.items()},
            "largest": fwd.nlargest(8, "magnitude")[["time", "magnitude", "latitude", "longitude"]]
                          .astype(str).to_dict("records")}
    (OUT_FWD / "ComCat_forward_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nforward events (M>={MCUT}, in polygon): {len(fwd)}   extended total: {len(ext)}")
    print("largest forward events:")
    for r in meta["largest"]:
        print("  ", r["time"], r["magnitude"], r["latitude"], r["longitude"])


if __name__ == "__main__":
    main()
