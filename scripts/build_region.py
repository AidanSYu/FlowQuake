"""Build an EarthquakeNPP-format catalog for a new region from USGS ComCat,
the same source/pipeline as the repo's Japan catalog. Downloads in windows,
dedupes, projects lat/lon -> x,y km (azimuthal equidistant about the region
centroid), and writes reference/Datasets/<name>/<name>_catalog.csv plus a
bounding-box region shape (<name>_shape.npy) for the ETAS inversion.

Run: python scripts/build_region.py Chile -40 -17 -76 -66 --dl-mag 3.5
"""
import argparse, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd

sys.path.append("reference/Datasets")
from plot_utils import azimuthal_equidistant_projection  # noqa

USGS = ("https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
        "starttime={s}&endtime={e}&minlatitude={a}&maxlatitude={b}"
        "&minlongitude={c}&maxlongitude={d}&minmagnitude={m}&eventtype=earthquake&orderby=time-asc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("minlat", type=float); ap.add_argument("maxlat", type=float)
    ap.add_argument("minlon", type=float); ap.add_argument("maxlon", type=float)
    ap.add_argument("--start", default="1990-01-01"); ap.add_argument("--end", default="2020-01-01")
    ap.add_argument("--dl-mag", type=float, default=3.5)
    ap.add_argument("--window-days", type=int, default=180)
    args = ap.parse_args()
    import urllib.request

    out_dir = Path("reference/Datasets") / args.name; out_dir.mkdir(parents=True, exist_ok=True)
    cur = datetime.fromisoformat(args.start); end = datetime.fromisoformat(args.end)
    frames = []
    while cur < end:
        ce = min(cur + timedelta(days=args.window_days), end)
        url = USGS.format(s=cur.strftime("%Y-%m-%dT%H:%M:%S"), e=ce.strftime("%Y-%m-%dT%H:%M:%S"),
                          a=args.minlat, b=args.maxlat, c=args.minlon, d=args.maxlon, m=args.dl_mag)
        for attempt in range(4):
            try:
                df = pd.read_csv(urllib.request.urlopen(url, timeout=60)); break
            except Exception as e:
                if attempt == 3: print("  fail", cur.date(), e); df = pd.DataFrame(); break
                time.sleep(2)
        if len(df): frames.append(df[["time", "latitude", "longitude", "mag", "id"]])
        cur = ce
    cat = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id")
    cat = cat.dropna(subset=["latitude", "longitude", "mag", "time"])
    cat["time"] = pd.to_datetime(cat["time"], utc=True).dt.tz_localize(None)
    cat = cat.sort_values("time").reset_index(drop=True)

    lat0, lon0 = cat["latitude"].mean(), cat["longitude"].mean()
    x, y = azimuthal_equidistant_projection(cat["latitude"].values, cat["longitude"].values, lat0, lon0)
    out = pd.DataFrame({"id": range(len(cat)), "time": cat["time"],
                        "longitude": cat["longitude"], "latitude": cat["latitude"],
                        "magnitude": cat["mag"], "x": x, "y": y})
    out.to_csv(out_dir / f"{args.name}_catalog.csv", index=False)

    # bbox region shape as LAT/LON corners (matches Japan_shape.npy; ETAS
    # projects internally and computes region area from these).
    corners_ll = np.array([[args.maxlat, args.minlon], [args.maxlat, args.maxlon],
                           [args.minlat, args.maxlon], [args.minlat, args.minlon]],
                          dtype=float)
    np.save(out_dir / f"{args.name}_shape.npy", corners_ll)

    print(f"{args.name}: {len(out)} events  {out.time.min()} -> {out.time.max()}")
    for mc in [3.5, 4.0, 4.5]:
        print(f"  mc>={mc}: {(out.magnitude >= mc).sum()}")
    print(f"  wrote {out_dir}/{args.name}_catalog.csv + shape (center {lat0:.2f},{lon0:.2f})")


if __name__ == "__main__":
    main()
