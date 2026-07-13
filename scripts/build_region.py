"""Build an EarthquakeNPP-format catalog for a region from an FDSN event service.

Two sources are supported:
  --source usgs : USGS ComCat (CSV)              -- as used for the repo's first pass.
  --source isc  : ISC reviewed Bulletin (text)   -- the AUTHORITATIVE, internationally
                  reviewed catalog. One consistent source across all foreign regions
                  removes inter-agency magnitude/completeness heterogeneity as a
                  confound for the cross-regime transfer claim.

Output: reference/Datasets/<name>/<name>_catalog.csv (id,time,longitude,latitude,
magnitude,x,y), a bounding-box <name>_shape.npy (lat/lon corners, for ETAS), and a
<name>_meta.json sidecar documenting source, magnitude-type mix, and an empirical
Mc estimate (maximum-curvature + b-value). Lat/lon -> x,y km is azimuthal equidistant
about the catalog centroid (same projection as every other catalog in the repo).

Run: python scripts/build_region.py Japan 22 46 122 150 --source isc --dl-mag 3.5
"""
import argparse, json, ssl, sys, time, urllib.request, urllib.error
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd

sys.path.append("reference/Datasets")
from plot_utils import azimuthal_equidistant_projection  # noqa

USGS = ("https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
        "starttime={s}&endtime={e}&minlatitude={a}&maxlatitude={b}"
        "&minlongitude={c}&maxlongitude={d}&minmagnitude={m}&eventtype=earthquake&orderby=time-asc")
# FDSN event services that return the standard pipe-delimited text schema.
FDSN_BASES = {
    "isc":    "http://www.isc.ac.uk/fdsnws/event/1/query",
    "ingv":   "https://webservices.ingv.it/fdsnws/event/1/query",
    "geonet": "https://service.geonet.org.nz/fdsnws/event/1/query",
}
FDSN_TMPL = ("{base}?starttime={s}&endtime={e}&minlatitude={a}&maxlatitude={b}"
             "&minlongitude={c}&maxlongitude={d}&minmagnitude={m}"
             "&format=text&orderby=time-asc")
# explicit &limit defeats default caps (INGV silently caps at 100); GeoNet 400s on
# any &limit, so we send none and rely on the window-splitter to stay under its cap.
FDSN_LIMIT = {"isc": 20000, "ingv": 20000, "geonet": 0}

# ISC FDSN returns one row per event (preferred origin + preferred magnitude).
# Cols: EventID|Time|Lat|Lon|Depth|Author|Catalog|Contributor|ContribID|MagType|Mag|MagAuthor|Loc|Type
SPLIT_AT = 4000          # if a window returns >= this, halve it (guards vs silent capping)
_NONEQ = ("explosion", "blast", "quarry", "mine", "mining", "nuclear", "rockburst", "collapse")
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE   # public catalog data; INGV's cert chain otherwise fails


def _get(url):
    return urllib.request.urlopen(url, timeout=240, context=_CTX).read().decode("utf-8", "replace")


def fetch_usgs(s, e, a, b, c, d, m):
    import io
    url = USGS.format(s=s.strftime("%Y-%m-%dT%H:%M:%S"), e=e.strftime("%Y-%m-%dT%H:%M:%S"),
                      a=a, b=b, c=c, d=d, m=m)
    df = pd.read_csv(io.StringIO(_get(url)))
    if not len(df):
        return pd.DataFrame(columns=["id", "time", "latitude", "longitude", "magnitude", "magtype", "eventtype"])
    return pd.DataFrame({"id": df["id"], "time": df["time"], "latitude": df["latitude"],
                         "longitude": df["longitude"], "magnitude": df["mag"],
                         "magtype": df.get("magType", ""), "eventtype": "earthquake"})


def _parse_isc(text):
    rows = []
    for ln in text.strip().split("\n"):
        if not ln or ln.startswith("#"):
            continue
        f = ln.split("|")
        if len(f) < 14:
            continue
        rows.append((f[0].strip(), f[1].strip(), f[2], f[3], f[10], f[9], f[13].strip().lower()))
    return pd.DataFrame(rows, columns=["id", "time", "latitude", "longitude", "magnitude", "magtype", "eventtype"])


def fetch_fdsn(base, limit, s, e, a, b, c, d, m, depth=0):
    """Fetch one FDSN-text window; recursively halve if truncated (>= SPLIT_AT)."""
    url = FDSN_TMPL.format(base=base, s=s.strftime("%Y-%m-%dT%H:%M:%S"),
                           e=e.strftime("%Y-%m-%dT%H:%M:%S"), a=a, b=b, c=c, d=d, m=m)
    if limit:
        url += f"&limit={limit}"
    splittable = (e - s) > timedelta(days=2)
    for attempt in range(4):
        try:
            df = _parse_isc(_get(url)); break
        except urllib.error.HTTPError as ex:
            if ex.code in (204, 404):    # no data
                return pd.DataFrame(columns=["id", "time", "latitude", "longitude", "magnitude", "magtype", "eventtype"])
            if ex.code == 413 and splittable:  # response too large (GeoNet) -> split now
                mid = s + (e - s) / 2
                print(f"    413 split {s.date()}..{e.date()}")
                return pd.concat([fetch_fdsn(base, limit, s, mid, a, b, c, d, m, depth + 1),
                                  fetch_fdsn(base, limit, mid, e, a, b, c, d, m, depth + 1)], ignore_index=True)
            if attempt == 3:
                raise
            time.sleep(3)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3)
    if len(df) >= SPLIT_AT and splittable:
        mid = s + (e - s) / 2
        print(f"    split {s.date()}..{e.date()} ({len(df)} rows) -> halving")
        lo = fetch_fdsn(base, limit, s, mid, a, b, c, d, m, depth + 1)
        hi = fetch_fdsn(base, limit, mid, e, a, b, c, d, m, depth + 1)
        return pd.concat([lo, hi], ignore_index=True)
    return df


def estimate_mc(mags, mbin=0.1):
    """Maximum-curvature Mc (Wiemer & Wyss) + b-value (Aki MLE) above it."""
    mags = np.asarray(mags, float)
    edges = np.arange(np.floor(mags.min() * 10) / 10, mags.max() + mbin, mbin)
    counts, _ = np.histogram(mags, bins=edges)
    centers = edges[:-1] + mbin / 2
    mc_maxc = centers[np.argmax(counts)] + 0.2  # MAXC + 0.2 correction (Woessner & Wiemer 2005)
    above = mags[mags >= mc_maxc - 1e-9]
    b = np.log10(np.e) / (above.mean() - (mc_maxc - mbin / 2)) if len(above) > 1 else float("nan")
    return float(mc_maxc), float(b), int(len(above))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("minlat", type=float); ap.add_argument("maxlat", type=float)
    ap.add_argument("minlon", type=float); ap.add_argument("maxlon", type=float)
    ap.add_argument("--source", choices=["usgs", "isc", "ingv", "geonet"], default="usgs")
    ap.add_argument("--start", default="1990-01-01"); ap.add_argument("--end", default="2020-01-01")
    ap.add_argument("--dl-mag", type=float, default=3.5)
    ap.add_argument("--window-days", type=int, default=365)
    args = ap.parse_args()

    out_dir = Path("reference/Datasets") / args.name; out_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "usgs":
        fetch = fetch_usgs
    else:
        base = FDSN_BASES[args.source]; lim = FDSN_LIMIT[args.source]
        fetch = lambda s, e, a, b, c, d, m: fetch_fdsn(base, lim, s, e, a, b, c, d, m)
    cur = datetime.fromisoformat(args.start); end = datetime.fromisoformat(args.end)
    frames = []
    while cur < end:
        ce = min(cur + timedelta(days=args.window_days), end)
        t0 = time.time()
        try:
            df = fetch(cur, ce, args.minlat, args.maxlat, args.minlon, args.maxlon, args.dl_mag)
        except Exception as ex:
            print("  FAIL", cur.date(), type(ex).__name__, str(ex)[:80]); df = pd.DataFrame()
        if len(df):
            frames.append(df)
        print(f"  {cur.date()}..{ce.date()}  {len(df):5d}  {time.time()-t0:5.1f}s")
        cur = ce

    cat = pd.concat(frames, ignore_index=True)
    cat["magnitude"] = pd.to_numeric(cat["magnitude"], errors="coerce")
    cat["latitude"] = pd.to_numeric(cat["latitude"], errors="coerce")
    cat["longitude"] = pd.to_numeric(cat["longitude"], errors="coerce")
    cat = cat.dropna(subset=["latitude", "longitude", "magnitude", "time"])
    cat = cat[~cat["eventtype"].fillna("").str.contains("|".join(_NONEQ), case=False)]
    cat = cat.drop_duplicates(subset="id")
    cat["time"] = pd.to_datetime(cat["time"], utc=True, errors="coerce").dt.tz_localize(None)
    cat = cat.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    lat0, lon0 = cat["latitude"].mean(), cat["longitude"].mean()
    x, y = azimuthal_equidistant_projection(cat["latitude"].values, cat["longitude"].values, lat0, lon0)
    out = pd.DataFrame({"id": range(len(cat)), "time": cat["time"],
                        "longitude": cat["longitude"], "latitude": cat["latitude"],
                        "magnitude": cat["magnitude"], "x": x, "y": y,
                        "magtype": cat["magtype"].fillna("?").str.upper().values})  # for Mw-homogenization
    out.to_csv(out_dir / f"{args.name}_catalog.csv", index=False)

    corners_ll = np.array([[args.maxlat, args.minlon], [args.maxlat, args.maxlon],
                           [args.minlat, args.maxlon], [args.minlat, args.minlon]], dtype=float)
    np.save(out_dir / f"{args.name}_shape.npy", corners_ll)

    mc, b, n_above = estimate_mc(out["magnitude"].values)
    magtype_mix = cat["magtype"].fillna("?").str.upper().value_counts().head(8).to_dict()
    meta = {"name": args.name, "source": args.source, "n_events": int(len(out)),
            "bbox": [args.minlat, args.maxlat, args.minlon, args.maxlon],
            "center": [float(lat0), float(lon0)], "dl_mag": args.dl_mag,
            "time_range": [str(out.time.min()), str(out.time.max())],
            "Mc_maxc": round(mc, 2), "b_value": round(b, 3), "n_above_Mc": n_above,
            "magtype_mix": {k: int(v) for k, v in magtype_mix.items()},
            "counts_by_mc": {str(m): int((out.magnitude >= m).sum()) for m in [3.5, 4.0, 4.5, 5.0]}}
    (out_dir / f"{args.name}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{args.name} [{args.source}]: {len(out)} events  {out.time.min()} -> {out.time.max()}")
    print(f"  Mc(maxc)={mc:.2f}  b={b:.2f}  N>=Mc={n_above}")
    print(f"  magtypes: {magtype_mix}")
    print(f"  counts: {meta['counts_by_mc']}")
    print(f"  wrote {out_dir}/ (catalog, shape, meta)  center {lat0:.2f},{lon0:.2f}")


if __name__ == "__main__":
    main()
