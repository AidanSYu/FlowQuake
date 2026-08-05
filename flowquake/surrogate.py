"""Surrogate nulls: destroy the coupling, keep everything else.

The ground-truth validation needs an arm whose true slope is zero. Three
attempts to get one from a simulator failed for reasons documented in
MOONSHOT.md invariants 1f and 1g -- the generated catalogs were near-Poisson at
the scored threshold (variance/mean 1.21 against 2.0-5.4 for the real regional
catalogs), so neither a trained neural model nor a fitted ETAS could forecast
them, and a flat null proved nothing.

Real catalogs do not have that problem: their clustering is real by
definition. What they lack is a known answer. Surrogates supply one. Keep a
real catalog's target events exactly, and replace its small events with a
version of THEMSELVES that has been decoupled from those targets. The small
events keep their count, magnitude distribution, spatial footprint and their
own clustering; what they lose is any relationship to when and where the
targets happened. A slope measured on that arm is a slope the estimator
invented.

Which coupling to destroy is a real choice, not a detail:

  `time_shift`  breaks the TEMPORAL relationship and preserves the spatial one.
                Small events still sit on the same faults as the targets, so
                the arm retains genuine information about WHERE targets occur
                and only loses WHEN. This is the conservative null: an
                estimator that shows no slope against it is not merely
                ignoring noise, it is failing to use timing.

  `rotate`      breaks the SPATIAL relationship and preserves the temporal one.
                The mirror test.

  `both`        breaks both, and is closest to "no information at all". Use it
                as the primary null; the other two localise WHICH kind of
                information a slope is coming from, which is a more interesting
                result than a single pass/fail.

The time shift is CIRCULAR within the catalog span, so the small-event rate
stays stationary and no window is left empty. A plain offset would push events
off the end and quietly thin the late windows -- which would itself produce an
mc-dependent artifact, since the thinning is proportional to event count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["surrogate_null", "circular_time_shift", "rotate_about_centroid"]


def circular_time_shift(t: np.ndarray, shift_days: float,
                        t0: float, t1: float) -> np.ndarray:
    """Shift times by `shift_days`, wrapping within [t0, t1)."""
    span = float(t1 - t0)
    if span <= 0:
        raise ValueError("empty time span")
    return t0 + np.mod(np.asarray(t, dtype=float) - t0 + shift_days, span)


def rotate_about_centroid(x: np.ndarray, y: np.ndarray,
                          angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Rigid rotation about the centroid: preserves all pairwise distances."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = x.mean(), y.mean()
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    dx, dy = x - cx, y - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def surrogate_null(
    df: pd.DataFrame,
    m_split: float,
    mode: str = "both",
    seed: int = 0,
    time_col: str = "time",
    min_shift_frac: float = 0.2,
    era_bounds: list | None = None,
) -> pd.DataFrame:
    """Targets untouched; sub-`m_split` events decoupled from them.

    `mode` is "time_shift", "rotate" or "both". `min_shift_frac` keeps the
    circular shift away from zero and from a full wrap -- a shift of a few days
    would leave most of the triggering relationship intact and the arm would
    not be a null at all.

    `era_bounds` are timestamps (typically `val_start` and `test_start`) inside
    which the shift is applied independently, so each era keeps exactly the
    event count it started with. Pass them whenever the arms will be compared
    after a train/test split; without them the shift silently moves events
    across the boundary and hands one arm more training data than the other.

    Requires `x`/`y` columns for the rotation modes; every catalog in
    `reference/Datasets/` carries them.
    """
    if mode not in ("time_shift", "rotate", "both"):
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed)

    d = df.copy()
    t = pd.to_datetime(d[time_col])
    keep = d[d["magnitude"] >= m_split].copy()
    small = d[d["magnitude"] < m_split].copy()
    if len(small) == 0:
        raise ValueError(f"no events below m_split={m_split}")

    if mode in ("time_shift", "both"):
        ts = t.astype("int64").to_numpy()
        t0, t1 = float(ts.min()), float(ts.max())
        st = pd.to_datetime(small[time_col]).astype("int64").to_numpy()

        # Shift WITHIN each era separately when era boundaries are given.
        #
        # A single shift across the whole catalog preserves the total count but
        # not its split between training and test: real small events cluster in
        # time around sequences, and spreading them uniformly moves some across
        # the boundary. Measured on WHITE, the surrogate arm ended up with 1,604
        # training events at mc 2.0 against the informative arm's 1,383, and
        # 4,871 against 3,864 at mc 1.5 -- a 26% advantage in training data for
        # the arm that is supposed to be the control. Shifting inside each era
        # keeps the per-era counts EXACTLY equal, so the only difference between
        # arms remains the coupling.
        bounds = [t0] + [float(pd.Timestamp(b).value) for b in (era_bounds or [])
                         if t0 < float(pd.Timestamp(b).value) < t1] + [t1]
        shifted = st.copy()
        for lo_e, hi_e in zip(bounds, bounds[1:]):
            sel = (st >= lo_e) & (st < hi_e) if hi_e < t1 else (st >= lo_e)
            if not sel.any():
                continue
            span_e = hi_e - lo_e
            if span_e <= 0:
                continue
            sh = float(rng.uniform(min_shift_frac * span_e,
                                   (1.0 - min_shift_frac) * span_e))
            shifted[sel] = circular_time_shift(st[sel], sh, lo_e, hi_e)
        shifted = shifted.astype("int64")
        # rebuild through the ORIGINAL dtype: going via a bare int64 -> datetime
        # loses any timezone, and the result then cannot even be sorted against
        # the untouched target rows
        small[time_col] = pd.Series(shifted, index=small.index).astype(
            df[time_col].dtype if str(df[time_col].dtype).startswith("datetime")
            else "datetime64[ns]")

    if mode in ("rotate", "both"):
        if not {"x", "y"} <= set(small.columns):
            raise ValueError("rotation needs projected x/y columns")
        # avoid angles near 0 or 2*pi, which would barely move anything
        ang = float(rng.uniform(0.25 * np.pi, 1.75 * np.pi))
        nx, ny = rotate_about_centroid(small["x"].to_numpy(),
                                       small["y"].to_numpy(), ang)
        small["x"], small["y"] = nx, ny

    out = pd.concat([keep, small], ignore_index=True)
    return out.sort_values(time_col).reset_index(drop=True)
