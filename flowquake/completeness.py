"""Catalog completeness: where and to what magnitude the catalog can be trusted.

This module exists because of a measurement that nearly invalidated the whole
scaling curve. `scripts/build_comcat_lowmc.py` reported Mc = 1.30, flat over
1995-2019, using the maximum-curvature estimator. That number licensed a grid
of {3.0, 2.5, 2.0, 1.5}. It is wrong as a statewide figure:

    mc     1.0    1.5    2.0    2.5    2.6    3.0
    b     0.740  0.837  0.914  0.968  0.977  0.978

b climbs monotonically and only stabilises around mc 2.6. Above true
completeness b must be flat; a downward drift as the threshold falls is the
signature of missing small events. Taken at face value that would put three of
the four planned grid points below Mc, where the curve measures DETECTION
rather than information -- the exact failure MOONSHOT.md invariant 1 forbids.

Taken at face value, and wrongly. Per 1-degree cell the same estimator gives
Mc 1.2-1.8 across the well-instrumented interior. The statewide 2.6 is a
mixture artifact: a superposition of Gutenberg-Richter laws with different Mc
is not itself a Gutenberg-Richter law, and the aggregate FMD rolls over at the
WORST cell's completeness, not the typical one. The offshore and border cells
(Mendocino, the Baja margin, the offshore Central Coast) are badly incomplete
and drag the aggregate down.

So the fix is not to abandon low mc. It is to score only where the catalog
supports it. Restricted to 1-degree cells whose TRAINING-ERA Mc is <= 1.5, the
union's own Mc is 1.3, and the {3.0, 2.5, 2.0, 1.5} grid sits entirely above
completeness. That costs targets -- 293 statewide down to 126 -- and buys back
the validity of every point on the x-axis.

Two properties this module must never lose:

  CAUSAL   The mask is estimated from the training era ONLY. Choosing the
           testing region with knowledge of where the test-period earthquakes
           fell is look-ahead, and it is the kind that inflates skill without
           looking like cheating.
  FIXED    ONE mask, computed once at the lowest mc on the grid, reused for
           every point and for every control model. A mask that moves with mc
           puts an mc-dependent quantity back into the comparison, which is
           the failure the thinned sub-process metric exists to avoid.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .target_process import aki_utsu_b

__all__ = [
    "shi_bolt_sigma", "mc_by_b_stability", "mc_by_max_curvature",
    "CompletenessMask", "mc_map", "build_mask",
]


def shi_bolt_sigma(mags: np.ndarray, b: float) -> float:
    """Shi & Bolt (1982) standard error of a maximum-likelihood b-value.

    Preferred over Aki's b/sqrt(n) because it uses the observed magnitude
    scatter rather than assuming the sample is exactly exponential, which is
    precisely the assumption under test near Mc.
    """
    m = np.asarray(mags, dtype=float)
    n = len(m)
    if n < 3:
        return float("inf")
    var = float(((m - m.mean()) ** 2).sum() / (n * (n - 1)))
    return float(2.30 * b * b * math.sqrt(var))


MC_SAFETY_MARGIN = 0.3
"""How far above an estimate a threshold must sit before it can be trusted.

Both estimators here are OPTIMISTIC on ground truth -- measured bias -0.08
(single-reference) and -0.11 (MBS), with a standard deviation near 0.23-0.27
over simulated catalogs with a known Ogata-Katsura detection curve. Since an
optimistic Mc licenses grid points below completeness, and that is the failure
that voids the whole curve, a threshold should clear the estimate by roughly
bias + 1 sd. Use it as `mc_estimate + MC_SAFETY_MARGIN <= threshold`.
"""


def _b_table(m: np.ndarray, lo: float, hi: float, step: float,
             min_events: int, sigma_floor: float) -> dict[float, tuple[float, float]]:
    table: dict[float, tuple[float, float]] = {}
    for c in np.arange(lo, hi + 1e-9, step):
        c = round(float(c), 4)
        x = m[m >= c - 1e-9]
        if len(x) < min_events:
            continue
        try:
            b = aki_utsu_b(m, c)
        except ValueError:
            continue
        table[c] = (b, max(shi_bolt_sigma(x, b), sigma_floor))
    return table


def mc_by_b_stability(
    mags: np.ndarray, lo: float = 0.0, hi: float = 4.0, step: float = 0.1,
    dm_ref: float = 0.5, min_events: int = 200, sigma_floor: float = 0.02,
    n_consecutive: int = 3, method: str = "mbs",
) -> float | None:
    """Lowest threshold at which b has stopped drifting.

    `method="mbs"` (default, Woessner & Wiemer 2005) compares `b(mc)` against
    the MEAN of b over `[mc, mc + dm_ref]` and requires the criterion to hold
    for `n_consecutive` thresholds in a row. `method="single"` is the older
    Cao & Gao (2002) form, comparing against the single point `b(mc + dm_ref)`
    on first crossing.

    Default to MBS. The single-reference first-crossing rule fires on noise:
    across nested estimation eras for one 1-degree cell it returned 1.3, 2.1,
    1.1, 1.8 -- not a physically possible sequence, since shrinking to a more
    recent era can only improve completeness. MBS gave 2.1, 2.0, 1.9 on the
    same data, and on ground truth its scatter is smaller (sd 0.23 vs 0.27).
    That stability matters because these estimates decide which cells enter
    the testing region, and an unstable mask makes the target set arbitrary.

    `sigma_floor` keeps very large samples from demanding an unattainably
    tight match: at 600k events the formal sigma is 0.0008, which no real
    catalog satisfies, and the answer would be driven to the top of the grid
    by sample size alone.

    Returns None when nothing on the grid stabilises. That is informative and
    must not be coerced to a number -- it means incomplete throughout, and the
    cell should be excluded. Compare the result against a threshold using
    `MC_SAFETY_MARGIN`; the estimator is biased optimistic.
    """
    m = np.asarray(mags, dtype=float)
    m = m[np.isfinite(m)]
    table = _b_table(m, lo, hi, step, min_events, sigma_floor)
    keys = sorted(table)

    if method == "single":
        for c in keys:
            ref = round(c + dm_ref, 4)
            if ref in table and abs(table[c][0] - table[ref][0]) <= table[c][1]:
                return c
        return None
    if method != "mbs":
        raise ValueError(f"unknown method {method!r}; use 'mbs' or 'single'")

    run = 0
    for c in keys:
        window = [table[k][0] for k in keys if c <= k <= c + dm_ref + 1e-9]
        if len(window) < 3:
            break
        run = run + 1 if abs(float(np.mean(window)) - table[c][0]) <= table[c][1] else 0
        if run >= n_consecutive:
            return round(c - (n_consecutive - 1) * step, 4)
    return None


def mc_by_max_curvature(mags: np.ndarray, dm: float = 0.1,
                        correction: float = 0.0) -> float:
    """Modal magnitude of the frequency-magnitude distribution.

    Kept for comparison only. MAXC is fast and it is what the original catalog
    build used, but it is known to be optimistic (Woessner & Wiemer 2005
    recommend adding 0.2 or more), and on a spatially heterogeneous catalog it
    is optimistic by far more than that: it returned 1.30 statewide where
    b-stability returns 2.6. Do not use it to set the grid.
    """
    m = np.asarray(mags, dtype=float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return float("nan")
    edges = np.arange(m.min() - dm / 2, m.max() + dm, dm)
    counts, _ = np.histogram(m, bins=edges)
    return float(edges[int(np.argmax(counts))] + dm / 2 + correction)


class CompletenessMask:
    """A causal, fixed set of cells the catalog is complete in.

    Deliberately dumb: it holds the accepted cell keys and the cell size, and
    can test membership. All the judgement lives in `build_mask`, and the
    resulting object is meant to be serialised and reused verbatim by every
    model on the curve so no arm can quietly score a different region.
    """

    def __init__(self, cells: set[tuple[float, float]], deg: float,
                 mc_threshold: float, mc_union: float | None,
                 mc_by_cell: dict[tuple[float, float], float | None],
                 train_end: str):
        self.cells = set(cells)
        self.deg = float(deg)
        self.mc_threshold = float(mc_threshold)
        self.mc_union = mc_union
        self.mc_by_cell = mc_by_cell
        self.train_end = train_end
        self.margin = MC_SAFETY_MARGIN
        self.mc_worst_cell: float | None = (
            max((mc_by_cell[k] for k in cells if mc_by_cell.get(k) is not None),
                default=None) if cells else None)

    def __len__(self) -> int:
        return len(self.cells)

    def keys_for(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        la = np.floor(np.asarray(lat, dtype=float) / self.deg) * self.deg
        lo = np.floor(np.asarray(lon, dtype=float) / self.deg) * self.deg
        return np.stack([np.round(la, 4), np.round(lo, 4)], axis=1)

    def contains(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        k = self.keys_for(lat, lon)
        return np.array([(float(a), float(b)) in self.cells for a, b in k],
                        dtype=bool)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.contains(df["latitude"].to_numpy(),
                                df["longitude"].to_numpy())].reset_index(drop=True)

    def to_dict(self) -> dict:
        return {
            "deg": self.deg,
            "mc_threshold": self.mc_threshold,
            "mc_union": self.mc_union,
            "mc_worst_cell": self.mc_worst_cell,
            "margin": self.margin,
            "train_end": self.train_end,
            "n_cells": len(self.cells),
            "cells": sorted([list(c) for c in self.cells]),
            "mc_by_cell": {f"{a},{b}": v for (a, b), v in
                           sorted(self.mc_by_cell.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompletenessMask":
        cells = {(float(a), float(b)) for a, b in d["cells"]}
        by_cell = {}
        for k, v in d.get("mc_by_cell", {}).items():
            a, b = k.split(",")
            by_cell[(float(a), float(b))] = v
        return cls(cells, d["deg"], d["mc_threshold"], d.get("mc_union"),
                   by_cell, d["train_end"])


def mc_map(df: pd.DataFrame, deg: float = 1.0, min_events: int = 2000,
           **kw) -> dict[tuple[float, float], float | None]:
    """Per-cell b-stability Mc. `df` must already be restricted to the
    training era -- this function does not enforce causality, `build_mask` does."""
    d = df.copy()
    d["_la"] = (np.floor(d["latitude"] / deg) * deg).round(4)
    d["_lo"] = (np.floor(d["longitude"] / deg) * deg).round(4)
    out: dict[tuple[float, float], float | None] = {}
    for (la, lo), g in d.groupby(["_la", "_lo"]):
        if len(g) < min_events:
            continue
        out[(float(la), float(lo))] = mc_by_b_stability(
            g["magnitude"].to_numpy(), **kw)
    return out


def build_mask(df: pd.DataFrame, train_end: str, mc_threshold: float,
               deg: float = 1.0, min_events: int = 2000,
               time_col: str = "time", margin: float = MC_SAFETY_MARGIN,
               **kw) -> CompletenessMask:
    """Cells whose TRAINING-ERA Mc is at or below `mc_threshold`.

    `mc_threshold` should be the LOWEST mc on the scaling grid: the mask has to
    justify the most demanding point, and the same mask is then used for every
    other point so the region never moves.

    A cell is accepted when `Mc_cell + margin <= mc_threshold`. The margin is
    not decoration: the estimator is biased optimistic (see MC_SAFETY_MARGIN),
    and an optimistic Mc is precisely what licenses a grid point below
    completeness.

    The returned mask also carries `mc_union`, the b-stability Mc of the pooled
    accepted cells. Treat that as a NECESSARY BUT NOT SUFFICIENT check. Pooling
    can make the union look better than its worst member -- abundant events
    from well-instrumented cells swamp a sparse incomplete one -- which is the
    same mixture artifact that made the statewide number unusable, running in
    the opposite direction. The binding constraint is `mc_worst_cell`, since
    the score is evaluated cell by cell and one incomplete cell contributes a
    biased rate no matter how good the average is.
    """
    d = df.copy()
    t = pd.to_datetime(d[time_col])
    train = d[t < pd.Timestamp(train_end)]
    if len(train) == 0:
        raise ValueError(f"no events before train_end={train_end}")

    by_cell = mc_map(train, deg=deg, min_events=min_events, **kw)
    cells = {k for k, v in by_cell.items()
             if v is not None and v + margin <= mc_threshold}
    if not cells:
        best = min((v for v in by_cell.values() if v is not None), default=None)
        raise ValueError(
            f"no cell has training-era Mc + {margin} <= {mc_threshold} "
            f"(best cell Mc = {best}); raise the threshold or use a denser catalog")

    mask = CompletenessMask(cells, deg, mc_threshold, None, by_cell, train_end)
    mask.margin = float(margin)
    mask.mc_worst_cell = max(by_cell[k] for k in cells)
    pooled = mask.apply(train)
    mask.mc_union = mc_by_b_stability(pooled["magnitude"].to_numpy(), **kw)
    return mask
