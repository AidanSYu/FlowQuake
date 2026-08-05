"""M>=M_TARGET sub-process scoring — the mc-invariant metric of `MOONSHOT.md`.

Why this module exists
----------------------
The scaling curve varies catalog completeness `mc` along the x-axis. Per-event
`tll`/`sll` (the metric everywhere else in this repo) is the density of the next
event *above mc*, so its units move with mc: at mc 0.5 events are minutes apart
and at mc 2.5 hours apart. Plotting nats/event against mc would manufacture an
enormous slope that is pure bookkeeping. `MOONSHOT.md` invariant 1b.

The comparable quantity is the intensity of the **thinned target process** —
the rate of M>=M_TARGET events per day per km^2:

    lambda_tgt(t,x,y) = lambda_total(t,x,y | H^mc) * P(m >= M_TARGET | H^mc)

An mc-0.5 model and an mc-2.5 model then forecast the *same physical quantity*
and their likelihoods live on one scale. This is also exactly what CSEP scores.

Estimation is simulation-based: a forecast is a set of simulated catalogs over
[t0, t0+H), which is the CSEP catalog-based forecast definition. That reuses the
validated forward simulator instead of hand-rolling a compensator, and it makes
this scorer **model-agnostic** — FlowQuake and ETAS are scored by identical code,
differing only in who produced the catalogs.

Metrics returned, per forecast window and aggregated:
  * `ll`      Poisson log-likelihood of the observed target events on the grid
              (the CSEP catalog-based L-test form)
  * `ig`      information gain per target event against a reference forecast
              (ETAS, or the time-independent Poisson null) — the currency a
              seismologist reads
  * `p_large` P(at least one M>=M_LARGE in the horizon), with Brier score, log
              score and reliability bins — the decision-relevant metric that
              nats alone cannot carry (`MOONSHOT.md` gate G5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Fixed spatial grid in the catalog's km frame.

    MUST be built once, at a reference completeness, and reused for every point
    on the curve. Deriving the bounding box per-mc makes the grid itself
    mc-dependent (lower mc -> more events -> slightly larger box), which leaks a
    second mc dependence into the metric. Use `Grid.from_bounds`.
    """

    xmin: float
    ymin: float
    nx: int
    ny: int
    bin_km: float

    @staticmethod
    def from_bounds(x, y, bin_km: float = 5.0, pad_km: float = 10.0) -> "Grid":
        xmin, ymin = float(np.min(x)) - pad_km, float(np.min(y)) - pad_km
        xmax, ymax = float(np.max(x)) + pad_km, float(np.max(y)) + pad_km
        return Grid(
            xmin=xmin, ymin=ymin,
            nx=int(math.ceil((xmax - xmin) / bin_km)),
            ny=int(math.ceil((ymax - ymin) / bin_km)),
            bin_km=float(bin_km),
        )

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def cell_area_km2(self) -> float:
        return self.bin_km ** 2

    def index(self, x, y) -> np.ndarray:
        """Flat cell index for each (x, y); points outside are clipped in."""
        gx = np.clip(((np.asarray(x) - self.xmin) / self.bin_km).astype(int), 0, self.nx - 1)
        gy = np.clip(((np.asarray(y) - self.ymin) / self.bin_km).astype(int), 0, self.ny - 1)
        return gx * self.ny + gy

    def counts(self, x, y) -> np.ndarray:
        out = np.zeros(self.n_cells, dtype=np.float64)
        if len(np.asarray(x)) == 0:
            return out
        np.add.at(out, self.index(x, y), 1.0)
        return out

    def active_mask(self, x, y) -> np.ndarray:
        """Cells containing at least one of the given events — the testing region.

        CSEP defines its testing region a priori (the RELM polygon); this is the
        data-driven equivalent. Build it from the **training era at all
        magnitudes**, which makes it causal (no test-window information) and
        mc-independent (it is a property of the network footprint, not of the
        threshold we choose). Scoring over the whole bounding box instead makes
        the Poisson likelihood a measurement of the water-level floor: a 25 km
        grid over a wandering catalog is ~10^6 cells against ~10^2 target
        events, and every observed event lands in a cell no simulation reached.
        """
        return self.counts(x, y) > 0

    def inside(self, x, y) -> np.ndarray:
        x, y = np.asarray(x), np.asarray(y)
        return (
            (x >= self.xmin) & (x < self.xmin + self.nx * self.bin_km)
            & (y >= self.ymin) & (y < self.ymin + self.ny * self.bin_km)
        )


@dataclass(frozen=True)
class TargetSpec:
    """What we score. Identical across every point of the scaling curve."""

    m_target: float = 4.0        # the sub-process being forecast
    m_large: float = 6.0         # the decision-relevant threshold
    horizon_days: float = 30.0
    # Fraction of total expected rate spread uniformly over the grid before
    # taking logs. Prevents log(0) in cells no simulation reached, and is the
    # standard CSEP water-level. Reported so the sensitivity is auditable.
    floor_frac: float = 1e-3

    # --- magnitude-tail handling (MOONSHOT.md invariant 1c) -----------------
    # "learned": thin the simulated catalog by the model's own sampled
    #     magnitudes. Sensitive to GR-head calibration, and that sensitivity is
    #     mc-DEPENDENT (at lower mc the head extrapolates further to reach
    #     m_target), so it leaks an artefactual slope into the scaling curve.
    #     This is the ABLATION.
    # "fixed": thin by a Gutenberg-Richter probability computed once from the
    #     training era and held constant across the whole curve:
    #         P(m >= m_target | m >= mc) = 10 ** (-b * (m_target - mc))
    #     The learned magnitude head is then entirely out of the metric and what
    #     remains is rate + location skill. This is the PRIMARY result.
    tail_mode: str = "fixed"
    b_value: float = 1.0
    mc: float = 2.5              # the completeness of the catalog being scored

    def tail_prob(self, m: float) -> float:
        """P(magnitude >= m | event is in the catalog), under the fixed GR tail."""
        return float(10.0 ** (-self.b_value * (m - self.mc)))


def rate_field(sims: list[np.ndarray], grid: Grid, spec: "TargetSpec") -> np.ndarray:
    """Expected M>=m_target counts per cell over the horizon, from simulations.

    `sims` is a list of (n_i, 4) arrays with columns [t_days, x_km, y_km, mag] —
    the return type of `flowquake.ntest.simulate_day_events`, and the format the
    ETAS producer is adapted to in `scripts/scaling_curve.py`.

    Under `tail_mode="fixed"` every simulated event contributes `tail_prob`
    instead of being accepted/rejected on its sampled magnitude: the analytic
    Poisson thinning of the simulated process by a constant GR tail. This makes
    the score independent of the learned magnitude head, which is the point.
    """
    if len(sims) == 0:
        raise ValueError("no simulated catalogs")
    total = np.zeros(grid.n_cells, dtype=np.float64)
    w = spec.tail_prob(spec.m_target) if spec.tail_mode == "fixed" else 1.0
    for ev in sims:
        if len(ev) == 0:
            continue
        sel = (np.ones(len(ev), dtype=bool) if spec.tail_mode == "fixed"
               else ev[:, 3] >= spec.m_target)
        if sel.any():
            total += w * grid.counts(ev[sel, 1], ev[sel, 2])
    return total / float(len(sims))


def _watered(lam: np.ndarray, floor_frac: float) -> np.ndarray:
    """Uniform water-level so empty cells are finite. Preserves total rate."""
    tot = float(lam.sum())
    if tot <= 0.0:
        return np.full_like(lam, 1e-12)
    return (1.0 - floor_frac) * lam + floor_frac * tot / len(lam)


def poisson_ll(lam: np.ndarray, obs_counts: np.ndarray) -> float:
    """CSEP catalog-based L-test form: sum_c [n_c log lam_c - lam_c - log n_c!]."""
    from scipy.special import gammaln

    return float(np.sum(obs_counts * np.log(lam) - lam - gammaln(obs_counts + 1.0)))


def poisson_ll_parts(lam: np.ndarray, obs_counts: np.ndarray) -> tuple[float, float, float]:
    r"""Split the L-test score into its CSEP N-test and S-test components.

    Writing L = sum_c(lam_c) and p_c = lam_c / L,

        ll     = -L + N log L  +  sum_c n_c log p_c  -  sum_c log n_c!
                 \____________/    \________________/
                    level              shape

    `level` is the total-count (N-test) term and `shape` the normalised
    spatial-density (S-test) term. The split matters here because `shape` is
    INVARIANT to any rescaling lam -> c*lam, so it cannot be moved by a
    mis-specified magnitude tail, while `level` absorbs that error in full.
    Since the fixed tail extrapolates over `m_target - mc` decades, any b error
    hits `level` with mc-dependent weight (measured: -0.19 nats/decade on the
    informative arm, -0.14 on the null, i.e. common-mode). Reporting the two
    separately is what makes the curve robust to the tail rather than hostage
    to it. `ll = level + shape - log-factorial` exactly.
    """
    from scipy.special import gammaln

    lam = np.asarray(lam, dtype=np.float64)
    n = np.asarray(obs_counts, dtype=np.float64)
    L = float(lam.sum())
    N = float(n.sum())
    level = -L + N * math.log(L) if L > 0 else float("-inf")

    # Sum the shape term over OCCUPIED cells only. Cells with n_c = 0
    # contribute zero by identity, and evaluating them anyway makes the result
    # depend on whether lam_c happens to be zero there: numpy gives
    # 0 * log(0) = 0 * -inf = nan, which would silently poison the whole score.
    # Production waters lam before this call so it does not arise today, but a
    # floor_frac of 0 -- or any future caller passing a raw field -- would hit
    # it, and a nan here is indistinguishable from a missing point downstream.
    occ = n > 0
    shape = (float(np.sum(n[occ] * (np.log(lam[occ]) - math.log(L))))
             if L > 0 else float("-inf"))
    ll = level + shape - float(np.sum(gammaln(n + 1.0)))
    return ll, level, shape


def p_at_least_one(sims: list[np.ndarray], spec: "TargetSpec",
                   grid: Grid | None = None) -> float:
    """P(at least one m >= m_large in the horizon), averaged over simulations.

    `tail_mode="fixed"` uses exact Poisson thinning rather than the sampled
    magnitudes: a simulation with n in-region events has
    P(none exceed m_large) = (1 - p)**n, with p the constant GR tail. Averaging
    that over simulations keeps the decision metric independent of the learned
    magnitude head, exactly as the likelihood is.
    """
    if len(sims) == 0:
        return 0.0
    if spec.tail_mode == "fixed":
        p = spec.tail_prob(spec.m_large)
        acc = 0.0
        for ev in sims:
            n = 0 if len(ev) == 0 else int(
                grid.inside(ev[:, 1], ev[:, 2]).sum() if grid is not None else len(ev))
            acc += 1.0 - (1.0 - p) ** n
        return acc / float(len(sims))
    hit = 0
    for ev in sims:
        if len(ev) == 0:
            continue
        sel = ev[:, 3] >= spec.m_large
        if grid is not None and sel.any():
            sel = sel & grid.inside(ev[:, 1], ev[:, 2])
        if np.any(sel):
            hit += 1
    return hit / float(len(sims))


def score_window(
    sims: list[np.ndarray] | None,
    obs: np.ndarray,
    grid: Grid,
    spec: TargetSpec,
    active: np.ndarray | None = None,
    lam_field: np.ndarray | None = None,
    p_large: float | None = None,
) -> dict:
    """Score one forecast window.

    `obs` is an (n, 4) array [t_days, x_km, y_km, mag] of the events actually
    observed in the window, **above the target threshold filter applied here** —
    pass the full observed catalog for the window and this selects.

    `active` is the testing-region cell mask from `Grid.active_mask`. Strongly
    recommended: without it the score is dominated by empty cells. It must be
    the SAME mask for every point on the curve and for the ETAS control.

    `lam_field` supplies an ANALYTIC rate field instead of estimating one from
    simulations, and `p_large` the matching decision probability. Use it when a
    closed form exists — ETAS has one, see `flowquake.etas_fit.etas_rate_field`.
    A simulated field carries a Monte-Carlo resolution bias of -4.98 nats per
    target event at 93 total simulated events, falling to -0.03 at 20,000; since
    that total scales with the catalog rate above mc, the bias masquerades as
    an mc-dependent slope. An analytic field has none of it, and costs less than
    the ~165M simulations per curve point needed to drive it down by brute
    force. When `lam_field` is given, `sims` may be None.
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, 4)
    keep = obs[:, 3] >= spec.m_target
    if len(obs):
        keep &= grid.inside(obs[:, 1], obs[:, 2])
        if active is not None:
            keep &= active[grid.index(obs[:, 1], obs[:, 2])]
    tgt = obs[keep]

    if lam_field is not None:
        lam_full = np.asarray(lam_field, dtype=np.float64)
        if len(lam_full) != grid.n_cells:
            raise ValueError(
                f"lam_field has {len(lam_full)} cells, grid has {grid.n_cells}")
    else:
        if sims is None:
            raise ValueError("pass either `sims` or `lam_field`")
        lam_full = rate_field(sims, grid, spec)
    obs_full = grid.counts(tgt[:, 1], tgt[:, 2])
    if active is not None:
        lam_full, obs_full = lam_full[active], obs_full[active]
    lam = _watered(lam_full, spec.floor_frac)
    obs_counts = obs_full

    if p_large is not None:
        p_lg = float(p_large)
    elif sims is not None:
        p_lg = p_at_least_one(sims, spec, grid)
    else:
        # analytic thinning of the expected count: P(at least one m >= m_large)
        # under a Poisson field with the GR tail applied
        lam_lg = float(lam_full.sum()) * (
            spec.tail_prob(spec.m_large) / max(spec.tail_prob(spec.m_target), 1e-300))
        p_lg = float(1.0 - math.exp(-lam_lg))
    y_lg = bool(np.any(obs[:, 3] >= spec.m_large)) if len(obs) else False
    eps = 1e-6
    p_clip = min(max(p_lg, eps), 1.0 - eps)

    ll, ll_level, ll_shape = poisson_ll_parts(lam, obs_counts)

    # Water-level sensitivity, carried on every window so it can never be
    # invisible. The floor exists to keep log(0) out of the score, but it is
    # not neutral: it adds a term that depends on how CONCENTRATED the forecast
    # is, and forecast concentration changes with mc. Measured on synthetic
    # fields with matched simulation budgets, the gap between a diffuse and a
    # sharp field moved from 0.61 to 1.05 nats as floor_frac went from 1e-4 to
    # 1e-2 -- the same order as the signal being measured. A slope is only
    # trustworthy if it survives this sweep, so report it rather than pick a
    # floor and hope.
    shape_by_floor = {
        f"{f:g}": poisson_ll_parts(_watered(lam_full, f), obs_counts)[2]
        for f in (1e-4, 1e-3, 1e-2)
    }

    # Effective number of occupied cells (inverse participation ratio).
    #
    # Matching the total simulated events T across mc is necessary but not by
    # itself sufficient: the per-cell relative variance is 1/(T*p_c), so what
    # actually sets resolution is T per EFFECTIVE cell, and a sharper forecast
    # concentrates the same T into fewer of them. Measured at matched T, the
    # residual concentration-dependence of the bias is 0.44 nats at T = 2,000
    # but only 0.023 at T = 20,000 -- so the budget in scaling_curve is large
    # enough, and this number is what proves it rather than assumes it.
    _p = lam_full / lam_full.sum() if lam_full.sum() > 0 else lam_full
    n_eff = float(1.0 / np.sum(_p ** 2)) if _p.sum() > 0 else float("nan")
    return {
        "n_target_obs": int(len(tgt)),
        "n_expected": float(lam.sum()),
        "ll": ll,
        "ll_level": ll_level,
        "ll_shape": ll_shape,
        "ll_shape_by_floor": shape_by_floor,
        "n_eff_cells": n_eff,
        "p_large": float(p_lg),
        "obs_large": y_lg,
        "brier": float((p_lg - float(y_lg)) ** 2),
        "log_score": float(math.log(p_clip) if y_lg else math.log(1.0 - p_clip)),
        "n_sims": int(len(sims)) if sims is not None else 0,
        "analytic": lam_field is not None,
        "tail_mode": spec.tail_mode,
    }


def magnitude_quantum(mags: np.ndarray, tol: float = 1e-6,
                      frac: float = 0.99) -> float:
    """Largest step the magnitudes are reported on, or 0.0 if continuous.

    The Aki-Utsu binning correction is `mc - dm/2`, and it is only correct when
    the magnitudes really are rounded to `dm`. Applying a 0.1 correction to
    continuous magnitudes biases b LOW by construction: it inflates the mean
    excess by dm/2, and since the fixed tail extrapolates over
    `m_target - mc` decades, that bias is amplified by
    `10**(db * (m_target - mc))` — an mc-DEPENDENT level error, which is the
    exact failure invariant 1c exists to prevent. Measured on the synthetic
    catalog: true b = 0.993-1.002, this function's caller returned 0.869, and
    the resulting over-forecast ran 1.16x at mc 2.5 to 1.83x at mc 1.0.
    """
    m = np.asarray(mags, dtype=float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return 0.0
    for step in (0.1, 0.05, 0.02, 0.01):
        r = np.abs(m / step - np.round(m / step))
        if float(np.mean(r < tol)) >= frac:
            return step
    return 0.0


def aki_utsu_b(mags: np.ndarray, mc: float, dm: float | None = None) -> float:
    """Aki-Utsu maximum-likelihood b-value for events above `mc`.

    Used to pin the fixed magnitude tail (invariant 1c). Estimate ONCE per
    region from the training era and hold it constant across the whole scaling
    curve — re-estimating per mc would put a second mc-dependent quantity back
    into the metric, which is the exact failure the fixed tail exists to avoid.

    `dm` defaults to the quantum measured from the data rather than an assumed
    0.1; see `magnitude_quantum` for why guessing it is not safe. Pass an
    explicit value only to override a known-wrong inference.
    """
    m = np.asarray(mags, dtype=float)
    m = m[m >= mc]
    if len(m) < 50:
        raise ValueError(f"only {len(m)} events >= mc={mc}; b-value unreliable")
    if dm is None:
        dm = magnitude_quantum(m)
    return float(1.0 / (math.log(10.0) * (m.mean() - (mc - dm / 2.0))))


def reliability(windows: list[dict], n_bins: int = 10) -> list[dict]:
    """Reliability diagram data for P(M>=m_large): forecast vs observed rate."""
    p = np.array([w["p_large"] for w in windows], dtype=float)
    y = np.array([float(w["obs_large"]) for w in windows], dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if not sel.any():
            continue
        out.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "n": int(sel.sum()),
            "mean_forecast": float(p[sel].mean()),
            "observed_freq": float(y[sel].mean()),
        })
    return out


def aggregate(windows: list[dict], reference: list[dict] | None = None) -> dict:
    """Aggregate window scores; if `reference` is given, information gain vs it.

    Information gain is per TARGET EVENT, the CSEP convention, and is the number
    that goes on the y-axis of the killer figure. Both models must have been
    scored on the same windows in the same order.
    """
    ll = float(np.sum([w["ll"] for w in windows]))
    n_tgt = int(np.sum([w["n_target_obs"] for w in windows]))
    # level/shape are absent from windows scored before the split existed
    has_parts = all("ll_shape" in w for w in windows)
    lvl = float(np.sum([w["ll_level"] for w in windows])) if has_parts else None
    shp = float(np.sum([w["ll_shape"] for w in windows])) if has_parts else None
    out = {
        "n_windows": len(windows),
        "n_target_events": n_tgt,
        "ll_total": ll,
        "ll_per_target_event": ll / n_tgt if n_tgt else None,
        "ll_level_total": lvl,
        "ll_shape_total": shp,
        # PRIMARY for the scaling curve: immune to magnitude-tail error by
        # construction, because shape is invariant to lam -> c*lam.
        "ll_shape_per_target_event": (shp / n_tgt) if (has_parts and n_tgt) else None,
        "ll_level_per_target_event": (lvl / n_tgt) if (has_parts and n_tgt) else None,
        "n_expected_total": float(np.sum([w["n_expected"] for w in windows])),
        "brier": float(np.mean([w["brier"] for w in windows])),
        "log_score_mean": float(np.mean([w["log_score"] for w in windows])),
        "base_rate_large": float(np.mean([float(w["obs_large"]) for w in windows])),
        "reliability": reliability(windows),
    }
    if reference is not None:
        if len(reference) != len(windows):
            raise ValueError("reference must cover the same windows")
        ref_ll = float(np.sum([w["ll"] for w in reference]))
        out["reference_ll_total"] = ref_ll
        out["ig_total"] = ll - ref_ll
        out["ig_per_target_event"] = (ll - ref_ll) / n_tgt if n_tgt else None
        out["reference_brier"] = float(np.mean([w["brier"] for w in reference]))
        out["brier_skill_score"] = (
            1.0 - out["brier"] / out["reference_brier"]
            if out["reference_brier"] > 0 else None
        )
    return out


def poisson_null_sims(
    rate_per_day: float, grid_density: np.ndarray, grid: Grid,
    spec: TargetSpec, n_sims: int, seed: int = 0,
) -> list[np.ndarray]:
    """Time-independent Poisson forecast, as the floor reference model.

    `grid_density` is a normalized (sums to 1) per-cell probability from the
    training-era target-magnitude seismicity. `rate_per_day` is the training-era
    M>=m_target rate. This is the "no skill beyond long-term seismicity" baseline
    every point on the curve should also be reported against, so a reader can see
    absolute skill and not only the FlowQuake-minus-ETAS difference.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(grid_density, dtype=float)
    p = p / p.sum()
    mu = rate_per_day * spec.horizon_days
    out = []
    xs = grid.xmin + (np.arange(grid.n_cells) // grid.ny + 0.5) * grid.bin_km
    ys = grid.ymin + (np.arange(grid.n_cells) % grid.ny + 0.5) * grid.bin_km
    for _ in range(n_sims):
        n = rng.poisson(mu)
        if n == 0:
            out.append(np.zeros((0, 4)))
            continue
        cells = rng.choice(grid.n_cells, size=n, p=p)
        jit = (rng.random((n, 2)) - 0.5) * grid.bin_km
        ev = np.column_stack([
            rng.random(n) * spec.horizon_days,
            xs[cells] + jit[:, 0],
            ys[cells] + jit[:, 1],
            np.full(n, spec.m_target),
        ])
        out.append(ev)
    return out
