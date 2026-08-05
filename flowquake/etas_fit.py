"""Self-contained ETAS: EM inversion, simulation, and a free smoothed background.

Three jobs, all of which the repository currently cannot do without external
state:

  1. **The scaling-curve control** (`MOONSHOT.md` invariant 3). ETAS must be
     re-inverted at *every* mc on the curve. Nothing else here can do that: the
     benchmark ships one inversion per catalog at one completeness.

  2. **Gate G1**, the control that could collapse §4.4. The benchmark's ETAS has
     a UNIFORM spatial background, and +0.051 of FlowQuake's +0.060 spatial gain
     comes from replacing it with smoothed seismicity — a feature operational
     ETAS has had since Zhuang, Ogata & Vere-Jones (2002). `background="smoothed"`
     gives ETAS that feature, so we can report what is left for the neural part.

  3. **Provenance.** `pyproject.toml:25-38` records two candidate `etas` forks,
     states they are different code, and marks the choice as blocking release;
     no committed artifact resolves which produced the published baselines
     (`WORKING.md` item 8). Every number measured against *this* module is
     reproducible from this repository alone.

Model (Ogata 1998), conditional intensity:

    lambda(t,x,y) = mu * u(x,y)
                  + sum_{t_j < t} K * 10^(a (m_j - mc))
                                    * (t - t_j + c)^(-p)
                                    * f(r_ij; d_j, q)

    f(r; d, q) = (q-1) / (pi d^2) * (1 + r^2/d^2)^(-q),
    d_j^2      = d^2 * 10^(gamma (m_j - mc))

Both the temporal and spatial kernels are exactly normalised, so the
log-likelihood below is a proper point-process likelihood and directly
comparable with FlowQuake's.

Inversion is Zhuang-style stochastic-declustering EM: the E-step computes each
event's probability of being background vs triggered by each prior event; the
M-step re-estimates parameters (and, for `background="smoothed"`, re-estimates
the background field as a variable-bandwidth KDE weighted by background
probability, which is exactly Zhuang's construction).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

# Parent interactions are truncated for tractability. Both cutoffs are far
# outside where the kernels carry mass (Omori is ~t^-1.1; the spatial kernel is
# ~r^-3), and the truncation is applied IDENTICALLY in fitting, likelihood and
# simulation, so it never biases a comparison between models.
MAX_PARENT_DAYS = 3652.5      # 10 years
MAX_PARENT_KM = 400.0


@dataclass(frozen=True)
class ETASParams:
    mu: float = 1e-3       # background events / day / km^2
    K: float = 0.02        # productivity
    a: float = 0.8         # magnitude-productivity scaling
    c: float = 0.01        # Omori offset, days
    p: float = 1.1         # Omori exponent
    d: float = 1.0         # spatial scale, km
    gamma: float = 0.5     # spatial magnitude scaling
    q: float = 1.8         # spatial decay exponent
    mc: float = 2.5

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

def omori(dt: np.ndarray, P: ETASParams) -> np.ndarray:
    """Normalised Omori-Utsu density in elapsed time (integrates to 1 over
    [0, inf) for p > 1)."""
    return (P.p - 1.0) / P.c * (1.0 + dt / P.c) ** (-P.p)


def omori_int(dt: np.ndarray, P: ETASParams) -> np.ndarray:
    """Integral of `omori` from 0 to dt."""
    return 1.0 - (1.0 + dt / P.c) ** (1.0 - P.p)


def spatial(r2: np.ndarray, mpar: np.ndarray, P: ETASParams) -> np.ndarray:
    """Normalised power-law spatial density (integrates to 1 over the plane)."""
    d2 = P.d ** 2 * 10.0 ** (P.gamma * (mpar - P.mc))
    return (P.q - 1.0) / (math.pi * d2) * (1.0 + r2 / d2) ** (-P.q)


def productivity(mpar: np.ndarray, P: ETASParams) -> np.ndarray:
    """Expected direct offspring of a parent of magnitude `mpar`."""
    return P.K * 10.0 ** (P.a * (mpar - P.mc))


def branching_ratio(P: ETASParams, b: float = 1.0) -> float:
    """Expected offspring per event, averaged over the GR magnitude
    distribution. >= 1 means the process is supercritical, which is a red flag
    on any inversion (the manuscript reports 0.968 at MANUSCRIPT.md:364 and
    never discusses it)."""
    if b <= P.a:
        return float("inf")
    return float(P.K * b / (b - P.a))


# --------------------------------------------------------------------------
# neighbour structure
# --------------------------------------------------------------------------

def radius_for_tol(m_par: float, P: ETASParams, tol: float) -> float:
    """Radius containing (1 - tol) of a parent's offspring.

    From the power-law kernel's exact tail, P(r > R) = (1 + R^2/d_j^2)^(1-q):
        R = d_j * sqrt(tol^(1/(1-q)) - 1)
    The magnitude dependence is the whole point — an M(mc+3) parent needs ~30x
    the radius of an M(mc) one, so a single global cutoff is either wrong for
    large events or ruinously expensive for small ones.
    """
    d2 = P.d ** 2 * 10.0 ** (P.gamma * (m_par - P.mc))
    return float(math.sqrt(d2 * (tol ** (1.0 / (1.0 - P.q)) - 1.0)))


def parent_pairs(t, x, y, m, P: ETASParams, max_days: float = MAX_PARENT_DAYS,
                 tol: float = 0.01, max_parents: int = 4096, block: int = 512):
    """(child, parent) index arrays for pairs that carry non-negligible intensity.

    Three bounded truncations, each with a stated cost, all applied IDENTICALLY
    in fitting, likelihood and simulation so no comparison is biased:

      * time     `max_days`. Omori with p=1.2 leaves ~9% of offspring beyond ten
                 years, but spread over that span their instantaneous rate is
                 negligible.
      * space    per-parent radius from `radius_for_tol`, so each parent keeps
                 (1 - tol) of its own offspring regardless of magnitude.
      * count    at most `max_parents` most-recent parents per child, a backstop
                 against dense swarms.

    The naive version of this function used a 10-year / 400 km box and a Python
    loop per event, which is O(E^2) in pairs — ~10^9 for a 50k catalog, and it
    hung. This one is blocked and fully vectorised: children are processed in
    time-contiguous blocks whose admissible parents form one contiguous slice,
    so the inner work is a single (block x slice) masked distance computation.
    """
    n = len(t)
    r_max = np.array([radius_for_tol(mi, P, tol) for mi in m])
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        j0 = int(np.searchsorted(t, t[i0] - max_days, side="left"))
        j0 = max(j0, i1 - 1 - max_parents) if i1 - 1 - max_parents > j0 else j0
        if i1 <= j0 + 1:
            continue
        js = np.arange(j0, i1 - 1)
        if len(js) == 0:
            continue
        ci = np.arange(i0, i1)
        dt = t[ci][:, None] - t[js][None, :]
        dx = x[ci][:, None] - x[js][None, :]
        dy = y[ci][:, None] - y[js][None, :]
        ok = (dt > 0) & (dt <= max_days) & \
             ((dx * dx + dy * dy) <= (r_max[js] ** 2)[None, :])
        if not ok.any():
            continue
        a, b = np.nonzero(ok)
        yield ci[a], js[b]


# retained name for older call sites
def _parent_pairs(t, x, y, max_days=MAX_PARENT_DAYS, max_km=MAX_PARENT_KM):
    raise RuntimeError(
        "_parent_pairs was O(E^2) and is removed; use parent_pairs(t,x,y,m,P)")


# --------------------------------------------------------------------------
# background field
# --------------------------------------------------------------------------

@dataclass
class Background:
    """Spatial background density, per km^2, normalised over the region."""

    mode: str                  # "uniform" | "smoothed"
    area_km2: float
    grid: np.ndarray | None = None      # (nx, ny) density per km^2
    xmin: float = 0.0
    ymin: float = 0.0
    bin_km: float = 5.0

    def at(self, x, y) -> np.ndarray:
        if self.mode == "uniform" or self.grid is None:
            return np.full(np.shape(x), 1.0 / self.area_km2)
        nx, ny = self.grid.shape
        gx = np.clip(((np.asarray(x) - self.xmin) / self.bin_km).astype(int), 0, nx - 1)
        gy = np.clip(((np.asarray(y) - self.ymin) / self.bin_km).astype(int), 0, ny - 1)
        return self.grid[gx, gy]


def smoothed_background(x, y, w, xmin, ymin, nx, ny, bin_km,
                        k_neighbour=6, sig_min=1.0, sig_max=50.0,
                        floor=0.02) -> np.ndarray:
    """Helmstetter/Zhuang variable-bandwidth KDE, weighted by `w`.

    Each event is smoothed with a bandwidth set by its k-th nearest neighbour
    distance, so dense fault traces stay sharp while isolated events spread. `w`
    is the background probability from the E-step, which is what makes this
    Zhuang's background estimator rather than a raw seismicity map.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    pts = np.column_stack([x, y])
    kk = min(k_neighbour + 1, len(pts))
    dk, _ = cKDTree(pts).query(pts, k=kk)
    sig = np.clip(dk[:, -1], sig_min, sig_max)
    gx = np.clip(((x - xmin) / bin_km).astype(int), 0, nx - 1)
    gy = np.clip(((y - ymin) / bin_km).astype(int), 0, ny - 1)

    edges = np.logspace(np.log10(sig.min()), np.log10(sig.max() + 1e-9), 9)
    grid = np.zeros((nx, ny))
    for b in range(len(edges) - 1):
        sel = (sig >= edges[b]) & (sig <= edges[b + 1] if b == len(edges) - 2
                                   else sig < edges[b + 1])
        if not sel.any():
            continue
        sub = np.zeros((nx, ny))
        np.add.at(sub, (gx[sel], gy[sel]), w[sel])
        grid += gaussian_filter(sub, sigma=math.sqrt(edges[b] * edges[b + 1]) / bin_km)
    tot = grid.sum() * bin_km ** 2
    if tot <= 0:
        return np.full((nx, ny), 1.0 / (nx * ny * bin_km ** 2))
    grid /= tot
    area = nx * ny * bin_km ** 2
    return (1.0 - floor) * grid + floor / area


# --------------------------------------------------------------------------
# likelihood and EM inversion
# --------------------------------------------------------------------------

def intensity_at_events(t, x, y, m, P: ETASParams, bg: Background):
    """Return (bg_rate, trig_rate) per event: the two additive parts of lambda."""
    n = len(t)
    # mu is events/day/km^2 region-wide; bg.at is a density per km^2 normalised
    # over the region, so mu * area * bg.at is the local rate per km^2.
    bg_rate = P.mu * bg.area_km2 * bg.at(x, y)
    trig = np.zeros(n)
    for ci, pj in parent_pairs(t, x, y, m, P):
        dt = t[ci] - t[pj]
        r2 = (x[ci] - x[pj]) ** 2 + (y[ci] - y[pj]) ** 2
        contrib = productivity(m[pj], P) * omori(dt, P) * spatial(r2, m[pj], P)
        np.add.at(trig, ci, contrib)
    return bg_rate, trig


def log_likelihood(t, x, y, m, P: ETASParams, bg: Background,
                   t_start: float, t_end: float) -> float:
    """Proper point-process log-likelihood: sum log lambda - integral lambda.

    The compensator uses the exact temporal integral of the (normalised) Omori
    kernel and treats the spatial kernel as fully contained in the region, which
    is accurate to the region-edge leakage and is applied identically to every
    model being compared.
    """
    bg_rate, trig = intensity_at_events(t, x, y, m, P, bg)
    inwin = (t >= t_start) & (t < t_end)
    lam = bg_rate[inwin] + trig[inwin]
    ll = float(np.sum(np.log(np.maximum(lam, 1e-300))))
    integ = P.mu * bg.area_km2 * (t_end - t_start)
    prior = t < t_end
    if prior.any():
        a = np.maximum(t_start - t[prior], 0.0)
        b = t_end - t[prior]
        integ += float(np.sum(productivity(m[prior], P) *
                              (omori_int(b, P) - omori_int(a, P))))
    return ll - integ


def fit_etas_em(t, x, y, m, mc: float, region: tuple, background: str = "uniform",
                n_iter: int = 20, bin_km: float = 5.0, init: ETASParams | None = None,
                t_start: float | None = None, t_end: float | None = None,
                max_branching: float = 0.99, verbose: bool = False) -> tuple[ETASParams, Background, dict]:
    """Zhuang-style stochastic-declustering EM.

    `region` is (xmin, xmax, ymin, ymax) in km. `background`:
      "uniform"  - the benchmark's ETAS: mu constant over the region.
      "smoothed" - free spatially-variable background, re-estimated each
                   iteration as a background-probability-weighted
                   variable-bandwidth KDE. THIS IS THE GATE G1 CONTROL.

    `max_branching` bounds the branching ratio below 1 so the fitted process is
    stationary and its simulations terminate. See the barrier in the M-step.
    """
    from scipy.optimize import minimize

    xmin, xmax, ymin, ymax = region
    area = (xmax - xmin) * (ymax - ymin)
    nx = max(int(math.ceil((xmax - xmin) / bin_km)), 1)
    ny = max(int(math.ceil((ymax - ymin) / bin_km)), 1)
    t_start = float(t.min()) if t_start is None else t_start
    t_end = float(t.max()) if t_end is None else t_end
    T = t_end - t_start

    P = init or ETASParams(mc=mc, mu=len(t) / (area * T) * 0.3)
    P = replace(P, mc=mc)
    bg = Background(mode=background, area_km2=area, xmin=xmin, ymin=ymin,
                    bin_km=bin_km)
    hist = {"ll": [], "branching_ratio": []}
    best_ll, best = -np.inf, (P, bg)

    for it in range(n_iter):
        # ---- E step: background probability per event -----------------------
        bg_rate, trig = intensity_at_events(t, x, y, m, P, bg)
        lam = np.maximum(bg_rate + trig, 1e-300)
        w_bg = bg_rate / lam

        # ---- M step: background --------------------------------------------
        mu_new = float(w_bg.sum() / (area * T))
        if background == "smoothed":
            grid = smoothed_background(x, y, w_bg, xmin, ymin, nx, ny, bin_km)
            bg = replace(bg, grid=grid)
        P = replace(P, mu=mu_new)

        # ---- M step: triggering kernel (numerical MLE on the pair set) ------
        pairs = list(parent_pairs(t, x, y, m, P))
        if pairs:
            ci = np.concatenate([a for a, _ in pairs])
            pj = np.concatenate([b for _, b in pairs])
            dt = t[ci] - t[pj]
            r2 = (x[ci] - x[pj]) ** 2 + (y[ci] - y[pj]) ** 2
            mp = m[pj]

            def nll(theta):
                K, a_, c_, p_, d_, g_, q_ = np.exp(theta)
                p_ += 1.0 + 1e-6      # p > 1 so Omori integrates
                q_ += 1.0 + 1e-6      # q > 1 so the spatial kernel integrates
                Q = replace(P, K=K, a=a_, c=c_, p=p_, d=d_, gamma=g_, q=q_)
                # Subcriticality barrier. An ETAS with branching ratio >= 1 is
                # non-stationary: it predicts infinite expected offspring and its
                # simulations diverge, so it is useless as a forecast whatever
                # its likelihood. Without this the EM will happily run away when
                # the background is mis-specified — measured n = 469.9 with
                # mu -> 1.7e-11 on a catalog whose region was set too large.
                # The published ComCat inversion sits at n = 0.968
                # (MANUSCRIPT.md:364), close enough that this bound matters on
                # real data too.
                if not np.isfinite(branching_ratio(Q)) or branching_ratio(Q) >= max_branching:
                    return 1e12
                trg = np.zeros(len(t))
                np.add.at(trg, ci, productivity(mp, Q) * omori(dt, Q) *
                          spatial(r2, mp, Q))
                lam_ = np.maximum(P.mu * bg.area_km2 * bg.at(x, y) + trg, 1e-300)
                ll_ = np.sum(np.log(lam_))
                integ = float(np.sum(productivity(m, Q) *
                                     (omori_int(np.maximum(t_end - t, 0.0), Q) -
                                      omori_int(np.maximum(t_start - t, 0.0), Q))))
                return -(ll_ - integ) / len(t)

            th0 = np.log(np.array([P.K, P.a, P.c, max(P.p - 1.0, 1e-3), P.d,
                                   P.gamma, max(P.q - 1.0, 1e-3)]))
            res = minimize(nll, th0, method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-4})
            K, a_, c_, p_, d_, g_, q_ = np.exp(res.x)
            P = replace(P, K=K, a=a_, c=c_, p=p_ + 1.0, d=d_, gamma=g_, q=q_ + 1.0)

        ll = log_likelihood(t, x, y, m, P, bg, t_start, t_end)
        hist["ll"].append(ll)
        hist["branching_ratio"].append(branching_ratio(P))
        if ll > best_ll:
            best_ll, best = ll, (P, bg)
        if verbose:
            print(f"  EM {it:>2}: ll={ll:.2f} n={branching_ratio(P):.3f} "
                  f"mu={P.mu:.3e} K={P.K:.4f} a={P.a:.3f} p={P.p:.3f} q={P.q:.3f}",
                  flush=True)
        if it > 2 and abs(hist["ll"][-1] - hist["ll"][-2]) < 1e-4 * abs(hist["ll"][-1]):
            break

    # The M-step maximises the true likelihood with the background held at its
    # E-step value rather than an exact EM Q-function, so the iteration is
    # coordinate ascent and is NOT guaranteed monotone — measured a peak at
    # iteration 1 followed by a slow decline on synthetic data. Returning the
    # best-likelihood iterate makes the estimator well-defined regardless.
    #
    # Note on identifiability: (d, q) are near-degenerate in this kernel — they
    # trade off while leaving the density almost unchanged, and recovery tests
    # show both drifting ~35% together while mu/K/a/p/gamma land within 10%.
    # That is a property of ETAS, not a defect here, and it does not affect the
    # forecast: everything downstream compares LIKELIHOODS and simulated rates,
    # never individual parameter values.
    hist["best_ll"] = best_ll
    P, bg = best
    return P, bg, hist


# --------------------------------------------------------------------------
# simulation — the producer for the scaling-curve ETAS control
# --------------------------------------------------------------------------

def simulate_etas(P: ETASParams, bg: Background, hist_t, hist_x, hist_y, hist_m,
                  t0: float, horizon_days: float, region: tuple, n_sims: int,
                  b_value: float = 1.0, seed: int = 0,
                  max_events: int = 20000) -> list[np.ndarray]:
    """Simulate `n_sims` catalogs over [t0, t0+horizon), conditioned on history.

    Returns the same format as `flowquake.ntest.simulate_day_events` — a list of
    (n_i, 4) arrays with columns [t_days, x_km, y_km, magnitude] — so both models
    feed the SAME `flowquake.target_process` scorer. That shared format is what
    makes the ETAS control a like-for-like comparison rather than two pipelines
    that happen to print similar numbers.

    Aftershocks of the observed history are included (the dominant term for a
    short horizon), then the simulated events branch among themselves.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = region
    area = (xmax - xmin) * (ymax - ymin)
    t_end = t0 + horizon_days
    beta = b_value * math.log(10.0)

    hist_t = np.asarray(hist_t); hist_x = np.asarray(hist_x)
    hist_y = np.asarray(hist_y); hist_m = np.asarray(hist_m)
    prior = hist_t < t0
    ht, hx, hy, hm = hist_t[prior], hist_x[prior], hist_y[prior], hist_m[prior]
    # only parents whose Omori tail still matters over the window
    recent = ht >= t0 - MAX_PARENT_DAYS
    ht, hx, hy, hm = ht[recent], hx[recent], hy[recent], hm[recent]

    # expected offspring of each historical parent falling inside the window
    w_hist = productivity(hm, P) * (omori_int(t_end - ht, P) - omori_int(t0 - ht, P))

    # background field sampler
    if bg.mode == "smoothed" and bg.grid is not None:
        flat = bg.grid.ravel()
        pgrid = flat / flat.sum()
        nx, ny = bg.grid.shape
    else:
        pgrid = None

    def draw_bg(n):
        if n == 0:
            return np.zeros(0), np.zeros(0)
        if pgrid is None:
            return rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)
        cell = rng.choice(len(pgrid), size=n, p=pgrid)
        gx, gy = cell // ny, cell % ny
        return (bg.xmin + (gx + rng.random(n)) * bg.bin_km,
                bg.ymin + (gy + rng.random(n)) * bg.bin_km)

    def offspring(pt, px, py, pm):
        """One generation of aftershocks inside the window."""
        lam = productivity(pm, P) * (omori_int(t_end - pt, P) -
                                     omori_int(np.maximum(t0 - pt, 0.0), P))
        k = rng.poisson(np.clip(lam, 0, 1e4))
        if k.sum() == 0:
            return (np.zeros(0),) * 4
        par = np.repeat(np.arange(len(pt)), k)
        n = len(par)
        # inverse-CDF sample of Omori truncated to [max(t0,pt), t_end)
        lo = omori_int(np.maximum(t0 - pt[par], 0.0), P)
        hi = omori_int(t_end - pt[par], P)
        u = lo + rng.random(n) * (hi - lo)
        dt = P.c * ((1.0 - u) ** (1.0 / (1.0 - P.p)) - 1.0)
        tn = pt[par] + dt
        # inverse-CDF sample of the power-law spatial kernel
        d2 = P.d ** 2 * 10.0 ** (P.gamma * (pm[par] - P.mc))
        r = np.sqrt(d2 * ((1.0 - rng.random(n)) ** (1.0 / (1.0 - P.q)) - 1.0))
        th = rng.uniform(0, 2 * np.pi, n)
        xn, yn = px[par] + r * np.cos(th), py[par] + r * np.sin(th)
        mn = P.mc + rng.exponential(1.0 / beta, n)
        keep = (tn >= t0) & (tn < t_end) & (xn >= xmin) & (xn <= xmax) \
            & (yn >= ymin) & (yn <= ymax)
        return tn[keep], xn[keep], yn[keep], mn[keep]

    out = []
    for _ in range(n_sims):
        # background events in the window
        n_bg = rng.poisson(P.mu * area * horizon_days)
        bt = t0 + rng.random(n_bg) * horizon_days
        bx, by = draw_bg(n_bg)
        bm = P.mc + rng.exponential(1.0 / beta, n_bg)

        # First generation from the observed history.
        #
        # `offspring` draws its OWN Poisson count from each parent's expected
        # offspring. An earlier version drew k ~ Poisson(w_hist) here and then
        # passed each parent REPEATED k times into `offspring`, which drew again
        # -- so the first generation came out as sum_i w_i^2 rather than
        # sum_i w_i. With w_i < 1 that suppresses history-driven triggering
        # quadratically (measured 0.102 against an analytic 0.277 on the same
        # history), leaving the constant background to dominate the forecast.
        # That is why the ETAS control produced a nearly constant rate field and
        # corr(n_expected, n_observed) of about zero on real data.
        acc_t, acc_x, acc_y, acc_m = [bt], [bx], [by], [bm]
        if len(ht):
            ot, ox, oy, om = offspring(ht, hx, hy, hm)
            acc_t.append(ot); acc_x.append(ox); acc_y.append(oy); acc_m.append(om)

        # cascade among simulated events
        cur = (np.concatenate(acc_t), np.concatenate(acc_x),
               np.concatenate(acc_y), np.concatenate(acc_m))
        allev = [cur]
        n_tot = len(cur[0])
        for _g in range(20):
            if len(cur[0]) == 0 or n_tot > max_events:
                break
            cur = offspring(*cur)
            n_tot += len(cur[0])
            if len(cur[0]):
                allev.append(cur)

        t_ = np.concatenate([a[0] for a in allev])
        if len(t_) == 0:
            out.append(np.zeros((0, 4)))
            continue
        o = np.argsort(t_)
        out.append(np.column_stack([
            t_[o],
            np.concatenate([a[1] for a in allev])[o],
            np.concatenate([a[2] for a in allev])[o],
            np.concatenate([a[3] for a in allev])[o]]))
    return out


# --------------------------------------------------------------------------
# analytic rate field — no Monte-Carlo, no resolution bias
# --------------------------------------------------------------------------

def etas_rate_field(P: ETASParams, bg: Background, hist_t, hist_x, hist_y, hist_m,
                    grid, t0: float, horizon_days: float, tail_weight: float = 1.0,
                    max_parent_days: float = MAX_PARENT_DAYS) -> np.ndarray:
    """Expected count of target events per grid cell over [t0, t0+horizon).

    Computed in CLOSED FORM rather than by simulation. That matters because the
    simulated alternative carries a Monte-Carlo resolution bias which is the
    largest single artifact found in this pipeline: the shape score's per-cell
    relative variance is 1/(T*p_c) with T the total simulated events, T grows
    directly with the catalog rate above mc, and the resulting bias reached
    +2.6 nats/decade on a null whose true slope is zero. Removing the Monte
    Carlo removes that bias exactly, rather than pushing it down by brute force
    -- reaching T = 20,000 per window on a sparse catalog at a 1-day horizon
    costs ~165M simulations per curve point, about 1.7 hours.

    Each kernel here is a normalised density, so the integrals are analytic:

        background      mu * cell_area * horizon           (or mu * bg.grid)
        direct offspring  sum_i  K*10^(a(m_i-mc))
                                 * [Omori_int(t_end-t_i) - Omori_int(t0-t_i)]
                                 * spatial(r_ic^2; m_i) * cell_area

    APPROXIMATION, and it is the one thing to state when reporting this: only
    DIRECT offspring of observed history are included. Events triggered by
    events that occur *inside* the forecast window are omitted. This is the
    standard first-generation approximation used for short-horizon operational
    forecasts, and its size here is measured rather than assumed.

    Measured against the full branching simulator (40,000 simulations) on the
    parameters actually fitted to WHITE at mc 2.0 -- which are near-critical,
    branching ratio 0.990 -- over four scattered 1-day windows:

        total rate      simulated / analytic = 1.10, 1.11, 1.17, 1.14
        spatial shape   total-variation distance 0.099 - 0.142

    So secondary generations add about 13% of the LEVEL. The infinite-time
    multiplier at n = 0.99 would be 1/(1-n) = 100; it does not apply because
    Omori decay puts almost all of that outside a 1-day window. The 13% is
    also mostly harmless to the primary metric, since the shape term is
    invariant to a uniform rescaling of lambda -- only the residual spatial
    difference matters, and the quoted TV is an upper bound because the
    simulated reference carries its own Monte-Carlo noise.

    `tail_weight` is the fixed GR thinning P(m >= m_target | m >= mc), applied
    analytically for the same reason the scorer applies it analytically.
    """
    ht = np.asarray(hist_t, dtype=float)
    hx = np.asarray(hist_x, dtype=float)
    hy = np.asarray(hist_y, dtype=float)
    hm = np.asarray(hist_m, dtype=float)

    t_end = t0 + horizon_days
    area = grid.cell_area_km2

    # cell centres
    gx = (np.arange(grid.nx) + 0.5) * grid.bin_km + grid.xmin
    gy = (np.arange(grid.ny) + 0.5) * grid.bin_km + grid.ymin
    cx = np.repeat(gx, grid.ny)          # flat index is gx * ny + gy
    cy = np.tile(gy, grid.nx)

    # --- background -------------------------------------------------------
    if bg.mode == "smoothed" and bg.grid is not None:
        # The smoothed background lives on its OWN grid, which generally has a
        # different origin and cell size from the scoring grid. Resample it by
        # looking up each scoring cell's centre, rather than assuming the two
        # rasters line up -- an earlier version silently fell back to a uniform
        # background when the shapes differed, which would have made gate G1
        # report "smoothed" numbers that were actually uniform.
        nbx, nby = bg.grid.shape
        bi = np.clip(((cx - bg.xmin) / bg.bin_km).astype(int), 0, nbx - 1)
        bj = np.clip(((cy - bg.ymin) / bg.bin_km).astype(int), 0, nby - 1)
        dens = np.asarray(bg.grid[bi, bj], dtype=float)
        if dens.sum() > 0:
            # Normalise to MEAN 1 over the scoring grid, so the smoothed field
            # carries the same TOTAL background rate as the uniform one and only
            # redistributes it. Normalising against bg.area_km2 instead would
            # make the two backgrounds differ in level as well as in shape, and
            # gate G1's comparison would be reading both at once.
            dens = dens / dens.mean()
            lam = P.mu * area * horizon_days * dens
        else:
            lam = np.full(grid.n_cells, P.mu * area * horizon_days)
    else:
        lam = np.full(grid.n_cells, P.mu * area * horizon_days)

    # --- direct offspring of prior events ---------------------------------
    prior = ht < t0
    if prior.any():
        ht, hx, hy, hm = ht[prior], hx[prior], hy[prior], hm[prior]
        recent = ht >= t0 - max_parent_days
        ht, hx, hy, hm = ht[recent], hx[recent], hy[recent], hm[recent]

    if len(ht):
        # expected offspring of each parent falling inside the window
        w = productivity(hm, P) * (omori_int(t_end - ht, P) - omori_int(t0 - ht, P))
        keep = w > 1e-12
        ht, hx, hy, hm, w = ht[keep], hx[keep], hy[keep], hm[keep], w[keep]

    if len(ht):
        # blocked over parents so the cells x parents array stays bounded
        block = max(1, int(4_000_000 // max(grid.n_cells, 1)))
        for s in range(0, len(ht), block):
            e = slice(s, s + block)
            r2 = ((cx[:, None] - hx[None, e]) ** 2 +
                  (cy[:, None] - hy[None, e]) ** 2)
            lam += (spatial(r2, hm[None, e], P) * w[None, e]).sum(axis=1) * area

        # Midpoint quadrature is wrong where the kernel varies inside one cell,
        # and it does: the spatial scale d fitted on WHITE is 1.0 km against
        # 2 km cells. Evaluating the density at the cell CENTRE and multiplying
        # by the area overestimates the parent's own cell by 1.94x and
        # underestimates its neighbours by 0.84x -- so the field is
        # over-concentrated exactly where a target is most likely to fall, and
        # the distortion lands in the SHAPE term, which is the primary metric.
        #
        # Correct it only near each parent, where it matters: at 4 km the error
        # is already 8% and at 10 km under 2%. A k x k sub-grid over the near
        # block brings every cell within ~2% at k=4 (1.94 -> 1.015 for the
        # parent's own cell). Cost is parents x near-cells x k^2, which is tiny
        # next to the full cells x parents product.
        _subgrid_correction(lam, P, grid, hx, hy, hm, w, cx, cy, area)

    return lam * tail_weight


def _subgrid_correction(lam, P, grid, hx, hy, hm, w, cx, cy, area,
                        k: int = 4, n_rad: int = 3) -> None:
    """Replace midpoint quadrature with a k x k sub-grid near each parent.

    Modifies `lam` in place. `n_rad` is the half-width of the corrected block
    in cells; 3 covers a 7x7 neighbourhood, beyond which midpoint is already
    accurate to a couple of percent for the kernels fitted here.
    """
    b = grid.bin_km
    off = (np.arange(k) + 0.5) / k * b - b / 2.0
    ox, oy = np.meshgrid(off, off, indexing="ij")
    ox, oy = ox.ravel(), oy.ravel()                        # (k^2,)

    di = np.arange(-n_rad, n_rad + 1)
    dix, diy = np.meshgrid(di, di, indexing="ij")
    dix, diy = dix.ravel(), diy.ravel()                    # (C,)

    pi = np.floor((hx - grid.xmin) / b).astype(np.int64)
    pj = np.floor((hy - grid.ymin) / b).astype(np.int64)

    ii = pi[:, None] + dix[None, :]                        # (P, C)
    jj = pj[:, None] + diy[None, :]
    ok = (ii >= 0) & (ii < grid.nx) & (jj >= 0) & (jj < grid.ny)
    flat = ii * grid.ny + jj

    ccx = grid.xmin + (ii + 0.5) * b                       # near-cell centres
    ccy = grid.ymin + (jj + 0.5) * b

    mid_r2 = (ccx - hx[:, None]) ** 2 + (ccy - hy[:, None]) ** 2
    mid = spatial(mid_r2, hm[:, None], P) * area

    sub_r2 = ((ccx[:, :, None] + ox[None, None, :] - hx[:, None, None]) ** 2 +
              (ccy[:, :, None] + oy[None, None, :] - hy[:, None, None]) ** 2)
    sub = spatial(sub_r2, hm[:, None, None], P).mean(axis=2) * area

    delta = ((sub - mid) * w[:, None])[ok]
    np.add.at(lam, flat[ok], delta)
