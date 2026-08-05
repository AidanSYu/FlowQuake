# ETAS, derived and dissected

The Epidemic-Type Aftershock Sequence model is the thing FlowQuake is measured
against. It is a marked Hawkes process with nine parameters and a hand-built
kernel, and it has beaten every neural point process anyone has published on the
EarthquakeNPP benchmark. This chapter derives it from scratch, transcribes the
*exact* variant this repository scores against, works out its normalization and
its branching ratio by hand, derives the EM inversion, and then attacks it.

---

## What this chapter buys you

- You can **write down ETAS's conditional intensity from first principles** —
  starting from "each earthquake independently triggers offspring" — and say
  which line of the derivation each of the three empirical laws enters at.
- You can **derive the two normalizers that the benchmark actually scores**:
  `Z_j = pi / (rho * d_j^rho)` (the spatial integral) and the compensator
  `Lambda(t)`, and explain precisely why `sll` is a *conditional density given
  that an event occurs*, not an intensity.
- You can **compute the branching ratio for this parameterization**, show that
  the effective productivity exponent is `a − rho*gamma` and not `a`, and state
  the convergence condition and what violating it means physically.
- You can **derive the EM / stochastic-declustering inversion**, name which
  M-step updates are closed-form and which are not, and explain the `O(N_ev²)` cost.
- You can **attack ETAS from six directions** (edge effects, incompleteness,
  identifiability, `m_c` sensitivity, non-stationarity, uniform background) and
  say what the standard fixes are and which ones this benchmark does not apply.
- You can **state formally in what sense FlowQuake's neural-ETAS head is a strict
  superset of ETAS**, verify the claim algebraically, and name the two places
  where the word "strict" is doing more work than it should.

---

## Prerequisites

Read these first. This chapter assumes them without re-deriving them.

- [Chapter 1 — point processes, conditional intensity, and the likelihood](01-point-processes.md):
  `lambda(t | H_t)`, the compensator `Lambda(t)`, the two equivalent forms of
  `log L`, and the `lambda ↔ f(tau | H)` correspondence.
- [Chapter 2 — seismology for the point-process modeller](02-seismology.md):
  Gutenberg–Richter (§5), Omori–Utsu (§6), Utsu productivity (§7), and the
  completeness magnitude `m_c` (§4).
- [STACK.md](../STACK.md) Parts 0 and I, for the repo's score conventions
  (`tll`, `sll`, `mll`, `nll`) and the two-models-in-one-repo structure. This
  chapter is the theory STACK.md's Part II assumes; it does not repeat the code
  walkthrough.

**Notation.** Following the shared convention: `t` in days since catalog start,
`tau` = inter-event gap, `s = (x, y)` in km, `m` magnitude, `m_c` completeness,
`beta = b * ln 10`. **Disambiguation:** the `etas` package and this repo call the
Omori exponential-taper timescale `tau`, colliding with the inter-event gap.
Throughout this chapter the taper timescale is written **`tau_tap`**. Wherever a
code snippet is transcribed verbatim you will see the repo's `tau`; that is
always `tau_tap`. Two further local conventions, to avoid collisions: `A` is the
region area (the Gutenberg–Richter intercept is written `a_GR`), and `N_ev` is
the number of events in a catalog (`n` is reserved throughout for the branching
ratio).

---

## 1. The problem ETAS was invented to solve

Seismicity is *clustered*. Fit a homogeneous Poisson process to a catalog and
you get exponential, independent inter-event times; real ones have a spike near
zero (aftershocks within minutes) and a heavy tail (quiet years), with daily-count
variance orders of magnitude above Poisson.

There were two responses. **Declustering** strips aftershocks with a windowing
rule (Gardner–Knopoff 1974, *BSSA* is the canonical one) and models the remainder
as Poisson — but "aftershock" is not a property an event carries, it is an
inference, and any deterministic window is arbitrary. Ogata's response was to
*model* the clustering and let the parent–child assignment stay latent. That is
ETAS.

The design goal, stated sharply, explains every subsequent choice:

> Build a conditional intensity `lambda(t, s, m | H_t)` that (i) reproduces the
> three empirical laws exactly by construction, (ii) has a closed-form
> space-time integral so the likelihood is computable, and (iii) has few enough
> parameters to invert from one region's catalog.

Requirement (ii) is the reason for every functional form in ETAS. Power laws in
`(t+c)` and `(r²+d)` are not chosen because they fit best; they are the
heavy-tailed shapes whose integrals you can do in closed form.

---

## 2. Lineage

| year | who | what |
|---|---|---|
| 1894 | Omori | aftershock rate `∝ 1/(t + c)` |
| 1944 | Gutenberg & Richter | `log10 N(≥m) = a_GR − b m` (`a_GR` written so as not to collide with the region area `A`) |
| 1961 | Utsu | generalizes Omori to `(t+c)^{−p}`, `p ≈ 1.1` |
| **1988** | **Ogata**, *JASA* 83(401), 9–27 | **temporal ETAS**: `lambda(t) = mu + Σ_j K e^{a(m_j−m_c)} (t−t_j+c)^{−p}`, plus residual analysis via the random-time-change theorem |
| **1998** | **Ogata**, *Ann. Inst. Statist. Math.* 50(2), 379–402 | **space–time ETAS**: adds an isotropic power-law spatial kernel whose scale grows with parent magnitude |
| 2002 | Zhuang, Ogata & Vere-Jones, *JASA* 97(458), 369–380 | **stochastic declustering**: the EM view; nonparametric background estimated from background probabilities |
| 2008 | Veen & Schoenberg, *JASA* 103(482), 614–624 | ETAS estimation as an EM-type algorithm, made explicit |
| 2005 | Gerstenberger et al., *Nature* | STEP — the first operational daily aftershock forecast (superimposed Omori sequences, not literally ETAS) |
| 2021 | Mizrahi, Nandan & Wiemer | the `etas` Python package (`lmizrahi/etas`) — the implementation this benchmark uses |
| 2026 | Stockman, Lawson & Werner, *TMLR* | EarthquakeNPP: five California catalogs, fixed splits, `etas`-fitted baselines, five neural point processes, none of which beat ETAS |

Ogata 1988 is the paper to be able to talk about. Its two contributions are the
model *and* the diagnostic: under the random time change `t → Lambda(t)`, a
correctly specified point process becomes a unit-rate Poisson process, so you can
test a seismicity model with a KS test on transformed times — the ancestor of
every modern point-process residual check.

Two operational descendants matter for framing FlowQuake. **ETAS-as-simulator**:
operational systems (USGS aftershock forecasts, Italian OEF, Switzerland's
`suiETAS` — Mizrahi et al. 2024, *BSSA* 114(5), 2591–2612) do not
evaluate `lambda` on a grid, they *simulate* catalog continuations from the
fitted branching process — the mode
[flowquake/etas_csep.py](../flowquake/etas_csep.py) runs ETAS in, via the
package's `ETASSimulation`. And **CSEP** (Schorlemmer et al. 2007, *SRL*), which
standardized the N/S/M consistency tests and makes "ETAS is the incumbent" an
institutional fact rather than a benchmark artifact.

---

## 3. Building ETAS from three laws

### 3.1 The branching construction

Define a marked branching (Hawkes) process on `[0, T] × A × [m_c, ∞)`, where `A`
is the study region:

1. **Immigrants** arrive as an inhomogeneous Poisson process with intensity
   `mu(s)` per km² per day. In the plain model `mu(s) = mu`, a constant.
2. **Every event** `j = (t_j, s_j, m_j)`, whether immigrant or offspring,
   independently spawns a Poisson process of *direct* offspring with intensity

   ```
   g_j(t, s) = kappa(m_j) · phi(t − t_j) · psi(s − s_j ; m_j)
   ```

   where `phi` integrates over time to 1 up to a constant, `psi` is a spatial
   density shape, and `kappa(m_j)` sets the expected number.
3. **Magnitudes** are drawn i.i.d. from the Gutenberg–Richter density,
   `f_m(m) = beta · exp(−beta (m − m_c))` for `m ≥ m_c`, **independently of
   everything else** — independent of the parent's magnitude, of the time since
   the parent, and of location.

The three empirical laws enter at exactly three places:

- Omori–Utsu → `phi(u) ∝ (u + c)^{−p}`.
- Utsu productivity → `kappa(m) ∝ e^{a(m − m_c)}`.
- Gutenberg–Richter → step 3, and (through the productivity) the branching ratio.

### 3.2 From branching to conditional intensity

This is the step people skip. Why does "a superposition of independent Poisson
offspring processes" give an *additive* conditional intensity?

Two facts about Poisson processes do all the work:

- **Superposition.** If `N_1, …, N_k` are independent Poisson processes with
  intensities `nu_1, …, nu_k`, then `Σ N_i` is Poisson with intensity `Σ nu_i`.
- **Conditioning on the past.** Given `H_t` — the times, locations and magnitudes
  of every event before `t` — the set of *already existing* parents is known, and
  each contributes a deterministic (given `H_t`) intensity `g_j(t, s)`. Events not
  yet born contribute nothing at time `t` because their parents do not exist yet.

Hence

```
lambda(t, s | H_t)  =  mu(s)  +  Σ_{j : t_j < t}  g_j(t, s)
```

and the full marked intensity factorizes as

```
lambda(t, s, m | H_t)  =  lambda(t, s | H_t) · f_m(m)
```

because magnitudes are drawn independently. That last factorization is exactly
the chain-rule split the benchmark scores as `tll`, `sll`, `mll`
([STACK.md §3](../STACK.md#3-marks-factorization-and-the-three-scores)). It is
an *assumption*, not a theorem — see the misconceptions section.

Note what is being assumed by "each event independently spawns offspring":
**the process has no state beyond the event list.** There is no stress variable,
no fault-loading term, no memory of what a previous sequence did to the crust.
Everything ETAS knows is encoded in `(t_j, s_j, m_j)` tuples. That is both its
robustness and its ceiling (§9).

### 3.3 The general form and its common variants

A common way to write the space–time ETAS conditional intensity (this is the
*unnormalized-kernel* convention; the pre-normalized convention is the
"spatial normalization" row of the table below, and the difference matters — §6.2):

```
                                       K0 · e^{a(m_j − m_c)}                1
lambda(t,s|H) = mu(s) +   Σ        ───────────────────────── · ──────────────────────────
                       t_j < t          (t − t_j + c)^p         (|s − s_j|² + d·e^{gamma(m_j−m_c)})^{1+rho}
```

Variants you should be able to name and distinguish:

| axis | variant A | variant B | consequence |
|---|---|---|---|
| Omori tail | pure `(u+c)^{−p}` | `e^{−u/tau_tap} (u+c)^{−p}` (**this repo**) | the taper makes the time integral converge even when `p ≤ 1` |
| spatial kernel | power law `(r²+d)^{−(1+rho)}` (**this repo**) | Gaussian `e^{−r²/2σ²}`; or short-term/long-term mixture | power law keeps distant triggering possible; Gaussian pays huge likelihood penalties on tail events |
| spatial normalization | kernel written **unnormalized**, magnitude scale in `d` (**this repo**) | kernel pre-normalized to integrate to 1, productivity carried entirely by `K0 e^{a Δm}` | changes the meaning of `a` — see §6.2, this is the single most common source of confusion |
| isotropy | isotropic (**this repo**) | anisotropic, elongated along strike; or finite-fault source geometry | isotropic is wrong for `M ≥ 6` ruptures whose length exceeds the aftershock-zone width |
| background | uniform `mu` (**this repo's ETAS**) | fixed smoothed-seismicity map; or nonparametric, estimated jointly by stochastic declustering | uniform is a real handicap in California — §12 shows it is where most of FlowQuake's spatial win comes from |
| parameters | global | spatially varying `(mu(s), K0(s), a(s), …)` | fixes some non-stationarity; costs identifiability |

---

## 4. The exact form used by this benchmark

Everything below is transcribed from code in this repository, not from memory.
It also matches the triggering kernel printed in the benchmark paper itself
(Stockman, Lawson & Werner, arXiv:2410.08226 §2), which writes

```
g(t, r², m) = [ e^{−t/tau} · k · e^{a(m − Mcut)} ]  /  [ (t + c)^{1+omega} · (r² + d·e^{gamma(m − Mcut)})^{1+rho} ]
```

— term for term the same, with the paper's `k`, `Mcut` for the repo's `k0`, `mc`.
So the *functional form* is pinned to a primary source even though the *fork*
that produced the fitted parameters is not (§7.5).

### 4.1 Transcription

From [flowquake/neural_etas.py](../flowquake/neural_etas.py) lines 78–87 (with
the learned offsets `dlogw`, `dlogd`, `drho` set to zero, which is the ETAS
special case):

```python
w   = (k0 * exp(a * (mag - mc)) * exp(-dt / tau) * pow(dt + c, -(1.0 + omega))) * valid
dmj = d * exp(gamma * (mag - mc))
kj  = pow(r2 + dmj, -(1.0 + rho))
zj  = pi / (rhoj * pow(dmj, rhoj))
near_num = (w * kj).sum(-1)
near_den = (w * zj).sum(-1)
```

The parameters are read from the `etas` package's `parameters_0.json` in
[scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
lines 52–60 — note that five of the nine are stored as base-10 logarithms:

```python
mu = 10 ** p["log10_mu"];  k0 = 10 ** p["log10_k0"];  c = 10 ** p["log10_c"]
tau = 10 ** p["log10_tau"];  d = 10 ** p["log10_d"]
a, omega, gamma, rho = p["a"], p["omega"], p["gamma"], p["rho"]
mc, area, R = meta["mc"], meta["area"], meta["earth_radius"]
```

In shared notation, the **triggering weight** of past event `j` (magnitude `m_j`,
elapsed time `Δt = t − t_j`) is

```
                             ┌ productivity ┐  ┌  taper  ┐  ┌── Omori ──┐
w_j(Δt) = k0 · exp(a (m_j − m_c)) · exp(−Δt / tau_tap) · (Δt + c)^{−(1+omega)}
```

the **aftershock zone** and **spatial kernel** are

```
d_j     = d · exp(gamma (m_j − m_c))            [km², a squared length]
K_j(r²) = (r² + d_j)^{−(1+rho)}                 [unnormalized]
```

and the conditional intensity of the marked process is

```
lambda(t, s | H_t) = mu  +  Σ_{t_j < t}  w_j(t − t_j) · K_j(|s − s_j|²)
```

with `mu` a **density** (events per km² per day) and `r` computed as a haversine
great-circle distance with `R = meta["earth_radius"]`
([scripts/etas_sll_repro.py](../scripts/etas_sll_repro.py), lines 61–64).

### 4.2 The nine parameters

| symbol | code name | units | what it controls | typical scale |
|---|---|---|---|---|
| `mu` | `mu` (`10^log10_mu`) | events km⁻² day⁻¹ | background rate density (uniform over the region) | `log10_mu ≈ −6.3` for ComCat_25 → `mu ≈ 4.6e−7`; times a ~10⁶ km² region gives ~0.45 background events/day |
| `k0` | `k0` (`10^log10_k0`) | mixed | overall productivity level at `m = m_c` | small; `log10_k0` around −3 |
| `a` | `a` | mag⁻¹ | how fast productivity grows with magnitude | ≈ 1.5–1.7 (ComCat_25: **1.556**) |
| `c` | `c` (`10^log10_c`) | days | Omori regularizer; flattens the rate inside `Δt < c` | 10⁻³–10⁻² d (minutes to a quarter hour) |
| `omega` | `omega` | — | Omori exponent offset; **`p = 1 + omega`** | small, near 0; can be *slightly negative* |
| `tau_tap` | `tau` | days | exponential cutoff on the Omori tail | very long, 10³–10⁴ d (years to decades) |
| `d` | `d` (`10^log10_d`) | km² | aftershock-zone area scale at `m = m_c` | ~0.5–5 km² |
| `gamma` | `gamma` | mag⁻¹ | how fast the aftershock zone grows with magnitude | ≈ 1.2 |
| `rho` | `rho` | — | spatial tail exponent; kernel decays as `r^{−2(1+rho)}` | ≈ 0.5–0.6 (ComCat_25: **0.557**) |

**Provenance warning, and you should raise it before a professor does.** The
three ComCat_25 values in bold (`a = 1.556`, `log10_mu = −6.333`, `rho = 0.557`,
plus the refit's branching ratio 0.968) are quoted from
[MANUSCRIPT.md](../MANUSCRIPT.md) lines 363–364. They are **not backed by any
committed artifact**: `results/CLAIMS.md` row **N2** records that
`runs/forward_etas_ComCat_25_refit2020/summary.json` holds only
`window / n / tll / sll / nll / etas_name / fit_window / params_frozen_from`, and
that *no ETAS parameter vector is committed anywhere under `runs/`*. The
`reference/` tree that holds `parameters_0.json` is gitignored. The other
"typical scale" entries in the table are order-of-magnitude values from the ETAS
literature, not from this repo — verify against your own `parameters_0.json`
before quoting any of them in a viva. The same applies to the region area `A`:
`meta["area"]` is read by the code but is **not committed anywhere** either, so
every "~10⁶ km²" in this chapter is a stated order of magnitude, never a
measurement.

### 4.3 `p = 1 + omega`, and why anyone would write it that way

The Omori–Utsu law is normally `n(t) ∝ (t + c)^{−p}`. The `etas` package
parameterizes the exponent as `1 + omega`, so

```
p = 1 + omega        omega = p − 1
```

`p ≈ 1.1` corresponds to `omega ≈ 0.1`. Two reasons for the offset
parameterization:

1. **The interesting regime is near `p = 1`.** Writing the exponent as an offset
   from 1 puts the physically special case (`p = 1`, logarithmic divergence of
   the aftershock count) at `omega = 0`, which is a better-conditioned place for
   an optimizer to sit than `p = 1` in an unbounded coordinate.
2. **It admits `p < 1` gracefully.** Fitted `omega` is often near zero and is
   sometimes slightly negative. Under a pure Omori law `p ≤ 1` makes
   `∫₀^∞ (u+c)^{−p} du` diverge — an infinite expected number of aftershocks per
   event, which is not a usable model. This parameterization is paired with the
   exponential taper (next), which makes the integral finite for *any* `omega`.

**Do not confuse `rho` with `p`.** `rho` is the *spatial* tail exponent. The
kernel `(r² + d_j)^{−(1+rho)}` decays as `r^{−2(1+rho)}`, so the radial survivor
function decays as `r^{−2 rho}` (derived in §5.2). Different letter, different
axis, both power laws.

### 4.4 What the taper buys, precisely

The time factor is `phi(u) = e^{−u/tau_tap} (u + c)^{−(1+omega)}`. Define

```
H(T) = ∫₀^T e^{−u/tau_tap} (u + c)^{−(1+omega)} du
```

which is exactly what [scripts/etas_forward_eval.py](../scripts/etas_forward_eval.py)
lines 57–64 computes by cumulative quadrature on a log mesh and interpolates.

- Without the taper, `H(∞)` converges iff `omega > 0`.
- **With** the taper, `H(∞) < ∞` for every `omega`, because `e^{−u/tau_tap}` kills
  the tail. Near `u = 0` the integrand behaves like `u^{−(1+omega)}` regularized
  by `c`, and `c > 0` makes it bounded.

So `tau_tap` is not cosmetic; it is what makes the model integrable in the
regime where the data actually pushes `omega`. It is also nearly unidentifiable
when `tau_tap` is decades: distinguishing "tapered at 27 years" from "not
tapered" requires watching individual sequences decay for much longer than
`tau_tap`, and a 40–50 year catalog gives you at most one taper timescale of
post-mainshock observation for its *earliest* mainshocks and far less for the
rest. Expect that question.

---

## 5. Normalization, in full

This section contains the two derivations the benchmark's scores rest on.

### 5.1 The spatial integral `Z_j`

**Claim.** For `rho > 0` and `d_j > 0`,

```
Z_j  =  ∫_{R²} (|s − s_j|² + d_j)^{−(1+rho)} dA(s)  =  pi / (rho · d_j^rho)
```

**Derivation.** Work in polar coordinates centered at `s_j`. The integrand
depends only on `r = |s − s_j|`, so `dA = 2 pi r dr`:

```
Z_j = ∫₀^∞ (r² + d_j)^{−(1+rho)} · 2 pi r dr
```

Substitute `u = r²`, so `du = 2r dr` and `2 pi r dr = pi du`:

```
Z_j = pi ∫₀^∞ (u + d_j)^{−(1+rho)} du
```

The antiderivative of `(u + d_j)^{−(1+rho)}` is `−(u + d_j)^{−rho} / rho`, so

```
                    [   (u + d_j)^{−rho}  ]^∞           d_j^{−rho}          pi
Z_j = pi · ( −1/rho)[                     ]    = pi · ─────────────  =  ───────────
                    [                     ]_0             rho            rho · d_j^rho
```

using `(u + d_j)^{−rho} → 0` as `u → ∞`, which needs **`rho > 0`**. ∎

This is line 85 of [flowquake/neural_etas.py](../flowquake/neural_etas.py) and
line 83 of
[scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py).
Note the magnitude dependence hiding inside: since `d_j = d e^{gamma(m_j−m_c)}`,

```
Z_j = (pi / (rho · d^rho)) · e^{−rho · gamma · (m_j − m_c)}
```

A bigger parent has a *smaller* `Z_j`. That is not a bug: `K_j` is unnormalized,
and a bigger parent's kernel is broader and hence *lower* at every fixed `r`
relative to its peak, so its integral shrinks even though its cloud is wider.
This is the coupling that reappears in §6.2 and it is the single most commonly
botched part of this parameterization.

### 5.2 The offspring location density, and its moments

Normalizing `K_j` gives the density of *where a given parent's offspring lands*:

```
g_j(r) = K_j(r²) / Z_j = (rho / pi) · d_j^rho · (r² + d_j)^{−(1+rho)}
```

Check: `∫₀^∞ g_j(r) 2 pi r dr = 2 rho d_j^rho ∫₀^∞ (r²+d_j)^{−(1+rho)} r dr = 1` by
the same `u = r²` substitution. ✓

The radial **survivor function** is

```
P(R > r) = d_j^rho · (r² + d_j)^{−rho}        →   ~ (d_j / r²)^rho   for r² ≫ d_j
```

Three consequences you should have at your fingertips:

- **Median offspring distance.** Set `P(R > r) = 1/2`:
  `r_med = sqrt( d_j · (2^{1/rho} − 1) )`. With `rho = 0.55`,
  `2^{1/0.55} = 3.527`, so `r_med ≈ 1.590 · sqrt(d_j)`. For an M6 with
  `d_j = 66.7 km²` that is **≈ 13.0 km**; for an M3.5 with `d_j = 3.32 km²` it
  is **≈ 2.9 km**. Those are physically right, which is a useful sanity check
  that you have the parameterization the right way round.
- **`E[R]` exists iff `rho > 1/2`.** `E[R] = ∫₀^∞ P(R>r) dr` and the integrand
  decays as `r^{−2 rho}`. At ComCat_25's fitted `rho = 0.557` the mean distance
  is finite but *barely*.
- **`E[R²]` never exists** for `rho ≤ 1`. The aftershock cloud has infinite
  variance in distance at every plausible fitted `rho`. Anyone who summarizes an
  ETAS spatial kernel by a standard deviation is wrong, and any Gaussian-mixture
  spatial head is going to be punished on the tail events — which is exactly why
  FlowQuake's production spatial head uses an ETAS-shaped component and not a
  Gaussian ([STACK.md §10](../STACK.md#10-headspy--the-spatial-and-magnitude-heads)).

### 5.3 The conditional spatial density — the quantity `sll` scores

This is the most-misunderstood object in the benchmark, so take it slowly.

`lambda(t, s | H_t)` has units of events per km² per day. It is *not* a
probability density: it does not integrate to 1 over anything. To score "did the
model know where the event would be?", the benchmark conditions on the event
having occurred at that instant and asks for the density of its location.

Fix a time `t` and condition on the event `{ an event occurs in [t, t+dt) }`.
By the definition of intensity, the probability of an event in
`[t, t+dt) × dA(s)` is `lambda(t, s | H_t) dt dA(s)`. The probability of an
event anywhere in the region in `[t, t+dt)` is
`(∫_A lambda(t, s' | H_t) dA(s')) dt`. Dividing, the `dt` cancels:

```
                      lambda(t, s | H_t)              mu  +  Σ_j w_j K_j(|s−s_j|²)
f_s(s | H_t)  =  ───────────────────────────  =  ────────────────────────────────────
                  ∫_A lambda(t, s' | H_t) dA        mu·A  +  Σ_j w_j Z_j
```

That is line 143 of
[scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py):
`np.log(mu + trig_num) - np.log(mu * area + trig_den)`. And
`sll = log f_s(s_i | H_{t_i})`, in log 1/km².

Four things to be precise about:

1. **The denominator is the total rate over the region**, `lambda*(t)` in the
   repo's naming ([scripts/etas_forward_eval.py](../scripts/etas_forward_eval.py)
   line 115: `lam_star = mu*area + (w * zint).sum()`), in events per day. It also
   serves as the *temporal* intensity — same object, two roles. That is why
   `TLL = log(lam_star) − ∫ lam_star` and `SLL = log(lam) − log(lam_star)` at
   lines 127–128, and `TLL + SLL = log(lam) − ∫ lam_star` recovers the joint
   space-time log-likelihood term. The factorization is exact, not an
   approximation.
2. **`sll` has no reference scale.** `sll = −8.6898` on ComCat_25
   (`runs/etas_sll_repro.json`, `mean_sll_ref`) means `e^{−8.6898} ≈ 1.68e−4`
   per km², i.e. as if the mass were spread uniformly over ≈ 5,940 km². Chile's
   ETAS gets `sll = −12.4848` (`runs/multiregion_master.json`, `Chile.ETAS.sll`)
   — an equivalent area of ≈ 264,000 km², about 44× larger. **`sll` is not
   comparable across regions.** Only the paired difference against the same
   region's ETAS is.
3. **`f_s` depends on the history but not on the current event's magnitude.**
   Because magnitudes are i.i.d. GR draws, `f_s` is the same whether the event
   about to happen is an M2.5 or an M7. This is a real assumption and a real
   failure mode (§9).
4. **`f_s` is sub-normalized over the region.** See §5.5.

### 5.4 The temporal factor and the compensator

The total-rate process `lambda*(t) = mu·A + Σ_j w_j(t − t_j) Z_j` is a
one-dimensional conditional intensity, and by
[Chapter 1](01-point-processes.md) its log-likelihood contribution for event `i`
is

```
TLL_i = log lambda*(t_i)  −  ∫_{t_{i-1}}^{t_i} lambda*(u) du
```

The compensator has a closed form up to the one-dimensional quadrature `H`:

```
Lambda(T) = ∫₀^T lambda*(u) du = mu·A·T  +  Σ_{t_j < T}  k0 e^{a(m_j − m_c)} · Z_j · H(T − t_j)
```

which is exactly
[scripts/etas_forward_eval.py](../scripts/etas_forward_eval.py) lines 116–117.
The `Z_j` factor pulls out of the time integral because the space and time parts
of `g_j` are separable — the model's entire tractability rests on that
separability.

One implementation subtlety worth knowing because it is the kind of thing a
professor loves: the compensator must be **anchored at the scoring-window
start**, not at the catalog start. Lines 121–126 compute
`anchor = Lambda(t_start)` including the contribution of every pre-window event,
then take first differences. Getting the anchor wrong corrupts only the first
target's `TLL` — the repo's own note says the effect is ~1.5e−4 nats on average
([MANUSCRIPT.md](../MANUSCRIPT.md) lines 344–348) — but the reproduction of
that temporal check is **not committed** (`results/CLAIMS.md` row N4).

### 5.5 The mass leak: `Z_j` integrates over the plane, not the region

`Z_j` was derived over all of `R²`. The region `A` is finite. Therefore

```
∫_A lambda(t, s | H) dA  =  mu·A + Σ_j w_j ∫_A K_j  ≤  mu·A + Σ_j w_j Z_j
```

with strict inequality whenever any parent carries positive weight: the
power-law kernel has infinite support, so *every* parent leaks something, and
the only question is how much. Consequently

```
∫_A f_s(s | H) dA  ≤  1
```

**ETAS's scored spatial density is sub-normalized.** It is a *conservative*
density: it under-claims, so `sll` is systematically slightly pessimistic. How
much? For a parent at distance `L` from the boundary,
`P(R > L) = d_j^rho (L² + d_j)^{−rho}`; for `d_j = 66.7 km²` (an M6),
`L = 50 km`, and `rho = 0.55`, that is **13.4%** of the parent's offspring mass
beyond radius `L`. Not all of that leaks: for a *straight* boundary at distance
`L`, the fraction of the annulus at radius `r` lying outside the half-plane is
`arccos(L/r)/pi`, and integrating that against the radial density gives a leak
of **4.2%** — about 31% of `P(R > L)`, not half, because most of the mass just
beyond radius `L` is still on the inside of the line. (Numerically integrated;
redo it in five lines if you want to check.) 4% of one M6's kernel is still not
a rounding error for boundary-adjacent sequences.

Two things follow, and you should volunteer both:

- The leak applies **identically in structure** to FlowQuake's neural-ETAS head,
  which inherits `Z'_j = pi/(rho'_j d'^{rho'_j}_j)`, and to its Gaussian KDE
  background components — the precompute docstring says so in as many words
  ("slightly mass-leaking at region edges, i.e. conservative",
  [scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
  lines 6–9). So the *comparison* is not obviously biased.
- But it is not *exactly* unbiased either: the head learns `dlogd` and `drho`
  offsets and a background that concentrates on the 1.5 km smoothing scale, so
  its components are on average **sharper** than ETAS's and therefore leak
  **less**. Renormalizing both densities over the true region polygon would add
  `−log(∫_A f_s)` to each, and would add *more* to ETAS than to the head. The
  reported spatial gain is therefore, to an unquantified degree, *flattered* by
  the leak asymmetry. Nobody in this repo has measured it. Section 16 gives the
  measurement I would run.

---

## 6. The branching ratio for this parameterization

The branching ratio `n` is the expected number of *direct* offspring of a
randomly chosen event. It is the single most diagnostic scalar in an ETAS fit.

### 6.1 Expected direct offspring of one parent

Integrate the parent's triggering term over all future time and all space:

```
N(m)  =  ∫₀^∞ ∫_{R²}  k0 e^{a(m−m_c)} e^{−u/tau_tap} (u+c)^{−(1+omega)} K(r²; m)  dA du

      =  k0 e^{a(m−m_c)} · Z(m) · H(∞)
```

using the separability of §5.4. Substituting `Z(m) = (pi/(rho d^rho)) e^{−rho gamma (m − m_c)}`:

```
                pi
N(m)  =  k0 · ──────── · H(∞) · exp( (a − rho·gamma) · (m − m_c) )
              rho·d^rho
```

### 6.2 The effective productivity exponent is `a − rho·gamma`

Read that last line again. **The exponent that governs how offspring count grows
with parent magnitude is not `a`.** It is

```
a_eff  =  a − rho · gamma
```

because the *unnormalized* kernel's integral shrinks as the aftershock zone
widens. In a parameterization where the spatial kernel is pre-normalized to
integrate to 1, `a_eff = a` and the productivity exponent is read straight off.
In *this* parameterization it is not.

Numbers: with `a = 1.556` and `rho = 0.557` (both quoted from `MANUSCRIPT.md`;
§4.2's provenance warning applies) and `gamma ≈ 1.2` (literature
order-of-magnitude), `a_eff ≈ 1.556 − 0.668 = 0.888`. So the offspring count
actually grows a factor `e^{0.668} ≈ 1.95` *per magnitude unit* more slowly than
`e^{a Δm}` would suggest — and, as §13 shows, that is the difference between a
subcritical and a supercritical model. If you quote `a = 1.556` as "the
productivity exponent" in a viva you will be corrected.

This also has a direct **identifiability** consequence (§8.4): only the
combination `a − rho·gamma` is constrained by *how many* aftershocks a big event
produces. Splitting it into `a` and `rho·gamma` requires the data to speak about
*how spread out* they are. In a sparse catalog with few large events, that split
is weakly determined.

### 6.3 Averaging over the magnitude distribution

`n` averages `N(m)` over the GR density `f_m(m) = beta e^{−beta(m − m_c)}`,
`m ≥ m_c`. Substituting `v = m − m_c`:

```
n  =  ∫₀^∞  N(m_c) e^{a_eff v} · beta e^{−beta v} dv
   =  N(m_c) · beta ∫₀^∞ e^{−(beta − a_eff) v} dv
   =  N(m_c) · beta / (beta − a_eff)
```

**Convergence condition: `a_eff < beta`.** With `b = 1`, `beta = ln 10 ≈ 2.303`.
Assembling everything:

```
            k0 · pi · H(∞)         beta
n  =  ───────────────────────  ·  ─────────────        provided  a − rho·gamma < beta
            rho · d^rho            beta − a_eff
```

**If `a_eff ≥ beta`**, the integral diverges: the expected number of offspring is
infinite, because the exponential growth of productivity with magnitude
outruns the exponential decay of the magnitude distribution. Physically this
says "the rare huge events dominate the total offspring count so completely that
the mean does not exist." Computationally the likelihood is still finite (you
only ever evaluate it on observed, finite magnitudes) — which is the trap. **A
model with `a_eff ≥ beta` can be fit and can score well, and will never
terminate when you simulate from it.** Real catalogs have a maximum magnitude,
so a tapered-GR magnitude distribution removes the divergence; the classical
`a < beta` condition is a statement about the untruncated model. In practice
fitted `a_eff` sits well below `beta`.

### 6.4 Subcriticality

`n` is the mean offspring count of a Galton–Watson branching process. Standard
result (not proved here; see Harris, *The Theory of Branching Processes*, 1963,
or any branching-process text): the expected total number of descendants of one
immigrant, including itself, is

```
1 + n + n² + n³ + …  =  1 / (1 − n)     for n < 1,     divergent for n ≥ 1
```

and hence, for a stationary ETAS with background rate `mu·A`, the total event
rate is

```
R_total  =  mu·A / (1 − n)
```

which inverts to a very useful diagnostic:

```
n  =  1  −  (background rate) / (total rate)  =  1 − (background fraction)
```

**Operational ETAS must be subcritical (`n < 1`)**, for three separate reasons:

1. **Stationarity.** For `n ≥ 1` the process is explosive; there is no stationary
   rate to which the catalog's observed rate could correspond.
2. **Simulation terminates.** Every operational use of ETAS
   ([flowquake/etas_csep.py](../flowquake/etas_csep.py) included) simulates
   catalog continuations. A supercritical fit produces simulations that grow
   without bound; a fit at `n = 0.999` produces simulations whose count
   distribution has enormous variance and whose N-test is uninterpretable.
3. **Interpretability.** `1 − n` *is* the background fraction. A fit with
   `n = 0.968` — which is what [MANUSCRIPT.md](../MANUSCRIPT.md) line 364
   reports for the 2020 refit of ComCat_25 — says only 3.2% of California's
   `M ≥ 2.5` events are independent immigrants. Near-critical fits are the norm
   for California; that is a genuine, much-discussed feature of the data, not
   necessarily a fitting pathology.

Same caveat as §4.2: that 0.968 lives only in `MANUSCRIPT.md`; no artifact under
`runs/` records it (`results/CLAIMS.md` row N2).

---

## 7. Fitting: the EM / stochastic-declustering inversion

### 7.1 Why the direct likelihood is awkward

The observed-data log-likelihood is

```
log L  =  Σ_i log[ mu + Σ_{j<i} w_j(t_i − t_j) K_j(r_ij²) ]  −  Lambda(T)
```

The first term is a **log of a sum**. Every parameter appears inside every log,
so nothing separates, no update is closed-form, and the surface has long curved
valleys (the `c`–`p` and `k0`–`a` trade-offs of §8.4). Direct maximization works
but is fragile.

### 7.2 Latent parents and the complete-data likelihood

Introduce for each event `i` a latent label

```
u_i ∈ {0, 1, …, i−1},     u_i = 0  →  i is a background (immigrant) event
                          u_i = j  →  i was triggered directly by event j
```

Conditional on the labels, the branching construction of §3.1 says the process
decomposes into `N_ev + 1` **independent** inhomogeneous Poisson processes (with
`N_ev` the number of events; `n` is reserved for the branching ratio): the
background, and one offspring process per parent. Poisson likelihoods are
sums-of-logs. So

```
log L_c(theta; u)  =  Σ_i [ 1{u_i=0} log mu  +  Σ_{j<i} 1{u_i=j} log( w_j(t_i−t_j) K_j(r_ij²) ) ]
                      −  mu·A·T  −  Σ_j k0 e^{a(m_j−m_c)} Z_j H(T − t_j)
```

The log-of-a-sum is gone. That is the whole reason to do EM here: **the
branching structure is literally a latent-variable model**, and the complete-data
likelihood is a mixture likelihood with the mixture indicator being "who was your
parent".

### 7.3 E-step: responsibilities

`Q(theta | theta_old) = E_{u | data, theta_old}[ log L_c ]`. The expectation only
touches the indicators, so we need `p_ij = P(u_i = j | data, theta_old)`.

Two routes to the same answer. The **superposition route**: given that a point
occurred at `(t_i, s_i)`, and that it came from one of a set of independent
Poisson processes with intensities `nu_0 = mu`, `nu_j = w_j K_j` evaluated at
that point, the probability it came from component `k` is `nu_k / Σ nu`. The
**Jensen route**, which is the generic EM argument: for any probability vector
`q`,

```
log Σ_k a_k  ≥  Σ_k q_k log(a_k / q_k)
```

with equality iff `q_k = a_k / Σ a_k`. Applying this to the log-of-a-sum in
§7.1 and choosing the equality case gives the same weights. Either way:

```
              mu                                    w_j(t_i − t_j) · K_j(r_ij²)
p_i0 = ────────────────────      p_ij = ──────────────────────────────────────────────
        lambda(t_i, s_i)                          lambda(t_i, s_i)
```

with `lambda(t_i, s_i) = mu + Σ_{j<i} w_j K_j` and `Σ_j p_ij = 1` for each `i`.

`p_i0` is the **background probability** of event `i`. Thresholding or sampling
on it is *stochastic declustering* (Zhuang, Ogata & Vere-Jones 2002) — a
probabilistic replacement for Gardner–Knopoff windowing, and a far better one,
because it comes out of a fitted model rather than a rule of thumb.

### 7.4 M-step: what is closed-form and what is not

Maximize

```
Q  =  Σ_i p_i0 log mu  +  Σ_i Σ_{j<i} p_ij log( w_j K_j )  −  mu·A·T  −  Σ_j k0 e^{a(m_j−m_c)} Z_j H(T−t_j)
```

**Uniform background — closed form.** `∂Q/∂mu = (Σ_i p_i0)/mu − A·T = 0`, so

```
mu_hat  =  ( Σ_i p_i0 ) / (A · T)
```

"Expected number of background events divided by the space-time volume." This is
as clean as EM ever gets.

**Nonparametric background — closed form up to a smoother.** If `mu(s)` is free,
the same argument gives `mu_hat(s) ∝ Σ_i p_i0 · delta(s − s_i)`, smoothed by a
kernel. Zhuang et al. use a variable bandwidth; Helmstetter, Kagan & Jackson
(2007, *SRL* 78(1), 78–86) use a nearest-neighbour adaptive bandwidth. **This is
the single upgrade the benchmark's ETAS does not have** (its `mu` is a scalar),
and §12 shows it is where most of FlowQuake's spatial win comes from.

**`k0` — closed form given the others.** `log w_j` contains `log k0` additively,
and the compensator contains `k0` linearly:

```
k0_hat  =  ( Σ_i Σ_{j<i} p_ij )  /  ( Σ_j e^{a(m_j−m_c)} Z_j H(T − t_j) )
```

Numerator = expected number of triggered events; denominator = the model's
predicted offspring count at `k0 = 1`. Again a rate-matching identity.

**`(a, c, omega, tau_tap, d, gamma, rho)` — no closed form.** They appear inside
the logs *and* inside `Z_j` and `H`. The M-step is a numerical maximization of a
smooth 7-dimensional objective (quasi-Newton or Nelder–Mead), evaluated with the
responsibilities held fixed. The `etas` package does this; the repo's
`--refit-globals` control in
[flowquake/neural_etas.py](../flowquake/neural_etas.py) lines 68–75 does a much
smaller version of the same thing by SGD.

Iterate E and M to convergence. [MANUSCRIPT.md](../MANUSCRIPT.md) line 363
reports the ComCat 2020 refit converging in 12 EM iterations.

### 7.5 Why 3–4 CPU hours per region

[REPRODUCE.md §2](../REPRODUCE.md) budgets ~3–4 h per region.

- The E-step needs `p_ij` for every ordered pair `(i, j)` with `t_j < t_i`. With
  `N_ev ≈ 10^5` events that is `N_ev²/2 ≈ 5 × 10^9` pairs; even blocked and
  vectorized (the repo's own reimplementation blocks at `BLOCK = 128`,
  [scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
  line 33) each sweep is minutes.
- The M-step calls the objective — which requires the same `O(N_ev²)` sums — tens of
  times per iteration.
- Ten-plus EM iterations of that.

Practical implementations truncate pairs beyond some space-time distance. Doing
so changes the estimator (it is no longer exactly the ETAS MLE) and is one of the
things that makes two ETAS implementations disagree. Which brings us to a real
weakness of this repo: **`results/CLAIMS.md` row N1 records that no version,
commit hash, package name, or provenance for the `etas` implementation is
recorded in any of the 136 committed run JSONs or 90 committed YAMLs**, and
`pyproject.toml` lines 25–38 names two candidate forks without deciding between
them. If a professor asks "which ETAS did you beat?", the honest answer today is
"the one that produced these artifacts, and we have not pinned it." That is a
known open item ([WORKING.md](../WORKING.md) item 8), not a hidden one.

---

## 8. Practical pathologies a professor will probe

### 8.1 Temporal edge effects

Events before the fitting window trigger events inside it. If you start the
likelihood at `T_0` and only sum over parents with `t_j ≥ T_0`, you attribute
their offspring to the background, biasing `mu` **up** and `k0` **down**.

**Standard fix:** an *auxiliary* window `[T_aux, T_0)` whose events act as
sources but never as targets. EarthquakeNPP does exactly this — ComCat_25's
auxiliary window is 1971-01-01 → 1981-01-01
([STACK.md Part III](../STACK.md#part-iii--the-benchmark-contract)) — and the
repo's `aux_start` convention in [flowquake/data.py](../flowquake/data.py)
mirrors it. `precompute_trigger_features.py` sums over **all** priors including
the aux era (line 105's mask is `j < i`, with no lower cutoff), which is the
correct behaviour.

There is a residual bias no auxiliary window removes: events before `T_aux` still
trigger inside the window. With an Omori tail and `tau_tap` of decades, that is a
real if small term. It is why long auxiliary windows are used.

### 8.2 Spatial edge effects

Events near the boundary trigger offspring outside `A`, which are never
observed. If the model's `Z_j` counts that unobservable mass (which, as §5.5
showed, it does), then the fitted productivity is biased **down**: the model is
being asked to explain fewer observed offspring than it predicts.

**Standard fixes**, none of which this benchmark applies:

- Restrict *targets* to an inner region while allowing *sources* from a buffer
  ring — the spatial analogue of the auxiliary window.
- Replace `Z_j` with `∫_A K_j`, computed numerically per parent. Exact, but it
  destroys the closed form and the `O(1)` per-parent cost.
- Weight each event by the fraction of its aftershock zone inside `A`.

Since neither ETAS nor FlowQuake's head applies any of these here, the effect is
shared. That makes the *comparison* defensible and the *absolute* `sll` values
mildly pessimistic for both.

### 8.3 Short-term aftershock incompleteness (STAI)

For minutes to hours after a large event, the seismic network cannot resolve
small events buried in the coda. The catalog rate immediately after a mainshock
is artificially depressed. Fitting a plain Omori law to that depressed early
rate systematically **biases `c` upward** (a larger `c` flattens the model's own
early rate to match the missing data) and distorts `p` and the productivity `a`
(Seif, Mignan, Zechar, Werner & Wiemer 2017, *JGR Solid Earth* 122(1), 449–469,
"Estimating ETAS: the effects of truncation, missing data, and model
assumptions"; Hainzl 2022, *BSSA* 112(1), 494–507, the ETASI model, which builds
a detection-blindness term into the intensity).

This is why fitted `c` values (10⁻³–10⁻² days = 1.5–15 minutes) should be read as
*detection* timescales at least as much as physical ones. It also means that a
model scored on a *complete* catalog and a model scored on an incomplete one are
not comparable. The repo's own [MANUSCRIPT.md](../MANUSCRIPT.md) line 356
acknowledges the 2024 M7.0 sequence's late incompleteness and argues it "affects
both models symmetrically" — a reasonable claim that is not measured.

### 8.4 Identifiability and parameter trade-offs

Four trade-offs to be able to name:

| trade-off | why | how to break it |
|---|---|---|
| `c` vs `p` (`omega`) | over a finite observed range of `Δt`, a small `c` with steep `p` mimics a larger `c` with shallow `p` | data at very short `Δt` — precisely where STAI destroys it |
| `k0` vs `a` | `k0` sets the level at `m = m_c`, `a` the slope; with few large events the lever arm is short | a wide magnitude range with many large events |
| `a` vs `rho·gamma` | **only `a_eff = a − rho·gamma` controls the offspring count** (§6.2) | the *spatial spread* of offspring, which fixes `rho` and `gamma` separately |
| `tau_tap` vs `omega` | a long taper and a slightly steeper Omori exponent both suppress the far tail | observing well *past* `tau_tap`, where the exponential and the power law finally separate. At `tau_tap ~ 10⁴ d (≈27 yr)` a 40–50 year catalog barely reaches one taper timescale, and almost none of that observing time is post-mainshock |

The third of these is specific to this parameterization and falls straight out of
the derivation in §6. It is the kind of thing that distinguishes someone who
derived the model from someone who read about it.

### 8.5 Sensitivity to `m_c`

`m_c` enters ETAS in three places: as the truncation of the catalog, as the
reference point in `e^{a(m − m_c)}` and `d e^{gamma(m − m_c)}`, and as the GR
threshold.

- **What is invariant.** In a *correctly specified* ETAS, the background rate and
  the total rate both scale by the same GR factor `10^{−b Δm_c}` when you lower
  the threshold, so the *background fraction*, and hence the branching ratio `n`,
  is invariant to `m_c`.
- **What is not.** `k0` and `d` are defined relative to `m_c` and must rescale;
  **`k0` values are meaningless to compare across catalogs with different `m_c`**.
- **What actually happens.** Empirically parameter estimates *do* drift with the
  cutoff, and the drift is approximately exponential in `m_c` — the model is
  misspecified and the drift is the diagnostic. See Schoenberg, Chu & Veen
  (2010), *JGR Solid Earth* 115, B04309, "On the relationship between lower
  magnitude thresholds and bias in epidemic-type aftershock sequence parameter
  estimates."

The repo takes `m_c` seriously: `scripts/check_completeness.py` verifies the
choice is stable across train and test eras, because a drifting `m_c` would
manufacture a fake temporal trend
([STACK.md §4](../STACK.md#4-three-empirical-laws)). The benchmark's five
California catalogs span `m_c` from 0.6 (WHITE_06) to 2.5 (ComCat_25), and the
five non-California regions with committed ETAS baselines sit at 2.5 (Italy) and
4.0 (Japan, Chile, Greece, Iran) (`runs/multiregion_master.json`, per-region
`mc`).

### 8.6 Non-stationarity

ETAS parameters are fit once and held fixed. But `b` varies in space and after
large events; aftershock productivity varies by tectonic regime; network
detectability improves over decades. Fitting on 1981–2007 and scoring on
2007–2020 assumes all of it is stationary.

The repo tests this directly and the answer is reassuring for ETAS: refitting
ComCat_25's ETAS on data *through* 2020-01-17 rather than through 2007 improves
its forward-window NLL by only **0.0159 nats** (7.464320811779553 →
7.448446148714125; `runs/forward_etas/summary.json` and
`runs/forward_etas_ComCat_25_refit2020/summary.json`, key `nll`). Thirteen extra
years of data buys ETAS 0.016 nats/event. That is a strong statement about how
stationary this problem is over decades — and, incidentally, a strong statement
about how hard the remaining signal is to extract.

### 8.7 The background is uniform

The benchmark's ETAS has `mu(s) = mu`, a constant over ~10⁶ km² of California.
The real background is concentrated on fault traces. Every background event that
occurs on a fault is scored against a density that is flat.

This is a *choice of the benchmark's configuration*, not an intrinsic limitation
of ETAS: nonparametric background estimation has been standard since Zhuang et
al. 2002. But it is the configuration FlowQuake is measured against, and §12
shows it accounts for most of the measured spatial gain. This is the single most
important thing to be honest about when presenting FlowQuake's spatial result.

---

## 9. Known failure modes

- **Isotropy vs elongated ruptures.** Wells–Coppersmith scaling puts an M7
  rupture at roughly 40 km and an M7.5 near 80 km
  ([Chapter 2 §9](02-seismology.md)). Its aftershocks decorate that strip;
  ETAS's `K_j` spreads them in a circle centred on the hypocentre, which is
  often near one *end* of the rupture — so the kernel is both the wrong shape
  and the wrong centre.
- **No fault geometry.** ETAS has never heard of a fault; it infers spatial
  structure only from the locations of past events. This is exactly why a
  smoothed-seismicity background helps so much: it is a cheap proxy for fault
  geometry.
- **Productivity/spatial-scale coupling (§6.2).** A single parameter `gamma`
  controls how the aftershock zone grows with magnitude, and it is *entangled*
  with productivity through `a_eff`. There is no way to say "big events have many
  aftershocks in a compact zone" and "moderate events have few aftershocks over a
  wide zone" in the same fit.
- **Swarms.** Geothermal and volcanic swarms (Salton Sea is the benchmark's
  example) have no mainshock, no Omori decay from a dominant parent, and often
  a different `b`. ETAS models them by piling up many mutually-triggering small
  events, which fits the counts poorly. SaltonSea_10's ETAS `sll` of
  **−2.3151** (`runs/fullsuite_summary.json`, `SaltonSea_10.etas_sll`) — an
  equivalent uniform area of only ~10 km² — shows how spatially concentrated
  that catalog is, and how much of the "skill" there is just "it happens in the
  same tiny place".
- **Magnitude independence.** Magnitudes are i.i.d. GR draws. Reality has
  foreshocks (a sequence's largest event is often not its first), spatially
  varying `b`, and post-mainshock `b` transients (Gulia & Wiemer 2019, *Nature*,
  the "foreshock traffic light"). ETAS cannot express any of it. FlowQuake's
  `GRMagnitudeHead` makes `beta` history-dependent, which is a genuine departure
  ([STACK.md §10](../STACK.md#10-headspy--the-spatial-and-magnitude-heads)).
- **No stress state.** Coulomb stress transfer, aseismic slip, fluid diffusion —
  none of it is representable. ETAS is a phenomenological description of an event
  list, not a physical model.

---

## 10. Variants and competitors

| model | idea | where it wins |
|---|---|---|
| **Spatially varying ETAS** | fit `(mu, k0, a, …)` per cell or per region, with smoothing/regularization | heterogeneous tectonics; California's north/south differences (Nandan et al.-style inversions) |
| **Anisotropic ETAS** | elliptical `K_j` aligned to strike, or to the local seismicity's principal axis | large-mainshock sequences; near-fault forecasting |
| **Finite-fault ETAS** | replace the point source with the rupture plane / slip distribution | days after an `M ≥ 6.5` when the finite-fault model exists; requires extra data ETAS normally does not have |
| **ETASI** (Hainzl 2022, *BSSA* 112(1), 494–507) | intensity includes a detection-blindness term | early aftershock forecasting where STAI dominates |
| **flETAS / refit variants** | re-estimate global parameters on a rolling window, or add a smoothed background | mild non-stationarity; this repo's `--refit-globals` control is a cut-down SGD version |
| **Smoothed seismicity / relative intensity** (Helmstetter, Kagan & Jackson 2007, *SRL*) | no time dependence at all; kernel-smooth past epicentres to a rate map | long-term hazard; and — the uncomfortable point — it is a very strong *spatial* baseline that the benchmark's ETAS lacks |
| **STEP** (Gerstenberger et al. 2005, *Nature*) | superimposed generic + sequence-specific Omori models, with time-varying `b` | operational daily aftershock forecasting; simpler to run than full ETAS |
| **EEPAS** (Rhoades & Evison) | "every earthquake a precursor according to scale": moderate events forecast larger ones months–decades later | medium-term forecasting of moderate-to-large events; a different time-scale than ETAS |
| **Neural point processes** | learn `lambda` or `f` with a network | nothing, yet, on this benchmark — Stockman et al. 2026's headline result |

The row that should make you uncomfortable is **smoothed seismicity**. It has no
temporal component at all, and it is a strong spatial forecaster. A benchmark
whose ETAS has a *uniform* background is not testing spatial skill against the
best available spatial baseline. That is a fair criticism of the benchmark and,
by inheritance, of FlowQuake's spatial claim.

---

## 11. Why ETAS is hard to beat — argued, not asserted

Four arguments, in increasing order of how much they should worry a challenger.

**1. The functional forms are close to right.** Power law in time, power law in
space, exponential in magnitude. These are not arbitrary; they are what a century
of data says. A flexible model must spend capacity rediscovering them from a
finite sample, and will rediscover them *worse* than a century of physics did.
Any neural model that does not build them in starts several nats behind.

**2. Full-history integration.** Every ETAS evaluation sums over *all* prior
events. Omori's tail means a 2011 mainshock still contributes in 2019. The
published NPP baselines truncate history — DeepSTPP sees 20 events
([STACK.md §6](../STACK.md#6-why-etas-is-hard-to-beat)) — and 20 events at
ComCat's rate is a few days. Truncating history discards the mass in the tail.
This repository quantifies exactly how much that matters: 64% of ComCat test
events recur within 0.5 km of a prior event, and **85% of those nearest priors
lie outside a 64-event window** ([MANUSCRIPT.md](../MANUSCRIPT.md) lines
507–510). The single design decision that produces FlowQuake's spatial win is
undoing this truncation.

**3. Small effective sample size for a heavy-tailed density.** ComCat_25 has
**55,442** training events — read from `runs/mw_robustness.json` →
`california.comcat_mc25_headline.train_events`, which *is* a committed artifact.
(Use that number, not the "~70,000" of
[STACK.md §6](../STACK.md#6-why-etas-is-hard-to-beat): 70,374 is
`92,263 − 21,889`, i.e. **every event before `test_start`** — auxiliary window
plus training window plus validation window — not the training window. The two
figures are both right about different windows and are the primer's one
recurring event-count trap; see [Ch. 5 §3.2](05-sequence-models-ssm.md#32-cost).)
Those 55,442 are not 55,442
independent samples: an
aftershock sequence is statistically closer to *one* observation. The target is a
density over `(tau, x, y, m)` with power-law tails in three of the four
coordinates. Under those conditions flexibility is a liability, and the
memorization result of §4.3 of the manuscript measures it: with a learned
whole-catalog embedding of only 4 dimensions, train NLL improves to 4.14 while
held-out NLL blows up to 19.65 — worse than the Poisson floor of 13.26
(`runs/ablation_h/memorization_figure.json`).

**4. ETAS is fit on the same region it is scored on.** This is the argument most
people miss and it is the strongest one. ETAS's nine parameters are inverted on
that region's own catalog, up to the test-window start. It is not a generic model
being transferred; it is a *region-specialized* model with the correct functional
form. Beating it means beating a model that has already absorbed the region's
productivity, its decay rate, its aftershock-zone scaling, and its background
level. Any claim of the form "our model generalizes better" has to contend with
the fact that ETAS did not need to generalize — it was refit.

Corollary for FlowQuake, and the thing to say before someone else says it: the
neural-ETAS head **starts from the region's ETAS inversion** and improves it. It
does not remove the inversion. It upgrades an ETAS deployment.
[MANUSCRIPT.md](../MANUSCRIPT.md) states this in as many words, and
[REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) is the document about it.

---

## 12. How FlowQuake's neural-ETAS head generalizes ETAS

### 12.1 The formal claim

The head ([flowquake/neural_etas.py](../flowquake/neural_etas.py) lines 59–96)
computes

```
                bg(s)  +  alpha·far_num(s)  +  Σ_{j ∈ near} w'_j K'_j(s)
f_head(s | H) = ──────────────────────────────────────────────────────────
                mu'·A  +  alpha·far_den     +  Σ_{j ∈ near} w'_j Z'_j
```

with

```
w'_j  = w_j · exp(dlogw_j)                       dlogw_j = MLP_0(m_j, Δt_j)
d'_j  = d·e^{gamma(m_j−m_c)} · exp(dlogd_j)      dlogd_j = MLP_1(m_j, Δt_j)
rho'_j= rho · exp(drho_j),  clamped to [0.05, 5] drho_j  = MLP_2(m_j, Δt_j)
K'_j  = (r² + d'_j)^{−(1+rho'_j)}                Z'_j    = pi / (rho'_j · d'^{rho'_j}_j)
bg(s) = mu'·A·[ (1−g)/A + g·Σ_k softmax(kde_logits)_k · kde_k(s) ],  g = sigmoid(kde_gate)
mu'   = mu · exp(log_mu_adj),   alpha = exp(log_alpha)
```

**Strict-superset claim.** There exists a parameter setting at which
`f_head(s | H) = f_ETAS(s | H)` for every `s` and every history `H` — pointwise,
not just in expectation.

**Proof.** Set `log_mu_adj = 0` (so `mu' = mu`), `log_alpha = 0` (so
`alpha = 1`), all MLP outputs `= 0` (so `w'_j = w_j`, `d'_j = d_j`,
`rho'_j = rho`, hence `K'_j = K_j` and `Z'_j = Z_j`), and `kde_gate → −∞` (so
`g → 0` and `bg(s) → mu·A·(1/A) = mu`, a constant). Then, using the near/far
construction of
[scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
line 155 — `far_num = trig_num − near_base_num`, `far_den = trig_den − near_base_den`
— the numerator becomes

```
mu + (trig_num − near_base_num) + Σ_{j∈near} w_j K_j  =  mu + trig_num
```

because `Σ_{j∈near} w_j K_j` **is** `near_base_num` by definition, and likewise
the denominator becomes `mu·A + trig_den`. And `trig_num`/`trig_den` are the
full-history ETAS sums over *all* priors (line 105's mask is `j < i`). So

```
log f_head  =  log(mu + trig_num) − log(mu·A + trig_den)  =  SLL_ETAS
```

which is line 143's expression exactly. ∎

**Two places where "strict" is doing more work than it should**, and you should
raise both yourself:

1. **It is a limit, not an attained point.** `sigmoid(kde_gate) = 0` only as
   `kde_gate → −∞`. The verification uses `kde_gate_init = −30`, at which
   `sigmoid(−30) ≈ 9.4e−14` — utterly negligible, far below float32 resolution,
   but formally the ETAS point lies in the *closure* of the parameter set, not in
   it. The parameter set is an open superset whose closure contains ETAS.
2. **It is a superset of *one* ETAS, not of the ETAS family.** The far field is
   frozen at the published inversion's parameters and can only be *scaled* by
   `alpha`; the modulations reach only the ≤384-parent near set. You cannot
   recover ETAS-with-different-`(a, c, rho, …)` from this head. The claim "we
   strictly generalize ETAS" should always be read as "we strictly generalize
   **this fitted** ETAS."

### 12.2 Why target-location-independent selection preserves the closed form

The MLP's inputs are `[(m_j − m_c)/2, (log(Δt_j + 1e−3) − 2)/3]`
([flowquake/neural_etas.py](../flowquake/neural_etas.py) lines 61–62) — the
parent's magnitude and elapsed time, **never** the target location `s`. Neither
does near-set *selection*: parents are the top-256 by ETAS weight `w_ij` plus the
128 nearest to the **previous event's** location
([scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
lines 116–129), with `NEAR_W, NEAR_P = 256, 128` and a deduplicated cap of 384.

Consequence: at a fixed forecast time, `w'_j`, `d'_j`, `rho'_j` are **constants
with respect to `s`**. Therefore `∫_{R²} w'_j K'_j dA = w'_j Z'_j` with `Z'_j`
given by the same closed form derived in §5.1 — the derivation only needed the
kernel's shape, and the shape is unchanged. The denominator is exactly the
integral of the numerator, with no numerical quadrature anywhere.

Let the MLP see `s` and this collapses instantly: you would be modelling an
unnormalized energy `exp(−E(s))` whose partition function has no closed form.
This is not a stylistic preference; it is the difference between a density and a
score. [MANUSCRIPT.md](../MANUSCRIPT.md) lines 512–516 records that the authors
walked into exactly this trap once — an "apparent −7.9 spatial ceiling" that
turned out to be the artifact of normalizing over a *target-dependent*
neighbour set.

Same caveat as §5.5 applies: "exactly normalized" means the denominator is
exactly `∫_{R²}` of the numerator. Over the finite region `A`, both ETAS's
density and the head's integrate to slightly *less* than 1.

### 12.3 The verification gates and their tolerances

Read from the code, with the tolerances stated exactly:

| check | code | tolerance | achieved | committed? |
|---|---|---|---|---|
| numpy float64 reimplementation vs the package's stored per-event `SLL` | [scripts/etas_sll_repro.py](../scripts/etas_sll_repro.py), `ok = dabs.max() < 1e-4` | `1e−4` | `max_abs_sll_err` = **1.7655796824556091e−09** over `n_test` = 21889 | **yes**, `runs/etas_sll_repro.json` |
| frozen precomputed sums vs the package's `SLL` | [scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py) line 147, `assert err < 1e-6` | `1e−6` | printed only | no |
| **gate-closed torch head** vs the numpy `etas_sll` recomputed from the frozen sums ([train_neural_etas.py](../scripts/train_neural_etas.py) line 52) | [scripts/train_neural_etas.py](../scripts/train_neural_etas.py) line 101, `assert abs(exact_err).max() < 2e-5` | `2e−5` | printed only (line 100) | **no** |
| training-init sanity (≈5% KDE gate) | [scripts/train_neural_etas.py](../scripts/train_neural_etas.py) line 106, `assert abs(init_test − etas_test) < 0.05` | `0.05` nats | printed only | no |

**A discrepancy to flag, because a professor checking the repo will find it.**
[STACK.md](../STACK.md) lines 943–949 put the number **1.77e−9** and the artifact
`runs/etas_sll_repro.json` inside the bullet describing the *gate-closed head*
check. Those are two different checks, run by two different scripts, against two
different references:

- `runs/etas_sll_repro.json` is written by `scripts/etas_sll_repro.py`, a NumPy
  float64 reimplementation of the ETAS spatial density, compared against the
  `etas` package's own stored per-event `SLL` column. Tolerance `1e−4`; achieved
  `1.77e−9`; committed.
- the gate-closed *torch* head is compared against `etas_sll`, which
  `scripts/train_neural_etas.py` line 52 recomputes in NumPy from the **frozen
  precompute sums** (`trig_num`, `trig_den`) — not against `etas_sll_repro.py`'s
  output. Tolerance `2e−5` (set by float32 precision, not by any modelling
  approximation); achieved value printed at line 100 and discarded.

The chain is still sound — the precompute sums are themselves asserted against
the package's `SLL` to `1e−6` at `precompute_trigger_features.py` line 147 — but
"1.77e−9" is evidence for the *first* link, not the last.
[MANUSCRIPT.md](../MANUSCRIPT.md) lines 522–524 gets the attribution right (it
says "we reimplement the benchmark's exact spatial density — verified to
reproduce … (`scripts/etas_sll_repro.py`)"); `results/CLAIMS.md` row S23 labels
the same artifact "gate-closed head". All three checks pass; only the labelling
is wrong.

### 12.4 What the head actually learns, from the artifacts

From `runs/neural_etas/ComCat_25/summary_full_s0.json`
(`bg_weights[unif,kde...]`, `alpha_far`, `mu_adj`):

```
background mixture weights [uniform, 1.5 km, 6 km, 25 km, 100 km]
   = [0.12905, 0.39849, 0.24101, 0.17523, 0.05622]
alpha_far = 0.96392        mu_adj = 2.69974
```

Read that: ~87% of the background mass moves off uniform and onto smoothed
seismicity, with the **1.5 km** map carrying the largest single share; the far
triggering field is left essentially alone (`alpha ≈ 0.96`); and the total
background level is multiplied by 2.7. That last number is the interesting one —
it says ETAS's inverted uniform `mu` is *too small* once the background is
allowed to have spatial structure, which is exactly what you would expect if the
inversion had been compensating for a flat background by pushing mass into
triggering.

And the ablation is honest about attribution
(`runs/neural_etas/ComCat_25/summary_{bg_only_s0,refit_globals_s0,full_s0}.json`,
key `dS_mean`):

| configuration | `dS` vs ETAS |
|---|---|
| learned background mixture + two scalars only (`--no-mlp`) | **+0.0513** |
| + SGD refit of global kernel params (`--refit-globals`) | **+0.0564** |
| + per-parent neural modulation (default) | **+0.0600** |

**Seed caveat, because someone will ask.** All three rows are **seed 0 only**;
the repo commits three seeds for the full configuration and one seed each for the
two ablations. The full config's three seeds give `dS_mean` 0.0600 / 0.0599 /
0.0607 (`summary_full_s{0,1,2}.json`), so the headline is stable, but the *85%
attribution* rests on single-seed ablations. `results/CLAIMS.md` also notes that
the "+0.060, CI [0.051, 0.069]" pairing quoted below and in `runs/total_win.json`
combines a 3-seed mean with seed 0's bootstrap CI; no pooled-across-seeds CI is
committed.

**85% of the spatial win is the smoothed background** — structure ETAS itself
could have had (§8.7), and which every NPP in the benchmark also lacked. The
neural component adds a real but small increment. The manuscript says this
plainly; make sure you do too.

One more precision on the `--refit-globals` control that the repo does not state:
`g_off` is a 4-vector `[a, log d, rho, log c]`
([flowquake/neural_etas.py](../flowquake/neural_etas.py) line 43). So the
"classical flETAS-style refit" control refits **4 of the 9** ETAS parameters
(`k0`, `omega`, `tau_tap`, `gamma`, `mu` are not in it — `mu` moves only through
`log_mu_adj`), on the near set only, with the far field scaled by `alpha`. Its
own docstring flags the near-set limitation; the 4-of-9 limitation is
unremarked. It is therefore a **conservative lower bound** on what a full
classical refit would achieve, and the manuscript's open-items list already
notes that a full flETAS EM baseline has not been run
(`results/CLAIMS.md` row N12).

---

## 13. Worked example

Everything here is arithmetic you can redo in five lines of Python. **These
parameters are illustrative round numbers, not this repo's fitted values** (which,
per §4.2, are not committed).

```
m_c = 2.5          mu  = 5e-7 /km²/day     A   = 1e6 km²   (so mu·A = 0.5 /day)
k0  = 0.01         a   = 1.5               c   = 0.01 day
omega = 0.05  (p = 1.05)                   tau_tap = 1000 days
d   = 1.0 km²      gamma = 1.2             rho = 0.55       b = 1 → beta = 2.302585
```

**History: two past events.**

| j | `m_j` | position (km) | elapsed `Δt` (days) |
|---|---|---|---|
| 1 | 6.0 | (0, 0) | 0.5 |
| 2 | 3.5 | (30, 0) | 0.02 |

### Step 1 — triggering weights `w_j`

```
w_1: productivity  k0·e^{a(6.0−2.5)}   = 0.01 · e^{5.25}   = 1.905663
     taper         e^{−0.5/1000}                            = 0.999500
     Omori         (0.5 + 0.01)^{−1.05}                      = 2.027922
     w_1 = 1.905663 × 0.999500 × 2.027922                    = 3.862604

w_2: productivity  0.01 · e^{1.5}                            = 0.044817
     taper         e^{−0.02/1000}                            = 0.999980
     Omori         (0.02 + 0.01)^{−1.05}                      = 39.721229
     w_2 = 0.044817 × 0.999980 × 39.721229                   = 1.780146
```

Note the M6 from twelve hours ago and the M3.5 from half an hour ago carry
weights of the same order. Omori decay is steep enough that recency competes with
a 2.5-magnitude-unit productivity difference.

### Step 2 — aftershock zones and normalizers

```
d_1 = 1.0 · e^{1.2 × 3.5} = e^{4.2}  = 66.6863 km²    → r_med ≈ 1.590·√66.6863 = 12.98 km
d_2 = 1.0 · e^{1.2 × 1.0} = e^{1.2}  =  3.3201 km²    → r_med ≈ 1.590·√3.3201  =  2.90 km

Z_1 = pi / (0.55 · 66.6863^{0.55}) = 0.566979
Z_2 = pi / (0.55 ·  3.3201^{0.55}) = 2.952248
```

The larger parent has the **smaller** `Z` — §5.1's coupling, made concrete.

### Step 3 — the denominator (this is history-only; it does not depend on `s`)

```
mu·A      = 0.500000
w_1 · Z_1 = 3.862604 × 0.566979 = 2.190015
w_2 · Z_2 = 1.780146 × 2.952248 = 5.255433
────────────────────────────────────────────
DEN = lambda*(t) = 7.945449 events / day
```

Sanity check: 7.9 events/day right after an M6, against a 0.5/day background.
Background fraction at this instant is 6.3%. Reasonable.

### Step 4 — numerators and `sll` at three candidate locations

```
s_A = (5, 0)      — 5 km from the M6
s_B = (60, 40)    — 72 km from the M6, 50 km from the M3.5
s_C = (300, 200)  — far from everything
```

```
s_A:  r_1² = 25          r_2² = 625
      K_1 = (25 + 66.6863)^{−1.55}  = 9.087154e−04     w_1 K_1 = 3.510008e−03
      K_2 = (625 + 3.3201)^{−1.55}  = 4.600653e−05     w_2 K_2 = 8.189836e−05
      NUM = 5.0e−07 + 3.510008e−03 + 8.189836e−05      = 3.592406e−03
      f_s = 3.592406e−03 / 7.945449 = 4.521338e−04 /km²
      sll = log(3.592406e−03) − log(7.945449) = −7.7015

s_B:  r_1² = 5200        r_2² = 2500
      w_1 K_1 = 6.584111e−06      w_2 K_2 = 9.610707e−06
      NUM = 1.669482e−05          f_s = 2.101180e−06 /km²      sll = −13.0730

s_C:  r_1² = 130000      r_2² = 112900
      w_1 K_1 = 4.570069e−08      w_2 K_2 = 2.622771e−08
      NUM = 5.719284e−07  (of which 5.0e−07, i.e. 87%, is the background mu)
      f_s = 7.198189e−08 /km²     sll = −16.4469
```

Read the three numbers as a picture of what the model believes, all against the
uniform-over-region density `1/A = 1e−6 /km²`, i.e. `log f = −13.8155`: 5 km from
the mainshock the model puts **452×** the uniform density there
(`−7.7015 − (−13.8155) = 6.114`, `e^{6.114} = 452`); 72 km away it is only
`e^{0.7425} ≈ 2.1×` better than uniform; and 361 km away it
is **`e^{−2.631} ≈ 0.072×`, i.e. 14× worse than uniform**, because a uniform
background over a fixed region is a bad forecast for a specific far-away point
when the denominator is inflated by an ongoing sequence.

That last observation is the entire motivation for the next step.

### Step 5 — switch the background from uniform to a smoothed map

Replace `mu` (constant) with `mu·A·bg(s)` where `bg` is a *normalized* density
over the region (`∫_A bg dA = 1`). Uniform corresponds to `bg = 1/A = 1e−6`.
Suppose the smoothed-seismicity map gives

```
bg(s_A) = 5e−5 /km²   (on an active fault trace, 50× uniform)
bg(s_B) = 2e−7 /km²   (off-fault,               0.2× uniform)
bg(s_C) = 5e−5 /km²   (a different active fault, 50× uniform)
```

**The denominator does not change**, because `∫_A mu·A·bg dA = mu·A` exactly —
this is the point of using a normalized map. Only the numerator moves:

| location | uniform `sll` | smoothed `sll` | Δ |
|---|---|---|---|
| `s_A` | −7.7015 | −7.6947 | **+0.0068** |
| `s_B` | −13.0730 | −13.0973 | **−0.0243** |
| `s_C` | −16.4469 | −12.6664 | **+3.7805** |

Three lessons, and they are exactly the lessons behind §12.4's ablation:

1. **Where triggering dominates, the background is irrelevant.** At `s_A` the
   background is 0.014% of the numerator; a 50× better background moves `sll` by
   0.007 nats.
2. **A better background can *hurt* individual events.** At `s_B` the map says
   "off-fault", and it was wrong for this event. The gain is an average, not a
   guarantee — which is why FlowQuake's spatial win has a **win rate of only
   0.4972** on the test window (`runs/total_win.json`,
   `test_2007_2020.dS.win_rate`) despite a strictly positive mean of +0.060. The
   spatial gain is **tail-driven, not median-driven.** Be ready for that question.
3. **Where triggering is weak, the background is everything.** At `s_C` the gain
   is 3.78 nats — the full `log(50)` = 3.912 minus the small dilution from the
   residual triggering term. Background events far from recent sequences are
   where the smoothed map earns its keep, and they are a minority of events,
   which is precisely why the aggregate gain (+0.060) is two orders of magnitude
   smaller than the per-event gain on the events that matter.

### Step 6 — the branching ratio for these parameters

```
H(∞) = ∫₀^∞ e^{−u/1000} (u + 0.01)^{−1.05} du = 10.574367       (numeric quadrature)
pi / (rho · d^rho) = pi / 0.55 = 5.711987

N(m_c) = k0 · 5.711987 · H(∞) = 0.01 × 5.711987 × 10.574367 = 0.604006
a_eff  = a − rho·gamma = 1.5 − 0.55 × 1.2 = 0.84
beta / (beta − a_eff) = 2.302585 / 1.462585 = 1.574326

n = 0.604006 × 1.574326 = 0.950903          ← subcritical, just
```

Cross-checks:

```
N(m = 6.0) = k0 e^{1.5×3.5} Z(6.0) H(∞) = 11.4253 direct offspring
N(m = 3.5) =                            =  1.3991 direct offspring
stationary total rate  =  mu·A / (1 − n)  =  0.5 / 0.049097  =  10.18 events/day
```

**And the error you must not make.** If you forget that `Z_j` depends on `m_j`
and use `beta/(beta − a)` instead of `beta/(beta − a_eff)`:

```
n_wrong = 0.604006 × 2.302585/(2.302585 − 1.5) = 0.604006 × 2.868961 = 1.732871
```

**1.73 — supercritical.** The same parameters, read with the wrong
normalization convention, turn a perfectly sensible fit into an explosive one.
This is exactly the mistake §6.2 warns about, and it is the single best
"do you actually understand this parameterization?" question in the chapter.

---

## 14. How this shows up in FlowQuake

Do not re-read [STACK.md](../STACK.md); this section is the map from theory to
artifact, and it links rather than restates.

| theory | where |
|---|---|
| ETAS intensity, transcribed | [flowquake/neural_etas.py](../flowquake/neural_etas.py) lines 78–87; [scripts/etas_sll_repro.py](../scripts/etas_sll_repro.py) docstring |
| parameter load from `parameters_0.json` | [scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py) lines 52–60 |
| `Z_j = pi/(rho d_j^rho)` (§5.1) | `neural_etas.py` line 85; `precompute_trigger_features.py` line 83 |
| conditional spatial density (§5.3) | `precompute_trigger_features.py` line 143; assertion at line 147 |
| compensator and `H(T)` (§5.4) | [scripts/etas_forward_eval.py](../scripts/etas_forward_eval.py) lines 57–64, 116–117, anchor at 121–126 |
| ETAS as a *simulator* (§2) | [flowquake/etas_csep.py](../flowquake/etas_csep.py) — the package's `ETASSimulation`, run through the same pyCSEP harness as FlowQuake |
| the auxiliary window (§8.1) | [STACK.md Part III](../STACK.md#part-iii--the-benchmark-contract); `aux_start` in [flowquake/data.py](../flowquake/data.py) |
| smoothed background as an ETAS upgrade (§8.7, §12.4) | causal multi-scale KDE at `precompute_trigger_features.py` lines 112–114, bandwidths `KDE_BWS = [1.5, 6.0, 25.0, 100.0]` km |
| the strict-superset gates (§12.3) | [scripts/train_neural_etas.py](../scripts/train_neural_etas.py) lines 94–106; `runs/etas_sll_repro.json` |

**Grounded ETAS baseline numbers** (all read from committed artifacts, not from
`STACK.md`):

| catalog | `m_c` | ETAS `tll` | ETAS `sll` | artifact |
|---|---|---|---|---|
| ComCat_25 | 2.5 | 1.4343428344882627 | −8.689770387238827 | `runs/fullsuite_summary.json` |
| WHITE_06 | 0.6 | 2.0210970061274423 | −4.2610686365574395 | `runs/fullsuite_summary.json` |
| SanJac_10 | 1.0 | 1.1325267069430716 | −5.398118234811221 | `runs/fullsuite_summary.json` |
| SaltonSea_10 | 1.0 | 2.332039202380453 | −2.3150835316085487 | `runs/fullsuite_summary.json` |
| SCEDC_20 | 2.0 | 2.5409825345527426 | −7.534222208042888 | `runs/fullsuite_summary.json` |
| Italy | 2.5 | 1.2512631525086881 | −8.847686897538015 | `runs/multiregion_master.json` |
| Japan | 4.0 | 1.445424447397327 | −12.649036904406483 | `runs/multiregion_master.json` |
| Chile | 4.0 | 0.19844856815110745 | −12.484753955646761 | `runs/multiregion_master.json` |
| Greece | 4.0 | **−1.0096944966712087** | −10.947214851142146 | `runs/multiregion_master.json` |
| Iran | 4.0 | **−1.234184097081222** | −11.531403497921753 | `runs/multiregion_master.json` |

Greece and Iran have **negative** `tll`. That is not a bug: `tll` is
`log f_t(tau)` in log(1/day), so a negative value just means the density of the
waiting time is below 1 per day — these are sparse `m_c = 4.0` catalogs where
typical gaps are many days. It also means "ETAS gets `tll ≈ 1.4`" is a statement
about California, not about ETAS.

**A trap in the artifacts, worth knowing before someone catches you.**
`flowquake/evaluate.py` line 60 defaults `--etas-dir` to
`reference/Experiments/ETAS/output_data_ComCat_25`, and lines 104–110 write
whatever `ll_scores.json` it finds there into the run's `baselines` block. So
`runs/chile_n1/eval_test.json` carries `baselines.ETAS.nll = 7.2554275527505645`
— **ComCat's** ETAS, not Chile's — and consequently
`beats_ETAS_nll: false` in that file was evaluated against the wrong region.
Chile's actual ETAS `nll` is 12.286305387495654
(`runs/multiregion_master.json`, `Chile.ETAS.nll`), against which the same run's
`nll` of 12.527615547180176 is still a loss, so no conclusion changes — but
**never quote `baselines` out of a non-ComCat `eval_test.json`**. Use
`runs/multiregion_master.json`. The repo already knows: `results/CLAIMS.md`'s
Family-4 notes record that `runs/{greece,iran,japan,chile}_*/eval_test.json` all
carry the California inversion for exactly this reason, and that nothing in the
manuscript reads those keys. The same file also flags a separate staleness trap
in `n1_density/eval_forward.json`.

---

## 15. Common misconceptions

**1. "ETAS predicts earthquakes."**
Actually: ETAS produces a *rate* — a conditional intensity — from which you get
probabilities, never a deterministic prediction. Its entire skill is short-term
clustering: after an M6 the rate goes up by orders of magnitude for days. It has
essentially no skill at predicting the *first* event of a sequence.
**Why it matters:** every operational claim about ETAS is a claim about
probability gain over a Poisson baseline, and the honest comparison
(ComCat_25: ETAS `nll` 7.2554 vs Poisson 13.2619, `runs/eval_test_N1.json`) is
about 6 nats/event — real, but not prophecy.

**2. "`lambda(t, s)` is a probability density over space."**
Actually: it has units of events km⁻² day⁻¹ and integrates to the expected count,
not to 1. The benchmark's `sll` scores `f_s = lambda / ∫_A lambda`, a *conditional*
density given that an event occurs at that instant (§5.3).
**Why it matters:** you cannot compare `sll` to a rate, and you cannot compare
`sll` across regions with different areas (California −8.69 vs Chile −12.48).

**3. "`a` is the productivity exponent."**
Actually: in this parameterization the exponent controlling *how many* offspring
a magnitude-`m` event produces is `a − rho·gamma`, because the unnormalized
kernel's integral `Z_j` shrinks with magnitude (§6.2).
**Why it matters:** it changes the branching ratio from 0.95 to 1.73 in the
worked example — subcritical to supercritical.

**4. "`p` and `rho` are both Omori exponents."**
Actually: `p = 1 + omega` is temporal; `rho` is spatial, and the kernel decays as
`r^{−2(1+rho)}` with radial survivor `r^{−2 rho}`.
**Why it matters:** at fitted `rho ≈ 0.557`, `E[R]` is finite but `E[R²]` is not.
Anyone quoting an aftershock-zone "standard deviation" has already lost.

**5. "The exponential taper is a minor regularizer."**
Actually: it is what makes the model integrable when `omega ≤ 0`, which fitted
values sit near and can fall below (§4.4). Without it the compensator diverges.
**Why it matters:** it is also nearly unidentifiable at `tau_tap ~ 10⁴ days` from
a 40-year catalog, so it is simultaneously essential and unconstrained.

**6. "ETAS's spatial density integrates to 1 over the region."**
Actually: `Z_j` is the integral over the whole plane, so `∫_A f_s ≤ 1` — the
density leaks mass outside the study region (§5.5).
**Why it matters:** it makes both ETAS's and FlowQuake's `sll` conservative, and
it makes their *difference* subtly dependent on how sharp each model's kernels
are near boundaries. Nobody has measured this.

**7. "EM is just a numerical convenience for ETAS."**
Actually: the E-step's responsibilities `p_ij` *are* the posterior probability
that event `i` was triggered by event `j`, and `p_i0` is the background
probability. EM is not a trick; it is the model's own latent structure made
explicit, and it is the definition of stochastic declustering (§7.3).
**Why it matters:** it means every ETAS fit hands you, for free, a probabilistic
decomposition of the catalog into background and triggered events — which is what
you would use to build a nonparametric background map.

**8. "FlowQuake beats ETAS, so ETAS is obsolete."**
Actually: the spatial win comes from a neural-ETAS head that is *initialized from
each region's ETAS inversion* and, at gate-closed settings, *is* ETAS; and 85% of
the measured gain comes from replacing the uniform background with smoothed
seismicity, which is a standard ETAS upgrade the benchmark's configuration lacked
(§12.4).
**Why it matters:** the defensible claim is "an ETAS deployment can be upgraded
by ~0.06 nats/event spatially and ~0.11 total", not "ETAS is replaced."
[REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) exists because the
authors take this distinction seriously.

**9. "A higher branching ratio means a better fit."**
Actually: `n = 1 − background fraction`, so `n` is a property of the catalog as
much as of the fit. Near-critical `n` (California's ~0.97) is a real feature.
`n ≥ 1` is a broken fit that will still evaluate a finite likelihood but will
never terminate under simulation (§6.4).
**Why it matters:** it is a check you should run on any ETAS fit before trusting
a forecast from it.

---

## 16. Questions a professor will ask

**Q1. Derive the branching ratio for the parameterization in your code.**
`N(m) = k0 e^{a(m−m_c)} Z(m) H(∞)` by separability, with
`Z(m) = (pi/(rho d^rho)) e^{−rho gamma (m−m_c)}` and
`H(∞) = ∫₀^∞ e^{−u/tau_tap}(u+c)^{−(1+omega)} du`. So
`N(m) = N(m_c) e^{a_eff (m−m_c)}` with `a_eff = a − rho·gamma`. Averaging over
GR gives `n = N(m_c)·beta/(beta − a_eff)`, requiring `a_eff < beta`. The
subtlety is `a_eff ≠ a`, because the kernel is written unnormalized. §6, §13
step 6.

**Q2. What exactly does `sll = −8.69` mean?**
`log f_s` in log(1/km²), where `f_s = lambda(t,s|H) / ∫_A lambda(t,s'|H) dA` — the
density of the location conditional on an event occurring at that instant.
`e^{−8.69} ≈ 1.68e−4 /km²`, equivalent to spreading unit mass over ~5,940 km².
It is not comparable across regions: Chile's ETAS gets −12.48
(`runs/multiregion_master.json`).

**Q3. Why is the spatial integral `pi/(rho d^rho)` and not something with a
`Gamma` function?**
Because `(r²+d)^{−(1+rho)}` integrated with `dA = 2 pi r dr` becomes
`pi ∫₀^∞ (u+d)^{−(1+rho)} du` under `u = r²`, and that is an elementary power-law
integral. Full derivation in §5.1. It needs `rho > 0`.

**Q4. Your model's spatial density — does it integrate to 1 over California?**
No. It integrates to slightly less than 1, because `Z_j` (and `Z'_j`, and the
Gaussian KDE components) integrate over `R²` while the region is finite. Both
ETAS and the head are sub-normalized in exactly the same structural way, so both
`sll` values are conservative. §5.5, and the precompute docstring says so
explicitly. What "exactly normalized" means in the manuscript is that the
denominator is exactly the `R²` integral of the numerator, which is what makes
the closed form work.

**Q5. Why EM rather than direct maximum likelihood?**
The observed-data likelihood contains `log Σ_j (…)`, coupling all parameters.
Introducing latent parent labels makes the complete-data likelihood a sum of
independent Poisson likelihoods, so the log-of-a-sum disappears, `mu` and `k0`
acquire closed-form updates, and the responsibilities have a direct
interpretation as background/triggering probabilities. §7. The E-step weights
follow either from Poisson superposition or from the equality case of Jensen's
inequality.

**Q6. What is stochastic declustering and how does it relate to your E-step?**
It *is* the E-step. `p_i0 = mu / lambda(t_i, s_i)` is the posterior probability
that event `i` is an immigrant. Zhuang, Ogata & Vere-Jones (2002, *JASA* 97) use
these weights to build a nonparametric background map, iterating with the
parameter fit. §7.3, §7.4.

**Q7. Why does ETAS need an auxiliary window?**
Because events before the fitting window trigger events inside it. Omitting them
attributes their offspring to the background, biasing `mu` up and `k0` down.
EarthquakeNPP uses 1971–1981 for ComCat_25; the repo's precompute sums over all
priors including the aux era. A residual bias from pre-auxiliary events remains
and is not corrected. §8.1.

**Q8. Which ETAS implementation, at which version, produced your baselines?**
*(Hostile.)* We do not know. `results/CLAIMS.md` row N1 records that no version,
commit, package, or provenance key exists in any of the 136 committed run JSONs
or 90 committed YAMLs, and `pyproject.toml` names two candidate forks without
deciding (upstream `lmizrahi/etas` vs the EarthquakeNPP author's `ss15859/etas`
fork). It is the Mizrahi et al. `etas` family: the triggering kernel is printed
in the benchmark paper (arXiv:2410.08226 §2) and matches this repo's code term
for term, and the numeric reproduction of the package's per-event `SLL` to
1.77e−9 (`runs/etas_sll_repro.json`) confirms the *functional form and the
fitted parameters* — but neither pins the fork or its version. This is a known
open item
([WORKING.md](../WORKING.md) item 8) and it is a genuine gap: a reviewer cannot
reproduce the baseline without it. The fix is one `pip freeze` and one commit
hash recorded in the run JSON.

**Q9. Your headline spatial win is 85% "add a smoothed background". Isn't that
just fixing a strawman ETAS?**
*(Hostile.)* Partly, yes, and the manuscript says so. The benchmark's ETAS has a
uniform background, whereas nonparametric background estimation has been standard
since Zhuang et al. 2002. Three honest responses: (i) the ablation is reported —
+0.0513 background-only, +0.0564 with a classical global refit, +0.0600 full
(`runs/neural_etas/ComCat_25/summary_*_s0.json`), so nothing is hidden; (ii) the
comparison is against the incumbent as the benchmark ships it, which is also the
incumbent every prior NPP was measured against; (iii) it is still a real
limitation, and the missing experiment is a full flETAS/EM refit with a free
nonparametric background — which `results/CLAIMS.md` row N12 records as *not
run*. If I had one more month of compute that is the experiment I would run.

**Q10. You claim a "strict superset" of ETAS. Prove it, and then tell me where
the claim is weaker than it sounds.**
The proof is §12.1: set `log_mu_adj = log_alpha = 0`, all MLP outputs to zero,
and `kde_gate → −∞`; the numerator collapses to `mu + trig_num` and the
denominator to `mu·A + trig_den`, which is the benchmark's `SLL` expression
verbatim. Two weakenings: (i) ETAS lies in the *closure* of the parameter set
(`sigmoid(kde_gate) = 0` only in the limit; the test uses `kde_gate = −30`,
giving `9.4e−14`), and (ii) it is a superset of *this fitted* ETAS, not of the
ETAS family — the far field is frozen at the inversion and only scaled by
`alpha`, and modulations reach only the ≤384-parent near set.

**Q11. Why is your spatial win real if the per-event win rate is under 50%?**
*(Hostile.)* It is tail-driven, not median-driven: `dS.win_rate = 0.4972` with
`dS.mean = 0.060` and 95% block-bootstrap CI [0.051, 0.0688]
(`runs/total_win.json`, `test_2007_2020`). The mechanism is visible in §13 step 5:
a smoothed background *loses* small amounts on the many trigger-dominated events
and *wins* large amounts on the minority of background events far from recent
sequences. That is a legitimate way to gain expected log-score, but it means the
model is not uniformly better, and a decision-maker who cares about the median
event would not see an improvement. Any claim should say "mean log-score", and
the repo's own artifacts do report the win rate.

**Q12. Two events at the same instant, one M2.5 and one M7. Does ETAS place them
differently in space?**
No. `f_s` does not depend on the magnitude of the event being placed, because
magnitudes are i.i.d. GR draws independent of location (§3.1 step 3). This is
false in reality — large events nucleate on mature faults, small ones do not —
and it is a structural limitation of the factorization, shared by FlowQuake.

**Q13. What breaks first if you halve `m_c`?**
Halving `m_c` (say 2.5 → 1.25) multiplies the event count by `10^{b·Δm_c}`, i.e.
about 18× at `b = 1`; `k0` and the reference of `d` must rescale
because they are defined at `m = m_c`; STAI worsens (more small events are lost
after mainshocks); and the `O(N_ev²)` E-step cost grows quadratically. In a
correctly specified ETAS the *background fraction* — and hence `n` — is invariant,
because background and triggered rates scale by the same GR factor. Empirically
estimates drift roughly exponentially in the cutoff (Schoenberg, Chu & Veen 2010,
*JGR* 115, B04309), which is a misspecification diagnostic. §8.5.

**Q14. Fitted `omega` can be negative. Is that physical?**
`p = 1 + omega < 1` means the untapered aftershock count diverges — unphysical as
a standalone statement. In this parameterization the exponential taper
`e^{−Δt/tau_tap}` makes the integral finite regardless, so the fit is
well-defined. The right reading is that `omega` and `tau_tap` jointly describe
the tail and are individually weakly identified from a catalog shorter than
`tau_tap`. §4.3, §4.4, §8.4.

**Q15. Your `--refit-globals` control is supposed to answer "is the gain just a
better ETAS fit?". Does it?**
*(Hostile.)* Not fully. It refits **4 of 9** parameters (`a`, `log d`, `rho`,
`log c` — [flowquake/neural_etas.py](../flowquake/neural_etas.py) line 43),
by SGD, on the near set only, with the far field scaled by a single scalar
`alpha`. `k0`, `omega`, `tau_tap`, `gamma` are not refit. It is therefore a
conservative lower bound (+0.0564 of the +0.0600 full gain) and not a proper
control. The proper control is a full EM refit with a free background, which
`results/CLAIMS.md` row N12 records as not run. I would state the result as "at
least 94% of our spatial gain survives a partial classical refit" and not claim
more.

**Q16. Why can't the neural modulation see the target location?**
Because then `w'_j`, `d'_j`, `rho'_j` would be functions of `s`, and
`∫ w'_j(s) K'_j(s) dA` would no longer be `w'_j Z'_j` — you would need a
partition function with no closed form. The near set is selected by ETAS weight
and by the *previous* event's location, both `s`-independent
([scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
lines 116–129). §12.2. The manuscript records that the authors hit exactly this
trap once, producing an illusory "−7.9 spatial ceiling."

**Q17. What is the compensator, and where does it appear in your code?**
`Lambda(T) = mu·A·T + Σ_j k0 e^{a(m_j−m_c)} Z_j H(T − t_j)`, where
`H(T) = ∫₀^T e^{−u/tau_tap}(u+c)^{−(1+omega)} du`. It appears at
[scripts/etas_forward_eval.py](../scripts/etas_forward_eval.py) lines 116–117,
with `H` built by cumulative quadrature on a log mesh at lines 57–64 and the
window-start anchor at 121–126. FlowQuake's *own* temporal head never computes a
compensator — it models `f(tau | H)` directly with a normalizing flow, which is
why its temporal likelihood is exact rather than quadrature-limited
([Chapter 1](01-point-processes.md), [STACK.md §2](../STACK.md#2-the-likelihood-and-the-choice-that-shapes-this-repo)).

**Q18. If ETAS is so good, why does it need refitting per region, and what does
that imply about your comparison?**
It implies the comparison is *hard*, in ETAS's favour: ETAS's nine parameters are
inverted on the very region and era it is scored on, so it is a
region-specialized model with the right functional form. §11 argument 4. It also
implies that FlowQuake's cross-region transfer results are the interesting ones,
because they test something ETAS does not do at all. But it is also why the head
being ETAS-initialized is a real caveat: FlowQuake inherits the region-specific
inversion rather than removing the need for it.

**Q19. How would you measure the boundary-leak asymmetry you raised in §5.5?**
Pick a stratified sample of test events (say 2,000, stratified by distance to the
region boundary). For each, evaluate both `f_ETAS` and `f_head` on a fine grid
over the region polygon, integrate numerically to get `Q_ETAS` and `Q_head`, and
compare the renormalized scores `sll + (−log Q)`. If `−log Q_ETAS > −log Q_head`
systematically, the reported `dS` is inflated by the difference. The
infrastructure exists —
[flowquake/neural_etas_forecast.py](../flowquake/neural_etas_forecast.py)
already evaluates the head as a field over the CSEP grid in one vectorized pass
— so this is an afternoon's work, and it has not been done.

**Q20. Give me one number from this repository that you think is
over-claimed.**
*(Hostile — answer it, do not dodge.)* Several are documented in
`results/CLAIMS.md`, which the repo maintains as an adversarial check on its own
manuscript: 11 `MISMATCH` rows covering 8 distinct contradictions and 13
`NO ARTIFACT` rows. The ones that touch this chapter: (i) row **N2** — the ETAS
parameter values and the branching ratio 0.968 appear only in `MANUSCRIPT.md`
and are backed by no committed artifact; (ii) row **N4** — the ETAS *temporal*
reproduction to ~1e−5/event is claimed but no artifact records it, while the
*spatial* reproduction is properly committed; (iii) `STACK.md` Part V attaches
`runs/etas_sll_repro.json` (a NumPy reimplementation check) to the gate-closed
*torch head* assertion, which is a different check with a different tolerance
(§12.3). None of these change a headline result; all of them are the kind of
thing a careful reviewer finds, and it is better to have found them first.

---

## 17. Further reading

1. **Ogata, Y. (1988).** *Statistical models for earthquake occurrences and
   residual analysis for point processes.* JASA 83(401), 9–27. — The temporal
   ETAS paper, and the random-time-change residual diagnostic. If you read one
   thing, read this.
2. **Ogata, Y. (1998).** *Space–time point-process models for earthquake
   occurrences.* Ann. Inst. Statist. Math. 50(2), 379–402. — The space–time
   extension; where the magnitude-scaled spatial kernel comes from.
3. **Zhuang, J., Ogata, Y., Vere-Jones, D. (2002).** *Stochastic declustering of
   space-time earthquake occurrences.* JASA 97(458), 369–380. — The EM /
   background-probability view, and nonparametric background estimation. This is
   the paper that makes §8.7 uncomfortable for the benchmark.
4. **Veen, A., Schoenberg, F. P. (2008).** *Estimation of space-time branching
   process models in seismology using an EM-type algorithm.* JASA 103(482),
   614–624. — The EM derivation written out as an estimation method.
5. **Daley, D. J., Vere-Jones, D.** *An Introduction to the Theory of Point
   Processes*, Vols I–II. — The reference for conditional intensity, compensators,
   the martingale machinery, and simulation by thinning. Chapters 7 and 14 are
   the relevant ones.
6. **Mizrahi, L., Nandan, S., Wiemer, S. (2021).** *Embracing data incompleteness
   for better earthquake forecasting.* JGR Solid Earth 126, e2021JB022379
   (doi:10.1029/2021JB022379). — The methods paper behind the `etas` package
   (`github.com/lmizrahi/etas`) that produced every baseline in this repository.
   Read it for the EM machinery of §7; §4's functional form is pinned by the
   EarthquakeNPP paper below, but *which fork* was installed is still open (§7.5).
7. **Stockman, S., Lawson, D. J., Werner, M. J. (2026).** *EarthquakeNPP: a
   benchmark for earthquake forecasting with neural point processes.* TMLR,
   arXiv:2410.08226. — The benchmark, its splits, and the result that five NPPs
   all lose to ETAS. The rules of the game FlowQuake plays.
8. **Helmstetter, A., Kagan, Y. Y., Jackson, D. D. (2007).** *High-resolution
   time-independent grid-based forecast for M ≥ 5 earthquakes in California.*
   SRL 78(1), 78–86. — Adaptive-bandwidth smoothed seismicity: the strong
   spatial baseline the benchmark's ETAS lacks, and the estimator
   [flowquake/data.py](../flowquake/data.py)'s `adaptive_bg_grid` implements.
9. **Seif, S., Mignan, A., Zechar, J. D., Werner, M. J., Wiemer, S. (2017).**
   *Estimating ETAS: the effects of truncation, missing data, and model
   assumptions.* JGR Solid Earth 122(1), 449–469 (doi:10.1002/2016JB012809). —
   Systematic study of how the pathologies of §8 bias the inversion.
10. **Schoenberg, F. P., Chu, A., Veen, A. (2010).** *On the relationship between
    lower magnitude thresholds and bias in epidemic-type aftershock sequence
    parameter estimates.* JGR Solid Earth 115, B04309. — The `m_c`-sensitivity
    result of §8.5. Pair it with Schorlemmer et al. (2007), *SRL* 78(1), 17–29,
    for CSEP's N/S/M consistency tests, which is the framework in which "ETAS is
    the incumbent" is an institutional fact.
