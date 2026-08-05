# Seismology for the point-process modeller

You know probability. You do not know rocks. This chapter closes that gap far enough that
you can argue with a seismologist about what a catalog *is*, why its statistics look the
way they do, and which of FlowQuake's numbers measure physics versus measuring the seismic
network. Everything here is upstream of the modelling: [STACK.md](../STACK.md) walks the code,
this chapter is the domain knowledge that walkthrough assumes.

## What this chapter buys you

- Define seismic moment and moment magnitude from first principles, with the right units
  convention, and explain *why* ML/mb/Ms saturate and Mw does not.
- Derive the Aki b-value MLE, its standard error, an exact F-test for two b-values, and the
  Utsu/Bender binning correction — then say what the `+0.005` in
  [flowquake/heads.py](../flowquake/heads.py) corrects and why the constant cannot be right for
  every catalog in this repo.
- Answer "why a power law and not an exponential?" two ways: Dieterich's rate-and-state
  seismicity equation, and superposition of exponentials over a scale-free population.
- Explain completeness magnitude m_c, and exactly how a drifting m_c manufactures a fake
  temporal trend a flexible model will happily learn and score well on.
- Attack or defend FlowQuake's cross-regime claim on seismological grounds, and know the
  five places where the repo's seismological hygiene is thin before a professor finds them.

## Prerequisites

- The point-process chapter ([docs/01-point-processes.md](01-point-processes.md)):
  lambda(t | H_t), the compensator, the likelihood. This chapter uses that vocabulary and
  derives no point-process theory.
- [STACK.md](../STACK.md) Part 0 only, for the two-model orientation (production TPP vs
  neural-ETAS spatial head).
- The ETAS chapter ([docs/03-etas.md](03-etas.md)) is *downstream*. Read this first.

Shared notation applies, with two collisions to disarm up front:

- The repo's ETAS parameter set is `(mu, k0, a, c, omega, tau, d, gamma, rho)` and its `tau`
  is the **Omori exponential taper timescale**, colliding with `tau` for the inter-event gap.
  Here the taper is **tau_tap**; a bare `tau` is always a waiting time.
- `rho` is used for two unrelated things in this codebase. In `neural_etas.py` it is the
  ETAS **spatial power-law decay exponent** (`(r² + d_j)^{-(1+rho)}`); in
  `heads.py`'s `KernelMixtureHead` it is the **aspect ratio** of an elliptical component
  (`rho ≥ 1`, axes `d*rho` and `d/rho`). Both usages appear below, always labelled.

---

## 1. What an earthquake physically is

### 1.1 Elastic rebound, stick–slip, and rupture

Plate motion loads the lithosphere at cm/yr. Faults inside it are frictionally locked, so the
imposed displacement accumulates as **elastic strain energy** rather than as steady slip.
When shear stress reaches frictional strength the patch slips, the rock springs back, and the
stored energy goes to frictional heat, fracture-surface energy and radiated waves. That is
**elastic rebound** (Reid 1910, Carnegie Institution report on the 1906 San Francisco
earthquake). Two consequences to state instantly:

1. **An earthquake is a stress-drop event.** It relieves stress on the ruptured patch and
   *raises* it elsewhere — at the rupture tips and in off-fault lobes. That transfer is the
   physical basis of aftershock triggering (§9).
2. **Loading is slow, release is fast** — decades-to-millennia versus seconds, nine to
   twelve orders of magnitude. Catalog inter-event gaps inherit that range, which is why
   [flowquake/data.py](../flowquake/data.py) works in `log tau` with `TAU_FLOOR_DAYS = 1e-7`.

Mature faults already have a fracture surface, so an earthquake is mostly **stick–slip
frictional instability**, not fresh fracture (Brace & Byerlee 1966, *Science*). Instability
requires the fault to weaken with slip faster than the elastic surroundings unload it: with
spring stiffness `k` and slip-weakening rate `W = -dtau/ddelta`, you need `W > k`. The modern
constitutive framework is **rate-and-state friction** (Dieterich, Ruina, ~1979–83), and §6.3
shows it *predicts* Omori decay — the most useful physics fact in this chapter.

Nor is a large earthquake a point: it nucleates in a small patch, a rupture front propagates
at 0.7–0.9 times the crustal shear-wave speed (`beta_s ≈ 3.5 km/s`, so ~2.5–3 km/s), slip
occurs behind the front and heals. "An
M7.1 at 35.77 N, 117.60 W" is really a 50–100 km long, 10–15 km deep surface slipping 1–3 m
over ~30 s. So **the point abstraction degrades with magnitude** — an M7 placed at its
*hypocentre* (the nucleation point, not the slip centroid) misplaces most of its aftershocks
by tens of km, which is why both models widen the spatial kernel with parent magnitude
(`d_j = d*exp(gamma*(m_j - m_c))` in ETAS, a learned per-component `d` in
[flowquake/heads.py](../flowquake/heads.py)) — and **aftershocks decorate the rupture surface**,
so their cloud is elongated along strike, which is the physical justification for
`(rho, theta)` in `KernelMixtureHead` (§9).

### 1.2 Seismic moment

```
M0 = mu_shear * A * D
```

| symbol | meaning | typical |
|---|---|---|
| `mu_shear` | shear modulus (rigidity) of the source rock | 30 GPa crust, 60–70 GPa mantle |
| `A` | ruptured fault area | 1 km² (M4) → 1e5 km² (M9) |
| `D` | average slip over `A` | mm (M3) → tens of m (M9) |
| `M0` | seismic moment | N·m (SI) or dyne·cm (cgs); 1 N·m = 1e7 dyne·cm |

`M0` is not an index: it is the amplitude of the equivalent **double-couple** force system
in the elastodynamic representation theorem — the strength of the body-force distribution
producing the same far-field radiation as the slip discontinuity. That is why it is the
right size measure and why it is recoverable from long-period waveforms without knowing `A`
and `D` separately.

Sanity check you should be able to do live: M7 rupture, `A = 60 x 15 km = 9e8 m²`,
`D = 1 m`, `mu_shear = 3e10 Pa` gives `M0 = 2.7e19 N·m`.

### 1.3 Moment magnitude, and the units trap

Magnitudes are logarithmic in amplitude. **Moment magnitude** (Kanamori 1977; Hanks &
Kanamori 1979, *JGR* 84, 2348–2350) is built to agree with ML where ML works and Ms where
Ms works. Hanks & Kanamori wrote it in cgs:

```
Mw = (2/3)*log10(M0) - 10.7                  [M0 in dyne·cm]      (HK79)
```

The IASPEI standard is SI:

```
Mw = (2/3)*( log10(M0) - 9.1 )               [M0 in N·m]          (IASPEI)
   = (2/3)*log10(M0) - 6.0667
```

**These are not the same equation.** With `log10 M0[dyne·cm] = log10 M0[N·m] + 7`,

```
HK79 in SI = (2/3)*(log10 M0[N·m] + 7) - 10.7 = (2/3)*log10 M0[N·m] - 6.0333
           = (2/3)*( log10 M0[N·m] - 9.05 )
```

so HK79 is the "−9.05" convention and IASPEI the "−9.1" convention, differing by
`(2/3)*0.05 = 0.033` magnitude units. Trivial for most purposes; **not** trivial when
reproducing a published Mw to two decimals, and not trivial inside a productivity term
`exp(a*(m - m_c))` with `a ≈ 2.3`, where 0.033 units is an 8% productivity shift.
**This chapter uses IASPEI** and says so at every computed number. Applying it to the M7
example: `log10(2.7e19) = 19.4314`, `Mw = (2/3)(19.4314 - 9.1) = 6.888`. Under HK79, 6.92.

**Why 2/3?** Radiated energy scales roughly linearly with moment
(`E ≈ M0 * Delta_sigma / (2 mu_shear)`, and stress drop is roughly magnitude-independent),
while the classical energy–magnitude relation is `log10 E = 1.5 M + const`. Inverting gives
`M = (2/3) log10 M0 + const`. The 2/3 is inherited from the historical 1.5, not new physics.
Memorise: **one magnitude unit = 10^1.5 ≈ 31.6× in moment; two units ≈ 1000×.**

---

## 2. Magnitude scales, saturation, and why mixing them is dangerous

| scale | measures | usable band | form |
|---|---|---|---|
| **ML** (Richter 1935) | peak amplitude on a Wood–Anderson short-period seismogram, distance-corrected | 2 ≲ M ≲ 6.5 | `log10 A - log10 A0(Delta)` |
| **mb** | teleseismic short-period P amplitude (~1 s) | 4 ≲ M ≲ 6 | `log10(A/T) + Q(Delta,h)` |
| **Ms** | ~20 s surface (Rayleigh) wave amplitude | 5 ≲ M ≲ 8 | `log10(A/T) + 1.66 log10 Delta + 3.3` |
| **Md** | coda duration above a noise threshold | 1 ≲ M ≲ 4, network-specific | `a + b log10(dur) + c*Delta`, calibrated to local ML |
| **Mw** | seismic moment from waveform inversion | all; no saturation | `(2/3)(log10 M0 - 9.1)` |

### 2.1 Saturation and why it happens

**Saturation** is the failure of an amplitude magnitude to keep growing once the source is
"bigger than" the measurement band. The mechanism is the source spectrum: to first order the
far-field displacement spectrum is the **omega-squared (Brune) model**
`|U(f)| ∝ M0 / (1 + (f/f_c)²)` — flat at a level ∝ M0 below the **corner frequency** `f_c`,
falling as `f^-2` above it, with `f_c ≈ k*beta_s/L` for rupture length `L`. Under constant
stress drop `M0 ∝ L³`, so `f_c ∝ M0^(-1/3)`. A scale measuring at fixed `f_meas` reads the
spectrum there:

- `f_meas < f_c` (small event): flat part, amplitude ∝ M0, the scale tracks moment.
- `f_meas > f_c` (large event): `|U(f_meas)| ∝ M0*f_c²/f_meas² ∝ M0*M0^(-2/3) = M0^(1/3)`.
  Amplitude grows as the **cube root** of moment, so the magnitude compresses; band
  limitation then flattens it entirely.

With `f_c ≈ 1 Hz` near M5: ML (1–10 Hz) saturates near **6.5**, mb (~1 Hz) near **6–6.5**,
Ms (0.05 Hz) near **8–8.5**, Mw (f → 0) never. The 1960 Chile earthquake is `Mw 9.5` but was
assigned `Ms ≈ 8.5`; the 2004 Sumatra earthquake is `Mw 9.1` with `Ms ≈ 8.8`. Both surface-wave
values sit an order of magnitude in moment below the truth. That is saturation, and it is why
Mw exists.

### 2.2 Three ways mixed scales bite, quantitatively

**(a) An offset corrupts productivity.** ETAS productivity is `k0*exp(a(m - m_c))` with
`a ≈ 2.0–2.3`. A systematic `delta_m = +0.2` in one era multiplies every event's inferred
productivity by `exp(2.3*0.2) = 1.58` — a 58% error, which a global EM fit splits across
both eras.

**(b) A scale change rescales b and a.** If `Mw = s*M_rep + o`, then

```
N(≥ Mw) = 10^(A - b_w*Mw) = 10^(A' - (b_w*s)*M_rep)     =>     b_rep = s * b_w
```

The repo's Italian conversion table is a single line —
[scripts/build_italy_mw.py:49](../scripts/build_italy_mw.py#L49), citing Gasperini, Lolli &
Vannucci (2013, *BSSA* 103(4), 2227–2246) — and it has **two** branches:

```python
_REL = {"ML": (1.0, 0.08), "MD": (1.456, -1.472)}   # (slope, intercept), Mw = slope*M + intercept
```

So ML is only *shifted* (`s = 1`, `b` unchanged) while Md is genuinely *stretched*
(`s = 1.456`). An Md-reported catalog with `b_rep = 1.0` therefore corresponds to
`b_w = 1/1.456 = 0.687`; an ML-reported one is unaffected. That asymmetry is the whole story
of §2.4: only the Md portion of the catalog is deformed.

**(c) A mid-catalog change of magnitude *type* is a step discontinuity in the marks.**
[scripts/build_italy_mw.py:42](../scripts/build_italy_mw.py#L42) records that INGV flips
Md → ML around **2005-04-16**. Under `Mw = 1.456*Md - 1.472` the Md branch's fixed point is
`Md = 1.472/0.456 = 3.228`: below it the conversion *shrinks* magnitudes, above it stretches
them; the ML branch just adds 0.08 everywhere. So the same physical event of a given size gets
a systematically different converted number before and after the flip — a jump of
`(1.456 m - 1.472) - (m + 0.08) = 0.456 m - 1.552`, i.e. −0.41 units at m = 2.5 and +0.04 at
m = 3.5. That is §4.3's fake-temporal-structure pathology, in the mark channel.

### 2.3 The repo's exposure

[REPRODUCE.md](../REPRODUCE.md) says it in its own caveats:

> Magnitudes are agency-preferred (ISC/INGV); types mixed (documented in
> `<name>_meta.json`). At the analysis mc, b ≈ 0.9–1.0; an Mw-consistent robustness check
> is a supplement item.

Three points to concede rather than defend:

1. **Mixed types are real.** [scripts/build_region.py:75](../scripts/build_region.py#L75)
   parses the ISC FDSN text schema and takes field 10 (`Mag`) with field 9 (`MagType`) —
   whatever ISC prefers for that event. Over Japan 1990–2020 that is a mixture of mb, Ms,
   Mw(GCMT) and contributed ML. `magtype` is preserved; **no homogenisation is applied to
   any of the five ISC/INGV regions used in the headline results.**
2. **"b ≈ 0.9–1.0" is not backed by a committed artifact at the analysis m_c.** The only
   committed b-values ([runs/completeness.json](../runs/completeness.json)) are at each era's
   *maximum-curvature* m_c, not at `mcut = 4.0`: Japan 0.79/0.85, Chile 0.94/0.76, Greece
   1.13/1.00, Iran 0.89/0.95 (train/test) — a range of 0.76–1.13. The `b_value` in
   `<name>_meta.json` ([scripts/build_region.py:183](../scripts/build_region.py#L183)) is also
   at MAXC m_c, and `reference/` is not committed ([README.md](../README.md)), so nothing here
   lets you check b at m_c = 4.0. The honest answer is "not reported; here is the four-line
   Aki MLE that would compute it."
3. **The Mw robustness test was run on Italy and FlowQuake lost badly.** From
   [runs/mw_robustness.json](../runs/mw_robustness.json):

   | catalog | mc | train events | test n | FQ tll | ETAS tll | dT | decision |
   |---|---|---|---|---|---|---|---|
   | native ML | 2.5 | 19,430 | 10,908 | 1.3225 | 1.2513 | **+0.0712** | win |
   | native ML, density control | 2.8 | 9,167 | 5,346 | 0.6769 | 0.6747 | +0.0022 | tie |
   | Mw-homogenised | 2.6 | 10,391 | 8,525 | 1.0371 | 1.2903 | **−0.2532** | loss |

### 2.4 A real internal inconsistency, stated plainly

The docstring of [scripts/mw_robustness.py](../scripts/mw_robustness.py) and the
`interpretation` field of [runs/mw_robustness.json](../runs/mw_robustness.json) both call the
Italy erosion "a density effect (a density-matched ML control at mc 2.8 also only ties),
not a magnitude-scale artifact." **The artifact's own numbers do not support that.** The
density control has 9,167 training events and ties (+0.0022); the Mw run has **10,391** —
*more* — and loses by **−0.2532**. And ETAS's own `tll` went **up** on the Mw catalog
(1.2513 → 1.2903) while FlowQuake's fell (1.3225 → 1.0371). Thinning cannot explain a run
with more data doing 0.255 nats worse.

A plausible mechanism, offered as hypothesis not finding: a homogeneous Mw scale is exactly
what ETAS's single-exponent productivity law assumes, while FlowQuake's relational features
consume raw magnitudes and are disrupted by the type-dependent stretch and the 2005 flip.

**Credit where due:** [MANUSCRIPT.md](../MANUSCRIPT.md) §4.5 gets this right — the further drop
"reflects a residual sensitivity to the heavy, type-dependent Md→Mw compression itself (the
stretch distorts the magnitude features the neural heads consume), which ETAS — re-fitting a
single productivity exponent — absorbs." The script docstring and the JSON `interpretation`
string are the stale ones. Saying that shows you read both.

---

## 3. How a catalog is made

A catalog is not data; it is the output of a five-stage inference pipeline, and every stage
leaves fingerprints.

**Detection.** Classically **STA/LTA**: trigger when a short-term envelope average (0.5–2 s)
exceeds a long-term average (10–60 s) by a threshold. Modern pipelines add **template
matching** (cross-correlate the continuous record against known event waveforms; pulls
events out of noise and lowers m_c by ~1 unit) and deep-learning pickers. The QTM catalog
behind `SanJac_10` and `SaltonSea_10` is exactly this: Ross, Trugman, Hauksson & Shearer
(2019), "Searching for hidden earthquakes in Southern California", *Science* 364, 767–771 —
1.81M events for 2008–2017 where the standard SCSN catalog has ~180k.
*Fingerprint:* every detector upgrade is a step change in m_c at a specific date, and
template-matching catalogs are supersets, not homogeneous extensions, of the parent catalog
(their magnitudes come from amplitude ratios relative to templates, a noisier estimator).

**Picking and association.** Pick P and S arrival times per station; then decide which picks
belong to the same event. Association is combinatorially hard exactly when rate is high —
i.e. during aftershock sequences — and fails by **splitting** one earthquake into two or
**merging** two into a mislocated average.

**Location inversion, and why depth is worst.** Solve for hypocentre `s = (x,y,z)` and origin
time `t0` from `t_obs^k = t0 + T(s, r_k; v)`. Geiger's method linearises; station `k`
contributes the row `[1, dT/dx, dT/dy, dT/dz]`, and those derivatives are the components of
the source-side slowness vector: with take-off angle `i_k` from the downward vertical and
azimuth `phi_k`,

```
dT/dz = -cos(i_k)/v(s) ;   dT/dx = -sin(i_k) cos(phi_k)/v(s) ;   dT/dy = -sin(i_k) sin(phi_k)/v(s)
```

When every station is far compared with the focal depth — the usual case — rays leave nearly
horizontally, so `cos(i_k)` is small **and nearly the same for all k**. The depth column is
then approximately a constant multiple of the all-ones origin-time column: near-collinear,
ill-conditioned normal matrix, and **depth trades off against origin time**. Move the source
5 km deeper, shift `t0` by ~1 s, and the arrivals fit almost as well. Cures: a station within
roughly one focal depth of the epicentre (so `i_k` varies), depth phases (`pP - P`) for
teleseisms, or a 3-D velocity model. *Fingerprint:* depth histograms spike at 5, 10, 33 km —
values agencies **fix** when the inversion will not resolve depth. Depth error also bleeds
into the horizontal solution through the velocity model, contaminating the coordinates this
repo *does* use.

**And FlowQuake never sees depth.**
[scripts/build_region.py:75](../scripts/build_region.py#L75) parses ISC fields
`(0,1,2,3,10,9,13)` = id, time, lat, lon, mag, magtype, eventtype — field 4 is `Depth` and
is **discarded**. Fine for California (seismicity ≲ 20 km deep). A genuine approximation for
the Japan box (22–46 N, 122–150 E, [REPRODUCE.md](../REPRODUCE.md) §1), which contains the
whole Japan–Kuril–Izu subduction system down to ~600 km, all projected onto one plane: a
slab event at 400 km and a crustal event above it are, to this model, in the same place.

**Uncertainties** (formal ones are optimistic, since they assume the velocity model):

| quantity | dense local network | regional | teleseismic |
|---|---|---|---|
| horizontal | 0.1–1 km | 2–10 km | 10–30 km |
| depth | 0.5–2 km | 5–20 km | 10–40 km |
| origin time | < 0.1 s | 0.2–1 s | 1–3 s |
| magnitude | ±0.1–0.2 | ±0.2–0.3 | ±0.2–0.3 |

Relative relocation (double-difference, waveform cross-correlation) reduces *relative* error
to tens of metres while leaving *absolute* error unchanged. The `WHITE_06` catalog is such a
product: White, Ben-Zion & Vernon (2019), *JGR Solid Earth* 124, 6908–6930 — 108,800 events,
M −1.8 to 5.4, probabilistic 3-D location plus cross-correlation relative locations.

**Network evolution** shows up as: more stations → lower m_c → a step in apparent rate at
fixed threshold; a relocation campaign → seismicity collapsing onto fault traces with no
change in physics; a new magnitude procedure → a step in the marks; a station outage → a
local temporary rise in m_c. Because FlowQuake's temporal head models the density of
`log tau` conditioned on lagged log-gaps at seven exponentially spaced lags
([flowquake/data.py:26](../flowquake/data.py#L26), `RECENCY_LAGS = (1,2,4,8,16,32,64)`), **a
monotone completeness trend is directly visible to the model as a monotone trend in gap
statistics**. §4.3 makes that quantitative.

**Non-tectonic contamination.** Quarry and mine blasts (diurnal, weekday-daytime, shallow);
mining-induced events; **geothermal seismicity** (the Geysers and the Salton Sea field are
among California's most productive patches, largely driven by fluid injection and
extraction); injection-induced seismicity; occasionally nuclear tests and landslides. The
repo filters what it can: the blocklist
`("explosion","blast","quarry","mine","mining","nuclear","rockburst","collapse")` is defined
at [scripts/build_region.py:45](../scripts/build_region.py#L45) and applied to the FDSN
`eventtype` column at [scripts/build_region.py:160](../scripts/build_region.py#L160);
the ComCat query carries `&eventtype=earthquake`
([scripts/build_comcat_forward.py:52-55](../scripts/build_comcat_forward.py#L52-L55)).
**This removes only what the agency already labelled.** Misclassified blasts and all
geothermal/induced seismicity — which agencies label `earthquake` because it is
indistinguishable at catalog level — remain. Both the Geysers and Salton Sea lie inside the
RELM polygon defining `ComCat_25`, and [MANUSCRIPT.md](../MANUSCRIPT.md) §4.4 names "Geysers,
Ridgecrest, Salton Sea" as the dense zones driving its spatial analysis. A nontrivial share
of the density FlowQuake exploits is industrial.

**One artifact the pipeline creates itself.** The benchmark's ComCat recipe, replicated at
[scripts/build_comcat_forward.py:121-138](../scripts/build_comcat_forward.py#L121-L138),
**jitters duplicate locations and times**: `sj, tj = 0.005, 0.1/86400.0` — 0.005 degrees
(≈ 0.55 km) and 0.1 s. A continuous density cannot be evaluated at a tie, so ties are broken
by hand. That jitter is the same order as the sub-kilometre clustering the density-adaptive
head is built to capture (`d_floor_km: 0.1` in every production config), which puts a floor
on how much sub-km structure can be real.

---

## 4. Completeness magnitude m_c

### 4.1 What it is, and why it moves

`m_c` is the lowest magnitude above which essentially all events in the region and period
are recorded. Below it the catalog is a sample from (process × unknown detection
probability), not from the process. Every catalog here is truncated at an assumed `m_c` —
the `mcut` field of the config.

Detection requires signal above noise at enough stations to locate, so `m_c` depends on
**station density and geometry** (varies strongly in *space*: ~1.0 near the dense San
Jacinto instrumentation, ~2.5 offshore), **noise** (time of day, season, weather),
**network history** (decadal, monotonically downward), and **seismicity rate** — the vicious
one.

### 4.2 Aftershock incompleteness (STAI)

Right after a large earthquake, the mainshock coda and overlapping aftershock waveforms
**swamp detection**, and analysts cannot keep up. `m_c` jumps to roughly `M - 4` to `M - 3`
and decays back over hours to weeks, approximately

```
m_c(t) ≈ M - 4.5 - 0.75*log10(t)          [t in days]
```

which is the form fitted by **Helmstetter, Kagan & Jackson (2006, *BSSA* 96(1), 90–106)** to
the Landers, Northridge and Hector Mine sequences and used ever since as the standard STAI
correction. (Coefficients are network-specific; treat 4.5 and 0.75 as southern-California
values, not constants of nature.)

After an M7.1, the first hour (`t = 0.04 d`) has `m_c ≈ 3.6`, not 1.0. The catalog is
missing most aftershocks exactly when the rate is highest. This is the field's single most
consequential data problem: it biases Omori `c` upward (§6.2), biases `b` downward in
sequences (missing small events raise `mean(m)`, and `b = log10(e)/(mean(m) - m_c)`),
biases ETAS `a` and `k0` downward, and biases *any* temporal model the same way.

Mizrahi, Nandan & Wiemer (2021, *JGR Solid Earth* 126, e2021JB022379) — the methods paper
for the `etas` package this benchmark uses — is titled "Embracing data incompleteness for
better earthquake forecasting" precisely for this. **Note for your viva:** the benchmark's
ETAS configuration uses a *fixed* `mc` per catalog, not that paper's time-varying `mc(t)`.
Both models inherit the STAI bias.

### 4.3 How a drifting m_c manufactures a fake temporal trend

Suppose true `m_c` was 2.7 in the 1980s and 2.3 by 2015, with the analysis threshold fixed
at 2.5. The early catalog is missing [2.5, 2.7). Under GR with `b = 1`,

```
fraction of M≥2.5 events lying in [2.5, 2.7)  =  1 - 10^(-1*0.2)  =  1 - 0.631  =  0.369
```

so the early catalog is **37% sparser at the same nominal threshold**, purely from
instrumentation, and the apparent rate rises by `1/(1 - 0.369) = 1.585` over the record.

Now: `tll = log f(tau | H)`. A 1.6× rate increase is a systematic shortening of gaps, and it
is learnable from lagged log-gap features. Because the benchmark splits chronologically
(train early, test late), the trend points into the test window and the model is *rewarded*
for learning an artifact. ETAS is partly protected by its rigid form (constant `mu` plus
Omori triggering has no free secular trend), which means **an m_c artifact would show up as
a FlowQuake win** — precisely the claim under test. Partial defences: conservative
thresholds; the gain is positive in 85% of 180-day windows across the test decade
([MANUSCRIPT.md](../MANUSCRIPT.md) §4.1); and the frozen model replicates out-of-time on
2020–2026 where m_c is essentially constant (`dT = +0.0574`,
[runs/total_win.json](../runs/total_win.json)). None of these is airtight — a slow monotone
drift gives a gain that is positive throughout, just larger later.

### 4.4 Estimating m_c

All methods fit the **frequency–magnitude distribution** and find where it departs from GR.

- **Maximum curvature (MAXC).** Take the bin of the *non-cumulative* FMD with the most
  events. Fast, trivially implemented, **biased low**; Woessner & Wiemer (2005, *BSSA*)
  recommend `+0.2`. This is what the repo uses, at
  [scripts/check_completeness.py:21-30](../scripts/check_completeness.py#L21-L30) and
  identically at [scripts/build_region.py:113-122](../scripts/build_region.py#L113-L122):

  ```python
  mc = centers[np.argmax(counts)] + 0.2                          # MAXC + W&W correction
  b  = np.log10(np.e) / (above.mean() - (mc - mbin/2))           # Aki MLE, binning-corrected
  ```

- **Goodness-of-fit (GFT; Wiemer & Wyss 2000, *BSSA*).** For each trial `m_c`, fit GR above
  it and take the smallest `m_c` whose residual `R = 100*(1 - sum|N_obs - N_pred|/sum N_obs)`
  reaches 90% (or 95%). More principled, needs more events, can fail to reach threshold.
- **b-value stability (MBS; Cao & Gao 2002, *GRL*).** `b(m_c)` is biased below completeness
  and flat above it; take the smallest `m_c` where `|b_avg(m_c) - b(m_c)| ≤ delta_b` (Shi &
  Bolt standard error). Robust; needs the most data.
- **EMR (Woessner & Wiemer 2005).** Model the *whole* FMD as GR above `m_c` times a normal-CDF
  detection probability below, fit all four parameters by ML. Uses the sub-complete data.

**Expect these to disagree by 0.2–0.5 magnitude units.** `m_c` is itself an estimate with real
uncertainty, propagating into `b`, into ETAS's `a`, and into how much data you have.

### 4.5 What `scripts/check_completeness.py` does — and does not

```python
REGIONS   = {"Japan": "1992-01-01", "Chile": "1992-01-01",
             "Greece": "1992-01-01", "Iran": "1992-01-01"}
TRAIN_END = "2011-01-01";  TEST_END = "2020-01-01"
rec_mc    = float(np.ceil(max(mc_tr, mc_te)*2)/2)     # round the worse era up to 0.5
```

It estimates MAXC+0.2 `m_c` separately on the train era (1992→2011) and test era
(2011→2020) — boundaries matching the four ISC configs — takes the worse, and rounds up to
the nearest 0.5. Committed output, [runs/completeness.json](../runs/completeness.json):

| region | m_c train | m_c test | b train | b test | rec. mcut | N train ≥ mcut | N test ≥ mcut |
|---|---|---|---|---|---|---|---|
| Japan | 3.65 | 3.75 | 0.79 | 0.85 | 4.0 | 19,929 | 14,886 |
| Chile | 3.95 | 3.65 | 0.94 | 0.76 | 4.0 | 11,820 | 6,546 |
| Greece | 3.65 | 3.85 | 1.13 | 1.00 | 4.0 | 2,612 | 1,748 |
| Iran | 3.85 | 3.85 | 0.89 | 0.95 | 4.0 | 2,010 | 1,121 |

**Four honest observations to make before anyone else does:**

1. **`m_c` is not stable across eras** — it moves by up to 0.30 (Chile 3.95 → 3.65).
   [MANUSCRIPT.md](../MANUSCRIPT.md) says `m_c` is "verified to be stable across the training
   and test eras". What is verified is that the *upper envelope*, rounded up to 0.5, lands
   at 4.0. Sound engineering; generous wording.
2. **Italy and all five California catalogs are never checked.** `REGIONS` has four names
   and `runs/completeness.json` has four keys. [REPRODUCE.md](../REPRODUCE.md) §1 says the
   script "confirms mc 4.0 (ISC) / 2.5 (Italy)" — it does not. Italy is the second-largest
   contributor to the headline temporal claim (`dT = +0.0712`, Holm p = 0.003,
   [runs/stats_hardening.json](../runs/stats_hardening.json)) and its completeness is asserted,
   not measured, in the committed artifacts.
3. **`m_c` is estimated once per region, pooled over space and time.** For Japan that pools
   the Ryukyu arc, Nankai, Tohoku and Hokkaido, and shallow crustal with 600 km-deep slab
   seismicity, into one number.
4. **No aftershock windows are excluded.** MAXC on a test era containing the 2011 M9.0
   Tohoku sequence treats STAI-censored magnitudes as a normal sample. Chile's `b` falling
   0.94 → 0.76 between eras is *suggestive* of this — its test era contains the 2014 Iquique
   M8.2 and 2015 Illapel M8.3 sequences — but say the honest version: Chile's estimated `m_c`
   also **fell** (3.95 → 3.65) between the same eras, which is the opposite of what pure STAI
   censoring predicts and is more naturally read as network improvement. Two effects are
   confounded in one number and the committed artifacts separate neither.

---

## 5. Gutenberg–Richter, and the b-value done properly

### 5.1 The law

Gutenberg & Richter (1944, *BSSA*): `log10 N(≥ m) = a_GR - b*m`. Above completeness this is
equivalent to exponentially distributed magnitudes:

```
P(M > m | M ≥ m_c) = 10^(-b (m - m_c)) = exp(-beta (m - m_c)),      beta = b * ln(10)
f_m(m) = beta * exp(-beta (m - m_c)),   m ≥ m_c
```

`b ≈ 1` almost everywhere, so `beta ≈ 2.3026`: ten times fewer M5s than M4s. Higher `b`
means relatively more small events (steeper size distribution).

| setting | typical b |
|---|---|
| global tectonic average | 1.0 |
| volcanic / geothermal / swarm | 1.2 – 2.5 |
| subduction megathrust | 0.7 – 0.9 |
| aftershock sequences | 0.8 – 1.1 (apparently lower under STAI) |
| creeping / high heat flow | > 1 |

The strongest physical correlate is **differential stress**: `b` decreases as differential
stress rises, which is why `b` is watched as a stress proxy and why "b dropped before the
mainshock" claims are perennial and perennially contested.

Two caveats to keep loaded:

- Pure GR is unbounded above, which is impossible (fault dimensions are finite). Fixes: a
  hard **truncation** at `m_max`, or a **tapered** (Kagan) rolloff. FlowQuake's
  `GRMagnitudeHead` uses the untruncated exponential and clamps to `[m_c, 8.5]` in the
  *sampler* only ([flowquake/model.py](../flowquake/model.py); [STACK.md](../STACK.md) §11).
- Kagan argued `b ≡ 1` universally, with all apparent variation coming from magnitude
  errors, incompleteness and finite samples. A minority position, not a fringe one — and the
  correct reply to "your b ranges 0.76 to 1.13, so the regimes differ."

### 5.2 The Aki MLE, derived

`n` magnitudes `≥ m_c`, i.i.d. from `f(m) = beta*exp(-beta(m - m_c))`. Write
`u_i = m_i - m_c ≥ 0`:

```
L(beta)     = sum_i [ log beta - beta*u_i ]  =  n log beta - beta * sum_i u_i
dL/dbeta    = n/beta - sum_i u_i = 0     =>    beta_hat = n / sum_i u_i = 1/ubar
d²L/dbeta²  = -n/beta² < 0                     (so it is a maximum)

b_hat = beta_hat / ln(10) = log10(e) / ( mean(m) - m_c ),      log10(e) = 0.4342945
```

That is **Aki (1965)**, *Bull. Earthq. Res. Inst. Tokyo* 43, 237–239. It is the reciprocal
of the mean excess above threshold, the only sufficient statistic an exponential has.

### 5.3 Exact sampling distribution, standard error, two-sample test

Three results from one identity. If `U ~ Exp(beta)` then `2*beta*U ~ chi²(2)`, so for
`S = sum_i u_i`,

```
2*beta*S ~ chi²(2n)        and, since beta_hat = n/S,

    beta / beta_hat  =  beta*S/n  =  X / (2n),     X ~ chi²(2n)                       (*)
```

**Standard error (Aki).** For `X ~ chi²(2n)`, `Var(log X) ≈ psi'(n) ≈ 1/n`. From (*),
`log beta_hat = log beta - log(X/(2n))`, so

```
sd( log beta_hat ) ≈ 1/sqrt(n)      =>      sd( b_hat ) ≈ b / sqrt(n)
```

Shi & Bolt (1982, *BSSA*) give the more conservative sample-based form
`sigma_b = 2.30*b²*sqrt( sum_i (m_i - mean(m))² / (n(n-1)) )`, which does not assume the
exponential model is exactly right. They agree closely when it is (worked example A:
0.1771 vs 0.1782 for n = 20).

**Exact two-sample test.** Applying (*) to two independent samples,

```
R = (beta_1/beta_hat_1) / (beta_2/beta_hat_2) = [X_1/(2n_1)] / [X_2/(2n_2)] ~ F(2n_1, 2n_2)
```

Under `H0: beta_1 = beta_2` this collapses to **`b_hat_2 / b_hat_1 ~ F(2n_1, 2n_2)`** — an
exact test for any sample sizes. For large samples,
`log(b_hat_2/b_hat_1) ~approx~ Normal(0, 1/n_1 + 1/n_2)`.

(You will also meet an **Utsu (1992)** likelihood-ratio approximation for comparing two
b-values. I do not reproduce its algebraic form here because I could not verify it against
the original; the F-test above is exact and derived from scratch, so prefer it and say why.)

**The weak point is independence.** Aftershocks are not independent draws, so the effective
sample size is below `n` and all these intervals are too narrow. The repo handles exactly
this problem for likelihood gains with a stationary block bootstrap
([flowquake/stats.py](../flowquake/stats.py)); nobody here does it for b-values.

### 5.4 The binning correction (Utsu / Bender)

Catalogs report magnitudes on a grid of width `dm`. An event *reported* as `m` has true
magnitude in `[m - dm/2, m + dm/2)`, so the *reported* threshold `m_c` corresponds to a
*true* threshold `m_c - dm/2`. Model `M - (m_c - dm/2) ~ Exp(beta)`; then for a grid point
`m ≥ m_c`,

```
P(reported = m) = exp(-beta(m - dm/2 - (m_c - dm/2))) - exp(-beta(m + dm/2 - (m_c - dm/2)))
                = exp(-beta(m - m_c)) * [ 1 - exp(-beta*dm) ]
```

Maximising `sum_i log P(m_i)`: the bracket is `i`-independent, so

```
-sum_i (m_i - m_c) + n*dm*exp(-beta dm)/(1 - exp(-beta dm)) = 0
```

Write `x = beta*dm` and `ubar = mean(m) - m_c`. The score equation says
`ubar = dm/(e^x - 1) = (1/beta)*[x/(e^x - 1)] = (1/beta)*(1 - x/2 + x²/12 - ...)`, i.e.

```
1/beta = ubar + dm/2 - beta*dm²/12 + O(dm⁴)          (exact-to-quartic)
=>  beta_hat ≈ 1 / ( ubar + dm/2 ),      b_hat = log10(e) / ( mean(m) - (m_c - dm/2) )
```

i.e. **replace `m_c` by `m_c - dm/2`** — the **Utsu (1966) / Bender (1983)** correction. It
always *lowers* `b_hat`. Two quantitative riders, both checkable in three lines of Python:

- **The correction is not small.** For `dm = 0.1` and true `b = 1` the correctly-shifted mean
  excess is `1/beta = 0.4343`, so `ubar = 0.3843` and the *uncorrected* estimate is
  `0.4343/0.3843 = 1.130` — **13% high**.
- **The correction is not exact either.** Dropping the `-beta*dm²/12` term leaves a *relative*
  bias of `-(beta*dm)²/12`: at `beta = 2.3026`, `dm = 0.1` the exact `ubar` is 0.38621 and the
  half-bin estimator returns `beta_hat = 2.2925` against a true 2.3026 — **0.44% low**. So the
  half-bin rule slightly over-corrects. Negligible next to the 13% it fixes, but know it
  exists before someone asks whether `dm/2` is exact. (It is not; only the *leading* term is.)

The repo applies the correction in both places it estimates `b`
([scripts/check_completeness.py:29](../scripts/check_completeness.py#L29),
[scripts/build_region.py:121](../scripts/build_region.py#L121)), with `mbin = 0.1`.

### 5.5 The `+0.005` in `heads.py` — what it corrects, and the doc bug

[flowquake/heads.py:170-174](../flowquake/heads.py#L170-L174):

```python
def log_prob(self, m, cond, mc):
    """m: (B,) raw magnitudes >= mc. Half-bin shift handles discretization."""
    beta = self.beta(cond)
    dm   = torch.clamp(m - mc, min=0.0) + 0.005
    return torch.log(beta) - beta * dm
```

**What it is for.** The head reports a *continuous density* `mll = log f_m`, but the data
are on a grid. Scoring a density at grid points is meaningful only up to `log(dm)`; the
comparable object is `log P(m) - log(dm)`. From §5.4, with `x = beta*dm`:

```
log P(m) - log(dm) = -beta(m - m_c) + log[ (1 - e^{-x})/dm ]
                   = log(beta) - beta(m - m_c) + log[ (1 - e^{-x})/x ]
                   = log(beta) - beta*[ (m - m_c) + dm/2 ]  +  x²/24 + O(x³)
```

So the code's *form* is exactly right and the correct shift is **`dm/2`, half the bin
width**. The neglected term is `(2.3026*0.1)²/24 = 0.0022` nats for a 0.1 grid — negligible.
With no shift at all, `mll` is too high by `beta*dm/2 ≈ 0.115` nats/event on a 0.1 grid.

**The problem, in its strongest form.** `0.005 = dm/2` implies `dm = 0.01`. But
[STACK.md:726-728](../STACK.md) and [MANUSCRIPT.md](../MANUSCRIPT.md) §2 both justify it as "a
half-bin shift for the catalog's **0.1**-magnitude discretization", and half of 0.1 is
**0.05**. Two things follow, and only the first depends on data I cannot see:

1. *Which* of code and documentation is wrong depends on the on-disk decimal precision of
   ComCat's magnitude column, and `reference/` is not committed ([README.md](../README.md)), so
   I could not check. USGS commonly reports two decimals (`2.53`), which would make the code
   right and the documentation wrong; ISC and INGV report one, which would make the code 10×
   too small. [flowquake/data.py:201](../flowquake/data.py#L201) applies only
   `magnitude >= mcut` and never rounds, so whatever the file holds is what the head sees.
2. **The constant is wrong for *some* catalog no matter how that resolves**, and this needs
   no uncommitted data at all. `0.005` is a **hardcoded literal** in
   [flowquake/heads.py:173](../flowquake/heads.py#L173) — one value shared by all eleven
   catalogs, with no per-catalog plumbing. The ISC and INGV catalogs are certainly on a 0.1
   grid (it is what the ETAS side assumes:
   [scripts/precompute_trigger_features.py:36](../scripts/precompute_trigger_features.py#L36)
   defines `round_half_up(x, delta=0.1)` and
   [:65](../scripts/precompute_trigger_features.py#L65) applies it, to match the `etas`
   package's `delta_m` binning). So either ComCat is also on 0.1 and *every* catalog is
   mis-shifted 10×, or ComCat is on 0.01 and the five agency catalogs used in the results are.
   Given that ISC/INGV are on 0.1, no assignment of ComCat's precision makes a single
   hardcoded `0.005` correct everywhere. **Make that the claim in a viva** — unlike point 1
   it cannot be overturned by whatever `reference/` turns out to contain.

(Two binning conventions circulate — "reported `m` is the bin's lower edge, true
magnitude uniform on `[m, m+dm)`" and "reported `m` is the bin *centre*". They give
the **same** `+dm/2` shift, provided that under the second you also re-anchor the
truncation at `m_c - dm/2`, which is what §5.4 does and what the code's
reported-magnitude cut requires. [Ch. 8 §2.8](08-flowquake-synthesis.md#28-conditional-gutenbergrichter-with-a-half-bin-shift)
works both and records that dropping the re-anchoring wrongly gives zero.)

Do **not** cite `mag_dequant: 0.01` as supporting evidence, even though it appears in every
production config. That field is dead: [flowquake/model.py:61](../flowquake/model.py#L61) takes
it as `mag_dequant: float = 0.0, # kept for config compat (GR head absorbs it)` and never
uses it. It is also `0.01` in the ISC and INGV configs, where a 0.01 grid is certainly wrong,
so it is a copied default, not a measurement.

**Quantify the mis-shifted branch.** The MLE with shift `s` is `beta_hat = 1/(ubar + s)`.
Take true `b = 1` on a 0.1 grid, so the correctly-shifted mean is `ubar + 0.05 = 1/2.3026 =
0.4343`, i.e. `ubar = 0.3843`:

```
s = 0.05  (correct):  beta_hat = 1/0.4343 = 2.3026  ->  b_hat = 1.000
s = 0.005 (code):     beta_hat = 1/0.3893 = 2.5687  ->  b_hat = 1.116
```

an **11.6% upward bias in fitted `beta`** on any 1-decimal catalog, and `mll` inflated by
`beta*(0.05 - 0.005) ≈ 0.104` nats/event.

**Consequences, stated precisely.**

- **The headline is unaffected.** `nll = -(tll + sll)` ([README.md](../README.md)) and `mll` is
  not in it. Stronger than that: production runs use `h_bottleneck = 0`
  ([README.md](../README.md)), in which case `_cond` returns the raw safe token dims with no
  learned encoder in the path ([flowquake/model.py:160-164](../flowquake/model.py#L160-L164)).
  `GRMagnitudeHead` is then a **standalone `nn.Linear`** sharing no trainable parameter with
  `head_t` or `head_s`, so a biased magnitude gradient cannot leak into `tll` or `sll` through
  a shared trunk. That is a real defence, not a hopeful one.
- **Simulation and the CSEP M-test are affected.**
  [flowquake/heads.py:176-180](../flowquake/heads.py#L176-L180) samples `mc - log(u)/beta` with
  *no* shift at all, so an inflated `beta` makes simulated magnitudes systematically too
  small. The manuscript credits this head with M-test consistency
  ([MANUSCRIPT.md](../MANUSCRIPT.md) §4.2: **M 89/92** for the production head against ETAS's
  87/92; **90/92** for the full-history head,
  [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md)). Note *which* catalog that is
  measured on — ComCat_25 only. A passing M-test on ComCat is therefore weak positive
  evidence that ComCat really is on a 0.01 grid, and says nothing about the agency catalogs,
  which are never CSEP-tested.

One command settles the precision question:

```bash
python -c "import pandas as pd; m=pd.read_csv('reference/Datasets/ComCat/ComCat_catalog.csv').magnitude; \
print(sorted(set((m*100).round().astype(int)%10))[:12])"
```

If that prints only `[0]`, ComCat is on a 0.1 grid too and the constant is wrong everywhere.

---

## 6. Omori–Utsu, and why the decay is a power law

### 6.1 The law

Omori (1894) found aftershock rate decaying as `1/(t + c)` after the 1891 Nobi earthquake;
Utsu (1961, *Geophys. Mag.* 30, 521–605) generalised the exponent:

```
n(t) = K / (t + c)^p
```

`K` = productivity (scales with mainshock magnitude, §7); `p` = decay exponent, typically
**0.9–1.2**, most often ~1.1, correlating weakly with heat flow (hotter → larger p); `c` = a
small offset regularising `t = 0`, quoted anywhere from 1e-4 to 1e-1 days. In the repo's
ETAS the term is `(Delta_t + c)^{-(1+omega)}`, so **`p = 1 + omega`**
([flowquake/neural_etas.py:80](../flowquake/neural_etas.py#L80)); the extra
`exp(-Delta_t/tau_tap)` taper makes the integral finite regardless of `p`.

**Integrability matters.** Expected aftershocks in `[t1, t2]`:

```
N(t1,t2) = K * [ (t+c)^(1-p) / (1-p) ] from t1 to t2
```

- `p > 1`: `N(t1, ∞)` finite — the sequence has a total.
- `p = 1`: `N = K*log((t2+c)/(t1+c))` — logarithmically divergent; formally never ends.
- `p < 1`: divergent, worse.

Fitted `p` sits near 1, so ETAS's `tau_tap` taper is doing real work guaranteeing a finite
branching ratio.

### 6.2 The c-value is partly an artifact

Fitted `c` varies by three orders of magnitude between studies of the same region. The
mainstream view is that much of that is **STAI** (§4.2): if the first hours are missing
their small events, the observed early rate is depressed and `c` absorbs the depression.
When detection improves — template matching, dedicated coda reanalysis — aftershock rates
continue as `1/t` down to seconds and `c` shrinks toward zero. (The case is made in Kagan
2004, *BSSA* 94(4), 1207–1228, and in Peng, Vidale and co-workers' high-resolution Parkfield
studies; check the exact Peng reference before quoting a number from it.) Honest position:
**`c` mostly encodes how bad your catalog is in the first hours**, even though rate-and-state
does predict a physical `c` (§6.3).

### 6.3 Why a power law, answer 1: rate-and-state friction

Dieterich (1994, *JGR* 99(B2), 2601–2618) derived a **seismicity rate equation** from
rate-and-state friction. Take the result as given — I do not derive the constitutive law:

> A population of nucleation sites obeying rate-and-state friction, driven at constant
> background stressing rate `tau_dot_r`, produces seismicity rate
> `R = r / (gamma_state * tau_dot_r)`, with
> `d(gamma_state) = [dt - gamma_state*d(tau)] / (A*sigma)`, where `r` is the background
> rate, `A` the direct-effect parameter and `sigma` the effective normal stress.

Impose a **step** in shear stress `Delta_tau` at `t = 0` (the mainshock's static transfer),
then resume constant loading. Solving gives

```
             r                                                  A * sigma
R(t) = ------------------------------------------ ,     t_a = -------------
        [ e^{-Delta_tau/(A sigma)} - 1 ] e^{-t/t_a} + 1         tau_dot_r
```

Check the limits: at `t = 0` the denominator is `exp(-Delta_tau/(A sigma))`, so
`R(0) = r*exp(Delta_tau/(A sigma))` — a large rate jump ✓; as `t → ∞` the denominator → 1
and `R → r` ✓.

Now write `E = exp(-Delta_tau/(A sigma))`, small for a large step. For `t << t_a`,
`e^{-t/t_a} ≈ 1 - t/t_a`, so

```
denominator ≈ (E - 1)(1 - t/t_a) + 1 = E + (1 - E)*t/t_a ≈ E + t/t_a

           r            r * t_a           K
R(t) ≈ ---------  =  --------------  =  -------,     K = r*t_a,  c = t_a*e^{-Delta_tau/(A sigma)}
        E + t/t_a     t + E*t_a          t + c
```

**Omori's law with p = 1 exactly, plus a natural c-value, from laboratory friction.** Two
riders a professor may probe: (i) the derivation gives `p = 1` exactly, so observed `p ≠ 1`
requires heterogeneity in `A*sigma`, in stress step size, or in `t_a` — the *deviation* is a
heterogeneity measurement, not a friction-law measurement; (ii) `t_a = A*sigma/tau_dot_r`
predicts an aftershock *duration*: with `A*sigma ≈ 0.1` MPa and `tau_dot_r ≈ 0.003` MPa/yr,
`t_a ≈ 30` yr, so sequences last decades and then return to background.

### 6.4 Why a power law, answer 2: superposition

A model-free argument that does not require believing rate-and-state. Suppose each triggered
site relaxes exponentially at its own rate `lam`, and the population has density
`p(lam) ∝ 1/lam` over `[lam_min, lam_max]` — scale-free, which is what a heterogeneous fault
system with no preferred length scale gives. Then

```
n(t) = ∫ p(lam) * lam * e^{-lam t} dlam  =  C * ∫_{lam_min}^{lam_max} e^{-lam t} dlam
     = C * ( e^{-lam_min t} - e^{-lam_max t} ) / t
     ≈ C / t          for   1/lam_max << t << 1/lam_min
```

Exactly `1/t` over the intermediate range, with exponential cutoffs at both ends — the short
one playing the role of `c`, the long one of `tau_tap`. **Power laws are the signature of a
system with no characteristic timescale; exponentials of a system with exactly one.** Faults
have neither one timescale nor one length scale, hence power laws in time *and* space.

### 6.5 The other mechanisms

| mechanism | timescale | predicts |
|---|---|---|
| **Rate-and-state nucleation** (Dieterich 1994) | s – decades | `p = 1` exactly, `c` from the stress step (§6.3) |
| **Afterslip** — aseismic continued slip loading nearby patches | days – years | Aftershock rate ∝ afterslip rate, which decays ~1/t for logarithmic (velocity-strengthening) creep. Also Omori. |
| **Fluid / pore-pressure diffusion** — a pressure pulse lowering effective normal stress | hours – years | A *migrating* triggering front at `r ≈ sqrt(4 pi D t)` for diffusivity `D` (0.01–10 m²/s) — a spatial signature static triggering does not have. Dominates swarms, geothermal fields, induced sequences. |
| **Viscoelastic relaxation** of lower crust / upper mantle | years – centuries | Delayed, distant triggering that static Coulomb transfer cannot explain |

**Why this matters here:** ETAS bundles all of these into `(c, p, tau_tap)`. That is a strong
prior — nearly right for mainshock–aftershock sequences, and *wrong* for fluid-driven swarms,
whose rate rises then falls and whose extent expands as `sqrt(t)`. Salton Sea is a geothermal
swarm region (§12.3) and is where FlowQuake's temporal margin is largest (+0.102,
[runs/fullsuite_summary.json](../runs/fullsuite_summary.json)). Suggestive; labelled a
hypothesis, not a result — the repo does not test it.

---

## 7. Productivity scaling and Båth's law

**Utsu productivity.** The number of aftershocks grows exponentially with mainshock size:

```
N_aft(≥ m_c)  ∝  10^( alpha*(m - m_c) )  =  exp( a*(m - m_c) ),      a = alpha * ln(10)
```

with `alpha ≈ 0.8–1.0`, `a ≈ 1.8–2.3`. This is the `k0*exp(a*(m_j - m_c))` factor at
[flowquake/neural_etas.py:78-83](../flowquake/neural_etas.py#L78-L83). The case `alpha = b` is
**self-similar branching**. Whether `alpha` is below, at, or above `b` is a live empirical
question, confounded by STAI (which removes small aftershocks of large mainshocks, biasing
`alpha` down).

**The branching ratio, and why `a < beta` is required.** Expected direct offspring of a
parent of magnitude `m`:

```
nu(m) = ∫_0^∞ k0*e^{a(m - m_c)} * e^{-t/tau_tap} * (t + c)^{-(1+omega)} dt
      = k0 * e^{a(m - m_c)} * I,       I = ∫_0^∞ e^{-t/tau_tap}(t+c)^{-(1+omega)} dt  < ∞
```

Averaging over the GR-distributed parent magnitude:

```
n_branch = ∫_{m_c}^∞ nu(m) * beta * e^{-beta(m - m_c)} dm
         = k0 * I * beta * ∫_0^∞ e^{(a - beta)u} du
         = k0 * I * beta / (beta - a)        provided  a < beta
```

**If `a ≥ beta` the expected offspring count is infinite.** A stationary ETAS *requires*
`a < beta = b*ln(10) ≈ 2.30` for `b = 1`. Published fits typically land at `a ≈ 2.0–2.3` and
`n_branch ≈ 0.8–0.95`, i.e. right at the edge and *nearly critical*, which is why the
likelihood is flat and ill-conditioned in `(a, k0)` — the mean is dominated by rare large
events — and why forecast variance is enormous. **Flag the provenance:** those are
literature values, not this repo's. The fitted ETAS parameters live in
`reference/Experiments/ETAS/output_data_<Cfg>/parameters_0.json`, which is not committed
([README.md](../README.md)), and no committed run JSON carries them, so I can state the
functional form and the `a < b ln 10` constraint but cannot quote FlowQuake's own `a`, `k0`
or branching ratio. Do not let anyone believe otherwise. What *is* committed and consistent
with near-criticality: EM inversion takes 3–4 CPU-hours per region
([REPRODUCE.md](../REPRODUCE.md) §2).

**Båth's law.** The largest aftershock is on average ~**1.2** magnitude units below the
mainshock, roughly independently of mainshock size (Båth 1965, *Tectonophysics* 2, 483–514 —
the observation appears in a paper whose title, "Lateral inhomogeneities of the upper
mantle", gives no hint of it). The standard explanation is that this is **not an independent
law** but a consequence of GR + productivity + **selection bias**: you only call an event a
mainshock if it is the largest in its sequence, which conditions on the largest aftershock
being smaller. (Vere-Jones argued this early; later derivations from GR alone include Vere-
Jones et al. and Console, Lombardi, Murru & Rhoades — check venues before citing a specific
one.) Worked example B shows a standard generic aftershock model implies a deficit of ~0.66
rather than 1.2 — a real, well-known tension, mostly about how "largest aftershock" is
defined and over what window.

---

## 8. Foreshocks, aftershocks, swarms — and declustering

**The labels are model-dependent, not physical.** No measurement distinguishes a foreshock
from a mainshock. A **mainshock** is the largest event in some space–time window; a
**foreshock** precedes it inside that window; an **aftershock** follows; a **swarm** is a
burst with no dominant event (the largest does not stand out by ~1 unit) and a non-Omori
envelope that often rises then falls and migrates. Change the window and labels change: an
M5 "mainshock" on Monday becomes a "foreshock" when an M6 occurs on Tuesday, and *nothing
about the M5 changed*. That is why ETAS and FlowQuake model all events with the same
machinery — in a branching formulation every event is simultaneously a potential parent and
a potential offspring, and there is no "mainshock" variable anywhere. Say that out loud; it
is a genuine conceptual advantage.

**Declustering** removes triggered events to leave a putative Poisson background:

| method | how | failure mode |
|---|---|---|
| **Gardner & Knopoff (1974, *BSSA* 64(5), 1363–1367)** | Fixed magnitude-dependent space–time windows (e.g. ~40 km, ~500 d for M6.5) | Windows arbitrary, tuned on 1970s southern California; over-removes in swarms, under-removes long sequences; deterministic, no uncertainty |
| **Reasenberg (1985, *JGR* 90(B7), 5479–5495)** | Links events whose interaction zones overlap; adaptive time windows keyed to the largest event so far in the cluster | Several tuning parameters; still deterministic |
| **Stochastic (Zhuang, Ogata & Vere-Jones 2002, *JASA*)** | Fit ETAS; compute `p_ij` that `i` was triggered by `j` and `p_i0 = mu/lambda(t_i, s_i)` that it is background; thin by those probabilities | Model-dependent by construction, but honest about uncertainty and gives a *distribution* of declustered catalogs |

Removing aftershocks changes the magnitude distribution (Mizrahi, Nandan & Wiemer 2021,
*SRL* 92(4), 2333–2342), the background rate, the b-value and the spatial distribution. Any
"the declustered catalog is Poisson" claim is partly a claim about the algorithm.

**Where this lands for FlowQuake:** neither ETAS nor FlowQuake declusters — both are fitted
to the full catalog, which is right. **But** `adaptive_bg_grid`
([flowquake/data.py:120-151](../flowquake/data.py#L120-L151)) smooths **all** training-era
epicentres, aftershocks included, so the "background" map is really a smoothed *total*
seismicity density that over-weights places with a big sequence during training. Frankel
(1995) and Helmstetter, Kagan & Jackson (2007) face the same choice and argue smoothing the
undeclustered catalog forecasts better. That argument is your defence; know it.

---

## 9. Triggering: static, dynamic, and why space is anisotropic

**Static Coulomb stress transfer.** An earthquake permanently changes the surrounding stress
field. The standard scalar summary on a receiver fault of specified orientation:

```
Delta_CFS = Delta_tau + mu_f * ( Delta_sigma_n + Delta_P )
```

`Delta_tau` = shear stress change in the slip direction; `Delta_sigma_n` = normal stress
change (positive = unclamping); `Delta_P` = pore-pressure change, often folded in via an
apparent friction `mu_f' ≈ 0.4`; `mu_f ≈ 0.6–0.8`. King, Stein & Lin (1994, *BSSA* 84(3),
935–953) showed 1992 Landers aftershocks concentrate in positive-`Delta_CFS` lobes and are
depleted in negative "stress shadows". **The geometry around a slipping rectangular
dislocation is a four-lobed butterfly, elongated along strike** — emphatically not isotropic.
Magnitudes are small: 0.01–0.1 MPa triggers abundantly, roughly **0.1–10%** of a typical
1–10 MPa stress drop. Small nudges matter because faults are already near failure.

**Dynamic triggering.** The passing wavefield produces transient stresses far larger than
the static change at distance. Hill et al. (1993, *Science* 260, 1617–1623) documented
seismicity across the western US within minutes of the M7.3 Landers earthquake, out to
~1250 km — beyond any static effect. The static field of a finite source falls as `r^-3` in
the far field; a surface wave loses only `r^-1/2` geometrically, and something like `r^-1.5`
to `r^-2` once anelastic attenuation is folded in — so dynamic stress dominates at distance
by many orders of magnitude. Velasco, Hernandez, Parsons & Pankow (2008, *Nature Geoscience*
1, 375–379, "Global ubiquity of dynamic earthquake triggering") found small-event rate
increases after 12 of 15 M > 7 earthquakes since 1990, independent of tectonic setting.
Proposed mechanisms: transient unclamping, shaking-driven pore-pressure redistribution,
bubble mobilisation, frictional weakening. Strongest in geothermal and volcanic areas —
again the Salton Sea and Geysers. *Implication:* a model with a single fitted decay exponent
`rho` is fitting a mixture of two mechanisms with different tails.

**Why the aftershock cloud is elongated, and what FlowQuake does.** Aftershocks decorate the
rupture surface and its tips. Rupture length scales with magnitude — Wells & Coppersmith
(1994, *BSSA* 84(4), 974–1002), roughly `log10(L[km]) ≈ 0.6*M - 2.5` for subsurface rupture
length, all slip types (an order-of-magnitude form: read the paper's Table 2 for the actual
regressions and their 0.25–0.35-unit scatter). So M6.5 → ~20–25 km, M7.5 → ~80–100 km. The
aftershock zone is a strip of that length, a few km wide, oriented along strike. Two model
responses:

- **ETAS as configured here is isotropic**: `K_j(r²) = (r² + d_j)^{-(1+rho)}`
  ([flowquake/neural_etas.py:82-84](../flowquake/neural_etas.py#L82-L84); here `rho` is the ETAS
  *spatial decay exponent*) with `d_j = d*exp(gamma(m_j - m_c))` — magnitude-dependent
  *size*, no orientation. (Ogata 1998,
  *AISM* 50, 379–402, proposed elliptical ETAS kernels; the `etas` package configuration
  used by this benchmark does not.)
- **FlowQuake's `KernelMixtureHead` is anisotropic.**
  [flowquake/heads.py:84-85](../flowquake/heads.py#L84-L85) builds an aspect ratio
  `rho = 1 + softplus(.) ∈ [1, 5]` (a *different* `rho` — see the notation note) and an
  orientation `theta = atan2(.,.)` per component, and
  [:89-94](../flowquake/heads.py#L89-L94) uses them as elliptical axes `d*rho` and `d/rho`.
  The axes are
  **area-preserving**: `(d*rho)*(d/rho) = d²`, so `sqrt(det M) = d²` and the *same*
  normalizer works — elongation costs no Jacobian term ([STACK.md](../STACK.md) §10;
  `tests/test_heads.py` numerically integrates a forcibly elongated component and confirms
  it still integrates to 1). Initialisation is isotropic (`rho_raw` bias `-5.0`, so
  `rho ≈ 1.007`, [flowquake/heads.py:62-69](../flowquake/heads.py#L62-L69)), and the MLP
  producing `(rho, theta)` is fed only `[cond, log Delta_t_j, m_j, log dist_j]` — the
  parent's age, size and offset, never an absolute coordinate.

This is the cleanest "we encoded real physics" story in the repo. Be ready for the
counterpunch: the production kernel-mixture head nonetheless **loses** to ETAS spatially on
all five California catalogs ([runs/fullsuite_summary.json](../runs/fullsuite_summary.json):
ComCat `sll` −9.059 vs ETAS −8.690). Anisotropy alone does not buy a spatial win; that comes
from the separate full-history `neural_etas` head.

---

## 10. Spatial structure and background models

**Seismicity is not uniform at any scale.** Earthquakes occur on faults, faults form
networks, and fault networks are scale-invariant over several decades of length: the
correlation (fractal) dimension of **epicentres in map view** is typically ~1.2–1.8 — between
a line (fault traces) and the plane — and ~2.0–2.3 for **hypocentres in 3-D**. (Quote the
right one: a map-view dimension above 2 is impossible, so a paper reporting 2.2 is talking
about hypocentres.) In California, density varies by **five or six orders of magnitude**
between the Salton Sea / Geysers hotspots and the Great Valley.

A uniform background gives `sll = log(1/A)`. Invert the benchmark's Poisson baseline
`sll = -13.7745` ([README.md](../README.md)) and you get `A = e^13.7745 = 9.6e5 km²` — which is
the RELM/CSEP California polygon's area to within rounding, so the baseline is exactly the
uniform model and nothing more. ETAS gets `-8.690`. **That ~5.1-nat gap is what a spatial
model buys, and most of it is knowing where the faults are.**

**Smoothed-seismicity models** say past seismicity locates future seismicity: smooth the
catalog, normalise, use it as the spatial density. **Frankel (1995, *SRL*)** — used for the
US National Seismic Hazard Maps — counts events in cells and smooths with a **fixed**-
bandwidth Gaussian (~50 km central/eastern US, less in the west). **Helmstetter, Kagan &
Jackson (2007, *SRL* 78(1), 78–86)** use an **adaptive** bandwidth: event `i` is smoothed
with `sigma_i` = distance to its `k`-th nearest neighbour. Dense clusters get narrow kernels
(fault traces survive); isolated events get wide ones (off-fault probability never collapses
to zero). This won the RELM 5-year California experiment. *Why adaptive wins:* a fixed
bandwidth must trade resolving a fault trace (small sigma) against not assigning zero
probability to a genuine off-fault event (large sigma), and a likelihood punishes the second
failure catastrophically — one event in a zero cell contributes `-inf`.

**`adaptive_bg_grid` is HKJ, bucketed for speed**
([flowquake/data.py:120-151](../flowquake/data.py#L120-L151)):

```python
dk, _ = cKDTree(pts).query(pts, k=min(k+1, len(pts)))     # k = 6
sig   = np.clip(dk[:, -1], sigma_min, sigma_max)          # 1 km <= sigma_i <= 60 km
edges = np.logspace(log10(sig.min()), log10(sig.max()+1e-6), n_buckets+1)     # 12 buckets
for b in range(n_buckets):
    sub = np.zeros((nx, ny)); np.add.at(sub, (gx[sel], gy[sel]), 1.0)
    grid += gaussian_filter(sub, sigma=sqrt(edges[b]*edges[b+1]) / bin_km)
grid = grid / (grid.sum() * bin_km**2 + 1e-12)            # density per km^2
grid = (1 - floor)*grid + floor / stats["bg_area"]        # floor = 0.03 uniform mix
```

- `sigma_i` = distance to the **6th nearest neighbour**, clipped to `[1, 60]` km — exactly
  the HKJ variable bandwidth.
- Events are **bucketed into 12 log-spaced bandwidth classes**, each smoothed once at the
  bucket's geometric-mean bandwidth: 12 FFT convolutions instead of one kernel per event,
  at the cost of quantising the bandwidth.
- Normalised over the region, then mixed with **3% uniform** — likelihood insurance, so a
  test event in an empty cell cannot score `-inf`. The fixed-bandwidth fallback
  ([flowquake/data.py:262-272](../flowquake/data.py#L262-L272)) uses a 2% floor and
  `sigma = 2` cells (≈ 4 km): that is the Frankel-style model, kept for comparison.

Two properties to state under fire: **(i)** the map is fitted on the training era only
(`adaptive_bg_grid(x[fit], y[fit], ...)`), so it does not leak test locations; **(ii)** it is
nonetheless *absolute geography baked into the model* — everything else in the production
spatial head is translation-invariant (mixture components sit at observed locations supplied
at eval time; `SAFE_TOKEN_DIMS = [0, 3] + range(4, 32)`,
[flowquake/model.py:35](../flowquake/model.py#L35), keeps `log tau` and `m` and the 28
relational features and drops raw dims 1 and 2, which are `x` and `y`). Both
[REPRODUCE.md](../REPRODUCE.md) ("lighter than an ETAS inversion but not zero target-catalog
preprocessing") and [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) ("the model is not
target-catalog-free") list the background map explicitly as a residual preprocessing
dependency. Concede it before you are asked.

---

## 11. Tectonic regimes, and whether one model should transfer

| regime | mechanics | here | what tends to differ |
|---|---|---|---|
| **Transform / strike-slip** | Plates slide horizontally past each other; near-vertical faults; 0–20 km | California (all five benchmark catalogs) | `b ≈ 1.0`; abundant M<7; well instrumented, low `m_c` |
| **Subduction** | Underthrusting; megathrust + intraslab + outer-rise; 0–700 km | Japan, Chile | `b ≈ 0.7–0.9` on the interface; M9+; enormous textbook Omori sequences; strong depth structure a 2-D model cannot see |
| **Extension / rifting** | Crust pulls apart; normal faults; < 15 km | Italy (Apennines), Greece (Aegean) | Frequent swarms and M5–6.5 sequences; `b` often > 1; high heat flow → faster decay (larger `p`) |
| **Collision** | Distributed continental convergence | Iran (Zagros/Alborz) | Diffuse over a huge area; low rate per unit area; often high `b`; sparse instrumentation → high `m_c` |
| **Volcanic / geothermal / induced** | Fluid- and heat-driven | Salton Sea, Geysers (inside `ComCat_25` and `SaltonSea_10`) | Swarms, migration fronts, `b` often 1.2–2.0; Omori a poor description |

**Why it *should* transfer.** The three empirical laws hold in every regime; only parameters
change. Rate-and-state friction is a property of rock, not plate-boundary type, so the
triggering *form* should be universal. FlowQuake's conditioning is deliberately relational
and translation-invariant (`SAFE_TOKEN_DIMS` = log-tau, magnitude and 28 relational
features; absolute `x, y` excluded), so it learns a *kernel*, not a *map* — and kernels are
the universal part. Empirically, leave-one-region-out pooled pre-training plus a 2,000-step
fine-tune lifts Greece from `dT = -0.1069` native to `-0.0125` few-shot and Iran from
`-0.2760` to `-0.0634` (native and few-shot both in
[runs/multiregion_master.json](../runs/multiregion_master.json); the few-shot numbers are what
[runs/stats_hardening.json](../runs/stats_hardening.json) reports as those regions'
`temporal_variant`). Note the same file's *zero*-shot column is the more honest test and is
worse: Greece `-0.0395`, Iran `-0.1049`, and Chile actually **flips to a loss** zero-shot
(`+0.0343` native → `-0.0271`). The transfer story needs the fine-tune.

**Why it should not.** `b` differs by regime and `b` sets `beta`, which sets the productivity
ceiling `a < beta`. `p` correlates with heat flow, so kernel *shape* genuinely differs
between the cold Zagros and the hot Aegean. Background rate per unit area differs by orders
of magnitude, and the repo concedes it "still recomputes per-region normalization and a
train-era smoothed-seismicity background map" ([REPRODUCE.md](../REPRODUCE.md)). Regime-specific
physics has no analogue elsewhere: slab seismicity at 400 km (invisible to a 2-D catalog),
swarm migration fronts, megathrust supercycles. And the sharpest counter-evidence is the
repo's own caveat ([REPRODUCE.md:152-155](../REPRODUCE.md)) that the pooled model "is one shared
deployment checkpoint with no per-region weight fitting after pooling; **it is not
leave-one-region-out zero-shot transfer** because each region's training window participates
in the pooled pre-training run."

**What the evidence actually supports** ([runs/stats_hardening.json](../runs/stats_hardening.json),
`family_dT_holm`):

| region | regime | m_c | `dT` | 95% CI | raw p | Holm p | sig. |
|---|---|---|---|---|---|---|---|
| California | transform | 2.5 | +0.0533 | [+0.0402, +0.0674] | 0.0005 | 0.003 | yes |
| Italy | extension | 2.5 | +0.0712 | [+0.0496, +0.0969] | 0.0005 | 0.003 | yes |
| Chile | subduction | 4.0 | +0.0343 | [+0.0069, +0.0651] | 0.009 | 0.036 | yes |
| Japan | subduction | 4.0 | −0.0139 | [−0.0319, +0.0049] | 0.137 | 0.274 | no |
| Iran | collision | 4.0 | −0.0634 (few-shot) | [−0.1335, +0.0112] | 0.091 | 0.273 | no |
| Greece | extension | 4.0 | −0.0125 (few-shot) | [−0.0642, +0.0417] | 0.648 | 0.648 | no |

The pattern is **not regime**; the manuscript's reading, which I would defend, is **catalog
density**: the two largest and most significant wins are the two lowest-`m_c` catalogs
(California and Italy, both `m_c = 2.5`), and every one of the five California catalogs
(`m_c` 0.6–2.5) is a positive `dT`. The regime framing is a *coverage* claim — the win is not
confined to transform faulting — not a claim that regime is the operative variable.

**But do not overstate density as a monotone predictor, because the artifacts do not support
that.** Among the four `m_c = 4.0` regions the ordering is *not* training-set size: Japan has
the **most** training events above `mcut` of the four (19,929, versus Chile's 11,820,
[runs/completeness.json](../runs/completeness.json)) and is the one that does not win, while
Chile with 40% fewer events does. So density-as-event-count fails on exactly this pair. The
manuscript's own explanation is a different one — Japan's test window is dominated by the
Tōhoku sequence, ETAS's best case (§12.2) — and the honest summary is: `m_c` separates the
dense wins from the sparse losses cleanly; *within* the sparse group the residual ordering is
explained by sequence composition, not by how much data there is.

---

## 12. The catalogs FlowQuake actually uses

### 12.1 The five EarthquakeNPP California catalogs

Read from the config YAMLs in [configs/](../configs/), which are what the training scripts
consume:

| catalog | source | `mcut` | aux | train | val | test | config |
|---|---|---|---|---|---|---|---|
| **ComCat_25** | USGS ComCat, RELM/CSEP polygon | **2.5** | 1971-01-01 | 1981-01-01 | 1998-01-01 | 2007-01-01 → 2020-01-17 | [configs/n1_density.yaml](../configs/n1_density.yaml) |
| **WHITE_06** | White et al. (2019) San Jacinto | **0.6** | 2008-01-01 | 2009-01-01 | 2015-01-01 | 2017-01-01 → 2021-01-01 | [configs/WHITE_06_n1.yaml](../configs/WHITE_06_n1.yaml) |
| **SanJac_10** | QTM (Ross et al. 2019), San Jacinto | **1.0** | 2008-01-01 | 2009-01-01 | 2014-06-01 | 2016-01-01 → 2018-01-01 | [configs/SanJac_10_n1.yaml](../configs/SanJac_10_n1.yaml) |
| **SaltonSea_10** | QTM (Ross et al. 2019), Salton Sea | **1.0** | 2008-01-01 | 2009-01-01 | 2014-06-01 | 2016-01-01 → 2018-01-01 | [configs/SaltonSea_10_n1.yaml](../configs/SaltonSea_10_n1.yaml) |
| **SCEDC_20** | SCEDC standard southern California | **2.0** | 1981-01-01 | 1985-01-01 | 2011-01-01 | 2014-01-01 → 2020-01-01 | [configs/SCEDC_20_n1.yaml](../configs/SCEDC_20_n1.yaml) |

`ComCat_25` is the headline: 92,263 events ≥ M2.5 from 1971 to 2020-01-17, of which
**21,889** are in the test window ([README.md](../README.md); `n` in
[runs/stats_hardening.json](../runs/stats_hardening.json)).

**The suite is a completeness ladder** — `m_c` from 0.6 to 2.5 in one tectonic regime,
designed to isolate catalog density. That is why the density-dependence reading of §11 is
well supported. 3-seed results ([runs/fullsuite_summary.json](../runs/fullsuite_summary.json)):

| catalog | `m_c` | FQ `tll` | ETAS `tll` | Δ | FQ `sll` | ETAS `sll` |
|---|---|---|---|---|---|---|
| ComCat_25 | 2.5 | 1.4868 | 1.4343 | +0.0525 | −9.0589 | −8.6898 |
| WHITE_06 | 0.6 | 2.0669 | 2.0211 | +0.0458 | −4.7259 | −4.2611 |
| SanJac_10 | 1.0 | 1.1610 | 1.1325 | +0.0284 | −5.9233 | −5.3981 |
| SaltonSea_10 | 1.0 | 2.4337 | 2.3320 | +0.1017 | −2.6375 | −2.3151 |
| SCEDC_20 | 2.0 | 2.6194 | 2.5410 | +0.0784 | −7.8483 | −7.5342 |

Read the `sll` column: **ETAS wins spatially on all five**, by 0.31 to 0.53 nats. Read its
*scale* too: −2.64 (Salton Sea) vs −9.06 (ComCat) is an **area** difference, not a quality
difference — `sll` is `log f_s` in `log(1/km²)`, and Salton Sea is a tiny box. Comparing
`sll` across catalogs is meaningless; only within-catalog paired differences mean anything.

### 12.2 The six agency-built regions

Built by [scripts/build_region.py](../scripts/build_region.py) from FDSN services
([REPRODUCE.md](../REPRODUCE.md) §1):

| region | source | bbox (lat, lon) | dl `m` | analysis `mcut` | regime |
|---|---|---|---|---|---|
| Japan | ISC reviewed Bulletin | 22–46 N, 122–150 E | 3.5 | 4.0 | subduction |
| Chile | ISC | −40..−17, −76..−66 | 3.5 | 4.0 | subduction |
| Greece | ISC | 34–42 N, 19–29 E | 3.5 | 4.0 | extension |
| Iran | ISC | 25–40 N, 44–63 E | 3.5 | 4.0 | collision |
| Italy | INGV | 36–47 N, 6–19 E | 2.0 | 2.5 | extension |
| (New Zealand) | GeoNet | — | 3.5 | 3.5 | **omitted** — ETAS inversion prohibitively slow at catalog scale ([REPRODUCE.md](../REPRODUCE.md)) |

All use `train_start: 1992-01-01` (Italy 1994-01-01), `val_start: 2009-01-01`,
`test_start: 2011-01-01`, `test_end: 2020-01-01`. Choosing **one source per completeness
group** (ISC for mc-4.0, INGV for Italy) is deliberate and defensible — it removes
inter-agency heterogeneity as a confounder ([scripts/build_region.py:1-17](../scripts/build_region.py#L1-L17)).
It does not remove *intra*-agency magnitude-type heterogeneity (§2.3).

**Japan's test window begins 2011-01-01**, ten weeks before the 2011-03-11 M9.0 Tohoku
earthquake, so its entire test set is dominated by the most textbook-Omori aftershock sequence
in the instrumental record — ETAS's best case, as [MANUSCRIPT.md](../MANUSCRIPT.md) §4.5 notes.
That is a fair explanation of `dT = -0.0139`, and a reminder that with one sequence dominating,
the effective sample size is far below n = 14,886.

### 12.3 Salton Sea, and why it matters for ETAS

`SaltonSea_10` is QTM-derived at `m_c = 1.0` over the Salton Sea / Brawley seismic zone at
the southern end of the San Andreas system. Three things make it special: it is a **step-over**
between the San Andreas and Imperial faults, with distributed deformation, extremely high
heat flow and active hydrothermal circulation; it hosts the **Salton Sea Geothermal Field**,
whose production and injection are documented to modulate local seismicity — Brodsky & Lajoie
(2013), "Anthropogenic seismicity rates and operational parameters at the Salton Sea
geothermal field", *Science* 341, 543–546, found that **net** fluid volume (extracted minus
injected) best tracks the background rate once aftershocks are removed; and
it is **the swarmiest place in California** — the 2005 and 2016 Bombay Beach swarms and the
2012 Brawley swarm, hundreds of events over days, migrating at ~km/day, with **no dominant
mainshock**. The 2016 swarm (late September 2016) falls inside the test window.

**Why it matters:** ETAS assumes every event is background (constant-rate Poisson) or an
Omori-decaying offspring. A fluid-driven swarm is neither — its rate rises then falls, its
extent expands as `sqrt(t)`, and its "productivity" bears no relation to its largest
magnitude. ETAS can only represent it by pretending the first event triggered everything.

FlowQuake's temporal margin is largest here (+0.1017). The tempting story — "FlowQuake wins
where ETAS's form is wrongest" — is a hypothesis I like and **cannot support from the
committed artifacts**: the repo does not stratify the Salton Sea gain by swarm episode, and
the competing explanation (Salton Sea is also the densest and most temporally clustered
catalog, the best case for a flexible temporal density) is at least as plausible. Offer it
as a hypothesis with a proposed test — stratify `dT` by whether the event falls inside a
detected swarm — and you look better than if you assert it.

---

## Worked example A — a b-value by hand, with and without the binning correction

Twenty magnitudes above a reported `m_c = 2.5`, on a `dm = 0.1` grid:

```
2.5 2.5 2.5 2.5 2.6 2.6 2.6 2.7 2.7 2.8 2.8 2.9 3.0 3.1 3.2 3.3 3.5 3.7 4.0 4.4
```

**1 — the sufficient statistic.**

```
sum = 4(2.5) + 3(2.6) + 2(2.7) + 2(2.8) + 2.9+3.0+3.1+3.2+3.3+3.5+3.7+4.0+4.4
    = 10.0 + 7.8 + 5.4 + 5.6 + 27.1  =  59.9
n = 20 ,   mean(m) = 59.9/20 = 2.995
```

**2 — uncorrected Aki MLE.**

```
b_hat = 0.4342945 / (2.995 - 2.50) = 0.4342945 / 0.495 = 0.8774
```

**3 — with the Utsu/Bender half-bin correction** (`m_c - dm/2 = 2.45`):

```
b_hat = 0.4342945 / (2.995 - 2.45) = 0.4342945 / 0.545 = 0.7969
```

The correction lowers `b` by **0.081, about 9%**, and always lowers it (the denominator only
grows). At the sample sizes typical of a spatial b-value map (n = 50–200 per cell),
forgetting it puts a systematic ~10% ridge into every map you produce.

**4 — is that different from 1?** Aki: `sigma_b = b/sqrt(n) = 0.7969/4.4721 = 0.1782`.
Shi & Bolt, using `sum_i (m_i - mean)² = 5.5895` (the four 2.5s give `4*0.495² = 0.9801`;
the single 4.4 gives `1.405² = 1.9740`):

```
sigma_b = 2.30 * 0.7969² * sqrt( 5.5895 / (20*19) )
        = 2.30 * 0.6350 * 0.121281  =  0.1771
```

The two agree to 0.6%, as §5.3 says they should when the exponential model holds. So
**b = 0.80 ± 0.18**: with 20 events you cannot distinguish `b = 0.8` from `b = 1.0`, let
alone claim a regional difference. Ask anyone showing you a b-value map how many events are
in each cell.

**5 — comparing two b-values with the exact F-test (§5.3).** Region A: `b_A = 0.79`,
`n_A = 200`. Region B: `b_B = 1.05`, `n_B = 180`. Under `H0`,

```
R = b_B/b_A = 1.05/0.79 = 1.3291   ~   F(2*200, 2*180) = F(400, 360)     under H0

exact:   two-sided tail of F(400, 360) at 1.3291      ->   p = 0.00588
normal:  log R      = 0.28451
         Var(log R) ≈ 1/n_A + 1/n_B = 0.005000 + 0.005556 = 0.010556 -> sd = 0.10274
         z          = 0.28451 / 0.10274 = 2.769       ->   p = 0.00562
```

The exact and asymptotic answers agree to 5% of each other at `n ≈ 200`, which is the point:
use the F-test when `n` is small and either when it is not. Different at the 1% level —
*if* the magnitudes are independent draws. They are not (aftershocks cluster), so the
effective sample size is below `n`, the true p is larger, and 0.006 is a floor on your
p-value, not an estimate of it.

```python
import numpy as np
m = np.array([2.5]*4 + [2.6]*3 + [2.7]*2 + [2.8]*2 + [2.9,3.0,3.1,3.2,3.3,3.5,3.7,4.0,4.4])
mc, dm = 2.5, 0.1
print(np.log10(np.e)/(m.mean()-mc), np.log10(np.e)/(m.mean()-(mc-dm/2)))   # 0.8774 0.7969
b = np.log10(np.e)/(m.mean()-(mc-dm/2))
print(b/np.sqrt(len(m)), 2.30*b**2*np.sqrt(((m-m.mean())**2).sum()/(len(m)*(len(m)-1))))
# step 5: exact F-test vs the normal approximation
from scipy import stats
R = 1.05/0.79
print(2*min(stats.f.cdf(R,400,360), stats.f.sf(R,400,360)),          # 0.005885 exact
      2*stats.norm.sf(np.log(R)/np.sqrt(1/200 + 1/180)))             # 0.005619 normal
```

---

## Worked example B — how many M ≥ 4 aftershocks does an M6.5 produce?

Use the **Reasenberg & Jones (1989, *Science* 243, 1173–1176) generic California model** —
the standard operational aftershock forecast, combining Omori–Utsu with GR and productivity:

```
lambda(t, M) = 10^( a_RJ + b*(M_main - M) ) / (t + c)^p        [events/day, magnitude >= M]
```

Generic California: `a_RJ = -1.67`, `b = 0.91`, `p = 1.08`, `c = 0.05` d. (Generic values;
regional and sequence-specific fits differ substantially — say so.) Take `M_main = 6.5`,
`M = 4.0`, `t ∈ [0, 365]` days.

**1 — magnitude/productivity factor.**

```
10^( -1.67 + 0.91*(6.5 - 4.0) ) = 10^( -1.67 + 2.275 ) = 10^0.605 = 4.027
```

(`10^0.6 = 3.9811`, times `10^0.005 = 1.01158`.)

**2 — time integral.** With `p = 1.08`, `1 - p = -0.08`:

```
     365                        (0.05)^(-0.08) - (365.05)^(-0.08)
I =   ∫ (t + 0.05)^(-1.08) dt = ---------------------------------
      0                                      0.08

(0.05)^(-0.08)   = exp(-0.08 * ln 0.05)   = exp( 0.23966) = 1.27082
(365.05)^(-0.08) = exp(-0.08 * ln 365.05) = exp(-0.47201) = 0.62375

I = (1.27082 - 0.62375)/0.08 = 0.64706/0.08 = 8.0883
```

**3 — multiply.** `E[N(M ≥ 4.0, first year)] = 4.0272 * 8.0883 = 32.57` — **about 33
aftershocks of M ≥ 4 in the first year.**

Cross-check via GR: the same formula at `M = 2.5` gives
`10^(-1.67 + 0.91*4) * 8.0883 = 93.33 * 8.0883 = 754.8` events, and
`754.8 * 10^(-0.91*1.5) = 754.8 * 0.043153 = 32.57` ✓ — consistent, because the model *is* GR
in the magnitude direction.

**4 — the largest aftershock, and the tension with Båth's law.** Assume the count above `M`
is Poisson with mean `N(M)`; then `P(none ≥ M) = exp(-N(M))`, so the **median** largest
aftershock solves `N(M) = ln 2 = 0.693`:

```
10^( -1.67 + 0.91*(6.5 - M) ) * 8.0883 = 0.6931
10^( -1.67 + 0.91*(6.5 - M) ) = 0.085698
-1.67 + 0.91*(6.5 - M) = -1.06703   ->   6.5 - M = 0.6626   ->   M = 5.837
```

A **median Båth deficit of 0.66**, not the canonical 1.2. Do not paper over the gap; it is
instructive. R&J's generic `a_RJ` counts *all* aftershocks in a fixed window including
secondary ones, whereas Båth's law was measured within particular space–time windows under
mainshock-selection rules that condition on the largest aftershock being smaller; and 1.2 is
an *average of observed deficits*, not the median of a Poisson-max under a fitted model.
Both numbers are regionally variable and the generic R&J parameters are known to be generous
for California. Right posture: quote 1.2 as the empirical rule, know it is largely GR +
productivity + selection, and be able to show that a specific model implies a specific
deficit which need not equal 1.2.

```python
import numpy as np
from scipy.optimize import brentq
a, b, p, c = -1.67, 0.91, 1.08, 0.05
I = ((0+c)**(1-p) - (365+c)**(1-p)) / (p-1)
print(10**(a + b*(6.5-4.0)) * I)                                   # 32.57
print(brentq(lambda M: 10**(a + b*(6.5-M))*I - np.log(2), 3, 6.5)) # 5.8374
```

---

## Worked example C — moment to magnitude, both conventions

The 2019 Ridgecrest mainshock had `M0 ≈ 5.3e19 N·m` (GCMT; order of magnitude only — verify
the exact value before quoting).

```
log10(5.3e19) = 19 + 0.72428 = 19.72428

IASPEI:  Mw = (2/3)(19.72428 - 9.1)  = (2/3)(10.62428) = 7.083
HK79:    Mw = (2/3)(19.72428 + 7) - 10.7                = 7.116     (i.e. +0.033)
```

The catalog value is 7.1; either convention rounds to it. **State which one you used.** And
note that 0.033, fed through an ETAS productivity term with `a = 2.3`, is
`exp(2.3*0.033) = 1.079` — an 8% productivity difference. Not fatal; not nothing.

---

## How this shows up in FlowQuake

Cross-references only; [STACK.md](../STACK.md) is the code walkthrough.

| concept | where | note |
|---|---|---|
| Gutenberg–Richter | [flowquake/heads.py:157-180](../flowquake/heads.py#L157-L180) | `m - m_c ~ Exp(beta(cond))`; `beta` init 2.0, i.e. `b ≈ 0.869`. History-dependent `beta` is a real departure from ETAS. |
| Binning correction | [flowquake/heads.py:173](../flowquake/heads.py#L173) | `+0.005`, hardcoded for all eleven catalogs; §5.5 — right form, and wrong for at least one catalog family whichever way ComCat's precision resolves. |
| Omori–Utsu | [flowquake/neural_etas.py:78-83](../flowquake/neural_etas.py#L78-L83); [flowquake/data.py:154-165](../flowquake/data.py#L154-L165) | `p = 1 + omega`; `RECENCY_LAGS = (1,2,4,8,16,32,64)` gives the Omori argument at seven scales. |
| Utsu productivity | same lines | Learned multiplicative modulation `g(m_j, Delta_t)` in the neural-ETAS head. |
| Power-law spatial decay | [flowquake/heads.py:104-105](../flowquake/heads.py#L104-L105) | `f(r) = (q-1)/(pi d²)(1 + r²/d²)^{-q}` — heavy-tailed by design. |
| Anisotropy / strike elongation | [flowquake/heads.py:89-94](../flowquake/heads.py#L89-L94) | Area-preserving ellipse `(d*rho, d/rho)` at angle `theta`; §9. |
| Magnitude-dependent rupture size | `d_j = d*exp(gamma(m_j - m_c))`; learned `d` per component | Wells–Coppersmith scaling in kernel form. |
| Long-lived triggering | [flowquake/data.py:49-73](../flowquake/data.py#L49-L73) | `BIG_M = 16` largest `M ≥ 4.5` events of a trailing 730-day window — the Omori tail a last-64 window misses. |
| Smoothed-seismicity background | [flowquake/data.py:120-151](../flowquake/data.py#L120-L151) | HKJ variable bandwidth, 12 buckets, 3% uniform floor; §10. |
| Completeness | `mcut` in every config; [scripts/check_completeness.py](../scripts/check_completeness.py) → [runs/completeness.json](../runs/completeness.json) | MAXC + 0.2, per era, four ISC regions only; §4.5. |
| Magnitude-type heterogeneity | [scripts/build_region.py](../scripts/build_region.py), [scripts/build_italy_mw.py](../scripts/build_italy_mw.py), [scripts/mw_robustness.py](../scripts/mw_robustness.py), [scripts/mag_robustness.py](../scripts/mag_robustness.py) | §2.3–2.4. |
| Non-tectonic filtering | [scripts/build_region.py:45](../scripts/build_region.py#L45) | Removes only agency-labelled non-earthquakes; §3. |
| Depth | nowhere | Discarded at build time; §3. |
| Declustering | nowhere | Neither model declusters. Correct choice; §8. |

---

## Common misconceptions

1. **"Magnitude is a measurement."** *Actually:* it is a model output — ML is an amplitude
   reading through a 1930s southern-California distance correction; Mw is a waveform
   inversion. Different scales read different parts of the source spectrum, so they are not
   noisy versions of one another. *Why it matters:* it is why `Mw = 1.456*Md - 1.472` has a
   slope ≠ 1, and why converting scales changes `b` by that slope (§2.2).

2. **"The Richter scale goes up to 10."** *Actually:* ML saturates near 6.5 and nothing has a
   "Richter magnitude" of 9; the upper bound on Mw is set by available fault area.
   *Why it matters:* an interviewer using "Richter" as a synonym is testing whether you will
   correct them.

3. **"m_c is a property of the region."** *Actually:* it is a property of region × time ×
   network × noise × seismicity rate — dropping a unit on a network upgrade, jumping three
   units for hours after an M7 (§4.2). *Why it matters:* the repo's `mcut` is one number per
   catalog for a 30–50 year window: a conservative approximation, not a fact.

4. **"The c-value is physical."** *Actually:* fitted `c` mostly encodes how incomplete your
   catalog is in the first hours; higher-resolution catalogs give smaller `c` (§6.2).
   Rate-and-state *does* predict `c = t_a*exp(-Delta_tau/(A sigma))`, but the fitted value is
   dominated by the artifact. *Why it matters:* comparing `c` across studies with different
   detection thresholds compares detection thresholds.

5. **"Aftershocks are a different kind of event."** *Actually:* labels come from a windowing
   rule applied after the fact and flip when a later larger event occurs (§8). ETAS and
   FlowQuake treat every event identically. *Why it matters:* it dissolves the "foreshock
   prediction" question — foreshocks are only foreshocks in hindsight.

6. **"Declustering gives the 'real' background rate."** *Actually:* it gives the background
   according to your algorithm; Gardner–Knopoff, Reasenberg and stochastic declustering give
   materially different catalogs with different b-values (§8). *Why it matters:* national
   seismic hazard maps rest on declustered rates.

7. **"Omori is just a curve fit; decay should be exponential."** *Actually:* the power law is
   derivable from rate-and-state with `p = 1` exactly (§6.3) and is also what superposing
   exponentials over a scale-free timescale distribution gives (§6.4). An exponential implies
   a single characteristic relaxation time, which a heterogeneous fault system lacks.
   *Why it matters:* it is *why* a fixed-window encoder (DeepSTPP's 20 events) is structurally
   fatal — the tail carries real mass.

8. **"b = 1 everywhere, so b is not interesting."** *Actually:* `b` does vary (lower on
   megathrusts, higher in geothermal areas), but much *reported* variation is scale
   heterogeneity, incompleteness and small samples — a 20-event estimate has
   `sigma_b ≈ 0.18` (example A). Kagan's `b ≡ 1` position is defensible.
   *Why it matters:* it is the correct reply to anyone reading a b difference as physics
   without checking `n`.

9. **"FlowQuake beats ETAS."** *Actually:* the *temporal* head beats ETAS on dense catalogs;
   the *production spatial* head loses on all five California catalogs
   ([runs/fullsuite_summary.json](../runs/fullsuite_summary.json)); the total win uses a
   **separate** neural-ETAS head **initialised from that region's own ETAS inversion**.
   *Why it matters:* [README.md](../README.md) itself frames the composite as "an *upgrade of a
   deployed ETAS system*, not an inversion-free replacement for one," and
   [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) puts it at rung 3 of 5. Anyone who
   has read the repo will check.

10. **"A uniform spatial background is a reasonable null."** *Actually:* density varies over
    five or six orders of magnitude; uniform scores ~5 nats/event worse than ETAS (Poisson
    `sll` −13.77 vs ETAS −8.69, [README.md](../README.md)). Almost all spatial skill is knowing
    where the faults are. *Why it matters:* it explains why `adaptive_bg_grid` is load-bearing
    and why the repo lists it as a preprocessing dependency it cannot yet remove.

11. **"Deep earthquakes are shallow ones, further down."** *Actually:* below ~60 km rock is
    too ductile for brittle stick–slip; intermediate and deep events need other mechanisms
    (dehydration embrittlement, transformational faulting) and are far less productive of
    aftershocks. *Why it matters:* the Japan catalog mixes deep slab and crustal events in a
    2-D model (§3), a real specification error for that region.

---

## Questions a professor will ask

**Q1. Define seismic moment and derive moment magnitude. State your units convention.**
`M0 = mu_shear * A * D` — shear modulus (30 GPa crust), rupture area, average slip. It is the
scalar amplitude of the equivalent double-couple in the elastodynamic representation theorem,
which is why it is recoverable from long-period waveforms without knowing `A` and `D`
separately. `Mw = (2/3)(log10 M0 - 9.1)` with `M0` in N·m — IASPEI. Hanks & Kanamori's
original is `(2/3) log10 M0 - 10.7` in dyne·cm, which converts to
`(2/3)(log10 M0[N·m] - 9.05)`; the two differ by 0.033 units. The 2/3 comes from
`log10 E = 1.5 M + const` combined with `E ∝ M0`.

**Q2. Why does ML saturate and Mw not?** ML reads the spectrum at 1–10 Hz. The Brune
omega-squared spectrum is flat below `f_c` and falls as `f^-2` above, and `f_c ∝ M0^(-1/3)`
under constant stress drop. Once `f_c` drops below the band, amplitude scales as
`M0*f_c² ∝ M0^(1/3)` instead of `M0`, so the magnitude compresses and then flattens. Mw uses
the zero-frequency level, which never saturates. ML saturates near 6.5, mb near 6–6.5, Ms
near 8–8.5 (§2.1) — hence Ms ≈ 8.5 for the Mw 9.5 Chile 1960 earthquake.

**Q3. Derive the Aki MLE and its standard error.** `u_i = m_i - m_c ~ Exp(beta)`;
`L = n log beta - beta sum u_i`; `dL/dbeta = 0` gives `beta_hat = 1/ubar`, so
`b_hat = log10(e)/(mean(m) - m_c)`. For the error: `2 beta S ~ chi²(2n)`, so
`beta/beta_hat = X/(2n)` with `X ~ chi²(2n)`; `Var(log X) ≈ psi'(n) ≈ 1/n`, giving
`sd(log b_hat) ≈ 1/sqrt(n)` and `sd(b_hat) ≈ b/sqrt(n)`. The same identity gives the exact
two-sample test `b_hat_2/b_hat_1 ~ F(2n_1, 2n_2)`.

**Q4. What does the `+0.005` in `heads.py` correct?** Reported `m` means true magnitude in
`[m - dm/2, m + dm/2)`, so the reported threshold corresponds to a true threshold
`m_c - dm/2`. Expanding `log P(m) - log dm` gives
`log beta - beta[(m - m_c) + dm/2] + (beta dm)²/24` — a half-bin *shift*, exactly the code's
form. **But `0.005 = dm/2` implies `dm = 0.01`, while STACK.md and MANUSCRIPT.md justify it as
a correction for 0.1-unit discretization.** Better: `0.005` is a *hardcoded literal* shared by
all eleven catalogs, and ISC/INGV are certainly on a 0.1 grid, so it is wrong for **some**
catalog whatever ComCat's precision turns out to be. On a 0.1 grid it biases fitted `beta`
11.6% high. The headline is unaffected — `nll = -(tll + sll)` excludes `mll`, and at
`h_bottleneck = 0` the GR head shares no trainable parameter with the other heads — but
simulated magnitudes and therefore the CSEP M-test are affected. §5.5 has the full argument.

**Q5. Why a power law and not an exponential?** Two answers. (a) Dieterich (1994): a
rate-and-state population under a step stress gives
`R(t) = r/{[e^{-Delta_tau/(A sigma)} - 1]e^{-t/t_a} + 1}` with `t_a = A sigma/tau_dot_r`; for
`t << t_a` this reduces exactly to `K/(t + c)` with `p = 1` and
`c = t_a e^{-Delta_tau/(A sigma)}`. (b) Model-free: superposing `lam e^{-lam t}` over
`p(lam) ∝ 1/lam` gives `∝ 1/t` between cutoffs. Both say a power law is the signature of no
characteristic timescale, an exponential of exactly one.

**Q6. Why does ETAS need `a < b ln 10`?** Expected direct offspring of a random event is
`n_branch = k0*I*beta/(beta - a)`, `I` the finite Omori-with-taper time integral; the
magnitude integral `∫ e^{(a - beta)u} du` diverges unless `a < beta = b ln 10 ≈ 2.30`. Fitted
`a` lands at 2.0–2.3, right at the edge — which is why the ETAS likelihood is ill-conditioned
in `(a, k0)` and EM inversion takes hours.

**Q7. Your regions have different b-values. Does that break the transfer claim?** The
committed b-values range 0.76–1.13, but they are computed at per-era MAXC thresholds, on
undeclustered catalogs containing M8+ sequences whose STAI biases `b` down, with an
independence assumption aftershock clustering violates — so the intervals are too narrow.
More importantly, FlowQuake's transferable component is the *kernel*, not the magnitude
distribution: `SAFE_TOKEN_DIMS` excludes absolute coordinates and `beta(cond)` is refit per
region anyway. The claim I would defend is §11's: the wins track catalog *density*, not
regime.

**Q8. Your Japan result is negative. Explain it.** `dT = -0.0139` [−0.0319, +0.0049], Holm
p = 0.274, TOST-equivalent at ±0.05 ([runs/stats_hardening.json](../runs/stats_hardening.json)).
Three reasons, in descending confidence: (i) `m_c = 4.0` makes it the sparse case where ETAS's
parametric form is near-optimal — the density-dependence pattern of §11; (ii) the test window
starts 2011-01-01 and is dominated by the M9.0 Tohoku aftershock sequence, ETAS's best case,
so the effective sample size is far below n = 14,886; (iii) the catalog contains slab
seismicity to ~600 km projected onto a plane (§3), a specification error I cannot fix without
adding depth.

**Q9 (hostile). Your temporal win is an m_c artifact. If completeness improved over the
record, gaps shorten systematically and your flexible model learns an instrumentation trend
ETAS's rigid form cannot fit. Refute that.** I cannot refute it fully, and the mechanism is
real: under `b = 1`, a 0.2-unit m_c improvement makes the early catalog 37% sparser at a
fixed threshold and raises the apparent rate 1.59× over the record (§4.3), and chronological
splits point that trend into the test window. Three partial defences. (i) Thresholds are
conservative — 2.5 for ComCat when modern southern-California m_c is nearer 1.8. (ii) The
gain is positive in 85% of 180-day windows across the test decade
([MANUSCRIPT.md](../MANUSCRIPT.md) §4.1), not concentrated late. (iii) The frozen model
replicates out-of-time on 2020–2026 with essentially constant m_c (`dT = +0.0574`,
[runs/total_win.json](../runs/total_win.json)). (ii) is weakest — a slow drift gives a gain
positive throughout, just larger later. The clean experiment is to re-run at a threshold
where m_c is provably flat across the whole record (say ComCat at 3.5) with a matched ETAS.
The repo has a `comcat_mc30` run that *loses* (`dT = -0.0792`,
[runs/mw_robustness.json](../runs/mw_robustness.json)) — but it is *retrained* on sparse data,
so it tests data efficiency, not the m_c artifact. The missing control is the dense-trained
model on a high-m_c subset with a matched ETAS; the repo has half of it
(`comcat_mc25_production_on_Mge3`, `dT = +0.074`).

**Q10 (hostile). Italy collapses under Mw homogenization — +0.071 to −0.253 — and your own
script calls that a density effect. It is not. Defend yourself.** You are right, and §2.4
walks the artifact's own numbers: the density-matched control ties on 9,167 training events
while the Mw run loses 0.2532 on **10,391** — more data, much worse — and ETAS's own `tll`
*rose* (1.2513 → 1.2903) where FlowQuake's fell. Thinning cannot explain that.
[MANUSCRIPT.md](../MANUSCRIPT.md) §4.5 already states it correctly ("a residual sensitivity to
the heavy, type-dependent Md→Mw compression itself"); the stale text is the docstring of
[scripts/mw_robustness.py](../scripts/mw_robustness.py) and the `interpretation` field of
[runs/mw_robustness.json](../runs/mw_robustness.json), which should be fixed. My hypothesis
(§2.4): the type-dependent stretch — Md ×1.456, ML ×1.0 — plus the 2005-04-16 flip distorts
exactly the raw-magnitude features the neural heads consume, while ETAS absorbs it by refitting
one productivity exponent. The correct claim, which the manuscript makes, is that Italy is a
**dense-native-catalogue** result, not claimed under Mw homogenization.

**Q11 (hostile). You claim completeness is verified across eras. Show me.** It is verified for
four of six regions. [scripts/check_completeness.py:17](../scripts/check_completeness.py#L17)
sets `REGIONS` to Japan, Chile, Greece and Iran only, and
[runs/completeness.json](../runs/completeness.json) has exactly those four keys. Italy is not
checked
despite [REPRODUCE.md](../REPRODUCE.md) §1 claiming the script "confirms mc 4.0 (ISC) / 2.5
(Italy)". No California catalog is checked. And "stable" is generous: `m_c` moves by up to
0.30 (Chile 3.95 → 3.65); what is verified is that the worse era, rounded up to 0.5, lands at
4.0. The estimate also pools an entire region over space and includes aftershock sequences,
both of which bias MAXC. The fix is cheap — add Italy and the five California catalogs to
`REGIONS`, and report m_c on a spatial grid — and I would run it before submission.

**Q12 (hostile). No depth, and half your regions are subduction zones. Why believe Japan or
Chile at all?** The catalogs genuinely have no depth:
[scripts/build_region.py:75](../scripts/build_region.py#L75) parses ISC fields 0,1,2,3,10,9,13
and drops field 4 (`Depth`). For Japan that projects the Ryukyu arc, Nankai and 600 km-deep
slab seismicity onto one plane, so a slab event and a crustal event above it are colocated.
A real specification error; I would not defend it as anything else. What rescues the
*comparison* is that ETAS is scored on the same 2-D catalog and is equally blind, so paired
`dT`/`dS` remain a fair head-to-head; what it damages is any claim the model learned
subduction *physics*. Honest scope: a benchmark-conformant 2-D marked point process, with
depth as future work — either a crustal-only filter or a genuinely 3-D kernel.

**Q13. What is STAI and who does it hurt more, you or ETAS?** After a mainshock, `m_c` jumps
to roughly `M - 4` and decays back over hours to weeks, because coda and overlapping waveforms
swamp detection. It biases `c` up, `b` down in sequences, and productivity down. It hurts
**both**: the benchmark's ETAS uses a fixed `mc`, not the time-varying `mc(t)` of Mizrahi et
al. (2021), and FlowQuake trains on the same censored catalog. So the head-to-head is fair;
the absolute calibration of both is off in the first hours of every large sequence. If
pressed on which is more hurt: FlowQuake, plausibly, since a flexible model can fit the
censoring pattern as if it were physics — but I have no measurement and would not assert it.

**Q14. Why is a Gaussian mixture the wrong spatial model?** Aftershock distances are power-law
distributed, and a Gaussian's `exp(-r²/2sigma²)` tail is catastrophically thin against them.
Compare the *shape* terms (normalizers dropped, both kernels at scale length `sigma = d`) for
an event five scale lengths out: the Gaussian pays `r²/2sigma² = 12.5` nats, the heavy-tailed
`(1 + r²/d²)^{-q}` pays `q*log(1 + 25) = 5.86` nats at `q = 1.8`. Nearly 7 nats *per event*,
and the gap grows without bound in `r` — logarithmically for the power law, quadratically for
the Gaussian. Over thousands of test events with a genuine power-law tail that is
unrecoverable, which is why
[flowquake/heads.py:104-105](../flowquake/heads.py#L104-L105) uses
`f(r) = (q-1)/(pi d²)(1 + r²/d²)^{-q}` despite the class name.

**Q15. Justify the anisotropic kernel physically. Then tell me whether it helped.**
Aftershocks decorate a rupture strip of length `L(m)` (`log10 L ≈ 0.6M - 2.5`: ~20–25 km at M6.5)
oriented along strike, and the Coulomb stress-change field around a dislocation is a
strike-elongated four-lobed pattern (King, Stein & Lin 1994). FlowQuake gives each component
an area-preserving ellipse `(d*rho, d/rho)` at angle `theta`
([flowquake/heads.py:89-94](../flowquake/heads.py#L89-L94)); area preservation means the same
normalizer works, so elongation is free. **Did it help? Not enough** — the production spatial
head still loses to ETAS on all five California catalogs. The manuscript's diagnosis (§4.4) is
that the residual is triggering *coverage*, not kernel shape — and quote it exactly, because
the numbers are conditional: **64%** of ComCat test events recur within 0.5 km of a prior
event, and **85% of those nearest priors** lie outside the model's last-64-event window. So
the head is being asked to place aftershocks of older moderate mainshocks it cannot see,
which ETAS captures by integrating triggering over the full history.

**Q16. Why is a uniform background so much worse, quantitatively?** Density varies by five to
six orders of magnitude across California. The benchmark's Poisson baseline scores
`sll = -13.7745` and ETAS `-8.690` ([README.md](../README.md)); inverting the baseline gives
`A = e^13.7745 = 9.6e5 km²`, i.e. it *is* the uniform model over the RELM polygon and nothing
else. So ~5.1 nats/event of spatial skill is "know where the faults are."
`adaptive_bg_grid` captures it with HKJ variable-bandwidth smoothing — `sigma_i` = distance to
the 6th nearest neighbour, 12 log-spaced bandwidth buckets, 3% uniform floor so no cell scores
`-inf`.

**Q17. Should aftershocks be removed before training?** No, and neither model removes them —
triggering *is* the signal in a Hawkes-type model. Declustering biases the magnitude
distribution, background rate and b-value, and different algorithms give materially different
answers from the same catalog. The one declustering-adjacent choice is `adaptive_bg_grid`,
which smooths the *undeclustered* training catalog, so the "background" over-weights places
with a big sequence during training. Frankel and HKJ make the same choice and argue it
forecasts better; I would say the same and flag it.

**Q18. Salton Sea is where you win biggest. Is that because ETAS cannot model swarms?** That
is my hypothesis and the committed artifacts cannot support it. Facts: `SaltonSea_10` is a QTM
catalog at `m_c = 1.0` over a geothermal step-over with documented fluid-driven swarms
including the 2016 Bombay Beach swarm inside the test window, and the temporal margin there is
the largest of the five (+0.1017). ETAS structurally cannot represent a swarm — no dominant
mainshock, rate rises then falls, extent grows as `sqrt(t)` under pore-pressure diffusion. But
the competing explanation is that Salton Sea is also the densest and most clustered catalog,
exactly where density-dependence says FlowQuake should win. The separating test is to stratify
`dT` by whether each event is inside a detected swarm episode; nobody has run it.

**Q19. What would a seismologist say is missing?** Four things, decreasing in importance:
(i) no prospective forecast — [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) says so and
makes it rung 4 of its ladder; (ii) no depth; (iii) no time-varying `m_c`, so neither model
handles STAI; (iv) no treatment of magnitude *uncertainty* — both models treat reported
magnitudes as exact, and ±0.2 fed through `exp(2.3*delta_m)` is a ~60% productivity error per
event.

**Q20. With six more months, what seismological work would you do first?** In order: (1) re-run
completeness on all eleven catalogs, spatially resolved, and re-run the headline at a
provably-flat-m_c threshold with a matched ETAS — this closes the biggest alternative
explanation (Q9); (2) fix the magnitude half-bin constant per catalog and re-run the CSEP
M-test (Q4); (3) add depth for the subduction regions or restrict to crustal seismicity, and
see whether Japan flips (Q12); (4) stratify the Salton Sea and Geysers gains by swarm episode
(Q18); (5) run the prospective forecast, since nothing else moves the claim past rung 3.

---

## Further reading

1. **Ogata (1988), *JASA* 83(401), 9–27** and **Ogata (1998), *Ann. Inst. Statist. Math.*
   50(2), 379–402** — the two papers that turned the three empirical laws into a likelihood,
   the second adding space and anisotropic kernels. [docs/03-etas.md](03-etas.md)
   descends entirely from them.
2. **Utsu, Ogata & Matsu'ura (1995), "The centenary of the Omori formula for a decay law of
   aftershock activity", *J. Phys. Earth* 43(1), 1–33** — definitive Omori–Utsu review:
   history, parameter ranges, every caveat about `c` and `p`.
3. **Dieterich (1994), *JGR* 99(B2), 2601–2618** — the derivation in §6.3; the single most
   useful citation for "why a power law?".
4. **Wiemer & Wyss (2000), *BSSA* 90(4), 859–869** and **Woessner & Wiemer (2005), *BSSA*
   95(2), 684–698** — the completeness methods of §4.4, including the MAXC + 0.2 correction
   the repo uses and the EMR method it does not.
5. **Helmstetter, Kagan & Jackson (2007), *SRL* 78(1), 78–86** — the adaptive-bandwidth model
   `adaptive_bg_grid` implements; read with **Frankel (1995), *SRL* 66(4), 8–21** for the
   fixed-bandwidth predecessor and the case for smoothing an undeclustered catalog.
6. **King, Stein & Lin (1994), *BSSA* 84(3), 935–953** — Coulomb stress transfer, the butterfly
   lobe geometry, and the physical case for anisotropic spatial kernels.
7. **Zhuang, Ogata & Vere-Jones (2002), *JASA* 97(458), 369–380** — the probabilistic view of
   what an "aftershock" is, and why the label is model-dependent.
8. **Mizrahi, Nandan & Wiemer (2021), *JGR Solid Earth* 126, e2021JB022379** — methods paper
   for the `etas` package used here, and the standard treatment of time-varying `m_c` (which
   the benchmark configuration does *not* enable).
9. **Stockman, Lawson & Werner (2026), *TMLR*, arXiv:2410.08226** — EarthquakeNPP: the five
   California catalogs, the splits, the ETAS baselines, and the finding that no neural point
   process beat ETAS. The contract FlowQuake is measured against.
10. **Ross, Trugman, Hauksson & Shearer (2019), *Science* 364, 767–771** and **White, Ben-Zion
    & Vernon (2019), *JGR Solid Earth* 124, 6908–6930** — the dense catalogs behind
    `SanJac_10`, `SaltonSea_10` and `WHITE_06`; read for what a template-matching catalog is
    and is not, and why its magnitudes and `m_c` are not homogeneous with the parent catalog.

Two more worth an afternoon each: **Hanks & Kanamori (1979), *JGR* 84, 2348–2350** (three
pages; the units convention of §1.3) and **Aki (1965), *Bull. Earthq. Res. Inst. Tokyo* 43,
237–239** (the MLE of §5.2 and its confidence limits, in four pages).

---

*Next: [docs/03-etas.md](03-etas.md) assembles the three empirical laws of §5–§7 into the
conditional intensity this repository is trying to beat.*
