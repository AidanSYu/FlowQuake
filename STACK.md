# FlowQuake, end to end

A first-principles walkthrough of the entire stack: the statistics it rests on,
the incumbent model it is trying to beat, every module in `flowquake/`, the
training and evaluation protocol, the generative/CSEP layer, and the statistical
machinery that turns raw numbers into claims. Written to be read top to bottom by
someone who has never seen a point process.

Companion docs: [README.md](README.md) (claims + setup),
[MANUSCRIPT.md](MANUSCRIPT.md) (the paper), [REPRODUCE.md](REPRODUCE.md) (commands),
[WORKING.md](WORKING.md) (current state and what is unfinished),
[results/CLAIMS.md](results/CLAIMS.md) (claim→artifact map).

---

## Table of contents

- [Part 0 — Orientation](#part-0--orientation)
- [Part I — The statistics, from scratch](#part-i--the-statistics-from-scratch)
  - [1. What a point process is](#1-what-a-point-process-is)
  - [2. The likelihood, and the choice that shapes this repo](#2-the-likelihood-and-the-choice-that-shapes-this-repo)
  - [3. Marks, factorization, and the three scores](#3-marks-factorization-and-the-three-scores)
- [Part II — ETAS, the incumbent](#part-ii--etas-the-incumbent)
  - [4. Three empirical laws](#4-three-empirical-laws)
  - [5. The exact ETAS used here](#5-the-exact-etas-used-here)
  - [6. Why ETAS is hard to beat](#6-why-etas-is-hard-to-beat)
- [Part III — The benchmark contract](#part-iii--the-benchmark-contract)
- [Part IV — The code, bottom up](#part-iv--the-code-bottom-up)
  - [7. `data.py` — turning a catalog into tensors](#7-datapy--turning-a-catalog-into-tensors)
  - [8. `ssm.py` — the selective state-space encoder](#8-ssmpy--the-selective-state-space-encoder)
  - [9. `flow.py` — the temporal head](#9-flowpy--the-temporal-head)
  - [10. `heads.py` — the spatial and magnitude heads](#10-headspy--the-spatial-and-magnitude-heads)
  - [11. `model.py` — assembly, and the memorization knob](#11-modelpy--assembly-and-the-memorization-knob)
  - [12. `train.py` / `config.py` — the training loop](#12-trainpy--configpy--the-training-loop)
  - [13. `evaluate.py` — scoring against ETAS](#13-evaluatepy--scoring-against-etas)
- [Part V — The second model: the neural-ETAS spatial head](#part-v--the-second-model-the-neural-etas-spatial-head)
- [Part VI — Generative evaluation: simulation and CSEP](#part-vi--generative-evaluation-simulation-and-csep)
- [Part VII — Statistics that turn numbers into claims](#part-vii--statistics-that-turn-numbers-into-claims)
- [Part VIII — The three results, and exactly what holds them up](#part-viii--the-three-results-and-exactly-what-holds-them-up)
- [Part IX — Boundaries, external dependencies, and how to check me](#part-ix--boundaries-external-dependencies-and-how-to-check-me)
- [Part X — Reading order and exercises](#part-x--reading-order-and-exercises)

---

## Part 0 — Orientation

**The one-sentence version.** Earthquake catalogs are the standard test bed for
point-process forecasting; the standard model, ETAS, is a hand-built recipe that
neural point processes have repeatedly failed to beat; FlowQuake replaces ETAS's
two weakest structural assumptions (fixed-window history, hand-crafted kernels)
while keeping its structure, and wins on likelihood in six regions.

**The thing that trips everyone up first.** There are *two different models* in
this repository, trained by two different scripts, and they are combined only at
scoring time:

| | production TPP | neural-ETAS spatial head |
|---|---|---|
| code | [flowquake/model.py](flowquake/model.py) | [flowquake/neural_etas.py](flowquake/neural_etas.py) |
| trainer | `python -m flowquake.train` | `scripts/train_neural_etas.py` |
| params | ~0.1–1 M, GPU | ~2 k, CPU, minutes |
| produces | `tll`, `sll`, `mll` | `sll` only |
| used for the headline | **temporal** (`tll`) | **spatial** (`sll`) |

The headline total-likelihood number is `tll` from the first model **plus** `sll`
from the second, paired event-by-event against the same region's ETAS
(`scripts/total_win_summary.py` → [runs/total_win.json](runs/total_win.json)).
The first model *also* has a spatial head (the kernel mixture), and that head
still **loses** to ETAS spatially. Keeping this straight explains most apparent
contradictions between sections of the manuscript.

**The second thing that trips people up.** The Mamba-style SSM encoder in
[flowquake/ssm.py](flowquake/ssm.py) — the most impressive-looking code in the
repo — is **not instantiated in any production run**. Every production config sets
`h_bottleneck: 0`, and [model.py:82-83](flowquake/model.py#L82-L83) sets
`self.encoder = None` in that case. Of the 123 tracked YAMLs, 114 set
`h_bottleneck: 0`, 3 omit the key (defaulting to 0), and 6 set it higher — and
those 6 are the *same three* ablation experiments counted twice, since each
appears both as `runs/ablation_h/h{4,16,64}.yaml` and as the corresponding
`runs/ablation_h/h*/config.yaml`. So: three ablation runs, everything else at
zero. The encoder exists to *make a scientific point*: §4.3 of
the manuscript shows that giving the heads access to a learned whole-catalog
embedding causes catastrophic memorization. The production model gets its
whole-catalog reach from hand-designed relational features and observation-anchored
mixture components instead. See [§11](#11-modelpy--assembly-and-the-memorization-knob)
and [Part VIII](#the-memorization-result-43).

**Repo map.**

```
flowquake/            the library (15 modules + __init__, ~3,000 lines)
  data.py             catalog → tensors: tokens, splits, mixture components, background map
  ssm.py              Mamba-2-style selective scan (ablation-only in production)
  flow.py             conditional rectified flow + exact ODE log-likelihood
  heads.py            kernel-mixture spatial head, Gutenberg–Richter magnitude head
  model.py            assembly: conditioning, losses, evaluation, sampling
  train.py            training loop, early stopping
  evaluate.py         test-window scoring, paired-vs-ETAS merge
  neural_etas.py      the full-history spatial head (second model)
  neural_etas_forecast.py   same head evaluated as a grid field, for CSEP
  ntest.py            autoregressive day simulation (the generative core)
  csep_forecast.py    pyCSEP N/S/M for the production model
  csep_forecast_head.py     pyCSEP N/S/M with the neural-ETAS head supplying locations
  etas_csep.py        ETAS pushed through the *identical* pyCSEP path (head-to-head)
  stats.py            block bootstrap, Holm, TOST
configs/              33 experiment YAMLs
scripts/              51 analysis/figure/robustness scripts
runs/                 226 committed artifacts (136 summary JSONs, 90 configs)
tests/                5 files / 16 tests: scan-vs-naive, flow-vs-analytic, head
                      normalization, data alignment, statistics
reference/            NOT COMMITTED — the EarthquakeNPP benchmark clone. Nothing runs without it.
```

---

## Part I — The statistics, from scratch

### 1. What a point process is

Start with the object. A **temporal point process** is a random set of points on
the time line: `t₁ < t₂ < t₃ < …`. An earthquake catalog is exactly that — a list
of times at which something happened. Nothing else.

The natural way to describe such a random set is not a density over the whole
configuration (that lives in an awkward infinite-dimensional space) but a
**conditional intensity**:

```
λ(t | H_t) = lim         P(one event in [t, t+dt) | history up to t) / dt
             dt → 0
```

Read it as an instantaneous rate, in events per unit time, that is allowed to
depend on everything that has already happened (`H_t`, the history). Two
canonical cases:

- **Poisson process**: `λ(t) = λ₀`, constant. The past tells you nothing. Gaps
  between events are Exponential(λ₀), memoryless.
- **Hawkes / self-exciting process**: `λ(t) = μ + Σ_{tⱼ < t} g(t − tⱼ)`. Every
  past event bumps the rate up by a decaying kernel `g`. Earthquakes are the
  textbook physical example: a mainshock is followed by aftershocks, and those
  aftershocks have their own aftershocks.

ETAS is a marked Hawkes process. That is the entire conceptual content of it.

### 2. The likelihood, and the choice that shapes this repo

Given a realization `t₁, …, t_n` on `[0, T]`, the log-likelihood of a point
process with intensity `λ` is

```
log L  =  Σᵢ log λ(tᵢ | H_{tᵢ})  −  ∫₀ᵀ λ(u | H_u) du
```

The first term rewards putting rate where events actually happened; the second
term is the penalty that stops you from just cranking `λ` up everywhere. It is
the normalizer, and it is the hard part: `λ` depends on the history, so the
integral has to be done along the realized path.

There is an equivalent formulation that matters enormously here. Define the
**next-event conditional density** — the density of the *waiting time* `τ` until
the next event, given the history:

```
f(τ | H)  =  λ(t_{i-1} + τ | H) · exp( − ∫₀^τ λ(t_{i-1} + u | H) du )
```

(rate now × probability of surviving until now). And then

```
log L  =  Σᵢ log f(τᵢ | H_{tᵢ₋₁})   + (a boundary term for the final censored gap)
```

**These are the same likelihood written two ways**, and the choice between them is
the single most consequential design decision in this repository:

- **Model `λ`** (what ETAS does): you get an interpretable additive
  self-excitation structure, but every likelihood evaluation needs `∫λ`. ETAS's
  kernels are chosen precisely so that integral has a closed form.
- **Model `f(τ | H)` directly** (what FlowQuake does): the normalization is now a
  1-D density's own normalization, which a normalizing flow gives you *exactly and
  for free*. You never compute `∫λ`. In exchange you lose the additive
  interpretation and you can only evaluate the process event-by-event, not as a
  rate field over continuous time.

FlowQuake takes the second road. [flowquake/flow.py](flowquake/flow.py) is a
density model over `log τ`, not an intensity model. This is why the temporal
scores are exact rather than approximated, and it is also why generating a forecast
requires *sequential simulation* ([Part VI](#part-vi--generative-evaluation-simulation-and-csep))
rather than just evaluating a rate on a grid.

### 3. Marks, factorization, and the three scores

Real catalogs carry **marks**: each event has a location `(x, y)` and a magnitude
`m`. A marked point process has a joint conditional density over
`(τ, x, y, m)`. FlowQuake factorizes it by the chain rule:

```
f(τ, x, y, m | H)  =  f_t(τ | H) · f_s(x, y | H) · f_m(m | H)
```

Note what is assumed: given the history, time / space / magnitude are treated as
conditionally independent. Each factor gets its own head. (ETAS makes the same
factorization — its productivity term couples magnitude to *future* rate, but the
current event's own mark distributions factor.)

The benchmark scores each factor separately, per test event, in physical units:

| score | definition | units | higher is |
|---|---|---|---|
| `tll` | `log f_t(τ)` | log (1/day) | better |
| `sll` | `log f_s(x, y)` | log (1/km²) | better |
| `mll` | `log f_m(m)` | log (1/mag unit) | better |
| `nll` | `−(tll + sll)` | nats | **lower** |

Two things to internalize:

1. **`nll` deliberately excludes `mll`.** That is the EarthquakeNPP convention.
   The magnitude head still matters — it is what makes the CSEP M-test pass — but
   it is not in the headline number.
2. **Units are absolute, not relative.** `sll ≈ −8.7` means the model assigns
   about `e^{−8.7} ≈ 1.7 × 10⁻⁴` per km² at the true location — as if the mass
   were spread evenly over ~5,900 km². These are genuine
   probability densities over California, not softmax scores. A "+0.11 nats/event"
   gain means the model assigns `e^{0.11} ≈ 1.12×` more probability mass to what
   actually happened — a 12% likelihood improvement per event, compounded over
   21,889 events.

The **targets to beat** on the flagship catalog (from
`reference/Experiments/ETAS/output_data_ComCat_25/ll_scores.json`):

| model | tll | sll | nll |
|---|---|---|---|
| ETAS | 1.4343 | −8.6898 | **7.2554** |
| Poisson | 0.5126 | −13.7745 | 13.2619 |

The Poisson row is the "you learned nothing" floor. ETAS sits 6 nats/event above
it. That gap is the size of the actual problem.

---

## Part II — ETAS, the incumbent

You cannot understand FlowQuake's design without understanding ETAS, because
FlowQuake is built as a structural upgrade to ETAS, not as an alternative to it.

### 4. Three empirical laws

Seismology has three robust century-old empirical laws. ETAS is what you get when
you write them as a Hawkes kernel.

1. **Omori–Utsu (1894/1961)** — aftershock rate decays as a power law in time
   since the mainshock: `n(t) ∝ (t + c)^{−p}`, with `p ≈ 1`. Power law, not
   exponential: aftershocks persist for years, and the tail carries real mass.
   This is why "look at the last 20 events" is a structurally fatal modeling
   choice.
2. **Gutenberg–Richter (1944)** — magnitudes are exponentially distributed above
   a completeness threshold: `P(M > m) = 10^{−b(m − m_c)}`, `b ≈ 1`. Ten times
   fewer M5s than M4s. Equivalently `m − m_c ~ Exponential(β)` with
   `β = b ln 10 ≈ 2.3`.
3. **Utsu productivity** — the number of aftershocks a mainshock triggers grows
   exponentially with its magnitude: `N ∝ e^{a(m − m_c)}`.

`m_c`, the **completeness magnitude**, appears everywhere and is not a nuisance:
below it, the seismic network simply misses events, so the catalog is not a
sample from the process. Every catalog in this repo is truncated at its `m_c`
(2.5 for ComCat, 0.6 for White, 4.0 for the ISC regions), and
`scripts/check_completeness.py` verifies the choice is stable across train and
test eras — because a `m_c` that drifts over time would manufacture a fake
temporal trend the model would happily learn.

### 5. The exact ETAS used here

The benchmark uses the Mizrahi et al. `etas` package. Its parameters are read
from `parameters_0.json` in
[scripts/precompute_trigger_features.py:52-60](scripts/precompute_trigger_features.py#L52-L60),
and the functional form is transcribed in
[flowquake/neural_etas.py:78-87](flowquake/neural_etas.py#L78-L87). Written out:

**Triggering weight** of past event `j` on the present (magnitude `mⱼ`, elapsed
time `Δt`):

```
w_j = k₀ · exp(a(mⱼ − m_c))  ·  exp(−Δt/τ)  ·  (Δt + c)^{−(1+ω)}
      └── productivity ──┘     └ taper ┘      └── Omori decay ──┘
```

**Spatial kernel** of that same parent, at squared distance `r²`:

```
K_j(r²) = (r² + dⱼ)^{−(1+ρ)},   dⱼ = d · exp(γ(mⱼ − m_c))
```

with normalizer `Z_j = ∫ K_j dA = π / (ρ · dⱼ^ρ)`. (Derivation: substitute
`u = r²`, `dA = π du`, so `∫ = π ∫₀^∞ (u + dⱼ)^{−(1+ρ)} du = π dⱼ^{−ρ}/ρ`.)
Bigger earthquakes get bigger `dⱼ` — a wider aftershock cloud. Power-law again,
so distant aftershocks stay possible.

**Conditional spatial density** — the probability density of *where* the next
event is, given that one occurs now:

```
                  μ  +  Σⱼ wⱼ Kⱼ(r²(s))
f_s(s | H)  =  ───────────────────────────
                μ·A  +  Σⱼ wⱼ Zⱼ
```

`μ` is the background rate density (per km² per day), `A` the region area. The
numerator is the intensity at `s`; the denominator is its integral over the
region, so this is intensity normalized to a density. Sanity check the code:
[precompute_trigger_features.py:143](scripts/precompute_trigger_features.py#L143)
computes exactly `log(mu + trig_num) − log(mu*area + trig_den)` and asserts it
reproduces the package's stored per-event `SLL` to `< 1e-6`. That assertion is
the contract between this repo and the benchmark; if it ever fails, everything
downstream is void.

**Fitting** is by expectation–maximization: E-step assigns each event a
probability of being background vs. triggered by each predecessor; M-step refits
`(μ, k₀, a, c, ω, τ, d, γ, ρ)`. It takes 3–4 CPU-hours per region
([REPRODUCE.md](REPRODUCE.md) §2). Nine parameters, hand-designed functional
forms, and it beats every neural point process in the published benchmark.

### 6. Why ETAS is hard to beat

Three reasons, and FlowQuake's whole design is a response to them.

1. **The functional forms are nearly right.** Power laws in time and space,
   exponential in magnitude — these are not arbitrary; they are what a century of
   data says. A neural net given a small catalog will spend its capacity
   rediscovering them, badly.
2. **It integrates over the whole history.** Every ETAS evaluation sums over
   *all* prior events. A 2011 M9 keeps contributing in 2019. The published NPP
   baselines truncate history — DeepSTPP sees 20 events — which discards exactly
   the Omori tail that carries the mass.
3. **The data is small and the target is a density.** ComCat_25 has ~70,000
   training events. Density estimation with heavy tails on that much data
   punishes flexibility: a flexible model can memorize where earthquakes happened
   in 1985 and score beautifully on train while catastrophically mis-locating
   2015. §4.3 measures exactly this.

FlowQuake's answer, in one line each:

- Keep the power-law functional forms; learn *modulations* of them. (`heads.py`,
  `neural_etas.py`)
- Reach the whole history — via relational features at 7 exponentially spaced
  lags, via long-lived big-trigger components spanning 2 years, and (in the second
  model) via literal full-history ETAS sums. (`data.py`, `neural_etas.py`)
- Structurally forbid the heads from seeing absolute coordinates, so memorizing
  geography is impossible rather than merely discouraged. (`model.py`
  `SAFE_TOKEN_DIMS`)

---

## Part III — The benchmark contract

**EarthquakeNPP** (Stockman, Lawson & Werner, TMLR 2026) is the benchmark. Its
finding: none of five neural point processes beat ETAS; ETAS wins spatial
log-likelihood against all of them. It ships five California catalogs with fixed
splits and a fitted ETAS baseline per catalog.

The ComCat_25 protocol, grounded from the harness:

| window | dates | role |
|---|---|---|
| auxiliary | 1971-01-01 → 1981-01-01 | history warm-up; never a target |
| train | 1981-01-01 → 1998-01-01 | gradient targets |
| val | 1998-01-01 → 2007-01-01 | early stopping only |
| test | 2007-01-01 → 2020-01-17 | 21,889 events ≥ M2.5, scored once |

Two conventions worth stating explicitly because they cause confusion:

- **"NPP training uses aux+train."** The auxiliary window's events are *inputs*
  (they build history for the first real targets) but are never *targets*. In
  code this is the difference between the token array (all events) and the target
  mask ([data.py:216-218](flowquake/data.py#L216-L218)).
- **Normalization statistics come from pre-val events only**
  ([data.py:221-229](flowquake/data.py#L221-L229), `fit = times < val_start`).
  So do the background seismicity map and every other fitted quantity. Test-era
  data touches nothing but the score.

The five California catalogs span a wide range of `m_c` and density — ComCat_25
(`m_c` 2.5), WHITE_06 (0.6), SanJac_10 (1.0), SaltonSea_10 (1.0), SCEDC_20 (2.0)
— and §4.5 adds six non-California regions built from ISC/INGV via
`scripts/build_region.py`. The spread in `m_c` is what makes the
"density-dependence" result of §4.5 measurable at all.

---

## Part IV — The code, bottom up

### 7. `data.py` — turning a catalog into tensors

[flowquake/data.py](flowquake/data.py) is where most of the modeling actually
happens, which is unusual and worth sitting with: the features here encode more
domain knowledge than the neural architecture does.

**Input.** A CSV with `time, x, y, magnitude, latitude, longitude`. `x, y` are km
in a local projection. Filter to `magnitude ≥ mcut`, sort by time, clip to
`[aux_start, test_end)`.

**Time.** `t_days` = days since the first event. `τᵢ = tᵢ − tᵢ₋₁` is the gap
*preceding* event `i`, floored at `TAU_FLOOR_DAYS = 1e-7` (~9 ms; the catalog's
smallest nonzero gap is ~5e-8 d). We model `log τ`, not `τ`, because inter-event
gaps span nine orders of magnitude (milliseconds to years) and a power-law tail
in `τ` becomes a roughly-tractable shape in `log τ`.

**The token.** Each event becomes a `TOKEN_DIM = 32` vector
([data.py:26-28](flowquake/data.py#L26-L28)):

```
[ log τᵢ , xᵢ , yᵢ , mᵢ ]                          ← 4 "core" dims
[ log(tᵢ − tᵢ₋ₖ), xᵢ − xᵢ₋ₖ, yᵢ − yᵢ₋ₖ, mᵢ₋ₖ ]     ← 4 dims × 7 lags = 28
      for k ∈ RECENCY_LAGS = (1, 2, 4, 8, 16, 32, 64)
```

That second block (`recency_matrix`,
[data.py:154-165](flowquake/data.py#L154-L165)) is the design's quiet core. Read
it as **raw material for an ETAS kernel, precomputed**:

- `log(tᵢ − tᵢ₋ₖ)` — the Omori argument at seven time scales at once.
- `(xᵢ − xᵢ₋ₖ, yᵢ − yᵢ₋ₖ)` — *displacements*, not positions. Translation
  invariant. Move the whole catalog 500 km east and these are unchanged.
- `mᵢ₋ₖ` — the productivity argument for each of those parents.

Exponential lag spacing means 7 features cover 64 events of history, the way a
dilated convolution covers a long receptive field cheaply. This is the mechanism
by which the production model "sees the whole catalog" without an encoder.

Everything is z-scored using train-era means and stds. `_lagged`
([data.py:40-46](flowquake/data.py#L40-L46)) clamps to the first event for the
earliest rows, which are aux-era only and never targets, so the clamping never
touches a scored prediction.

**Mixture components (`lastk`).** A separate tensor of shape
`(E, MIX_K, 4)` holding raw `[x, y, log Δt, m]` for each event's candidate
*triggering parents*. Three tiers:

1. **Recency tier**, `LAST_K = 64`: events `i, i−1, …, i−63`. Component 0 is the
   current event itself.
2. **Big-trigger tier**, `BIG_M = 16` (`big_trigger_matrix`,
   [data.py:49-73](flowquake/data.py#L49-L73)): the 16 largest events with
   `m ≥ 4.5` in the trailing 730 days. Rationale, verbatim from the code: *"the
   last-K window spans only ~70 days at the catalog's average rate, but large
   mainshocks keep triggering for years."* Empty slots get an "ancient,
   far-decayed" null row so the tensor stays rectangular and those components are
   effectively switched off by their features.
3. **Near tier** (optional, `n_near > 0`, `near_trigger_matrix`,
   [data.py:76-117](flowquake/data.py#L76-L117)): the `n_near` spatially nearest
   *prior* events within `rmax_km`, over all history, via a KD-tree. The comment
   quantifies the gap it closes: 64% of ComCat events recur within 0.5 km of a
   prior event, and 85% of those are outside the last-64 window. Selection is
   purely spatial and strictly causal (`cand[cand < i]`).

So `MIX_K = 64 + 16 = 80` components by default. This tiering is FlowQuake's
substitute for ETAS's full-history sum: rather than summing over all 70,000
prior events, select ~80 that plausibly matter, by three different criteria.

**Background map.** [data.py:255-273](flowquake/data.py#L255-L273) builds a
smoothed-seismicity grid on 2 km bins: histogram the train-era epicenters,
Gaussian-blur (σ = 2 bins ≈ 4 km), normalize to a density per km², then mix 98%
of that with 2% uniform so off-fault locations never get zero density. With
`adaptive_bg: true` you instead get `adaptive_bg_grid`
([data.py:120-151](flowquake/data.py#L120-L151)), the Helmstetter et al. (2007)
variable-bandwidth estimator: each event is smoothed by its own distance to its
6th-nearest neighbor, so dense fault traces stay sharp while isolated events
spread wide. The fixed-bandwidth version's failure mode — off-fault "holes" that
genuine background events fall into and get charged for — is exactly what the
variable bandwidth fixes.

This map **is absolute geography**, fit on train-era data. Be precise about the
memorization claim: FlowQuake forbids the *learned* conditioning from seeing
absolute coordinates, and it uses a *fitted, frozen, train-era-only* background
map. Those are different things, and the second is standard practice (ETAS's own
background is uniform; smoothed seismicity is the standard upgrade).

**The +1 shift.** This is the single most bug-prone convention in the file, so it
is worth being explicit. Token `i` describes event `i`. The prediction made *at*
position `i` is about event `i+1`. Therefore:

```
mask[i]        = "event i+1 is a target in this split"
target[i]      = feats[i+1][:4]          (normalized; only log τ is used)
raw_next[i]    = raw[i+1][1:4]           (x, y, m in physical units)
```

`full_sequence_batch` ([data.py:358-371](flowquake/data.py#L358-L371)) builds
this for the whole catalog as one length-`E` sequence for exact evaluation, and
`evaluate.py` inverts it (`mask_np[1:] = nxt[:-1]`,
[evaluate.py:114-116](flowquake/evaluate.py#L114-L116)) to recover which *events*
were scored, for timestamp-pairing against ETAS. `tests/test_data.py` guards this
alignment.

**`CropDataset`.** Training uses random contiguous crops of `window = 2048`
events, with the first `burn_in = 256` positions masked out of the loss. The
burn-in exists because a crop starts with no history: the SSM state and the lag
features at position 0 are garbage, and you do not want gradients from garbage.
`n_crops` is set to `batch_size × 1e6` — effectively an infinite stream; the
`steps` count and early stopping decide when to stop.

### 8. `ssm.py` — the selective state-space encoder

Not used in production (see [Part 0](#part-0--orientation)), but it is the
mechanism behind the memorization result, it is the best-tested code in the repo,
and it is genuinely instructive. Read it as a self-contained implementation of
Mamba-2's SSD algorithm in ~300 lines of pure PyTorch, no CUDA kernels.

**First principles.** A linear state-space model is a linear recurrence with a
hidden state:

```
H_t = A_t · H_{t-1} + B_t · x_t
y_t = C_t · H_t
```

If `A, B, C` are fixed, this is an LTI system and you can compute the whole
sequence with an FFT convolution. The **selective** part of Mamba is that
`A_t, B_t, C_t` depend on the input `x_t` — which is what gives it the ability to
"decide what to remember" — and that breaks the convolution trick.

The specific form here ([ssm.py:7-9](flowquake/ssm.py#L7-L9)) uses a *scalar*
decay per head:

```
H_t = a_t · H_{t-1} + Δt_t · B_t x_tᵀ,     a_t = exp(−Δt_t · A_h)
y_t = C_t · H_t + D · x_t
```

`H_t` is an `(N, P)` matrix per head — an outer-product memory, like fast weights.
`A_h > 0` is a learned per-head decay rate; `Δt_t = softplus(linear(x_t) + bias)`
is the input-dependent step size. **Selectivity lives entirely in `Δt_t`**: a large
`Δt` means "this event matters, decay the old state hard and write strongly";
small `Δt` means "skip this one." (This is a genuinely nice fit for earthquake
data — a M7 should reset the state, a background M2.5 should not.)

**The chunked scan.** Naive evaluation is a sequential loop over `L` — correct
(`selective_scan_ref`, [ssm.py:102-118](flowquake/ssm.py#L102-L118), in fp64,
used only by tests) but hopeless on a GPU for `L = 2048`. The SSD trick: split
the sequence into chunks of `Q = 64` and note that the contribution to `y_t`
decomposes into

- **intra-chunk**: the part from inputs inside `t`'s own chunk, and
- **inter-chunk**: the part carried in by the state at the chunk boundary.

*Intra-chunk* becomes a masked quadratic form — literally a causally-masked
attention matrix ([ssm.py:72-76](flowquake/ssm.py#L72-L76)):

```
y_intra[t] = Σ_{s ≤ t}  decay(t,s) · (C_t · B_s) · Δt_s · x_s
```

with `decay(t,s) = exp(Σ_{r=s+1..t} log a_r)` computed as a cumulative-sum
difference in `segsum_decay` ([ssm.py:21-33](flowquake/ssm.py#L21-L33)) — this is
the "duality" in *state-space duality*: within a chunk the SSM *is* linear
attention with a decay mask.

*Inter-chunk* needs only one summary per chunk
([ssm.py:78-93](flowquake/ssm.py#L78-L93)): `S_c = Σ_s decay(end, s) Δt_s B_s x_sᵀ`,
then a short sequential recurrence over the `L/Q` chunks. Cost:
`O(L·Q)` for the quadratic parts, `O(L/Q)` sequential steps. Parallel where it
can be, sequential only where it must be.

Everything runs in fp32 (`x, dt, A, Bm, Cm = (t.float() for t in ...)`) because
the decays are exponentials of cumulative sums and fp16 loses them.

**Correctness.** `tests/test_ssm.py` checks the chunked scan against the fp64
naive recurrence at `L = 200` (deliberately not a multiple of 64, exercising the
padding path) to `atol=1e-4`, and separately checks that scanning `[0:64]` then
`[64:128]` with the carried state equals scanning `[0:128]` — the invariant that
makes streaming valid.

**Streaming.** `prefill` / `step` ([ssm.py:274-295](flowquake/ssm.py#L274-L295))
support the simulation path: absorb the observed catalog in segments carrying
state, then advance one event at a time. `SSDBlock.step`
([ssm.py:207-232](flowquake/ssm.py#L207-L232)) is the O(1) single-event update,
including a manual depthwise-conv window so the causal conv also streams. This is
the same prefill/decode split as a transformer's KV cache, except the state is
constant-size instead of growing with sequence length.

**Block structure.** `SSDBlock` is Mamba-2's: one input projection producing
`(z, x, B, C, Δt)`, a depthwise causal conv over `(x, B, C)`, SiLU, the scan, a
gated RMSNorm (`norm(y * silu(z))`), an output projection. `SSMEncoder` stacks
them pre-norm with residuals.

### 9. `flow.py` — the temporal head

This is the exact-likelihood machinery. It models the density of one scalar: the
normalized log inter-event time.

**Continuous normalizing flows in one paragraph.** Pick a simple base density
`p₀ = N(0, I)`. Define a time-dependent velocity field `v(z, t)` and let samples
flow along `dz/dt = v(z, t)` from `t=0` to `t=1`. This transports `p₀` into some
`p₁`. The density transforms by the *instantaneous change of variables* formula:

```
d/dt log p(z(t), t)  =  − ∇·v(z(t), t)
```

so, integrating,

```
log p₁(u)  =  log p₀(z(0))  −  ∫₀¹ ∇·v dt        where z(1) = u
```

This is exact — no ELBO, no bound. The cost is an ODE solve per evaluation plus
a divergence. In high dimensions the divergence needs a stochastic (Hutchinson)
estimator; here `dim = 1`, so the "Jacobian trace" is a single derivative and can
be computed exactly and cheaply. **This is why the temporal head is a flow and the
spatial head is not** — 1-D exactness is free, 2-D is not.

**Training without solving the ODE.** Flow matching (Lipman et al. 2023 /
rectified flow) makes training simulation-free. Pick a straight path between
noise and data:

```
z_t = (1 − (1 − σ_min)·t) · z₀  +  t · u,     z₀ ~ N(0, I)
```

Its time derivative is constant along the path: `dz_t/dt = u − (1 − σ_min) z₀`.
So just regress the network onto that
([flow.py:70-78](flowquake/flow.py#L70-L78)):

```python
t  = torch.rand(B)
z0 = torch.randn_like(u)
zt = (1 - (1 - s) * t) * z0 + t * u
loss = mse(velocity(zt, t, cond), u - (1 - s) * z0)
```

Three lines, an MSE, no ODE. The theory says the minimizer of this *conditional*
objective is the marginal velocity field that transports `p₀` to the data
distribution.

**What `σ_min` buys.** At `t = 1` the path lands on `u + σ_min·z₀`, so the modeled
density is the data convolved with `N(0, σ_min²)` — a KDE-style bandwidth floor.
Configs use `sigma_min: [0.02, 0.01, 0.05]`. Without it, a flow trained on
*discretized* data (magnitudes on a 0.1 grid, timestamps at finite resolution)
will happily collapse onto the discrete atoms and report absurd likelihoods.
Treat it as regularization that makes the reported density honest.

**Exact likelihood.** `log_prob` ([flow.py:109-134](flowquake/flow.py#L109-L134))
integrates the ODE *backward* from the datum with RK4, accumulating the
divergence, then returns `log N(z(0)) − logdet`. `_vel_and_div`
([flow.py:94-107](flowquake/flow.py#L94-L107)) uses
`torch.func.jacrev` + `vmap` for a per-sample exact Jacobian trace. Default 64
steps at eval (`--steps 96` in `evaluate.py`), 32 during validation.

**Units.** The flow models `u = (log τ − μ)/σ`. Converting to `f(τ)` in 1/day
needs two Jacobian corrections
([model.py:233-235](flowquake/model.py#L233-L235)):

```
log f(τ) = log p(u)  −  log σ  −  log τ
                       └──┬──┘   └──┬──┘
              u = (logτ−μ)/σ    τ = exp(log τ)
```

That one line is the difference between a number you can compare to ETAS and a
number that means nothing. `tests/test_flow.py` verifies the whole path
end-to-end by training the flow on a known Gaussian and checking `log_prob`
against the analytic density to 0.15 nats.

**Initialization.** The final layer is zero-initialized
([flow.py:61-62](flowquake/flow.py#L61-L62)), so the initial velocity field is
identically zero and `p₁ = p₀ = N(0,1)` exactly at step 0 — a well-defined,
finite-likelihood starting point.

### 10. `heads.py` — the spatial and magnitude heads

**`KernelMixtureHead`.** The production spatial head. A mixture with one
component sitting at each of the `MIX_K` candidate parents from `data.py`, plus a
uniform component, plus the background-map component:

```
f_s(s) = Σ_j w_j · Kernel_j(s − s_j)  +  w_unif · (1/A)  +  w_kde · kde(s)
```

The per-component kernel is ETAS's shape, not a Gaussian
([heads.py:104-105](flowquake/heads.py#L104-L105)):

```
f(r) = (q − 1)/(π d²) · (1 + r²/d²)^{−q}
```

Verify it normalizes: `∫₀^∞ f(r) 2πr dr`, substitute `u = r²/d²`:
`= (q−1) ∫₀^∞ (1+u)^{−q} du = (q−1)·1/(q−1) = 1`. ✓ Heavy-tailed, unlike a
Gaussian — which matters, because aftershock distances are power-law distributed
and a Gaussian mixture pays enormous likelihood penalties on the tail events.

**Anisotropy.** Real aftershock clouds are elongated along the fault strike. Each
component gets `(ρ, θ)` giving elliptical axes `d·ρ` and `d/ρ` rotated by `θ`
([heads.py:88-94](flowquake/heads.py#L88-L94)). Crucially the axes are
**area-preserving**: `(dρ)·(d/ρ) = d²`, so `√det M = d²` and the *same*
normalizer works — elongation is free, no extra Jacobian term.
`tests/test_heads.py` numerically integrates a forcibly-elongated component over
a grid and confirms it still integrates to 1.

**Where the parameters come from.** A small MLP
([heads.py:76-86](flowquake/heads.py#L76-L86)) maps
`[cond, log Δt_j, m_j, log dist_j]` → `(mixture logit, d, q, ρ, cos θ, sin θ)` per
component. Those three per-component features *are the arguments of an ETAS
kernel*: recency (Omori), magnitude (productivity), distance. The head is not
learning "where earthquakes happen"; it is learning **how to weight and shape a
triggering kernel given a parent's age, size, and offset**. Softplus floors keep
`d ≥ d_floor` and `q ≥ q_floor` so the density can never spike to infinity.

Initialization ([heads.py:61-74](flowquake/heads.py#L61-L74)) puts the head at
`d ≈ 2.5 km`, `q ≈ 1.8`, isotropic, ~35% background — i.e. at a plausible ETAS
kernel, before any training.

**Why this head resists memorization.** Its components sit at *observed event
locations supplied at evaluation time*. Nothing in the weights encodes "there is
a fault at (−120.3, 36.1)". Move the catalog, and the mixture moves with it. The
only absolute-geography term is the KDE background component, which is a fitted
frozen map, not a learned one.

**Sampling** ([heads.py:115-154](flowquake/heads.py#L115-L154)) draws a component,
then an elliptical radius by inverse CDF (`F(u) = 1 − (1+u²)^{1−q}`, invertible in
closed form), then rotates. Note the defensive block at lines 126-129: during
long autoregressive simulation a single NaN in the conditioning would poison the
logits and trip a device-side assert inside `multinomial`, so weights are
sanitized and the uniform component is floored. This kind of code is what makes a
10,000-catalog CSEP run finish.

**`GRMagnitudeHead`.** Gutenberg–Richter, made conditional:
`m − m_c ~ Exponential(β(cond))`, so
`log f(m) = log β − β(m − m_c)`. One linear layer produces `β` via softplus,
initialized at 2.0 (i.e. `b ≈ 0.87`). Two details carry real weight:

- The `+0.005` in `dm = clamp(m − mc, 0) + 0.005`
  ([heads.py:173](flowquake/heads.py#L173)) is a half-bin shift for the catalog's
  0.1-magnitude discretization. Without it, evaluating a continuous density at
  the exact grid points is systematically biased.
- Making `β` history-dependent is a real departure from ETAS, which holds `b`
  fixed. The manuscript credits this head with restoring the CSEP magnitude test.

### 11. `model.py` — assembly, and the memorization knob

[flowquake/model.py](flowquake/model.py) wires the three heads together and holds
the central experimental control.

**The conditioning vector.** Everything hinges on `SAFE_TOKEN_DIMS`
([model.py:32-35](flowquake/model.py#L32-L35)):

```python
SAFE_TOKEN_DIMS = [0, 3] + list(range(4, TOKEN_DIM))
# = log τ, magnitude, and all 28 relational features
# EXCLUDES dims 1 and 2 — absolute x and y.
```

30 dimensions, none of which is an absolute coordinate. The comment states the
principle exactly: these are *"translation-invariant statistics that cannot
fingerprint a specific catalog position-era."* Then:

```python
cond_dim = len(SAFE_TOKEN_DIMS) + h_bottleneck    # 30 + h
```

**The `h_bottleneck` knob** ([model.py:38-44](flowquake/model.py#L38-L44)):

- `h = 0` → no encoder is even constructed; heads see the 30 relational dims
  only. Memorization through learned conditioning is *structurally impossible*,
  not merely penalized. **This is production.**
- `h > 0` → build the SSM encoder, project its output to `h` dims, add Gaussian
  noise at train time, concatenate. The encoder has seen absolute `x, y` (it
  consumes the full 32-dim token), so this channel *can* carry geography.

That knob is the independent variable of §4.3. Results at `ckpt_last`, from
[runs/ablation_h/memorization_figure.json](runs/ablation_h/memorization_figure.json):

| h | train nll | held-out nll | gap | where the *best* val checkpoint landed |
|---|---|---|---|---|
| 0 | 7.28 | 7.62 | **0.34** | step 7,750 (of 11,750 trained) |
| 4 | 4.14 | 19.65 | **15.50** | step 250 (the first check ever run) |

The `h = 4` model reaches a *train* NLL of 4.14 — far better than ETAS's 7.26 —
while its held-out NLL blows up to 19.65, worse than the Poisson baseline of
13.26. And the "best checkpoint" column is the kill shot: for every `h > 0` the
best held-out checkpoint is *the first one ever evaluated*, at step 250
([runs/ablation_h/ablation_h.json](runs/ablation_h/ablation_h.json) — `h = 4, 16,
64` all report `step: 250`). Memorization is not a late-training pathology you can
early-stop your way out of; it begins immediately.

**The three entry points.**

- `fm_losses` ([model.py:175-197](flowquake/model.py#L175-L197)) — training.
  Flow-matching MSE for time; direct closed-form negative log-likelihood for
  space and magnitude. Weighted `(1.0, 1.0, 0.5)`. Note the asymmetry: the
  temporal head is trained by regression on a velocity field, the other two by
  their actual likelihood. Only the temporal head needs the flow-matching
  surrogate; the closed-form heads can be optimized directly.
- `log_likelihood` ([model.py:219-243](flowquake/model.py#L219-L243)) —
  evaluation. Chunks over events (`event_chunk = 4096`) because the ODE + Jacobian
  is memory-hungry; converts to physical units; returns per-event `tll/sll/mll`.
- `sample_next` / `build_token` / `lastk_from_bufs`
  ([model.py:247-315](flowquake/model.py#L247-L315)) — simulation. `build_token`
  is the exact inverse of `data.py`'s feature construction, reconstructing a
  normalized token from physical `(τ, x, y, m)` plus the lane's history buffers.
  Any drift between these two is a silent, catastrophic bug — the simulator would
  feed the model out-of-distribution inputs — which is why `data.py` and
  `build_token` should always be read as a pair.

Physical clamps in `sample_next` ([model.py:265-268](flowquake/model.py#L265-L268))
bound `τ ∈ [1e-7, 60]` days, `x, y` to the region box, `m ∈ [m_c, 8.5]`. Without
them one bad draw propagates into the next step's features and derails an entire
simulated catalog.

### 12. `train.py` / `config.py` — the training loop

Deliberately plain, which is the right call for a research codebase where the
interesting variance should live in the model, not the optimizer.

```
AdamW, lr 3e-4, weight_decay 0.03
500-step linear warmup → cosine decay
grad clip 1.0
crops of 2048 events, batch 8, up to 20,000 steps
validate every 250 steps on 4,096 subsampled val targets (32 ODE steps)
early stop after 16 checks with no improvement in val nll
```

Two things to notice:

- **The early-stopping criterion is `nll = −(tll + sll)`**, matching the
  benchmark's headline metric — not the training loss, which includes `mll` at
  weight 0.5. Model selection targets the reported metric.
- **Validation is subsampled and uses fewer ODE steps** (32 vs 96 at test). Cheap
  enough to run every 250 steps; the final number is always recomputed at full
  fidelity by `evaluate.py`.

Checkpoints (`ckpt_best.pt`, `ckpt_last.pt`) carry the config, weights, *and*
`cat.stats` — the normalization constants and background grid. That last part is
what makes frozen-checkpoint forward evaluation possible: `evaluate.py --catalog
… --test-start …` swaps the evaluation catalog and window **without** touching the
checkpoint's stats ([evaluate.py:62-66](flowquake/evaluate.py#L62-L66)), which is
exactly the discipline an out-of-time test requires.

`--init-from` loads weights with `strict=False` for warm-started few-shot
transfer, which is how the Greece/Iran results in §4.5 are produced.

`config.py` is a dataclass-backed YAML loader with one nice property: unknown keys
**raise** ([config.py:81-82](flowquake/config.py#L81-L82)). A typo'd hyperparameter
fails loudly instead of being silently ignored — the classic way research configs
lie to you.

### 13. `evaluate.py` — scoring against ETAS

Loads a checkpoint, runs `log_likelihood` over the whole catalog as one sequence
(exact, full history, no crops), and writes both a summary JSON and a per-event
CSV.

The important function is `paired_vs_etas`
([evaluate.py:28-51](flowquake/evaluate.py#L28-L51)): it merges FlowQuake's
per-event scores with ETAS's per-event `TLL`/`SLL` from
`augmented_catalog.csv` **on timestamp**, then reports mean gain, standard error,
and win rate for temporal, spatial, and joint.

Why paired comparison rather than comparing two means: consecutive earthquakes
are massively correlated (an aftershock sequence is one "event" statistically),
so unpaired standard errors are wildly optimistic. Pairing on the same events
removes the shared difficulty, and the block bootstrap of
[Part VII](#part-vii--statistics-that-turn-numbers-into-claims) handles the
remaining autocorrelation.

`res["beats_ETAS_nll"]` is a boolean written straight into the artifact — a small
thing, but it means the claim is stored next to the number that supports it.

---

## Part V — The second model: the neural-ETAS spatial head

This is the piece that produces the spatial and total-likelihood wins, and it is
architecturally unlike everything above. Four files:

```
scripts/precompute_trigger_features.py   full-history ETAS sums, once, ~30-60 min CPU
flowquake/neural_etas.py                 the head (~2k params)
scripts/train_neural_etas.py             trainer (CPU, minutes)
flowquake/neural_etas_forecast.py        the same head as a grid field, for CSEP
```

### The idea

Take ETAS's spatial density and make it a **strict superset**: add learnable
pieces that can be switched off to recover ETAS *exactly*. Then any measured gain
is a gain over ETAS by construction, not an artifact of a different setup.

```
                bg(s)  +  α·far_num(s)  +  Σ_{j ∈ near} w'_j K'_j(s)
f_s(s | H)  =  ────────────────────────────────────────────────────────
                μ'·A   +  α·far_den     +  Σ_{j ∈ near} w'_j Z'_j
```

([neural_etas.py:59-96](flowquake/neural_etas.py#L59-L96).) Three learnable
extensions over ETAS:

1. **Background** — ETAS uses uniform `μ`. Here `bg` is a learned mixture of
   uniform and four causal multi-scale KDE maps (bandwidths 1.5, 6, 25, 100 km),
   gated by `sigmoid(kde_gate)`. "Causal" is load-bearing: the KDE at event `i`
   is built from events `j < i` only
   ([precompute_trigger_features.py:112-114](scripts/precompute_trigger_features.py#L112-L114)),
   so it is a legitimate online forecast quantity, not a fitted map.
2. **Per-parent neural modulations** — a 2→32→32→3 MLP maps each near-set
   parent's `(magnitude, log Δt)` to offsets `(Δlog w, Δlog d, Δρ)` on its ETAS
   weight and kernel shape. The head learns *"triggers of this size and this age
   should be weighted a bit more and spread a bit wider than the global inversion
   says"*.
3. **Global scalars** — `α` scales the frozen far field, `μ'` adjusts the
   background rate.

### The normalization argument (this is the whole trick)

The MLP takes `(m_j, Δt_ij)` and **never the target location `s`**
([neural_etas.py:61-62](flowquake/neural_etas.py#L61-L62)). Neither does near-set
*selection*: parents are chosen by top-256 ETAS weight plus 128 nearest to the
**previous event's** location
([precompute_trigger_features.py:116-129](scripts/precompute_trigger_features.py#L116-L129)),
never to `s`.

Consequence: at a fixed forecast time, the weights `w'_j` and shapes `d'_j, ρ'_j`
are *constants* with respect to `s`. So `Z'_j = π/(ρ'_j d'^{ρ'_j}_j)` remains the
exact closed-form integral of `K'_j`, and the denominator is exactly the integral
of the numerator. **The density is properly normalized by construction, with no
numerical integration anywhere.** Let the MLP see `s` and this collapses
immediately — you would be learning an unnormalized energy and would need a
partition function you cannot compute.

This same property is what makes CSEP evaluation tractable
([neural_etas_forecast.py:6-13](flowquake/neural_etas_forecast.py#L6-L13)): every
query point on the CSEP grid shares one near set and one far field, so the whole
spatial field for a forecast day is one vectorized pass — the same cost as a
gridded ETAS forecast.

### The near/far split

The far field (all priors outside the near set) is **precomputed and frozen**:
`far_num = trig_num − near_base_num`, `far_den = trig_den − near_base_den`
([precompute_trigger_features.py:155](scripts/precompute_trigger_features.py#L155)).
The near set (≤ 384 parents) is recomputed live in the trainer so gradients can
reach its modulations. The result: full-history ETAS fidelity at the cost of a
384-term sum per event.

### Two verification gates

The trainer refuses to run unless both pass — an unusually good practice worth
copying:

1. **Gate-closed reproduction** — build a second head with `kde_gate_init = -30`
   (KDE mass ≈ 0), no MLP, all offsets zero, and assert its per-event `sll`
   matches the package's ETAS `SLL` to `< 2e-5`
   ([train_neural_etas.py:94-101](scripts/train_neural_etas.py#L94-L101)). This is
   the operational proof of "strict superset." The committed run reports agreement
   to **1.77e-9 nats** ([runs/etas_sll_repro.json](runs/etas_sll_repro.json)).
2. **Init sanity** — the actual training init (with a ~5% KDE gate) must be within
   0.05 nats of ETAS ([train_neural_etas.py:103-106](scripts/train_neural_etas.py#L103-L106)).

Why the gate opens to 5% rather than 0 at init: the comment
([neural_etas.py:45-47](flowquake/neural_etas.py#L45-L47)) records a real
debugging finding — with a hard "exact" logit at +16, the softmax starves the KDE
components and their weights freeze at `[1, 0, …]`. A small open gate keeps
gradients alive. And the reported gains are always measured against the *package*
ETAS scores, never against this near-ETAS init.

### Controls

Three configurations are trained and reported separately:

| flag | what learns |
|---|---|
| `--no-mlp` | background mixture + `α` + `μ'` only |
| `--refit-globals` | classical (flETAS-style) SGD refit of global kernel params, no MLP |
| *(default)* | + per-parent neural modulations |

The `--refit-globals` control is the honest one: it asks *"is the gain just from
refitting ETAS's parameters better?"* Its own docstring flags its limitation —
it reweights the near set exactly but scales the far field by `α`, making it a
conservative lower bound on a true full refit.

### Training

Full-batch-ish Adam on CPU, two param groups (scalars at lr 1e-2, MLP at 2e-3
with weight decay), early stopping on val `sll`, splits identical to the
benchmark's. Test scoring is paired against the package's per-event ETAS `SLL`
with a stationary block bootstrap. The docstring
([train_neural_etas.py:5-8](scripts/train_neural_etas.py#L5-L8)) explicitly warns
that grids and multiple seeds were run, so this is **not** a
test-scored-once protocol — a caveat the repo states about itself.

---

## Part VI — Generative evaluation: simulation and CSEP

Likelihood answers "does the model assign high density to what happened?" CSEP
answers a different and operationally more important question: "if you generate
forecasts, are they *calibrated*?" A model can win on likelihood and still be
systematically wrong about counts.

### Simulation: `ntest.py`

[flowquake/ntest.py](flowquake/ntest.py) `simulate_day_events` is the generative
core. For a forecast day:

1. Absorb the observed catalog up to the day start (streaming SSM prefill if the
   encoder exists; otherwise just history buffers).
2. Broadcast that state across `n_sims` independent lanes.
3. Loop: sample `(τ, x, y, m)` from the heads, advance the lane's history
   buffers, rebuild the token, repeat — until the sampled time leaves the 1-day
   window or `MAX_EVENTS_PER_DAY = 200` is hit.

Vectorized over lanes with `torch.where` masking so finished lanes stop updating
without breaking the batch.

**The truncated-first-event subtlety** ([ntest.py:88-104](flowquake/ntest.py#L88-L104))
is a genuinely subtle piece of correctness. The last observed event is at some
`t_last < day_start`, and we *know* nothing happened between `t_last` and
`day_start`. So the first simulated event must be drawn from the conditional
`f(τ | τ > day_start − t_last)`, not from `f(τ)`. The code rejection-samples this
truncated conditional (up to 200 rounds), and lanes that never accept are marked
as having no event that day. Skip this and you systematically over-forecast every
day's first event.

### The three CSEP tests

pyCSEP catalog-based consistency tests, each comparing an observed statistic
against its distribution across simulated catalogs:

- **N-test** — is the observed *count* consistent? Reports
  `δ₁ = P(N_sim ≥ N_obs)`, `δ₂ = P(N_sim ≤ N_obs)`.
- **S-test** — are the observed *locations* consistent with the forecast's
  spatial rate density?
- **M-test** — is the observed frequency-magnitude distribution consistent?

Pass criterion at two-sided 95%: every reported quantile `≥ 0.025`
([csep_forecast.py:233-258](flowquake/csep_forecast.py#L233-L258)). Note the
`csep_summary` helper carefully excludes non-evaluable days (NaN, or pyCSEP's
`(-1,-1)` sentinel) from the denominator rather than counting them as failures —
which is why you see `S 79/85` and not `S 79/100`.

### Three forecast pipelines, one harness

The head-to-head design is the strongest methodological feature of this section:

| module | counts/times/mags | locations |
|---|---|---|
| [csep_forecast.py](flowquake/csep_forecast.py) | FlowQuake simulator | FlowQuake kernel-mixture head |
| [csep_forecast_head.py](flowquake/csep_forecast_head.py) | FlowQuake simulator | **neural-ETAS head**, sampled from its grid field |
| [etas_csep.py](flowquake/etas_csep.py) | ETAS simulation | ETAS |

All three write **byte-compatible** `csep_ascii` CSVs and are scored by the same
`csep_summary`, on the same forecast days, at matched simulation budgets. The
`etas_csep.py` docstring records the verification that makes the days comparable:
ETAS's `timewindow_end` equals FlowQuake's `test_start` (2007-01-01 for
ComCat_25), so day offset `d` is the identical wall-clock window in both
pipelines. `--rerun` mode recomputes tests from saved CSVs with no GPU, so
scoring is reproducible independently of generation.

Supporting details worth knowing:

- `fit_xy_to_lonlat` ([csep_forecast.py:44-69](flowquake/csep_forecast.py#L44-L69))
  inverts the catalog's km projection with a cubic polynomial, max residual
  ~0.36 km against ~11 km CSEP cells. Fit on the full catalog — legitimate,
  because it is a fixed geometric transform, not seismicity.
- The observed catalog is **reloaded fresh every day**
  ([csep_forecast.py:189-194](flowquake/csep_forecast.py#L189-L194)) because
  pyCSEP's `.filter*` methods mutate in place; reusing the object would silently
  shrink the observed set day after day. That comment documents a real bug that
  was found and fixed.
- `neural_etas_forecast.py --validate` reconstructs the field evaluator at 40
  real events' own history states and checks it reproduces the training-path
  per-event `sll` — closing the loop between "head as a density scored at events"
  and "head as a field sampled on a grid."

**Reported outcome** (§4.2, matched 10³-catalog budget, 100 identical forecast
days): the full-history head scores N 95/100, S 79/85, M 90/92. Against ETAS on
83 shared days, both pass the S-test 77 times, with 10 discordant days split 5–5
→ McNemar exact **p = 1.0000**. Read that correctly: it is *not* evidence the head
is better on consistency. It is evidence the spatial likelihood gain **cost
nothing** in calibration, which is the claim being made.

---

## Part VII — Statistics that turn numbers into claims

[flowquake/stats.py](flowquake/stats.py) plus `scripts/stats_hardening.py`. This
part is where a lot of ML papers are weakest and where this repo is unusually
careful.

**The problem.** Per-event gains are strongly autocorrelated. An aftershock
sequence is thousands of events that are, statistically, close to one
observation. A naive standard error over 21,889 "independent" events
overstates precision by a large factor.

**Stationary block bootstrap** ([stats.py:42-81](flowquake/stats.py#L42-L81)).
Resample *blocks* of consecutive events with geometric lengths (mean 50) that
wrap around the series, rather than individual events. Blocks preserve
within-sequence correlation; geometric lengths keep the resampled series
stationary (Politis & Romano 1994). 2,000–4,000 replicates, percentile CI.

Everything downstream is built on it:

- `block_bootstrap_pvalue` — two-sided, add-one smoothed so a p-value is never
  exactly 0. The floor at 4,000 replicates is `p = 0.0005`, which is why that
  exact number appears all over the artifacts. **`p_boot: 0.0005` means "at the
  resolution floor", not "p = 0.0005".**
- `holm_bonferroni` ([stats.py:129-138](flowquake/stats.py#L129-L138)) — step-down
  family-wise correction. Six regions tested → six chances to get lucky → the
  family is corrected as a unit.
- `tost_equivalence` ([stats.py:141-158](flowquake/stats.py#L141-L158)) — two
  one-sided tests. This one deserves emphasis. **A confidence interval crossing
  zero is not evidence of a tie; it is absence of evidence.** TOST asks the
  affirmative question: is the 90% CI of the mean gain *contained within*
  ±margin? Any "ties ETAS" statement in the manuscript is required to pass this
  at ±0.1 nats/event. That is a discipline most papers skip.
- **McNemar's exact test** (in `scripts/audit_readiness.py`) for the paired
  CSEP pass/fail comparison — the right test for paired binary outcomes, keyed on
  the discordant pairs only.

`ci_decision` ([stats.py:32-39](flowquake/stats.py#L32-L39)) reduces a CI to
`win` / `loss` / `tie`, and that string is what gets written into artifacts —
so the decision rule is recorded alongside the number, not applied later by a
human reading a table.

---

## Part VIII — The three results, and exactly what holds them up

### Result 1 — the temporal loss flips (§4.1)

3-seed mean test `tll`, FlowQuake vs region-fitted ETAS
([runs/fullsuite_summary.json](runs/fullsuite_summary.json)):

| catalog | m_c | FQ tll | ETAS tll | Δ |
|---|---|---|---|---|
| ComCat_25 | 2.5 | 1.486833 | 1.434343 | +0.052 |
| WHITE_06 | 0.6 | 2.066893 | 2.021097 | +0.046 |
| SanJac_10 | 1.0 | 1.160957 | 1.132527 | +0.028 |
| SaltonSea_10 | 1.0 | 2.433720 | 2.332039 | +0.102 |
| SCEDC_20 | 2.0 | 2.619408 | 2.540983 | +0.078 |

Five for five. Under the block bootstrap, four are significant; **San Jacinto's
interval crosses zero and is recorded as a tie** — the repo's own bookkeeping,
stated in [WORKING.md](WORKING.md). The aggregation is verified: all 30 values in
`fullsuite_summary.json` recompute from the 15 per-seed `eval_test.json` files, 29
to exactly zero difference and one to 8.7e-19 (float summation order).

Also note: `sll` is *worse* than ETAS on all five (e.g. −9.06 vs −8.69 on
ComCat). The production model's headline `nll` is therefore **worse** than ETAS
(7.572 vs 7.255). That is not hidden — it is the gap Result 3 closes with the
second model.

### The memorization result (§4.3)

Covered in [§11](#11-modelpy--assembly-and-the-memorization-knob). The claim is
mechanistic, not just empirical: exposure to a learned whole-catalog embedding
lets the heads fingerprint catalog position-eras; the fix is not regularization
but **structural exclusion** of absolute coordinates from the conditioning. The
evidence that rules out "just early-stop it" is that every `h > 0` run's best
held-out checkpoint is the first one evaluated.

Honest caveat, and the repo says it in [NOVELTY.md](NOVELTY.md): this sub-claim
was searched for in the literature and found neither confirmed nor contradicted.
It should be framed as a diagnostic contribution, not a "first."

### Result 3 — total likelihood beats ETAS in six regions (§4.4)

Composite = FlowQuake temporal `tll` + neural-ETAS head `sll`, paired per event
against the same region's own ETAS inversion
([runs/stats_hardening.json](runs/stats_hardening.json) →
`total_with_head_family`):

| region | ΔTotal (nats/event) | 95% CI | Holm p | pairing coverage |
|---|---|---|---|---|
| California (ComCat_25) | +0.1133 | [0.1006, 0.1261] | 0.003 | 100% |
| Italy | +0.2095 | [0.1862, 0.2332] | 0.003 | 100% |
| Japan | +0.0390 | [0.0163, 0.0620] | 0.0045 | 96.3% |
| Chile | +0.0608 | [0.0349, 0.0900] | 0.003 | 97.1% |
| Greece | +0.0756 | [0.0224, 0.1316] | 0.011 | 92.0% |
| Iran | +0.0844 | [0.0098, 0.1711] | 0.0185 | 89.0% |

All six positive and Holm-significant. On ComCat that is `nll` 7.1421 vs ETAS
7.2554 ([runs/total_win.json](runs/total_win.json)).

Read the fine print, which the repo supplies:

- **Japan's +0.039 is flagged by the artifact itself** — `dTot_abs_below_0.05:
  true`. Statistically positive, below the stated 0.05-nat interpretability
  margin.
- **Greece and Iran use `temporal_variant: "fewshot"`**, not native training.
  Natively they lose; transfer is what makes them positive.
- **Coverage < 100% outside California/Italy** because the ETAS pipeline bins
  magnitudes before the completeness cut while FlowQuake's temporal CSV uses the
  raw cut. Reported per region rather than papered over.
- **The temporal family tells a different story than the total family.** Holm-
  corrected *temporal-only* p-values: California 0.003, Italy 0.003, Chile 0.036
  significant; Japan 0.274, Iran 0.273, Greece 0.648 **not** significant. The
  six-for-six total win rests substantially on the spatial head.

### Out-of-time replication (2020–2026)

Frozen checkpoints, 10,187 events never used for fitting or model selection
([runs/total_win.json](runs/total_win.json) → `forward_2020_2026`):

| | mean | 95% CI | win rate |
|---|---|---|---|
| dT | +0.0574 | [0.0376, 0.0819] | 60.5% |
| dS | +0.0666 | [0.0553, 0.0784] | 47.9% |
| dTot | +0.1241 | [0.1035, 0.1455] | 55.2% |

All three replicate, and dTot is slightly *larger* than in-sample. The dS win
rate of 47.9% with a positive mean is worth understanding: the head wins less
than half the events but wins by more when it wins — consistent with fixing the
tail (events ETAS badly mislocates) rather than shifting the bulk.

The label matters: this is **retrospective out-of-time scoring**, not a
registered prospective forecast. `total_win.json` carries that sentence in its
own `notes` field.

---

## Part IX — Boundaries, external dependencies, and how to check me

### The bound on every claim

Quoting [REPLACEMENT_READINESS.md](REPLACEMENT_READINESS.md), which is the file
to cite publicly:

> FlowQuake is a transferable neural point-process candidate that beats ETAS
> temporally on dense catalogs and, with a full-history neural-ETAS spatial head
> initialized from each region's ETAS inversion, beats ETAS on total likelihood
> across the six tested regions; it is not yet an operational replacement for
> ETAS systems.

The load-bearing qualifier: **the spatial head is initialized from each region's
ETAS inversion.** It is an *upgrade of a deployed ETAS system*, not an
inversion-free replacement for one. Per-region normalization and a train-era
smoothed-seismicity background map are also still required — lighter than an ETAS
inversion, but not zero target-catalog preprocessing.

### What is not in this repository

**`reference/` is not committed, and nothing runs without it.** It is a clone of
the EarthquakeNPP benchmark plus derived data. All 90 committed run configs and
all 33 in `configs/` — 123 tracked YAMLs — have `catalog_path` under it. Not one
resolves on a fresh clone.

Worse, per [WORKING.md](WORKING.md): six ETAS configs the manuscript depends on
(`Japan_25`, `Chile_25`, `Greece_25`, `Iran_25`, `Italy_25`, `ComCat_25_refit2020`)
are not shipped by the benchmark, are written by no script here, and exist only
inside the gitignored tree. Nobody but the author can currently regenerate the
§4.5 region baselines or the §4.1 refit control.

Also missing from the README's setup list but required: `Datasets/NewZealand/`,
`Datasets/Italy_Mw/` + `Italy_mw_raw/`, `Datasets/ComCat_forward/`,
`Datasets/ComCat_extended/`, `Experiments/ETAS/pycsep_tests_parallel.py`, and
`output_data_<Cfg>/parameters_0.json`.

Per-event score CSVs are `.gitignore`d, so block-bootstrap CIs cannot be
recomputed from the committed tree — only read from the stored summaries.

### Claim-tracing status

From [results/CLAIMS.md](results/CLAIMS.md) and [WORKING.md](WORKING.md), across
142 traced claim rows (134 distinct claims): 114 match the artifact exactly or to
rounding (63 rows at the artifact's own precision, 51 rounding to the printed
value), 2 are ambiguous between two committed artifacts, **8 distinct claims are
contradicted by their artifact**, and **12 distinct claims have no committed
backing at all**. If you are going to rely on a specific number, check
`results/CLAIMS.md` for its row first.

### Open items before submission

From [NOVELTY.md](NOVELTY.md): a fresh literature sweep (the RECAST team is
active on transfer), a targeted search for prior art on the memorization
finding, and a precise point-process-vs-classifier boundary against SafeNet
(which preempts any unqualified "first transfer learning for earthquake
forecasting" claim).

### Verifying without `reference/`

These run on a fresh clone:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q     # 16 pass; data-alignment tests skip without reference/
```

The passing tests are meaningful, not smoke: chunked scan vs fp64 naive
recurrence, streaming-state continuation, flow `log_prob` vs analytic Gaussian
density, and numerical grid integration confirming the spatial mixture (including
a forcibly elongated anisotropic component) integrates to 1.

With `reference/` present, the gates that must pass before any result is
trusted:

```bash
python scripts/etas_sll_repro.py             # ETAS spatial reproduction, must pass
python -m flowquake.neural_etas_forecast --validate ComCat_25   # field == training path
python scripts/audit_readiness.py            # -> runs/replacement_readiness.json
```

---

## Part X — Reading order and exercises

### If you have 30 minutes

1. [flowquake/data.py:154-165](flowquake/data.py#L154-L165) — `recency_matrix`.
   The relational features are the model.
2. [flowquake/model.py:32-44](flowquake/model.py#L32-L44) — `SAFE_TOKEN_DIMS` and
   `h_bottleneck`. The central scientific control.
3. [flowquake/neural_etas.py:59-96](flowquake/neural_etas.py#L59-L96) — the whole
   spatial head, 38 lines.
4. [runs/total_win.json](runs/total_win.json) — the headline result with its CIs.

### If you have a day

Add, in order: `heads.py` (with `tests/test_heads.py` open beside it), `flow.py`
(with `tests/test_flow.py`), `train.py`, `precompute_trigger_features.py`,
`ntest.py`, then `csep_forecast.py` and `etas_csep.py` as a pair.

### Exercises that will actually teach you the stack

1. **Derive the unit conversion.** Starting from `u = (log τ − μ)/σ`, derive
   `log f(τ) = log p(u) − log σ − log τ` and match it to
   [model.py:233-235](flowquake/model.py#L233-L235). If you can't, you don't yet
   understand what `tll` is.
2. **Verify the kernel normalizes.** Show
   `∫₀^∞ (q−1)/(πd²)(1+r²/d²)^{−q} · 2πr dr = 1`. Then explain why the
   area-preserving anisotropy (`dρ`, `d/ρ`) needs no change to that normalizer.
3. **Break the normalization on purpose.** Add the target location `s` as an
   input to `NeuralETASSpatialHead.mlp`. Explain precisely which line of
   [neural_etas.py](flowquake/neural_etas.py) becomes wrong, and why no amount of
   retraining fixes it.
4. **Find the coupling.** `data.py` builds tokens from a catalog;
   `model.build_token` rebuilds them from simulated events. Identify every field
   that must agree between them and construct a mismatch that would corrupt
   simulation while leaving likelihood evaluation untouched.
5. **Reproduce the memorization result's shape.** Predict, before looking, what
   `h = 128` would do to train NLL, held-out NLL, and best-checkpoint step. Then
   check your reasoning against the `h = 4, 16, 64` rows of
   [runs/ablation_h/ablation_h.json](runs/ablation_h/ablation_h.json).
6. **Interrogate a claim.** Pick a number from the README, find its row in
   `results/CLAIMS.md`, open the cited artifact, and confirm it. Try one of the 8
   contradicted claims and work out what happened.

---

## Appendix — cheat sheet

**Constants** ([data.py](flowquake/data.py)):
`TAU_FLOOR_DAYS = 1e-7` · `RECENCY_LAGS = (1,2,4,8,16,32,64)` · `TOKEN_DIM = 32` ·
`LAST_K = 64` · `BIG_M = 16` · `BIG_MAG_MIN = 4.5` · `BIG_WINDOW_DAYS = 730` ·
`MIX_K = 80` · `SAFE_TOKEN_DIMS` = 30 dims (excludes absolute x, y) ·
head `cond_dim = 30 + h_bottleneck`.

**Production hyperparameters** ([configs/n1_density.yaml](configs/n1_density.yaml)):
`d_model 96` · `n_layers 4` · `flow_hidden 96` · `mix_hidden 64` ·
`loss_weights [1.0, 1.0, 0.5]` · `sigma_min [0.02, 0.01, 0.05]` · `dropout 0.1` ·
`h_bottleneck 0` · `window 2048` · `burn_in 256` · `batch 8` · `lr 3e-4` ·
`steps 20000` · `val_every 250` · `patience 16`.

**Reading an artifact:** `decision` ∈ {win, loss, tie} from the bootstrap CI ·
`p_boot: 0.0005` = at the resolution floor · `p_holm` = family-corrected ·
`coverage_vs_etas` = fraction of ETAS-scored events successfully paired ·
`dTot_abs_below_0.05: true` = below the interpretability margin.

**The one-line mental model:** *keep ETAS's power laws, replace its fixed
parameters with history-conditioned modulations, forbid the learned parts from
ever seeing an absolute coordinate, and prove normalization stays closed-form at
every step.*
