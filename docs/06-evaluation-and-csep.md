# Forecast evaluation: scoring rules, information gain, and CSEP

Everything in this repository — every table, every claim, every "beats ETAS" —
is a statement produced by an evaluation procedure. This chapter is about the
procedures. It builds scoring rules from the definition, proves the one theorem
you must be able to prove on demand (propriety of the log score), derives what a
"nats per event" number actually means, and then walks the entire CSEP testing
apparatus from the RELM experiment through to the exact pass/fail arithmetic in
[flowquake/csep_forecast.py](../flowquake/csep_forecast.py).

It also reports several things the repository's own prose gets wrong or
oversells. That is deliberate. A viva is an adversarial exercise, and the fastest
way to survive one is to have already found the weak points yourself.

## What this chapter buys you

- You can define a scoring rule, define propriety, and **prove** that the
  logarithmic score is strictly proper, in about four lines, from Jensen's
  inequality — and say exactly which regularity condition makes the proof work.
- You can state the Bernardo uniqueness theorem precisely, including what
  "essentially unique" means (affine invariance), what "local" means, and where
  the theorem fails (binary sample spaces; higher-order locality).
- You can convert `+0.113 nats/event` into a sentence a seismologist and a
  statistician both accept, and you can say why multiplying it by N does *not*
  give you a defensible Bayes factor here.
- You can define every CSEP consistency test — N, M, S, L, CL — with its null
  hypothesis, its test statistic, and its rejection rule, and explain why the
  repo's uniform two-sided `min(quantile) >= 0.025` criterion is **not** the
  classical convention for the S- and M-tests, and what changes when you use the
  classical one instead (numbers below; the head-to-head S ranking flips).
- You can explain why the M-test is a worse problem still: its catalog-based
  statistic is a *discrepancy*, not a likelihood, so the repo's two-sided rule
  counts "the magnitude histogram fitted better than the simulations" as a
  failure — which is where 32 of the 33 M-test rejections in this repository
  come from (§8.5).
- You can explain why "95/100 days pass" is exactly what a *perfect* forecast
  should produce, why 100/100 would be a red flag, and why the McNemar test in
  §4.2 of [MANUSCRIPT.md](../MANUSCRIPT.md) had essentially no power to detect
  anything.
- You can explain, without hand-waving, why FlowQuake structurally cannot emit a
  gridded rate forecast and therefore must Monte-Carlo its way into the CSEP
  harness, and what that costs.

## Prerequisites

Chapter filenames follow `docs/NN-topic.md`; links below are relative to this
file. If one 404s the chapter has been renamed — the chapter number is the stable
reference.

- **[Chapters 1](01-point-processes.md)–[2](02-seismology.md)** (point processes;
  the conditional intensity and the two
  equivalent forms of the likelihood). The short version is
  [STACK.md](../STACK.md) Part I §1–§3 — read that at minimum. You need
  `lambda(t | H_t)`, the compensator `Lambda(t)`, and the identity
  `log L = sum_i log f(tau_i | H)` before this chapter makes sense.
- **[Chapter 3 — ETAS](03-etas.md)** for what the incumbent is and what its
  parameters `mu, k0, a, c, omega, tau_tap, d, gamma, rho` do. (The repo names
  the Omori taper timescale `tau`, which collides with the inter-event gap
  `tau`; throughout this chapter the taper is `tau_tap`.)
- **[Chapter 7 — statistics for dependent data](07-statistics-dependent-data.md)**
  for the stationary block bootstrap,
  Holm–Bonferroni, and TOST. This chapter forward-references TOST in §11 and
  §16; the implementation is [flowquake/stats.py](../flowquake/stats.py).
- [STACK.md](../STACK.md) Part VI is the *code* walkthrough of the CSEP harness.
  This chapter is the theory that walkthrough assumes; where they overlap I link
  rather than restate.

Notation is the repo's: `t` in days since catalog start, `tau` = inter-event
gap, `s = (x, y)` in km, `m` = magnitude, `m_c` = completeness magnitude,
`b` = Gutenberg–Richter b-value, `beta = b * ln(10)`. Scores are
`tll = log f_t` (log 1/day), `sll = log f_s` (log 1/km^2), `mll = log f_m`,
`nll = -(tll + sll)`.

---

## 1. What is being scored, and by whom

There are two distinct evaluation questions in this repository, and conflating
them is the single most common error a reader makes.

- **Question A — density scoring.** Given the history, how much probability
  density did the model put on *what actually happened next*? This is the
  per-event `tll`/`sll`/`mll`/`nll` machinery in
  [flowquake/evaluate.py](../flowquake/evaluate.py). It is *relative*: it means
  nothing except against another model on the same events.
- **Question B — calibration of issued forecasts.** If the model issues a
  one-day forecast, is the world it predicts statistically compatible with the
  world that happened? This is CSEP. It is *absolute*: a model is tested against
  itself, no reference model needed, and the verdict is "consistent" or
  "rejected".

Neither implies the other; §5 makes that precise. §2–§4 build Question A, §6–§12
build Question B, §14 maps both onto the repo.

One framing to hold throughout: a probabilistic forecast is a *claim*, and an
evaluation procedure is the mechanism by which the claim can be wrong. If you
cannot state, for a given number in a paper, the procedure that could have
produced the opposite number, that number is decoration.

---

## 2. Scoring rules from first principles

### 2.1 The definition

Let `Y` be the quantity to be forecast, taking values in a sample space `Omega`
(for us: an inter-event time in `(0, inf)`, a location in `R^2`, a magnitude in
`[m_c, inf)`, or a whole catalog). Let `F` be a probability distribution on
`Omega` — the forecast. Let `y` be the value that materializes.

```
A SCORING RULE is a function

    S : (F, y)  ->  R u {-inf}

that assigns a real-valued reward to the forecast F when y is observed.
```

Convention here: **positively oriented**, higher is better. (Half the
literature uses the negative orientation — "loss" — including the repo's `nll`.
Say which you mean, every time.)

The *expected score* of forecast `F` when the truth is `G` is

```
    S(F, G)  :=  E_{Y ~ G} [ S(F, Y) ]  =  integral  S(F, y) dG(y)
```

Note the notational overload: the second argument is a value in the first form
and a distribution in the second. That is standard (Gneiting & Raftery 2007).

### 2.2 Propriety, and the proof for the log score

```
S is PROPER on a class P of distributions if, for all F, G in P,

    S(G, G)  >=  S(F, G)

and STRICTLY PROPER if equality holds only when F = G.
```

In words: **the forecaster maximizes their expected score by reporting what they
actually believe.** This is an incentive-compatibility property. A scoring rule
that is not proper rewards lying, and any model trained or selected on an
improper rule is being pushed toward a distribution that is not the truth.

The **logarithmic score** is

```
    S_log(F, y)  =  log f(y)
```

where `f` is the density of `F` with respect to a fixed dominating measure
(Lebesgue on `R^2` for locations, counting measure for a discrete outcome). The
"fixed dominating measure" clause is not pedantry: it is why `sll` has units of
`log(1/km^2)` and why you may not compare an `sll` computed in km^2 with one
computed in degrees^2. See [STACK.md](../STACK.md) Part I §3 on units.

**Theorem.** `S_log` is strictly proper on the class of distributions with
densities w.r.t. a common dominating measure.

**Proof.** Let `g` be the true density and `f` the reported one. Then

```
  S(G,G) - S(F,G)  =  E_{Y~g} [ log g(Y) - log f(Y) ]
                   =  integral  g(y) log( g(y) / f(y) ) dy
                   =  KL(g || f)
```

so we need `KL(g || f) >= 0` with equality iff `f = g` almost everywhere. By
Jensen's inequality applied to the concave function `log`, with the expectation
taken under `g`:

```
  -KL(g||f) = E_g[ log( f(Y)/g(Y) ) ]
            <= log E_g[ f(Y)/g(Y) ]                      (Jensen, log concave)
            =  log integral_{ g>0 }  g(y) * f(y)/g(y) dy
            =  log integral_{ g>0 }  f(y) dy
            <= log integral f(y) dy  =  log 1  =  0      (f >= 0 and integrates to 1)
```

Hence `KL(g||f) >= 0`. Equality in Jensen for a *strictly* concave `log`
requires `f(Y)/g(Y)` to be `g`-almost-surely constant; combined with equality in
the last line (`f` puts no mass outside the support of `g`) and the fact that
both integrate to 1, the constant must be 1, so `f = g` a.e. Strict propriety
follows. **QED.**

Two things a professor will probe about this proof:

1. **`f` must be a genuine normalized density.** The final step used
   `integral f = 1`. A model reporting an unnormalized `f~ = c * f` with `c > 1`
   gains `log c` free on every event and the rule stops being proper. This is
   exactly why FlowQuake's temporal score is trustworthy — the rectified flow
   gives a change-of-variables normalization for `f_t(tau|H)` with no variational
   bound (the repo calls this "exact", and the Jacobian trace really is computed
   exactly rather than by a Hutchinson estimator — [STACK.md](../STACK.md) Part IV
   §9; the residual approximation is the ODE solver's discretization, not a
   likelihood bound), and
   ETAS's kernels are chosen so `integral lambda` is closed form. Two models are
   comparable on log score only if both are honestly normalized over the same
   space. [runs/etas_sll_repro.json](../runs/etas_sll_repro.json) exists to certify
   exactly that for the spatial side: FlowQuake's re-implementation of the ETAS
   spatial density reproduces the benchmark's own per-event values to
   `max_abs_sll_err` `1.766e-09`, `match: true`.
2. **The score is unbounded below**, and that is the rule doing its job: a model
   that declares something impossible and is then wrong *should* be destroyed.
   Practically it means one event can dominate a mean, implementations floor the
   density, and the floor is a modelling choice that moves the number. Bounded
   rules (Brier, CRPS) forgive this; the log score does not.

### 2.3 Locality

```
S is LOCAL (strictly, 0-local) if S(F, y) depends on F only through the
value of its density at the observed point:

    S(F, y)  =  s( f(y), y )
```

The log score is local by construction. The Brier score is not: to score a
categorical forecast it uses the probabilities assigned to outcomes that did
*not* occur. CRPS is not: it integrates the CDF over the whole line.

Locality is a strong philosophical position. It says: *what you said about
things that did not happen is irrelevant*. This is the likelihood principle in
scoring-rule clothing.

It has a practical consequence that matters for the spatial score. `sll` is
0-local, so it does not care about *distance*. If model A puts density
`1.7e-4 /km^2` at the true location and so does model B, they score identically
— even if A's mass is all within 5 km and B's is scattered over the whole state
with a coincidental bump at the right place. That is a real limitation of the
headline spatial number, and §16 has the hostile question.

### 2.4 The uniqueness theorem, stated precisely

> **Bernardo (1979), *Expected Information as Expected Utility*, Annals of
> Statistics 7:686–690.** Among scoring rules that are *smooth*, *local*, and
> *proper*, the only ones are of the form
>
> ```
>     S(F, y)  =  a * log f(y)  +  b(y),     a > 0
> ```
>
> where `b` is an arbitrary real function of the outcome alone.

That is what "essentially unique" means: **unique up to a positive affine
transformation whose additive part may depend on the outcome but not on the
forecast.**

Why that latitude is harmless:

- The multiplicative constant `a > 0` is a change of units. `a = 1` gives nats;
  `a = 1/ln 2` gives bits. It cannot change any ordering of forecasts.
- The additive `b(y)` is the same for every forecast being compared on the same
  event, so it **cancels in every difference**. Since every comparison in this
  repository is a *difference* of log scores on the same event (§4), `b` is
  invisible. It is exactly the freedom to add "how hard was this event" to the
  score.

Fine print you should volunteer before you are asked:

- **The discrete case needs `|Omega| >= 3`.** On a binary sample space every
  proper rule is trivially local (`f(y)` determines the whole distribution, so
  Brier is local there too). The uniqueness statement has content only from
  three outcomes up (Dawid, Lauritzen & Parry, *Proper local scoring rules on
  discrete sample spaces*, AoS 2012, 40:593–608).
- **Relaxing locality reopens the field.** Let the score depend on the density
  *and its first `m` derivatives* at `y` ("`m`-local") and a large class of
  proper rules exists for every even `m >= 2` — and none for odd `m`. Those
  rules can be computed **without the normalizing constant** (Parry, Dawid &
  Lauritzen, *Proper local scoring rules*, AoS 2012, 40:561–592); the Hyvärinen
  score / score matching lives here. Relevant to FlowQuake: score matching was
  the alternative route to an unnormalized temporal model; the repo instead
  bought exact normalization from the flow and therefore gets to use the 0-local
  rule.

### 2.5 Brier: proper, not local

For a binary event (`y in {0,1}`) with reported probability `p`:

```
    S_Brier(p, y)  =  -(p - y)^2
```

**Propriety.** Let the true probability be `q`, so `Y ~ Bernoulli(q)`:

```
  E[ -(p - Y)^2 ] = -[ (p-1)^2 q + (p-0)^2 (1-q) ]
                  = -[ p^2 - 2pq + q ]
                  = -(p - q)^2  -  q(1 - q)
```

The second term does not involve `p`; the first is maximized uniquely at
`p = q`. So Brier is strictly proper. That decomposition is worth memorizing: it
is *reliability* (how far your probability is from the truth) plus *uncertainty*
(the irreducible variance of the event). The multi-category version
`-sum_k (p_k - 1{y=k})^2` is proper by the same argument applied coordinatewise,
and is manifestly non-local.

### 2.6 CRPS: proper, not local, in the units of the outcome

For a real-valued outcome with predictive CDF `F`:

```
    CRPS(F, y)  =  integral_{-inf}^{inf}  ( F(z) - 1{ z >= y } )^2  dz
```

(negatively oriented — lower is better; negate for the positive convention).

**Propriety, in one line.** For each threshold `z`, the quantity
`(F(z) - 1{z >= y})^2` is exactly the Brier loss for the binary event
`{Y <= z}` with reported probability `F(z)`. Brier is proper for each `z`, and a
non-negatively weighted integral of proper rules is proper. Hence CRPS is proper
on the class of distributions with finite first moment (which is what makes the
integral finite). Strictness: if `F != G` as distributions then, both being
right-continuous and monotone, `F(z) != G(z)` on a non-degenerate interval — a
set of positive Lebesgue measure — and Brier is *strictly* proper at each such
`z`, so the integrated expected score is strictly larger for `G`.

Three properties that matter operationally: CRPS **has the units of `y`** (a
CRPS of 12 km is interpretable in a way that `-9.06 log(1/km^2)` is not); it is
**distance-sensitive** (missing by 100 km always costs more than missing by 5
km, which the log score does not guarantee); and it **reduces to absolute error**
when `F` is a point mass, making it the natural distributional generalization of
MAE.

### 2.7 So when would you *not* use the log score?

| situation | preferred rule | why |
|---|---|---|
| Model comparison / selection over full densities | **log** | strictly proper, local, and its expectation gap *is* the KL divergence — the ordering it induces is the information-theoretic one |
| A specific decision threshold ("M>=6 in 30 days?") | **Brier** | the decision is binary; Brier's reliability/resolution decomposition maps onto the decision |
| Location error where distance matters | **CRPS / energy score** | log score is 0-local and blind to how far the miss was |
| Heavy tails, unreliable extremes, one event could dominate | **CRPS** | bounded influence; log score is unbounded below |
| Model is unnormalized (EBM, score-based) | **Hyvärinen / m-local** | does not need the normalizing constant |
| Reporting to a non-statistical stakeholder | **CRPS or Brier** | physical or probability units |

The honest summary: the log score is the right *default* for point-process model
comparison because it is the unique local proper rule and because its
differences are log-likelihood ratios, which is the currency of statistical
evidence. It is *not* the right rule if the decision you care about is
threshold-shaped or distance-shaped. FlowQuake reports only log scores. That is
a defensible choice inherited from EarthquakeNPP, and it is also a gap: no
distance-sensitive spatial score is reported anywhere in this repository.

---

## 3. Information gain: turning nats into a sentence

### 3.1 Definition

Let model A and model B both issue proper predictive densities for the same `N`
test outcomes `y_1, ..., y_N` under the same histories. Define the per-event
log-score difference

```
    d_i  =  log f_A(y_i | H_i)  -  log f_B(y_i | H_i)
```

and the **information gain per event** (IG, or "information gain per
earthquake", IGPE, in the CSEP literature):

```
    IG  =  (1/N) sum_{i=1..N} d_i
```

Units: **nats** if the logarithm is natural, **bits** if base 2. Convert with
`bits = nats / ln 2 = nats / 0.6931`.

### 3.2 What the number means, derived

Exponentiate the definition:

```
    exp(IG)  =  exp( (1/N) sum_i log( f_A(y_i)/f_B(y_i) ) )
             =  ( prod_i  f_A(y_i)/f_B(y_i) )^{1/N}
             =  GEOMETRIC MEAN of the per-event density ratio
```

So `IG = +0.113 nats/event` means: **taking the geometric mean over test events,
model A assigned `exp(0.113) = 1.120` times as much probability density to what
actually happened as model B did.** A 12.0% per-event density improvement.

The arithmetic for the numbers actually in this repo
([runs/total_win.json](../runs/total_win.json), ComCat_25, test window 2007-01-01 →
2020-01-17, N = 21,889 paired events):

| quantity | nats/event | `exp(IG)` | bits/event | events per bit |
|---|---|---|---|---|
| `dT` temporal | +0.0533 | 1.055 | 0.0769 | 13.0 |
| `dS` spatial (full-history head) | +0.0600 | 1.062 | 0.0866 | 11.6 |
| `dTot` total | +0.1133 | 1.120 | 0.1635 | 6.1 |

And for the 2020-01-17 → 2026 forward window (N = 10,187, same file):
`dT` +0.0574, `dS` +0.0666, `dTot` +0.1241 → `exp(0.1241) = 1.132`, 0.179
bits/event.

### 3.3 The sentence to say to a professor

> "Both models emit genuine normalized densities over the same outcome space, so
> the difference of their log scores is a per-event log-likelihood ratio. A gain
> of +0.113 nats per event means that, in geometric mean over the 21,889 test
> events, FlowQuake assigned 1.12× the probability density that region-fitted
> ETAS assigned to what actually happened — a 12% per-event improvement. In
> information units that is 0.164 bits per event, so it takes about six
> earthquakes to accumulate one bit of evidence discriminating the two models."

The "six earthquakes per bit" clause is the one that lands. It converts an
abstract score into a *rate of evidence accumulation*, which is what a
statistician actually wants to know.

For scale, the same arithmetic on the benchmark's floor: ETAS `nll` 7.2554 vs a
homogeneous Poisson `nll` 13.2619 (quoted in [README.md](../README.md); the ETAS
row is independently backed by `etas_nll` in
[runs/fullsuite_summary.json](../runs/fullsuite_summary.json), the Poisson row's
underlying artifact `ll_scores.json` lives in the un-committed `reference/`
tree). That is a 6.006 nat/event gap, `exp(6.006) = 406`. **ETAS is ~406×
better than "you learned nothing"; FlowQuake is ~1.12× better than ETAS.** State
both, always. It is the honest scale of the contribution and pre-empts the
"neural models beat ETAS by a mile" caricature.

### 3.4 The `N * IG` trap

It is true that

```
   N * IG  =  sum_i d_i  =  log [ prod_i f_A(y_i) / prod_i f_B(y_i) ]
           =  the total log-likelihood ratio of A vs B on the test set
```

and for `N = 21889`, `IG = 0.1133` that is **2,480 nats**, i.e. a likelihood
ratio of `e^2480`. Do not say this out loud without three immediate caveats:

1. **It is a likelihood ratio, not a Bayes factor.** A Bayes factor integrates
   over parameter uncertainty. This is a plug-in ratio at two point estimates,
   both of which were fitted on data. It equals a Bayes factor only under
   degenerate (point-mass) priors on the fitted parameters.
2. **The events are not independent.** `prod_i f(y_i | H_i)` *is* the correct
   joint likelihood by the chain rule for a point process — that part is fine.
   The problem is inference: a per-event standard error computed as
   `sd(d)/sqrt(N)` treats 21,889 events as 21,889 independent pieces of
   evidence, and an aftershock sequence of 3,000 events is statistically closer
   to one. This is exactly why [flowquake/stats.py](../flowquake/stats.py) uses a
   stationary block bootstrap with mean block length 50 instead of a naive SE
   (Chapter 7). The reported CI on `dTot` is `[0.1006, 0.1268]`
   ([runs/total_win.json](../runs/total_win.json)) — that is the number to quote,
   not `e^2480`.
3. **`p_boot: 0.0005` is a resolution floor, not a p-value.** The bootstrap uses
   4,000 replicates with add-one smoothing, so `2 * (0 + 1)/(4000 + 1) =
   0.0005` is the smallest value it can emit. [STACK.md](../STACK.md) Part VII says
   this explicitly and it appears all over the artifacts.

---

## 4. Why paired comparison, and what the estimand is

### 4.1 The variance argument

Suppose you evaluated A and B *separately*: compute `mean(log f_A)` over test
events, compute `mean(log f_B)`, subtract. The true variance of that difference
is

```
    Var( mean_A - mean_B )  =  ( Var(A) + Var(B) - 2 Cov(A,B) ) / N
```

which is exactly `Var(d)/N`. The point is not that pairing changes the
estimator's variance — it does not, it is the same random variable — but that
*reporting two marginal means and their separate standard errors throws the
covariance term away*, and any error bar built that way (`sqrt(SE_A^2 + SE_B^2)`)
is wrong by the amount below.

Per-event log scores in a catalog have enormous variance, and almost all of it
is *event difficulty*, not model quality. An aftershock 40 seconds and 300 m
from an M7.1 has a temporal log-density of order `+5` under any competent model;
an isolated background event after a three-week quiescence has one of order
`-2`. Both models see the same event and both are pushed the same way. So
`Cov(A,B)` is large and positive.

Write `Var(A) = Var(B) = sigma^2` and `Corr(A,B) = rho`:

```
    Var(d)  =  2 sigma^2 (1 - rho)
```

With `rho = 0.95`, `Var(d) = 0.1 sigma^2`, against `2 sigma^2` for the
covariance-ignoring version — so the honest SE is `sqrt(2/0.1) = 4.5x` **smaller**
than the one you would quote by combining two marginal standard errors.
**Pairing does not change the estimate; it stops you discarding the nuisance
covariance when you state the uncertainty.**

*Honesty note:* I could not compute the empirical `rho` for this repository.
Per-event score CSVs are excluded by `.gitignore`
([WORKING.md](../WORKING.md): "Per-event score CSVs are excluded by `.gitignore`, so
every block-bootstrap CI and everything in `runs/stats_hardening.json` is a
summary"), and the one tracked per-event file,
`runs/neural_etas/ComCat_25/per_event_forward_full.json`, contains only
aggregates. The argument above is analytic; the actual `rho` in this repo is not
verifiable from the committed artifacts.

### 4.2 The estimand, stated carefully

```
    theta  :=  E[ d ]  =  E[ log f_A(Y | H) - log f_B(Y | H) ]
```

where the expectation is over the joint law of `(H, Y)` **as generated by the
real seismicity process on the evaluation window**, with both models held fixed
(frozen weights, frozen ETAS inversion). The sample estimate is `IG` and the
uncertainty comes from resampling the *series* `d_1, ..., d_N`.

Three things this is not:

- **It is not a statement about parameters.** No parameter of either model
  appears. The estimand is a property of two fixed predictive machines on one
  window.
- **It is not a likelihood ratio test.** An LRT compares *nested* models fitted
  to the *same* data and gets its null distribution from Wilks' theorem
  (`2 log LR -> chi^2_k`). Here the models are non-nested, fitted on a different
  (earlier) window, and scored out of sample. There is no chi-square null and no
  assumption that either model is correct — both are certainly misspecified. The
  relevant reference class is *predictive accuracy comparison*: Diebold &
  Mariano (1995, *Journal of Business & Economic Statistics* 13:253–263) is the
  canonical treatment of a paired loss-differential test with autocorrelation-
  robust variance, and the repo's block bootstrap is a nonparametric analogue of
  it. For non-nested in-sample model selection the classical reference is Vuong
  (1989, *Econometrica* 57:307–333).
- **It is not a claim that A is "the true model".** Under a proper rule, the
  best-scoring model is the one closest in expected-KL to the truth *among those
  compared*. Both can be terrible.

### 4.3 The `win_rate` field, and why it is not the effect size

The artifacts report a `win_rate` beside each mean: `dS.win_rate` **0.4972** for
the ComCat spatial gain ([runs/total_win.json](../runs/total_win.json)). Read it
carefully — **the spatial head wins on fewer than half of individual events while
having a mean gain of +0.060 nats.** Not a contradiction: the gains are
asymmetric, losing slightly on many events and winning a lot on some. It is a
useful diagnostic and exactly the number a hostile examiner seizes on; §16 Q3
has the response.

---

## 5. Calibration and sharpness

### 5.1 Definitions

Let `F_1, F_2, ...` be a sequence of predictive distributions and `y_1, y_2, ...`
the realizations.

**Probabilistic calibration (PIT).** Define the probability integral transform

```
    u_i  =  F_i( y_i )
```

If each `F_i` is the true conditional law of `Y_i` given the information used to
form it, and `F_i` is continuous, then `u_i ~ Uniform(0,1)`. A histogram of the
`u_i` is the standard calibration diagnostic: U-shaped means the forecasts are
too narrow (over-confident), hump-shaped means too wide (under-confident),
sloped means biased.

For a *temporal point process* there is an exact and much sharper version of
this. The random time change theorem (usually credited to Meyer 1971 and
Papangelou 1972; brought into seismology as residual analysis by Ogata 1988,
JASA 83:9–27) says: if `lambda(t | H_t)` is the true conditional
intensity, then transforming event times by the compensator,
`t_i -> Lambda(t_i)`, produces a **unit-rate Poisson process**. Equivalently the
transformed gaps are i.i.d. Exponential(1), so `u_i = 1 - exp(-(Lambda(t_i) -
Lambda(t_{i-1})))` is i.i.d. Uniform(0,1). In FlowQuake's parameterization the
same object is available directly as `u_i = F_t(tau_i | H_i)`, the flow's own
CDF. **This is the calibration test the repository does not run.** No PIT
histogram, no KS test on transformed times, appears anywhere in `flowquake/` or
`scripts/`. Naming that gap yourself is worth several minutes of goodwill in a
viva.

**Marginal calibration.** The average predictive CDF equals the empirical CDF of
the observations. Weaker and easier to satisfy.

**Sharpness.** The concentration of the predictive distributions — mean
predictive variance, or mean 90% interval width. **Sharpness is a property of
the forecasts alone**; it never looks at the observations. A forecast that always
says "M2.5, in the 3 m^2 around my desk, in the next 4 milliseconds" is maximally
sharp and utterly uncalibrated.

### 5.2 The paradigm

> **Gneiting, Balabdaoui & Raftery (2007), *Probabilistic forecasts,
> calibration and sharpness*, JRSS-B 69:243–268.** The goal of probabilistic
> forecasting is to **maximize the sharpness of the predictive distributions
> subject to calibration**.

Calibration is the constraint; sharpness is the objective. The reason this
formulation is so durable is the companion fact: **proper scoring rules assess
calibration and sharpness simultaneously.** A strictly proper score cannot be
improved by being sharper-but-wrong or by being right-but-vague; the optimum is
the true conditional law.

### 5.3 Why a log-score win does not imply calibration, and vice versa

This is the crux of why FlowQuake runs *both* likelihood scoring and CSEP, and
you must be able to argue it in both directions.

**Direction 1 — win on log score, fail a consistency test.** The log score is an
*average*. A specific, structured calibration defect that affects a minority of
days can be invisible in the mean while being systematic enough for a targeted
test to catch. Concretely: suppose a model's daily count distribution is
correctly centred on typical days but systematically over-dispersed on
post-mainshock days (5% of days). The per-event log scores on those days are
only mildly affected — the events still happen where and when the model roughly
expects — but the N-test, which looks at the *count distribution* on each day
separately, rejects on exactly those days. The average hides what the
day-by-day test exposes. This is not hypothetical: the *first* ETAS run through
this harness scored a perfectly respectable set of likelihoods and yet failed
the N-test on 27 of 100 days ([runs/etas_csep_pod/csep_results.json](../runs/etas_csep_pod/csep_results.json),
`summary.N.n_pass` 73).

**Direction 2 — pass every consistency test, lose badly on log score.** Take the
homogeneous Poisson model with the correct long-run rate and a long-run smoothed
spatial density. Its daily count distribution is Poisson with the right mean, so
over many days it passes the N-test at close to the nominal rate, and its
spatial density is a reasonable long-run description, so it often passes the
S-test. It is *calibrated in the tested respects and completely unsharp*: it
never says "today is different". And it loses to ETAS by 6.006 nats/event
(§3.3), a factor of 406 in density. **A consistency test asks whether the
forecast is compatible with the data. A climatology is compatible with the data.
It is just useless.**

That asymmetry is the whole argument for reporting both: **likelihood** measures
sharpness-given-calibration and ranks models but cannot certify one as
acceptable; **CSEP consistency** measures calibration and can reject a model
outright but cannot tell you one is good.

---

## 6. The CSEP programme

### 6.1 Where it came from and why

Earthquake prediction has a long and unhappy history of claims that could not be
falsified: alarm windows specified loosely enough that almost any event counted
as a hit, magnitude thresholds adjusted after the fact, and evaluation performed
retrospectively by the claimant. The field's institutional response was to move
the evaluation *out of the modeller's hands*.

- **RELM** (Regional Earthquake Likelihood Models), a Southern California
  Earthquake Center project, invited groups to submit five-year forecasts for
  California on a fixed 0.1° grid with fixed magnitude bins, registered in
  advance, to be scored by an independent centre against an authoritative
  catalog. The testing methodology was set out in **Schorlemmer, Gerstenberger,
  Wiemer, Jackson & Rhoades (2007), *Earthquake likelihood model testing*,
  Seismological Research Letters 78(1):17–29** — the paper that defines the
  N-, L-, R- (and by extension M-, S-) tests. First results were published in
  Schorlemmer et al. (2010, *Pure and Applied Geophysics*).
- **CSEP** (Collaboratory for the Study of Earthquake Predictability)
  generalized RELM into a standing, international infrastructure: multiple
  testing centres (California/SCEC, Europe/ETH Zurich, New Zealand/GNS,
  Japan/ERI), a shared codebase, and automated prospective evaluation. The
  programmatic framing is set out in **T. H. Jordan (2006), *Earthquake
  predictability, brick by brick*, SRL 77(1):3–6, doi:10.1785/gssrl.77.1.3**
  (an editorial, not a methods paper — cite it for the philosophy, not for a
  test). The modern open-source implementation is **pyCSEP** (Savran, Bayona,
  Iturrieta, Asim, Bao, Bayliss, et al., 2022, *SRL* 93(5):2858–2870), which is
  the package this repository calls.

The four commitments that make CSEP work: **registered in advance** (the
forecast file exists before the evaluation window opens, so post-hoc tuning is
impossible); **fully specified** (rates for every space-magnitude-time bin, or a
simulator — no verbal hedging); **independent authoritative data** (the
evaluation catalog is drawn by the testing centre from a designated network
catalog at a fixed lag and completeness threshold); and **automated blind
scoring** (the modeller does not run the test).

### 6.2 Prospective, pseudo-prospective, retrospective

| mode | model fitted on | evaluation data | look-ahead risk |
|---|---|---|---|
| **Prospective** | data before `T0` | data after `T0`, *which did not exist when the model was registered* | none, structurally |
| **Pseudo-prospective** | data before `T0` | data after `T0`, already recorded | modeller has seen the world; selection pressure is real but the model is frozen |
| **Retrospective** | any data, including the evaluation window | any | severe |

The distinction is not pedantic. Pseudo-prospective evaluation is honest and
useful — Savran et al. (2020) is exactly that for UCERF3-ETAS on Ridgecrest —
but it cannot rule out the modeller having, consciously or not, selected among
architectures, seeds, or hyperparameters using knowledge of what happened.

**FlowQuake's status, stated plainly.** The 2020-01-17 → 2026 window is
described in the repository's own artifact as
"a retrospective out-of-time/pseudo-prospective replication, not a registered
prospective forecast" ([runs/total_win.json](../runs/total_win.json), `notes[0]`).
[REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) lists rung 4 —
"Prospective deployment: freeze a checkpoint and run rolling forecasts on a
future catalog window that was not used for model selection" — as **not done**.
Take that framing as your own; do not let an examiner discover it.

### 6.3 What look-ahead bias actually looks like

Enumerate these; a good examiner will ask you to. (1) Completeness `m_c` or the
b-value estimated on the full catalog including test. (2) Feature normalization
statistics computed over all data rather than train-only. (3) Declustering or
catalog cleaning that uses future events to classify a past one. (4)
Hyperparameter, seed, or early-stopping selection on the test window. (5) A
background rate map smoothed from a catalog that includes test events. (6)
Reporting the best of many runs where "best" was measured on test. (7) Data
revision leakage — using a catalog version whose magnitudes were revised *after*
the forecast date.

Where this repository sits on each:

- (2) is handled: normalization is computed on the training split
  ([STACK.md](../STACK.md) Part IV §7).
- The projection inverse `fit_xy_to_lonlat` in
  [flowquake/csep_forecast.py:44-69](../flowquake/csep_forecast.py#L44-L69) *is*
  fitted on the full catalog. The code's own defence is that it is a fixed
  geometric transform of coordinates, not a function of seismicity, with a max
  residual ~0.36 km against ~11 km CSEP cells. I find that defence sound: the
  map `(x,y) -> (lon,lat)` would be identical if you fitted it on synthetic
  points on a lattice.
- (4) and (6) are the live risk. [STACK.md](../STACK.md) Part V records that the
  neural-ETAS head's training script docstring "explicitly warns that grids and
  multiple seeds were run, so this is **not** a test-scored-once protocol".
  That is the repo flagging its own exposure, and you should repeat it rather
  than defend it.
- (5) applies by construction: the head uses a train-era smoothed-seismicity
  background map, acknowledged in [README.md](../README.md) and
  [REPRODUCE.md](../REPRODUCE.md).

---

## 7. Gridded vs catalog-based forecasts, and why FlowQuake must simulate

### 7.1 The two forecast formats

**Gridded (rate-based).** The forecast is a table of expected counts

```
    lambda_{ijk}  =  expected number of events in
                     spatial cell i, magnitude bin j, time bin k
```

Classically these are treated as independent Poisson, so the whole forecast is a
product of Poisson likelihoods and every test statistic is available in closed
form or by trivial simulation. **What the model must be able to do: emit an
expected rate per bin.**

**Catalog-based (simulation-based).** The forecast *is* a collection of `J`
synthetic catalogs `C_1, ..., C_J` drawn from the model's predictive law for the
window. **What the model must be able to do: simulate.** No parametric
assumption about the count distribution is needed, because the null distribution
of any statistic is just its empirical distribution over the `J` catalogs. This
is the format introduced for CSEP by **Savran, Werner, Marzocchi, Rhoades,
Jackson, Milner, Field & Michael (2020), *Pseudoprospective evaluation of
UCERF3-ETAS forecasts during the 2019 Ridgecrest sequence*, BSSA
110(4):1799–1817**, whose motivating point is that the Poisson assumption of the
gridded tests is *wrong* for clustered seismicity — an ETAS-like process is
strongly over-dispersed, and a Poisson null rejects it for the wrong reason.

### 7.2 Why FlowQuake cannot emit a gridded rate

This is a structural consequence of the modelling choice in
[STACK.md](../STACK.md) Part I §2, and you should be able to derive it.

FlowQuake models `f_t(tau | H)`, the conditional density of the waiting time to
the *next* event, not `lambda(t | H_t)`. The two are related. For `t` in the open
interval between the last observed event `t_last` and the next event, with
`tau = t - t_last`:

```
    lambda(t | H_t)  =  f_t(tau | H) / S_t(tau | H),      S_t = 1 - F_t
```

*Derivation.* `S_t(tau) = P(no event in (t_last, t_last + tau] | H)`. The hazard
of the waiting time is by definition `h(tau) = f_t(tau)/S_t(tau)`, and the
conditional intensity of a point process restricted to the interval before the
next event is exactly that hazard. Equivalently, integrating,
`S_t(tau) = exp(- integral_0^tau lambda(t_last + u | H) du)`, which is the
identity in [STACK.md](../STACK.md) Part I §2 read backwards.

So FlowQuake *does* have an intensity — **but only up to the next event.** The
moment an event occurs, the history changes and the intensity must be
re-derived from the new conditioning. To produce a gridded expected count for a
whole day you need

```
    E[ N(day) | H ]  =  integral_{day}  E[ lambda(u | H_u) ]  du
```

and the inner expectation is over *all possible continuation paths* — every
possible number, timing, location and magnitude of events earlier in the day,
each of which changes `H_u`. There is no closed form. ETAS escapes this only
because its intensity is *additive* over past events with kernels whose integrals
are closed form, so its expected daily rate given history can be written down
(modulo the branching contribution of within-day events, which ETAS's simulator
also handles by simulation).

**Conclusion: FlowQuake must Monte-Carlo.** That is not an implementation
choice; it follows from modelling `f(tau|H)` instead of `lambda`. The trade the
repo made is: exact temporal likelihood, no `integral lambda`, at the price of
sequential simulation for anything forecast-shaped.

### 7.3 Tracing the consequence through the code

[STACK.md](../STACK.md) Part VI walks `simulate_day_events`
([flowquake/ntest.py](../flowquake/ntest.py)) and its CSEP wrapper
([flowquake/csep_forecast.py](../flowquake/csep_forecast.py)) line by line. Here is
what the *theory* demands of them and where the approximations enter.

**(a) The truncated first gap — a correctness requirement, not an optimization.**
At the forecast origin `day_start`, the last observed event is at some
`t_last < day_start`, and we *know* nothing happened in between. So the first
simulated gap must be drawn from the conditional

```
    f( tau | tau > day_start - t_last )  =  f(tau) / S(day_start - t_last),
                                             for tau > day_start - t_last
```

not from `f(tau)`. [flowquake/ntest.py:88-104](../flowquake/ntest.py#L88-L104)
implements this by rejection: sample from `f`, keep only draws with
`t_last + tau >= day_start`. Rejection sampling from `f` with acceptance region
`{tau > a}` returns exactly a draw from `f(tau | tau > a)` — that is the whole
proof, and it is why the method is correct rather than approximate.

**Where it becomes approximate.** The acceptance probability per round is
`S(day_start - t_last)`, and the loop is capped at `MAX_REJECTION_ROUNDS = 200`.
Lanes that never accept are marked as having **no event that day**. If
`S = 0.02`, the probability a lane never accepts in 200 rounds is
`(1 - 0.02)^200 = 0.0176` — so ~1.8% of simulation lanes are *wrongly* assigned
zero events, biasing the simulated count distribution downward on days following
a long quiescence. That is a real, quantifiable, small bias in the null
distribution of the N-test, and it is not documented in
[MANUSCRIPT.md](../MANUSCRIPT.md).

**(b) The per-lane event cap.** `MAX_EVENTS_PER_DAY = 200`
([flowquake/ntest.py:28](../flowquake/ntest.py#L28)) right-truncates each simulated
catalog. In [runs/csep_h2h_fq/csep_results.json](../runs/csep_h2h_fq/csep_results.json)
the largest per-day `sim_mean` is **151.568** and the largest observed daily
count `n_obs` is **196**, both on day 4570 — which is `test_start` 2007-01-01
plus 4,570 days, i.e. **2019-07-07**, the day after the M7.1 Ridgecrest
mainshock. On that day the cap happens not to distort the N-test (a capped lane
still records 200 >= 196, so `delta_1 = P(N_sim >= 196)` is unaffected; the file
gives `[0.134, 0.891]`, a comfortable pass), but the general point stands:
**the simulated count distribution is right-truncated and no artifact records
how often the cap binds.**

**(c) Format and bookkeeping.** `fit_xy_to_lonlat` inverts the km projection so
simulated events can be written as `csep_ascii` rows, with `catalog_id` = the
simulation index (how pyCSEP recognizes `J` separate catalogs). The observed
catalog is reloaded fresh every day
([flowquake/csep_forecast.py:189-194](../flowquake/csep_forecast.py#L189-L194))
because pyCSEP's `.filter*` methods mutate in place — that comment documents a
real bug where each day's filters compounded and the observed set silently
shrank.

---

## 8. The consistency tests, defined

Setup for the catalog-based versions, which is what this repo runs. A forecast
is `J` simulated catalogs `C_1, ..., C_J` on a window; the observation is one
catalog `C_obs`. Every test is of the form: compute a statistic `T` on `C_obs`,
compute it on each `C_j`, and locate `T(C_obs)` in the empirical distribution
`{T(C_j)}`.

pyCSEP reports the location as a pair of quantiles:

```
    delta_1  =  P( T_sim >= T_obs )  =  (1/J) #{ j : T(C_j) >= T(C_obs) }
    delta_2  =  P( T_sim <= T_obs )  =  (1/J) #{ j : T(C_j) <= T(C_obs) }
```

Note `delta_1 + delta_2 = 1 + P(T_sim = T_obs) >= 1`. For a continuous statistic
they sum to 1; for a discrete one (counts) the excess is the tie probability,
and that excess is what makes the discrete test **conservative** (§11).

### 8.1 N-test (number)

- **H0:** the observed event count `N_obs` in the window is a draw from the
  forecast's count distribution.
- **Statistic:** `T = N`, the total number of events (above `m_c`, inside the
  region, inside the time window).
- **Quantiles:** `delta_1 = P(N_sim >= N_obs)`, `delta_2 = P(N_sim <= N_obs)`.
- **Reading:** `delta_1` near 0 means the observed count exceeded almost every
  simulation — the forecast **under-predicts**. `delta_2` near 0 means the
  forecast **over-predicts**. The docstring in
  [flowquake/ntest.py:8-9](../flowquake/ntest.py#L8-L9) states exactly this.
- **Rejection (classical and here):** two-sided,
  `min(delta_1, delta_2) < alpha/2` with `alpha = 0.05`.
- The gridded version (Zechar, Gerstenberger & Rhoades 2010, BSSA 100(3),
  doi:10.1785/0120090192) uses the same `delta_1, delta_2` but obtains the null
  from a Poisson with mean equal to the forecast's total rate.

### 8.2 M-test (magnitude)

- **H0:** the observed frequency-magnitude distribution is consistent with the
  forecast's.
- **Gridded version (Zechar et al. 2010).** Collapse the forecast over space,
  rescale the magnitude distribution so its total equals `N_obs`, and compute
  the Poisson log-likelihood of the observed magnitude histogram. Higher is
  better, so the classical rejection is **one-sided lower**: reject when the
  observed likelihood is in the lower tail (`kappa < alpha`).
- **Catalog-based version (what this repo runs) is a different statistic, and
  the difference matters.** `csep.core.catalog_evaluations.magnitude_test`
  builds a union histogram over all `J` simulated catalogs, rescales it to
  `N_obs` events, and computes for the observation

  ```
      d_obs = sum_k ( log10( Omega(k) + 1 ) - log10( Lambda_U_scaled(k) + 1 ) )^2
  ```

  over magnitude bins `k`, where `Omega` is the observed histogram. Each
  simulated catalog is rescaled to `N_obs` events the same way and gives one
  `D_j`; the null is `{D_j}`. The rescaling is what stops the M-test being a
  re-run of the N-test.
- **This statistic is a non-negative discrepancy, not a likelihood: larger is
  worse.** The committed artifacts confirm it — in
  [runs/csep_h2h_fq/csep_results.json](../runs/csep_h2h_fq/csep_results.json) every
  `M.observed` lies in `[0.056, 3.920]` and correlates with the observed daily
  count at `r = 0.95` (my computation), which is the signature of a
  squared-difference statistic on `log10(count + 1)` histograms and not of a
  log-likelihood (contrast the `S.observed` values, all negative, `[-8.24,
  -1.47]`).
- **Consequence for the rejection rule.** pyCSEP defines the quantile score as
  `gamma_m = P(D_j <= d_obs) = delta_2`, the same functional form as the
  S-test's `gamma_s` — but because the underlying statistic is oriented the
  opposite way, the tail that indicates a *bad* magnitude forecast is the
  **upper** one (`delta_1` small), not the lower. pyCSEP's own documentation
  states the quantile definition and does **not** state a rejection rule for the
  catalog-based tests, so the widely-repeated "M, S and CL are rejected
  one-sided lower" convention is inherited from the *gridded* tests, where all
  three statistics are likelihoods. I flag this as an unresolved convention
  rather than asserting a correction to the literature — but the orientation of
  the statistic is source-verified, and §8.5 shows what it does to the repo's
  M-test numbers.
- *Provenance:* the catalog-based description above is read from the pyCSEP
  `main` branch on GitHub (`csep/core/catalog_evaluations.py`,
  `csep/utils/stats.py`, `csep/utils/calc.py`) plus pyCSEP's "Theory of CSEP
  Tests" page. pyCSEP is not vendored in this repository and is not
  installed in the environment I ran the artifact arithmetic in, so I could not
  execute it against the committed forecasts; the version the repo actually ran
  may differ from `main`.

### 8.3 S-test (space)

- **H0:** the observed event *locations* are consistent with the forecast's
  spatial rate density.
- **Gridded version (Zechar et al. 2010).** Collapse the forecast over
  magnitude to a spatial rate `lambda_i` per cell; normalize it so that
  `sum_i lambda_i = N_obs`, which removes the count effect; compute the joint
  Poisson log-likelihood of the observed per-cell counts under the normalized
  rate; simulate catalogs from the normalized rate to get the null; report the
  quantile score `= P(L_sim <= L_obs)`. **Reject one-sided if it is `< alpha`.**
  (Notation warning: Zechar et al. 2010 write `zeta` for this S-test quantile and
  `kappa` for the M-test's, reserving `gamma` for the CL-test; pyCSEP writes
  `gamma_s` and `gamma_m`. Say which convention you are using.)
- **Catalog-based version (what this repo runs).** The forecast's expected
  spatial rate field is estimated from the simulated catalogs
  (`forecast.get_expected_rates(...)` at
  [flowquake/csep_forecast.py:187](../flowquake/csep_forecast.py#L187)). pyCSEP's
  `_compute_likelihood` then normalizes that field to sum to unity over cells
  and returns

  ```
      S_obs  =  (1 / N_obs) * sum_cells  n_i * ln( lambda_i / sum_j lambda_j )
  ```

  i.e. **the mean natural log of the normalized spatial probability at the cells
  where events actually landed** — a per-event mean, so it does not scale with
  `N_obs`, and higher is better. The null is the same statistic on each
  simulated catalog. (This is the `likelihood_norm` return value; the
  un-normalized Poisson log-likelihood, which keeps the `-expected_cond_count`
  term, is computed but is not what the S-test scores.)
- pyCSEP returns the `(delta_1, delta_2)` pair for this test, and the
  correspondence to the classical quantity is **`gamma_s = delta_2 =
  P(S_sim <= S_obs)`**. You can see the pair in the artifacts: e.g.
  `runs/n1_density/csep/csep_results.json`, day 96,
  `"S": {"quantile": [0.9715593331153972, 0.02844066688460281]}` — the two
  entries sum to 1.0 because the statistic is continuous. Note that the repo's
  own `csep_summary` docstring
  ([flowquake/csep_forecast.py:235-237](../flowquake/csep_forecast.py#L235-L237))
  says pyCSEP returns "a single gamma for the S-test"; the artifacts show it
  returns a pair, so the docstring is wrong on that point even though the code
  handles both shapes.

### 8.4 L-test and CL-test (not run in this repo)

- **L-test.** `H0`: the whole observed catalog (space, magnitude and rate
  jointly) is consistent with the forecast. Statistic: the joint log-likelihood
  of the observation under the forecast. **Problem:** it is dominated by the
  count. A forecast with the right spatial and magnitude structure but a
  slightly wrong rate fails L, and you cannot tell why.
- **CL-test (conditional likelihood).** The same statistic with the forecast
  rescaled so its total expected count equals `N_obs`, i.e. *conditioning on the
  observed number*. This isolates the space-magnitude distribution and is the
  test you should quote when you want "given that this many events happened, did
  they happen in the right places at the right sizes".
- **Neither is computed here.**
  [flowquake/csep_forecast.py:147](../flowquake/csep_forecast.py#L147) imports only
  `number_test`, `spatial_test`, `magnitude_test`. That is a defensible choice —
  N, S and M are the decomposition, and CL is close to S⊗M — but it is a gap
  relative to the RELM/CSEP standard suite, and you should say so rather than
  imply the full battery was run.

### 8.5 The repo's pass criterion, verified against the code

```python
# flowquake/csep_forecast.py:240-251  (the `passes` closure inside csep_summary,
#                                      which spans 233-258)
def passes(q):
    if q is None: return None
    if isinstance(q, (list, tuple)):
        if any(v != v for v in q) or min(q) < -0.5:   # NaN or (-1,-1) sentinel
            return None                                # -> excluded from denominator
        return bool(min(q) >= 0.025)
    if q != q or q < -0.5: return None
    return bool(q >= 0.025)
```

So the rule is: **a day passes iff every reported quantile is `>= 0.025`**, and
days where the test was not evaluable (NaN observed statistic, or pyCSEP's
`(-1,-1)` sentinel) are dropped from the denominator rather than counted as
failures. That is why the artifacts report `S 79/85`, not `S 79/100`.

**This is where I must flag a deviation from convention.** Applying
`min(delta_1, delta_2) >= 0.025` uniformly means the S- and M-tests are run
**two-sided**, whereas the classical CSEP convention for likelihood-type
statistics (Zechar et al. 2010) is **one-sided lower at `alpha = 0.05`**. For
the S-test — where the statistic really is a likelihood (§8.3) — the two rules
differ in both directions:

- On the classical failure tail (observed likelihood implausibly *low*) the
  repo's threshold is 0.025, i.e. **more permissive** than the conventional 0.05.
- The repo additionally rejects days where the observed likelihood is
  implausibly *high* (`delta_1 < 0.025`), which the classical test never does.

Recomputing both criteria from the committed per-day quantiles (my computation,
from `results[].S.quantile` in each file):

| run | S: repo 2-sided @0.025 | S: classical 1-sided @0.05 | fails via low tail (<0.025) | fails via high tail (<0.025) |
|---|---|---|---|---|
| `csep_h2h_fq` (FlowQuake N1, 10^3) | 82/85 | **79/85** | 0 | 3 |
| `csep_h2h_etas` (ETAS, 10^3) | 80/86 | **84/86** | 0 | 6 |
| `n1_density/csep_head` (full-history head, 10^3) | 79/85 | **77/85** | 4 | 2 |
| `n1_density/csep` (production, 10^4) | 85/91 | **80/91** | 4 | 2 |

**The head-to-head ranking flips.** Under the repo's criterion FlowQuake (82/85)
beats ETAS (80/86) on the S-test; under the classical criterion ETAS (84/86)
beats FlowQuake (79/85). Re-running the paired McNemar test on the 83 commonly
evaluable days under the classical criterion gives head 75 vs ETAS 81, 8
discordant split 1/7, **exact p = 0.0703** — marginal, and in ETAS's favour.

**The M-test is worse, and in a different way.** Because the catalog-based M
statistic is a discrepancy rather than a likelihood (§8.2), `delta_2` small
means the observed magnitude histogram sits *closer* to the forecast's union
histogram than a typical simulated catalog does — a "fits too well" day, not a
failure in any recognizable sense. Yet that is where almost every M-test
rejection in the repository comes from. Decomposing the repo criterion's M
failures by tail (my computation from `results[].M.quantile`):

| run | M: repo 2-sided @0.025 | fails via `delta_2 < 0.025` ("too good") | fails via `delta_1 < 0.025` ("too bad") |
|---|---|---|---|
| `csep_h2h_fq` | 89/92 | 3 | 0 |
| `csep_h2h_etas` | 87/92 | 4 | 1 |
| `n1_density/csep_head` | 90/92 | 2 | 0 |
| `n1_density/csep` | 90/92 | 2 | 0 |
| `final_s1555/csep` | 90/92 | 2 | 0 |
| `etas_csep_pod` | 73/92 | 19 | 0 |

32 of the 33 M-test failures across all six committed runs are low-discrepancy
days. Under an upper-tail-only rule at `alpha = 0.05` (`delta_1 >= 0.05`) the
counts become 90, 89, 87, 88, 89 and **92/92** respectively. Two things follow.
First, the repo's M pass rates are not measuring what the label suggests, and
the FlowQuake-vs-ETAS M ordering (89 vs 87) is not robust either — it becomes 90
vs 89 upper-tail-only. Second, `etas_csep_pod`'s M 73/92 is not an independent
magnitude-distribution defect at all: every one of its 19 failures is a
too-good-a-fit day, which is what you get when the *simulated* catalogs are so
small that their rescaled magnitude histograms are wildly erratic. That is a
downstream symptom of the same count deficit discussed in §11.6, not a second
piece of evidence.

Caveat, stated plainly: this paragraph rests on the orientation of the
catalog-based M statistic (source-verified) plus my inference about which tail
should reject (not settled by pyCSEP's documentation, §8.2). If you disagree
with the inference, the arithmetic still stands and the conclusion weakens to
"the repo's M-test rejections are concentrated in the tail whose interpretation
is least clear".

Two honest readings of the S-test choice, and you should offer both. *For the
repo:* a two-sided criterion is arguably more informative — ETAS's six high-tail failures mean its
simulated catalogs are spatially **more dispersed than reality** (real events
land in higher-rate cells than ETAS's own simulations do), which is precisely
the over-smoothing §4.4 of the manuscript claims the neural head fixes, and the
one-sided test is blind to it by construction. *Against the repo:* the criterion
is nonstandard, it is flagged as nonstandard nowhere in
[MANUSCRIPT.md](../MANUSCRIPT.md), [STACK.md](../STACK.md) or
[results/CLAIMS.md](../results/CLAIMS.md), and the ranking it produces is not robust
to the conventional one. Any sentence of the form "FlowQuake is at least as well
spatially calibrated as ETAS" ([MANUSCRIPT.md](../MANUSCRIPT.md) §4.2 table) must
carry that caveat.

---

## 9. Comparative tests: R, T, W — and McNemar

Consistency tests compare a model to the data. Comparative tests compare two
models to each other.

- **R-test (RELM, Schorlemmer et al. 2007).** Statistic: the log-likelihood
  ratio of model A to model B on the observed catalog. Null distribution:
  simulate catalogs *from B* and recompute. Awkward in practice — it is
  asymmetric (you must run it twice, once under each null), and its power
  depends on which model you simulate from.
- **T-test (Rhoades, Schorlemmer, Gerstenberger, Christophersen, Zechar &
  Imoto, 2011, *Efficient testing of earthquake forecasting models*, Acta
  Geophysica 59:728–747).** Replace the simulation-based R-test with the
  classical **paired t-test on the per-earthquake log-likelihood-ratio
  contributions**. The estimand is the *information gain per earthquake*,
  exactly the `IG` of §3, and the output is a confidence interval for it. This
  is the direct ancestor of what
  [flowquake/stats.py](../flowquake/stats.py) `paired_gain_summary` computes.
- **W-test (same paper).** The Wilcoxon signed-rank companion: distribution-free,
  tests the median of the paired differences, robust when the `d_i` are
  heavy-tailed or skewed — which log-likelihood differences invariably are.

**How FlowQuake relates.** The repo's paired per-event gain with a stationary
block bootstrap is the T-test's estimand computed with an **autocorrelation-
robust** variance rather than the t-test's independence assumption. That is
strictly stronger, and it is the right sentence to have ready: *"we report the
Rhoades et al. information gain per earthquake, but we do not use the paired
t-test's standard error, because the per-event differences are strongly
serially dependent within aftershock sequences; we use a stationary block
bootstrap (Politis & Romano 1994) with mean block length 50."*

**McNemar's test** is a different animal and is the one used for the CSEP
head-to-head. Given the *same* 83 forecast days scored by two models, each day
yields a paired binary outcome (pass/fail, pass/fail). Arrange them:

```
                       ETAS pass   ETAS fail
    model A pass           a           b
    model A fail           c           d
```

The concordant cells `a` and `d` carry no information about a *difference*
(both models did the same thing). Conditioning on the number of discordant
pairs `n = b + c`, under `H0` (the two models have equal probability of passing)
each discordant pair is equally likely to go either way, so

```
    b | (b + c = n)  ~  Binomial(n, 1/2)
```

and the two-sided exact p-value is

```
    p  =  min( 1,  2 * sum_{i=0}^{min(b,c)}  C(n, i) / 2^n )
```

That is the exact McNemar test. §13.2 works it numerically on the repo's
actual table.

---

## 10. Catalog-based CSEP: simulated catalogs as the null

Why the catalog-based formulation (Savran et al. 2020) is the natural home for a
generative model like FlowQuake:

1. **No Poisson assumption.** The gridded tests assume bin counts are Poisson.
   Clustered seismicity is strongly over-dispersed — variance far exceeds mean —
   so a Poisson null rejects a *correct* clustered forecast far above the
   nominal rate. The catalog-based null inherits whatever dispersion the model
   actually has.
2. **The format matches what the model can produce.** FlowQuake can simulate and
   cannot emit a rate field (§7.2). A gridded protocol would force a Monte-Carlo
   approximation of the rate *and then* impose a wrong parametric null on top.
3. **Any statistic is testable.** With `J` catalogs you get the null
   distribution of *anything* — largest event, number of M>=5, Ripley's K,
   the inter-event time distribution. The repo runs only N, S, M; an examiner
   may reasonably ask why you did not test the statistic you actually care
   about.
4. **Cost, and why matched budgets matter.** The quantile resolution is `1/J`,
   and `get_expected_rates` estimates the spatial rate field *from the
   simulations*, so its smoothness also depends on `J`. Comparing a
   10^4-catalog FlowQuake forecast against a 10^3-catalog ETAS forecast would
   confound model quality with Monte-Carlo budget. The head-to-head in §4.2 of
   [MANUSCRIPT.md](../MANUSCRIPT.md) holds `J = 1000` for both — that matching is
   its strongest methodological feature.

---

## 11. The fine print a professor will attack

### 11.1 "95/100 days pass" — is that good?

**Under a perfectly calibrated forecast and a continuous test statistic, each
day passes with probability exactly `1 - alpha = 0.95`.** So the *expected*
number of passes in 100 days is 95, with standard deviation
`sqrt(100 * 0.95 * 0.05) = 2.18`. A pass count of 95 is dead on nominal.
Anything in roughly 91–99 is within ±2 sd.

The corollary that catches people out: **a forecast that passes 100/100 is
suspicious, not excellent.** Either the test has no power against that forecast
(e.g. the predictive distribution is so wide that nothing is surprising), or the
statistic is discrete enough to be conservative, or the days are so dependent
that they are effectively one test. "Our model passed every test" is a claim to
interrogate, not celebrate.

Running the binomial arithmetic on the three matched-budget head-to-head runs
(my computation, repo pass criterion;
`P(X >= observed failures | Binomial(n_eval, 0.05))`):

| run | test | failures / evaluable | expected failures | P(X >= obs) |
|---|---|---|---|---|
| `csep_h2h_fq` | N | 5/100 | 5.00 | 0.564 |
| `csep_h2h_fq` | S | 3/85 | 4.25 | 0.804 |
| `csep_h2h_fq` | M | 3/92 | 4.60 | 0.844 |
| `csep_h2h_etas` | N | 3/100 | 5.00 | 0.882 |
| `csep_h2h_etas` | S | 6/86 | 4.30 | 0.260 |
| `csep_h2h_etas` | M | 5/92 | 4.60 | 0.490 |
| `n1_density/csep_head` | N | 5/100 | 5.00 | 0.564 |
| `n1_density/csep_head` | S | 6/85 | 4.25 | 0.252 |
| `n1_density/csep_head` | M | 2/92 | 4.60 | 0.948 |

**Not one of these deviates significantly from a perfectly calibrated forecast.**
That is simultaneously the strongest and the weakest thing you can say about the
CSEP section: every model tested here is consistent, and the tests had no power
to separate them.

### 11.2 The discrete N-test is conservative

For a continuous statistic the two-sided rule rejects with probability exactly
`alpha`. For the *count* statistic it does not, because
`delta_1 + delta_2 = 1 + P(N_sim = N_obs) > 1` whenever ties are likely — small
counts, few distinct values. The achieved rejection rate under `H0` is therefore
**below** the nominal 5%, so the null pass rate is *above* 0.95 and an observed
95/100 is, if anything, slightly worse than nominal rather than exactly on it.
The discrete test is biased toward passing; a 95% pass rate does not certify 95%
calibration.

Concrete case. On a day with `N_obs = 0`, `delta_1 = P(N_sim >= 0) = 1` always,
so the test can only ever reject through `delta_2 = P(N_sim = 0) < 0.025`. Real
example from [runs/csep_h2h_fq/csep_results.json](../runs/csep_h2h_fq/csep_results.json),
day 817 (= 2009-03-28): `n_obs` 0, `sim_mean` 6.511, quantile
`[1.0, 0.009]` — only 0.9% of the 1,000 simulated catalogs were empty, so the
model effectively declared "today will not be quiet" and was wrong. The same day
in the 10^4-catalog run gives `[1.0, 0.0086]`, confirming it is a model
property, not Monte-Carlo noise.

### 11.3 Multiplicity and dependence

**Multiplicity.** 3 tests × 100 days × 4 pipelines is ~1,200 hypothesis tests.
If the object you report is a **pass rate**, no correction is needed — you are
estimating a proportion, not making 100 decisions, and the right uncertainty
statement is a binomial CI (§11.1). If the object is **"the model passed all
three tests"**, that *is* a compound decision over a family and family-wise
error matters. [MANUSCRIPT.md](../MANUSCRIPT.md) phrases the result as pass rates,
which is the correct framing.

**Dependence between days.** Here the repo is in better shape than it
advertises. The 100 forecast days come from `np.linspace(0, n_test_days - 1,
100)` ([flowquake/csep_forecast.py:131](../flowquake/csep_forecast.py#L131)), and I
verified the committed day lists have consecutive differences of exactly 48 or
49 across `day` 0 → 4763. Omori decay makes triggering correlation between
windows 48 days apart negligible, so the *data* dependence across days is weak.

What is **not** independent: all 100 days share one trained checkpoint and one
fitted ETAS inversion. The pass counts are therefore correlated through the
model, and a binomial CI quantifies "how often does *this* checkpoint pass", not
"how often would a checkpoint from *this method* pass". Seed variation is a
separate and larger source of uncertainty, handled for the likelihood scores by
the 3-seed aggregation in
[runs/fullsuite_summary.json](../runs/fullsuite_summary.json) (`n: 3`) and **not
handled at all for CSEP**, where every result comes from a single checkpoint.

### 11.4 A pass rate is not a p-value

A pass rate has no null distribution attached to it until you supply one. Given
the (idealized) binomial null of §11.1 you can compute a p-value for "is this
pass rate consistent with 95%?" — that is the last column of the table above.
Reporting "S 79/85 (92.9%)" without that column invites the reader to compare
92.9% to 95% and conclude something. There is nothing to conclude: at `n = 85`
and `p = 0.95` the binomial standard deviation is `sqrt(85 * 0.95 * 0.05) = 2.0`
days, so 79 and the nominal 80.75 are less than one sd apart.

### 11.5 Failing to reject is not evidence of equality — and here the power was ~zero

This is the most important paragraph in the chapter.

[MANUSCRIPT.md](../MANUSCRIPT.md) §4.2, [README.md](../README.md),
[REPRODUCE.md](../REPRODUCE.md) and [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md)
all state that the full-history head's S-test is "statistically indistinguishable
from ETAS (McNemar exact p = 1.00)". I re-derived the table from the raw per-day
quantiles and confirm the arithmetic exactly: on 83 commonly evaluable days,
head 77 passes, ETAS 77 passes, 10 discordant days split 5/5, exact two-sided
p = 1.0000.

Now compute the power. With `b + c = 10` discordant pairs, the two-sided exact
McNemar p-value as a function of the split is:

| split (min side) | exact two-sided p |
|---|---|
| 0 / 10 | 0.00195 |
| 1 / 9 | 0.02148 |
| 2 / 8 | 0.10938 |
| 3 / 7 | 0.34375 |
| 5 / 5 | 1.00000 |

**The test could only have rejected at 5% if at most 1 of the 10 discordant days
had gone the "wrong" way.** In other words, the design was capable of detecting
only a near-total asymmetry. `p = 1.00` here carries essentially no evidential
weight for equality; it is the *maximum possible* p-value for this table and
would have arisen from a coin flip.

The honest statement is: *"we cannot detect a difference in S-test pass rates,
and with 10 discordant days we had power to detect only an overwhelming one."*
The affirmative claim "the models are equivalent" requires an **equivalence
test**: pre-specify a margin (e.g. "the pass-rate difference is within ±5
percentage points") and show the confidence interval lies inside it — the TOST
logic of [flowquake/stats.py](../flowquake/stats.py) `tost_equivalence`, covered in
Chapter 7. **The repo applies TOST rigorously to its likelihood gains and not at
all to its CSEP comparison.** That asymmetry is a real gap and you should name it
before anyone else does.

Two further audit findings you should be holding:

- **No code in this repository computes a McNemar p-value.**
  [scripts/audit_readiness.py:298](../scripts/audit_readiness.py#L298) computes the
  discordant *count* and then hard-codes the string `"McNemar p~1.0"` into its
  message at [line 314](../scripts/audit_readiness.py#L314).
  [STACK.md](../STACK.md) Part VII lists "McNemar's exact test (in
  `scripts/audit_readiness.py`)" among the repo's statistical machinery — that
  is **not accurate as written**; the number is correct (I verified it
  independently) but it is not computed by any committed code.
  [results/CLAIMS.md](../results/CLAIMS.md) row C19 makes the same observation
  ("states 'McNemar p~1.0' in prose but stores no p-value key").
- **Two of the committed CSEP summaries are stale.**
  [results/CLAIMS.md](../results/CLAIMS.md) finding A2 reports that the two
  10^4-catalog runs' stored `summary` blocks predate the current
  `csep_summary()` sentinel handling: `n1_density/csep` day 2982 records
  `S.quantile [-1.0, -1.0]` with `observed: NaN`, which current code excludes,
  so re-aggregating that file's own `results[]` yields **S 85/91 = 93.4%**, not
  the stored and manuscript-quoted **85/92 = 92.4%**. I reproduced the same
  off-by-one independently. The direction is *against* the author (the corrected
  rate is better), and the four 10^3-catalog runs are self-consistent with
  current code.

### 11.6 The N 73/100 → 97/100 story cannot be isolated

[MANUSCRIPT.md](../MANUSCRIPT.md) §4.2 attributes the first ETAS run's N-test
failure rate (73/100, in
[runs/etas_csep_pod/csep_results.json](../runs/etas_csep_pod/csep_results.json))
solely to the fitted inversion's cached source set not being re-conditioned on
post-`test_start` mainshocks — the fix documented at
[flowquake/etas_csep.py:80-89](../flowquake/etas_csep.py#L80-L89).

[results/CLAIMS.md](../results/CLAIMS.md) points out a confound: the pod file
records `n_sims: 10000` while the fixed run records `1000`, and `n_sims` is
passed straight through as pyCSEP's `n_cat`
([flowquake/csep_forecast.py:180](../flowquake/csep_forecast.py#L180)). If ~1,000
real catalogs were scored with `n_cat = 10000`, pyCSEP pads with ~9,000 *empty*
catalogs, which alone produces exactly the observed under-prediction signature.

My own forensics on the pod file support the confound rather than resolving it:
all 27 N-test failures occur via `delta_1 < 0.025` (observed count above nearly
every simulation), and the median `delta_2 = P(N_sim <= N_obs)` across the 100
days is **0.955** — the simulated count distribution is piled up at very low
counts. That is what padding with empty catalogs looks like. It is also what a
genuinely under-conditioned ETAS looks like. The pod file's depressed M rate
(73/92) is *not* independent corroboration of either story: §8.5 shows all 19 of
those M failures are low-discrepancy days, the signature of simulated catalogs
that are too small, i.e. the same count deficit seen twice. **The two endpoints
are supported by artifacts; the single stated cause is not.**

Two further complications, both from the repo's own files. Both `etas_csep_pod`
and `csep_h2h_etas` record `n_nonempty: 1` and `sim_mean: NaN` on every day,
because those fields are hard-coded placeholders in `--rerun` mode
([flowquake/csep_forecast.py:159-160](../flowquake/csep_forecast.py#L159-L160)), so
neither file can tell you how many catalogs were really scored. And
`csep_h2h_etas` carries an `n_sims_note` key that speaks directly to the
confound: *"actual ETAS forecasts simulated at 1000 sims/day (matched to
FlowQuake); earlier 10000 was the `--rerun` default arg, not the simulation
count."* Read carefully, that note is evidence **for** the padding mechanism, not
against it: it says the recorded `n_sims` can be a `--rerun` default rather than
a count of catalogs, and it is exactly that default which
[csep_forecast.py:180](../flowquake/csep_forecast.py#L180) hands to pyCSEP as
`n_cat`. The note is attached to the fixed run, not the pod run, and does not
retract the pod file's `n_sims: 10000`.

---

## 12. Alarm-based evaluation: Molchan, ROC, and why likelihood won

The older evaluation tradition treats a forecast as an **alarm function**
`a(s, t)`: threshold it, declare alarms over some fraction of the space-time
volume, and count what you catch.

- `tau` (overloaded again — here the *alarm fraction*): the fraction of the
  space-time volume under alarm.
- `nu`: the **miss rate**, the fraction of target events not inside any alarm.

**Molchan diagram (Molchan, 1990, *Strategies in strong earthquake prediction*,
Physics of the Earth and Planetary Interiors 61:84–98; the "error diagram").**
Plot `nu` against `tau` as the threshold sweeps. Random guessing gives the
diagonal `nu = 1 - tau`; a perfect predictor reaches `(0, 0)`; skill is a
trajectory *below* the diagonal.

**Area skill score (Zechar & Jordan, 2008, *Testing alarm-based earthquake
predictions*, GJI 172(2):715–724).** The normalized area *above* the Molchan
trajectory. 1 = perfect skill, 0 = perfect non-skill, **1/2 is the expected
score for a random alarm function**. It is closely related to the ROC AUC and
coincides with it in the rare-event limit, where target events occupy a
negligible fraction of the space-time volume so that "fraction of volume under
alarm" and "false-alarm rate among non-event units" become the same quantity —
which is the usual seismological regime, but is an approximation, not an
identity.

**ROC** plots hit rate against false-alarm rate; the difference from Molchan is
what goes on the x-axis — Molchan uses the fraction of space-time occupied
(the natural cost when alarms are spatially extended), ROC uses the false-alarm
rate among non-event units.

**Why likelihood-based evaluation is preferred for point processes:**

1. **Alarm scores are rank-only, therefore not proper.** ASS and AUC depend on
   the alarm function only through the *ordering* it induces, so a forecaster
   can report any strictly increasing transform of their true probabilities —
   `p^3`, `sqrt(p)`, anything — and score identically. The reported *magnitudes*
   of the probabilities are unconstrained, so the evaluation cannot certify
   them. There is no incentive-compatibility guarantee.
2. **Binarization discards information.** You must pick a target magnitude and a
   window; count structure and the magnitude distribution are thrown away. A
   point process is a joint law over times, locations and marks; an alarm score
   sees a binary field.
3. **The cost ratio is implicit.** Choosing a point on the Molchan curve is
   choosing a miss/alarm-cost trade-off. Reporting the curve dodges the choice;
   reporting one point smuggles it in.
4. **The honest counterpoint.** Alarm-based evaluation is *closer to the actual
   decision* — a civil-protection agency does not consume a density, it decides
   whether to warn. The right answer to "why not Molchan?" is "because we are
   comparing models, not choosing an alarm policy — but a decision-relevant
   evaluation, or a proper score weighted toward the outcome region you care
   about, is a legitimate separate exercise we did not do."

---

## 13. Worked examples

### 13.1 N-test `delta_1`, `delta_2` by hand

Suppose a forecast produced `J = 20` simulated catalogs whose event counts are

```
    N_sim = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9, 12]
```

**Case A: `N_obs = 9`.**

```
    #{ N_sim >= 9 }  =  |{9, 12}|         =  2   ->  delta_1 = 2/20  = 0.100
    #{ N_sim <= 9 }  =  20 - 1 (the 12)   = 19   ->  delta_2 = 19/20 = 0.950
    min(delta_1, delta_2) = 0.100  >=  0.025      ->  PASS
```

Sanity check the identity: `delta_1 + delta_2 = 1.050 = 1 + 1/20`, and indeed
exactly one simulated catalog ties `N_obs`. Good.

**Case B: `N_obs = 13`.**

```
    #{ N_sim >= 13 } = 0   ->  delta_1 = 0.000
    #{ N_sim <= 13 } = 20  ->  delta_2 = 1.000
    min = 0.000  <  0.025  ->  FAIL, by UNDER-PREDICTION
```

The observed count exceeded every simulation: the model did not think a day that
busy was possible.

**Case C: `N_obs = 12`.**

```
    delta_1 = 1/20 = 0.050,   delta_2 = 20/20 = 1.000,   min = 0.050  ->  PASS
```

**The lesson hiding in Case C.** With `J = 20`, `delta_1` can only take values
`k/20`, so the smallest non-zero value is 0.05. The test can therefore *never*
reject via `delta_1` except when `delta_1` is exactly 0 — the criterion
`delta_1 < 0.025` is unreachable otherwise. In general you need `k/J < 0.025`
for some `k >= 1`, i.e. **`J > 40`** for the threshold to be attainable at all,
and `J` in the hundreds to thousands for the estimate to be stable. This is why
the repo runs `J = 1000` and `J = 10000` and why the matched-budget design of
§10 matters.

**Cross-check against a real record.** From
[runs/csep_h2h_fq/csep_results.json](../runs/csep_h2h_fq/csep_results.json),
day 2934 (= 2015-01-13): `n_obs` 16, `sim_mean` 3.639, `n_nonempty` 946 of
`n_sims` 1000, `N.quantile [0.002, 0.999]`. Read it: only 0.2% of the 1,000
simulated catalogs had 16 or more events, so `delta_1 = 0.002 < 0.025` — a
**failure by under-prediction**. The model expected ~3.6 events and 16 happened.

**A worked M-test record, and what its denominator tells you.** Same file, day
481 (= 2008-04-26): `n_obs` 43, `sim_mean` 7.125, `n_nonempty` 995 of 1000,
`M.quantile [0.9909547738693467, 0.009045226130653266]`. Multiply through: those
are exactly `986/995` and `9/995`. So the M-test null contains **995** values,
not 1000 — pyCSEP skips empty simulated catalogs when building the magnitude
test distribution (it has to; an empty catalog has no histogram to rescale), and
the same 995 shows up in that day's `S.quantile` (`0.1678… = 167/995`). Two
lessons. First, the effective Monte-Carlo budget for the S- and M-tests is
`n_nonempty`, not `n_sims`, and on quiet days it can be much smaller. Second,
read the day as a whole: `N.quantile [0.011, 0.989]` is a clear under-prediction
failure (43 events against a simulated mean of 7.1), and the M-test's
`delta_2 = 0.009` is *also* counted as a failure by the repo criterion — but it
says the observed magnitude histogram matched the forecast's union histogram
*better* than 99% of the simulated catalogs did (§8.5). One badly-forecast day,
two "failures", pointing in opposite directions.

### 13.2 The 2×2 discordance table and the exact McNemar p-value

Take the repo's headline paired comparison: the full-history neural-ETAS head
versus ETAS, S-test, on the days where both are evaluable. Sources:
[runs/n1_density/csep_head/csep_results.json](../runs/n1_density/csep_head/csep_results.json)
and [runs/csep_h2h_etas/csep_results.json](../runs/csep_h2h_etas/csep_results.json),
scoring each day with the repo's own criterion `min(quantile) >= 0.025` and
dropping non-evaluable days.

83 days are commonly evaluable. The table (my recomputation from
`results[].S.quantile`):

```
                        ETAS pass   ETAS fail   | total
    head pass               72           5      |   77
    head fail                5           1      |    6
    -------------------------------------------+------
    total                   77           6      |   83
```

Check the margins: head passes `72 + 5 = 77`; ETAS passes `72 + 5 = 77`;
concordant `72 + 1 = 73`; discordant `5 + 5 = 10`; total `73 + 10 = 83`. ✔

Now the exact test. Condition on `n = b + c = 10` discordant pairs. Under `H0`,
`b ~ Binomial(10, 1/2)`. With `b = 5`, `min(b, c) = 5`:

```
    sum_{i=0}^{5} C(10, i)  =  1 + 10 + 45 + 120 + 210 + 252  =  638
    P(X <= 5)               =  638 / 1024                      =  0.62305
    p_two-sided             =  min(1, 2 * 0.62305)             =  1.00000
```

**p = 1.0000.** Which, given that `b = c = n/2` exactly, is the largest value the
test can return — the observation is precisely the null's centre. §11.5 explains
why this is not evidence of equality.

For contrast, the *production* kernel-mixture head versus ETAS on the same 83
days:

```
                        ETAS pass   ETAS fail   | total
    FQ pass                 76           4      |   80
    FQ fail                  1           2      |    3

    n = b + c = 4 + 1 = 5,   min(b,c) = 1
    sum_{i=0}^{1} C(5, i)  =  1 + 5  =  6
    p  =  min(1, 2 * 6/32)  =  12/32  =  0.375
```

And the manuscript's third comparison — full-history head versus production
kernel-mixture head, 81 commonly evaluable days, 75 vs 78 passes, 9 discordant
split 3/6 — gives `p = 2 * (C(9,0)+C(9,1)+C(9,2)+C(9,3))/2^9 = 2*130/512 =
0.5078`, matching [results/CLAIMS.md](../results/CLAIMS.md) row C20 exactly
(the manuscript rounds it to 0.51).

Five lines of Python to redo all of it:

```python
import json
from math import comb
ok  = lambda q: None if (any(v!=v for v in q) or min(q)<-0.5) else min(q) >= 0.025
day = lambda f: {r["day"]: ok(r["S"]["quantile"]) for r in json.load(open(f))["results"] if "S" in r}
A, B = day("runs/n1_density/csep_head/csep_results.json"), day("runs/csep_h2h_etas/csep_results.json")
sh = [d for d in A if A[d] is not None and B.get(d) is not None]
b, c = sum(A[d] and not B[d] for d in sh), sum(B[d] and not A[d] for d in sh)
print(len(sh), b, c, min(1, 2*sum(comb(b+c, i) for i in range(min(b, c)+1))/2**(b+c)))
# -> 83 5 5 1.0
```

### 13.3 Information gain, end to end

Take `dTot.mean = 0.1133` from
[runs/total_win.json](../runs/total_win.json) (`test_2007_2020`), with
`n = 21889`.

```
    density ratio (geometric mean)  =  exp(0.1133)          =  1.1200
    bits per event                  =  0.1133 / ln 2        =  0.1635
    events per bit of evidence      =  ln 2 / 0.1133        =  6.12
    total log-likelihood ratio      =  21889 * 0.1133       =  2480 nats
    reported 95% block-bootstrap CI =  [0.1006, 0.1268]     (same file)
```

Sanity-check the components against the same file: `fq_nll` 7.142121886887271,
`etas_nll` 7.255427552750566, difference 0.113305665863295 → rounds to the
stored `dTot.mean` 0.1133. ✔ And `dT.mean` 0.0533 + `dS.mean` 0.0600 = 0.1133 ✔
(the total is exactly the sum of the temporal and spatial gains because
`nll = -(tll + sll)` — the magnitude term is excluded by the EarthquakeNPP
convention).

Say the sentence from §3.3, then immediately say: "and the 2,480-nat total is a
plug-in likelihood ratio over serially dependent events, so the CI on the
per-event gain is the number I would defend."

---

## 14. How this shows up in FlowQuake

[STACK.md](../STACK.md) Part VI is the code walkthrough; this section is the map
from theory to artifact. Nothing here restates the code.

| theory (this chapter) | code | artifact |
|---|---|---|
| log score, per-event `tll`/`sll`/`mll` (§2) | [flowquake/evaluate.py](../flowquake/evaluate.py) | `runs/*/eval_test.json`, `runs/fullsuite_summary.json` |
| information gain, paired (§3–§4) | [flowquake/stats.py](../flowquake/stats.py) `paired_gain_summary` | `runs/total_win.json`, `runs/stats_hardening.json` |
| ETAS density re-implementation certified (§2.2) | `scripts/etas_sll_repro.py` | `runs/etas_sll_repro.json` (`max_abs_sll_err` 1.766e-09) |
| simulation because there is no rate field (§7) | [flowquake/ntest.py](../flowquake/ntest.py) `simulate_day_events` | only two `ntest.json` files are committed (`runs/comcat25_s1555/`, `runs/smoke/`); the CSEP runs consume `simulate_day_events` directly |
| catalog-based N/S/M (§8, §10) | [flowquake/csep_forecast.py](../flowquake/csep_forecast.py) | `runs/n1_density/csep/csep_results.json`, `runs/csep_h2h_fq/…` |
| same harness, ETAS side (§10) | [flowquake/etas_csep.py](../flowquake/etas_csep.py) | `runs/csep_h2h_etas/csep_results.json` |
| same harness, neural-ETAS head (§10) | [flowquake/csep_forecast_head.py](../flowquake/csep_forecast_head.py) | `runs/n1_density/csep_head/csep_results.json` |
| pass criterion (§8.5) | `csep_summary`, [csep_forecast.py:233-258](../flowquake/csep_forecast.py#L233-L258) | the `summary` block of every `csep_results.json` |
| paired pass/fail comparison (§9, §13.2) | [scripts/audit_readiness.py:286-317](../scripts/audit_readiness.py#L286-L317) | `runs/replacement_readiness.json` (counts only; **no p-value key**) |

**The three-pipeline design** — three different forecast producers writing
byte-compatible `csep_ascii` CSVs, scored by one `csep_summary` on the same days
at the same `J` — is tabulated in [STACK.md](../STACK.md) Part VI and not repeated
here. What matters statistically is that the matched budget removes `J` as a
confound (§10), and that [results/CLAIMS.md](../results/CLAIMS.md) row C13 verifies
the `results[].day` lists are element-for-element identical across the four runs
it checks. I extended the check: all **six** committed `csep_results.json` files
(adding `final_s1555/csep` and `etas_csep_pod`) carry the identical day list
`np.linspace(0, 4763, 100, dtype=int)`, so "identical 100 forecast days" holds
literally and across the pre-fix run too.

**The actual numbers, read from the artifacts.**

Standalone, production N1 model, 100 days × 10^4 catalogs
([runs/n1_density/csep/csep_results.json](../runs/n1_density/csep/csep_results.json),
`summary`):

| test | stored | recomputed with current `csep_summary` |
|---|---|---|
| N | 95/100 (95.0%) | 95/100 |
| S | 85/92 (92.4%) | **85/91 (93.4%)** — day 2982 is the `(-1,-1)` sentinel |
| M | 90/92 (97.8%) | 90/92 |

Base (non-density-adaptive) seed 1555
([runs/final_s1555/csep/csep_results.json](../runs/final_s1555/csep/csep_results.json),
byte-identical to the loose [runs/csep_results_s1555.json](../runs/csep_results_s1555.json)):
N 95/100, S 81/92 (88.0%; recomputed 81/91 = 89.0%), M 90/92. The manuscript's
"S 88% → 92%" density-adaptive improvement is this pair, and both halves carry
the same +1 denominator inflation.

Matched 10^3-catalog head-to-head, identical 100 days:

| test | FlowQuake N1 (`csep_h2h_fq`) | ETAS (`csep_h2h_etas`) | full-history head (`csep_head`) |
|---|---|---|---|
| N | 95/100 | 97/100 | 95/100 |
| S | 82/85 | 80/86 | 79/85 |
| M | 89/92 | 87/92 | 90/92 |

Paired McNemar (exact, two-sided), my recomputation, repo criterion:

| comparison | shared days | passes | discordant | p |
|---|---|---|---|---|
| head vs ETAS, S | 83 | 77 vs 77 | 5 / 5 | **1.0000** |
| FQ vs ETAS, S | 83 | 80 vs 77 | 4 / 1 | 0.3750 |
| head vs FQ, S | 81 | 75 vs 78 | 3 / 6 | 0.5078 |
| head vs ETAS, N | 100 | 95 vs 97 | 1 / 3 | 0.6250 |
| head vs ETAS, M | 92 | 90 vs 87 | 3 / 0 | 0.2500 |

The pre-fix ETAS run
([runs/etas_csep_pod/csep_results.json](../runs/etas_csep_pod/csep_results.json)):
N 73/100, S 61/63, M 73/92 — see §11.6 for why the attributed cause is not
isolable, and §8.5 for why the M row is a restatement of the N row rather than a
second finding.

**What "statistically indistinguishable from ETAS" licenses.** It licenses:
*"we ran the incumbent through our own harness on identical days at a matched
budget, and we cannot detect a difference in consistency-test pass rates."* It
does **not** license *"the models are equivalently calibrated"*, because (i) the
test had power only against a near-total asymmetry (§11.5), (ii) no equivalence
margin was pre-specified and no TOST was run on the CSEP comparison (contrast
Chapter 7, where TOST *is* applied to the likelihood gains and is what gates the
"ties ETAS" language for Japan and Greece), and (iii) the S-test ranking is not
robust to using the classical one-sided criterion (§8.5).

---

## 15. Common misconceptions

1. **People think** a lower NLL means the model is better calibrated.
   **Actually** NLL measures sharpness *given* calibration, aggregated. A model
   can win on NLL while being systematically miscalibrated on a minority of
   days, and a climatology can be perfectly calibrated while losing by 6
   nats/event. **Why it matters:** it is the entire reason this repo runs both
   likelihood scoring and CSEP; if you conflate them you cannot explain why §4.1
   and §4.2 of the manuscript are separate results.

2. **People think** passing all CSEP tests means the forecast is good.
   **Actually** a consistency test only fails to reject; a wide, vague forecast
   passes everything. **Why it matters:** 100/100 is a red flag for power, not a
   badge. Under perfect calibration you should fail ~5% of days.

3. **People think** the log score cares how far the miss was. **Actually** it is
   0-local: it sees only the density value at the observed point. Two spatial
   models with identical `sll` can have wildly different typical miss distances.
   **Why it matters:** the spatial claim in this repo is entirely `sll`-based;
   no distance-sensitive score (CRPS, energy score) is reported anywhere.

4. **People think** `delta_1` and `delta_2` are two independent p-values.
   **Actually** they are the two tails of one location statistic and they
   satisfy `delta_1 + delta_2 = 1 + P(tie)`. **Why it matters:** the two-sided
   test is `min(delta_1, delta_2) < alpha/2`, one decision, not two.

5. **People think** the S-test is two-sided. **Actually** the classical CSEP
   S-test (and the gridded M-, L-, CL-tests) is one-sided lower; only the N-test
   is conventionally two-sided. This repo applies a two-sided rule to all three,
   and §8.5 shows the head-to-head S-test ranking flips under the classical
   rule. **Why it matters:** it is the single most attackable methodological
   choice in the CSEP section.

6. **People think** the catalog-based M-test is just the gridded M-test with a
   simulated null. **Actually** its statistic is a cumulative squared difference
   between `log10(histogram + 1)` vectors, which is non-negative and *larger is
   worse* — the opposite orientation to a likelihood (§8.2). **Why it matters:**
   32 of the 33 M-test rejections in this repository are on the "fits better
   than the simulations" tail (§8.5), so the M pass rates do not mean what their
   label suggests.

7. **People think** `p = 1.00` on a McNemar test is strong evidence of
   equivalence. **Actually** it is the *maximum possible* p-value, achieved when
   the discordant days split exactly evenly, and with 10 discordant pairs the
   test could only reject on a 0/10 or 1/9 split. **Why it matters:** absence of
   evidence is not evidence of absence; equivalence requires TOST with a
   pre-specified margin.

8. **People think** multiplying information gain by N gives a Bayes factor.
   **Actually** it gives a plug-in log-likelihood ratio, and treating 21,889
   serially dependent events as independent evidence inflates it enormously.
   **Why it matters:** quote the per-event gain and its block-bootstrap CI, not
   `e^2480`.

9. **People think** FlowQuake could just emit a gridded rate forecast if someone
   wrote the code. **Actually** it models `f(tau | H)` rather than
   `lambda(t|H_t)`; the hazard identity recovers the intensity only up to the
   next event, and a daily expected count is a path integral over all
   continuations. Simulation is structural, not lazy. **Why it matters:** it
   explains the whole architecture of `ntest.py` and its Monte-Carlo cost.

10. **People think** "out-of-time evaluation" and "prospective evaluation" are the
    same. **Actually** out-of-time on already-recorded data is
    *pseudo-prospective*; the model is frozen but the modeller has seen the world.
    The repo says this about itself in
    [runs/total_win.json](../runs/total_win.json) `notes[0]`. **Why it matters:**
    rung 4 of [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) is explicitly
    not done, and claiming otherwise is the fastest way to lose a viva.

11. **People think** the CSEP pass rates carry seed uncertainty like the
    likelihood tables do. **Actually** the likelihood headline is a 3-seed mean
    ([runs/fullsuite_summary.json](../runs/fullsuite_summary.json) `n: 3`); every
    CSEP result comes from a **single checkpoint**. **Why it matters:** the
    binomial CI on a pass rate quantifies day-to-day noise for *this* checkpoint
    and says nothing about seed-to-seed variability.

---

## 16. Questions a professor will ask

**Q1. Define a proper scoring rule and prove the log score is proper.**
A scoring rule `S(F, y)` rewards a forecast distribution `F` when `y`
materializes; it is proper on a class `P` if `S(G,G) >= S(F,G)` for all
`F, G in P`, strictly if equality implies `F = G`. For the log score,
`S(G,G) - S(F,G) = E_g[log g - log f] = KL(g||f)`, and Jensen on the concave
`log` gives `-KL = E_g[log(f/g)] <= log E_g[f/g] = log integral_{g>0} f <= 0`,
with equality iff `f = g` a.e. The clause that carries the weight is
`integral f <= 1`: propriety requires the reported density to be genuinely
normalized.

**Q2. What does "the log score is the essentially unique local proper rule"
mean?** Bernardo (1979, AoS 7:686–690): among smooth, local, proper rules, all
are `a log f(y) + b(y)` with `a > 0`. "Essentially" = up to positive affine
transformation with an outcome-dependent additive term. `a` is a unit change
(nats vs bits); `b(y)` cancels in any comparison of two forecasts of the same
outcome. Caveats: on a discrete space you need at least three outcomes for the
statement to have content, and if you weaken 0-locality to `m`-locality (density
plus `m` derivatives) a large class exists for even `m >= 2` — that is where
score matching lives.

**Q3. Your spatial gain is +0.060 nats/event but the win rate is 0.4972 — your
model is worse on most events. Explain.**
Correct, and it is in the artifact
([runs/total_win.json](../runs/total_win.json) `test_2007_2020.dS.win_rate`). The
mean and the median disagree because the gains are asymmetric: the head loses a
little on many isolated background events and wins a lot on clustered ones,
where the spatial structure is actually informative. The estimand I claim is the
mean paired log-likelihood ratio — that is what accumulates into evidence and
what the T-test/IGPE tradition targets. But you are right that a *median*-based
claim would fail, and the honest framing is "a better forecast on average, not a
uniformly better forecast". A Wilcoxon signed-rank (the W-test of Rhoades et al.
2011) would be the appropriate companion statistic, and it is not reported here.

**Q4. Why do you need CSEP at all if you already win on likelihood?**
Because the two answer different questions (§5.3). Likelihood ranks; it cannot
certify. Consistency tests can reject a model outright and can catch structured
defects that averaging hides — as the first ETAS run in this repo demonstrates
(perfectly ordinary likelihoods, N-test failure on 27 of 100 days).

**Q5. You report `S 79/85`. Where did the other 15 days go?**
Days on which the test was not evaluable — pyCSEP returns NaN or the `(-1,-1)`
sentinel, chiefly when the observed catalog has no events inside the region
after filtering. `csep_summary` drops them from the denominator rather than
counting them as failures
([csep_forecast.py:244-247](../flowquake/csep_forecast.py#L244-L247), the sentinel
branch that returns `None`, filtered out at
[:255](../flowquake/csep_forecast.py#L255)). That is the right
call, but it does mean the denominator varies per test and per model, so the
pass rates are not on identical day sets. For the paired comparison I intersect
to the 83 days where both models are evaluable.

**Q6 (hostile). Your S-test criterion is not the CSEP S-test. You use a
two-sided `min(delta) >= 0.025` where the literature uses one-sided lower at
0.05. Did your conclusion survive the conventional rule?**
It does not survive intact (§8.5). Under the classical rule FlowQuake N1 goes
82/85 → 79/85 and ETAS goes 80/86 → **84/86**; the paired head-vs-ETAS McNemar
goes from 77–77 (p = 1.00) to 75–81, 8 discordant split 1/7, **p = 0.0703** —
marginal, against us. The defence: the two-sided criterion is the more
informative one for this claim, because ETAS's high-tail rejections mean its
simulated catalogs are spatially *more dispersed than reality*, exactly the
over-smoothing the neural head is meant to fix, and the one-sided test is blind
to that by construction. What I would not claim is that the criterion is
standard. It is not, it is nowhere flagged as nonstandard, and the S-test result
should be reported under both rules.

**Q6b (hostile). What is your M-test actually testing? Its statistic is not a
likelihood.**
Correct, and this is the sharpest version of the previous question. pyCSEP's
catalog-based M-test statistic is `sum_k (log10(Omega(k)+1) -
log10(Lambda_U_scaled(k)+1))^2` — a non-negative discrepancy in which larger is
worse (§8.2). Applying `min(delta_1, delta_2) >= 0.025` to it therefore rejects
days on which the observed magnitude histogram matched the forecast *better*
than the simulated catalogs did, and that is where 32 of the 33 M-test failures
in this repository come from (§8.5). Concretely, `csep_h2h_fq` M 89/92 becomes
90/92 under an upper-tail-only rule, ETAS 87/92 becomes 89/92 — the ordering
reverses — and the pre-fix ETAS run's alarming M 73/92 becomes 92/92, i.e. the
M-test never had an independent complaint about that run at all. What I will not
claim is that the field has settled which tail rejects: pyCSEP documents the
quantile score `gamma_m = P(D_j <= d_obs)` and states no rejection rule for the
catalog-based tests, and the "one-sided lower" convention everyone quotes comes
from the gridded tests, where the statistic genuinely is a likelihood. The
defensible position is that the M-test numbers in this repository should not be
reported as a headline until that convention is pinned down.

**Q7 (hostile). You claim the head is "statistically indistinguishable from
ETAS" on the basis of a McNemar p of 1.00. What is the power of that test?**
Essentially none (§11.5). With 10 discordant days the exact test rejects at 5%
only on a 0/10 or 1/9 split; 2/8 already gives p = 0.109, and p = 1.00 is the
largest value the table can return. The honest claim is "we cannot detect a
difference", not "there is none". Equivalence needs a pre-specified margin and a
TOST — a discipline the repo applies to its likelihood gains
([flowquake/stats.py](../flowquake/stats.py) `tost_equivalence`) and not here. The
fix is cheap: pre-register a pass-rate margin, bootstrap a paired CI on the
pass-rate difference over days, report whether it sits inside the margin.

**Q8 (hostile). Two of your committed CSEP summaries disagree with your own
aggregation code. Explain.**
Correct, and it is the repo's own audit finding A2, which I reproduced
independently (§11.5, §14). The two 10^4-catalog runs' stored `summary` blocks
predate the current sentinel handling; re-aggregating `n1_density/csep` from its
own `results[]` gives S 85/91 = 93.4%, not the stored and manuscript-quoted
85/92 = 92.4%. The direction is against the author, and the four 10^3-catalog
runs are self-consistent with current code.

**Q9 (hostile). Your ETAS N-test improvement, 73/100 → 97/100, is attributed to
a source-set conditioning bug. But the two runs also differ in `n_sims` (10,000
vs 1,000), which pyCSEP consumes as `n_cat`. Did you fix a science bug or a
padding artifact?**
The artifacts cannot settle it (§11.6). Padding ~1,000 real catalogs out to
`n_cat = 10000` with empty ones produces exactly the observed signature, and my
forensics are consistent with it: all 27 failures occur via `delta_1 < 0.025`
and the median `delta_2` is 0.955. Both files also record placeholder
`n_nonempty: 1` / `sim_mean: NaN` in `--rerun` mode, so they cannot report how
many catalogs were really scored. The endpoints are real; the causal attribution
is not established, and the clean experiment — re-run the pre-fix source set at
`n_sims = 1000` — has not been done.

**Q10 (hostile). None of your CSEP pass rates differs significantly from 95%,
for either model. So what did the CSEP section establish?**
That both models are consistent, and that the tests lacked power to separate
them. That is genuine but modest: it rules out the specific objection that the
head buys spatial likelihood by concentrating probability so hard its simulated
catalogs become implausible. It does not establish superiority. The quantitative
version of "no power" is the binomial table in §11.1 — for every run,
`P(X >= observed failures | Bin(n, 0.05))` exceeds 0.25.

**Q11. Why can FlowQuake not produce a gridded forecast?**
Because it models `f(tau | H)` rather than `lambda(t | H_t)`. The hazard identity
`lambda = f/(1-F)` recovers the intensity only on the interval before the next
event; a daily expected count requires `E[integral lambda]` over all
continuation paths, which has no closed form once a within-day event can change
the history. So a rate over any window with a non-negligible chance of
containing events must be Monte-Carloed. ETAS avoids this because its intensity
is additive with closed-form kernel integrals.

**Q12. Your simulator caps each lane at 200 events per day. Does that matter?**
It is a right-truncation of the simulated count distribution, and no artifact
records how often it binds. The worst case in the committed runs is day 4570
(2019-07-07, the day after the Ridgecrest M7.1): `sim_mean` 151.568, observed
count 196. On that day the cap happens not to bias the N-test, because a capped
lane still records 200 >= 196 and `delta_1 = P(N_sim >= 196)` is unaffected. But
if an observed count had exceeded 200 the test would have been mechanically
unable to see it. The clean fix is to log the fraction of capped lanes per day
and raise the cap where it binds.

**Q13. The first simulated event is rejection-sampled. Why, and what does it
cost?**
Because at the forecast origin we know no event occurred between `t_last` and
`day_start`, so the first gap must come from the truncated conditional
`f(tau | tau > day_start - t_last)`; rejection sampling from `f` with acceptance
set `{tau > a}` is exactly a draw from that conditional. Skipping it would
systematically over-forecast the first event of every day. The cost is that the
acceptance probability is `S(day_start - t_last)`, the loop caps at 200 rounds,
and non-accepting lanes are recorded as "no event today". At an acceptance
probability of 0.02 that mislabels `(1-0.02)^200 = 1.8%` of lanes, biasing the
simulated count distribution down after long quiescence.

**Q14. What is the difference between the paired log-score comparison and a
likelihood ratio test?**
An LRT compares nested models fitted to the same data with a Wilks chi-square
null. Here the models are non-nested, fitted on an earlier window, and scored
out of sample; the estimand is the population mean of a paired predictive-loss
differential and the reference distribution comes from resampling the series.
The right literature is predictive-accuracy comparison (Diebold & Mariano 1995,
JBES 13:253–263), of which the repo's stationary block bootstrap is a
nonparametric analogue, and the CSEP T-test (Rhoades et al. 2011) which targets
the same information-gain-per-earthquake estimand with a normal-theory SE we
deliberately do not use.

**Q15. Why does your NLL exclude the magnitude term?**
EarthquakeNPP convention: `nll = -(tll + sll)`. The magnitude head still exists
and still matters — it is what makes the CSEP M-test available at all, which the
benchmark's other neural point processes cannot run because they do not forecast
magnitudes. But `mll` is not in the headline. If you wanted a *total* proper
score over the full marked outcome you would include it; excluding it makes the
number comparable to the benchmark's published ETAS baseline, which is the
point.

**Q16. Why not evaluate with a Molchan diagram or ROC?**
Because alarm-based scores depend only on the ranking induced by the alarm
function, so they are invariant to monotone re-labelling of the probabilities
and cannot be strictly proper — they cannot certify the *magnitudes* of the
forecast probabilities, which is what a point-process model claims. They also
require binarizing a marked point process. The counterpoint I would concede: a
Molchan/ASS evaluation is closer to an actual civil-protection decision, and a
decision-relevant evaluation would be a legitimate separate exercise we did not
attempt.

**Q17. What calibration diagnostic would you add tomorrow?**
The PIT/random-time-change residual test. FlowQuake gives `F_t(tau_i | H_i)`
directly, so `u_i = F_t(tau_i|H_i)` should be i.i.d. Uniform(0,1) under a
correctly specified temporal model — the same statement as Ogata's (1988)
transformed-time residual analysis. A histogram plus a KS or Anderson–Darling
test would give a *per-model* temporal calibration check at essentially zero
cost, and would sit between the aggregate `tll` and the day-level N-test. It is
not implemented anywhere in this repository.

**Q18. Is 100 forecast days enough?**
For estimating a pass rate, the standard error at `n = 100` and `p = 0.95` is
2.2 days, so you can resolve differences of about 5 percentage points and no
better. For a paired comparison the resolution is set by the number of
*discordant* days, which was 10 — far too few (§11.5). Increasing to every day
in the 4,764-day test window would help the marginal rates but not the pairing
as much as you would hope, because adjacent days are strongly dependent through
aftershock sequences; the current 48-day spacing was chosen (probably
implicitly, via `np.linspace`) to keep days near-independent, and that is the
right trade for the marginal rate.

**Q19. Which single number would you defend as the contribution?**
`dTot = +0.113 nats/event` on ComCat, 95% block-bootstrap CI `[0.1006, 0.1268]`,
replicated out-of-time at `+0.124 [0.1035, 0.1455]`
([runs/total_win.json](../runs/total_win.json)). Its scale relative to the problem:
ETAS beats a homogeneous Poisson by 6.006 nats/event (a factor of 406), and this
is 0.113 on top of that (a factor of 1.12). It is a real, replicated, modest
improvement over a strong incumbent, and it is bounded by the caveat that the
spatial half comes from a head initialized on the target region's own ETAS
inversion — an *upgrade* of a deployed ETAS system, not an inversion-free
replacement.

**Q20. What would make you believe FlowQuake should replace an operational ETAS
system?**
Rung 4 of the repo's own ladder: a registered, frozen, genuinely prospective
rolling forecast on a window that did not exist at registration time, scored by
an independent testing centre, with pre-specified equivalence margins on the
consistency tests and a pre-specified primary endpoint. Nothing short of that
addresses the model-selection exposure the repo itself flags (grids and multiple
seeds were run against the test window for the spatial head), and nothing short
of that satisfies the CSEP standard the field adopted precisely because
retrospective claims were not falsifiable.

---

## 17. Further reading

1. **Gneiting & Raftery (2007), *Strictly proper scoring rules, prediction, and
   estimation*, JASA 102:359–378.** The reference for the whole scoring-rule
   framework: definitions, propriety, the Savage representation, CRPS, energy
   score. Read §1–§4 and you have §2 of this chapter with proofs.
2. **Gneiting, Balabdaoui & Raftery (2007), *Probabilistic forecasts,
   calibration and sharpness*, JRSS-B 69:243–268.** Where "maximize sharpness
   subject to calibration" comes from, plus the PIT diagnostics. This is the
   paper that justifies running both likelihood scoring and consistency tests.
3. **Bernardo (1979), *Expected Information as Expected Utility*, Annals of
   Statistics 7:686–690.** The uniqueness of the log score among smooth local
   proper rules. Short. Read it so you can state the theorem exactly rather than
   approximately.
4. **Parry, Dawid & Lauritzen (2012), *Proper local scoring rules*, Annals of
   Statistics 40:561–592** (and the companion *…on discrete sample spaces*,
   40:593–608). What happens when you relax locality; why score matching is
   possible; the `|Omega| >= 3` caveat.
5. **Schorlemmer, Gerstenberger, Wiemer, Jackson & Rhoades (2007), *Earthquake
   likelihood model testing*, SRL 78(1):17–29.** The RELM testing methodology.
   The origin of the N-, L-, R-test vocabulary.
6. **Zechar, Gerstenberger & Rhoades (2010), *Likelihood-based tests for
   evaluating space-rate-magnitude earthquake forecasts*, BSSA 100(3),
   doi:10.1785/0120090192.** The definitive statement of the gridded N-, S-, M-,
   L-, CL-tests and the `delta_1`/`delta_2`/`gamma` quantities. This is the paper
   whose one-sided S-test convention §8.5 compares against.
7. **Rhoades, Schorlemmer, Gerstenberger, Christophersen, Zechar & Imoto (2011),
   *Efficient testing of earthquake forecasting models*, Acta Geophysica
   59:728–747.** The T-test and W-test; information gain per earthquake as the
   estimand. Directly ancestral to `paired_gain_summary`.
8. **Savran, Werner, Marzocchi, Rhoades, Jackson, Milner, Field & Michael
   (2020), *Pseudoprospective evaluation of UCERF3-ETAS forecasts during the
   2019 Ridgecrest sequence*, BSSA 110(4):1799–1817.** Catalog-based CSEP tests:
   why the Poisson gridded null is wrong for clustered seismicity, and how
   simulated catalogs furnish the null instead. The methodological basis of
   everything in §10.
9. **Savran, Bayona, Iturrieta, Asim, Bao, Bayliss, et al. (2022), *pyCSEP: A
   Python Toolkit for Earthquake Forecast Developers*, SRL 93(5):2858–2870.**
   The package this repo calls. Read it before making any precise claim about
   what `number_test`/`spatial_test`/`magnitude_test` compute — this chapter is
   explicit about where it describes documented behaviour rather than
   source-verified behaviour.
10. **Zechar & Jordan (2008), *Testing alarm-based earthquake predictions*, GJI
    172(2):715–724.** Molchan diagram, area skill score, and its equivalence to
    ROC AUC in the continuum limit. The alternative evaluation tradition, stated
    properly.
11. **Ogata (1988), *Statistical models for earthquake occurrences and residual
    analysis for point processes*, JASA 83:9–27.** Both the ETAS model and the
    residual/random-time-change analysis that would give you the temporal
    calibration diagnostic §16 Q17 says is missing.
12. **[results/CLAIMS.md](../results/CLAIMS.md)** — the repository's own adversarial
    claim-to-artifact audit, 142 rows, including the CSEP family (C1–C21) and the
    A2 sentinel finding. Read it before you read [MANUSCRIPT.md](../MANUSCRIPT.md),
    not after.
