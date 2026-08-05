# The question bank: 157 questions with model answers

This is the drilling chapter. Everything else in this primer teaches; this one
tests. Read it with the artifacts open, and answer out loud before reading the
model answer — the failure mode this chapter exists to prevent is *recognising*
an answer instead of *producing* one.

## What this chapter buys you

- You can answer, cold, any of the standard "do you actually know this field"
  questions: intensity vs density, the point-process likelihood, time rescaling,
  Hawkes branching, Gutenberg–Richter, Omori–Utsu, completeness magnitude,
  magnitude scales, b-value estimation.
- You can derive ETAS from three empirical laws, state every parameter's
  meaning and units, compute its branching ratio, and explain why it is hard to
  beat rather than just asserting that it is.
- You can defend every FlowQuake design choice from first principles — the flow
  on `log tau`, the closed-form spatial heads, the normalization argument for the
  neural-ETAS head, the near/far split, the truncated-first-event correction.
- You can quote the headline numbers **with their artifact filename**, and you
  know which of them the repository's own audit flags as unbacked or
  contradicted.
- You can take a hostile question that has no fully satisfying answer, concede
  it cleanly, and pivot to what the evidence does support — which is the single
  highest-return viva skill and the thing bluffers cannot do.

## Prerequisites

Read these first. This chapter assumes all of them and does not re-derive their
content, only tests it.

- [Chapter 1 — Point processes from first principles](01-point-processes.md)
  — intensity, the likelihood, compensator, time rescaling. Tier 1 assumes all of it.
- [Chapter 2 — Seismology for the point-process modeller](02-seismology.md)
  — Gutenberg–Richter, Omori–Utsu, completeness, magnitude scales.
- [Chapter 3 — ETAS, derived and dissected](03-etas.md)
  — the incumbent. Tier 2 is unanswerable without it.
- [Chapter 4 — Neural density estimation, normalizing flows, and flow matching](04-flows-and-density-estimation.md)
  — the CNF likelihood theorem and flow matching's key theorem, for Tier 3.
- [Chapter 5 — Sequence models and selective state-space models](05-sequence-models-ssm.md)
  — SSD duality and the chunked scan.
- [Chapter 6 — Forecast evaluation: scoring rules, information gain, and CSEP](06-evaluation-and-csep.md)
  — proper scoring rules and the N/S/M tests, for Tier 4.
- [Chapter 7 — Statistics with dependent data](07-statistics-dependent-data.md)
  — block bootstrap, Holm, TOST, McNemar. The rest of Tier 4.
- [Chapter 8 — FlowQuake: the whole argument, and every joint where it can be attacked](08-flowquake-synthesis.md)
  — **read this immediately before Tier 5.** Tier 5 is Chapter 8's attack
  surface rephrased as a hostile examiner would put it.

For the code itself, [STACK.md](../STACK.md) is the walkthrough; this chapter
cross-links to it rather than restating it. For claim provenance,
[results/CLAIMS.md](../results/CLAIMS.md) is the ground truth and
[WORKING.md](../WORKING.md) is the honest current-state document. Note that there
is no standalone chapter on the spatial heads; that material lives in Chapter 8
and in [STACK.md](../STACK.md) Parts IV–V, which is what Tier 3's spatial questions
draw on.

**Notation** (identical to the rest of the primer): `t` days since catalog
start; `tau` inter-event gap; `T` horizon; `N(t)` counting process; `H_t`
history; `lambda(t | H_t)` conditional intensity; `Lambda(t)` compensator;
`f(tau | H)`, `F`, `S` the next-gap density, cdf, survivor; `s = (x, y)` km;
`m` magnitude; `m_c` completeness. ETAS parameters `mu, k0, a, c, omega,
tau_tap, d, gamma, rho` — note that **the repository calls the Omori taper
timescale `tau`, colliding with the inter-event gap**; throughout this primer
the taper is written `tau_tap`. `b` is the Gutenberg–Richter b-value and
`beta = b * ln 10`. Scores: `tll = log f_t` (log 1/day), `sll = log f_s`
(log 1/km^2), `mll = log f_m`, `nll = -(tll + sll)`. Flow state `z`, flow time
`t_flow` in [0, 1].

---

## 1. How to use this bank

Four passes, in this order. Do not skip pass 0.

| pass | what you do | pass criterion |
|---|---|---|
| 0 | Read Tier 1 only. Answer each aloud, no notes. | 28 of 32 correct |
| 1 | Tiers 2–4. Write the derivation on paper where one is asked for. | you can produce every derivation without the text |
| 2 | Tier 5 (hostile) with a friend playing the sceptic, out loud, timed. | you never bluff; every concession is followed by a pivot |
| 3 | Tiers 6–7 with the source file open, then closed. | you can say what breaks if the line is deleted |

Three rules that matter more than the content:

1. **Name the artifact.** "The six-region total win is in
   `runs/stats_hardening.json` under `total_with_head_family`" beats "about a
   tenth of a nat" every time. Examiners test whether you know where your
   numbers live.
2. **Distinguish the two models.** Half of all apparent contradictions in this
   project come from confusing the production kernel-mixture spatial head
   (which *loses* to ETAS) with the neural-ETAS full-history head (which wins).
   `runs/stats_hardening.json` contains a key called `dTot_mean` in **both**
   blocks, with opposite verdicts: `per_region.California.dTot_mean = -0.3107`
   (`"loss"`) and `total_with_head_family.California.dTot_mean = +0.1133`
   (`"win"`).
3. **Concede in one sentence, then pivot.** "That is a real limitation" costs
   you nothing. "That is a real limitation, and here is the narrower claim the
   evidence does support" wins the room.

---

## 2. Tier 1 — Foundations (Q1–Q32)

*Can you talk about this field at all?*

**Q1.** What is a temporal point process, formally, and why is a density over
the whole realization an awkward object?

**A.** A random countable subset of the line, equivalently a counting measure
`N` with `N(A)` = number of points in a Borel set `A`. A "density over the
realization" would need a reference measure on the space of all locally finite
configurations, which is infinite-dimensional and has no canonical Lebesgue
analogue; the standard construction (unit-rate Poisson as reference) works but
is unwieldy. The conditional intensity is the practical parameterisation
because it reduces the object to one scalar function of time and history.

**Q2.** Define the conditional intensity. State what `H_t` includes and, in
particular, whether it includes `t` itself.

**A.** `lambda(t | H_t) = lim_{dt -> 0} P(N[t, t+dt) = 1 | H_t) / dt`, the
instantaneous event rate given the past. `H_t` is the filtration generated by
the process strictly *before* `t` — it is left-continuous / predictable, so it
does **not** include an event at `t`. If it did, the intensity would be able to
see the thing it is predicting and the likelihood would be degenerate.

*Trap:* saying "the history up to and including t". That is the standard slip
and an examiner listens for it.

**Q3.** Write the log-likelihood of a point process observed on `[0, T]` and
explain each term.

**A.**

```
log L = sum_i log lambda(t_i | H_{t_i})  -  integral_0^T lambda(u | H_u) du
```

The sum rewards placing rate where events actually occurred. The integral — the
compensator `Lambda(T)` — is the normalizer: it is what stops you from setting
`lambda` arbitrarily large everywhere. It must be evaluated along the realized
path because `lambda` depends on history.

**Q4.** Sketch a derivation of that likelihood.

**A.** Partition `[0, T]` into `n` bins of width `h`. In each bin the
probability of an event is `lambda_k h + o(h)` and of no event
`1 - lambda_k h + o(h)`. The likelihood of the observed pattern is the product
over bins; taking logs, occupied bins contribute `log(lambda_k h)` and empty
bins `log(1 - lambda_k h) ≈ -lambda_k h`. Summing the empty-bin terms gives
`-sum_k lambda_k h -> -integral lambda`. The `log h` terms are a constant
depending only on the discretisation and are dropped (they cancel in any
likelihood *ratio*, which is all we ever use). This is a sketch, not a proof;
the rigorous statement uses the Jacod likelihood / Doob–Meyer decomposition —
see Daley & Vere-Jones, *An Introduction to the Theory of Point Processes*.

**Q5.** Give the exact relation between `lambda` and the next-gap density
`f(tau | H)`, and derive it.

**A.** Let `t_{i-1}` be the last observed event and `S(tau)` the probability of
no event in `(t_{i-1}, t_{i-1}+tau]`. The hazard of the waiting time is exactly
the conditional intensity, so `S'(tau) = -lambda(t_{i-1}+tau) S(tau)`, giving
`S(tau) = exp(-integral_0^tau lambda(t_{i-1}+u) du)` and

```
f(tau | H) = lambda(t_{i-1} + tau | H) * exp( - integral_0^tau lambda(t_{i-1}+u | H) du )
```

Equivalently `lambda = f / S = f / (1 - F)`: the intensity *is* the hazard rate
of the next gap.

**Q6.** Why is that equivalence the single most consequential design decision in
FlowQuake?

**A.** Because you can model either side. Modelling `lambda` (ETAS) gives an
interpretable additive self-excitation structure but requires `integral lambda`
for every likelihood evaluation — ETAS's kernels are chosen precisely so that
integral is closed-form. Modelling `f(tau | H)` directly (FlowQuake) makes the
normalization the 1-D density's own, which a normalizing flow supplies exactly
and for free, so `integral lambda` is never computed. The price: you lose the
additive interpretation and can only evaluate event-by-event rather than as a
rate field over continuous time, which is why forecasting requires sequential
simulation. See [STACK.md §2](../STACK.md).

**Q7.** State the time-rescaling theorem and say what it is used for.

**A.** If `Lambda(t) = integral_0^t lambda(u | H_u) du` is the compensator and
the model is correct, then the transformed times `Lambda(t_i)` form a unit-rate
Poisson process; equivalently the rescaled gaps
`Lambda(t_i) - Lambda(t_{i-1})` are i.i.d. Exponential(1). It is the basis of
residual analysis for point processes (Ogata 1988, JASA) — you Q-Q plot the
rescaled gaps against Exponential(1), or apply a KS test, to diagnose
misspecification. It is the point-process analogue of the probability-integral
transform.

*Trap:* claiming FlowQuake can do a standard time-rescaling residual check. It
cannot directly: it has no `lambda`, only `f(tau | H)`. What it *can* do is the
equivalent PIT check, `u_i = F(tau_i | H)` should be Uniform(0,1) — which is
the same test written in the density parameterisation.

**Q8.** What is a Poisson process's likelihood, and why is it the right floor
for this benchmark?

**A.** Homogeneous Poisson: `lambda(t) = lambda_0`, so
`log L = n log lambda_0 - lambda_0 T`, and the gaps are Exponential(lambda_0),
memoryless. It is the "you learned nothing from the history" baseline. On
ComCat_25 the benchmark's Poisson row is `tll` 0.5126406686259881,
`sll` -13.774504128914366, `nll` 13.261863460288378
(`runs/n1_density/eval_test.json` -> `baselines.Poisson`, copied verbatim from
the harness by [flowquake/evaluate.py](../flowquake/evaluate.py) lines 104–110).
ETAS sits 6.0064 nats/event below it — that gap is the size of the actual
problem.

**Q9.** Define a Hawkes process and write its intensity.

**A.** `lambda(t) = mu + sum_{t_j < t} g(t - t_j)` with `g >= 0` a triggering
kernel — a linear self-exciting process (Hawkes 1971, Biometrika). Every past
event permanently raises future rate by a decaying amount. ETAS is a *marked*
Hawkes process where the kernel amplitude depends on the parent's magnitude and
the offspring's location is drawn from a spatial kernel.

**Q10.** What is the branching-process representation of a Hawkes process, and
why does it matter?

**A.** A Hawkes process is equivalent to a Poisson cluster process: immigrants
arrive at rate `mu`, and each event independently produces offspring as an
inhomogeneous Poisson process with intensity `g(t - t_j)`, recursively. This is
what makes simulation trivial (thinning is unnecessary — you simulate
generation by generation), it gives the branching ratio its meaning, and it is
the object the EM inversion works with: the E-step assigns each event a
probability of being an immigrant vs. an offspring of each predecessor.

**Q11.** Define the branching ratio and state the stability condition.

**A.** `n = expected number of direct offspring per event`. Since generations
form a Galton–Watson tree, the expected cluster size is `1/(1-n)` for `n < 1`
and the process is stationary; for `n >= 1` the expected number of descendants
diverges and the process is explosive (non-stationary). Every fitted ETAS on a
real catalog must have `n < 1`, and fitted values on regional catalogs are
commonly in the 0.6–0.95 range and can sit close to 1 — which is one reason the
model is hard to fit stably. (Do not quote a specific branching ratio for *this*
project: the only one the manuscript states, 0.968 for the refit-2020 control,
is flagged `NO ARTIFACT` at `results/CLAIMS.md` N2.)

**Q12.** Derive the branching ratio for ETAS.

**A.** With triggering weight
`w_j(dt) = k0 * exp(a(m_j - m_c)) * exp(-dt/tau_tap) * (dt + c)^{-(1+omega)}`,
the expected direct offspring of a parent of magnitude `m` is

```
N(m) = k0 * exp(a(m - m_c)) * I,     I = integral_0^inf exp(-u/tau_tap) (u + c)^{-(1+omega)} du
```

`I` is an upper incomplete gamma (no elementary closed form); with the taper
removed (`tau_tap -> inf`) it collapses to `I = c^{-omega} / omega`. Averaging
over the Gutenberg–Richter magnitude law
`f(m) = beta exp(-beta(m - m_c))`:

```
n = k0 * I * integral_0^inf exp(a x) beta exp(-beta x) dx = k0 * I * beta / (beta - a)
```

which requires `a < beta`. If `a >= beta` the productivity integral diverges:
large events produce so many aftershocks that the mean is infinite. That
constraint is why fitted `a` is always compared against `beta`.

**Q13.** State Gutenberg–Richter and convert it to a density.

**A.** `log10 N(>= m) = A - b m`, i.e.
`P(M > m) = 10^{-b(m - m_c)}` for `m >= m_c` (Gutenberg & Richter 1944, BSSA).
Writing `10^{-bx} = exp(-b ln(10) x)`, the excess `m - m_c` is
Exponential(`beta`) with `beta = b ln 10`. For `b = 1`, `beta = 2.302585…`. The
density is `f(m) = beta exp(-beta (m - m_c))`, so `log f(m) = log beta - beta
(m - m_c)` — exactly the form of
[flowquake/heads.py](../flowquake/heads.py) `GRMagnitudeHead.log_prob`.

**Q14.** Derive the Aki maximum-likelihood estimator of `b`.

**A.** With `x_i = m_i - m_c ~ Exp(beta)` i.i.d., the log-likelihood is
`n log beta - beta sum x_i`; setting the derivative to zero gives
`beta_hat = 1 / xbar` and therefore

```
b_hat = log10(e) / xbar = 0.4342944819 / (mbar - m_c)
```

With magnitudes binned at width `dm`, the correct threshold is the bin's lower
edge, so use `m_c - dm/2` in the denominator (Aki 1965, Bull. Earthq. Res.
Inst.). The standard error is `SE(b_hat) ≈ b_hat / sqrt(n)` (delta method on
`1/xbar`); Shi & Bolt's slightly different variance estimate is the usual
refinement — verify before quoting it.

**Q15.** Given `b = 1.13`, what is the mean magnitude excess above completeness,
and what is the standard error on `b` from 2,612 events?

**A.** `xbar = log10(e)/b = 0.4342945/1.13 = 0.3843` magnitude units.
`SE = 1.13/sqrt(2612) = 0.0221`. Those are Greece's training-era values in
`runs/completeness.json` (`b_train` 1.13, `n_train_at_mcut` 2612).

**Q16.** State the Omori–Utsu law and integrate it.

**A.** Aftershock rate `n(t) = K (t + c)^{-p}` with `p ≈ 1` (Omori 1894; the
modified form with `p` free is Utsu 1961, *Geophys. Mag.* — verify the exact
citation before using it). The cumulative count to time `T`:

```
p != 1 : integral_0^T K(t+c)^{-p} dt = K [ c^{1-p} - (T+c)^{1-p} ] / (p - 1)
p == 1 : K [ log(T + c) - log c ]
```

For `p <= 1` the total diverges as `T -> inf`, so a taper (ETAS's
`exp(-dt/tau_tap)`) or `p > 1` is required for a finite branching ratio. The
key qualitative point: this is a power law, not an exponential, so aftershocks
persist for years and the tail carries real probability mass — which is why
"condition on the last 20 events" is a structurally fatal modelling choice.

**Q17.** What is `c` in the Omori law, physically, and why is it contentious?

**A.** A short-time regularizer that keeps the rate finite at `t = 0`. Fitted
values are seconds to hours. Part of it is genuinely physical (finite rupture
duration, afterslip) and part is an artefact of catalog incompleteness in the
minutes after a mainshock, when the coda of the mainshock masks small events.
This matters because it makes `c` partly a property of the *network*, not the
earth, and hence not transferable between catalogs.

**Q18.** State Utsu's productivity relation and its interaction with `b`.

**A.** The number of direct aftershocks scales as `N ∝ exp(a(m - m_c))`, i.e.
`10^{alpha(m - m_c)}` with `alpha = a/ln 10`. Because magnitudes are
exponentially distributed with rate `beta`, the branching ratio integral
converges only when `a < beta` (Q12). Empirically `alpha ≈ b` is common, which
puts real catalogs near the boundary — a fact that makes ETAS inversion
numerically delicate.

**Q19.** What is `m_c` and why is it not a nuisance parameter?

**A.** The completeness magnitude: below it the seismic network misses events,
so the catalog is no longer a sample from the process being modelled — it is a
sample from the process *thinned by an unknown detection function*. Every
likelihood in this project is conditional on `m >= m_c`. Getting it wrong
biases `b` (downward if `m_c` is too low), biases ETAS productivity, and — if
`m_c` drifts over time — manufactures a spurious temporal trend that any
flexible model will happily learn.

**Q20.** How is `m_c` estimated here, and what check is run?

**A.** Maximum-curvature: the magnitude bin with the most events, i.e. the peak
of the non-cumulative frequency–magnitude histogram, usually plus a safety
margin (commonly attributed to Wiemer & Wyss 2000, BSSA; verify before citing
the exact page). `scripts/check_completeness.py` writes
`runs/completeness.json`, which reports `mc_train` and `mc_test` *separately*
per region — the whole point is stability across eras. For example Japan gives
`mc_train` 3.65, `mc_test` 3.75 and `recommend_mcut` 4.0; Chile gives 3.95 /
3.65 -> 4.0. The chosen cut is above both era estimates, deliberately
conservative.

**Q21.** Name the main magnitude scales and say when each is valid.

**A.** `M_L` (Richter, local, amplitude on a Wood–Anderson-equivalent
instrument, saturates around 6–7); `m_b` (body wave, saturates ~6.5); `M_S`
(surface wave, saturates ~8.3); `M_d` (coda duration, used for small events in
regional networks); `M_w` (moment magnitude, no saturation). Saturation happens
because amplitude-based scales measure a fixed-period wave whose amplitude stops
growing once the rupture is much longer than that period.

**Q22.** Define moment magnitude.

**A.** From the seismic moment `M_0 = mu_shear * A * D` (rigidity × rupture
area × average slip), in N·m:

```
Hanks & Kanamori 1979:  M_w = (2/3) log10(M_0) - 10.7     (M_0 in dyn·cm)
IASPEI standard form:   M_w = (2/3) (log10(M_0) - 9.1)    (M_0 in N·m)
```

Be careful with the constant. Converting the first to N·m (1 N·m = 10^7 dyn·cm)
gives `(2/3) log10(M_0) - 6.033`; the IASPEI form gives `(2/3) log10(M_0) -
6.067`. The two conventions differ by ~0.03 magnitude units, which is why you
see both "−6.03" and "−6.06/−6.07" quoted. Say which convention you mean rather
than a bare constant. The 2/3 exponent was chosen so `M_w` agrees with `M_S` in
the range where `M_S` is unsaturated. Consequence worth stating: one magnitude
unit is a factor of `10^{1.5} ≈ 31.6` in radiated energy (and in `M_0`).

**Q23.** Why does the mixture of magnitude scales matter for this project?

**A.** Because the b-value, the ETAS productivity `a`, and the GR head are all
fitted on whatever scale the agency reports. The cross-regime catalogs use
agency-preferred magnitudes with mixed types (ISC, INGV), documented in each
region's `<name>_meta.json` per [REPRODUCE.md](../REPRODUCE.md). The manuscript
handles this by running an `M_w`-homogenization robustness check
(`scripts/mw_robustness.py`) and explicitly **not** claiming Italy's total win
under `M_w` homogenization — the README says "native-catalogue scale, not
claimed under Mw homogenization". That is the right kind of scoping.

**Q24.** What is a marked point process, and what factorization does FlowQuake
assume?

**A.** A point process where each point carries a mark — here location `s` and
magnitude `m`. FlowQuake factorizes the joint conditional density by the chain
rule and then assumes conditional independence given history:

```
f(tau, x, y, m | H) = f_t(tau | H) * f_s(x, y | H) * f_m(m | H)
```

Each factor gets its own head. ETAS makes the same factorization for the
*current* event's marks (its magnitude–location coupling acts on *future* rate
through productivity, not on the present event's own marks).

**Q25.** Is that conditional-independence assumption defensible?

**A.** Partly. It is standard in ETAS and in the benchmark, so the comparison is
apples-to-apples. It is nonetheless wrong in a known way: aftershock magnitude
and distance-from-parent are weakly dependent, and larger events in a sequence
tend to be spatially offset. The honest defence is (i) the benchmark's scoring
convention factorizes the same way, so any coupling would have to be *added* to
both models to compare fairly, and (ii) the effect size is small relative to
the 0.1-nat gains being reported. It is a real modelling limitation, not a
scoring error.

**Q26.** Define `tll`, `sll`, `mll`, `nll` with units and sign conventions.

**A.** Per test event: `tll = log f_t(tau)` in log(1/day); `sll = log f_s(x,y)`
in log(1/km^2); `mll = log f_m(m)` in log(1/magnitude unit); and
`nll = -(tll + sll)`. Higher `tll`/`sll`/`mll` is better; lower `nll` is better.
`nll` **excludes** `mll` — that is the EarthquakeNPP convention, not an
oversight. The magnitude head still matters because it is what makes the CSEP
M-test pass.

**Q27.** `sll = -8.6898`. Translate that into something physical.

**A.** `exp(-8.689770387238827) = 1.6830e-4` per km^2. Its reciprocal is
5,942 km^2: the model is as informative about location as spreading the whole
probability mass uniformly over about 5,900 km^2 of California. That is the
ETAS number from `runs/fullsuite_summary.json` -> `ComCat_25.etas_sll`. The
neural-ETAS head's `-8.6297607421875`
(`runs/replacement_readiness.json` -> `checks[spatial_win_comcat]`) is
equivalent to about 5,596 km^2 — a 5.8% shrink, which is exactly
`exp(0.06) = 1.0618`.

**Q28.** What does a gain of "+0.113 nats/event" actually mean?

**A.** `exp(0.1133) = 1.1200`: the model assigns 12.0% more probability density
to what actually happened, per event, on average. Compounded over the 21,889
test events it is a likelihood ratio of `exp(0.1133 * 21889) = e^{2480}` —
which is why per-event nats, not total log-likelihood, is the only sane unit
here.

**Q29.** Is a log score a proper scoring rule, and what does that buy you?

**A.** Yes, and strictly so: for a scoring rule `S(P, y)`, propriety means
`E_{y~Q}[S(Q, y)] >= E_{y~Q}[S(P, y)]` for all `P`, with equality only at
`P = Q` for strict propriety (Gneiting & Raftery 2007, JASA). For the log score
this is the non-negativity of KL divergence. Consequence: you cannot gain
expected score by reporting anything other than your true predictive
distribution — no hedging, no sharpening tricks. It is what makes a likelihood
comparison a legitimate forecast comparison rather than a fitting contest.

**Q30.** What is the difference between calibration and sharpness, and why do
you need both?

**A.** Calibration is agreement between stated probabilities and observed
frequencies; sharpness is the concentration of the predictive distribution.
The climatological forecast is perfectly calibrated and useless; a
point-forecast is maximally sharp and usually wrong. Proper scoring rules
reward "sharpness subject to calibration". In this project the log score
measures the combination, and the CSEP N/S/M tests measure calibration alone —
which is why the manuscript reports both, and why "we win on likelihood and tie
on CSEP" is the correct reading of §4.2, not a weakness.

**Q31.** Why are per-event standard errors over 21,889 earthquakes not
trustworthy?

**A.** Because consecutive earthquakes are massively correlated: an aftershock
sequence of a thousand events is, statistically, close to one observation. A
naive i.i.d. standard error over `n` events *understates* the true standard
error by roughly `sqrt(n / n_eff)`, where `n_eff` is the effective sample size —
i.e. it overstates precision by that same factor, which is large when whole
aftershock sequences behave as single observations. The fix used here is a
stationary block bootstrap with mean
block length 50 events ([flowquake/stats.py](../flowquake/stats.py) line 45),
which resamples blocks of consecutive events and thus preserves within-sequence
correlation.

**Q32.** What does "the model conditions on the entire history" actually mean
at evaluation time, and how is it implemented without an encoder?

**A.** `evaluate.py` builds the whole catalog as one length-`E` sequence
(`full_sequence_batch`) and scores test events in that single pass, so every
test event's conditioning is computed from all preceding events. With
`h_bottleneck = 0` there is no learned encoder; the reach comes from
hand-designed relational features at seven exponentially spaced lags
(`RECENCY_LAGS = (1,2,4,8,16,32,64)`) plus mixture components at up to 80
selected parents — 64 recent, 16 big triggers over a trailing 730 days. The
second model, the neural-ETAS head, does a literal full-history ETAS sum.

---

## 3. Tier 2 — ETAS (Q33–Q53)

**Q33.** Write out the exact ETAS used in this repository, with units.

**A.** Triggering weight of past event `j` (magnitude `m_j`, elapsed `dt` days):

```
w_j = k0 * exp(a(m_j - m_c)) * exp(-dt / tau_tap) * (dt + c)^{-(1+omega)}
      \___ productivity ___/   \___ taper ____/    \___ Omori decay ___/
```

Spatial kernel at squared distance `r^2` (km^2):

```
K_j(r^2) = (r^2 + d_j)^{-(1+rho)},    d_j = d * exp(gamma (m_j - m_c))
```

Conditional spatial density given that an event occurs:

```
                 mu + sum_j w_j K_j(r^2(s))
f_s(s | H)  =  ------------------------------
                 mu*A + sum_j w_j Z_j
```

`mu` is background rate density per km^2 per day, `A` the region area. This is
transcribed at [flowquake/neural_etas.py](../flowquake/neural_etas.py) lines
78–87 and computed at
[scripts/precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
line 143.

**Q34.** Derive `Z_j`.

**A.** `Z_j = integral K_j dA = integral_0^inf (r^2 + d_j)^{-(1+rho)} 2 pi r dr`.
Substitute `u = r^2`, `du = 2r dr`, so `2 pi r dr = pi du`:

```
Z_j = pi * integral_0^inf (u + d_j)^{-(1+rho)} du
    = pi * [ -(u + d_j)^{-rho} / rho ]_0^inf
    = pi * d_j^{-rho} / rho
    = pi / (rho * d_j^rho)
```

which requires `rho > 0`. That is exactly `zj = math.pi / (rhoj *
torch.pow(dmj, rhoj))` in `neural_etas.py`.

**Q35.** Give each ETAS parameter's meaning and units.

**A.**

| symbol | meaning | units |
|---|---|---|
| `mu` | background rate density | events / (km^2 · day) |
| `k0` | productivity scale | day^omega (so `k0 · I` is dimensionless) |
| `a` | productivity exponent in magnitude | 1 / magnitude unit |
| `c` | Omori short-time regularizer | days |
| `omega` | Omori exponent, `p = 1 + omega` | dimensionless |
| `tau_tap` | exponential taper timescale on Omori | days |
| `d` | spatial kernel scale at `m = m_c` | km^2 (it is added to `r^2`) |
| `gamma` | magnitude scaling of the spatial scale | 1 / magnitude unit |
| `rho` | spatial power-law exponent | dimensionless |

Note the parameterisation trap: `d` has units of km^2 because it is added to
`r^2`, not to `r`.

**Q36.** How is ETAS fitted, and what makes it slow?

**A.** Expectation–maximization on the branching representation
(Veen & Schoenberg 2008, JASA; Zhuang, Ogata & Vere-Jones 2002, JASA for the
stochastic-declustering view). The E-step computes, for every event `i`, the
probability that it was background versus triggered by each predecessor `j` —
an `O(n^2)`-flavoured object even with truncation. The M-step refits the nine
parameters. [REPRODUCE.md](../REPRODUCE.md) §2 budgets 3–4 CPU hours per large
region.

**Q37.** What are edge effects in ETAS, and how are they handled?

**A.** Two kinds. **Temporal**: events before the fit window's start triggered
events inside it, so a fit that ignores them under-attributes triggering and
over-attributes background. The standard fix is an auxiliary warm-up window
whose events are inputs but never targets — exactly the benchmark's
1971–1981 auxiliary window. **Spatial**: parents outside the polygon trigger
events inside it, and the `Z_j` normalizer integrates over all of `R^2` while
the catalog only covers the polygon. Both bias `mu` upward and the branching
ratio downward if unhandled.

**Q38.** What is the "source events" / re-conditioning issue that appears in the
CSEP section?

**A.** When ETAS simulates forward from a fitted inversion, it needs the set of
parent events that seed the simulation. If that set is the fit-window catalog
and is not re-conditioned on events that happened *after* the fit window (i.e.
after `test_start`), the simulator has no knowledge of post-2007 mainshocks and
systematically under-forecasts counts. The manuscript attributes an earlier
N 73/100 result to exactly this, fixed by
`reload.source_events = reload.prepare_source_events()`. **Be careful here** —
see Q117; the repository's own audit shows this attribution is not isolable.

**Q39.** Why does ETAS beat every neural point process in the published
benchmark?

**A.** Three reasons. (i) Its functional forms are nearly right: power law in
time and space, exponential in magnitude — a century of data says so, and a
neural net on **55,442** training events
([runs/mw_robustness.json](../runs/mw_robustness.json) →
`california.comcat_mc25_headline.train_events` — a committed number, so quote it
rather than STACK.md's "~70,000", which is the *pre-`test_start`* prefix
`92,263 − 21,889 = 70,374` and includes the auxiliary and validation eras;
[Ch. 5 §3.2](05-sequence-models-ssm.md#32-cost) states the trap once)
spends its capacity rediscovering them badly.
(ii) It integrates over the *whole* history, so a 2011 M9 still contributes in
2019, while the published NPP baselines truncate (DeepSTPP sees 20 events).
(iii) The data is small and the target is a heavy-tailed density, which
punishes flexibility — a flexible model can memorize where 1985 earthquakes
happened and catastrophically mis-locate 2015 ones. §4.3 measures exactly that.

**Q40.** What is ETAS's background model, and why is that the exploitable
weakness?

**A.** Uniform over the region: `mu` is a constant density per km^2. Real
background seismicity is concentrated on fault systems and spans four orders of
magnitude spatially. Replacing it with a smoothed-seismicity map is the single
largest source of the neural head's spatial gain: the background-only ablation
(no per-parent modulations) already delivers `dS` +0.0513 of the +0.06 total
(`runs/neural_etas/ComCat_25/summary_bg_only_s0.json` -> `dS_mean` 0.0513,
CI [0.0434, 0.0595]). Note that smoothed seismicity is standard practice
(commonly attributed to Helmstetter, Kagan & Jackson 2007, SRL; verify before
citing), so this is an *upgrade to the benchmark's ETAS configuration*, not a
novel idea.

**Q41.** Then isn't the spatial win just "ETAS with a better background"?

**A.** Substantially yes, and the repository says so: 0.0513 of the 0.060 is
background alone. The remaining ~0.009 comes from the per-parent neural
modulations. The honest framing is: the largest single component of the gain is
a known, standard background upgrade the benchmark's ETAS did not have; the
neural modulations add a smaller, separately-reported increment; and the
`--refit-globals` control shows an SGD refit of global kernel parameters alone
gives +0.0564 (`summary_refit_globals_s0.json`). Those three numbers are the
whole decomposition and should be quoted together.

**Q42.** What is the `--refit-globals` control and what is its stated
limitation?

**A.** A classical flETAS-style control that refits global kernel parameters by
SGD with the learned background but no per-parent MLP — it asks "is the gain
just from refitting ETAS's parameters better?". Its own docstring flags the
limitation: it reweights the *near set* exactly but scales the far field by a
single `alpha`, making it a conservative lower bound on a true full refit. The
full flETAS (EM, free background) baseline is listed as not run
(`results/CLAIMS.md` N12).

**Q43.** Where do the ETAS parameters used here come from?

**A.** Read from `reference/Experiments/ETAS/output_data_<Cfg>/parameters_0.json`
at [precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
lines 52–60, which converts `log10_mu`, `log10_k0`, `log10_c`, `log10_tau`,
`log10_d` out of log space. For California those are the benchmark's *published*
inversion (`runs/forward_etas/summary.json` -> `params_frozen_from`:
`"ComCat_25 inversion (train<=2007, published with benchmark)"`). For the five
foreign regions they are author-run inversions using configs that live only in
the gitignored `reference/` tree.

**Q44.** How do you *know* the repository's ETAS reimplementation is faithful?

**A.** A hard assertion, not an eyeball check.
`precompute_trigger_features.py` line 143 computes
`log(mu + trig_num) - log(mu*area + trig_den)`, and line 147 asserts it
reproduces the package's stored per-event `SLL` to `< 1e-6`, aborting the
precompute otherwise. The committed
result is `runs/etas_sll_repro.json`: `max_abs_sll_err` 1.7655796824556091e-09,
`mean_sll_ours` -8.689770387238818 vs `mean_sll_ref` -8.689770387238829,
`n_test` 21889, `match` true. That is float precision.

**Q45.** Is the *temporal* term similarly verified?

**A.** No, and this is a genuine gap. `runs/etas_sll_repro.json` covers the
**spatial** term only. `MANUSCRIPT.md` claims ~1e-5/event temporal agreement
and a 1.5e-4-nat first-target anchor effect; `results/CLAIMS.md` lists both as
**N4, NO ARTIFACT**. The right answer in a viva is to say so, and to note that
the temporal comparison does not depend on the reimplementation — ETAS's
per-event `TLL` is read off disk from `augmented_catalog.csv`, never
recomputed by this repo at scoring time.

**Q46.** Why is a power-law spatial kernel used rather than a Gaussian?

**A.** Because aftershock distances are power-law distributed. A Gaussian
mixture pays enormous likelihood penalties on tail events: `log f` falls like
`-r^2` instead of `-2(1+rho) log r`, so a single aftershock 100 km from its
parent can cost hundreds of nats under a Gaussian and single digits under a
power law. Since the log score is dominated by its worst events, this is not a
cosmetic choice.

**Q47.** Under what condition is the ETAS spatial kernel integrable, and what
happens if `rho -> 0`?

**A.** `Z_j = pi/(rho d_j^rho)` requires `rho > 0`. As `rho -> 0`, `Z_j ->
infinity`: the kernel's tail becomes non-integrable (the density has infinite
mass) and the normalized conditional density collapses to zero everywhere.
[neural_etas.py](../flowquake/neural_etas.py) line 83 clamps
`rhoj = (rho * exp(drho)).clamp(0.05, 5.0)` for exactly this reason.

**Q48.** Where does the magnitude dependence of the spatial kernel come from?

**A.** `d_j = d exp(gamma (m_j - m_c))` — bigger earthquakes get a bigger
`d_j`, hence a wider aftershock cloud, consistent with rupture-length scaling
(rupture dimension grows roughly as `10^{0.5 M}`). Note the exponent: `d_j` has
units of km^2, so `gamma` here scales an *area*, and a `gamma` of ~0.5–1.0
corresponds to a linear-dimension exponent of ~0.25–0.5 per magnitude unit.

**Q49.** What is the `omega` parameterisation and why is it used instead of `p`?

**A.** `p = 1 + omega`, so `omega > 0` guarantees `p > 1` and hence a finite
Omori integral without the taper. It is a reparameterisation for optimizer
stability: an unconstrained `omega` with a softplus or exponential link keeps
`p` in the physical region during EM.

**Q50.** Name three ETAS failure modes.

**A.** (i) **Incompleteness after mainshocks**: for hours to days after a large
event the catalog is short-changed, so the early Omori decay is fit to
partially-missing data — the motivation for Mizrahi et al. (2021)'s
incompleteness-aware `etas`. (ii) **Swarms and induced seismicity**: fluid- or
creep-driven sequences are not Omori-shaped, and ETAS attributes them to
triggering it cannot explain. (iii) **Uniform background**: everything in Q40.
A fourth worth knowing is **near-critical instability** — with `n` near 1 the
likelihood surface is flat along a `k0`–`a` ridge and the inversion is
ill-conditioned.

**Q51.** ETAS has nine parameters and beats a neural network with ~0.1–1 M
parameters. What is the lesson?

**A.** That the relevant capacity is not parameter count but the match between
the hypothesis class and the data-generating process, given the sample size.
With 55,442 training events (Q39) and a heavy-tailed 4-D target density, the bias
introduced by ETAS's functional forms is small and its variance is tiny; a
flexible model trades a small bias reduction for a huge variance increase. §4.3
measures the variance side directly: with a learned whole-catalog channel open
(`h = 4`), train `nll` drops to 4.14 while held-out `nll` reaches 19.65 — worse
than the Poisson baseline of 13.26.

**Q52.** Why is ETAS still the operational incumbent despite these weaknesses?

**A.** Because it is calibrated, interpretable, cheap to evaluate on a grid,
transparent to regulators, and has a two-decade record of CSEP testing. A
replacement must beat it not just on likelihood but on consistency, auditability
and deployment cost. That is the standard `REPLACEMENT_READINESS.md` sets, and
the reason its ladder has five rungs of which only three are marked `[DONE]`.

**Q53.** What exactly is the benchmark contract you are evaluated under?

**A.** EarthquakeNPP (Stockman, Lawson & Werner, TMLR 2026, arXiv:2410.08226).
For ComCat_25: auxiliary 1971-01-01 -> 1981-01-01 (inputs, never targets),
train -> 1998-01-01, val 1998-01-01 -> 2007-01-01 (early stopping only), test
2007-01-01 -> 2020-01-17, 21,889 events at `m_c` 2.5. Normalization statistics,
the background map and every other fitted quantity come from pre-val events
only ([data.py](../flowquake/data.py) line 221, `fit = times < val_start`).
The benchmark's finding is that none of five NPPs beat ETAS and ETAS wins
spatial log-likelihood against all of them.

---

## 4. Tier 3 — Method (Q54–Q81)

**Q54.** What is a continuous normalizing flow, and state its likelihood
theorem.

**A.** Pick a base density `p_0 = N(0, I)`, a velocity field `v(z, t_flow)`, and
transport samples along `dz/dt_flow = v(z, t_flow)` from `t_flow = 0` to 1. The
instantaneous change-of-variables formula (Chen et al. 2018, NeurIPS, *Neural
Ordinary Differential Equations*) says

```
d/dt_flow  log p(z(t_flow), t_flow)  =  - div v(z(t_flow), t_flow)
```

so integrating,

```
log p_1(u) = log p_0(z(0)) - integral_0^1 div v dt_flow,     z(1) = u
```

This is **exact** — no ELBO, no bound. Cost: one ODE solve per evaluation plus
a divergence.

**Q55.** Derive the instantaneous change of variables.

**A.** Take the continuity (transport) equation for a density carried by a flow:
`d p/d t_flow + div(p v) = 0`. Expand: `d p/dt_flow + v · grad p + p div v = 0`.
The first two terms are the material derivative `Dp/Dt_flow` along a
trajectory, so `Dp/Dt_flow = -p div v`. Divide by `p`:
`D log p / D t_flow = - div v`. Integrating along the trajectory gives the
stated result. (This is the sketch; the rigorous version needs `v` Lipschitz in
`z` and integrable in `t_flow` so the flow map exists and is a diffeomorphism.)

**Q56.** Why is the temporal head a flow while the spatial head is not?

**A.** **Not because of divergence cost — give the inductive-bias answer, and be
ready to say why the cost answer is wrong.** At `dim = 1` the "Jacobian trace" is
a single derivative computed exactly by `torch.func.jacrev` + `vmap`
([flow.py](../flowquake/flow.py) lines 94–107), with no Hutchinson estimator
anywhere in the repository. But 2-D is *also* cheap: an exact trace at `d = 2` is
two backward passes against Hutchinson's one, `flow.py`'s own module docstring
(lines 6–8) says "dimension is 1-2, so the full Jacobian trace is cheap via
`torch.func.jacrev`", and `tests/test_flow.py::test_log_prob_matches_gaussian_2d_conditional`
trains and scores a 2-D conditional flow in the standard suite. Hutchinson only
earns its variance at `d` in the hundreds. So "2-D is not free" is false, and
[STACK.md §9](../STACK.md#9-flowpy--the-temporal-head) is wrong where it draws
that conclusion — [Chapter 4 §4.4](04-flows-and-density-estimation.md#44-where-the-repos-own-explanation-is-wrong)
lays this out in full.

The real reason is stated in [model.py](../flowquake/model.py) lines 12–15: a flow
over **absolute** `(x, y)`, conditioned on features that exclude absolute
coordinates (`SAFE_TOKEN_DIMS`), has nowhere to get "where" from except its
weights, so it must encode training-era geography — memorization by construction.
An observation-anchored mixture places components at event locations supplied at
evaluation time and is therefore translation-equivariant. Time has no "where" to
memorize in a scalar gap, so the flow's flexibility is safe there. Concede the
limit of the argument before you are asked: this indicts *absolute-coordinate
parameterization*, not flows — a flow over the displacement
`(x − x_last, y − y_last)` would be equivariant too, and that ablation was never
run ([Ch. 4 §10.2](04-flows-and-density-estimation.md#102-the-limits-of-the-argument--state-these-before-you-are-asked), H3).

*If you give the cost argument in a viva and the examiner opens `test_flow.py`,
you lose the room.*

**Q57.** State flow matching's key theorem and why it makes training
simulation-free.

**A.** Lipman et al. (2023, ICLR). Define a probability path `p_t` via
*conditional* paths `p_t(z | u)` with known conditional velocity `v_t(z | u)`.
The marginal velocity is `v_t(z) = E_{u ~ p(u|z)}[v_t(z|u)]`. The theorem: the
conditional flow-matching loss
`E_{t,u,z_t}||v_theta(z_t,t) - v_t(z_t|u)||^2` has the *same gradients* with
respect to `theta` as the intractable marginal objective
`E||v_theta - v_t||^2`. Proof sketch: expand both squares; the quadratic terms
in `v_theta` agree, and the cross terms agree because `E[v_t(z|u) | z] =
v_t(z)`; the remaining terms are `theta`-independent. So you can regress on a
per-sample target and never solve an ODE during training.

**Q58.** Write FlowQuake's specific path and target.

**A.** Rectified / straight path with a variance floor:

```
z_{t_flow} = (1 - (1 - sigma_min) t_flow) z_0 + t_flow * u,    z_0 ~ N(0, I)
```

Its `t_flow`-derivative is constant along the path: `u - (1 - sigma_min) z_0`.
So the loss is `MSE(v_theta(z_t, t_flow, cond), u - (1-sigma_min) z_0)` —
[flow.py](../flowquake/flow.py) lines 70–78. Three lines, no ODE.

**Q59.** What does `sigma_min` buy, and what breaks without it?

**A.** At `t_flow = 1` the path lands on `u + sigma_min * z_0`, so the modelled
density is the data convolved with `N(0, sigma_min^2)` — a KDE-style bandwidth
floor. Without it, a flow trained on *discretized* data (magnitudes on a 0.1
grid, timestamps at finite resolution) can place arbitrarily tall spikes on the
discrete atoms and report absurd likelihoods that are artefacts of the
discretisation. The production config uses `sigma_min: [0.02, 0.01, 0.05]`
(`configs/n1_density.yaml`) for time / space / magnitude respectively. **Only the
first is live**: [model.py:85-92](../flowquake/model.py#L85-L92) passes
`sigma_min[0]` to `CondFlow`, and `KernelMixtureHead` and `GRMagnitudeHead` take
no `sigma_min` argument at all — `sigma_min[1]` and `sigma_min[2]` are dead
config keys and the comment's "per head" describes a mechanism that does not
exist. The equivalent guards for those heads are the softplus floors
`d >= d_floor_km`, `q >= q_floor` and the `+0.005` magnitude shift. Do not say
"we floored all three heads"
([Ch. 4 §7.3](04-flows-and-density-estimation.md#73-a-discrepancy-in-the-repos-sigma_min-accounting),
[Ch. 8 §2.9](08-flowquake-synthesis.md#29-the-sigma_min-bandwidth-floors)).

**Q60.** Derive the unit conversion from the flow's output to `tll`.

**A.** The flow models `u = (log tau - mu_n)/sigma_n` where `mu_n, sigma_n` are
train-era normalization constants. Two change-of-variables steps:

```
f_{log tau}(l) = p_u(u) * |du/dl| = p_u(u) / sigma_n
f_tau(tau)     = f_{log tau}(log tau) * |d log tau / d tau| = f_{log tau} / tau
```

therefore

```
log f(tau) = log p(u) - log sigma_n - log tau
```

which is [model.py](../flowquake/model.py) lines 233–235 verbatim. That one line
is the difference between a number comparable to ETAS's `tll` and a number that
means nothing.

**Q61.** Why model `log tau` rather than `tau`?

**A.** Inter-event gaps span about nine orders of magnitude (milliseconds to
years). A heavy power-law tail in `tau` becomes a roughly tractable, roughly
unimodal shape in `log tau`, which a small MLP velocity field can represent.
Numerically, `tau` is floored at `TAU_FLOOR_DAYS = 1e-7` days (~9 ms) so the
log is finite; the catalog's smallest nonzero gap is ~5e-8 d.

**Q62.** How is the exact likelihood computed at evaluation, and with what
integrator?

**A.** `CondFlow.log_prob` integrates the ODE **backward** from the datum,
`t_flow: 1 -> 0`, with RK4, accumulating the divergence, and returns
`log N(z(0)) - logdet` ([flow.py](../flowquake/flow.py) lines 109–134). **Be
precise about the step count, because there are three of them and only one is
the headline.** `flowquake.evaluate`'s CLI *default* is 96
([evaluate.py:57](../flowquake/evaluate.py#L57)) and `README.md` documents that,
but `train.py --eval-after` hard-codes `--steps 64`
([train.py:185](../flowquake/train.py#L185)) and that is the path every headline
number came through: of the 68 committed `eval_test*.json` artifacts, **66 record
`ode_steps: 64`**, one records 96 (`runs/comcat25_s1555`) and one 32
(`runs/smoke`). Validation during training uses 32 (`val_ode_steps`). So say "the
reported results are 64-step results", concede the 64-vs-96 inconsistency between
the in-window and forward-window evaluations, and quote the convergence evidence:
the two same-seed runs at 64 and 96 differ by 5.6e-9 nats/event in the float64
paired temporal gain ([Ch. 4 §8.3](04-flows-and-density-estimation.md#83-is-the-result-step-count-sensitive-what-the-repo-shows),
[Ch. 8 §10.6](08-flowquake-synthesis.md#106-ode-step-count-sensitivity)). The
correctness test trains the flow on a known Gaussian and checks `log_prob`
against the analytic density to 0.15 nats (`tests/test_flow.py`).

**Q63.** Why is the flow's final layer zero-initialized?

**A.** With `net[-1]` weight and bias zero, the initial velocity field is
identically zero, so the flow map is the identity and `p_1 = p_0 = N(0,1)`
*exactly* at step 0. That is a well-defined, finite-likelihood starting point;
without it early flow-matching training can start from a density that is
already pathological in the tails and produce enormous initial gradients.

**Q64.** Explain the SSD scan and its "duality".

**A.** The selective SSM recurrence is `H_t = a_t H_{t-1} + dt_t B_t x_t^T`,
`y_t = C_t · H_t`, with `a_t = exp(-dt_t A_h)` input-dependent. Split the
sequence into chunks of `Q = 64`. Within a chunk,

```
y_intra[t] = sum_{s <= t} decay(t,s) * (C_t · B_s) * dt_s * x_s
```

which is literally a causally-masked attention matrix with a decay mask — that
is the duality in *state-space duality*. Across chunks, only one summary state
per chunk is needed, giving a short sequential recurrence over `L/Q` steps.
Cost: `O(L·Q)` for the quadratic parts and `O(L/Q)` sequential steps
(Dao & Gu 2024, ICML, *Transformers are SSMs*).

**Q65.** What does the "selective" in selective SSM mean here?

**A.** `A`, `B`, `C` and the step size `dt_t` depend on the input, which breaks
the LTI convolution trick that makes classical SSMs fast. In this
implementation selectivity lives entirely in `dt_t = softplus(linear(x_t) +
bias)`: a large `dt_t` decays the old state hard and writes strongly ("this
event matters"); a small one skips. That is a genuinely good fit for
earthquakes — an M7 should reset the state, a background M2.5 should not.

**Q66.** Why is the whole scan run in fp32?

**A.** The decays are exponentials of cumulative sums of `log a_t`. In fp16 the
cumulative sums lose precision, so the pairwise decay `exp(cs[t] - cs[s])`
degrades — both inside a 64-step chunk and, worse, along the inter-chunk
recurrence (32 chunks for a 2048-event crop), where the errors compound
multiplicatively. [ssm.py](../flowquake/ssm.py) line 52 casts every input to
`float()` unconditionally, and the module docstring states the scan is
"sequential across chunks in fp32".

**Q67.** How is the chunked scan verified?

**A.** `tests/test_ssm.py` checks it against a naive fp64 sequential recurrence
(`selective_scan_ref`) at `L = 200` — deliberately not a multiple of 64, so the
padding path is exercised — to `atol = 1e-4`, and separately checks that
scanning `[0:64]` then `[64:128]` with the carried state equals scanning
`[0:128]`. The second is the invariant that makes streaming simulation valid.

**Q68.** What are the relational features, and why are they the real model?

**A.** Each event becomes a `TOKEN_DIM = 32` vector: four core dims
`[log tau_i, x_i, y_i, m_i]` plus, for each of `RECENCY_LAGS = (1,2,4,8,16,32,
64)`, the block `[log(t_i - t_{i-k}), x_i - x_{i-k}, y_i - y_{i-k}, m_{i-k}]`
(28 dims). Read them as precomputed ETAS-kernel raw material: `log dt` is the
Omori argument at seven scales, the displacements are translation-invariant
"where the recent activity is relative to me", and `m_{i-k}` is the
productivity argument for each parent. Exponential lag spacing covers 64 events
of history with 7 features, the way a dilated convolution buys a long receptive
field cheaply.

**Q69.** State `SAFE_TOKEN_DIMS` and the principle behind it.

**A.** `SAFE_TOKEN_DIMS = [0, 3] + list(range(4, TOKEN_DIM))` — 30 dims:
`log tau`, magnitude, and all 28 relational features, **excluding dims 1 and 2,
absolute `x` and `y`** ([model.py](../flowquake/model.py) lines 32–35). The
principle, quoted from the code comment, is that these are
"translation-invariant statistics that cannot fingerprint a specific catalog
position-era". Memorizing geography through the learned conditioning becomes
structurally impossible rather than merely penalized.

**Q70.** Describe the kernel-mixture spatial head and verify it normalizes.

**A.** `f_s(s) = sum_j w_j Kernel_j(s - s_j) + w_unif/A + w_kde * kde(s)`, with
one component at each of the `MIX_K = 80` candidate parents. Per-component
radial density:

```
f(r) = (q - 1) / (pi d^2) * (1 + r^2/d^2)^{-q}
```

Check: `integral_0^inf f(r) 2 pi r dr`. Substitute `u = r^2/d^2`, so
`r dr = d^2 du / 2`:

```
= (q-1)/(pi d^2) * 2 pi * (d^2/2) * integral_0^inf (1+u)^{-q} du
= (q-1) * 1/(q-1) = 1     (requires q > 1)
```

`q` is floored at `q_floor = 1.15` by a softplus, so `q > 1` always holds.

**Q71.** Explain the anisotropy and why it needs no extra normalizer.

**A.** Each component gets `(rho, theta)` giving elliptical axes `d*rho` and
`d/rho` rotated by `theta` ([heads.py](../flowquake/heads.py) lines 88–94). Do
the change of variables rather than asserting it: with
`A = R_theta · diag(d*rho, d/rho)` and the component written as
`(q-1)/(pi d^2)(1 + |A^{-1}s|^2)^{-q}`, substituting `s = A w` gives
`integral = |det A| / d^2 * [ (q-1)/pi * integral (1+|w|^2)^{-q} dw ] = |det A|/d^2`,
and `det A = det R_theta · (d rho)(d/rho) = d^2` — the rotation contributes 1 and
the two `rho` factors cancel. So the integral is exactly 1 with the *isotropic*
prefactor: elongation along fault strike costs no Jacobian term. With free
semi-axes `(a,b)` you would carry `d^2/(ab)` through every normalizer, every
`logsumexp` and every sampler. Full derivation:
[Ch. 8 §2.7](08-flowquake-synthesis.md#27-area-preserving-anisotropy).
`tests/test_heads.py` numerically integrates a forcibly elongated, rotated
component over a grid and confirms it still integrates to 1 within 0.04.
**Concede the cost unprompted:** area preservation forbids "longer *and* wider",
so a large rupture's zone can only be traded in shape, not grown in area, at
fixed `d`.

**Q72.** Why does the kernel-mixture head resist memorization?

**A.** Its components sit at *observed event locations supplied at evaluation
time*. Nothing in the weights encodes "there is a fault at (-120.3, 36.1)"; per
component the MLP sees `[cond, normalized log dt_j, normalized m_j,
log(dist_j + 0.1)/3]` and — because the production config sets
`spatial_density_feat: true` — a fourth feature, `log1p(#components within
`density_radius_km`)/3` ([model.py](../flowquake/model.py) lines 99–121). All four
are functions of *relative* geometry, recency and magnitude, i.e. the arguments
of an ETAS kernel. Translate the whole
catalog 500 km east and the mixture translates with it. The one absolute-geography
term is the KDE background component, and that is a *fitted, frozen, train-era*
map, not learned weights.

**Q73.** Is that a full defence against memorization?

**A.** No, and be precise about it. The KDE background map *is* absolute
geography, fit on train-era events. The claim that survives is narrower:
FlowQuake forbids the **learned** conditioning from seeing absolute
coordinates, and separately uses a fitted frozen background map, which is
standard practice (ETAS's own background is uniform; smoothed seismicity is the
standard upgrade). Those are two different things and conflating them is the
easiest way to be caught overclaiming.

**Q74.** Describe the neural-ETAS spatial head and its three learnable
extensions.

**A.**

```
                 bg(s) + alpha*far_num(s) + sum_{j in near} w'_j K'_j(s)
f_s(s | H)  =  ----------------------------------------------------------
                 mu'*A + alpha*far_den    + sum_{j in near} w'_j Z'_j
```

(i) **Background**: a learned mixture of uniform and four *causal* multi-scale
KDE maps (bandwidths 1.5, 6, 25, 100 km), gated by `sigmoid(kde_gate)`; the KDE
at event `i` is built from events `j < i` only, so it is a legitimate online
forecast quantity, not a fitted map. (ii) **Per-parent modulations**: a
2 -> 32 -> 32 -> 3 MLP maps each near-set parent's `(magnitude, log dt)` to
offsets `(dlog w, dlog d, drho)`. (iii) **Global scalars** `alpha` and `mu'`.
Trainable parameters: 96 + 1056 + 99 = 1,251 in the MLP plus 4 KDE logits and 3
scalars = **1,258** (arithmetic from the module definition; torch is not
installed in the environment this chapter was written in, so that is a hand
count, not a runtime one).

**Q75.** State the normalization argument for that head, precisely.

**A.** The MLP takes `(m_j, dt_ij)` and **never the target location `s`**
([neural_etas.py](../flowquake/neural_etas.py) lines 61–62). Nor does near-set
*selection*: parents are chosen as the top-256 by ETAS weight plus the 128
nearest to the **previous event's** location
([precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
lines 116–129), never to `s`. Therefore, at a fixed forecast time, `w'_j`,
`d'_j` and `rho'_j` are *constants* with respect to `s`. So
`Z'_j = pi/(rho'_j d'^{rho'_j}_j)` remains the exact closed-form integral of
`K'_j`, and the denominator is exactly the integral of the numerator over the
plane. The density is normalized **by construction**, with no numerical
integration anywhere.

**Q76.** What breaks if you let the MLP see `s`?

**A.** The weights become functions of the query point, so
`integral [sum_j w'_j(s) K'_j(s)] ds != sum_j w'_j Z'_j`. The denominator stops
being the integral of the numerator; you are now learning an unnormalized
energy and need a partition function that has no closed form and would have to
be estimated per event on a grid. No amount of retraining fixes it — the model
is no longer a density, so the reported `sll` is not a log-density and is not
comparable to ETAS's.

**Q77.** Explain the near/far split and why it is a good engineering trade.

**A.** The far field — the contribution of all priors outside the near set —
is precomputed and frozen: `far_num = trig_num - near_base_num`,
`far_den = trig_den - near_base_den`
([precompute_trigger_features.py](../scripts/precompute_trigger_features.py)
line 155). The near set (at most 256 + 128 = 384 parents) is recomputed live in
the trainer so gradients can reach its modulations. Result: full-history ETAS
fidelity at the cost of a 384-term sum per event, and the whole precompute is a
one-off 30–60 minute CPU pass.

**Q78.** How is "the head is a strict superset of ETAS" *operationally* proved?

**A.** Two gates the trainer refuses to run without
([train_neural_etas.py](../scripts/train_neural_etas.py) lines 94–106).
Gate 1: build a second head with `kde_gate_init = -30` (KDE mass ≈ 0), no MLP,
all offsets zero, and assert its per-event `sll` matches the package's ETAS
`SLL` to `< 2e-5`. Gate 2: assert the *actual training init* (with its ~5% KDE
gate) is within 0.05 nats of ETAS. The committed reproduction is
`runs/etas_sll_repro.json` -> `max_abs_sll_err` 1.7655796824556091e-09.

**Q79.** Explain the CSEP simulation path.

**A.** `simulate_day_events` in [ntest.py](../flowquake/ntest.py): absorb the
observed catalog up to the day start; broadcast that state across `n_sims`
independent lanes; then loop sampling `(tau, x, y, m)` from the heads, advancing
each lane's history buffers and rebuilding its token, until the sampled time
leaves the 1-day window or `MAX_EVENTS_PER_DAY = 200` is hit. Vectorized over
lanes with `torch.where` masking so finished lanes stop updating without
breaking the batch. The set of simulated catalogs *is* the catalog-based CSEP
forecast.

**Q80.** State the truncated-first-event correction and why it is necessary.

**A.** The last observed event is at `t_last < day_start`, and we *know* nothing
happened in between. So the first simulated event must be drawn from the
conditional `f(tau | tau > day_start - t_last)`, not from `f(tau)`. The code
rejection-samples that truncated conditional for up to
`MAX_REJECTION_ROUNDS = 200` rounds; lanes that never accept are marked as
having no event that day ([ntest.py](../flowquake/ntest.py) lines 88–104).
Skip it and you systematically over-forecast every day's first event, because
you keep drawing gaps that should already have been ruled out.

**Q81.** Is rejection sampling the right way to do that, and what is the bias?

**A.** It is unbiased *conditional on acceptance*: rejection sampling from
`f` restricted to `{tau >= threshold}` gives exactly the truncated conditional.
The bias enters through the 200-round cap: lanes that never accept are recorded
as "no event today" rather than resampled, which under-counts days where the
truncation probability is tiny (i.e. long quiet gaps where the model thinks an
event was overdue). The direction is toward under-forecasting counts on quiet
days. A cleaner alternative would be inverse-CDF sampling of the truncated
distribution, which the flow supports in principle (integrate the ODE to get
`F`) but at much higher cost. This is worth conceding as an approximation
rather than defending as exact.

---

## 5. Tier 4 — Evaluation and statistics (Q82–Q103)

**Q82.** What makes a scoring rule proper, and is `nll` one?

**A.** See Q29 for propriety. `nll = -(tll + sll)` is the negative of a sum of
two log scores over a factorized density, so it is a proper scoring rule for the
joint `(tau, s)` distribution under the assumed factorization. Dropping `mll`
does not break propriety; it just changes which distribution you are scoring —
you are scoring the time-and-location marginal model, not the full marked
process.

**Q83.** Define information gain and compute it here.

**A.** Information gain per event over a reference model is the mean
log-likelihood-ratio, `IG = mean(log L_model - log L_ref)`, in nats/event.
Against Poisson on ComCat_25: ETAS gains `13.261863 - 7.255428 = 6.0064`
nats/event; the FlowQuake composite gains `13.261863 - 7.142122 = 6.1197`. The
FlowQuake-over-ETAS gain of 0.1133 is therefore **1.9% of ETAS's own gain over
"you learned nothing"**. Say that number before an examiner does.

**Q84.** Break that down by axis.

**A.** Temporal: FlowQuake beats ETAS by 0.0533 where ETAS beats Poisson by
`1.4343 - 0.5126 = 0.9217`, so **5.8%**. Spatial (neural head): 0.060 against
ETAS's `-8.6898 - (-13.7745) = 5.0847` over Poisson, so **1.2%**. The temporal
axis is where the relative gain is real; the spatial gain is small in relative
terms even though it is the axis that flips the total.

**Q85.** Explain the stationary block bootstrap and why "stationary".

**A.** Resample *blocks* of consecutive observations with **geometric** lengths
(mean 50) that wrap around the series, rather than individual observations
(Politis & Romano 1994, JASA). Blocks preserve within-sequence correlation. The
geometric length distribution is what makes the resampled series stationary — a
fixed block length induces a periodic non-stationarity at the block boundaries.
Implementation: [flowquake/stats.py](../flowquake/stats.py) lines 42–81,
`mean_block = 50`, `n_boot = 2000` for CIs, percentile method.

**Q86.** Why block length 50, and how sensitive are the conclusions?

**A.** It is a judgement call, chosen to span a typical aftershock burst. The
honest answer is that **no sensitivity analysis is committed**: the block length
lives only in code, is not recorded in any result JSON
(`results/CLAIMS.md` N11), and no `scripts/` call site overrides the default.
That is a legitimate reviewer request and the fix is one line in `stats.py` plus
a re-run. If pressed on the direction of the risk: too-short blocks understate
uncertainty (intervals too narrow), so the conservative move is to show the CI
widening as `mean_block` grows.

**Q87.** What is the bootstrap p-value's floor and why does 0.0005 appear
everywhere?

**A.** `block_bootstrap_pvalue` is add-one smoothed:
`p = min(1, 2 * min((#{means <= 0}+1)/(B+1), (#{means >= 0}+1)/(B+1)))` with
`B = 4000`. With zero replicate means on the wrong side of zero this gives
`2 * 1/4001 = 0.00049988`, stored as 0.0005. **`p_boot: 0.0005` means "at the
resolution floor", not "p equals 0.0005"** — the true p could be arbitrarily
small.

**Q88.** Reconstruct the six-region Holm correction from first principles.

**A.** Every stored `p_boot` in `runs/stats_hardening.json` is exactly
`2(k+1)/4001` for an integer `k` — the count of replicate means on the wrong
side of zero. For `total_with_head_family`: California, Italy and Chile have
`k = 0` (`p = 0.00049988`), Japan `k = 2` (0.00149963), Greece `k = 10`
(0.00549863), Iran `k = 36` (0.01849538). Holm step-down with `m = 6`
([stats.py](../flowquake/stats.py) lines 129–138) takes a running maximum of
`(m - rank) * p` over the ascending order:

| rank | region | p_raw | (6-rank)·p | running max |
|---|---|---|---|---|
| 0 | California | 0.00049988 | 0.00299925 | 0.00299925 |
| 1 | Italy | 0.00049988 | 0.00249938 | 0.00299925 |
| 2 | Chile | 0.00049988 | 0.00199950 | 0.00299925 |
| 3 | Japan | 0.00149963 | 0.00449888 | 0.00449888 |
| 4 | Greece | 0.00549863 | 0.01099725 | 0.01099725 |
| 5 | Iran | 0.01849538 | 0.01849538 | 0.01849538 |

Rounded: 0.003, 0.003, 0.003, 0.0045, 0.011, 0.0185 — **exactly** the stored
`p_holm` values. If you can do this at the board you have proved you read the
artifact.

**Q89.** Why does Chile's temporal `p_holm` read 0.03599 and not 0.036?

**A.** Because `p_raw` is stored rounded to 5 decimals but `p_holm` was computed
from the unrounded value ([stats_hardening.py](../scripts/stats_hardening.py)
lines 169–172 round only at write time). Chile's true `p` is `2*18/4001 =
0.00899775`, and `4 * 0.00899775 = 0.03599`. Recomputing Holm from the *printed*
0.009 gives 0.036. The same explains Japan 0.27393 (`k = 273`) and Iran 0.27293
(`k = 181`).

**Q90.** State Holm's procedure and why it is preferred to Bonferroni.

**A.** Sort p-values ascending `p_(1) <= … <= p_(m)`; reject `H_(i)` if
`p_(j) <= alpha/(m-j+1)` for all `j <= i`. Equivalently the adjusted p-value is
the running maximum of `(m-j+1) p_(j)`. It controls the family-wise error rate
in the strong sense with no assumptions on dependence (Holm 1979, Scand. J.
Statist.), and it is uniformly more powerful than Bonferroni because only the
smallest p-value is multiplied by the full `m`.

**Q91.** What is TOST, and why does this project insist on it?

**A.** Two one-sided tests for *equivalence*: conclude equivalence at margin
`delta` if the `(1-2alpha)` CI of the effect lies entirely inside
`(-delta, +delta)` (commonly attributed to Schuirmann 1987; verify the exact
citation). Here it is a bootstrap 90% CI against `±0.1` nats/event
([stats.py](../flowquake/stats.py) lines 141–158). The reason: **a confidence
interval crossing zero is not evidence of a tie, it is absence of evidence** —
it can reflect low power. Every "ties ETAS" statement in the manuscript is
required to pass TOST. That is a discipline most papers skip.

**Q92.** Give a case where the distinction bites.

**A.** Iran. Its few-shot temporal `dT` is -0.0634 with 90% CI
[-0.1221, -0.0014] — the 95% CI crosses zero (`dT_decision_raw: "tie"`), but
`dT_tost_0.1.equivalent` is **false**. So Iran is *not* shown to be equivalent
to ETAS temporally; it is merely not shown to be different. Greece, by
contrast, is `equivalent: true` at ±0.1. Both are in
`runs/stats_hardening.json` -> `per_region`.

**Q93.** What is McNemar's test and why is it the right one for paired CSEP
pass/fail?

**A.** For paired binary outcomes, the concordant pairs carry no information
about which model is better; all the evidence is in the discordant counts
`b_01` and `b_10`. The exact test is a two-sided binomial test of
`b_01 ~ Binomial(b_01 + b_10, 0.5)` (McNemar 1947, Psychometrika). Here, head
vs ETAS on 83 shared S-test days: **each model passes 77 of the 83** and there
are 10 discordant days, which forces the split to be 5–5 and the table to be
72 both-pass / 5 head-only / 5 ETAS-only / 1 both-fail — i.e. they *agree* on 73
of 83, not 77. (`runs/replacement_readiness.json` stores 83 / 77 / 77 / "10
discordant"; [WORKING.md](../WORKING.md) item 15 flags the manuscript for
misreading the 77 as an agreement count, so get this right.) Then
`p = 2 * P(X <= 5)` with `X ~ Bin(10, 0.5)`, and since 5 is the median that
exceeds 1 and is capped: `p = 1.0000`. Head vs the production kernel-mixture
head: 81 shared days, head 75, production 78, 9 discordant split 3–6, so
`p = 2 * (C(9,0)+C(9,1)+C(9,2)+C(9,3))/2^9 = 2*130/512 = 0.5078125`
— matching `results/CLAIMS.md` C20 to all digits.

**Q94.** What does `p = 1.00` on that McNemar test actually let you claim?

**A.** Only that the head's spatial *calibration* is statistically
indistinguishable from ETAS's on those days. It is **not** evidence the head is
better on consistency, and with 10 discordant days the test has almost no power
to detect a moderate difference. The claim being made is narrower and correct:
the spatial likelihood gain **cost nothing** in consistency. Say it that way.

**Q95.** Describe the three CSEP tests.

**A.** Catalog-based consistency tests, each comparing an observed statistic to
its distribution across simulated catalogs (Schorlemmer et al. 2007, SRL;
Zechar, Gerstenberger & Rhoades 2010, BSSA; implemented via pyCSEP, Savran et
al. 2022, SRL). **N-test**: is the observed *count* consistent? reports
`delta_1 = P(N_sim >= N_obs)` and `delta_2 = P(N_sim <= N_obs)`. **S-test**: are
the observed *locations* consistent with the forecast's spatial rate density?
**M-test**: is the observed frequency–magnitude distribution consistent?

**Q96.** What is the pass criterion and what is excluded from the denominator?

**A.** Two-sided 95%: pass iff every reported quantile is `>= 0.025`
([csep_forecast.py](../flowquake/csep_forecast.py) lines 233–258). Non-evaluable
days — NaN, or pyCSEP's `(-1, -1)` sentinel — are excluded from the denominator
rather than counted as failures, which is why the reported numbers are
`S 79/85`, not `S 79/100`.

**Q97.** Isn't excluding those days a way of hiding failures?

**A.** It would be if the exclusions were outcome-dependent, and you should say
so. They are not: the sentinel means the test *could not be computed* (e.g. the
observed statistic is NaN because the day has no observed events in-region),
not that it failed. The exclusion rule is a committed function, not a
per-day judgement. What *is* fair to criticise is that the two 10^4-catalog runs
carry stale stored summaries: re-running the committed `csep_summary()` over
`runs/n1_density/csep/csep_results.json`'s own `results[]` gives S 85/91 =
0.9341, not the stored 85/92 = 0.9239 (`results/CLAIMS.md` A2). The correction
runs *in the author's favour*.

**Q98.** What does a pass rate of, say, 95/100 on the N-test mean?

**A.** Under a correct forecast, each day's test rejects with probability ~5% at
the two-sided 95% level, so 95/100 is exactly nominal. Two caveats an examiner
will want: (i) the 100 days are **not independent** — consecutive days share the
same aftershock sequences — so the binomial confidence interval on the pass rate
is optimistic; (ii) a pass rate *above* nominal (e.g. 98%) is not "better", it
suggests the forecast distribution is too wide.

**Q99.** How do you know FlowQuake and ETAS were tested on the same days?

**A.** All three pipelines (`csep_forecast.py`, `csep_forecast_head.py`,
`etas_csep.py`) write byte-compatible `csep_ascii` CSVs scored by the same
`csep_summary`. `results/CLAIMS.md` C13 records that the `results[].day` lists
in all four committed CSEP runs are **element-for-element identical** (day
indices 0…4763). And `etas_csep.py`'s docstring records the alignment check:
ETAS's `timewindow_end` equals FlowQuake's `test_start` (2007-01-01 for
ComCat_25), so day offset `d` is the identical wall-clock window in both.

**Q100.** What is the simulation budget, and was it matched?

**A.** The standalone production run used 10^4 simulated catalogs per forecast
day; the head-to-head runs used a matched 10^3 for both models
(`n_sims: 1000` in `csep_h2h_fq` and `csep_h2h_etas`). Matching matters because
`n_sims` is passed straight through as pyCSEP's `n_cat`
([csep_forecast.py](../flowquake/csep_forecast.py) line 180). The repository's own
audit (`results/CLAIMS.md`, "The N 73/100 → 97/100 attribution cannot be
isolated"; [WORKING.md](../WORKING.md)) argues that if the
declared `n_cat` exceeds the number of catalogs actually written, pyCSEP pads
with empty catalogs and that alone produces apparent under-forecasting. State
that as the audit's stated mechanism, not as something this repository
demonstrates: nothing here isolates it experimentally (Q117).

**Q101.** Why is a *paired* comparison used rather than comparing two means?

**A.** Because the difficulty of each event is shared between the two models: an
event in the middle of a well-behaved Omori sequence is easy for both, and an
isolated background event is hard for both. Pairing removes that shared
variance, which is enormous here. `paired_vs_etas`
([evaluate.py](../flowquake/evaluate.py) lines 28–51) merges FlowQuake's
per-event scores against ETAS's per-event `TLL`/`SLL` from
`augmented_catalog.csv` on timestamp, and the block bootstrap then handles the
residual autocorrelation.

**Q102.** What is the pairing coverage issue, and how big is it?

**A.** Outside California and Italy the two pipelines do not score identical
event sets, because the ETAS pipeline bins magnitudes (`round_half_up` to 0.1)
*before* the completeness cut while FlowQuake's temporal CSV uses the raw cut.
Events sitting exactly at the completeness edge land in one set and not the
other. Coverage is reported per region in
`runs/stats_hardening.json`: Japan 96.26%, Chile 97.09%, Greece 91.95%, Iran
89.04%. Iran's 89% is the one worth flagging yourself — 11% of ETAS-scored
events are dropped, and they are not a random 11%, they are completeness-edge
events.

**Q103.** How were the reported CIs made, and can a reader recompute them?

**A.** They come from the block bootstrap over per-event paired gains. A reader
**cannot** recompute them from this repository: per-event score CSVs are
excluded by `.gitignore`, so every CI and everything in
`runs/stats_hardening.json` exists only as a stored summary. Exactly one
per-event file is tracked repo-wide
(`runs/neural_etas/ComCat_25/per_event_forward_full.json`). That is stated
plainly in [results/CLAIMS.md](../results/CLAIMS.md) and is a real
reproducibility limitation.

---

## 6. Tier 5 — Hostile / defence (Q104–Q129)

*These are phrased the way a sceptic phrases them. Several correct answers are
concessions.*

**Q104.** *"Your spatial head is initialized from the ETAS inversion you claim
to beat. Isn't this circular?"*

**A.** It is not circular — it is a **bounded** claim, and the boundary is
stated in [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md), the README
and the manuscript. The head is a strict superset of ETAS's spatial density,
verified to 1.77e-9 nats with the gate closed, and the reported gains are always
measured against the *package's* ETAS scores, never against the near-ETAS
initialization. So "we beat ETAS" means "we strictly improve on ETAS's own
density from its own starting point" — which is a valid comparison, but it means
the artefact is an **upgrade of a deployed ETAS system, not an inversion-free
replacement for one**. If you want the inversion-free claim, this evidence does
not give it to you, and I would not defend it.

**Q105.** *"You report a six-region win but your own file says three of those
regions are not significant temporally. Which is it?"*

**A.** Both, and they are different families. `runs/stats_hardening.json` has
two Holm families. The **temporal** family: California 0.003, Italy 0.003, Chile
0.036 significant; Japan 0.274, Iran 0.273, Greece 0.648 **not**. The **total**
family (temporal + neural-ETAS spatial): all six Holm-significant at
p ≤ 0.0185. So the six-for-six claim is about *total* likelihood and rests
substantially on the spatial head; the temporal win is a three-region result,
not a six-region one. Anyone reporting "six-region win" without that sentence is
overclaiming.

**Q106.** *"Your Mamba encoder is switched off in every production run. Why is
it in the title?"*

**A.** It should not be, and the README's own first line is vulnerable here:
it opens "Selective-SSM (Mamba-style) whole-catalog encoder + flow-matching
marked point process", while every production config sets `h_bottleneck: 0` and
[model.py](../flowquake/model.py) lines 82–83 sets `self.encoder = None` in that
case. Counting tracked YAMLs: 114 set `h_bottleneck: 0`, 3 omit it (defaulting
to 0), and 6 files — three distinct ablation configs under `runs/ablation_h/`,
each committed twice — set it higher. The encoder's role is to *make a
scientific point* (§4.3: the memorization mechanism), not to be the production
model. The defensible framing is "a structured TPP whose whole-catalog reach
comes from relational features, with an SSM ablation that explains why the
learned alternative fails". Note also that [STACK.md](../STACK.md) states "181
set `h_bottleneck: 0` and exactly three set it higher" — the 181 is wrong; the
tracked count is 114, and 6 files set it higher.

**Q107.** *"You looked at the test set how many times?"*

**A.** More than once, and the repository says so. `train_neural_etas.py`'s own
docstring reads: *"The reported grids include ablations and multiple seeds; do
not describe these runs as a test-scored-once protocol."* Count what is actually
committed under `runs/neural_etas/`: three seeds of the `full` head in each of
the six regions (18 runs), plus seed-0 `--no-mlp` background-only ablations in
four regions (ComCat, Italy, Japan, Chile) and seed-0 `--refit-globals` controls
in two (ComCat, Italy). Do **not** say "three seeds of every configuration" —
only `full` has three. Model selection uses the *validation* window
(early stopping on val `sll`, and on val `nll` for the TPP), which is the
protocol contract; but the final test numbers were computed for many variants
before the headline was chosen. The mitigation that carries real weight is the
2020–2026 forward window, which was fetched after the fact with the benchmark's
own recipe and used for neither fitting nor early stopping — and the gains
replicate there (dT +0.0574, dS +0.0666, dTot +0.1241 [0.1035, 0.1455],
`runs/total_win.json`). It is still retrospective.

**Q108.** *"You call 2020–2026 an out-of-time test, but those earthquakes
already happened. What stops you from having tuned to them?"*

**A.** Nothing structural, and the repository refuses to claim otherwise.
`runs/total_win.json`'s own `notes` field says: *"forward_2020_2026 is a
retrospective out-of-time/pseudo-prospective replication, not a registered
prospective forecast."* The manuscript says "We do not call this a registered
prospective forecast because those events existed during development." The
defence is procedural, not logical: the checkpoint is frozen, `evaluate.py`
swaps the catalog and window without touching the checkpoint's stats
([evaluate.py](../flowquake/evaluate.py) lines 62–66), and the ETAS baseline is
the published 2007-fit inversion. But the objection is correct in principle and
the only answer is rung 4 of the readiness ladder — a registered prospective
forecast with external custody.

**Q109.** *"Your headline gain is 0.113 nats. ETAS beats a Poisson process by
6.0. Isn't your contribution 2% of the problem?"*

**A.** Yes, 1.9% (Q83), and that is the right way to size it. Two things make it
non-trivial anyway. First, the benchmark's finding was that **no** NPP beat ETAS
on total likelihood at all, so the sign is the result, not the size. Second, the
axis that flipped is the one every prior NPP lost worst on. But I would not
defend "a large improvement" — it is a small, statistically robust improvement
in the regime where the incumbent was thought unbeatable.

**Q110.** *"Your production model's spatial head loses to ETAS by 0.37 nats and
you report a spatial win. Explain."*

**A.** Two different models. The production TPP's kernel-mixture head has
`dS` -0.3691 on ComCat and loses on all five California catalogs
(`runs/replacement_readiness.json` -> `checks[california_spatial_total_gap]`,
level **WARN**). The spatial win comes from the *second* model, the
full-history neural-ETAS head trained by `scripts/train_neural_etas.py`, `dS`
+0.060. The headline total is `tll` from model 1 plus `sll` from model 2, paired
against the same region's ETAS. This is the single most common source of
confusion in the project and the reason `runs/stats_hardening.json` has two keys
called `dTot_mean` with opposite verdicts.

**Q111.** *"So your headline number is a Frankenstein of two separately-trained
models. Is that even a model?"*

**A.** It is a legitimate composite because the benchmark's scoring convention
factorizes `f = f_t · f_s · f_m` and scores the factors separately, so
combining a temporal factor from one model with a spatial factor from another
still yields a valid joint density — provided both condition on the same
history, which they do. But it is not a single deployable artefact, and the CSEP
work is what makes that honest: `csep_forecast_head.py` runs the *combination*
through the simulator (FlowQuake supplies counts/times/mags, the neural-ETAS
head supplies locations) and it passes N 95/100, S 79/85, M 90/92. Without that
run, "composite" would be a fair accusation.

**Q112.** *"You claim per-seed spread is ≤0.003. Your own audit says three
regions exceed it."*

**A.** Correct, and the audit is right. `results/CLAIMS.md` M1: the per-seed
`dS` ranges are ComCat 0.0008, Italy 0.0019, Japan 0.0003, **Chile 0.0070**,
**Iran 0.0056**, **Greece 0.0045**. The manuscript states ≤0.003 in one place
and ≤0.006 in another; the correct bound is ≤0.007. The companion clause — "all
six clear zero at every seed" — **is** true: every per-seed `dS_ci` is strictly
positive. So the conclusion survives; the stated bound does not.

**Q113.** *"Are the six composite totals three-seed or one-seed?"*

**A.** As committed, one seed — and this answer is more precise than
`results/CLAIMS.md` M2, so check it yourself. The manuscript's front matter says
"Per-event and full-suite results are 3-seed (mean ± std)" and §4.4 reports the
totals in the same paragraph as 3-seed head means. The arithmetic settles it:
Italy's `0.2095 - 0.0712 = 0.1383`, which is the seed-0 `dS` exactly, not the
3-seed 0.1373. **But the code has since been fixed and the artifact has not been
regenerated.** `head_seed_csvs` in
[stats_hardening.py](../scripts/stats_hardening.py) lines 121–135 now globs
`per_event_full_s*.csv` and its docstring says outright: *"Previously this was
hardcoded to `per_event_full_s0.csv`, so the total-likelihood headline was a
single training seed while the manuscript described a three-seed mean."* That
landed in commit `2e8fa8a`, which also fixed `total_win_summary.py` and
`transfer_neural_etas.py` and added a `single_seed_warning` flag. Neither
`runs/stats_hardening.json` nor `runs/total_win.json` has changed since commit
`9507356`, and neither carries `single_seed_warning` — so the committed numbers
are still seed 0 and the fix is one CPU re-run away, blocked only on the
gitignored per-event CSVs. Material only for Chile (stated +0.061; 3-seed would
be +0.064).

**Q114.** *"You claim the temporal wins are significant in every era of the test
window."*

**A.** That claim is wrong and the artifact contradicts it — `results/CLAIMS.md`
M6, the one claim in the manuscript that its own evidence actively refutes.
`runs/prospective.json` stores **no per-window CI or p-value of any kind**.
Chile has only 10 of 19 180-day windows positive
(`bins_dT_positive_frac` 0.5263), with negatives to -0.0557; California is 23
of 27 (0.8519). What is backed is the window-fraction statement and the overall
CI [0.0404, 0.0678]. The era-level significance claim should be cut.

**Q115.** *"Which `etas` package produced your baselines?"*

**A.** Unresolved, and this is the highest-priority open item in the repository
(`results/CLAIMS.md` N1). `pyproject.toml` lines 25–38 records **two** candidate
forks, `lmizrahi/etas` and `ss15859/etas`, states they are different code, and
marks the choice `TODO [USER, blocks release]`. No version, commit, sha,
package, env or provenance key exists in any of the 136 committed run JSONs or
the 90 YAMLs under `runs/` (the tracked YAML total is 123 — 90 in `runs/`, 33 in
`configs/`; `results/CLAIMS.md` N1 audits the 90); the inversion logs are
gitignored and absent. Scope of the
damage: the fork affects only the five foreign-region inversions plus
`ComCat_25_refit2020` and the §4.2 ETAS CSEP column. **California is safe either
way**, because `runs/forward_etas/summary.json` -> `params_frozen_from` is
`"ComCat_25 inversion (train<=2007, published with benchmark)"` — the
benchmark's own shipped output. The decisive artifact is
`etas-*.dist-info/direct_url.json` in the training environment.

**Q116.** *"Can I reproduce your foreign-region baselines?"*

**A.** No. Six ETAS configs the manuscript depends on — `Japan_25`, `Chile_25`,
`Greece_25`, `Iran_25`, `Italy_25` and `ComCat_25_refit2020` — are not shipped
by the benchmark, are written by no script in this repository, and exist only
inside the gitignored `reference/` tree. Nobody but the author can regenerate
the §4.5 region baselines or the §4.1 refit control. That is item 10 of
[WORKING.md](../WORKING.md)'s laptop list and it is a genuine reproducibility
failure, not a packaging inconvenience.

**Q117.** *"You attribute an N-test jump from 73/100 to 97/100 to a source-set
bug. Prove the cause."*

**A.** I cannot isolate it, and the repository says so. `runs/etas_csep_pod`
records `n_sims` 10000 while `runs/csep_h2h_etas` records 1000, and `n_sims` is
passed straight through as pyCSEP's `n_cat`. If the pod run scored ~1000 real
catalogs while declaring `n_cat = 10000`, pyCSEP pads with ~9000 empty catalogs,
which alone produces the observed count under-prediction — and that run's M-test
rate is *also* depressed (73/92), which padding explains and a source-set bug
does not. The two endpoints are backed; the single stated cause is not. The
passage should report both differences between the runs.

**Q118.** *"Your model needs the target region's ETAS inversion, a per-region
normalization, and a train-era background map. What exactly is 'transferable'
about it?"*

**A.** Less than the word suggests, and the manuscript's own caveats say so. The
transferable object is the *learned relational structure* — the temporal flow's
weights and the head's per-parent MLP — which is translation-invariant and does
transfer: zero-shot within-regime spatial transfer is 7 of 7 win-or-tie, and
few-shot recalibration of four scalars turns all seven into wins
(`runs/neural_etas/spatial_transfer_summary.json`). What does *not* transfer is
the preprocessing: input standardization, a smoothed-seismicity map, and — for
the spatial head — the target's ETAS inversion. Cross-completeness transfer
fails outright, 0 of 4. So: "lighter than an ETAS inversion, but not zero
target-catalog preprocessing" is the sentence, and it is the README's.

**Q119.** *"Your 'foundation model' had the target region in its pre-training
pool. That is not zero-shot."*

**A.** Correct for the *pooled global* checkpoint, and the audit flags it: the
`pooled_global_temporal` check is level **WARN** with the message "this is
pooled deployment, not leave-one-region-out zero-shot". The genuine LOO
experiments are separate (`runs/pool_loo_{japan,chile,greece,iran}`), and their
results are weaker: pooled zero-shot loses on Greece (-0.0552) and Iran
(-0.1580), and only few-shot brings them to ties. The honest statement is
"one pooled checkpoint deployed across regions", not "zero-shot transfer".

**Q120.** *"Your Japan result is +0.039 nats. Is that meaningful?"*

**A.** The artifact answers this against the author: `dTot_abs_below_0.05: true`
in `runs/stats_hardening.json`. It is statistically positive
(CI [0.0163, 0.0620], Holm p 0.0045) but below the project's own stated 0.05-nat
interpretability margin. And Japan's *temporal* component is negative
(-0.0139, Holm p 0.274), so the Japan total is carried entirely by the spatial
head. That should be said before it is asked.

**Q121.** *"Greece and Iran are 'wins' using few-shot transfer, on 1,748 and
1,121 events, with 92% and 89% pairing coverage. Are those results?"*

**A.** They are the weakest two rows and should be presented as such. Both use
`temporal_variant: "fewshot"`, not native training — natively they *lose*
temporally (Greece -0.107, Iran -0.276). Iran's total CI is [0.0098, 0.1711],
which nearly touches zero, and its Holm p of 0.0185 is the family maximum. Its
89.04% coverage means 138 ETAS-scored events are unpaired, and they are
completeness-edge events, not a random subset. I would report the six-region
result as "four regions on native temporal models plus two on transfer, with the
two transfer rows the least secure".

**Q122.** *"Your background map is fit on train-era data and is absolute
geography. So you do memorize geography."*

**A.** Yes — deliberately, and the distinction has to be stated carefully. What
§4.3 shows is that a *learned* whole-catalog embedding lets the heads
fingerprint catalog position-eras and blows held-out `nll` from 7.62 to 19.65.
What FlowQuake does instead is use a fitted, frozen, train-era-only
smoothed-seismicity map — the standard background method in operational
forecasting, and strictly less expressive than a learned map because it cannot
condition on time. Calling both "memorization" would be sloppy in both
directions, but the model is not geography-free and the README should not be
read as saying it is.

**Q123.** *"Your CSEP S-test denominators are wrong in two runs."*

**A.** They are, and the direction is in my favour, which is the only reason it
is not worse. `runs/n1_density/csep/csep_results.json` day 2982 records
`S.quantile [-1.0, -1.0]` with `observed` NaN — the harness's own not-evaluable
sentinel that `csep_forecast.py` line 246 excludes. Re-running the committed
`csep_summary()` over that file's own `results[]` gives S 85/91 = 0.9341, not
the stored 85/92 = 0.9239. Same off-by-one in `runs/final_s1555/csep`. The
manuscript claims at lines 977–979 that this denominator was already corrected;
that fix landed in prose only. The four 10^3-catalog runs are self-consistent
with current code.

**Q124.** *"Show me a number in your paper that your own audit cannot back."*

**A.** Twelve of them, enumerated N1–N12 in
[results/CLAIMS.md](../results/CLAIMS.md). The ones a reviewer would care about
most: the `etas` fork (N1); the refit-2020 parameter table and its "12 EM
iterations" (N2, N3); the ETAS *temporal* reproduction to 1e-5/event (N4); the
gridded simulator's 9.5e-7-nat validation (N5) — the code actually prints a
max-abs error over 40 randomly sampled events with a 1e-3 threshold, which is a
weaker statement than the manuscript makes; and the §4.4 distance-band
localization numbers (N6–N8), for which
`runs/n1_density/spatial_gap_decomp.json` contains no distance strata at all.
Across 142 traced rows the tally is 63 MATCH, 51 ROUNDING, 13 NO ARTIFACT,
11 MISMATCH, 2 AMBIGUOUS, plus two special-status rows (T23, a MATCH carrying a
NO ARTIFACT caveat, and X31, MISMATCH-adjacent wording) — 63+51+13+11+2+2 = 142.
Those 142 rows cover 134 distinct claims, because several findings are cited
from more than one family table.

**Q125.** *"Your temporal win on San Jacinto is reported as a win in the table
and a tie in the text."*

**A.** It is a **tie** and the table's Δ column is a 3-seed mean, not a
significance verdict. The block-bootstrap CI is
[-0.005686476386749143, 0.07592596149130082] with stored `decision: "tie"`
(`runs/replacement_readiness.json`). The manuscript says the interval "touches
zero"; it *crosses* zero, which is a different statement, and
[WORKING.md](../WORKING.md) item 15 flags exactly this wording. The correct
summary of §4.1 is: five of five positive in 3-seed means, four of five
block-bootstrap significant, San Jacinto a tie.

**Q126.** *"The two artifacts holding your California total disagree."*

**A.** Slightly, and it is a bootstrap-seed artefact rather than a
contradiction: `runs/total_win.json` -> `test_2007_2020.dTot.ci` is
[0.1006, 0.1268] while `runs/stats_hardening.json` ->
`total_with_head_family.California.dTot_ci` is [0.1006, 0.1261]. Same point
estimate (0.1133), same lower bound, upper bounds differing by 0.0007 — two
runs of a 2000-replicate percentile bootstrap with different seeds. It should be
reconciled, but nothing turns on it. I would rather point it out than be shown
it.

**Q127.** *"Your ablation says 'reproducible from the committed checkpoints'.
Are the checkpoints committed?"*

**A.** No. `git ls-files | grep '\.pt$'` returns nothing; `*.pt` is gitignored.
The §4.3 *metrics* are committed
(`runs/ablation_h/memorization_figure.json`), the checkpoints are not. The
sentence should say "reproducible from the committed metrics". That is
`results/CLAIMS.md` X31.

**Q128.** *"Your model can't produce a rate field, only a next-event density.
Isn't that useless operationally?"*

**A.** It is a real cost of choosing to model `f(tau | H)` instead of `lambda`.
You cannot read a hazard off a grid; you must simulate. The mitigation is that
the simulation path exists, is vectorized over lanes, and has been run through
the incumbent's own pyCSEP harness at a matched budget — so the *operationally
relevant* object (a catalog-based forecast) is produced and tested. The residual
cost is compute: a 100-day × 10^3-catalog run is expensive relative to
evaluating an ETAS rate on a grid, and `runs/replacement_readiness.json` lists
"GPU-vectorize the full-history head grid simulator" as a next rung for exactly
this reason.

**Q129.** *"Give me the one sentence you are prepared to defend in public."*

**A.** The one in [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md):
*"FlowQuake is a transferable neural point-process candidate that beats ETAS
temporally on dense catalogs and, with a full-history neural-ETAS spatial head
initialized from each region's ETAS inversion, beats ETAS on total likelihood
across the six tested regions; it is not yet an operational replacement for ETAS
systems."* Every clause in it is load-bearing and every qualifier is there
because dropping it would make the sentence false.

---

## 7. Tier 6 — Code level (Q130–Q145)

*For each: what does the line do, why is it there, and what breaks without it?*

**Q130.** `dm = torch.clamp(m - mc, min=0.0) + 0.005`
([heads.py](../flowquake/heads.py) line 173).

**A.** A half-bin shift so that a continuous density is evaluated at the
*centre* of the magnitude bin rather than its lower edge. Catalog magnitudes are
discretized; evaluating `f(m) = beta exp(-beta(m - m_c))` exactly at grid points
systematically over-scores, because the true probability of the bin is
`integral` over the bin, whose density-equivalent evaluation point is the
midpoint. Remove it and `mll` inflates by `beta * 0.005 ≈ 0.012` nats — small,
and `mll` is not in `nll`, so the headline is untouched. **Flag the
inconsistency:** both `STACK.md` and `MANUSCRIPT.md` describe this as a half-bin
for "the catalog's 0.1-unit magnitude discretization"
([STACK.md](../STACK.md) lines 726–727, [MANUSCRIPT.md](../MANUSCRIPT.md) line 242),
but half of 0.1 is 0.05, not 0.005. The constant 0.005 is a half-bin for a
**0.01** grid — numerically `mag_dequant/2` for `configs/n1_density.yaml`'s
`mag_dequant: 0.01`, though note the 0.005 is *hardcoded* in `heads.py`, not
read from the config (`model.py` line 61 keeps `mag_dequant` only "for config
compat (GR head absorbs it)"), so that agreement is a coincidence rather than a
wiring. Switching to 0.05 would move `mll` by `beta * 0.045 ≈ 0.10` nats.
**The strong form of the criticism, and the one to use** — because it needs no
uncommitted data: `0.005` is hardcoded for *all eleven catalogs*, and the
ISC/INGV catalogs are certainly on a 0.1 grid (the ETAS side assumes it,
`round_half_up(x, delta=0.1)` at
[precompute_trigger_features.py:36](../scripts/precompute_trigger_features.py#L36)),
so **no assignment of ComCat's precision makes a single constant right
everywhere** ([Ch. 2 §5.5](02-seismology.md#55-the-0005-in-headspy--what-it-corrects-and-the-doc-bug);
on a 0.1 grid it biases fitted `beta` 11.6% high). Neither choice touches `nll`,
which excludes `mll` — but simulated magnitudes and hence the CSEP M-test move.

**Q131.** `torch.where(mask, diff, torch.full_like(diff, -torch.inf)).exp()`
([ssm.py](../flowquake/ssm.py) line 33).

**A.** `segsum_decay` builds the within-chunk pairwise decay matrix
`M[t,s] = exp(sum_{r=s+1..t} log a_r)` as a cumulative-sum difference. The
`-inf` fills the strictly-upper triangle so that after `.exp()` those entries
are exactly 0 — **causal masking**. Without it, `y_intra[t]` would sum
contributions from `s > t`, i.e. the model would read the future, and every
likelihood would be invalid while looking perfectly healthy. Using a large
negative finite number instead of `-inf` would leave `exp` at a tiny nonzero
value and leak.

**Q132.** `obs = csep.load_catalog(str(obs_path))` inside the per-day loop
([csep_forecast.py](../flowquake/csep_forecast.py) lines 189–194).

**A.** pyCSEP's `.filter*` methods mutate the catalog object **in place**. If
the observed catalog were loaded once outside the loop, each day's window and
region filters would compound and the observed set would silently shrink day
after day, driving the N-test toward apparent over-forecasting. The comment
documents a real bug that was found and fixed. Remove the fresh reload and every
CSEP number after day 1 is wrong, monotonically.

**Q133.** The rejection loop at [ntest.py](../flowquake/ntest.py) lines 92–103.

**A.** Draws the first simulated event from the *truncated* conditional
`f(tau | t_last + tau >= day_start)` (Q80). Remove it and every forecast day
over-counts, because the simulator is allowed to place its first event before
the day began — an event we know did not happen. The `MAX_REJECTION_ROUNDS = 200`
cap and the `active = active & ~need` line are the honest approximation: lanes
that never accept in 200 rounds are recorded as having no event that day.

**Q134.** `SAFE_TOKEN_DIMS = [0, 3] + list(range(4, TOKEN_DIM))`
([model.py](../flowquake/model.py) line 35).

**A.** Excludes token dims 1 and 2 — absolute `x` and `y` — from head
conditioning, leaving 30 translation-invariant dims. Remove the exclusion (i.e.
pass all 32 dims) and the heads can condition on absolute position, which is
precisely the channel §4.3 shows leads to memorization; you would expect the
`h > 0` pathology to appear even at `h = 0`. Note it also determines
`cond_dim = len(SAFE_TOKEN_DIMS) + h_bottleneck = 30 + h`, so changing it
silently changes every head's input width and breaks checkpoint loading.

**Q135.** `mask[: self.burn_in] = False` in `CropDataset.__getitem__`
([data.py](../flowquake/data.py) line 346).

**A.** A training crop starts with no history: the SSM state is zero and the
lag features at position 0 are clamped to the crop's first event, so the first
few hundred positions are garbage. The burn-in of 256 positions removes them
from the loss. Remove it and you train on systematically corrupted conditioning
— the model would learn to predict from features that never occur at evaluation
time, where the full sequence is scored in one pass with real history.

**Q136.** `mask_np[1:] = nxt[:-1]`
([evaluate.py](../flowquake/evaluate.py) lines 114–116).

**A.** The `+1` convention: token `i` describes event `i`, but the prediction
made *at* position `i` is about event `i+1`. So `mask[i]` means "event `i+1` is
a target"; to recover which *events* were scored (for timestamp pairing against
ETAS's `augmented_catalog.csv`) you shift the mask forward by one. Get it wrong
by one and every FlowQuake score is paired against the *neighbouring* ETAS score
— a bug that would not crash, would barely change the mean, and would destroy
the paired CI. `tests/test_data.py` guards this alignment.

**Q137.** `w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)` and
`w[:, self.n_comp] = w[:, self.n_comp].clamp(min=1e-6)`
([heads.py](../flowquake/heads.py) lines 128–129).

**A.** During long autoregressive simulation, a single NaN anywhere in the
conditioning poisons the mixture logits, and `torch.multinomial` on an all-zero
or NaN weight vector trips a device-side assert that kills the whole run.
Sanitizing and flooring the uniform-background component guarantees a valid
distribution in every lane. Remove them and a 10,000-catalog CSEP run dies part
way through with a CUDA assert and no useful diagnostic. Note the cost: it
converts a NaN into a silent draw from the uniform background rather than an
error, so it can mask an upstream bug.

**Q138.** `tau = torch.nan_to_num(tau, nan=1.0).clamp(TAU_FLOOR_DAYS, 60.0)` and
the `x`, `y`, `m` clamps ([model.py](../flowquake/model.py) lines 265–268).

**A.** Physical bounds on a sampled event: gaps in [1e-7, 60] days, positions
inside the region box, magnitude in `[m_c, 8.5]`. Without them one extreme draw
propagates into the next step's relational features — a `log tau` far outside
the normalization range — and derails the entire simulated catalog from that
point on. The 60-day cap matters specifically because the simulation horizon is
1 day: any `tau > 1` ends the lane anyway, so the clamp costs nothing and
prevents `inf`.

**Q139.** `self.kde_gate = nn.Parameter(torch.tensor(kde_gate_init))` with
`kde_gate_init = -2.94`
([neural_etas.py](../flowquake/neural_etas.py) lines 44–48).

**A.** `sigmoid(-2.94) = 0.0502`, so the background starts at ~5% KDE mass and
~95% uniform — near-ETAS at init but with live gradients into the KDE
components. The code comment records the debugging finding that motivated it: a
hard "exact" logit at +16 makes the softmax starve the KDE components and their
weights freeze at `[1, 0, …]`. Set it to -30 and you recover ETAS *exactly*
(that is the gate-closed verification head) but the background never learns. The
reported gains are always measured against the package ETAS scores, never
against this near-ETAS init, which is what keeps the 5% opening honest.

**Q140.** `nn.init.zeros_(self.net[-1].weight)` / `bias`
([flow.py](../flowquake/flow.py) lines 61–62).

**A.** Makes the initial velocity field identically zero, so `p_1 = p_0 =
N(0,1)` exactly at step 0 — a finite, well-defined starting likelihood.
Without it the flow starts as a random diffeomorphism whose tails can be
arbitrarily bad, producing enormous early gradients and, in the worst case, an
initial `log_prob` of `-inf` on tail events.

**Q141.** `self.mlp[-1].weight.zero_()` / `bias.zero_()`
([neural_etas.py](../flowquake/neural_etas.py) lines 55–57).

**A.** The same idea for the spatial head: with the last MLP layer zeroed, all
per-parent offsets `(dlog w, dlog d, drho)` are exactly 0 at init, so the head
starts *at* the ETAS inversion (modulo the 5% KDE gate). This is what makes the
init-sanity gate — "within 0.05 nats of ETAS" — pass by construction rather than
by luck, and what makes "strict superset" an operational statement instead of a
claim.

**Q142.** `q = self.q_floor + F.softplus(out[..., 2])` with `q_floor = 1.15`
([heads.py](../flowquake/heads.py) line 83).

**A.** The radial kernel `(q-1)/(pi d^2) (1 + r^2/d^2)^{-q}` only normalizes for
`q > 1` (Q70). The softplus floor guarantees it. Remove the floor and the
optimizer will happily push `q` toward 1 to fatten the tail, at which point
`log(q - 1) -> -inf` and the density is not a density. Similarly
`d = d_floor + softplus(...)` with `d_floor_km` (0.1 in the production config)
stops the kernel spiking to infinite density at a component centre.

**Q143.** `rhoj = (self.rho * torch.exp(drho)).clamp(0.05, 5.0)`
([neural_etas.py](../flowquake/neural_etas.py) line 83).

**A.** Keeps the ETAS spatial exponent in the integrable region (Q47). At
`rho -> 0` the normalizer `Z_j = pi/(rho d^rho)` diverges and the conditional
density collapses; at large `rho` the kernel becomes a spike and the far field
is starved. The clamp is applied *after* the neural offset, so the MLP cannot
escape it.

**Q144.** `if h_bottleneck > 0: self.encoder = SSMEncoder(...)` else
`self.encoder = None` ([model.py](../flowquake/model.py) lines 76–83).

**A.** The central experimental control. At `h = 0` the encoder is *not
constructed at all*, so memorization through learned conditioning is
structurally impossible rather than merely penalized; heads see only the 30
relational dims. At `h > 0` the encoder consumes the full 32-dim token —
including absolute `x, y` — and its `h`-dim noisy projection is concatenated
onto the conditioning, so that channel *can* carry geography. Delete the
branch and you either always pay for an unused encoder or lose the ablation
that is the paper's mechanistic result.

**Q145.** `self.hi = int(valid.max()) - burn_in` and `self.n_crops = n_crops`
([data.py](../flowquake/data.py) lines 331–332), where the trainer passes
`n_crops=tc.batch_size * 1_000_000`
([train.py](../flowquake/train.py) line 114).

**A.** `n_crops` makes the crop dataset an effectively infinite stream, so
`steps` and early stopping decide when training ends rather than an epoch
boundary — appropriate when crops are random and overlapping. The `hi` bound
ensures a sampled crop always contains at least one target position past the
burn-in; without it a crop can be entirely burn-in or entirely aux-era, the loss
mask is empty, and the step produces a NaN mean over zero elements.

---

## 8. Tier 7 — What would you do next (Q146–Q157)

*These test judgement, not recall. There is no single right answer; there are
answers that show you understand what the evidence is missing.*

**Q146.** You have one month of GPU time. What do you run?

**A.** Nothing new on GPU. The binding constraints are all CPU or data:
re-running `scripts/stats_hardening.py` with `HEAD_COMBOS` pointed at seeds 1
and 2 to fix the seed-0 labelling (M2); capturing the six author-authored ETAS
configs so the foreign baselines are reproducible (N10); and persisting the
distance-strata that §4.4's localization claims need (N6–N8). GPU time buys
another seed; none of those three is a compute problem.

**Q147.** What single experiment would most increase your confidence in the
result?

**A.** A registered prospective forecast — rung 4. Not because the retrospective
evidence is weak, but because the objection it answers is categorically
different: not "is the model good" but "did anyone see the answers first". No
amount of extra regions or seeds substitutes, because the evidential value comes
from the custody being external. Concretely: deposit the frozen `runs/n1_density`
checkpoint plus the head with a CSEP testing centre, on the 1-day-ahead N/S/M
protocol already implemented in `csep_forecast_head.py`, against ETAS, for a
declared future period.

**Q148.** What is the cheapest experiment that could *falsify* your central
claim?

**A.** A full flETAS baseline: EM with a free (smoothed-seismicity) background,
inverted properly rather than the SGD `--refit-globals` control. The committed
control gives `dS` +0.0564 against the head's +0.060 — that is 94% of the gain
from refitting global parameters with a learned background and **no** neural
modulation. If a proper EM refit with a free background reached +0.060 or
beyond, the neural component of the spatial claim would be dead. It is CPU,
hours per region, and the manuscript already lists it as not run (N12). I would
run it before submission.

**Q149.** How would you make the model inversion-free?

**A.** Replace the ETAS-inversion initialization with a fit of the head's own
global parameters directly by maximum likelihood on the target's training
window — the head already contains ETAS as a special case, so the parameters
`(mu, k0, a, c, omega, tau_tap, d, gamma, rho)` are estimable from the same data
by SGD on `sll` rather than by EM. The reason this is not trivial: the far-field
precompute freezes those parameters, so the whole `trig_num`/`trig_den` pass
would have to be inside the optimization loop or re-run per outer iteration.
That is the single highest-value engineering item, because it removes the
qualifier that bounds every headline claim.

**Q150.** How would you test whether the temporal win is really about data
density?

**A.** The current evidence is observational — five California catalogs at
different `m_c` plus six foreign regions — and confounds density with region.
The controlled version is a within-catalog ladder: take ComCat at `m_c` 2.5 and
re-cut it at 2.75, 3.0, 3.25, 3.5, 4.0, re-invert ETAS at each cut, and measure
`dT` as a function of `m_c` on *the same region and the same era*. If the win
decays monotonically with `m_c` on one catalog, the density story survives; if
it does not, the cross-regime pattern is a region effect wearing a density
costume.

**Q151.** How would you strengthen the memorization result?

**A.** Three things. (i) A mechanistic probe: show the `h > 0` model's
bottleneck activations linearly decode absolute `(x, y)` at high `R^2` while the
`h = 0` conditioning does not — that turns a correlation into a mechanism.
(ii) An intervention: shuffle the mapping between catalog eras and embeddings at
test time and show the `h > 0` model's `sll` collapses to the `h = 0` level.
(iii) A control that separates capacity from geography: give the encoder the
same width but feed it *translation-invariant* tokens only, and show the
pathology disappears. Currently the claim rests on the `h` sweep alone, and
`NOVELTY.md` correctly frames it as a diagnostic contribution rather than a
"first".

**Q152.** The reviewer says the six-region result is a multiple-comparisons
artefact. What do you do?

**A.** Point at the Holm family that is already there — the correction is
applied and the maximum adjusted p is 0.0185 — and then concede the *real*
version of the objection, which is that the family was defined after the
regions were chosen. The clean fix is pre-registration of the next region set:
declare the regions, the `m_c`, and the analysis before inverting ETAS on them.
That is cheap and it is the only thing that makes "six for six" mean what a
reader thinks it means.

**Q153.** How would you decide whether to add a magnitude–location coupling?

**A.** Measure the effect size before building anything: on the test set,
regress per-event `sll` residual (FlowQuake minus ETAS) on the event's own
magnitude, and separately compute the mutual information between magnitude and
distance-to-nearest-prior. If the coupling is worth less than ~0.01 nats/event
it is not worth the normalization risk — a location-dependent magnitude density
is fine (it does not break `f_s`'s normalizer), but a magnitude-dependent
*spatial* density would need the near-set weights to depend on `m`, which they
already do, so the incremental structure is small. This is a "measure first"
answer and that is the point.

**Q154.** What would you do about the `etas` fork question if the training
environment is gone?

**A.** API fingerprinting, in this order: check whether
`inspect.signature(etas.simulation.ETASSimulation.simulate)` in each candidate
fork accepts the streaming `chunksize` kwarg and whether
`ETASParameterCalculation` exposes `prepare_source_events()` —
[etas_csep.py](../flowquake/etas_csep.py) imports both classes at lines 70–71,
calls `reload.prepare_source_events()` at line 88 and passes `chunksize` at line
93, so only a fork supporting both can have run. Then check
whether either fork's `invert_etas.py` requirements force the choice. If neither
discriminates, state in Methods that the fork is unresolved and note that
California — the flagship result — uses the benchmark's own published inversion
and is unaffected. Guessing is worse than conceding.

**Q155.** What is the right next region to add, and why?

**A.** A dense, low-`m_c`, non-California catalog in a regime not yet
represented — the honest gap is that every dense catalog tested is either
California (transform) or Italy (extension), and every subduction region is at
`m_c` 4.0. New Zealand (GeoNet) was attempted and dropped because its ETAS
inversion was prohibitively slow ([REPRODUCE.md](../REPRODUCE.md) notes). Making
that inversion tractable — or picking another dense subduction catalog — is what
would separate "the win is about density" from "the win is about California".

**Q156.** How would you package this for someone who actually operates a
forecasting system?

**A.** Rung 5 of the ladder, and it is mostly not research: one checkpoint plus
its `cat.stats` (already carried in the `.pt`), a documented preprocessing
contract (catalog schema, projection, `m_c`, background-map recipe), a forecast
export in the CSEP ASCII format the code already writes, audit logs recording
the exact inputs per forecast day, and failure-mode monitoring — specifically
alarms on the sanitization paths of Q137/Q138, because they currently convert
bugs into silent uniform draws.

**Q157.** If you had to cut one claim from the paper to make the rest
bulletproof, which?

**A.** The era-level significance claim (M6), because it is the only one the
artifact actively contradicts — everything else is either backed, or unbacked
and removable. Second on the list would be relabelling the six composite totals
as seed 0 (M2), because it is a front-matter claim about the whole paper's
methodology and a reviewer who catches it will re-read everything else with
suspicion.

---

## 9. Worked examples

Five you should be able to do at the board. Each is redoable by hand or in five
lines of Python.

### W1 — b-value by Aki MLE, and its standard error

Greece's training era, from `runs/completeness.json`: `b_train` 1.13,
`n_train_at_mcut` 2612, `recommend_mcut` 4.0.

Forward: given `b`, the mean magnitude excess above completeness is

```
xbar = log10(e) / b = 0.4342944819 / 1.13 = 0.38433 magnitude units
```

so the mean magnitude in that catalog above `m_c = 4.0` is about 4.384 (using
the bin-edge convention, `m_c - dm/2 = 3.95`, it is 4.334).

Standard error:

```
SE(b) = b / sqrt(n) = 1.13 / sqrt(2612) = 1.13 / 51.10 = 0.0221
```

So Greece's train-era `b` is `1.13 ± 0.022`, and the test-era value of 1.00 is
about six standard errors away — a real change, which is exactly why
`check_completeness.py` reports both eras and why the analysis cut (4.0) is set
above both maximum-curvature estimates (3.65 and 3.85).

```python
import math
b, n = 1.13, 2612
print(math.log10(math.e)/b, b/math.sqrt(n))   # 0.38433..., 0.02211...
```

### W2 — reading `sll` and a nat gain physically

```
ETAS  sll = -8.689770387238827   ->  exp = 1.68299e-4 / km^2  ->  1/exp = 5,942 km^2
head  sll = -8.6297607421875     ->  exp = 1.78707e-4 / km^2  ->  1/exp = 5,596 km^2
```

The head is as informative as spreading the mass over 5,596 km^2 instead of
5,942 km^2. Check the consistency: `5942 / 5596 = 1.0618 = exp(0.06)`, and 0.06
is exactly `dS_mean` in `runs/replacement_readiness.json` ->
`checks[spatial_win_comcat]`. A log-density difference *is* a ratio of effective
areas.

For the total: `exp(0.1133) = 1.1200`. Twelve percent more probability density
per event.

### W3 — reproducing the Holm-adjusted p-values exactly

Every `p_boot` in `runs/stats_hardening.json` is `2(k+1)/4001` for an integer
`k` — this is the add-one-smoothed two-sided bootstrap p-value at `B = 4000`
([stats.py](../flowquake/stats.py) lines 113–126).

```python
p  = lambda k: 2*(k+1)/4001
raw = {"California":p(0), "Italy":p(0), "Chile":p(0),
       "Japan":p(2), "Greece":p(10), "Iran":p(36)}
items, run, adj = sorted(raw.items(), key=lambda kv: kv[1]), 0.0, {}
for r,(lab,pv) in enumerate(items):
    run = max(run, (len(raw)-r)*pv); adj[lab] = min(1.0, run)
print({k: round(v,4) for k,v in adj.items()})
# {'California': 0.003, 'Italy': 0.003, 'Chile': 0.003,
#  'Japan': 0.0045, 'Greece': 0.011, 'Iran': 0.0185}
```

Those six values are the stored `p_holm` in `total_with_head_family`, exactly.
Repeat with `k = 0, 0, 273, 17, 1295, 181` for California, Italy, Japan, Chile,
Greece, Iran to reproduce `family_dT_holm`: 0.003, 0.003, 0.27393, 0.03599,
0.64784, 0.27293. Note that **three** of the six cannot be reproduced from the
*printed* `p_raw` values, which are rounded to five decimals: Chile (printed
0.009 → 0.03600, stored 0.03599), Iran (0.09098 → 0.27294, stored 0.27293) and
Japan (0.13697 → 0.27394, stored 0.27393). California, Italy and Greece do
reproduce from the printed values, because the rounding does not bite at their
Holm multipliers (6, 5 and 1 respectively, and the first two land on the same
running maximum either way).

### W4 — McNemar by hand

Head vs ETAS on 83 shared S-test days: **77 passes each** and 10 discordant
days. Two passes-totals of 77 with 10 discordant pins the whole 2×2 table:

```
both pass 72 | head-only 5 | ETAS-only 5 | both fail 1     (72 + 5 = 77 = 72 + 5)
agreement    = 72 + 1 = 73 of 83   <- NOT 77; 77 is each model's pass count
```

```
p = 2 * P(X <= 5),  X ~ Binomial(10, 0.5)
  = 2 * (1 + 10 + 45 + 120 + 210 + 252) / 1024
  = 2 * 638 / 1024  =  1.246  ->  min(1, ·) = 1.0000
```

(The two-sided exact test is capped at 1; here 5 is the median so the p-value is
exactly 1.) Head vs the production kernel-mixture head: 81 shared days, head 75,
production 78, 9 discordant split 3–6.

```
p = 2 * (C(9,0)+C(9,1)+C(9,2)+C(9,3)) / 2^9
  = 2 * (1 + 9 + 36 + 84) / 512  =  260/512  =  0.5078125
```

matching `results/CLAIMS.md` C20 to all seven digits.

### W5 — ETAS spatial density for a single parent (do this on paper)

One parent, magnitude `m_j = 6.0`, `m_c = 2.5`, `dt = 1` day. Take illustrative
parameters `d = 0.01` km^2, `gamma = 1.0`, `rho = 0.55`. Then

```
d_j = d * exp(gamma (m_j - m_c)) = 0.01 * exp(3.5) = 0.01 * 33.115 = 0.33115 km^2
Z_j = pi / (rho * d_j^rho) = 3.14159 / (0.55 * 0.33115^0.55)
    = 3.14159 / (0.55 * 0.54452) = 3.14159 / 0.29949 = 10.4899
```

At `r = 1 km`, `K_j = (1 + 0.33115)^{-1.55} = 1.33115^{-1.55} = 0.64187`. So the
kernel's *normalized* density at 1 km from this parent is
`K_j / Z_j = 0.06119` per km^2, i.e. `log = -2.794`. Compare that to a uniform
background over the benchmark's RELM/CSEP polygon — **9.6e5 km^2**, obtained by
inverting the Poisson baseline `sll = -13.7745` (`e^13.7745 = 959,823`,
`runs/n1_density/eval_test.json` -> `baselines.Poisson.sll`) — so
`log(1/959823) = -13.77`. (Use that number, not California's ~424,000 km^2 land
area: the polygon is the scoring region and includes offshore. Ch. 1 §10 and
Ch. 2 §10 use 9.6e5 for the same reason.) The
triggering term dominates by eleven nats near an active parent — which is why the
spatial score is essentially a question of *which parents you include*, and
hence why full-history coverage is the whole game.

---

## 10. How this shows up in FlowQuake

A map from tier to source. [STACK.md](../STACK.md) is the walkthrough; this is
just the index for revision.

| tier | primary sources | key artifact |
|---|---|---|
| Foundations | [STACK.md](../STACK.md) Part I | `runs/n1_density/eval_test.json` -> `baselines` |
| ETAS | [neural_etas.py](../flowquake/neural_etas.py) 78–87, [precompute_trigger_features.py](../scripts/precompute_trigger_features.py) 143 | `runs/etas_sll_repro.json` |
| Flows | [flow.py](../flowquake/flow.py), [model.py](../flowquake/model.py) 233–235 | `tests/test_flow.py` |
| SSM | [ssm.py](../flowquake/ssm.py) 21–99 | `tests/test_ssm.py`, `runs/ablation_h/` |
| Relational features | [data.py](../flowquake/data.py) 154–165, [model.py](../flowquake/model.py) 32–35 | `configs/n1_density.yaml` |
| Kernel-mixture head | [heads.py](../flowquake/heads.py) 96–154 | `tests/test_heads.py` |
| Neural-ETAS head | [neural_etas.py](../flowquake/neural_etas.py) 59–96 | `runs/neural_etas/*/summary_full_s*.json` |
| Simulation / CSEP | [ntest.py](../flowquake/ntest.py), [csep_forecast.py](../flowquake/csep_forecast.py), [etas_csep.py](../flowquake/etas_csep.py) | `runs/*/csep*/csep_results.json` |
| Statistics | [stats.py](../flowquake/stats.py), [stats_hardening.py](../scripts/stats_hardening.py) | `runs/stats_hardening.json` |
| Claim provenance | — | [results/CLAIMS.md](../results/CLAIMS.md), [WORKING.md](../WORKING.md) |

---

## 11. Common misconceptions

1. **People think** FlowQuake models an intensity like ETAS. **Actually** it
   models the next-gap density `f(tau | H)` directly and never computes
   `integral lambda`. **Why it matters:** it explains why the temporal scores
   are exact, why forecasts require sequential simulation rather than grid
   evaluation, and why a standard time-rescaling residual plot is not directly
   available.

2. **People think** the Mamba encoder is the model. **Actually** it is disabled
   in every production run (`h_bottleneck: 0`) and exists to demonstrate a
   failure mode. **Why it matters:** the README's own framing invites this
   misread, and an examiner who spots it will assume the rest of the paper is
   equally loose.

3. **People think** "FlowQuake beats ETAS" is one result. **Actually** it is
   two models: the production TPP wins temporally and *loses* spatially by 0.37
   nats; the separate neural-ETAS head wins spatially; the headline is their
   composite. **Why it matters:** `runs/stats_hardening.json` contains
   `dTot_mean` twice with opposite signs, and quoting the wrong one inverts the
   headline.

4. **People think** a confidence interval crossing zero means the models tie.
   **Actually** it means there is no evidence of a difference, which is not the
   same thing. **Why it matters:** the project requires TOST for every "ties"
   claim, and Iran fails it while Greece passes — a distinction that would be
   invisible under the naive reading.

5. **People think** `p_boot: 0.0005` is a p-value. **Actually** it is the
   resolution floor of a 4,000-replicate add-one-smoothed bootstrap,
   `2/4001 = 0.00049988`. **Why it matters:** you cannot compare two such
   p-values, and reporting "p = 0.0005" as if it were measured is wrong.

6. **People think** `nll` includes the magnitude score. **Actually**
   `nll = -(tll + sll)` by the benchmark's convention. **Why it matters:** the
   magnitude head is still what makes the CSEP M-test pass, so "we don't score
   magnitude" and "magnitude doesn't matter" are different statements.

7. **People think** the CSEP pass rates exclude days to flatter the result.
   **Actually** they exclude days on which the test was not *computable* (NaN or
   pyCSEP's `(-1,-1)` sentinel), by a committed rule. **Why it matters:** the
   real defect is different and worse — two stored 10^4-run summaries use a
   stale denominator, and correcting them moves the numbers *up*.

8. **People think** the model is geography-free. **Actually** only the *learned*
   conditioning is; the smoothed-seismicity background map is fitted absolute
   geography, frozen on train-era events. **Why it matters:** it is the
   difference between a defensible claim and one that a single `grep` for
   `kde_log_grid` demolishes.

9. **People think** the head's spatial gain is the neural modulations.
   **Actually** the background-only ablation delivers +0.0513 of the +0.060, and
   the global-refit control delivers +0.0564. **Why it matters:** the neural
   part is the smallest component and should be reported as such.

10. **People think** "six regions, all significant" is a strong generalization
    claim. **Actually** four of the six use native temporal models and two use
    few-shot transfer, and the *temporal* family is significant in only three of
    six. **Why it matters:** the sentence has to carry those qualifiers or it is
    an overclaim.

---

## 12. The ten questions most likely to sink you

Ranked by (probability asked) × (damage if fumbled). For each: the strategy, and
whether a fully satisfying answer exists.

1. **"Isn't initializing from the ETAS inversion circular?"** (Q104) —
   *No fully satisfying answer.* Concede immediately that this bounds the claim
   to an ETAS *upgrade*, quote the `REPLACEMENT_READINESS.md` sentence, and
   pivot to the gate-closed 1.77e-9 verification, which is what makes the
   comparison honest rather than circular. Do not try to argue it is not a
   limitation.

2. **"Six-region win, but three regions aren't temporally significant."**
   (Q105) — *Fully answerable.* Two families, two answers. Have both p-vectors
   memorized. The failure mode is hesitating, which reads as being caught.

3. **"Why is Mamba in the title if it is off?"** (Q106) — *Concession.* Say the
   framing is wrong and should change; explain what the encoder is for (the
   §4.3 mechanism) and what actually supplies whole-catalog reach (relational
   features + 80 mixture components + the head's full-history sums).

4. **"How many times did you touch the test set?"** (Q107) — *Concession with a
   strong pivot.* Quote your own docstring ("do not describe these runs as a
   test-scored-once protocol"), then move to the 2020–2026 window and be the
   first to say it is retrospective.

5. **"Which `etas` package?"** (Q115) — *No answer exists in the repository.*
   State that plainly, bound the damage (California uses the benchmark's own
   published inversion and is unaffected), name the decisive file
   (`direct_url.json`), and say the pin will carry a SHA. Guessing here is
   fatal because a reviewer can check.

6. **"Can I reproduce your foreign-region baselines?"** (Q116) — *No.* Six ETAS
   configs exist only on the author's machine. Concede, say the fix is
   committing them to `configs/etas/`, and note it is item 10 of the
   working list — showing that you already knew.

7. **"Are your composite totals three-seed?"** (Q113) — *Concession with a
   recovery.* The committed artifacts are seed 0; the arithmetic (Italy
   0.2095 − 0.0712 = 0.1383) proves it. But the scripts were fixed in commit
   `2e8fa8a` to glob all seeds and emit a `single_seed_warning`, and the JSONs
   simply have not been regenerated. Say which number changes (Chile
   +0.061 → +0.064) and which do not.

8. **"Significant in every era?"** (Q114) — *The claim is wrong.* Say so
   without hedging: Chile is 10 of 19 windows positive and no per-era statistic
   exists. Replace it with the backed window fractions. A wrong claim you
   volunteer costs a fraction of a wrong claim that is extracted.

9. **"0.113 nats against ETAS's 6.0 over Poisson — 2% of the problem."**
   (Q109) — *Fully answerable but only if you have the number first.* Say 1.9%
   before they do, then make the sign-not-size argument.

10. **"Your production spatial head loses to ETAS."** (Q110) — *Fully
    answerable.* Two models. Have `-0.3691` and `+0.060` and their artifact
    paths ready. Fumbling this one makes every other number suspect, because it
    looks like you do not know which model produced your headline.

**The meta-strategy.** For 1, 3, 4, 5, 6, 7 and 8 the correct answer is a
concession. A defensible concession — "that is a real limitation; here is
exactly how far it reaches, and here is the narrower claim that survives" —
ends the line of questioning. A bluff invites three follow-ups and loses all of
them.

---

## 13. One-page cheat sheet

**Definitions to say verbatim**

```
lambda(t | H_t) = lim_{dt->0} P(event in [t,t+dt) | H_t)/dt        (H_t predictable, excludes t)
log L  = sum_i log lambda(t_i | H_{t_i}) - integral_0^T lambda(u | H_u) du
f(tau|H) = lambda(t_{i-1}+tau) exp(-integral_0^tau lambda(t_{i-1}+u) du)   [lambda = hazard]
Time rescaling: Lambda(t_i) - Lambda(t_{i-1}) ~ iid Exp(1) if the model is correct
GR:    P(M > m) = 10^{-b(m-m_c)},  m - m_c ~ Exp(beta),  beta = b ln 10   (b=1 -> beta=2.3026)
Aki:   b_hat = log10(e)/(mbar - (m_c - dm/2)) = 0.4342945/xbar,  SE ≈ b/sqrt(n)
M_w:   Hanks-Kanamori (2/3)log10(M_0) - 10.7  [dyn.cm]  =  (2/3)log10(M_0) - 6.033 [N.m]
       IASPEI          (2/3)(log10(M_0) - 9.1) [N.m]    =  (2/3)log10(M_0) - 6.067 [N.m]
Omori: n(t) = K(t+c)^{-p};  integral_0^T = K[c^{1-p} - (T+c)^{1-p}]/(p-1),  p != 1
ETAS w_j = k0 exp(a(m_j-m_c)) exp(-dt/tau_tap) (dt+c)^{-(1+omega)}
ETAS K_j = (r^2+d_j)^{-(1+rho)},  d_j = d exp(gamma(m_j-m_c)),  Z_j = pi/(rho d_j^rho)
Branching ratio n = k0 * I * beta/(beta-a),  I = integral exp(-u/tau_tap)(u+c)^{-(1+omega)} du
Mixture kernel f(r) = (q-1)/(pi d^2) (1+r^2/d^2)^{-q},  integrates to 1 for q > 1
CNF:   d/dt_flow log p = -div v  =>  log p_1(u) = log p_0(z(0)) - integral_0^1 div v dt_flow
FM path: z_t = (1-(1-sigma_min)t) z_0 + t u;  target  u - (1-sigma_min) z_0
Units: log f(tau) = log p(u) - log sigma_n - log tau,   u = (log tau - mu_n)/sigma_n
```

**Numbers, with their file**

| number | value | file / key |
|---|---|---|
| ETAS ComCat tll / sll / nll | 1.4343428 / −8.6897704 / 7.2554276 | `runs/fullsuite_summary.json` -> `ComCat_25.etas_*` |
| Poisson floor | 0.5126407 / −13.7745041 / 13.2618635 | `runs/n1_density/eval_test.json` -> `baselines.Poisson` |
| FQ 3-seed tll (ComCat) | 1.4868326 ± 0.0008107 | `fullsuite_summary.json` -> `ComCat_25.tll` |
| Production head sll (ComCat) | −9.0588652 (dS −0.3691 vs ETAS) | `fullsuite_summary.json`; `replacement_readiness.json` WARN |
| Neural head sll | −8.6297607, dS +0.060 [0.0509, 0.0692] | `replacement_readiness.json` -> `spatial_win_comcat` |
| Gate-closed ETAS repro | max abs err 1.7655797e-09 on 21,889 events | `runs/etas_sll_repro.json` |
| Composite ComCat total | nll 7.1421219 vs 7.2554276; dTot +0.1133 | `runs/total_win.json` -> `test_2007_2020` |
| Six-region totals | CA .1133, IT .2095, JP .0390, CL .0608, GR .0756, IR .0844 | `stats_hardening.json` -> `total_with_head_family` |
| Their Holm p | .003 .003 .0045 .003 .011 .0185 | same |
| Temporal-family Holm p | CA .003, IT .003, CL .036 sig; JP .274, IR .273, GR .648 not | `stats_hardening.json` -> `family_dT_holm` |
| Forward 2020–2026 | n 10,187; dT +.0574, dS +.0666, dTot +.1241 [.1035,.1455] | `total_win.json` -> `forward_2020_2026` |
| Memorization (ckpt_last) | h0 7.28/7.62/gap 0.34; h4 4.14/19.65/gap 15.50 | `runs/ablation_h/memorization_figure.json` |
| Best ckpt for every h>0 | step 250, the first ever evaluated | `runs/ablation_h/ablation_h.json` |
| CSEP, full-history head | N 95/100, S 79/85, M 90/92 | `runs/n1_density/csep_head/csep_results.json` |
| Paired S-test vs ETAS | 83 shared days, 77 passes each (72 both-pass, 1 both-fail, agreement 73), 10 discordant 5–5, McNemar p 1.0000 | `replacement_readiness.json` -> `full_history_head_csep` |
| Claim audit | 142 rows: 63 MATCH, 51 ROUNDING, 13 NO ARTIFACT, 11 MISMATCH, 2 AMBIGUOUS + 2 special (T23, X31); 134 distinct claims | `results/CLAIMS.md` |
| Readiness | `RESEARCH_PREVIEW_READY`, 15 checks, 11 PASS / 4 WARN | `runs/replacement_readiness.json` |

**Constants** — `TAU_FLOOR_DAYS 1e-7` · `RECENCY_LAGS (1,2,4,8,16,32,64)` ·
`TOKEN_DIM 32` · `LAST_K 64` · `BIG_M 16` · `BIG_MAG_MIN 4.5` ·
`BIG_WINDOW_DAYS 730` · `MIX_K 80` · `SAFE_TOKEN_DIMS` 30 dims (no absolute
x, y) · `cond_dim = 30 + h` · `KDE_BWS [1.5, 6, 25, 100] km` ·
`NEAR_W 256 + NEAR_P 128 = 384` · `kde_gate_init −2.94` (sigmoid = 0.0502) ·
`MAX_EVENTS_PER_DAY 200` · `MAX_REJECTION_ROUNDS 200` · block bootstrap
`mean_block 50`, `n_boot 2000` (CI) / `4000` (p, TOST), `seed 0` ·
p-value floor `2/4001 = 0.00049988`.

**Production config** (`configs/n1_density.yaml`) — `d_model 96` · `n_layers 4`
· `flow_hidden 96` · `flow_layers 3` · `mix_hidden 64` ·
`loss_weights [1.0, 1.0, 0.5]` · `sigma_min [0.02, 0.01, 0.05]` · `dropout 0.1`
· `input_noise 0.1` · `h_bottleneck 0` · `spatial_density_feat true` ·
`d_floor_km 0.1` · `window 2048` · `burn_in 256` · `batch 8` · `lr 3e-4` ·
`weight_decay 0.03` · `warmup 500` · `steps 20000` · `val_every 250` ·
`patience 16` · `seed 1555`.

**The one-line mental model** — *keep ETAS's power laws, replace its fixed
parameters with history-conditioned modulations, forbid the learned parts from
ever seeing an absolute coordinate, and prove normalization stays closed-form at
every step.*

---

## 14. Further reading

Ten sources, one line each on why. Where I am not certain of a detail I say so;
verify before citing precisely.

1. **Daley & Vere-Jones, *An Introduction to the Theory of Point Processes*
   (Springer, 2 vols).** The rigorous version of everything in Tier 1 — the
   conditional intensity, the Jacod likelihood, and the compensator. Read Vol. I
   ch. 7 for the likelihood and Vol. II for the martingale machinery.

2. **Ogata, Y. (1988), *Statistical models for earthquake occurrences and
   residual analysis for point processes*, JASA 83(401), 9–27.** The ETAS paper
   and the time-rescaling residual method in one. If you cite one thing, cite
   this.

3. **Ogata, Y. (1998), *Space–time point-process models for earthquake
   occurrences*, Ann. Inst. Statist. Math. 50(2), 379–402.** The spatial
   extension, i.e. where the `(r^2 + d)^{-(1+rho)}` kernel comes from.

4. **Zhuang, Ogata & Vere-Jones (2002), *Stochastic declustering of space-time
   earthquake occurrences*, JASA 97(458), 369–380.** The branching
   representation used by the EM inversion — read this before claiming to know
   how ETAS is fitted.

5. **Mizrahi, Nandan & Wiemer (2021), *Embracing data incompleteness for better
   earthquake forecasting*, JGR Solid Earth 126(12), e2021JB022379.** The
   methods paper for the `etas` package that produced every baseline in this
   repository.

6. **Stockman, Lawson & Werner (2026), *EarthquakeNPP*, TMLR, arXiv:2410.08226.**
   The benchmark. Its finding — no NPP beats ETAS, ETAS wins spatial LL against
   all of them — is the thing FlowQuake is arguing with.

7. **Savran et al. (2022), *pyCSEP*, SRL 93(5), 2858–2870**, with
   **Zechar, Gerstenberger & Rhoades (2010), BSSA 100(3), 1184–1195** for the
   likelihood-based N/S/M tests. Read the second to know what the tests mean,
   the first to know what the code does.

8. **Lipman, Chen, Ben-Hamu, Nickel & Le (2023), *Flow Matching for Generative
   Modeling*, ICLR**, with **Chen, Rubanova, Bettencourt & Duvenaud (2018),
   *Neural Ordinary Differential Equations*, NeurIPS** for the instantaneous
   change-of-variables. Together these are the entire temporal head.

9. **Dao & Gu (2024), *Transformers are SSMs* (Mamba-2), ICML**, with
   **Gu & Dao (2023), *Mamba*, arXiv:2312.00752.** The SSD decomposition
   implemented in [ssm.py](../flowquake/ssm.py); read the duality section
   specifically.

10. **Politis & Romano (1994), *The stationary bootstrap*, JASA**, and
    **Gneiting & Raftery (2007), *Strictly proper scoring rules, prediction,
    and estimation*, JASA 102(477), 359–378.** The two statistical papers this
    project's claims actually rest on: the first is why the CIs are what they
    are, the second is why a likelihood comparison is a legitimate forecast
    comparison.

*Also worth reading for context, from [NOVELTY.md](../NOVELTY.md)'s prior-art
sweep:* Dascher-Cousineau et al. (2023) RECAST, GRL 50(17), e2023GL103909 (the
closest neural-TPP prior work); and Shchur, Biloš & Günnemann (2020),
*Intensity-free learning of temporal point processes*, ICLR — the general
argument for modelling `f(tau | H)` instead of `lambda`, which is the design
decision FlowQuake inherits.
