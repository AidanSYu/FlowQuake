# FlowQuake: the whole argument, and every joint where it can be attacked

Chapters 1–7 built the machinery. This chapter assembles it into the actual
scientific claim FlowQuake makes, shows you where every load-bearing number
comes from, and then tries to break it. The last part is the point. A viva is
not a test of whether you can recite the abstract; it is a test of whether you
found the weak joints before the examiner did.

Everything numeric in this chapter was read from a file in this repository, and
the file is named at the point of use. Where the repository's own prose
(`MANUSCRIPT.md`, `STACK.md`, `README.md`) disagrees with the artifact, the
disagreement is stated plainly. **Six** such disagreements below are **not**
recorded in [results/CLAIMS.md](../results/CLAIMS.md) — §2.8 (STACK.md's magnitude
grid), §2.9 (two dead `sigma_min` entries), §3.1 (STACK.md's parameter count),
§5.4(a) (a stale headline artifact), §5.4(b) (a single temporal seed under both
Holm families), and §10.2, which concerns the headline spatial result.

---

## What this chapter buys you

- You can state FlowQuake's claim in three sentences and immediately name the
  three qualifiers that make each sentence true rather than overclaimed, and you
  can separate **claim** from **evidence** from **interpretation** for each.
- You can justify every architectural decision by naming the alternative that
  was not taken and the specific failure that ruled it out — not "we chose a
  flow" but "we model `f(tau|H)` rather than `lambda(t|H)` because X, at cost Y".
- You can explain, mechanically and information-theoretically, why widening the
  learned global channel destroys held-out likelihood, why 0.3-sigma noise on a
  4-dimensional bottleneck does not stop it, and why early stopping cannot
  rescue it.
- You can tell an examiner, unprompted, which reported numbers are backed by a
  committed artifact, which are backed only to rounding, which are contradicted,
  and which have no backing at all — with the tallies.
- You can defend the two-model composite, and concede precisely the part of it a
  referee is right to object to.
- You can rank the objections by how much damage they do, and name the
  experiment that would settle each.

## Prerequisites

Read these first. Chapter files follow the `NN-topic.md` convention in `docs/`.

| chapter | what you need from it |
|---|---|
| [Chapter 1 — point processes](01-point-processes.md) | `N(t)`, `H_t`, `lambda(t\|H_t)`, `Lambda(t)`, the hazard/survivor identities, the marked factorization and `tll`/`sll`/`mll`/`nll`, time rescaling |
| [Chapter 2 — seismology](02-seismology.md) | Gutenberg–Richter and `b`, Omori–Utsu, completeness `m_c`, magnitude scales and why mixing them is dangerous, the catalogs FlowQuake uses |
| [Chapter 3 — ETAS](03-etas.md) | `mu, k0, a, c, omega, tau_tap, d, gamma, rho`; Omori; Gutenberg–Richter |
| [Chapter 4 — density estimation and flow matching](04-flows-and-density-estimation.md) | change of variables, CNFs, rectified flow, `z`, `t_flow`, the step-count question, why the spatial and magnitude heads are *not* flows |
| [Chapter 5 — sequence models and SSMs](05-sequence-models-ssm.md) | the Mamba-style encoder, the chunked scan, and its §15 — "this encoder is off in every production run" |
| [Chapter 6 — evaluation and CSEP](06-evaluation-and-csep.md) | scoring rules, information gain, paired comparison, CSEP N/S/M, McNemar |
| [Chapter 7 — statistics for dependent data](07-statistics-dependent-data.md) | stationary block bootstrap, block-length choice, Holm, TOST, family definition, test-set hygiene |

[Chapter 9](09-viva-question-bank.md) is the question bank; §15 below is the
subset that is specific to this chapter's argument.

And [STACK.md](../STACK.md), which is the code walkthrough. **This chapter does not
repeat it.** Where you need "which function does this", STACK.md has it; this
chapter has "why, on what evidence, and where it breaks".

Notation reminder, used throughout: `tau` is the inter-event gap; the repo's
Omori taper timescale is *also* called `tau` in code and configs, and is written
`tau_tap` here to disambiguate. `t_flow` in `[0,1]` is the flow's integration
variable, never event time.

---

## 1. The claim, in three sentences

Here is the claim at the precision a referee will hold you to. Each sentence is
followed by its evidence and by the interpretation that is *not* licensed.

### Sentence 1 — temporal

> On all five California catalogs of the EarthquakeNPP benchmark, a
> flow-matching temporal head conditioned on relational, translation-invariant
> history features beats the benchmark's region-fitted ETAS on per-event
> temporal log-likelihood `tll`, in 3-seed means.

**Evidence.** [runs/fullsuite_summary.json](../runs/fullsuite_summary.json):

| catalog | m_c | FQ tll (3-seed) | ETAS tll | Δ |
|---|---|---|---|---|
| ComCat_25 | 2.5 | 1.486832578976949 | 1.4343428344882627 | +0.05249 |
| WHITE_06 | 0.6 | 2.0668934186299643 | 2.0210970061274423 | +0.04580 |
| SanJac_10 | 1.0 | 1.1609567801157634 | 1.1325267069430716 | +0.02843 |
| SaltonSea_10 | 1.0 | 2.433719793955485 | 2.332039202380453 | +0.10168 |
| SCEDC_20 | 2.0 | 2.619408051172892 | 2.5409825345527426 | +0.07843 |

**Qualifier that makes it true.** Under the stationary block bootstrap, *four* of
five are significant; SanJac_10's interval is
`[-0.005686476386749143, 0.07592596149130082]` with stored `decision: "tie"`
([runs/replacement_readiness.json](../runs/replacement_readiness.json) →
`checks[california_block_bootstrap_temporal]`). `MANUSCRIPT.md:307` says the
interval "touches zero"; it *crosses* zero. Say "four of five, one tie".

**Interpretation not licensed.** This is not a claim about total likelihood. The
same three seeds give `sll` −9.0589 against ETAS's −8.6898 and `nll` 7.5720
against ETAS's 7.2554 — the production model **loses** overall on ComCat_25, and
on all five catalogs.

### Sentence 2 — spatial and total

> A separate, full-history spatial head that is a strict learnable superset of
> ETAS's spatial density — initialized from each region's own ETAS inversion —
> beats that inversion spatially in six regions, and combining its `sll` with
> the production model's `tll` flips **total** likelihood in all six.

**Evidence.** [runs/stats_hardening.json](../runs/stats_hardening.json) →
`total_with_head_family`; six regions, ΔTotal `+0.1133, +0.2095, +0.0390,
+0.0608, +0.0756, +0.0844` nats/event, all Holm-adjusted `p <= 0.0185`. On
ComCat_25 that is `nll` 7.142121886887271 vs ETAS 7.255427552750566
([runs/total_win.json](../runs/total_win.json)).

**Qualifiers that make it true.** (a) The head is *initialized from the target
region's ETAS inversion* and consumes precomputed ETAS triggering sums, so this
upgrades an ETAS deployment rather than replacing one. (b) The composite mixes
two separately trained models. (c) Greece and Iran use `temporal_variant:
"fewshot"`, not native training. (d) Japan's `+0.0390` carries the artifact's own
`dTot_abs_below_0.05: true` flag. (e) Pairing coverage outside
California/Italy is 89.0–97.1%, not 100%.

**Interpretation not licensed.** "FlowQuake beats ETAS" as a single-model
statement. No single trained model in this repository beats ETAS on total
likelihood.

### Sentence 3 — mechanism

> Exposing the output heads to a learned whole-catalog embedding causes
> catastrophic memorization — train `nll` collapses to 4.14 while held-out `nll`
> explodes to 19.65 — and the cure is structural exclusion of absolute
> coordinates from the learned conditioning, not regularization or early
> stopping.

**Evidence.** [runs/ablation_h/memorization_figure.json](../runs/ablation_h/memorization_figure.json),
`ckpt == "last"` rows; and every `h > 0` run's best held-out checkpoint is step
250, the first evaluated ([runs/ablation_h/ablation_h.json](../runs/ablation_h/ablation_h.json)).

**Qualifier.** [NOVELTY.md](../NOVELTY.md) records this sub-claim as **"Unclaimed
but UNCONFIRMED — no source either way"**. Frame it as a diagnostic
contribution, never as a "first".

**Interpretation not licensed.** That h>0 is *intrinsically* unlearnable. What is
shown is that *this* encoder, at *this* scale, with *this* fixed-sigma noise
regularizer and *this* data volume, memorizes. See §4.4.

---

## 2. The design as a chain of decisions

Every one of these is a place an examiner can ask "why not the obvious thing?"
Each row gives the decision, the alternative not taken, and the reason.

### 2.1 Model `f(tau | H)` rather than `lambda(t | H_t)`

**Not taken:** the standard neural-TPP construction, parameterizing the
conditional intensity and getting the likelihood from
`log L = sum_i log lambda(t_i | H_{t_i}) − integral_0^T lambda(u | H_u) du`.

**Reason.** The compensator integral has no closed form for a general neural
`lambda`, so you either restrict `lambda` to an integrable family (RMTPP's
exponential decay, Hawkes kernels) and lose flexibility, or Monte-Carlo the
integral and lose exactness. Modelling `f(tau|H)` sidesteps it: the survivor is
implied, not integrated. Nothing is lost in principle — the two are equivalent by

```
f(tau | H)              = lambda(t_prev+tau | H) * exp(-integral_0^tau lambda(t_prev+u | H) du)
lambda(t_prev + tau |H) = f(tau | H) / S(tau | H)
```

**Cost paid.** No `lambda` for free: simulation must sample `tau` from the flow
rather than thin a Poisson process, and any instantaneous-rate statement needs
`f/(1−F)`, which the repo does not compute. And the exact log-density needs the
instantaneous change-of-variables ODE along `t_flow`, 64–96 steps per event —
which is why `tll` is the only score requiring an ODE.

### 2.2 A flow for time, closed-form structured heads for space and magnitude

**Not taken:** a flexible generative head per mark (a 2-D flow or diffusion model
for `(x, y)`).

**Reason, empirical.** A head free to place density anywhere places it on the
training epicentres. At `h=4, ckpt_last` the *train* `sll` is −7.2697 (vs −8.8574
at `h=0`) while the *held-out* `sll` is **−13.4651**, against a
homogeneous-Poisson `sll` of **−13.7745**
([runs/n1_density/eval_test.json](../runs/n1_density/eval_test.json) →
`baselines.Poisson.sll`). A memorized spatial head retains 0.31 nats/event over
*uniform*. The temporal axis behaves oppositely: `log tau` has no
absolute-position analogue to memorize, so flexibility pays there.

**Reason, structural.** In a CSEP run the spatial density is normalized over the
region at every forecast time for every simulated catalog. A closed-form mixture
costs one `logsumexp`; a flow costs an ODE per query point. `sll`/`mll` are exact
by construction, no quadrature anywhere.

### 2.3 Relational, translation-invariant conditioning

Heads condition on `SAFE_TOKEN_DIMS = [0, 3] + range(4, 32)` — `log tau`,
magnitude, and seven lag blocks of `(log(t_i − t_{i−k}), x_i − x_{i−k},
y_i − y_{i−k}, m_{i−k})` for `k in {1,2,4,8,16,32,64}`. **Absolute `x`, `y` (dims
1 and 2) are excluded** ([flowquake/model.py:32-35](../flowquake/model.py#L32-L35)).

**Not taken:** feed absolute coordinates and regularize (dropout, weight decay,
noise). Exclusion makes memorization *structurally impossible* rather than
penalized. The lag block is not arbitrary featurization: `log(t_i − t_{i−k})` is
the Omori argument at seven timescales, `(dx, dy)` the spatial-kernel argument,
`m_{i−k}` the productivity argument — an ETAS kernel's inputs, precomputed at
exponentially spaced lags, the cheap-receptive-field trick of a dilated
convolution. It is also what makes the model portable: translate the catalog
500 km and every conditioning feature is unchanged.

**Honest boundary.** The *background map* is absolute geography, fitted on
train-era data ([flowquake/data.py:255-273](../flowquake/data.py#L255-L273)). The
claim is about the **learned conditioning**, not about the model containing no
geography. Get this distinction wrong in a viva and you have overclaimed.

### 2.4 Observation-anchored mixture components

Every spatial mixture component sits at an *observed prior event location
supplied at evaluation time*; no component location is a weight.

**Not taken:** a learned mixture of Gaussians over the region (the DeepSTPP-style
construction). A learned component centre *is* a memorized coordinate. The MLP
here maps `[cond, log Δt_j, m_j, log dist_j]` to
`(logit, d, q, rho, cos θ, sin θ)` — it learns *how to shape a triggering kernel
given a parent's age, size and offset*, not *where earthquakes happen*.

### 2.5 Three tiers of trigger candidates

`MIX_K = 80` components: 64 by recency (`i, i−1, …, i−63`), 16 "big triggers"
(largest `m >= 4.5` in the trailing 730 days), plus an optional spatial-nearest
tier via KD-tree over all history
([flowquake/data.py:49-117](../flowquake/data.py#L49-L117)).

**Not taken:** ETAS's full sum over all ~70,000 priors (`92,263 − 21,889 = 70,374`
events precede `test_start`; the *training* window itself is 55,442 —
[Ch. 5 §3.2](05-sequence-models-ssm.md#32-cost)). The code's own comment
gives the reason: the last-64 window spans ~70 days at the catalog's mean rate,
but large mainshocks keep triggering for years, so recency alone is blind to a
M6 from 400 days ago. The three tiers are three answers to "which parents
plausibly matter": *recent*, *large*, *co-located*.

**The residual this leaves.** The manuscript's diagnosis is triggering
*coverage* — recurrence at locations older than any fixed context window. Be
careful quoting the support: the "64% recur within 0.5 km / 85% outside the
last-64 window" figures at `MANUSCRIPT.md:507-508` are **NO ARTIFACT (N8)**,
printed by `scripts/trigger_coverage.py` and written nowhere, and the script's
printed statistic is for a **<5 km** neighbour, not <0.5 km.

### 2.6 ETAS-shaped power-law kernels, not Gaussians

Component density `f(r) = (q − 1)/(pi d^2) * (1 + r^2/d^2)^(−q)`
([flowquake/heads.py:104-105](../flowquake/heads.py#L104-L105)), with
`q >= q_floor = 1.15` and `d >= d_floor_km`. **Not taken:** a Gaussian mixture.

Normalization, by `u = r^2/d^2`:

```
integral_0^inf (1 + r^2/d^2)^(-q) 2*pi*r dr = pi d^2 integral_0^inf (1+u)^(-q) du
                                            = pi d^2 / (q-1)      for q > 1
```

so the prefactor is exactly right and `q_floor = 1.15` keeps `q − 1 >= 0.15 > 0`.

The tail is the reason. Aftershock distances are power-law distributed. Under a
Gaussian of scale `sigma`, an event at `r = 10 sigma` costs
`r^2/(2 sigma^2) = 50` nats; under this kernel it costs
`q log(1 + 100) ~ 1.8 * 4.6 = 8.3` nats. Mean per-event log-likelihood is
dominated by exactly those tail events, which is why ETAS uses
`(r^2 + d_m)^(−(1+rho))` and not a Gaussian either.

**Reading trap.** The two models parameterize this differently. The production
head uses `(1 + r^2/d^2)^(−q)` with `d` in km. The neural-ETAS head uses
`(r^2 + d_m)^(−(1+rho))` with `d_m = d * exp(gamma (m − m_c))` — here `d` has
units of **km²**, because it is added to `r^2`
([flowquake/neural_etas.py:82-85](../flowquake/neural_etas.py#L82-L85)), and its
normalizer is `Z = pi / (rho * d_m^rho)` (verified in §12.4).

### 2.7 Area-preserving anisotropy

Each component gets `(rho, theta)`, giving elliptical axes `d*rho` and `d/rho`
rotated by `theta` ([flowquake/heads.py:88-94](../flowquake/heads.py#L88-L94)).

**Not taken:** independent semi-axes `(a, b)`.

**The derivation, because it is asserted everywhere in this primer and worked
nowhere else.** The isotropic component is

```
f_iso(s) = (q-1)/(pi d^2) * ( 1 + |s|^2/d^2 )^{-q},     integral = 1  for q > 1
```

To make it elliptical, replace the argument `|s|^2/d^2` by a quadratic form.
Write `R_theta` for the rotation by `theta` and

```
A  =  R_theta · diag( d*rho,  d/rho )            so an ellipse with semi-axes d*rho, d/rho
Q(s)  =  |A^{-1} s|^2      (the "squared radius" in ellipse units)
f_ell(s)  =  (q-1)/(pi d^2) * ( 1 + Q(s) )^{-q}
```

Now integrate by the change of variables `s = A w`, `ds = |det A| dw`:

```
integral f_ell(s) ds  =  (q-1)/(pi d^2) * |det A| * integral (1 + |w|^2)^{-q} dw
                      =  (q-1)/(pi d^2) * |det A| * pi/(q-1)
                      =  |det A| / d^2
```

using `integral_{R^2}(1+|w|^2)^{-q} dw = 2 pi integral_0^inf (1+u^2)^{-q} u du =
pi/(q-1)` (substitute `v = u^2`). So the elliptical component integrates to 1
**iff `|det A| = d^2`**. And

```
det A  =  det R_theta · (d*rho)·(d/rho)  =  1 · d^2  =  d^2      independent of rho, theta
```

∎ — that is the whole content of "area-preserving". The rotation contributes
`det R = 1`, and the two scalings' `rho` factors cancel, so the isotropic
prefactor `(q-1)/(pi d^2)` is still the right normalizer and **elongation along
fault strike costs no Jacobian term**. With free semi-axes `(a, b)` you would get
`|det A| = ab != d^2` and would have to carry `d^2/(ab)` — equivalently
`sqrt(det M)` with `M = A A^T` — through every normalizer, every `logsumexp` and
every sampler, and get it right in all three.

Note what is *not* free: area preservation constrains the ellipse to trade width
for length at fixed area, so the head cannot say "this parent's aftershock zone
is both longer and wider". That is a genuine modelling restriction, not just a
convenience, and it is the honest answer if an examiner asks what the choice
costs.

`tests/test_heads.py` numerically integrates a forcibly elongated, rotated
component over a grid and confirms unit mass — which is exactly the `|det A| = d^2`
identity checked numerically.

### 2.8 Conditional Gutenberg–Richter with a half-bin shift

`m − m_c ~ Exponential(beta(cond))`, so `log f(m) = log beta − beta * dm` with
`dm = clamp(m − m_c, 0) + 0.005`
([flowquake/heads.py:170-175](../flowquake/heads.py#L170-L175)).

**Not taken:** ETAS's fixed `b` (equivalently fixed `beta = b ln 10`), and no
shift. `b` varies with stress state, depth, and position in a sequence; the
manuscript credits the history-dependent `beta` with restoring the CSEP
**M**-test, which the benchmark's other generative NPPs fail.

The shift, derived — and be precise about the binning convention, because the
answer depends on it. Magnitudes are reported on a grid of width `Δ`. Under the
*dequantization* convention the config names — the reported `m` is the **lower
edge** of a `Δ`-wide bin, i.e. the true magnitude is uniform on `[m, m+Δ)` — the
likelihood of the reported value, expressed as a density on the dequantized
scale, is `(F(m+Δ) − F(m)) / Δ`. Writing `dm = m − m_c`, for the exponential

```
(F(m+Delta) - F(m)) / Delta = exp(-beta*dm) * (1 - exp(-beta*Delta)) / Delta

log(...) = log(beta) - beta*dm - beta*Delta/2 + (beta*Delta)^2/24 + ...
         = log(beta) - beta*(dm + Delta/2) + O((beta*Delta)^2)
```

which is exactly what the code computes with `dm + 0.005`. So `+0.005` implies
`Δ = 0.01`, matching `mag_dequant: 0.01` ("uniform dequantization over the raw
0.01-mag grid", [configs/comcat25.yaml](../configs/comcat25.yaml)). At `beta ~ 2`
the correction is `beta*Δ/2 = 0.01` nats.

**The nearest-rounding branch, and a correction this chapter used to get wrong.**
Suppose the catalog rounds to the *nearest* `Δ` instead, so bins are centred on
the reported value. Evaluating `(F(m + Δ/2) − F(m − Δ/2))/Δ` with the exponential
anchored at `m_c` gives `beta*exp(−beta*dm)*(1 + (beta*Δ)²/24 + …)` — the
first-order term vanishes and the shift looks like **zero**. That algebra is
right and the conclusion is wrong, because it forgets that the *threshold* moves
too. The catalog is cut on the **reported** magnitude (`magnitude >= mcut`,
[flowquake/data.py:201](../flowquake/data.py#L201), with no rounding), so under
nearest-rounding the retained set is `{true M ≥ m_c − Δ/2}` and the exponential
must be anchored there. Redo it with the truncation at `m_c − Δ/2` and
`P(reported = m) = exp(−beta(m − m_c))·(1 − exp(−beta*Δ))`, giving
`log P(m) − log Δ = log beta − beta[(m − m_c) + Δ/2] + (beta*Δ)²/24`. **The
correct shift is `+Δ/2` under both conventions** — which is the classical
Utsu (1966) / Bender (1983) correction, derived in full at
[Ch. 2 §5.4](02-seismology.md#54-the-binning-correction-utsu--bender). Use
Chapter 2's version; it is the standard result and it is the one that always
*lowers* fitted `b`.

So the shift's *form* is right in both readings and the only live question is
the value of `Δ`. Two things bound the damage either way: `+0.005` is a constant
offset in `dm`, not a shape change, and the benchmark's headline score is
`nll = −(tll + sll)` ([flowquake/train.py:76](../flowquake/train.py#L76)), which
**excludes `mll` entirely** — so no reported total-likelihood number moves. The
shift reaches the headline only indirectly, through the `0.5`-weighted `mll` term
in the training loss, and through simulated magnitudes and hence the CSEP M-test
([Ch. 2 §5.5](02-seismology.md#55-the-0005-in-headspy--what-it-corrects-and-the-doc-bug)).

> **STACK.md is wrong here, and so is any defence of `0.005` as universally
> correct.** [STACK.md:726-728](../STACK.md) calls `+0.005` "a half-bin shift for
> the catalog's **0.1**-magnitude discretization"; half of 0.1 is 0.05, so the
> walkthrough is internally inconsistent with the code. `0.005 = Δ/2` implies
> `Δ = 0.01`, matching `mag_dequant: 0.01`.
> **But do not stop there** — `0.005` is a *hardcoded literal* at
> [flowquake/heads.py:173](../flowquake/heads.py#L173), one value shared by all
> eleven catalogs with no per-catalog plumbing, and the ISC/INGV catalogs are
> certainly on a 0.1 grid (the ETAS side assumes it: `round_half_up(x, delta=0.1)`
> at [scripts/precompute_trigger_features.py:36](../scripts/precompute_trigger_features.py#L36)).
> So **no assignment of ComCat's precision makes a single hardcoded `0.005`
> correct everywhere**: either ComCat is also on 0.1 and every catalog is
> mis-shifted 10×, or ComCat is on 0.01 and the five agency catalogs are. That is
> the version of the claim that cannot be overturned by whatever `reference/`
> turns out to contain, and it is
> [Ch. 2 §5.5](02-seismology.md#55-the-0005-in-headspy--what-it-corrects-and-the-doc-bug)'s
> point 2 — make it the claim you offer in a viva. On a 0.1 grid the
> mis-shift biases fitted `beta` **11.6% high** and inflates `mll` by ~0.104
> nats/event.
> Separately, `mag_dequant` is a dead config key:
> [flowquake/model.py:61](../flowquake/model.py#L61) marks it "kept for config
> compat (GR head absorbs it)" and nothing consumes it — so its agreement with
> `0.005` is a coincidence, not a wiring.

### 2.9 The `sigma_min` bandwidth floors

`sigma_min: [0.02, 0.01, 0.05]`, described in the config as "KDE-style bandwidth
floors (normalized units) per head [time, space, mag]: prevent density collapse
onto (discretized) training targets".

**Reason, derived.** Event times are recorded at finite precision, so the
empirical distribution of `log tau` is atomic. If the model family contains
arbitrarily narrow continuous densities the empirical log-likelihood is
**unbounded above** — put a spike of width `eps` on each atom and the log-density
grows like `−log eps → +infinity`. This is the classic non-existence of the MLE
for a KDE bandwidth: a genuine degeneracy of the objective, not a numerical
nuisance. The flow implements the floor by changing the probability path
([flowquake/flow.py:41-76](../flowquake/flow.py#L41-L76)):

```
z_{t_flow} = (1 - (1 - sigma_min) * t_flow) * z0  +  t_flow * u
```

At `t_flow = 1` the conditional path has residual sd `sigma_min` rather than 0,
so the modelled density is the data convolved with `N(0, sigma_min^2 I)`. This is
*conservative in the sense that matters for the claim*: the floor shrinks the
model family, so it cannot manufacture held-out likelihood that the unfloored
family could not also attain — against a true density sharper than `sigma_min`
it costs. (Empirically it *helps*, because the unfloored family's empirical
optimum is a degenerate spike; that is a bias–variance benefit, not an inflation
of the attainable ceiling.)

> **A second discrepancy, in the repo's own config.** Only `sigma_min[0]` is ever
> used — [flowquake/model.py:86](../flowquake/model.py#L86) passes it to the flow,
> and `grep -n "sigma_min\[" flowquake/model.py` returns exactly that one line.
> `sigma_min[1]` (space) and `sigma_min[2]` (magnitude) are **never consumed**;
> the config comment describing three per-head floors is inaccurate. The
> equivalent guards exist by other means: `d_floor_km` / `q_floor` softplus
> floors ([flowquake/heads.py:82-83](../flowquake/heads.py#L82-L83)) and the
> `+0.005` half-bin shift.

### 2.10 A train-era smoothed-seismicity background

**Not taken:** ETAS's uniform `mu`. Instead, a fitted smoothed-seismicity map —
histogram train-era epicentres on 2 km bins, Gaussian-blur, mix 98% map with 2%
uniform; or with `adaptive_bg: true`, a Helmstetter-style variable-bandwidth
estimator ([flowquake/data.py:120-151](../flowquake/data.py#L120-L151)) in which
each event is smoothed by its distance to its 6th nearest neighbour, so dense
fault traces stay sharp and isolated events spread wide. The fixed-bandwidth
failure mode this fixes is off-fault "holes" that genuine background events fall
into and are charged for.

The benchmark's ETAS has a *uniform* background, which is a known weakness of the
incumbent; closing it is legitimate — but see §10.2, because it is also where
most of the spatial win comes from.

The neural-ETAS head's background is a different object built on the same idea:
a learnable softmax mixture over **four** causal Gaussian smoothed-seismicity
maps at bandwidths 1.5 / 6 / 25 / 100 km (`KDE_BWS` in
[scripts/precompute_trigger_features.py:31](../scripts/precompute_trigger_features.py#L31)),
plus a uniform component, gated by `kde_gate`. The manuscript's interpretability
claim — "the 1.5 km map dominant" — is a statement about that softmax.

---

## 3. The two-model structure, in full

This is the single most attackable feature of the paper, and you must be able to
narrate it without flinching.

### 3.1 What the two models are

| | production TPP | neural-ETAS spatial head |
|---|---|---|
| files | [flowquake/model.py](../flowquake/model.py), `flow.py`, `heads.py`, `data.py` | [flowquake/neural_etas.py](../flowquake/neural_etas.py), `scripts/train_neural_etas.py` |
| params | **≈2.95e4** at `h_bottleneck: 0` (see below) | **1,258** at `n_kde = 4` |
| trained on | GPU, ~45 min (`README.md:156`, RTX 4090), random 2048-event crops | CPU, minutes, full-batch Adam |
| produces | `tll`, `sll`, `mll` | `sll` only |
| history | 80 selected trigger candidates + lag features | **all** priors: frozen far-field ETAS sums + live near set of <=384 |
| initialization | zero-init flow, ETAS-plausible kernel init | **the target region's published ETAS inversion** |

The composite reported as "FlowQuake total likelihood" is
`tll` from column 1 plus `sll` from column 2, paired per event against the same
region's ETAS.

> **A third discrepancy, and one you should raise before an examiner does.**
> [STACK.md:61](../STACK.md) gives the production model as "~0.1–1 M" parameters.
> That count is only reachable if the SSM encoder is built, and at
> `h_bottleneck: 0` — every production config — it is not
> ([flowquake/model.py:76-84](../flowquake/model.py#L76-L84): the `else` branch sets
> `self.encoder = None`). What is actually constructed, at `cond_dim = 30`,
> `flow_hidden = 96`, `flow_layers = 3`, `mix_hidden = 64`:
>
> | module | params |
> |---|---|
> | `CondFlow` (`Linear(40,96)`, `Linear(96,96)`×2, `Linear(96,1)`) | 22,657 |
> | `KernelMixtureHead` (`h_proj` 1,984 + `comp_mlp` 4,806 + `bg_logit` 62) | 6,852 |
> | `GRMagnitudeHead` (`Linear(30,1)`) | 31 |
> | **total** | **29,540** |
>
> (`comp_feat_dim = 4` because `spatial_density_feat: true` in
> [runs/n1_density/config.yaml](../runs/n1_density/config.yaml); the per-component
> MLP is *shared* across all 80 mixture components, which is why `MIX_K` does not
> enter the count.) So the production model is a **30-thousand**-parameter model,
> not a million-parameter one — which strengthens rather than weakens the paper,
> and makes §14's "the Mamba encoder is what makes it work" misconception
> concrete. Likewise the head: `kde_gate` 1 + `kde_logits` 4 + `log_mu_adj` 1 +
> `log_alpha` 1 + a `2→32→32→3` MLP (1,251) = **1,258**, not "~2 k"
> ([flowquake/neural_etas.py:31-58](../flowquake/neural_etas.py#L31-L58)).

### 3.2 Why the totals are mixed this way

Because the production model's own spatial head loses to ETAS on every catalog.
Read it straight off [runs/fullsuite_summary.json](../runs/fullsuite_summary.json):

| catalog | FQ `sll` (production, 3-seed) | ETAS `sll` | ΔS | FQ `nll` | ETAS `nll` |
|---|---|---|---|---|---|
| ComCat_25 | −9.058865229288736 | −8.689770387238827 | **−0.3691** | 7.5720 | 7.2554 |
| WHITE_06 | −4.725900491078694 | −4.2610686365574395 | **−0.4648** | 2.6590 | 2.2400 |
| SanJac_10 | −5.923290252685547 | −5.398118234811221 | **−0.5252** | 4.7623 | 4.2656 |
| SaltonSea_10 | −2.637502113978068 | −2.3150835316085487 | **−0.3224** | 0.2038 | −0.0170 |
| SCEDC_20 | −7.848306496938069 | −7.534222208042888 | **−0.3141** | 5.2289 | 4.9932 |

Five for five losses, 0.31–0.53 nats/event. And the corresponding *total* for the
production model, from [runs/stats_hardening.json](../runs/stats_hardening.json) →
`per_region`, is a loss in every region: California `dTot_mean` **−0.3107**
`"loss"`, Italy +0.0041 `"tie"`, Japan −0.5382, Chile −0.2305, Greece −0.1936,
Iran −0.4297.

> **Key collision to memorize.** `stats_hardening.json` contains
> `per_region.California.dTot_mean = -0.3107` (`"loss"`, production model) and
> `total_with_head_family.California.dTot_mean = +0.1133` (`"win"`, composite).
> Same key name, opposite verdict. A checker who opens the wrong block concludes
> the headline is inverted. [WORKING.md](../WORKING.md) lists "do not cite
> `per_region...dTot_mean` as the headline" in its "Do not do" section.

### 3.3 What is legitimate about the composite

- **The decomposition is exact and the metric is additive.** With the standard
  mark factorization `f(tau, s, m | H) = f_t(tau|H) f_s(s|tau,H) f_m(m|tau,s,H)`,
  the joint log-density is `tll + sll + mll` and the benchmark's headline
  `nll = −(tll + sll)`. Adding a temporal score from one model to a spatial score
  from another yields the log-density of a **well-defined product model**: use
  model A's `f_t` and model B's `f_s`. That composite is a legitimate probability
  model — it normalizes, it can be simulated from, and it can be CSEP-tested.
  It is not a metric hack.
- **It was CSEP-tested as one object.** `flowquake/csep_forecast_head.py` runs
  exactly this composite through pyCSEP on the same 100 forecast days at a
  matched 1e3-catalog budget: N 95/100, S 79/85, M 90/92
  ([runs/n1_density/csep_head/csep_results.json](../runs/n1_density/csep_head/csep_results.json)).
  So the mixed model is not a paper-only artifact.
- **The pairing is per-event and honest.** `scripts/stats_hardening.py` merges
  three per-event tables (FlowQuake temporal, head spatial, ETAS) on
  `index_from_zero` or `(time, duplicate-rank)`, and reports coverage.

### 3.4 What a referee will object to, correctly

1. **You never trained the thing you report.** The two heads were optimized
   separately, on the same data, with separate early stopping. A jointly trained
   model with the neural-ETAS spatial head has not been fitted. The composite's
   `tll` and `sll` were each selected to be good; nothing checked that they are
   jointly calibrated.
2. **Model selection was per-component.** The temporal head's early stopping
   criterion is `nll = −(tll + sll)` using the **production** spatial head
   ([flowquake/train.py](../flowquake/train.py)) — i.e. selected against a spatial
   head that is not the one reported. The head's early stopping is on val `sll`.
   Neither selection targeted the reported composite.
3. **Two models, two initialization stories.** The temporal head is trained from
   scratch. The spatial head starts *at ETAS*. So the composite's spatial
   advantage is measured against the very object it was initialized from.
4. **It obscures the honest single-model result**, which is that the production
   model loses on total likelihood in five of five California catalogs and in
   five of six cross-regime regions.

**The best honest answer.** Say it in this order: (a) the composite is a valid
product model, stated as such, and CSEP-tested as such; (b) the paper's own
`REPLACEMENT_READINESS.md` lists "the production kernel-mixture spatial head
still trails ETAS" as the first entry under "Holes That Still Matter"; (c) the
right next experiment is a *single* model with the flow temporal head and the
neural-ETAS spatial head trained jointly, selected on the composite `nll`, and
reported end to end. That experiment is not in the repository.

---

## 4. The memorization result, in depth

### 4.1 The numbers, exactly, with the checkpoint each row is

From [runs/ablation_h/memorization_figure.json](../runs/ablation_h/memorization_figure.json),
which stores both `best` and `last` for each `h`:

| h | ckpt | step | train nll | test nll | gap | train sll | test sll | test tll |
|---|---|---|---|---|---|---|---|---|
| 0 | best | 7750 | 7.300671696662903 | 7.604310989379883 | 0.30363929271698 | −8.87424087524414 | −9.092926025390625 | 1.4886150360107422 |
| 0 | **last** | 11750 | 7.281167030334473 | 7.621030569076538 | **0.33986353874206543** | −8.857418060302734 | −9.106239318847656 | 1.4852087497711182 |
| 4 | best | **250** | 8.006531953811646 | 8.21579110622406 | 0.20925915241241455 | −9.40820598602295 | −9.508623123168945 | 1.2928320169448853 |
| 4 | **last** | 4250 | 4.143446922302246 | **19.64580488204956** | **15.502357959747314** | −7.2697343826293945 | −13.46508502960205 | **−6.18071985244751** |
| 16 | best | **250** | 7.8804404735565186 | 8.124865412712097 | 0.2444249391555786 | −9.264986038208008 | −9.417007446289062 | 1.2921420335769653 |
| 16 | **last** | 4250 | 4.182443857192993 | 18.731383323669434 | 14.54893946647644 | −7.24488639831543 | −13.684921264648438 | −5.046462059020996 |
| 64 | best | **250** | 7.789060831069946 | 8.063421964645386 | 0.27436113357543945 | −9.17601203918457 | −9.355245590209961 | 1.2918236255645752 |
| 64 | **last** | 4250 | 4.2729010581970215 | 18.33090305328369 | 14.05800199508667 | −7.310171604156494 | −13.037601470947266 | −5.293301582336426 |

> **Sibling-artifact trap.** [runs/ablation_h/ablation_h.json](../runs/ablation_h/ablation_h.json)
> is the file whose name invites you to open it for the manuscript's §4.3 table,
> and it stores
> only the **best**-checkpoint rows (h=4: train 8.0065, test 8.2158, gap 0.2093,
> step 250). Those are not the manuscript's numbers. The manuscript is correct
> because it says "at the converged checkpoint (`ckpt_last`)" and cites
> `memorization_figure.json`, but the two files invite a misread.

Three derived facts worth having ready:

- `h=4` at `ckpt_last` has held-out `nll` 19.646 against a homogeneous-Poisson
  baseline of 13.2619 — **6.38 nats/event worse than assuming nothing**.
- Held-out temporal: `tll` = −6.18 versus Poisson's +0.5126 — 6.69 nats/event
  worse than a constant rate.
- Held-out spatial: `sll` = −13.465 versus a uniform-region Poisson's −13.7745.
  The memorized spatial head retains **0.31 nats** of skill over uniform, against
  `h=0`'s 4.67.

`exp(19.6458 − 7.6210) = 1.67e5`: the memorized model assigns, per event, a
joint density about 170,000 times smaller than the structural model's.

### 4.2 What "fingerprinting a catalog position-era" means mechanically

1. The SSM encoder consumes the **full 32-dim token**, dims 1 and 2 included
   ([flowquake/model.py:163-170](../flowquake/model.py#L163-L170)) — absolute `x, y`
   enter the encoder even at `h > 0`; only the *heads* are protected by
   `SAFE_TOKEN_DIMS`.
2. The SSM state `s_i` is a deterministic function of the entire prefix. Because
   tokens carry absolute coordinates, `s_i` is a continuous, near-injective
   summary of *which stretch of this particular catalog you are standing in*.
3. `h_i = h_proj(s_i)` compresses it to `h` dims; Gaussian noise `+ 0.3*eps` is
   added at **train time only** ([flowquake/model.py:169-170](../flowquake/model.py#L169-L170)).
4. `h_i` joins the 30 relational dims in `cond`. The spatial head's mixture
   weights and kernel shapes are functions of `cond`, so given a reliable index
   into "where am I in the catalog" it can put near-all its weight on the one or
   two components sitting on the epicentre that occurred *there*, and shrink `d`
   toward `d_floor`.
5. That is a learned lookup from a position-code to an answer. On the training
   era the lookup is correct (`train sll` −7.27); on the test era the code
   indexes nothing — the state trajectory has left the memorized manifold — and
   confident, narrow kernels in the wrong place are what a log-density punishes
   hardest (`test sll` −13.47).

The repo's own operational definition of the fingerprint is in
[scripts/probe_fingerprint_claim.py](../scripts/probe_fingerprint_claim.py):
"top-1 re-identification: nearest neighbour of `h` under noise draw 1 among all
2048 positions' `h` under an **independent** noise draw 2". If you can recover
the position index from a noisy `h`, `h` is a positional code.

> **This probe writes nothing.** `grep -rl "reident\|fingerprint" runs/` is empty,
> so the mechanism's *direct* evidence — the re-identification rate — is not a
> committed artifact, and neither is the "probe: train sll −8.3 / val −15.2 with
> h" figure in [configs/comcat25.yaml](../configs/comcat25.yaml)'s comments. The
> committed evidence is the `h`-sweep (strong, quantitative) plus a design
> argument (not measured). Do not present the mechanism as measured.

### 4.3 Why fixed-sigma noise on a narrow bottleneck did not save it

The repo already knows the answer; it is in a comment in
[configs/comcat25.yaml](../configs/comcat25.yaml):

> `# h channel rejected twice (B: best-at-first-checkpoint again; encoder can`
> `# amplify h_proj past fixed-sigma noise). Relational features only.`

Here is that argument made quantitative. Model the bottleneck as `h` parallel
additive white Gaussian noise channels, `Y = X + N`, `N ~ N(0, sigma^2)`,
`sigma = 0.3`, with signal power `E[X_j^2] <= P`. The capacity per channel use is

```
C_1 = (1/2) * ln(1 + P / sigma^2)      nats
C_h = h * (1/2) * ln(1 + P / sigma^2)  nats, for h independent dims
```

(Cover & Thomas, *Elements of Information Theory*, Gaussian channel.) The
improvement in expected log-likelihood available from conditioning on `Y` is at
most the mutual information `I(target ; Y | rest)`, which is at most `C_h`. So
`C_h` upper-bounds how much likelihood the `h` channel can buy — under whatever
distribution you take the expectation over, including the **empirical training
distribution**, which is the one being memorized.

How much does `h=4` need? Its train `nll` improves from 7.281167 (`h=0`,
`ckpt_last`) to 4.143447 — a gain of 3.1377 nats/event. Solve

```
4 * (1/2) * ln(1 + P/sigma^2) = 3.1377
    ln(1 + P/sigma^2) = 1.5688
    1 + P/sigma^2 = e^1.5688 = 4.801
    P/sigma^2 = 3.801
    P = 3.801 * 0.09 = 0.3421  ->  RMS signal amplitude 0.585 per dimension
```

(This is an upper bound on what the channel *could* carry, not a measurement of
what it did carry: capacity bounds the achievable mutual information, so an
encoder achieving the 3.14-nat gain must have `P >= 0.342`, but nothing here
says the code was operating at capacity.)

The encoder needs an output RMS of at least **0.585** in each of four dimensions to
carry the whole memorization budget past a 0.3-sigma noise floor. `h_proj` is an
unconstrained `nn.Linear` — nothing bounds `P`. Weight decay 0.03 penalizes it,
but a 3-nat-per-event likelihood gain buys an enormous amount of weight norm.

Three further reasons the regularizer was toothless:

- **Noise is train-time only** (`if self.training`). At evaluation `h` is read
  cleanly, so the effective SNR at test is infinite. Training-time noise only
  forces the code to be *robust*; it does not force it to be *small*.
- **Narrow != low-capacity for real-valued channels.** Four float32 numbers carry
  up to 128 bits ~ 89 nats of raw precision. Width bounds capacity only under a
  power or precision constraint. Compare what memorization actually needs:
  pinning a location to 0.5 km inside a ~500 km box costs
  `2 ln(500/0.5) = 13.8` nats. One dimension would do.
- **Noise is applied *after* the projection.** If it were applied to a
  norm-constrained representation — LayerNorm, `tanh`, or explicitly
  variance-normalized `h` — then `P` would be bounded and `C_h` would be a real
  budget. That, and not `h_noise`, is the fix a reviewer will suggest, and it is
  a legitimate experiment the repo does not run.

### 4.4 Why "best held-out checkpoint is the first one evaluated" is decisive

Validation runs every 250 steps. For `h = 4, 16, 64` the best val checkpoint is
step **250** — the first check that ever ran
([runs/ablation_h/ablation_h.json](../runs/ablation_h/ablation_h.json), `step: 250`
in all three rows). For `h = 0` it is step 7,750 of 11,750.

Why this kills the "just early-stop it" defence:

1. **There is no non-degenerate stopping point.** Early stopping selects the
   argmin of a validation curve; when the argmin is at the left boundary of the
   grid, the procedure has not found a good model, it has declined to train.
2. **The step-250 models lose to `h=0` anyway.** Their test `nll` is 8.2158 /
   8.1249 / 8.0634 against `h=0`'s converged 7.6210 — even the optimally-stopped
   `h > 0` model is 0.44–0.59 nats/event worse than the structural one.
3. **It dates the pathology to initialization.** Memorization is present within
   250 optimizer steps, so it is not a late-training phase that appears after
   generalizable structure has been learned — it is the *first* thing gradient
   descent finds, because the fingerprint is the lowest-loss direction available.
4. **A finer grid probably would not help.** With `val_every = 25` one might find
   a better point in `[0, 250]`; but across `h` the best-checkpoint gap *grows*
   (0.209, 0.244, 0.274) while the best test `nll` *falls* (8.216, 8.125, 8.063),
   i.e. the models converge toward each other from an already-losing position.
   That is an argument, not a measurement; nobody has run the finer grid.

### 4.5 Relation to the ML literature and to bias–variance

- **Classical framing fails.** Adding the `h` channel adds capacity, so bias
  falls and variance rises — but the classical story predicts a *U-shaped* test
  curve with a useful minimum. What is observed is monotone divergence from step
  250 and a 15.5-nat gap. That is not a variance term; it is *distribution
  shift* — the conditioning input `h_i` at test time lies off the manifold of `h`
  values the heads were fitted on. `scripts/probe_h_mismatch.py` exists to test
  exactly that ("is eval-time conditioning OOD vs training-time conditioning?")
  and its output is not committed.
- **Memorization literature.** Zhang, Bengio, Hardt, Recht & Vinyals (ICLR 2017)
  established that high-capacity networks fit arbitrary labels. Carlini et al.
  (USENIX Security 2019, *The Secret Sharer*) showed generative sequence models
  memorize rare training strings and that this is measurable by *exposure* —
  precisely what the re-identification probe measures. Feldman (STOC 2020) argues
  long-tail memorization can be *necessary* for generalization, which is the
  strongest counter-framing an examiner can deploy ("your fix may have thrown
  away useful long-tail fitting"). The answer: here the memorized tail is
  *geographic position in the training era*, which by construction does not recur
  in the test era, so Feldman's mechanism does not apply.
- **Information-bottleneck framing.** This is an information bottleneck with the
  wrong constraint: it bounds the channel's *dimension* rather than its *rate*.
  §4.3 derives why that is the wrong knob.
- **[NOVELTY.md](../NOVELTY.md) records this sub-claim as unconfirmed either way** —
  the literature sweep found "no source either way" and instructs: "frame as
  diagnostic novelty, not 'first'". Say it before an examiner does.

---

## 5. The full claim inventory

This section is the most valuable thing here for a viva. Know the weak claims
before the examiner does.

### 5.1 The tallies, from `results/CLAIMS.md`

[results/CLAIMS.md](../results/CLAIMS.md) traces **142 claim rows** covering **134
distinct claims**:

| status | rows | meaning |
|---|---|---|
| `MATCH` | 63 | artifact equals the stated value at the artifact's own precision |
| `ROUNDING` | 51 | artifact carries more digits and rounds exactly to the stated value |
| `NO ARTIFACT` | 13 | nothing committed backs the number |
| `MISMATCH` | 11 | the artifact contradicts the manuscript |
| `AMBIGUOUS` | 2 | two committed artifacts could back the value and they disagree |
| `MATCH` with a `NO ARTIFACT` caveat (T23) | 1 | |
| MISMATCH-adjacent wording (X31) | 1 | |
| **total** | **142** | |

Reading that down: **114 rows match exactly or to rounding** (63 exact + 51 to
rounding), the 11 `MISMATCH` rows are **8 distinct** contradictions (M1–M8), and
the 13 `NO ARTIFACT` rows plus T23 cover **12 distinct** unbacked claims
(N1–N12). Note that "matches to rounding" is very nearly half of the 114 — sound,
but only the 63 are exact.

### 5.2 The eight contradictions

| # | what the manuscript says | what the artifact says | artifact |
|---|---|---|---|
| M1 | per-seed spread of the six head `dS` values is <=0.003 (`:570`), or <=0.006 (`:972`) | Chile range 0.0070, Iran 0.0056, Greece 0.0045 — 3 of 6 exceed it | `neural_etas/{Chile,Iran,Greece}_25/summary_full_s{0,1,2}.json` |
| M2 | "Per-event and full-suite results are 3-seed" while reporting the six composite totals | the six totals are **seed 0 only**; a 3-seed Chile total would be +0.064, not +0.061 | `stats_hardening.json` → `total_with_head_family` |
| M3 | a transferred background-only head "still wins"; Japan→Greece modulation converts a non-win into a win | Japan→Greece bg-only is **−0.015**, a loss; full transferred head is `dS` 0.0282, `decision: "tie"`. Loss → tie | `neural_etas/spatial_transfer_summary.json` |
| M4 | Greece native `dT` 95% CI **[−0.144, −0.070]** | **[−0.16309, −0.04480]** — stated interval ~35% too narrow; point estimate −0.107 exact | `multiregion_master.json` → `Greece.native.paired.dT_ci` |
| M5 | Iran native `dT` 95% CI **[−0.347, −0.205]** | **[−0.36976, −0.17389]** | `multiregion_master.json` → `Iran.native.paired.dT_ci` |
| M6 | California and Chile temporal wins "individually significant in every era" | no per-window CI or p-value is stored anywhere; Chile has 10 of 19 180-day windows positive (0.5263), California 23 of 27 | `prospective.json` |
| M7 | (former note) figures are gitignored | 12 figures are tracked; `git ls-files figures/` returns 12 | `figures/` |
| M8 | `README.md`'s expected `reference/` tree is complete | incomplete: also needs `Datasets/NewZealand/`, `Datasets/Italy_Mw/`+`Italy_mw_raw/`, `Datasets/ComCat_forward/`, `Datasets/ComCat_extended/`, `Experiments/ETAS/pycsep_tests_parallel.py`, `output_data_<Cfg>/parameters_0.json` | `runs/*/config.yaml` |

M6 is the one that matters most rhetorically: it is the only claim where the
cited artifact *actively contradicts* the sentence rather than merely failing to
support it.

### 5.3 The twelve unbacked claims, by category

| category | items | what is missing |
|---|---|---|
| **Baseline provenance** | N1 | which `etas` fork produced every ETAS baseline. No version/commit/sha/package key exists in any of the 136 committed run JSONs or 90 committed YAMLs. `pyproject.toml:25-38` records two candidate forks and says outright the repo cannot decide. This is the *incumbent every gain is measured against* |
| **ETAS refit parameters** | N2, N3 | the refit2020 parameter vector (`a` 1.556→1.603, log10 mu −6.333→−6.389, rho 0.557→0.571, branching 0.968) and "converged in 12 iterations". The summary JSON holds only `window, n, tll, sll, nll, etas_name, fit_window, params_frozen_from` |
| **Numerical-agreement claims** | N4, N5 | the ETAS **temporal** term reproducing the package to ~1e-5/event and the 1.5e-4 anchor effect (the committed repro covers the **spatial** term only); the gridded simulator reproducing the head's per-event `sll` to 9.5e-7 (an exhaustive scan of `runs/**/*.json` finds no value in 1e-8…1e-5; the code only *prints* a max-abs-err over 40 sampled events with a 1e-3 threshold) |
| **`MANUSCRIPT.md` §4.4 localization** | N6, N7, N8 | the <0.5 km deficit shrinking −0.218 → −0.062; the "2–10 km band"; the 64%/85% recurrence-coverage figures. `spatial_gap_decomp.json` has **no distance strata at all**, and no script computes a "2–10 km" band |
| **Training-configuration claims** | N9, N11 | initialization starting "+0.002 to +0.004 above ETAS"; the block-bootstrap block length of 50 events (correct in code at `flowquake/stats.py:45`, recorded in no result JSON) |
| **Reproducibility** | N10 | that the region baselines and the refit control are regenerable from this repo — no script writes `reference/Experiments/ETAS/config/*.json` |
| **A baseline not run** | N12 | the full flETAS (EM, free background) spatial control. The committed substitute is the SGD `--refit-globals` control |

### 5.4 Two things `results/CLAIMS.md` has not caught

Both were found for this chapter by reading the artifacts directly. Expect an
examiner who does the same.

**(a) `stats_hardening.json` predates its own generating code.** The HEAD commit
(`2e8fa8a`, "Stop the headline statistics resting on a single training seed")
changed `scripts/stats_hardening.py` to glob `per_event_full_s*.csv`
([`:131`](../scripts/stats_hardening.py#L131)) and average over seeds, writing
`head_seeds`, `n_head_seeds`, `dTot_seed_means`, `dTot_seed_std`,
`seed_aggregation` and a `single_seed_warning` flag into every
`total_with_head_family` row ([`:214-231`](../scripts/stats_hardening.py#L214-L231)).
The committed `runs/stats_hardening.json` contains **none of those keys** — its
`total_with_head_family` entries are `n, dTot_mean, dTot_ci, decision, p_boot,
dTot_abs_below_0.05, temporal_variant, pairing, p_holm, significant_05_holm`. So
the committed headline numbers were produced by the *old*, seed-0 code. Re-running
the current script changes them (Chile +0.0608 → about +0.064). The code fix
landed; the artifact refresh did not.

**(b) The temporal side is still single-seed even after that fix.** `HEADLINE` in
[scripts/stats_hardening.py](../scripts/stats_hardening.py) is a dict of one
per-event CSV per region — `runs/n1_density/per_event_test.csv` for California
(seed 1555), `runs/italy_n1/`, `runs/japan_n1/`, `runs/chile_n1/`,
`runs/greece_fewshot/`, `runs/iran_fewshot/`. The seed-globbing fix touched only
the *head* seeds. So both `family_dT_holm` and `total_with_head_family` rest on
**one temporal training seed per region**.

---

## 6. Where the temporal family and the total family diverge

This is the single most under-advertised fact in the paper, and the first thing
a statistically literate examiner will find.

Both families live in [runs/stats_hardening.json](../runs/stats_hardening.json).
Both are Holm–Bonferroni step-down over six regions. They disagree.

| region | temporal `dT` | temporal p_raw | temporal **p_holm** | temporal verdict | total `dTot` | total p_raw | total **p_holm** | total verdict | coverage |
|---|---|---|---|---|---|---|---|---|---|
| California | +0.0533 | 0.0005 | **0.003** | significant | +0.1133 | 0.0005 | **0.003** | significant | 100% |
| Italy | +0.0712 | 0.0005 | **0.003** | significant | +0.2095 | 0.0005 | **0.003** | significant | 100% |
| Japan | −0.0139 | 0.13697 | **0.27393** | **not significant** | +0.0390 | 0.0015 | **0.0045** | significant | 96.26% |
| Chile | +0.0343 | 0.009 | **0.03599** | significant | +0.0608 | 0.0005 | **0.003** | significant | 97.09% |
| Greece | −0.0125 | 0.64784 | **0.64784** | **not significant** | +0.0756 | 0.0055 | **0.011** | significant | 91.95% |
| Iran | −0.0634 | 0.09098 | **0.27293** | **not significant** | +0.0844 | 0.0185 | **0.0185** | significant | 89.04% |

Read plainly:

- **Three of six regions that are significant on TOTAL are not significant on
  temporal alone** — Japan, Greece, Iran. In all three the temporal point
  estimate is **negative**.
- The temporal family is 3 significant wins, 3 non-significant (2 stored as
  `"tie"` after TOST, 1 — Iran — not equivalent at either margin: its 90% CI
  `[-0.12210, -0.00135]` excludes zero on the wrong side).
- The total family is 6 significant wins.
- **The six-for-six total win therefore rests substantially on the spatial
  head**, and in three regions *entirely* on it, in the sense that the temporal
  component is a non-significant negative there.
- Greece and Iran's temporal numbers are `temporal_variant: "fewshot"`. Their
  *native* temporal results are much worse: Greece `dT` −0.10692, Iran −0.27596
  ([runs/multiregion_master.json](../runs/multiregion_master.json)).

**On the additive decomposition.** `dTot = dT + dS` holds exactly for California
(0.0533 + 0.0600 = 0.1133) and Italy (0.0712 + 0.1383 = 0.2095), and only
approximately elsewhere — Japan 0.0386 vs 0.0390, Chile 0.0610 vs 0.0608, Greece
0.0732 vs 0.0756, Iran 0.0831 vs 0.0844. The reason is that the six-region `dS`
values come from `neural_etas/<region>/summary_full_s0.json`, paired on the
head's own event set (`n_test`), whereas `dTot` is paired on the three-way
intersection of temporal/head/ETAS-scored events. Different subsets, different
means. Say this before you are asked why the arithmetic does not close.

**On family definition.** Holm's procedure controls the family-wise error rate
*for the family you declare*. Two families of six were declared here, so the
effective correction is over six, not twelve. That is defensible if the two
families answer different questions (they do: "is the temporal head better" and
"is the composite better"), and indefensible if the total family is only
reported *because* the temporal family was disappointing. The artifact's own
`notes` field pre-states the family as "one headline `dT` claim per region",
which is the right form of protection — but it is stated in the artifact, not
in a pre-registration.

---

## 7. The out-of-time 2020–2026 replication

### 7.1 What it is

`scripts/build_comcat_forward.py` re-runs the benchmark's exact ComCat recipe
(USGS query, RELM/CSEP polygon filter, m_c 2.5 cut, duplicate jitter, azimuthal
equidistant projection) for the window after the benchmark's test end. Both
models are **frozen at their benchmark state**: FlowQuake's checkpoint carries
its own `cat.stats` (normalization constants and background grid), and
[flowquake/evaluate.py:61-66](../flowquake/evaluate.py#L61-L66) swaps the evaluation
catalog and window *without touching them*. ETAS is the 2007-fitted inversion,
frozen (`params_frozen_from: "ComCat_25 inversion (train<=2007, published with
benchmark)"`, [runs/forward_etas/summary.json](../runs/forward_etas/summary.json)).

### 7.2 The exact numbers

From [runs/total_win.json](../runs/total_win.json) → `forward_2020_2026`,
n = 10,187 events:

| quantity | value |
|---|---|
| FlowQuake `tll` | 1.0677136320078393 |
| ETAS `tll` | 1.0102738057926097 |
| FlowQuake `sll` (head) | −8.40797007548935 |
| ETAS `sll` | −8.474594617572162 |
| FlowQuake `nll` | 7.340256443481511 |
| ETAS `nll` | 7.464320811779553 |
| `dT` | mean +0.0574, CI [0.0376, 0.0819], win rate 0.6051, p_boot 0.0005 |
| `dS` | mean +0.0666, CI [0.0553, 0.0784], win rate 0.4785, p_boot 0.0005 |
| `dTot` | mean +0.1241, CI [0.1035, 0.1455], win rate 0.5516, p_boot 0.0005 |

All three replicate, and `dTot` is *larger* out of time than in-window
(+0.1241 vs +0.1133).

**The `dS` win rate of 47.85% with a positive mean** is worth understanding and
is the kind of thing an examiner probes. The head wins fewer than half the
individual events but wins by more when it wins — consistent with fixing the
heavy tail (events ETAS badly mislocates) rather than shifting the bulk. That is
the expected signature of a *background-map* improvement: most events are
aftershocks where ETAS's triggering term already dominates, and the map only
helps the minority that are background or off-trigger.

**Fairness control.** ETAS was also *re-inverted* through 2020 and re-scored on
the same window: its `nll` improves from 7.464320811779553 to 7.448446148714125
(0.0159 nats). Against that refit ETAS the composite's total win narrows to
+0.1082 (7.340256443481511 − 7.448446148714125), of which temporal +0.0522 and
spatial +0.0560. Note that **no artifact stores +0.108 or a CI for it** — it is a
derived difference of two stored means.

### 7.3 What it is not

- **Not prospective.** These events existed during development. The model was
  never shown them, but the author knew the period's seismicity — including the
  2024 M7.0 Cape Mendocino sequence — while designing.
- **Not registered.** No third party held the frozen model before the window
  opened.
- **Not operational.** `runs/total_win.json`'s own `notes[0]` says:
  "forward_2020_2026 is a retrospective out-of-time/pseudo-prospective
  replication, not a registered prospective forecast." Quote that sentence; it
  is the paper defending itself.

> **A stale block to avoid.** `runs/n1_density/eval_forward.json` carries
> `baselines.ETAS.tll = 1.4343428344882627` (the *in-window* value, not the
> forward window's 1.0102738057926097), its `split` field says `"test"`, and its
> `sll`/`nll` are the production kernel head, not the neural-ETAS head. Only
> `paired_vs_ETAS.temporal` in that file is safe to quote — and it is correct
> (1.0677136 − 1.0102738 = 0.0574398).

---

## 8. Transfer and the "foundation model" framing

### 8.1 What was actually done

Two distinct things, often conflated:

1. **Leave-one-region-out (LOO) pooled pre-training + few-shot fine-tuning.**
   Pre-train on the *other* m_c-4.0 regions, then evaluate the held-out region
   zero-shot, then fine-tune 2,000 steps on it
   ([REPRODUCE.md](../REPRODUCE.md) §4). This is genuine held-out-region transfer.
2. **A pooled *global* checkpoint** (`runs/pool_global`) trained on **all**
   regions, then evaluated everywhere ([runs/global_eval.json](../runs/global_eval.json)).
   This is a single-deployment-checkpoint demonstration, **not** transfer.

### 8.2 The LOO temporal numbers

From [runs/multiregion_master.json](../runs/multiregion_master.json), paired `dT`
against each region's own ETAS:

| region | native | zero-shot (LOO) | few-shot (LOO) |
|---|---|---|---|
| Japan | −0.015218 (tie) | −0.020584 (loss) | −0.021953 (loss) |
| Chile | +0.034257 (win) | −0.027069 (tie) | +0.041783 (win) |
| Greece | −0.106919 (loss) | −0.039520 (tie) | −0.012454 (tie) |
| Iran | −0.275965 (loss) | −0.104903 (loss) | −0.063428 (tie) |

The data-efficiency story is Greece and Iran: native training on 2,612 and 2,010
train events fails badly; transfer recovers Greece to TOST equivalence at the
±0.1 margin (90% CI `[-0.05599, +0.03520]`, `equivalent: true`) and narrows Iran
about fourfold (−0.276 → −0.063) *without* reaching equivalence
(90% CI `[-0.12210, -0.00135]`, `equivalent: false`).

Note that transfer **hurts** Japan at every stage. The honest sentence is
"transfer rescues data-poor regions and does not help data-rich ones", not
"transfer works".

### 8.3 The pooled global checkpoint: what it is and is not

[REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) is explicit, under "Holes
That Still Matter":

> The pooled global checkpoint is not unseen-region zero-shot transfer when the
> target region's training window was included in pooled pre-training.

[REPRODUCE.md](../REPRODUCE.md) says the same: "it is not leave-one-region-out
zero-shot transfer because each region's training window participates in the
pooled pre-training run." Its `replacement_readiness` check
`pooled_global_temporal` is a **WARN**, not a PASS. Its numbers
([runs/global_eval.json](../runs/global_eval.json)) are positive `dT` in four
regions (California +0.0313, Italy +0.1017, Japan +0.0114, Chile +0.0220) and
negative in two (Greece −0.0552, Iran −0.1580), with `dTot` negative in five of
six — because `global_eval.json` uses the *production* spatial head.

**How to describe it without overclaiming.** "One shared checkpoint, with no
per-region weight fitting after pooling, is temporally positive in four of six
regions under ordinary paired z-scores." Not: "a foundation model generalizes to
new regions."

### 8.4 The spatial head's transfer, and its hidden dependency

[runs/neural_etas/spatial_transfer_summary.json](../runs/neural_etas/spatial_transfer_summary.json)
reports 7 within-completeness-regime transfers, zero-shot: 6 `"win"` + 1
`"tie"` (Japan→Greece, `dS` 0.0282). Cross-completeness (m_c 2.5 source → m_c 4.0
target): **0 of 4**, all losses (−0.0672, −0.2292, −0.1464, −0.3280).

Read the file's own `note`: *"source-trained head applied to **target
ETAS-init features**"*. Zero-shot spatial transfer still requires the target
region's ETAS inversion and its precomputed full-history triggering sums. What
transfers is the ~1.3k learned parameters — the background mixture weights and the
`g(m_j, Δt)` modulation surface — not the ability to skip an inversion.

### 8.5 The framing guardrail

[NOVELTY.md](../NOVELTY.md) is unambiguous:

> Claim "first neural point-process / point-process-likelihood forecaster to
> transfer across tectonic regimes and beat region-fitted ETAS on temporal
> log-likelihood." Do **NOT** claim "first transfer learning for earthquake
> forecasting" unqualified — **SafeNet** preempts the broad version.

SafeNet (Zhang et al. 2025, *Scientific Reports*, DOI recorded in NOVELTY.md as
10.1038/s41598-025-93877-7) does cross-region few-shot transfer that beats ETAS
— but it is a 4°×4° gridded annual-max-magnitude **classifier** scored on
F1/recall. No likelihood, no intensity, no point process. The boundary you must
be able to state crisply: *different model class and different target
functional*. A classifier that predicts whether a cell exceeds a magnitude in a
year is not a forecaster of `f(tau, s, m | H)`.

---

## 9. External dependencies and reproducibility

Everything in this section was verified against the repository for this chapter.

### 9.1 `reference/` is not committed, and nothing runs without it

```
$ git ls-files '*.yaml' | wc -l                                  ->  123
$ git ls-files 'configs/*.yaml' | wc -l                          ->   33
$ git ls-files 'runs/**/*.yaml' | wc -l                          ->   90
$ git ls-files '*.yaml' | xargs grep -l 'catalog_path: *reference/' | wc -l
                                                                 ->  123
```

**All 123 tracked YAMLs — 33 in `configs/` and 90 under `runs/` — have
`catalog_path` under `reference/`, and not one of them resolves on a fresh
clone.** `.gitignore` line 10 excludes `reference/` outright.

### 9.2 Six ETAS configs no script in the repo can regenerate

`Japan_25`, `Chile_25`, `Greece_25`, `Iran_25`, `Italy_25` and
`ComCat_25_refit2020` are **not shipped by the EarthquakeNPP benchmark**. Grepping
the tree for anything that writes `reference/Experiments/ETAS/config/*.json`
returns only prose in `WORKING.md`, `MANUSCRIPT.md`, `REPRODUCE.md`, `README.md`
and `results/CLAIMS.md` — **no Python**.
[scripts/run_etas_regions.py](../scripts/run_etas_regions.py) only `subprocess.Popen`s
the benchmark's `invert_etas.py` / `predict_etas.py` with a config *stem*; it
never authors a config. Consequence: `README.md:126-129` and `REPRODUCE.md` §2
describe a reproduction path that works only on the author's machine, and those
six configs are the ones behind `MANUSCRIPT.md` §4.5's region baselines and
§4.1's refit
control.

### 9.3 Per-event CSVs are gitignored, so no CI can be recomputed

`.gitignore` excludes `per_event*.csv`, `*_per_event.csv`, `*.pt`, `*.npz`,
`CSEP_day_*.csv`, and admits only `*.json` / `*.yaml` under `runs/`. Verified:

```
$ git ls-files | grep -c '\.csv$'      ->  0
$ git ls-files | grep -c '\.pt$'       ->  0
$ git ls-files runs/ | wc -l           ->  226   (136 JSON + 90 YAML)
$ git ls-files | grep -i per_event     ->  runs/neural_etas/ComCat_25/per_event_forward_full.json
```

**Every stationary-block-bootstrap CI in the paper, and everything in
`runs/stats_hardening.json`, is a stored summary that cannot be re-derived from
the committed tree.** Exactly one per-event artifact is tracked repo-wide, and it
is a JSON summary of the forward window, not the raw pairing.

Because no `.pt` is tracked, `MANUSCRIPT.md`'s §4.3 phrase "reproducible from the
committed checkpoints" is wrong as written: the *metrics* are committed, the
checkpoints are not.

### 9.4 The `etas` fork is unpinned

`pyproject.toml:25-38` records two candidate implementations (`lmizrahi/etas`,
`ss15859/etas`), states they are different code, and marks the choice
`TODO [USER, blocks release]`. No version, commit, sha, package, env, provenance,
repo or git key exists in any of the 136 committed run JSONs or 90 committed
YAMLs. The fork matters for exactly two things: the five `MANUSCRIPT.md` §4.5
region inversions plus `ComCat_25_refit2020`, and its §4.2 ETAS CSEP column
(which imports the
installed package at runtime, [flowquake/etas_csep.py:70-71](../flowquake/etas_csep.py#L70-L71)).
California is safe either way — its baseline is the benchmark's own shipped
inversion.

### 9.5 What *can* be verified on a fresh clone

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q     # 16 pass; 6 data-alignment tests skip without reference/
```

Those 16 are meaningful: chunked SSD scan against an fp64 naive recurrence,
streaming-state continuation, flow `log_prob` against an analytic Gaussian
density, and numerical grid integration confirming the spatial mixture — including
a forcibly elongated anisotropic component — integrates to 1.

---

## 10. The attack surface, ranked

Ranked by how much damage a successful attack does. For each: (a) the strongest
form of the objection, (b) the best honest answer, (c) the experiment that would
settle it.

### 10.1 The head is initialized from the target region's ETAS inversion

**(a)** "Your spatial head *starts at* the object you claim to beat, and consumes
its precomputed triggering sums. You have not built a competitor to ETAS; you
have built a fine-tuner for ETAS, and you are reporting the fine-tuning gain as
a model comparison. Any parametric model plus a few thousand free parameters
fitted on the same data will beat itself."

**(b)** Concede the framing entirely — the repo already does, in
`REPLACEMENT_READINESS.md`, `README.md:11-13`, the abstract, and the conclusion.
Then make three defences. First, "strict superset" is a *stronger* claim than
"initialized from": with `kde_gate_init = -30`, modulations zero, `alpha = 1`,
the head reproduces the package's per-event ETAS `sll` to
**1.7655796824556091e-09** nats ([runs/etas_sll_repro.json](../runs/etas_sll_repro.json)),
so the comparison is exactly nested and the measured gain is a gain over ETAS
*by construction*, not an artifact of a different setup. Second, the reported
gains are always measured against the **package's** ETAS scores, never against
the near-ETAS training initialization. Third, there is a classical control:
`--refit-globals`, an flETAS-style SGD refit of ETAS's own global kernel
parameters with the same learned background, which asks "is the gain just
refitting ETAS better?"

**(c)** Train the head from a *neutral* initialization (isotropic kernel,
inversion-free `k0/a/c/omega`), and report the gain against the package ETAS. If
the gain survives, the inversion is scaffolding rather than substance. Not run.

### 10.2 The spatial win might be mostly the smoothed background — and the `--no-mlp` ablation settles it, badly

**(a)** "ETAS's background here is *uniform*. You replaced it with a fitted
multi-scale smoothed-seismicity map. Smoothed seismicity has beaten uniform
backgrounds since the 1990s. Your 'neural' contribution may be nothing."

**(b)** This is the strongest technical objection, and the committed artifacts
partly concede it. Read the ComCat ablation, all seed 0, all paired against the
same package ETAS:

| configuration | `dS_mean` | 95% CI | share of the full gain |
|---|---|---|---|
| `bg_only` (background mixture + `alpha` + `mu'`, **no MLP**) | 0.0513 | [0.0434, 0.0595] | 85.5% |
| `refit_globals` (classical flETAS-style SGD refit, no MLP) | 0.0564 | [0.0477, 0.0654] | 94.0% |
| `full` (+ per-parent neural modulations) | 0.0600 | [0.0509, 0.0692] | 100% |

Sources: `runs/neural_etas/ComCat_25/summary_{bg_only_s0,refit_globals_s0,full_s0}.json`.

So on ComCat the neural modulation adds **+0.0087** over background-only and
**+0.0036** over a classical global refit — 6–15% of the spatial gain — and the
three confidence intervals mutually overlap on `[0.0509, 0.0595]`. **No committed
artifact provides a paired CI on the increment**, which is the statistic that
would settle whether the neural part is doing anything at all.

> **And it is worse than the manuscript says.** `MANUSCRIPT.md:549-560` states
> "the neural component contributes a smaller, **consistent** increment on top"
> (`:559-560`),
> and reports the ablation for ComCat only. The committed artifacts for the other
> regions contradict "consistent":
>
> | region | `bg_only_s0` `dS` | `full_s0` `dS` | neural increment |
> |---|---|---|---|
> | ComCat_25 | 0.0513 | 0.0600 | **+0.0087** |
> | Italy_25 | 0.0542 | 0.1383 | **+0.0841** |
> | Japan_25 | 0.0556 | 0.0525 | **−0.0031** |
> | Chile_25 | 0.0351 | 0.0267 | **−0.0084** |
> | Greece_25 | — | 0.0857 | no `bg_only` run committed |
> | Iran_25 | — | 0.1465 | no `bg_only` run committed |
>
> **In two of the four regions where the ablation exists, background-only beats
> the full head.** The CIs overlap, so this is not evidence that the modulation
> *hurts*; it is decisive evidence that "consistent increment" is not supported.
> This finding is **not** in `results/CLAIMS.md` — it is a ninth contradiction.
> Also note Italy's `refit_globals` control gives `dS` 0.0102 [0.0051, 0.0155],
> far *below* its own `bg_only` 0.0542, which suggests the SGD global refit can
> actively damage the fit and that the control is not uniformly the
> "conservative lower bound" it is described as.

**(c)** Two experiments. (i) Add the same causal multi-scale smoothed-seismicity
background to ETAS itself and re-invert — the "flETAS with free background"
baseline flagged as N12 and never run. If ETAS-with-background matches the head,
the neural contribution is zero. (ii) Compute the *paired per-event* difference
`sll_full − sll_bg_only` and bootstrap it, in all six regions and all three
seeds. That is a CPU-minutes experiment and it is the one that decides the
sub-claim.

### 10.3 The composite mixes two models

Covered in §3.4. **Settling experiment:** joint training of flow + neural-ETAS
head, model-selected on composite `nll`, reported end to end.

### 10.4 Per-region normalization and a fitted background map are still required

**(a)** "You claim transferability, but deployment in a new region still needs
z-scoring statistics from that region's catalog, a train-era smoothed-seismicity
map from that region's catalog, and (for the head) that region's ETAS inversion.
That is most of an ETAS deployment."

**(b)** True and stated: `README.md:78-80`, `REPRODUCE.md` notes,
`REPLACEMENT_READINESS.md`. The defensible version of the claim is *"lighter than
an ETAS inversion, not zero target-catalog preprocessing"*. Quantitatively: the
normalization is a handful of means and sds; the background map is a KDE over
train-era epicentres (no optimization); the ETAS inversion is 3–4 h of EM per
large region. Those are different orders of cost — but they are not zero.

**(c)** Deploy on a region with *no* prior catalog beyond the minimum needed for
the map, and report degradation as a function of the length of the
map-fitting window.

### 10.5 Test-set reuse across the reported grid

**(a)** "You report a grid of ablations, controls and three seeds. Every one of
them was scored on the 2007–2020 test set. Your headline p-values are
conditioned on a search you do not correct for."

**(b)** The repo says this about itself, in two places. The trainer's docstring
([scripts/train_neural_etas.py:5-8](../scripts/train_neural_etas.py#L5-L8)): *"The
reported grids include ablations and multiple seeds; do not describe these runs
as a test-scored-once protocol."* And `MANUSCRIPT.md:542-545`: *"the reported
grid includes ablations, controls and three seeds that were all scored on the
2007–2020 test set during development; we therefore lean on the 2020–2026
out-of-time replication as the cleaner single-window confirmation."* That is the
right defence — the forward window was scored once, with frozen models — and it
is why §7 carries as much weight as it does. It is *not* a complete defence,
because architecture and hyperparameters were chosen with knowledge of in-window
test performance, and that knowledge transfers to the forward window.

**(c)** Pre-register the model and score a genuinely future window (rung 4).

### 10.6 ODE step-count sensitivity

**(a)** "Your `tll` is an ODE integral. You evaluated the in-window test set at
one step count and the forward window at another. How do I know the 0.05-nat
effect is not integration error?"

**(b)** This is a real inconsistency and there is also a decisive answer.
The inconsistency first: `README.md:160` documents `--steps 96`, but the three
headline ComCat seeds were scored at **64** steps
(`runs/{n1_density,n1_s1553,n1_s1554}/eval_test.json` → `ode_steps: 64`) while
`runs/n1_density/eval_forward.json` used **96**. Concede it, and name the cause:
`flowquake.train`'s `--eval-after` path hard-codes `--steps 64`
([flowquake/train.py:185](../flowquake/train.py#L185)), whereas the `flowquake.evaluate`
CLI default is 96 ([flowquake/evaluate.py:57](../flowquake/evaluate.py#L57)). The
in-window numbers came from the trainer, the forward-window numbers from a
separate invocation.

Now the answer. Two run directories, `final_s1555` and `comcat25_s1555`, are the
*same* configuration — their `config.yaml`s differ only in `out_dir` and in
`final_s1555` explicitly writing `mix_hidden: 64`, which is
`flowquake/config.py:34`'s default — trained with the same seed, and evaluated at
**64** and **96** ODE steps respectively:

| file | ode_steps | `tll` | paired temporal `mean_gain` |
|---|---|---|---|
| `runs/final_s1555/eval_test.json` | 64 | 1.485485315322876 | 0.051142535875441514 |
| `runs/comcat25_s1555/eval_test.json` | 96 | 1.485485315322876 | 0.051142530276869186 |

Be precise about which half of that table carries the argument. The `tll` values
are stored as float32, so their bit-for-bit agreement only bounds the difference
below float32 resolution (~1e-7 relative) — necessary but not impressive. The
informative number is the float64 paired mean gain, which differs by
**5.6e-9 nats/event** — nearly seven orders of magnitude (10^6.7) below the
smallest reported effect (SanJac's +0.028). RK4 at 64 steps is converged for this
integrand. Use this pair; it is the only committed same-model, two-step-count
comparison in the repository.

**(c)** A three-point step ladder (32 / 64 / 128) on one checkpoint, written to
JSON, would make the argument citable rather than reconstructed.

### 10.7 Block length choice for the bootstrap

**(a)** "You chose `mean_block = 50` events. Aftershock sequences run to
thousands of events. Too short a block leaves autocorrelation in the resample and
your CIs are too narrow — which is exactly the direction that manufactures
significance."

**(b)** Honest answer: the choice is defensible but undocumented and untested.
`mean_block = 50` is the default at
[flowquake/stats.py:45](../flowquake/stats.py#L45) (and `:87, :116, :145, :164`),
and no `scripts/` call site overrides it. **No result JSON records it** (N11), so
a reviewer working from the evidence pack cannot even confirm the resampling
scheme, let alone its sensitivity. The stationary bootstrap of Politis & Romano
(1994, JASA) is the right tool — geometric block lengths, wrap-around, and it is
consistent for the mean of a weakly dependent series — but consistency is
asymptotic in block length, and 50 is a guess.

**(c)** Report the headline CIs at `mean_block ∈ {10, 50, 200, 1000}`. If the
California CI `[0.0402, 0.0674]` survives at 1000, the objection dies. This is
minutes of CPU on per-event CSVs that are not committed.

### 10.8 Family definition for multiple testing

Covered in §6. **(c)** Pre-register the family (one `dT` and one `dTot` claim per
region, twelve tests, one Holm family) and report both the six-family and
twelve-family adjusted p-values. Iran's total (`p_holm` 0.0185) is the row most
at risk under a twelve-member family.

### 10.9 Magnitude heterogeneity in the foreign catalogs

**(a)** "Your ISC/INGV catalogs use agency-preferred magnitudes of mixed type. A
model that learns `beta(cond)` can exploit magnitude-type artifacts that have
nothing to do with seismology, and your Italy result — the largest total gain,
+0.2095 — is on a native-scale catalog (`runs/mw_robustness.json` keys it `ml_*`;
its own `interpretation` calls it M_d-dominated — the repo is not internally
consistent about which local scale it is, which is itself the point)."

**(b)** Partly answered, and the answer is uncomfortable.
[runs/mw_robustness.json](../runs/mw_robustness.json):

- **California** is fine: on the unambiguously-M_w M>=3 subset the production
  model's `dT` is +0.074 [0.0503, 0.1005], a win.
- **Italy is not.** Under M_w homogenization (`mw_uniform_mc26`, n = 8,525) the
  temporal result *inverts*: `dT` **−0.2532** [−0.2885, −0.2205], `dS` −0.2090,
  `dTot` −0.4622. And a density-matched native-scale control at m_c 2.8
  (`ml_mc28_density_control`,
  n = 5,346) gives `dT` +0.0022 [−0.0244, 0.0308], a **tie**. The file's own `interpretation` says
  the win "erodes as a density effect".
- `REPRODUCE.md` flags magnitude type as a caveat and `README.md` says the Italy
  total is "native-catalogue scale, not claimed under Mw homogenization".

So the correct statement is: *Italy's win is a native-catalogue result that does
not survive M_w homogenization, and the erosion is attributable to catalog
density rather than magnitude type per se.* If you say "Italy +0.21" without that
sentence, you have overclaimed.

- **Completeness** across the four ISC regions ([runs/completeness.json](../runs/completeness.json))
  is also heterogeneous in time: Japan `mc_train` 3.65 → `mc_test` 3.75,
  Chile 3.95 → 3.65, Greece 3.65 → 3.85. All were analysed at m_c 4.0, which is
  above every one of those, so the cut is conservative. But `b_train` vs `b_test`
  moves a lot (Chile 0.94 → 0.76), which is a real non-stationarity the GR head
  must absorb.

**(c)** Re-invert ETAS and re-train on M_w-homogenized catalogs for all five
foreign regions, and report the whole table on that footing.

### 10.10 Salton Sea swarms

**(a)** "Your largest California temporal gain is Salton Sea, +0.102 nats/event.
Salton Sea is a geothermal swarm region. Swarms are precisely where Omori is
wrong, so a flexible temporal density will win there for reasons that say
nothing about aftershock forecasting. And the catalog is small — 4,104 test
events."

**(b)** Half of this is the paper's own thesis and should be embraced: FlowQuake
wins where ETAS's parametric form is wrong, and ties where it is right (Japan's
Tohoku-dominated m_c-4.0 test set is textbook Omori and is a tie). Saying "we win
on swarms" is a *mechanistic* claim, not an embarrassment. The half you must
concede: **nothing in this repository stratifies performance by swarm-vs-mainshock
sequence.** There is a magnitude stratification
([runs/mag_robustness.json](../runs/mag_robustness.json)) and a time-window
stratification (`prospective.json`), but no sequence-type stratification. Also
note the SaltonSea bootstrap `n` is 4,103 against an eval `n` of 4,104 — one event
lost in the FlowQuake/ETAS time merge — and its per-seed `tll` spread (sd 0.0070)
is the largest of the five catalogs.

**(c)** Declare swarm episodes (e.g. by a standard swarm criterion on the
Baiesi–Paczuski or Zaliapin–Ben-Zion nearest-neighbour statistic) and report `dT`
inside and outside them, in all five California catalogs.

### 10.11 The McNemar comparison's power

**(a)** "Your headline CSEP claim is that the full-history head's S-test is
'statistically indistinguishable from ETAS', with McNemar exact p = 1.00. A
p-value of 1.00 is not evidence of equivalence. What is your power?"

**(b)** Concede immediately and quantify. The pairing
([results/CLAIMS.md](../results/CLAIMS.md) C17–C19, recomputed from the raw
`results[].S.quantile` arrays): 83 commonly evaluable days, head passes 77, ETAS
passes 77, **10 discordant days split 5–5**, two-sided exact binomial p = 1.0000.

McNemar's test conditions on the discordant pairs, so *all* the information is in
those 10 days. The exact 95% Clopper–Pearson interval for 5 successes in 10 is
**[0.1871, 0.8129]**, i.e. an odds ratio anywhere in **[0.230, 4.345]**. The data
are consistent with ETAS's S-test pass odds being four times better *or* four
times worse than the head's. The most extreme possible split (10–0) would have
given p = 0.00195, so the test *can* reject — but only for an effect near total
dominance.

The honest sentence is: *"the paired S-test did not detect a difference, at very
low power; this is a non-degradation result, not an equivalence result."* Never
say "indistinguishable" without the power caveat.

**(c)** Extend to 1,000 forecast days rather than 100, or move to a paired
*continuous* statistic (per-day S-test quantile difference, bootstrapped) rather
than a binary pass/fail, which throws away almost all the information.

### 10.12 CSEP consistency is not a win

**(a)** "You present N 95/100, S 79/85, M 90/92 as a result. Those are
*consistency* tests. Passing them means your forecast is not detectably
miscalibrated. A climatological forecast passes them too."

**(b)** Entirely correct, and the repo agrees: `REPLACEMENT_READINESS.md` calls
this "the spatial likelihood gain costs nothing in consistency". The role of CSEP
here is a **guard**, not a claim: it rules out the failure mode where a
likelihood gain comes from a density that is systematically wrong about counts or
magnitudes. Note also the nominal rate: at alpha = 0.05 you expect ~95% pass, so
95/100 and 90/92 are on target and 79/85 (92.9%) is slightly under. There is also
a known off-by-one — the stored `summary.S` denominators of the two 1e4-simulation
production-head runs (`runs/n1_density/csep`, `runs/final_s1555/csep`) include
the harness's own `[-1.0, -1.0]` not-evaluable sentinel day, so the stored
85/92 = 92.4% should be 85/91 = 93.4%, and 81/92 = 88.0% should be 81/91 = 89.0%
(AMBIGUOUS row A2; I re-ran the exclusion over both files' `results[]` and get
91 evaluable days in each). The correction runs *in the author's favour*.

**(c)** Nothing; this is a framing fix, not an experiment. State CSEP as a
non-degradation guard.

### 10.13 One temporal seed under the headline statistics

Covered in §5.4(b). **(a)** "Your Holm families rest on one temporal training
seed per region, and the across-seed sd of `tll` in the five California catalogs
runs 0.0007–0.0070 (`fullsuite_summary.json` → `tll_sd`) — the largest of those
is a quarter of the smallest reported effect, SanJac's +0.0284." **(b)** Concede; the fullsuite table *is* 3-seed, but
`stats_hardening.json` is not. **(c)** Re-run `stats_hardening.py` over all three
temporal seeds per region and report seed-averaged families.

### 10.14 The `etas` fork

Covered in §9.4. **(a)** "Every number is measured against an implementation you
cannot name." **(b)** California is safe (benchmark-shipped inversion); the five
foreign regions and the refit control are not. **(c)** Read
`etas-*.dist-info/direct_url.json` in the training environment — for a VCS
install it records the git URL and the resolved commit. `results/CLAIMS.md` lists
seven commands under "ETAS provenance", and calls this one "the single decisive
file".

---

## 11. What would have to be true for an operational replacement

[REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md) defines a five-rung ladder.
Rungs 1–3 are `[DONE]`;
[runs/replacement_readiness.json](../runs/replacement_readiness.json) reports
`overall: "RESEARCH_PREVIEW_READY"` across 15 checks, 11 PASS and 4 WARN
(`california_spatial_total_gap`, `california_block_bootstrap_temporal`,
`pooled_global_temporal`, `legacy_package_surface`).

| rung | status | the concrete experiment |
|---|---|---|
| 1. Research preview | DONE | tests pass; California temporal suite reproduced; standalone CSEP; cross-regime tables; `audit_readiness.py` |
| 2. Incumbent head-to-head | DONE | ETAS and FlowQuake through the *same* pyCSEP harness, 100 identical days, matched 1e3 catalogs (`csep_h2h_{fq,etas}`; I checked that the `results[].day` lists are element-for-element identical across all **seven** committed 100-day CSEP runs) |
| 3. Full-head CSEP | DONE | the neural-ETAS head through the same harness: N 95/100, S 79/85, M 90/92, paired S-test vs ETAS p = 1.00 |
| **4. Prospective deployment** | **NOT DONE** | freeze a checkpoint, deposit and hash it with a third party *before* the forecast period opens, run rolling 1-day-ahead N/S/M against ETAS on a declared future window |
| 5. Operational artifact | NOT DONE | package one checkpoint + preprocessing + calibration + forecast export + audit logs + failure-mode monitoring |

**Rung 4 is structurally impossible to do alone**, and this is the honest reason
the work needs a collaborator rather than more compute. A registered prospective
forecast requires a third party to hold two things the author cannot
self-certify: the frozen model, deposited before the period opens, and the future
catalog, which does not exist yet. The evidential value comes entirely from the
custody being external. No amount of additional retrospective work — more
regions, more seeds, a full flETAS baseline — substitutes, because the objection
being answered is not "is the model good" but "did anyone see the answers first".
`flowquake/csep_forecast_head.py` already implements the exact protocol; what is
missing is a custodian and a clock.

Beyond rung 4, the experiments that would convert the *scientific* claim into a
defensible operational one, in priority order:

1. **Paired-increment CIs on the `--no-mlp` ablation, all six regions, all three
   seeds** (§10.2). Cheapest, and it decides whether the "neural" part of the
   neural-ETAS head exists.
2. **A jointly trained single model** (§3.4). Removes the composite objection
   entirely.
3. **Neutral-initialization head** (§10.1). Removes the "you fine-tuned ETAS"
   objection.
4. **Full flETAS with a free smoothed background** (N12). The baseline that would
   show whether ETAS-plus-background already matches the head.
5. **M_w-homogenized re-inversion of all five foreign regions** (§10.9).
6. **Block-length sensitivity and a step-count ladder, both written to JSON**
   (§10.6, §10.7).
7. **Commit the six ETAS configs and a pinned `etas` commit SHA** (§9.2, §9.4).

---

## 12. Worked examples

### 12.1 The ComCat total win, by hand

The claim is `nll` 7.142 vs 7.255. Reconstruct it from
[runs/total_win.json](../runs/total_win.json) → `test_2007_2020`:

```
FlowQuake:  tll = +1.487639097333936
            sll = -8.629760984221207
            nll = -(tll + sll)
                = -( 1.487639097333936 - 8.629760984221207 )
                = -(-7.142121886887271)
                =  7.142121886887271

ETAS:       tll = +1.4343428344882627
            sll = -8.689770387238829
            nll =  7.255427552750566

dTot = 7.255427552750566 - 7.142121886887271 = 0.113305665863296
```

Cross-check the decomposition against the stored paired means: `dT` 0.0533 plus
`dS` 0.0600 equals 0.1133 — matching `dTot.mean` exactly, because ComCat's
pairing coverage is 100%.

**Interpretation.** 0.1133 nats/event means the composite assigns each observed
event a joint density `exp(0.1133) = 1.1200` times higher than ETAS does — a 12%
improvement in likelihood per event. Over the 21,889 test events that is
`0.1133 × 21889 = 2480` nats of total log-likelihood, or about 3,578 bits. In
information-gain-per-event terms — the currency CSEP and the forecasting
literature use — it is 0.1133 / ln 2 = 0.163 bits/event.

Sanity anchor: the gap between ETAS and a homogeneous Poisson on the same data is
`13.261863460288378 − 7.255427552750566 = 6.006` nats/event. So the composite
closes an additional **1.9%** of the distance from Poisson to ETAS. That framing
keeps you honest: the effect is real and significant, and it is small next to the
structure ETAS already captures.

### 12.2 Is the neural modulation doing anything? (five lines of Python)

```python
bg, refit, full = 0.0513, 0.0564, 0.0600      # runs/neural_etas/ComCat_25/summary_*_s0.json
print(bg / full, refit / full)                # 0.855   0.940
print(full - bg, full - refit)                # 0.0087  0.0036
# CIs: bg [0.0434,0.0595]  refit [0.0477,0.0654]  full [0.0509,0.0692]
```

The intervals overlap on `[0.0509, 0.0595]`. Since these are three *separate*
bootstrap CIs on three *separate* means and not a CI on the paired difference,
overlap does not by itself prove non-significance — but it does mean the repo
supplies no evidence that the increment is significant. Combine with the Japan
(−0.0031) and Chile (−0.0084) increments from §10.2 and the honest conclusion is:
**the causal multi-scale smoothed-seismicity background is the spatial result;
the neural modulation is unproven and region-dependent.**

### 12.3 How much noise would have been enough? (the channel-capacity calculation)

The `h=4` model buys `7.2812 − 4.1434 = 3.1377` nats/event of *train*
log-likelihood over `h=0`. Treat `h` as 4 parallel AWGN channels with noise
`sigma = 0.3`:

```python
import math
gain, h, sigma = 3.1377, 4, 0.3
P = sigma**2 * (math.exp(2*gain/h) - 1)     # required signal power per dim
print(P, math.sqrt(P))                      # 0.3421  0.5849
```

An encoder output RMS of **0.585** per dimension is all the channel needs to be
*capable* of carrying the entire memorization budget past the noise floor. Now
invert the question: what `sigma`
would have bounded the gain at, say, 0.1 nats/event *given* an encoder that has
already learned to output RMS 0.586?

```python
P = 0.586**2
for s in (0.3, 1.0, 3.0, 10.0):
    print(s, h * 0.5 * math.log(1 + P/s**2))
# 0.3 -> 3.1437    1.0 -> 0.5904    3.0 -> 0.0749    10.0 -> 0.0069
```

You would need `sigma ~ 3`, ten times larger, and the encoder would simply scale
`h_proj` by 10 in response. **The fix is not a bigger `sigma`; it is bounding
`P`** — LayerNorm or `tanh` on `h`, or noise scaled to the signal
(`h * (1 + sigma*eps)`), either of which makes the SNR, and therefore the
capacity, a genuine hyperparameter. That experiment is not in the repository.

### 12.4 Verifying the ETAS spatial normalizer

The neural-ETAS head asserts `Z_j = pi / (rho_j * d_mj^rho_j)` is the exact
integral of `K_j(r) = (r^2 + d_mj)^{-(1+rho_j)}`
([flowquake/neural_etas.py:83-85](../flowquake/neural_etas.py#L83-L85)). Confirm:

```
integral over the plane = integral_0^inf (r^2 + D)^(-(1+p)) * 2*pi*r dr
    substitute u = r^2,  du = 2r dr
  = pi * integral_0^inf (u + D)^(-(1+p)) du
  = pi * [ (u+D)^(-p) / (-p) ]_0^inf
  = pi * D^(-p) / p                                   for p > 0
  = pi / (p * D^p)                                    ✓
```

This is why the head is exactly normalized *with no numerical integration
anywhere*, and why the MLP must never see the target location `s`: if `rho_j` or
`d_mj` depended on `s`, `Z_j` would no longer be the integral of `K_j` and you
would be learning an unnormalized energy with an intractable partition function.

Numerically, on a 1000×1000 km grid at 1 km resolution with `p = 0.557`,
`D = 1.0`:

```python
import numpy as np
p, D = 0.557, 1.0
g = np.arange(-500, 500, 1.0) + 0.5
X, Y = np.meshgrid(g, g)
K = (X**2 + Y**2 + D)**(-(1+p))
print(K.sum()*1.0, np.pi/(p*D**p))     # 5.5890 vs 5.6402 -- the truncation gap
```

The finite grid under-integrates by 0.9% because the tail is heavy: with
`p = 0.557` the mass outside radius `R` falls only as `R^(-2p) = R^(-1.11)`, so
even a 500 km half-width leaves measurable mass outside.
That is the same heavy tail that makes this kernel beat a Gaussian, and it is a
useful reminder that "normalized" is a statement about the whole plane, while
CSEP grids are finite.

### 12.5 The McNemar power calculation

```python
from math import comb
n, k = 10, 5
probs = [comb(n, i)/2**n for i in range(n+1)]
p_two_sided = sum(p for p in probs if p <= probs[k] + 1e-15)
print(p_two_sided)                 # 1.0
print(2*probs[10])                 # 0.001953125  -- the smallest attainable p
```

Clopper–Pearson 95% for 5/10 (solve the two binomial tail equations):
**[0.1871, 0.8129]**, so the odds ratio lies in **[0.230, 4.345]**. Report the
interval, not the p-value.

---

## 13. How this shows up in FlowQuake

This section maps theory to files. For *what the code does*, read
[STACK.md](../STACK.md); the pointers below are for locating the argument, not
re-explaining the implementation.

| section of this chapter | code / artifact | STACK.md part |
|---|---|---|
| §2.1 `f(tau\|H)` not `lambda` | [flowquake/flow.py](../flowquake/flow.py), `model.py::log_likelihood` | Part I §2, Part IV §9 |
| §2.3 relational conditioning | `SAFE_TOKEN_DIMS`, [flowquake/model.py:32-35](../flowquake/model.py#L32-L35) | Part IV §11 |
| §2.4–2.7 the spatial head | [flowquake/heads.py:55-160](../flowquake/heads.py#L55-L160) | Part IV §10 |
| §2.5 three trigger tiers | [flowquake/data.py:49-117](../flowquake/data.py#L49-L117) | Part IV §7 |
| §2.8 GR half-bin shift | [flowquake/heads.py:170-175](../flowquake/heads.py#L170-L175) | Part IV §10 |
| §2.9 `sigma_min` | [flowquake/flow.py:41-76](../flowquake/flow.py#L41-L76) | Part IV §9 |
| §2.10 background map | [flowquake/data.py:120-151, 255-273](../flowquake/data.py#L120-L273) | Part IV §7 |
| §3 the second model | [flowquake/neural_etas.py](../flowquake/neural_etas.py), `scripts/train_neural_etas.py` | Part V |
| §4 memorization | [runs/ablation_h/](../runs/ablation_h/), `scripts/ablation_h.py`, `scripts/memorization_eval.py` | Part IV §11, Part VIII |
| §5 claim inventory | [results/CLAIMS.md](../results/CLAIMS.md), [WORKING.md](../WORKING.md) | Part IX |
| §6 the two families | [runs/stats_hardening.json](../runs/stats_hardening.json), `scripts/stats_hardening.py` | Part VII |
| §7 out-of-time | [runs/total_win.json](../runs/total_win.json), `scripts/build_comcat_forward.py` | Part VIII |
| §8 transfer | [runs/multiregion_master.json](../runs/multiregion_master.json), `runs/neural_etas/spatial_transfer_summary.json` | Part V, Part VIII |
| §10.11–10.12 CSEP | `runs/{csep_h2h_etas,csep_h2h_fq,n1_density/csep_head}/csep_results.json` | Part VI |
| §11 the ladder | [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md), [runs/replacement_readiness.json](../runs/replacement_readiness.json) | Part IX |

---

## 14. Common misconceptions

**1. "FlowQuake beats ETAS."**
Actually: no single trained model in this repository beats ETAS on total
likelihood. The production model beats it temporally and loses spatially by
0.31–0.53 nats/event on all five California catalogs. The total win belongs to a
*composite* of two separately trained models, one of which starts from ETAS.
*Why it matters:* this is the first sentence an examiner will test, and the
one-model version is false.

**2. "The Mamba encoder is what makes it work."**
Actually: `h_bottleneck = 0` in every production config, so the encoder is not
even constructed ([flowquake/model.py:76-84](../flowquake/model.py#L76-L84)) — the
whole trained model is 29,540 parameters (§3.1). The
"whole catalog" is seen through hand-built exponentially-spaced lag features in
`data.py`, and (for the spatial head) through precomputed full-history ETAS sums.
The SSM exists to *demonstrate a failure mode*.
*Why it matters:* the README's headline still leads with "Selective-SSM
(Mamba-style) whole-catalog encoder", which reads as the contribution.

**3. "The model contains no absolute geography, so it cannot memorize."**
Actually: the *learned conditioning* excludes absolute coordinates. The model
still uses a train-era-fitted smoothed-seismicity background map, which is
absolute geography, and the neural-ETAS head uses the target region's ETAS
inversion. The claim is about a channel, not about the whole model.
*Why it matters:* the strong version is false and easy to falsify by opening
`data.py`.

**4. "Early stopping would fix the memorization."**
Actually: for every `h > 0` the best held-out checkpoint is step 250, the first
one evaluated, and its test `nll` (8.06–8.22) is already worse than `h = 0`'s
converged 7.62. There is no interior optimum to stop at.
*Why it matters:* it is the first fix anyone proposes, and the artifact refutes
it in one column.

**5. "The `h_noise = 0.3` regularizer limits how much the channel can leak."**
Actually: a fixed-variance additive noise on the *output* of an unconstrained
linear projection imposes no capacity bound, because the encoder can scale the
signal. §12.3 shows that an RMS of 0.585 per dimension is already enough capacity
for the whole 3.14-nat memorization budget. The noise is also train-time only.
*Why it matters:* this is the mechanism question, and the repo's own config
comment gets it right in seven words.

**6. "CSEP consistency means FlowQuake forecasts better."**
Actually: N/S/M are *consistency* tests. Passing means not detectably
miscalibrated. A climatology passes. And the paired S-test against ETAS
(p = 1.00 on 10 discordant days, OR interval [0.23, 4.35]) is a non-detection at
very low power, not an equivalence result.
*Why it matters:* "statistically indistinguishable from the incumbent" sounds
like a win and is not one.

**7. "The 2020–2026 result is a prospective forecast."**
Actually: it is a retrospective out-of-time replication. The models were frozen,
but the events existed and were publicly known during development. The artifact's
own `notes` field says so.
*Why it matters:* calling it prospective is the single fastest way to lose
credibility with a seismologist.

**8. "The spatial gain is the neural part of the neural-ETAS head."**
Actually: on ComCat, the background-only ablation delivers 85.5% of the gain and
a classical SGD refit delivers 94%. In Japan and Chile, background-only *exceeds*
the full head. See §10.2.
*Why it matters:* it is the paper's most attackable technical claim and the
artifacts do not support the manuscript's word "consistent".

**9. "Transfer means you can deploy in a new region with no local data."**
Actually: zero-shot spatial transfer still consumes the *target* region's ETAS
inversion and its precomputed triggering sums (the transfer summary's own
`note`), plus per-region normalization and a train-era background map. What
transfers is ~1.3k learned parameters (1,258; §3.1).
*Why it matters:* the deployment claim is the most commercially attractive and
the most easily overstated.

**10. "The pooled global model is the foundation model."**
Actually: the pooled global checkpoint saw every region's training window, so it
is not held-out transfer. The LOO checkpoints are.
`REPLACEMENT_READINESS.md` and `REPRODUCE.md` both say this explicitly, and its
readiness check is a WARN.
*Why it matters:* "one checkpoint, all regions" and "generalizes to unseen
regions" are different claims and only the first is supported.

---

## 15. Questions a professor will ask

**Q1. State the claim in one sentence, with its qualifier.**
FlowQuake beats region-fitted ETAS on temporal log-likelihood on all five
California EarthquakeNPP catalogs in 3-seed means (four of five
block-bootstrap-significant, one tie), and a composite of its temporal head with
a separate full-history spatial head — initialized from each region's own ETAS
inversion — beats ETAS on total likelihood in six regions; it is not an
inversion-free replacement for ETAS.

**Q2. Why model the density of the gap instead of the intensity?**
Because the likelihood of a point process requires the compensator
`Lambda(t) = integral lambda`, which has no closed form for a general neural
`lambda`. Modelling `f(tau|H)` gives exact likelihood with no integral over event
time. The cost: no `lambda` for free, so simulation must sample `tau` and any
instantaneous-rate statement needs `f/(1−F)`. The two are equivalent via
`lambda = f/S`.

**Q3. Your temporal likelihood needs an ODE. How do I know 0.05 nats is not
integration error?**
Two run directories share a configuration and seed and were scored at 64 and 96
RK4 steps: `runs/final_s1555/eval_test.json` and
`runs/comcat25_s1555/eval_test.json` (their `config.yaml`s differ only in
`out_dir` and an explicitly written `mix_hidden: 64`, which is the default). The
float64 paired mean gain differs by 5.6e-9 nats/event — nearly seven orders of
magnitude below the smallest reported effect. (The `tll` agreement is float32 and
proves less.) Separately, I concede an inconsistency: the in-window test used 64
steps and the forward window 96, because `train.py --eval-after` hard-codes 64
while `flowquake.evaluate` defaults to the 96 that `README.md` documents.

**Q4. Derive the normalizer of your spatial kernel and tell me why not a
Gaussian.**
`integral (1 + r²/d²)^(−q) 2πr dr = πd²/(q−1)` by `u = r²/d²`, so the density is
`(q−1)/(πd²)(1+r²/d²)^(−q)` for `q > 1`; `q_floor = 1.15` enforces that. Not a
Gaussian because aftershock distances are power-law: an event at `10σ` costs 50
nats under a Gaussian and ~8 nats here, and mean per-event log-likelihood is
dominated by exactly those tail events.

**Q5. Why is the anisotropy area-preserving, and why does your magnitude head add
0.005 to `m − m_c`?**
Axes `(dρ, d/ρ)` give `det M = d⁴`, so `sqrt(det M) = d²` independent of `ρ` and
the isotropic normalizer still applies with no extra Jacobian — free elongation
along fault strike, verified by grid integration in `tests/test_heads.py`. The
`+0.005` is a half-bin correction for the 0.01-magnitude reporting grid, under
the *dequantization* convention the config states (true magnitude uniform on
`[m, m+Δ)`): the likelihood is `(F(m+Δ) − F(m))/Δ`, and
`log((1 − e^{−βΔ})/Δ) = log β − βΔ/2 + O((βΔ)²)`, so evaluating the continuous
density at `dm + Δ/2` recovers it to first order. At `β ≈ 2` the shift is 0.01
nats. The *nearest*-rounding convention gives the same `+Δ/2`, provided you also
move the truncation to `m_c − Δ/2` — that is the classical Utsu/Bender
correction ([Ch. 2 §5.4](02-seismology.md#54-the-binning-correction-utsu--bender)),
and §2.8 above records that an earlier version of this chapter dropped the
threshold re-anchoring and wrongly concluded the shift was zero. I would flag
three things unprompted: `STACK.md` misstates the grid as 0.1; `0.005` is a
*hardcoded literal* shared by all eleven catalogs while ISC/INGV are certainly on
a 0.1 grid, so **it is wrong for some catalog whichever way ComCat's precision
resolves** ([Ch. 2 §5.5](02-seismology.md#55-the-0005-in-headspy--what-it-corrects-and-the-doc-bug));
and `mll` is not part of the benchmark's `nll = −(tll + sll)`, so no reported
total moves — but simulated magnitudes and hence the CSEP M-test do.

**Q6. What exactly does "memorization" mean here, mechanically?**
The SSM encoder consumes absolute `x, y`, so its state is a near-injective index
into "which stretch of this catalog am I in". `h_proj` compresses that to `h`
dims and hands it to the heads. The spatial head then places narrow, confident
mass on the epicentres that occurred in that stretch. Train `sll` reaches −7.27;
test `sll` collapses to −13.47, which is only 0.31 nats better than a uniform
region. On held-out data the position code indexes nothing.

**Q7. Why did the 0.3-sigma noise not stop it?**
Because a fixed-variance additive channel bounds nothing when the signal power is
free. Capacity is `(h/2) ln(1 + P/σ²)`; solving for the 3.14 nats/event of train
gain gives `P = 0.342`, i.e. an RMS of 0.585 per dimension. `h_proj` is an
unconstrained `nn.Linear` and weight decay 0.03 is no match for a three-nat
likelihood gradient. The noise is also train-time only. The fix is to bound `P`
— LayerNorm/`tanh` on `h`, or signal-proportional noise — and that experiment has
not been run.

**Q8. Could you have early-stopped your way out?**
No. For `h ∈ {4, 16, 64}` the best held-out checkpoint is step 250, the first
validation ever run, and its test `nll` is 8.06–8.22 against `h = 0`'s converged
7.62. There is no interior optimum. It also dates the pathology to
initialization: the fingerprint is the lowest-loss direction available from step
one.

**Q9 (hostile). Your spatial head is initialized from the very ETAS inversion
you claim to beat, and consumes its precomputed triggering sums. Is this not
circular?**
The framing is fair and I concede it in the paper's own words: this upgrades a
deployed ETAS system rather than replacing one. Two things stop it being
*circular*. First, the nesting is exact: gate closed, the head reproduces the
package's per-event `sll` to 1.77e-9 nats, so the measured gain is a gain over
ETAS by construction. Second, gains are always scored against the package's ETAS
output, never against the near-ETAS initialization. What would settle it is
training the head from a neutral, inversion-free initialization and re-measuring
against package ETAS. That experiment is not in the repository, and until it is,
"upgrade" is the only word I will use.

**Q10 (hostile). Your spatial win looks like it is just replacing a uniform
background with smoothed seismicity, which people have done since the 1990s.
Does the `--no-mlp` ablation settle it?**
It settles it against me, more than the manuscript admits. On ComCat,
background-only gives `dS` 0.0513 of 0.0600 — 85.5% — and a classical SGD refit
of ETAS's global parameters gives 0.0564, 94%. Worse, in Japan (0.0556 vs 0.0525)
and Chile (0.0351 vs 0.0267) the background-only head *beats* the full head, so
`MANUSCRIPT.md:559-560`'s "smaller, consistent increment" is not supported —
and that is not recorded in `results/CLAIMS.md`. No committed artifact gives a
paired CI on the increment, which is the statistic that would decide it. The
defensible claim is that the causal multi-scale smoothed-seismicity background is
the result and that the neural modulation is unproven. The experiments that would
resolve it are a paired bootstrap on `sll_full − sll_bg_only` across all six
regions and three seeds, and a full flETAS baseline with a free background.

**Q11 (hostile). You mix a temporal score from one model with a spatial score
from another. Why is that not a metric hack?**
Because the product `f_t^A · f_s^B · f_m^A` is a well-defined normalized model —
it can be simulated from and CSEP-tested, and it was: N 95/100, S 79/85, M 90/92
at a matched budget on the same 100 days. But three of your objections stand.
Neither head was model-selected on the composite; the temporal head's early
stopping used the *production* spatial head, not the reported one; and no jointly
trained model exists. I would not defend the composite as the final artifact —
I would defend it as a decomposition result, and point to joint training as the
next experiment.

**Q12 (hostile). Three of your six regions with a significant total win have a
non-significant, *negative* temporal effect. Is the "temporal win" claim not
region-specific in a way the abstract hides?**
Yes, and I would report both families side by side, not in separate sections.
Holm-adjusted temporal p: California 0.003, Italy 0.003, Chile 0.036 significant;
Japan 0.274, Iran 0.273, Greece 0.648 not. Total: all six ≤ 0.0185. Japan, Greece
and Iran have negative temporal point estimates (−0.0139, −0.0125, −0.0634). The
six-for-six total win rests substantially — in those three regions, essentially
entirely — on the spatial head. The paper's own thesis explains why (the temporal
edge is density-dependent and fades at m_c 4.0), but the abstract's ordering does
not make the dependence obvious. On the related question of *why six Holm members
and not twelve*: two families were declared, answering different questions, and
the artifact's `notes` field pre-states the family as "one headline `dT` claim per
region" — the right form of protection, but living in an artifact rather than a
pre-registration. Under a single twelve-member family Iran's total (`p_holm`
0.0185) is the row most at risk, and I would report both.

**Q13 (hostile). Your headline statistics rest on one training seed.**
Correct for `stats_hardening.json`, and worse than the paper says. The
`total_with_head_family` numbers are seed-0-only on the spatial side (M2), and
`HEADLINE` in `scripts/stats_hardening.py` is one temporal CSV per region, so
both families use one temporal seed. Note also that the HEAD commit changed the
script to average over head seeds, but the committed JSON has no `head_seeds` or
`single_seed_warning` key — it predates its own generating code. Re-running moves
Chile's total from +0.0608 to about +0.064. The `fullsuite_summary.json` temporal
table *is* genuinely 3-seed and verified to recompute from its 15 per-seed files.

**Q14. Justify `mean_block = 50` for the stationary bootstrap.**
I cannot justify it from evidence, only from code: it is the default at
`flowquake/stats.py:45` and no call site overrides it. No result JSON records it
(N11), so a reviewer working from the evidence pack cannot confirm the resampling
scheme. Aftershock sequences run far longer than 50 events, and too short a block
biases CIs narrow — the direction that manufactures significance. The fix is a
sensitivity table at block lengths 10/50/200/1000, which is minutes of CPU on
per-event CSVs that are not committed.

**Q15. What does CSEP consistency buy you?**
A guard, not a claim. N/S/M test whether a forecast is detectably miscalibrated
in number, space and magnitude; a climatology passes them. Their role here is to
rule out the failure mode where a likelihood gain comes from a density that is
systematically wrong about counts. FlowQuake is, uniquely among the benchmark's
generative NPPs, consistent on the magnitude test, which I attribute to the
history-conditional `beta`.

**Q16. Your paired S-test gives p = 1.00. What is your power?**
Essentially none. 83 shared days, 77 passes each, 10 discordant split 5–5. The
Clopper–Pearson 95% interval for 5/10 is [0.187, 0.813], odds ratio [0.23, 4.35]
— the data are consistent with ETAS being four times better or four times worse.
The most extreme split would give p = 0.00195, so the test can only detect near
total dominance. I would call this a non-degradation result and report the odds
ratio interval, never the p-value.

**Q17. Italy is your biggest total gain. Is it real?**
It is real on the native catalogue and does not survive M_w homogenization. Under
`mw_uniform_mc26` the temporal effect inverts to −0.2532 [−0.2885, −0.2205] and
`dTot` to −0.4622; the density-matched M_L m_c-2.8 control is a tie (+0.0022).
The repo's own interpretation attributes the erosion to catalog density rather
than magnitude type. `README.md` already flags Italy as "native-catalogue scale,
not claimed under Mw homogenization" and I would not quote +0.21 without that
clause.

**Q18. Salton Sea is your largest California gain. Is that a swarm artifact?**
Partly, and that is the thesis rather than a defect: FlowQuake wins where ETAS's
parametric Omori form is wrong (swarms) and ties where it is right (Japan's
Tohoku-dominated m_c-4.0 set). What I cannot show is a sequence-type
stratification — nothing in the repository separates swarm from mainshock-
aftershock behaviour. The experiment is to declare swarms by a nearest-neighbour
statistic and report `dT` inside and outside them across all five California
catalogs.

**Q19. Can I reproduce your headline number from this repository?**
No, and I will not pretend otherwise. `reference/` is not committed and all 123
tracked YAMLs point into it. No checkpoint is tracked. Per-event CSVs are
gitignored, so every bootstrap CI is a stored summary. Six ETAS configs behind
the foreign-region baselines are authored by no script here. The `etas` fork is
unpinned.
What *is* reproducible on a fresh clone is the 16-test suite — the chunked scan
against an fp64 naive recurrence, flow `log_prob` against an analytic Gaussian, grid
integration of the anisotropic mixture to 1 — plus the trace in
`results/CLAIMS.md`, which for each reported number names either the committed
JSON key that backs it or the fact that nothing does (13 rows are `NO ARTIFACT`).

**Q20. What is the one experiment you would run next?**
The paired-increment bootstrap on the `--no-mlp` ablation, six regions, three
seeds. It costs CPU-minutes and it decides whether the "neural" in
"neural-ETAS" is load-bearing. Everything else — joint training, neutral
initialization, a prospective registration — is more important for the
*deployment* claim, but that one decides a scientific claim currently made on
one region's point estimate.

---

## 16. Further reading

1. **Ogata (1988), *JASA* 83, 9–27** — "Statistical models for earthquake
   occurrences and residual analysis for point processes". The temporal ETAS
   paper and the origin of residual analysis via time rescaling. Read it for what
   "beating ETAS" is beating.
2. **Ogata (1998), *Ann. Inst. Statist. Math.* 50(2), 379–402** — "Space–time
   point-process models for earthquake occurrences"
   (doi:10.1023/A:1003403601725). Space–time ETAS with the anisotropic spatial
   kernel; the direct ancestor of `flowquake/neural_etas.py`'s functional form.
3. **Daley & Vere-Jones, *An Introduction to the Theory of Point Processes*,
   Springer** — the reference for conditional intensity, compensators, and the
   likelihood identities used in §2.1. Volume I is the one you want.
4. **Stockman, Lawson & Werner (2026), *EarthquakeNPP: A benchmark for earthquake
   forecasting with neural point processes*, TMLR; arXiv:2410.08226** (venue and
   identifier as `MANUSCRIPT.md`'s bibliography records them) —
   the benchmark. Establishes that none of five NPPs beat ETAS and that ETAS wins
   spatial likelihood against all of them. Every number in this repository is
   measured inside its protocol.
5. **Zechar, Gerstenberger & Rhoades (2010), *BSSA* 100(3), 1184–1195** —
   "Likelihood-based tests for evaluating space–rate–magnitude earthquake
   forecasts" (doi:10.1785/0120090192). The S- and M-tests, and the basis of the
   pyCSEP path used in `MANUSCRIPT.md` §4.2.
6. **Politis & Romano (1994), *JASA*** — "The stationary bootstrap". The exact
   resampling scheme in `flowquake/stats.py`, including why geometric block
   lengths preserve stationarity. Read it before defending §10.7.
7. **Holm (1979), *Scandinavian Journal of Statistics*** — the step-down
   procedure behind `family_dT_holm` and `total_with_head_family`. Short, and it
   makes the family-definition issue in §6 obvious.
8. **Lipman, Chen, Ben-Hamu, Nickel & Le (ICLR 2023), *Flow Matching for
   Generative Modeling*; Liu, Gong & Liu (ICLR 2023), *Rectified flow*** — the
   temporal head's training objective and the straight-path construction the
   `sigma_min` schedule modifies.
9. **Zhang, Bengio, Hardt, Recht & Vinyals (ICLR 2017), *Understanding deep
   learning requires rethinking generalization*; Carlini, Liu, Erlingsson,
   Kos & Song (USENIX Security 2019), *The Secret Sharer*** — the memorization
   literature §4.5 situates the `h`-sweep in. The second gives you "exposure",
   which is the right metric for the un-committed re-identification probe.
10. **Cover & Thomas, *Elements of Information Theory*, ch. 9** — the Gaussian
    channel capacity `(1/2) ln(1 + P/σ²)` used in §4.3 and §12.3. One page, and
    it converts the memorization argument from a story into a calculation.
11. **[NOVELTY.md](../NOVELTY.md)** — the repository's own adversarial prior-art
    sweep, with the SafeNet / NMRP / RECAST / FERN differentiations and the
    framing guardrail. Read it before you claim any "first"; it records the
    memorization sub-claim as unconfirmed either way.
