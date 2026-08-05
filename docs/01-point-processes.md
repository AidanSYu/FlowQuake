# Point processes from first principles

The theory that everything else in this repository stands on. Written for someone
with strong maths and strong ML who has never seen a point process, and who needs
to survive a viva on it.

[STACK.md](../STACK.md) is the code walkthrough — it *assumes* this chapter. This
chapter derives what STACK.md states. Where the two disagree, or where the repo
does something the theory says you should not, I say so.

---

## What this chapter buys you

After reading this you should be able to:

- **Write down the conditional intensity `lambda(t | H_t)` from its defining limit**,
  say precisely what `H_t` is as a filtration, and explain why the *compensator*
  `Lambda(t)` — not `lambda` — is the object the theory is actually built on.
- **Derive the point-process log-likelihood twice**, once by bin-partitioning
  `[0, T]` and taking a limit, once by chaining conditional waiting-time densities,
  and prove the two are the same number. This is the single fact that licenses
  FlowQuake's entire architecture.
- **Derive and use the time-rescaling theorem**, run Ogata's residual analysis on a
  sequence by hand, and explain why FlowQuake does *not* report this diagnostic
  (it doesn't; that is a real gap, see [§5.5](#55-the-repo-does-not-do-this)).
- **Derive Ogata's thinning algorithm and prove it correct**, derive the
  inverse-compensator method, and explain exactly which of them FlowQuake avoids
  and what it pays for the privilege.
- **Derive `n = integral of the kernel`, `E[cluster size] = 1/(1 - n)` and the
  stationary rate `mu/(1 - n)`** for a Hawkes process, from both the branching
  representation and the intensity representation, and say what happens at
  `n >= 1`.
- **State exactly what conditional independence the mark factorization
  `f_t · f_s · f_m` assumes**, and name two concrete seismological situations where
  it is false.

---

## Prerequisites

- Undergraduate probability: densities, conditional expectation, the exponential
  distribution, the probability integral transform.
- Enough measure theory to be comfortable with the *words* sigma-algebra,
  filtration, adapted, and predictable. You do not need to prove anything
  measure-theoretic here; you need to know what the words are guarding against.
- Calculus: you will integrate `exp` and power laws, and differentiate under an
  integral once.
- Optional but useful: [STACK.md, Part 0](../STACK.md#part-0--orientation) for
  orientation on what FlowQuake is. You do **not** need any seismology yet —
  earthquakes enter this chapter only as examples. The seismology is
  [Chapter 2](02-seismology.md); its point-process instantiation is
  [Chapter 3 — ETAS](03-etas.md).

Notation is the shared notation from the primer's front matter. Two collisions you
must hold in your head, because both are inherited and neither is mine to rename:

- The repo's ETAS parameter set calls the Omori taper timescale `tau`, and the
  inter-event gap is *also* called `tau`. **Throughout this chapter `tau` is always
  an inter-event gap; the taper is written `tau_tap`.**
- `n` is the standard symbol for *both* the number of observed events (§1, §4) and
  the Hawkes branching ratio (§8). Context always disambiguates — the branching
  ratio is a dimensionless number in `[0, 1)`, the count is an integer in the
  thousands — but where the two could be confused I write "the branching ratio `n`"
  or spell the count out in words.

---

## 1. The object: counting processes, histories, and simplicity

### 1.1 A realization

A **temporal point process** on `[0, T]` is a random, locally finite set of points
`0 <= t_1 <= t_2 <= ... <= t_n <= T`. "Locally finite" means every bounded interval
holds finitely many points — no infinitely many events crammed into a second. An
earthquake catalog is literally this: a sorted list of origin times, plus (later)
marks; [flowquake/data.py:200-205](../flowquake/data.py#L200-L205) is the CSV sorted by
`time` and converted to `t_days`, days since the first event.

Two equivalent descriptions of the same random object. Be fluent in both, because
the likelihood proof of [§4](#4-the-likelihood-derived-twice) moves between them:

| representation | the random variables | natural for |
|---|---|---|
| **counting** | `N(t)` = number of points in `[0, t]` | intensities, martingales, compensators |
| **interval** | `n`, and the gaps `tau_i = t_i - t_{i-1}` (with `t_0 = 0`) | densities, autoregressive models, FlowQuake |

FlowQuake lives entirely in the interval representation; ETAS lives entirely in the
counting one.

### 1.2 The counting process

`N(t) = #{ i : t_i <= t }` is right-continuous, non-decreasing, integer-valued, with
`N(0) = 0`, jumping by `+1` at each `t_i` for a **simple** process (defined below).
Knowing `N(.)` as a function *is* knowing the point set: `N` is not a summary, it is
a re-encoding.

### 1.3 History: what `H_t` formally means

Everything conditional in this subject conditions on `H_t`. Informally: "everything
that has happened strictly before `t`". Formally:

```
H_t  =  sigma( N(u) : u < t )      (the internal / natural filtration)
```

the sigma-algebra generated by the whole path of the counting process on `[0, t)`.
Three properties matter and you should be able to say why:

1. **It is a filtration**: `s < t` implies `H_s` is a subset of `H_t`. Information
   only accumulates. Nothing in this subject lets you forget.
2. **Strictly before, not up to and including.** `H_t` is *left-continuous* — it
   excludes the event at `t` itself. If it did not, `lambda(t | H_t)` would be
   allowed to look at whether an event happens at `t` and "predict" it perfectly.
   The technical name for a process that is measurable with respect to the
   left-continuous filtration is **predictable**, and predictability is exactly the
   formalisation of "no peeking".
3. **It can be enlarged.** Nothing forces `H_t` to be the *internal* history. You
   may condition on covariates: GPS strain, a previous catalog, another region.
   The theory is unchanged as long as the enlarged filtration still makes `N`
   adapted and `lambda` predictable. This is the licence under which FlowQuake
   conditions on 30 engineered relational features rather than the raw point set —
   they are `H_t`-measurable functions of the past *events*, so the conditioning
   sigma-algebra is a sub-sigma-algebra of the internal history and the likelihood
   theory below applies verbatim (see [Q17](#q17-your-model-conditions-on-30-engineered-features-not-the-raw-history-does-the-likelihood-theory-still-hold)).

> **The one line to remember.** A conditional-intensity model is a *forecasting
> rule* — a map from the past to an instantaneous rate. Everything else is
> bookkeeping about how to score that rule.

### 1.4 Simple vs non-simple, and the repo's tie problem

A point process is **simple** if, with probability 1, all points are distinct:
`t_i < t_{i+1}` strictly, so `N` only ever jumps by `+1`. A related condition,
**orderliness**, says

```
P( N(t + dt) - N(t) >= 2 )  =  o(dt)     as dt -> 0
```

i.e. double events in a shrinking window are negligible at first order. For simple
processes with a conditional intensity these are equivalent for our purposes.

Simplicity matters enormously in [§4](#4-the-likelihood-derived-twice): the
bin-partition derivation needs "each small bin holds 0 or 1 event", which is
exactly orderliness.

**And real catalogs are not simple.** Earthquake origin times are reported at
finite precision, and genuinely coincident or near-coincident events happen. The
repo confronts this in two visible places:

- [data.py:206-210](../flowquake/data.py#L206-L210) computes `tau = np.diff(t_days)`
  then `np.clip(tau, TAU_FLOOR_DAYS, None)` with
  `TAU_FLOOR_DAYS = 1e-7` days (`1e-7 * 86400 = 8.64` ms; the source comment on
  [data.py:25](../flowquake/data.py#L25) rounds this to "~9 ms"). The comment on that line records that
  *"catalog's smallest nonzero gap is ~5e-8 d"* — i.e. the floor **binds**, on at
  least some events.
- [runs/total_win.json](../runs/total_win.json) reports
  `"pairing_key": "time+duplicate_rank"`. You only need a duplicate rank if
  duplicate timestamps exist.

So the process the model is fit to is a *floored* version of the catalog. That is
a modelling decision with a measurable consequence on the headline number, and it
is the first thing I would attack if I were examining this work — see
[Q13](#q13-hostile-the-tau-floor) for the arithmetic and the honest answer.

---

## 2. The conditional intensity

### 2.1 Definition as a limit

```
                            P( N(t + dt) - N(t) = 1 | H_t )
lambda(t | H_t)  =   lim   ---------------------------------
                    dt->0                dt
```

Units: events per unit time (here, per day). Three readings, all correct:

- **Rate**: the instantaneous expected number of events per unit time, given
  everything so far.
- **Hazard**: the instantaneous risk of an event *right now*, given none has
  happened since the last one. ([§3](#3-the-hazard-view) makes this precise — and
  it is a theorem, not a definition.)
- **Predictable projection of `dN`**: `E[ dN(t) | H_t ] = lambda(t | H_t) dt`.
  This is the version that generalises and the version [§8.5](#85-the-stationary-rate)
  needs.

For a simple process, by orderliness, `P(>= 2 events in dt) = o(dt)`, so
`E[dN(t) | H_t] = 1 * lambda dt + o(dt)` and all three readings coincide.

Two canonical instances, to have concrete objects in hand:

```
homogeneous Poisson :  lambda(t | H_t) = lambda_0            (history-blind)
Hawkes              :  lambda(t | H_t) = mu + sum_{t_j < t} g(t - t_j)
```

### 2.2 Existence and uniqueness

Two statements, both of which you should be able to quote and neither of which I
prove here.

**(a) Uniqueness.** *If two simple point processes on `[0, T]` admit conditional
intensities and those intensities agree almost everywhere (with respect to the
same filtration), the processes have the same law.* This is the workhorse: it means
"I have specified `lambda`" is a complete specification of the model, and it is the
engine of the thinning-correctness proof in [§6.2](#62-ogatas-thinning-algorithm).
*We do not prove this here*; see Daley & Vere-Jones, *An Introduction to the Theory
of Point Processes* (Springer; Vol. I, 2nd ed. 2003), Ch. 7, and Rasmussen (2018)
for a readable statement.

**(b) Existence.** Not every point process has a conditional intensity. The
compensator (next) always exists under mild conditions; the intensity exists only
when the compensator is *absolutely continuous* in `t`. A process whose compensator
has jumps — e.g. one that fires at a deterministic time with probability 1/2 — has
no intensity. Nothing in seismology behaves like that, so this is a footnote, but a
professor may ask whether you know `lambda` is a convenience rather than a
primitive. It is.

### 2.3 The compensator and the Doob–Meyer decomposition

Define the **compensator**

```
Lambda(t)  =  integral_0^t  lambda(u | H_u) du
```

Note carefully: the integrand is evaluated *along the realized path*. `lambda(u|H_u)`
depends on the events that actually occurred before `u`. `Lambda` is therefore a
random, path-dependent, non-decreasing process — not a deterministic function.

Now the statement that justifies calling it a compensator.

> **Doob–Meyer, specialised to counting processes.** Let `N` be a counting process
> adapted to `{H_t}` with `E[N(t)] < infinity` for all `t`. Then `N` is a
> submartingale, and there exists a unique **predictable**, non-decreasing process
> `Lambda` with `Lambda(0) = 0` such that
> ```
> M(t)  =  N(t) - Lambda(t)
> ```
> is a martingale with respect to `{H_t}`.
>
> *We do not prove this here.* It is the general Doob–Meyer decomposition (Doob
> 1953; Meyer 1962–63) applied to a class-(D) submartingale. See Daley &
> Vere-Jones Vol. II (2008) Ch. 14, or Brémaud, *Point Processes and Queues* (1981).

**Why the compensator is the right object**, in four steps — say this out loud:

1. `N(t)` never decreases, so it is a submartingale: `E[N(t)|H_s] >= N(s)`. It has a
   systematic upward drift, and that drift is the signal, not noise.
2. Doob–Meyer peels that drift off **uniquely**, provided you insist it be
   *predictable*. Without predictability the decomposition is not unique — you could
   hide information about the jump at `t` inside the "drift" and still make the
   residual a martingale. Predictability formalises "forecast before the event".
3. What is left, `M = N - Lambda`, is a martingale, so
   `E[N(t)-N(s) | H_s] = E[Lambda(t)-Lambda(s) | H_s]`: **the compensator is the best
   predictable forecast of the number of events yet to come.** That is exactly what
   a forecaster wants, and exactly what CSEP's N-test measures
   ([STACK.md, Part VI](../STACK.md#part-vi--generative-evaluation-simulation-and-csep)).
4. `lambda` is merely the density of `Lambda` when one exists. The intensity is
   derived; the compensator is primitive — and the likelihood
   ([§4](#4-the-likelihood-derived-twice)), time rescaling ([§5](#5-time-rescaling))
   and inverse simulation ([§6](#6-simulation)) are all statements about `Lambda`.

**Immediate consequence.** `E[N(T)] = E[Lambda(T)]`. For homogeneous Poisson,
`Lambda(T) = lambda_0 T` is deterministic. For a Hawkes, `Lambda` is random and
correlated with `N`, so the count distribution is over-dispersed relative to
Poisson — which is not a bug but the entire point, and is what lets a Hawkes model
pass an N-test on a day containing an aftershock sequence.

---

## 3. The hazard view

This derives the single most-used formula in neural TPPs. It is elementary once one
subtle step is made explicit — and that step is the one textbooks skip.

### 3.1 Setup

Fix the history up to and including event `i-1` at `t_{i-1}`; write `H` for
`H_{t_{i-1}+}`. Let `T_i` be the next event time and `tau = T_i - t_{i-1}` the gap.
Define

```
S(tau)  =  P( T_i > t_{i-1} + tau | H )          survivor
F(tau)  =  1 - S(tau)                            cdf
f(tau)  =  F'(tau) = -S'(tau)                    density
h(tau)  =  f(tau) / S(tau)                       hazard
```

with `S(0) = 1`. All four are conditional on `H`; I suppress it for readability and
restore it when it matters.

### 3.2 The subtle step: the hazard *is* the intensity

Claim:

```
h(tau)  =  lambda( t_{i-1} + tau | H_{t_{i-1}+tau} )   evaluated on the event
                                                       { no points in (t_{i-1}, t_{i-1}+tau] }
```

Why. Start from the hazard's own limit definition:

```
h(tau)  =  lim_{dt->0} (1/dt) P( T_i in [t_{i-1}+tau, t_{i-1}+tau+dt) | T_i >= t_{i-1}+tau, H )
```

Now look at the conditioning event `{ T_i >= t_{i-1}+tau } and H`. Because `T_i` is
the *next* event time, this event says: the history is `H`, **and nothing has
happened since**. But that is a complete description of `H_{t_{i-1}+tau}` — the
filtration at time `t_{i-1}+tau` contains exactly the points of `H` and no others.
The two conditioning sigma-algebras coincide on this event. Therefore

```
h(tau)  =  lim_{dt->0} (1/dt) P( N(t+dt) - N(t) = 1 | H_t )  with t = t_{i-1}+tau
        =  lambda( t_{i-1} + tau | H_t )
```

**This step is where the whole subject lives.** The conditional intensity looks like
it needs a full filtration, but between two events the filtration is frozen — the
only new information is "still nothing" — so `lambda` restricted to an inter-event
interval is an ordinary deterministic hazard function of elapsed time. That is why
the intensity representation and the interval representation are interchangeable,
and it is the reason a neural net that maps "the past 30 features" to "a density
over `tau`" is not cutting a corner.

The step *fails* if the filtration is enlarged with covariates that evolve during
the gap (e.g. a strain-meter reading that updates while you wait). Then `H_t` is
strictly richer than `H` plus "nothing happened", and `f(tau | H)` is a marginal
over the covariate path, not a hazard. FlowQuake's features are all functions of
past *events*, so it is safe. Say this if you are asked whether the derivation
survives external covariates — the honest answer is "not without an extra
integral".

### 3.3 From `S' = -h S` to the density

Rearranging the definition of the hazard:

```
h(tau) = f(tau)/S(tau) = -S'(tau)/S(tau) = - d/dtau [ log S(tau) ]
```

so, provided `S(u) > 0` on `[0, tau]`, integrate both sides from `0` to `tau`:

```
- [ log S(tau) - log S(0) ]  =  integral_0^tau h(u) du
```

With `S(0) = 1`, `log S(0) = 0`:

```
S(tau)  =  exp( - integral_0^tau h(u) du )
```

and therefore

```
f(tau | H)  =  h(tau) S(tau)
            =  lambda(t_{i-1} + tau | H) · exp( - integral_0^tau lambda(t_{i-1} + u | H) du )
```

which is the formula quoted in [STACK.md §2](../STACK.md#2-the-likelihood-and-the-choice-that-shapes-this-repo).
Read it as **(rate now) × (probability of having survived to now)**.

### 3.4 Two conditions people forget

**(a) Non-defectiveness.** `S(tau) -> 0` iff `integral_0^inf lambda du = infinity`.
If the intensity decays fast enough that its integral converges (say
`lambda(u) = e^{-u}`) then `S(infinity) = e^{-1} > 0`: with positive probability
**there is no next event, ever**, and the waiting-time "density" is defective. A
Hawkes process with constant background `mu > 0` cannot do this
(`integral >= mu*tau -> infinity`) — one of several quiet jobs the background does.

**(b) The behaviour at `tau = 0`.** If `lambda` blows up as `tau -> 0` (Omori with
`c -> 0` does), `f(tau)` has an integrable singularity at the origin. Still a
density, but numerically it is what the `log tau` parameterisation and the floor of
[Q13](#q13-hostile-the-tau-floor) are wrestling with.

---

## 4. The likelihood, derived twice

This is the most important section in the chapter.

### 4.1 Derivation A: infinitesimal bins

Partition `[0, T]` into `K` bins of width `Delta = T/K`, with bin `k` being
`B_k = [(k-1)Delta, k Delta)`. Let `u_k = (k-1)Delta` be its left endpoint. Take `K`
large enough that (by orderliness) each bin contains 0 or 1 event; the probability
of a bin with 2+ events is `o(Delta)` per bin, hence `o(1)` in total.

The data determine, for each bin, whether it is occupied. Because the bins are
processed left to right and each conditioning uses only earlier bins, the joint
probability factorises exactly as a product of conditionals:

```
P(data)  ~=  prod_{k : occupied}  lambda(u_k | H_{u_k}) Delta
             ×
             prod_{k : empty}     ( 1 - lambda(u_k | H_{u_k}) Delta )
```

Take logs:

```
log P  ~=  sum_{k occupied} [ log lambda(u_k) + log Delta ]
           + sum_{k empty} log( 1 - lambda(u_k) Delta )
```

Handle the two sums separately.

**Empty bins.** Use `log(1 - z) = -z + O(z^2)`:

```
sum_{k empty} log(1 - lambda(u_k) Delta)
   =  - sum_{k empty} lambda(u_k) Delta  +  O( K Delta^2 )
   =  - sum_{k=1}^{K} lambda(u_k) Delta  +  sum_{k occupied} lambda(u_k) Delta  +  O(T Delta)
```

The first term is a Riemann sum for `integral_0^T lambda(u | H_u) du`. The second is
a sum over exactly `n` bins of terms of size `lambda * Delta`, which is `O(n Delta)
-> 0`. The error term is `O(T Delta) -> 0`. So

```
sum_{k empty} log(1 - lambda Delta)  ->  - integral_0^T lambda(u | H_u) du
```

**Occupied bins.** There are exactly `n` of them, and `u_k -> t_i` as `Delta -> 0`:

```
sum_{k occupied} log lambda(u_k)  ->  sum_{i=1}^{n} log lambda(t_i | H_{t_i})
```

and the leftover `n log Delta` is the **volume element**. This is the honest place
to be careful: `P(data)` is a probability of a discretised observation, and it goes
to zero like `Delta^n`. Dividing by `Delta^n` — equivalently, subtracting
`n log Delta` — converts it into a *density* with respect to the natural reference
measure on point configurations (counting measure on `n`, times Lebesgue measure on
the ordered `n`-tuple of times). That density is the likelihood:

```
                 n
log L  =   sum  log lambda(t_i | H_{t_i})   -   integral_0^T lambda(u | H_u) du
                i=1
```

**Read the two terms.** The first rewards putting rate where events actually were.
The second is the *only* thing stopping you from setting `lambda = infinity`
everywhere: it is the normaliser. Everything hard about likelihood-based
point-process fitting is that second term, because `lambda` depends on the history,
so the integral must be taken along the realized path.

Sanity check on the homogeneous Poisson: `log L = n log lambda_0 - lambda_0 T`,
maximised at `lambda_0 = n/T`. Correct.

### 4.2 Derivation B: chaining conditional densities

Alternatively, describe the same data as: gap `tau_1`, then gap `tau_2`, ..., then
gap `tau_n`, then *no further event in the remaining time* `T - t_n`. By the chain
rule of probability, with `t_0 = 0`:

```
              n
log L  =   sum  log f(tau_i | H_{t_{i-1}})   +   log S(T - t_n | H_{t_n})
             i=1
```

The last term is the **right-censoring** term: you observed the interval `[t_n, T]`
and saw nothing, and that is informative. Dropping it is not "an approximation", it
is answering a different question.

### 4.3 Proof that A and B are the same object

Substitute the [§3.3](#33-from-s---h-s-to-the-density) formula into B. For each `i`,

```
log f(tau_i | H_{t_{i-1}})
   =  log lambda(t_i | H_{t_i})  -  integral_{t_{i-1}}^{t_i} lambda(u | H_u) du
```

and

```
log S(T - t_n | H_{t_n})  =  - integral_{t_n}^{T} lambda(u | H_u) du
```

Sum over `i = 1..n` and add the censoring term:

```
              n                          n
log L  =   sum  log lambda(t_i)   -   sum  integral_{t_{i-1}}^{t_i} lambda du
             i=1                        i=1
                                     -   integral_{t_n}^{T} lambda du
```

The integrals **telescope**: the intervals `[t_0, t_1], [t_1, t_2], ..., [t_{n-1},
t_n], [t_n, T]` partition `[0, T]` exactly, and `lambda(u|H_u)` is the same function
of `u` in each (the history changes at the boundaries, but that is already baked
into `H_u`). Therefore

```
   sum_i integral_{t_{i-1}}^{t_i} lambda du  +  integral_{t_n}^{T} lambda du
      =  integral_0^T lambda(u | H_u) du
```

and B reduces to A. QED. There is no approximation and no condition beyond the
existence of `lambda`.

**The one-line moral, and the design decision it licenses:**

> A point-process likelihood can be written as a sum of ordinary 1-D conditional
> densities over waiting times. You never have to compute `integral lambda` if you
> can model `f(tau | H)` directly and correctly normalised.

That is why [flowquake/flow.py](../flowquake/flow.py) is a **density model on
`log tau`**, not an intensity model. A normalizing flow gives you an exactly
normalised 1-D density for free (its normalisation is the change-of-variables
Jacobian, which is exact); by the theorem above, an exactly normalised `f(tau|H)`
is an exactly normalised point-process likelihood. See
[STACK.md §2](../STACK.md#2-the-likelihood-and-the-choice-that-shapes-this-repo) and
[§9](../STACK.md#9-flowpy--the-temporal-head) for the mechanics.

The trade you are making, stated plainly:

| | model `lambda` (ETAS) | model `f(tau | H)` (FlowQuake) |
|---|---|---|
| normalisation | needs `integral lambda`; forces kernels whose 1-D integral is cheap (closed form, or a tabulated quadrature) | free and exact from the flow |
| interpretability | additive triggering decomposition | none; the density is a black box |
| evaluate rate at arbitrary `t` | yes, `lambda(t)` on a grid | **no** — only event-by-event |
| simulate | thinning or inverse-compensator | direct sampling ([§6.4](#64-what-flowquake-does)) |
| residual analysis | time rescaling is immediate | needs `Lambda`, which you no longer have ([§5.5](#55-the-repo-does-not-do-this)) |

### 4.4 What FlowQuake actually reports (and the term it drops)

Be precise, because this is checkable and a professor will check it.
[flowquake/model.py:219-243](../flowquake/model.py#L219-L243) computes a **per-event**
`tll = log f_t(tau_i)` for every test-window target, and
[flowquake/evaluate.py:98-101](../flowquake/evaluate.py#L98-L101) reports the **mean**:

```python
"tll": float(tll.mean()),
"sll": float(sll.mean()),
"mll": float(mll.mean()),
"nll": float(-(tll.mean() + sll.mean())),
```

There is **no censoring term**. What is reported is `(1/n) * sum_i log f(tau_i | H)`,
not `(1/n) * log L`. Three things follow:

1. That is fine *for a paired comparison* — the EarthquakeNPP baseline scores ETAS
   the same way, per event, from `augmented_catalog.csv`
   ([evaluate.py:28-51](../flowquake/evaluate.py#L28-L51)). Both sides drop the same
   single boundary term.
2. It is **not** the log-likelihood of the test window, and you should not call it
   that. The missing term is one number, `log S(T - t_n)`, over 21,889 events
   ([runs/comcat25/eval_test.json](../runs/comcat25/eval_test.json), `n_events`), so
   its per-event contribution is `O(1/n)` and negligible — but "negligible" is a
   quantitative claim, and the honest phrasing is "mean per-event predictive
   log-density".
3. It means the reported score is a **prequential / one-step-ahead predictive
   score**, which is arguably the *more* appropriate object for forecast evaluation
   than a retrospective full-window likelihood.
   [Chapter 6](06-evaluation-and-csep.md) develops this.

---

## 5. Time rescaling

### 5.1 Statement

> **Time-rescaling (random time change) theorem.** Let `N` be a simple point process
> on `[0, infinity)` with conditional intensity `lambda(t | H_t)` and compensator
> `Lambda(t) = integral_0^t lambda du`. Suppose `Lambda` is continuous and strictly
> increasing with `Lambda(infinity) = infinity`. Define
> ```
> t*_i  =  Lambda(t_i)
> ```
> Then `{t*_1, t*_2, ...}` is a realization of a **homogeneous Poisson process of
> unit rate**.

Equivalently and more usefully in practice: the **rescaled gaps**

```
xi_i  =  Lambda(t_i) - Lambda(t_{i-1})  =  integral_{t_{i-1}}^{t_i} lambda(u | H_u) du
```

are i.i.d. `Exponential(1)`.

Attribution: the general martingale form is due to Meyer (1971, "Démonstration
simplifiée d'un théorème de Knight", *Séminaire de Probabilités V*, Lecture Notes in
Mathematics 191, 191–195) and Papangelou (1972, "Integrability of expected
increments of point processes and a related random change of scale", *Trans. Amer.
Math. Soc.* 165, 483–506). The version everyone in applied work actually cites is
Brown, Barbieri, Ventura, Kass & Frank (2002), "The time-rescaling theorem and its
application to neural spike train data analysis", *Neural Computation* 14(2),
325–346, which gives an elementary proof and the goodness-of-fit recipe. Ogata
(1988, *JASA* 83(401), 9–27) introduced the corresponding residual analysis for
seismicity.

### 5.2 Derivation

I give the elementary interval-representation proof; it is complete for processes
with an intensity, and it is the one you can reproduce on a board.

Fix `i` and condition on `H_{t_{i-1}}`. By [§3.2](#32-the-subtle-step-the-hazard-is-the-intensity),
on the interval `(t_{i-1}, t_i)` the intensity acts as an ordinary hazard function
of elapsed time; write `h(u) = lambda(t_{i-1} + u | H_{t_{i-1}})` and
`Hc(v) = integral_0^v h(u) du` for its cumulative hazard. Note `xi_i = Hc(tau_i)`.

`Hc` is continuous and strictly increasing (since `h > 0`) with `Hc(infinity) =
infinity` (non-explosion, [§3.4](#34-two-conditions-people-forget)), so it maps
`(0, infinity)` bijectively onto `(0, infinity)` and `Hc^{-1}(z)` exists for every
`z > 0`. Then:

```
P( xi_i > z | H_{t_{i-1}} )   =   P( Hc(tau_i) > z )
                              =   P( tau_i > Hc^{-1}(z) )
                              =   S( Hc^{-1}(z) )                        (definition of S)
                              =   exp( - Hc( Hc^{-1}(z) ) )              (by §3.3)
                              =   exp( - z )
```

So `xi_i | H_{t_{i-1}} ~ Exponential(1)`. **The conditional law does not depend on
`H_{t_{i-1}}` at all.** Therefore `xi_i` is independent of `H_{t_{i-1}}`, hence of
`xi_1, ..., xi_{i-1}` (which are `H_{t_{i-1}}`-measurable). By induction the `xi_i`
are i.i.d. `Exp(1)`. A point process whose gaps are i.i.d. `Exp(1)` is a unit-rate
Poisson process. QED.

Three observations worth making unprompted:

- This is **the probability integral transform**, applied one gap at a time:
  `F(tau_i | H) = 1 - exp(-xi_i) ~ Uniform(0,1)`, i.i.d. That is the same trick as
  the PIT / calibration diagnostics used everywhere in probabilistic forecasting.
- The rescaled gaps are *exactly* the compensator increments — which are *exactly*
  the terms that appear in `log f(tau_i) = log lambda(t_i) - xi_i`. Look back at
  [§4.3](#43-proof-that-a-and-b-are-the-same-object): the numbers you compute for
  the likelihood **are** the residuals. Nothing extra is needed.
- Non-explosion (`Lambda(infinity) = infinity`) is required, and is exactly the
  non-defectiveness condition from [§3.4](#34-two-conditions-people-forget).

### 5.3 Ogata's residual analysis

The recipe:

1. Fit the model, obtaining `lambda_hat`.
2. Compute `Lambda_hat(t_i)` for every event; form `xi_i` (the rescaled gaps).
3. Test `H_0: xi_i` i.i.d. `Exp(1)`. Standard batteries:
   - **KS test** on `xi_i` against `Exp(1)`, or equivalently on
     `u_i = 1 - exp(-xi_i)` against `Uniform(0,1)`.
   - **Q–Q plot** of sorted `xi` against `Exp(1)` quantiles, with the
     Kolmogorov confidence band. Bends at the top-right diagnose a mis-modelled
     tail — for seismicity, usually the Omori decay.
   - **Serial-correlation / Ljung–Box on `u_i`**, or the "`u_i` vs `u_{i+1}`
     scatter" that Brown et al. recommend. KS is blind to ordering; independence
     is half the null and must be tested separately.
   - **Cumulative-count plot**: `N` against `Lambda_hat(t)` should be the 45° line.
     Systematic departures localise *when* the model is wrong, which a KS `p`-value
     cannot.

This is *the* standard TPP diagnostic, and [§5.2](#52-derivation) says why: it
converts an arbitrary, history-dependent, non-stationary model into a single fixed
null — `Exp(1)` — for which exact tests exist. It is the closest thing the subject
has to a model-agnostic goodness-of-fit tool for the ground process.

### 5.4 The three caveats

1. **In-sample fitting breaks the null.** If `lambda_hat` was fit on the same data
   the `xi_i` are not exactly `Exp(1)` under `H_0` and the KS `p` is
   anti-conservative. Fix: rescale on a held-out window (a benchmark split gives you
   this free), or bootstrap the null Lilliefors-style.
2. **KS is weak against local misfit.** Be wrong on 2% of the catalog (the first hour
   after each M6) and you still pass. Look at the plots, not the `p`.
3. **Marks are not tested.** Time rescaling checks the *ground process* only; a model
   can pass perfectly and mis-locate every event. That is what the CSEP S- and
   M-tests are for ([STACK.md, Part VI](../STACK.md#part-vi--generative-evaluation-simulation-and-csep)).

### 5.5 The repo does not do this

**Honest gap.** Grep `flowquake/` and `scripts/` for `rescal`, `kstest`,
`ks_1samp`, `Kolmogorov`: **zero hits.** (`scipy` is imported 16 times across the
tree — spatial trees, filters, quadrature, optimisers — and `scipy.stats` exactly
zero times.) **There is no time-rescaling residual analysis, no KS test on
rescaled gaps, and no Q–Q plot anywhere in the repository.** Be precise about the
scope of that claim, because two nearby things *do* exist and a professor who greps
will find them: the word "residual" appears in the tree in unrelated senses (SSM
residual connections, projection residuals, "residual deficit" in
[MANUSCRIPT.md](../MANUSCRIPT.md)); and the ETAS compensator *is* computed — see the
next bullet.

Why not, and how bad is it:

- **Why not, for FlowQuake**: it never forms `lambda`, so it never forms `Lambda`.
  Getting `xi_i` from an `f(tau)`-model requires the *other* route —
  `xi_i = -log S(tau_i | H) = -log(1 - F(tau_i | H))` — and `F` is a 1-D integral of
  the flow density with no closed form. It is entirely computable (the flow is 1-D;
  a fixed quadrature over `log tau` would do it), just not implemented.
- **Why not, for ETAS — and this one has no excuse.** The ETAS side already computes
  the rescaled gaps and then throws them away.
  [scripts/etas_forward_eval.py:126-127](../scripts/etas_forward_eval.py#L126-L127)
  is literally
  ```python
  int_lam = np.ediff1d(cum, to_begin=cum[0] - anchor)
  TLL = np.log(lam_star) - int_lam
  ```
  i.e. `TLL_i = log lambda(t_i) - xi_i` — exactly [§4.3](#43-proof-that-a-and-b-are-the-same-object),
  with `int_lam` *being* the vector of `xi_i`. A KS test against `Exp(1)` on
  `int_lam` is one added line. (Note in passing what this also shows: in the repo's
  own ETAS evaluator, per-event `TLL` *is* `log f(tau_i | H)` — the same object
  FlowQuake's `tll` is — so
  [§4.4](#44-what-flowquake-actually-reports-and-the-term-it-drops)'s "both sides
  drop the same boundary term" is verifiable in source for that path. The `baselines`
  block in `eval_test.json` comes instead from EarthquakeNPP's
  `augmented_catalog.csv` under the gitignored `reference/` tree
  ([evaluate.py:117](../flowquake/evaluate.py#L117)), so for *that* path it remains
  an assumption about the benchmark's convention.)
- **How bad**: the repo substitutes CSEP consistency tests, which are a *stronger*
  and more operationally relevant calibration check on counts, locations and
  magnitudes ([STACK.md, Part VI](../STACK.md#part-vi--generative-evaluation-simulation-and-csep)).
  But CSEP tests are daily-aggregated; time rescaling is per-event and would
  localise temporal misfit at the scale where FlowQuake claims its win. Absent it,
  the temporal claim rests entirely on a likelihood difference and not at all on a
  calibration diagnostic.
- **What I would do**: compute `u_i = F_hat(tau_i | H)` for the 21,889 test events by
  quadrature against the flow, and report a KS test plus a `u_i`-vs-`u_{i+1}`
  independence check, side by side with the same quantities for ETAS. It is a
  half-day of work — the ETAS half is one line, per the previous bullet — and it
  would materially strengthen the temporal claim of
  [§4.4](#44-what-flowquake-actually-reports-and-the-term-it-drops). Say this if
  asked; it is a better answer than defending the omission.

---

## 6. Simulation

Three standard algorithms, and then what FlowQuake actually does.

### 6.1 The inverse method (inverse compensator)

Direct corollary of [§5](#5-time-rescaling), run backwards. Given the history up to
`t_{i-1}`:

```
1.  draw  xi ~ Exponential(1)
2.  solve  integral_{t_{i-1}}^{t_i} lambda(u | H_u) du  =  xi   for t_i
3.  append t_i, repeat
```

**Correctness** is immediate: by [§5.2](#52-derivation) a correctly simulated
process has `Exp(1)` compensator increments, and `tau -> Hc(tau)` is a bijection, so
drawing `xi` and inverting gives a gap with exactly the law `f(tau|H)`. **Cost**: you
need `Hc` and must invert it. For the classical Omori–Utsu kernel `(u+c)^{-p}` that
is elementary — which is *why* that kernel is convenient — and the inversion is a
1-D root find. Exact, no rejection, no wasted draws.

**Do not overstate this.** The kernel this repository's ETAS actually uses carries an
exponential taper, `exp(-u/tau_tap) (u+c)^{-(1+omega)}`, whose integral is **not**
elementary: [scripts/etas_forward_eval.py:57-64](../scripts/etas_forward_eval.py#L57-L64)
builds `H` by `scipy.integrate.quad` on a 6,000-point log mesh and then interpolates.
So the honest statement of the ETAS advantage is not "closed-form compensator" but
"a *one-dimensional, history-independent* compensator that can be tabulated once and
reused" — cheap, but numerical. [Chapter 3 §5.4](03-etas.md) says the same thing.

### 6.2 Ogata's thinning algorithm

Due to Ogata (1981), "On Lewis' simulation method for point processes", *IEEE
Trans. Inform. Theory* 27(1), building on the nonhomogeneous-Poisson thinning
of Lewis & Shedler (1979).

```
given history H_t at current time t:
  1.  find lambda_bar >= lambda(u | H_u) for all u in [t, t+L],
      valid as long as no new point is accepted
  2.  draw  w ~ Exponential(lambda_bar);  set  s = t + w
  3.  draw  U ~ Uniform(0,1)
  4.  if  U <= lambda(s | H_s) / lambda_bar :   accept s as a point;  t <- s
      else:                                     reject;               t <- s
  5.  repeat
```

**Correctness.** Consider the process of *accepted* points. Its conditional
intensity at time `s`, given the accepted history `H_s`, is

```
lambda_accepted(s | H_s)
   =  (rate of candidates at s) × P(accept | candidate at s, H_s)
   =  lambda_bar × [ lambda(s | H_s) / lambda_bar ]
   =  lambda(s | H_s)
```

The candidate process is Poisson at rate `lambda_bar`, so its intensity is
`lambda_bar` regardless of history. The retention probability is **predictable** —
it depends only on `H_s`, i.e. on accepted points strictly before `s` — which is
what makes the product legitimate as a conditional intensity. By the uniqueness
theorem of [§2.2](#22-existence-and-uniqueness), a simple point process with
conditional intensity `lambda(s|H_s)` has the target law. QED.

**Where the dominating rate comes from, for Hawkes.** Between events the intensity
`mu + sum_j g(t - t_j)` is non-increasing whenever `g` is non-increasing (true for
exponential and for Omori). So `lambda_bar = lambda(t+ | H_t)` — the intensity
immediately after the last accepted point — dominates until the next acceptance. It
must be recomputed after every acceptance, because acceptances make `lambda` jump
up. This is exactly what the code in [§12](#12-worked-example-b--time-rescaling)
does.

**Efficiency.** The acceptance rate is `E[lambda]/lambda_bar`. For a
near-critical Hawkes (`n` close to 1) with a spiky kernel, `lambda_bar` right after
a large event is enormous while the average intensity is small, so you can waste
90%+ of draws. This is why the inverse method is preferred when `Hc` is available.

### 6.3 Cluster (branching) simulation

For Hawkes specifically, the cleanest simulator uses the branching representation of
[§8.2](#82-the-branching-representation) directly:

```
1.  simulate immigrants as a homogeneous Poisson(mu) process on [0, T]
2.  for each existing event at time t_j (immigrant or offspring):
       draw its number of direct offspring  ~ Poisson(n),   n = integral g
       place each offspring at  t_j + v,  with  v ~ g(.)/n   (the normalised kernel)
3.  recurse until no new events; discard events past T
```

This terminates almost surely iff `n < 1` ([§8.4](#84-stability-cluster-size-and-what-happens-at-n--1)).
It is exact, embarrassingly parallel, and gives you the parent labels for free —
which is why it is what ETAS EM fitting is built around. Its weakness is edge
effects: clusters seeded before `t = 0` contribute to `[0, T]` and must be handled
by burn-in or by an explicit stationary initialisation.

### 6.4 What FlowQuake does

**It uses none of the three.** [flowquake/ntest.py](../flowquake/ntest.py)
`simulate_day_events` and [model.py:247-269](../flowquake/model.py#L247-L269)
`sample_next` do this instead:

```
1.  build the conditioning vector from the lane's history
2.  sample u ~ p_flow(. | cond)          [integrate the flow ODE forward from N(0,1)]
3.  tau = exp( u * log_tau_std + log_tau_mean )          [model.py:257]
4.  sample the marks from the closed-form heads
5.  push (tau, x, y, m) into the lane's history buffers, rebuild the token, repeat
```

Step 2–3 is a **direct draw from `f(tau | H)`**. No `lambda`, no `Lambda`, no
dominating rate, no rejection, no root find.

**Why that is a real advantage.**

- **No wasted work.** One ODE solve per event, always. Thinning's acceptance rate
  problem simply does not exist. This is what makes a 10^3–10^4-catalog CSEP run
  feasible ([STACK.md, Part VI](../STACK.md#part-vi--generative-evaluation-simulation-and-csep)).
- **No integrability constraint on the model.** The inverse method forces you to
  choose kernels whose 1-D integral you can compute cheaply and repeatedly — closed
  form, or (as here) a precomputed quadrature table
  ([§6.1](#61-the-inverse-method-inverse-compensator)). That constraint is precisely
  what ETAS is paying to keep, and it is one of the two structural weaknesses
  FlowQuake targets.
- **Sampling and scoring use the same object.** In an intensity model, the
  likelihood path (`integral lambda`) and the simulation path (thinning) are
  different code with different failure modes. Here both go through the flow, so a
  bug in one shows up in the other.

**What it costs.**

- **No rate field.** You cannot ask "what is `lambda` at 3pm next Tuesday at this
  location". Every forecast quantity must be obtained by *simulating forward*, which
  is why the CSEP machinery in this repo is a Monte-Carlo catalog simulator rather
  than a gridded rate. That is expensive and it is stochastic.
- **No compensator, hence no time-rescaling diagnostic** ([§5.5](#55-the-repo-does-not-do-this)).
- **No closed-form truncated conditional** — which brings us to the interesting bug
  that isn't.

### 6.5 The truncated first event, and why the naive version biases every forecast day

This is [ntest.py:88-104](../flowquake/ntest.py#L88-L104), and it is the subtlest
correct thing in the repo.

**The situation.** You are forecasting the 1-day window `[d, d+1)`. The last
*observed* catalog event is at `t_last`, with `t_last < d`. Let

```
a  =  d - t_last     (the observed quiet interval, "backward recurrence time")
```

You know something the model's unconditional `f(tau)` does not: **no event occurred
in `(t_last, d)`**. That is data.

**The correct law.** The first simulated event's gap must be drawn from the
*truncated* conditional

```
                      f(tau | H)
f(tau | H, tau > a) = ----------- ,     tau > a
                        S(a | H)
```

which is a perfectly ordinary conditional density (and note that by
[§3.3](#33-from-s---h-s-to-the-density), this is the same thing as saying "restart
the hazard clock at `a`").

**Why FlowQuake cannot just compute it.** `S(a | H) = 1 - F(a | H)` requires the cdf
of the flow, which has no closed form ([§6.4](#64-what-flowquake-does)). So the code
uses the one method that needs only samples:

```python
need = active & (t_last + tau < day_start_days)
for _r in range(MAX_REJECTION_ROUNDS):          # 200, ntest.py:28
    if not need.any(): break
    tau2, x2, y2, m2 = model.sample_next(...)
    take = need & (t_last + tau2 >= day_start_days)
    tau = torch.where(take, tau2, tau)          # ... and x, y, m
    need = need & ~take
active = active & ~need   # lanes that never accepted: no event today
```

**Correctness of rejection sampling for a truncation.** Draw `tau ~ f`, accept iff
`tau > a`. The accepted draw has density

```
              f(tau) 1{tau > a}          f(tau)
p_acc(tau) = -------------------  =  -----------  on tau > a
              P(tau > a)               S(a)
```

which is exactly the target. Acceptance probability per round is `S(a)`; the number
of rounds is `Geometric(S(a))`. No tuning, no proposal, exact.

**The bias if you skip it.** Suppose you sample `tau ~ f` unconditionally and keep
it. With probability `F(a) = 1 - S(a)` the first "simulated" event lands *before*
`d` — in a window you have already observed to be empty. Trace the code: the live
test at [ntest.py:107](../flowquake/ntest.py#L107) is `live = active & (t_next <
day_end)`, which such an event passes trivially. So it is **recorded as an event in
the forecast day**, and worse, the chain then continues from a time earlier than
`d`, giving the lane extra time to accumulate more events.

Leading-order size, under a Poisson approximation to make the arithmetic checkable
(this is my estimate, not a repo number). If the local rate is `lambda`, the backward
recurrence time `a` measured back from a fixed clock time is `Exponential(lambda)`,
so the *average* rejection probability is

```
E[ F(a) ]  =  E[ 1 - e^{-lambda a} ]  =  1 - lambda/(lambda + lambda)  =  1/2
```

(using `E[e^{-s a}] = lambda/(lambda + s)` for `a ~ Exp(lambda)`, at `s = lambda`)

— note this is **not** `1 - e^{-lambda E[a]} = 1 - e^{-1} = 0.632`; plugging the
mean of `a` into a concave function is the wrong order of operations, and the
correct average is exactly `1/2`. ComCat_25's test window has 21,889 events over
4,764 days ([runs/comcat25/eval_test.json](../runs/comcat25/eval_test.json),
[runs/comcat25/config.yaml](../runs/comcat25/config.yaml)), i.e. `~4.59` events/day.
So the naive scheme would add roughly `0.5` spurious events per simulated lane-day
on a mean of `~4.6`, before knock-on excitation — a systematic **over-forecast of
order 10%**, on every single day. The N-test's `delta_2 = P(N_sim <= N_obs)` would collapse
toward 0 and the model would fail consistency everywhere. STACK.md's "skip this and
you systematically over-forecast every day's first event" is correct, and this is
its magnitude.

**The residual bias that remains.** A lane that fails all 200 rounds is declared to
have no event that day ([ntest.py:103](../flowquake/ntest.py#L103)). That happens with
probability `(1 - S(a))^200`. If `S(a) = 0.02` (a long observed quiet gap relative
to the model's expectations), that is `0.98^200 = exp(-4.04) = 1.8%` of lanes
silently zeroed — a small *downward* count bias, concentrated on exactly the days
where the model is most surprised by the quiet. It is small, it is in the
conservative direction for an over-forecasting concern, and it is undocumented. A
professor who reads the code will find it. Be ready to say: *"yes, `(1-S(a))^200`
lanes are dropped; on a quiet day with `S(a)=0.02` that is 1.8%; the fix is to
sample the truncated conditional by inverting a quadrature-based cdf instead of
rejecting."*

---

## 7. A ladder of processes

Each of these isolates one property, which is why they are the standard contrasts.

| process | `lambda(t | H_t)` | what the past does | gaps |
|---|---|---|---|
| **homogeneous Poisson** | `lambda_0` | nothing | i.i.d. `Exp(lambda_0)` |
| **inhomogeneous Poisson** | `lambda_0(t)` | nothing (but clock time matters) | independent, not identical |
| **renewal** | `h(t - t_{N(t)})` | only the time since the *last* event | i.i.d. with density `f` |
| **self-correcting** | `exp(mu t - sum_{t_j<t} alpha)` | each event *lowers* the rate | under-dispersed, regular |
| **Hawkes (self-exciting)** | `mu + sum_{t_j<t} g(t-t_j)` | each event *raises* the rate | over-dispersed, clustered |

Reading the ladder:

- **Poisson** is the null. Complete independence; `N(t) - N(s) ~ Poisson(integral)`;
  memoryless gaps. In the repo it is the floor: ComCat_25 Poisson baseline
  `tll = 0.5126`, `sll = -13.7745`, `nll = 13.2619`
  ([runs/comcat25/eval_test.json](../runs/comcat25/eval_test.json), `baselines.Poisson`).
- **Inhomogeneous Poisson** adds clock-time structure but *still* no dependence on
  what happened. This is the crucial contrast for ML people: a big flexible neural
  net that predicts a rate as a function of `t` alone is still a Poisson process. It
  cannot represent "an aftershock sequence started". Test: if a model's `lambda`
  does not change when you delete a past event, it is Poisson.
- **Renewal** adds *one bit* of memory — time since the last event. This is where
  most naive "predict the next gap from the last gap" models live. It cannot
  represent an Omori tail, because Omori says an event from 400 days ago still
  matters and a renewal process has forgotten it.
- **Self-correcting** (Isham & Westcott 1979, *Stochastic Processes and their
  Applications* 8(3), 335–347) is the physical opposite: stress builds linearly and each earthquake
  releases it. This is the "characteristic earthquake / seismic gap" intuition and
  it produces *regular*, anti-clustered sequences. Real catalogs are emphatically
  clustered, which is a first-order empirical fact and the reason ETAS won this
  argument in seismology.
- **Hawkes** is the one that matches. And what it adds over renewal is precisely:
  **the whole history, additively, with a decaying kernel**.

Two takeaways for a viva. **(1)** "Self-exciting" is about *sign and memory length*,
not nonlinearity: Hawkes is linear in the past counting measure, and its clustering
comes from the feedback loop, not curvature. **(2)** The Omori tail is why truncated
history is fatal — `g(u) ~ u^{-p}` with `p ~= 1` has a barely-convergent integral
tail, so a large fraction of a big event's total triggering happens long after the
last 20 events have gone past. That is exactly the criticism
[STACK.md §6](../STACK.md#6-why-etas-is-hard-to-beat) levels at the published NPP
baselines.

---

## 8. Hawkes processes

### 8.1 Definition

```
lambda(t | H_t)  =  mu  +  sum_{t_j < t}  g(t - t_j)
```

with `mu > 0` the **background** (immigration) rate and `g : (0, inf) -> [0, inf)`
the **triggering kernel** (also: excitation kernel, response function). Standard
choices:

```
exponential :   g(u) = alpha * exp(-beta u)                  n = alpha / beta
power law   :   g(u) = k (u + c)^{-(1+omega)}                n = k c^{-omega} / omega
```

The power-law form is the Omori–Utsu law and is what ETAS uses; the repo's exact
triggering weight is transcribed at
[neural_etas.py:78-87](../flowquake/neural_etas.py#L78-L87) — with a taper
`exp(-dt / tau_tap)` that guarantees the integral converges. (Note again: the repo
calls that timescale `tau`; I write `tau_tap`.) The full ETAS treatment is
[Chapter 3](03-etas.md).

### 8.2 The branching representation

**Immigrant–offspring construction:**

- **Immigrants** arrive as a homogeneous Poisson process of rate `mu`. They are
  "spontaneous" — background seismicity, tectonic loading.
- **Offspring**: independently for each event (immigrant or not) at time `t_j`, its
  direct offspring form an *inhomogeneous Poisson process on `(t_j, infinity)` with
  rate `g(t - t_j)`*, independent of everything else.
- The realized process is the superposition of immigrants and all descendants of
  all generations.

**Claim:** this construction and the intensity definition of
[§8.1](#81-definition) give the same law. (Hawkes & Oakes 1974, "A cluster process
representation of a self-exciting process", *J. Appl. Prob.* 11(3), 493–503.)

**Sketch of the equivalence** — I give the argument, not a full proof:

Compute the conditional intensity of the constructed process. Condition on the whole
past `H_t`, which is the set of realized points `{t_j < t}` (note: *not* their family
labels — the observer does not see who triggered whom). By construction the process
is a superposition of:

- the immigrant Poisson process, contributing rate `mu` at `t`;
- for each realized past point `t_j`, an offspring Poisson process contributing
  rate `g(t - t_j)` at `t`.

Superposed independent processes have additive intensities, so the total rate is
`mu + sum_{t_j<t} g(t - t_j)`. The step that needs care is *why conditioning on
`H_t` does not change any of those rates*. It is because each offspring process is
**Poisson given its parent time**, and a Poisson process has independent increments:
knowing which offspring of `t_j` already occurred before `t` tells you nothing about
its rate after `t`. Likewise the unobserved parent labels are integrated out
harmlessly, because the sum runs over all past points regardless of label. Hence the
intensity of the constructed process equals the target, and by uniqueness
([§2.2](#22-existence-and-uniqueness)) the laws agree.

*We do not prove existence, non-explosion, or the uniqueness of the stationary
version here.* Hawkes & Oakes (1974) and Daley & Vere-Jones Vol. I Ch. 6 do.

**Why the branching view earns its keep** even though it is "just" a re-description:
it gives the cluster simulator of [§6.3](#63-cluster-branching-simulation); it gives
**EM fitting** (the E-step assigns each event a posterior probability of being
background vs. triggered by each predecessor — exactly the latent parent label; this
is how the benchmark's ETAS is fitted,
[STACK.md §5](../STACK.md#5-the-exact-etas-used-here)); it gives **declustering**
(separating mainshocks from aftershocks is a statement about the latent tree); and it
gives the stability theory below, which is a branching-process fact, not a
point-process one.

### 8.3 The branching ratio

```
n  =  integral_0^infinity  g(u) du
```

By the offspring construction, the number of *direct* offspring of any one event is
`Poisson(n)`, so `n` is the expected number of direct children per event. It is
dimensionless. It is also called the *criticality parameter* or *reproduction
number* — it is exactly the `R_0` of epidemiology, and Hawkes processes are the same
mathematics as epidemic branching.

Check on the exponential kernel: `integral_0^inf alpha e^{-beta u} du = alpha/beta`.

### 8.4 Stability, cluster size, and what happens at `n >= 1`

**Expected cluster size.** Take one immigrant and count all its descendants,
including itself. Generation 0 has 1 member. Given generation `k` has `Z_k` members,
each independently produces `Poisson(n)` children, so

```
E[Z_{k+1} | Z_k]  =  n Z_k     =>     E[Z_k]  =  n^k
```

Total cluster size `C = sum_{k>=0} Z_k`, so by monotone convergence

```
                inf                    1
E[C]  =  sum   n^k   =   ---------      provided  n < 1
               k=0                     1 - n
```

**Subcriticality.** `n < 1` is exactly the condition for a Galton–Watson branching
process with mean offspring `n` to die out almost surely and have finite expected
total progeny. So:

| regime | branching process | Hawkes process |
|---|---|---|
| `n < 1` **subcritical** | dies out a.s., `E[C] = 1/(1-n)` finite | stationary, finite rate `mu/(1-n)` |
| `n = 1` **critical** | dies out a.s. but `E[C] = infinity` | no stationary distribution; rate drifts, infinite expected cluster |
| `n > 1` **supercritical** | survives forever with positive probability | explodes: infinitely many events in finite time with positive probability |

At `n >= 1` the process is **not stationary and its likelihood is not the likelihood
of a stationary model**. Fitting can still be done on a finite window (the
likelihood of [§4](#4-the-likelihood-derived-twice) never assumed stationarity), but
simulation will not terminate and any statement about long-run rates is meaningless.

**Where this bites in this repository.** [MANUSCRIPT.md:364](../MANUSCRIPT.md) reports
a **branching ratio of 0.968** for the `ComCat_25_refit2020` ETAS inversion. That is
extremely close to critical: `1/(1 - 0.968) = 31.25` events per cluster in
expectation, versus e.g. `1/(1-0.8) = 5` at `n = 0.8`. The derivative of `1/(1-n)`
is `1/(1-n)^2 = 977` at that point, so a `0.01` shift in `n` moves expected cluster
size by ~10 events. Any downstream quantity that depends on `n` is badly
conditioned there.

**And you must flag this number's status.** [results/CLAIMS.md](../results/CLAIMS.md)
rows **N2**, **X8** and **P5** record that the refit parameter vector — including
`branching ratio 0.968` — has **no committed artifact**:
`runs/forward_etas_ComCat_25_refit2020/summary.json` holds only
`window / n / tll / sll / nll / etas_name / fit_window / params_frozen_from` (I
checked; it does), and no ETAS parameter vector is committed anywhere under
`runs/`. So: quote 0.968 as *"the manuscript reports 0.968, from an inversion whose
parameter file lives in the gitignored `reference/` tree"* — never as a verified
number. This is one of the 12 claims WORKING.md itself lists as having no committed
backing.

### 8.5 The stationary rate

Two derivations. Give both; the agreement is the point.

**(a) Via branching.** Immigrants arrive at rate `mu`. Each seeds a cluster of
expected size `1/(1-n)`. Every event in the process belongs to exactly one cluster.
So the total event rate is

```
Lambda_bar  =  mu / (1 - n)
```

**(b) Via the intensity and the compensator.** Assume the process is stationary with
`E[lambda(t)] = Lambda_bar` for all `t`, and extend the process to `(-infinity,
infinity)`. Take expectations of the intensity:

```
E[ lambda(t) ]  =  mu  +  E[ sum_{t_j < t} g(t - t_j) ]
                =  mu  +  E[ integral_{-inf}^{t} g(t - u) dN(u) ]
```

Now use the **compensator/martingale property** — this is where
[§2.3](#23-the-compensator-and-the-doobmeyer-decomposition) earns its keep. Because
`g(t-u)` for `u < t` is deterministic given `t`, and `E[dN(u)] = E[lambda(u)] du`:

```
E[ integral_{-inf}^{t} g(t-u) dN(u) ]  =  integral_{-inf}^{t} g(t-u) E[lambda(u)] du
                                       =  Lambda_bar * integral_0^{inf} g(v) dv
                                       =  Lambda_bar * n
```

(substituting `v = t - u`). So

```
Lambda_bar  =  mu + n Lambda_bar     =>     Lambda_bar (1 - n) = mu
            =>  Lambda_bar  =  mu / (1 - n)          for n < 1
```

Note the two places the derivation announces `n < 1`: the geometric series in (a),
and the sign of `(1-n)` in (b). At `n >= 1` (b) returns a negative or infinite rate,
which is the algebra telling you no stationary solution exists.

Numerical check is in [§12](#12-worked-example-b--time-rescaling): `mu = 0.5`,
`n = 0.4` predicts `0.8333`/day; the simulation of 2,000 days produced 1,686 events,
i.e. `0.843`/day.

---

## 9. Marked point processes

### 9.1 Ground process plus mark kernel

A **marked point process** attaches to each `t_i` a mark `k_i` in a mark space `K`;
here `k = (s, m)` with `s = (x, y)` in km and `m` the magnitude. The **ground
process** `N_g` is the times alone. The full specification factorises as

```
lambda(t, dk | H_t)  =  lambda_g(t | H_t) · f(k | t, H_t) dk
```

- `lambda_g(t | H_t)` — the ground intensity: *when* does something happen.
- `f(k | t, H_t)` — the mark kernel: *given* something happens at `t`, what is it.
  A probability density on `K` for each `(t, H_t)`.

Everything in [§3](#3-the-hazard-view)–[§6](#6-simulation) applies unchanged to
`lambda_g`; the marks come along for the ride. The likelihood picks up one extra
term per event:

```
              n                              n
log L  =   sum log lambda_g(t_i | H_{t_i})  + sum log f(k_i | t_i, H_{t_i})  -  integral_0^T lambda_g du
             i=1                              i=1
```

(There is no extra integral: the mark density integrates to 1 by construction, so it
contributes nothing to the compensator.)

### 9.2 The chain rule and what FlowQuake assumes

Write the joint density of the next event's full description given the history. The
exact chain rule is:

```
f(tau, s, m | H)  =  f_t(tau | H) · f_s(s | tau, H) · f_m(m | tau, s, H)
```

This is an identity. No assumption yet. What FlowQuake writes
([model.py:1-16](../flowquake/model.py#L1-L16),
[STACK.md §3](../STACK.md#3-marks-factorization-and-the-three-scores)) is

```
f(tau, s, m | H)  =  f_t(tau | H) · f_s(s | H) · f_m(m | H)
```

**Two conditional independences are being assumed:**

1. `s  ⫫  tau  |  H` — where the next event is does not depend on how long you
   waited for it.
2. `m  ⫫  (tau, s)  |  H` — how big the next event is does not depend on how long
   you waited or where it happened.

Note precisely what is *not* assumed. Each head sees the same conditioning vector
`cond` ([model.py:160-171](../flowquake/model.py#L160-L171)). In the committed
ComCat_25 configuration `h_bottleneck: 0`
([runs/comcat25/config.yaml](../runs/comcat25/config.yaml)), so `cond` is exactly
`SAFE_TOKEN_DIMS` — the *previous* event's token minus absolute `x, y`: its
`log tau`, its magnitude, and the 7 lags × 4 relational features, 30 numbers in all.
(With `h_bottleneck > 0` the SSM state is concatenated on; the argument below is
unaffected either way.) So the model absolutely does capture "the last event was an M6 15 minutes
ago, therefore expect small events nearby and soon". What it cannot capture is
coupling between the three components of the **same, next** event.

ETAS makes the same factorization, and it is worth saying why that is defensible:
in ETAS the current event's magnitude affects *future* rate (through the
productivity term `exp(a(m_j - m_c))`) but the current event's own
`(tau, s, m)` are drawn independently given `H`. So this is a fair fight — but "ETAS
does it too" is not a defence of the assumption's *truth*.

### 9.3 Where the factorization is wrong, concretely

Two counterexamples with real seismological content. Have both ready.

**(a) Short-term aftershock incompleteness — violates `m ⫫ tau | H`.**
Immediately after a large earthquake the network physically cannot detect small
events: their waveforms are buried in the mainshock's coda and in each other. The
effective `m_c` is *elevated* for minutes to hours and relaxes back. So **the
observed magnitude distribution depends directly on `tau`**: 30 seconds after an M6
the catalog contains essentially no M2.5s; three days later it does. A factorised
`f_m(m | H)` can partially compensate (it knows the previous event was an M6) but
cannot condition on *this* event's realized `tau`, which is the variable driving
detection. Not hypothetical — it is the subject of Mizrahi, Nandan & Wiemer (2021),
"Embracing data incompleteness for better earthquake forecasting", *JGR Solid Earth*
126(12), by the authors of the very `etas` package this benchmark uses
([MANUSCRIPT.md](../MANUSCRIPT.md) references). The benchmark's flat `m_c` cut (2.5 for
ComCat, [runs/comcat25/config.yaml](../runs/comcat25/config.yaml)) is applied uniformly
in time, so the incompleteness sits in the data for both models.

**(b) Magnitude-dependent location uncertainty — violates `s ⫫ m | H`.**
Catalog locations are *estimates* from arrival-time inversion. An event near `m_c`
is recorded by few stations with poor azimuthal coverage, so its location error is
kilometres and biased; a moderate event is recorded by dozens and located to a few
hundred metres. The **observed** spatial scatter around a source region is therefore
systematically wider for small events — the observed `(s, m)` pair is dependent
through the measurement process, which is all a catalog likelihood can see. A
factorised model cannot say "this event is small, so widen my spatial prediction";
`f(s | m, H)` with a magnitude-dependent width could.

Do not confuse (b) with the *parent*-side effect — rupture length grows with
magnitude, so an M7's aftershocks lie along a line rather than radially about the
hypocentre. That one **is** modelled, by both ETAS and FlowQuake, via
`d_j = d exp(gamma (m_j - m_c))` ([neural_etas.py:82](../flowquake/neural_etas.py#L82)).
Parent-magnitude coupling is handled; *current-event* mark coupling is not.

**How much does it cost?** Unknown, and unmeasured here. The diagnostic is cheap:
bin test events by `tau` and by `m` and check whether residual `sll`/`mll` show
structure. If `mll` degrades systematically at small `tau`, assumption (1) is costing
you. Nothing in `scripts/` does this.

---

## 10. Why log-likelihood

Full treatment is [Chapter 6](06-evaluation-and-csep.md); here is the preview you
need to not be blindsided.

**Log-likelihood is a strictly proper scoring rule.** Let `p*` be the true
conditional density and `q` your model's. Then

```
E_{X ~ p*} [ log q(X) ]  =  E_{p*}[ log p*(X) ]  -  KL( p* || q )
                            \_________________/    \____________/
                              a constant             >= 0, = 0 iff q = p*
```

Maximising expected log-score is exactly minimising KL to the truth, and is uniquely
maximised by reporting your true belief — no hedging can game it. That is what
"strictly proper" means, and it is why this is the benchmark's metric.

**Three consequences for reading this repo's numbers:**

1. **The units are absolute, not relative.** `sll = -8.6898` for ETAS
   ([runs/comcat25/eval_test.json](../runs/comcat25/eval_test.json)) is
   `log(1/km^2)`. Exponentiating, `e^{-8.6898} = 1.683e-4` per km², i.e. as if the
   probability mass were spread uniformly over `1/1.683e-4 = 5,942` km². The
   FlowQuake composite's `sll = -8.6298` ([runs/total_win.json](../runs/total_win.json))
   corresponds to `5,596` km². These are honest densities over California, not
   softmax logits.
2. **Differences are log-likelihood ratios.** `dTot = +0.1133` nats/event
   ([runs/total_win.json](../runs/total_win.json), `test_2007_2020.dTot.mean`) means
   `e^{0.1133} = 1.120`× more probability density on what actually happened, per
   event. Over 21,889 events that is `0.1133 × 21,889 = 2,480` nats of accumulated
   log-likelihood ratio. Those two framings sound wildly different and are the same
   number; be able to move between them.
3. **A proper score does not certify calibration.** A model can win on log-score and
   still be badly calibrated in the tail, or systematically wrong about counts. That
   is exactly why this repo also runs CSEP consistency tests.

**A sanity check you can do in your head, and which I recommend memorising** because
it proves you know what `tll` is. For a homogeneous Poisson at rate `lambda_0`,
`f(tau) = lambda_0 e^{-lambda_0 tau}` and `E[tau] = 1/lambda_0`, so

```
E[ log f(tau) ]  =  log lambda_0  -  lambda_0 E[tau]  =  log lambda_0 - 1
```

The benchmark's Poisson baseline has `tll = 0.5126407`
([runs/comcat25/eval_test.json](../runs/comcat25/eval_test.json)). Inverting:
`lambda_0 = exp(0.5126407 + 1) = 4.5387` events/day. The ComCat_25 test window runs
2007-01-01 to 2020-01-17 ([runs/comcat25/config.yaml](../runs/comcat25/config.yaml)) —
4,764 days — and contains 21,889 events, an empirical rate of **4.5947/day**. The
two agree to 1.2%, and the small shortfall is what you would expect from a baseline
rate fitted on the earlier training window. That is a complete, independent
verification that `tll` is what the docs say it is: a log-density in `log(1/day)`.

Similarly `sll = -13.7745` for the Poisson baseline gives an implied uniform area of
`e^{13.7745} = 959,823` km² — a plausible California bounding box, and consistent
with a genuinely uniform spatial baseline.

---

## 11. Worked example A — a three-event Hawkes, both likelihood forms

**This is the exercise that proves you understand the chapter.** Do it by hand once.

### Setup

Exponential-kernel Hawkes on `[0, T]`:

```
lambda(t)  =  mu  +  sum_{t_j < t}  alpha * exp( -beta (t - t_j) )

mu = 0.5 /day      alpha = 0.8 /day      beta = 2.0 /day      T = 3.0 days
events at   t_1 = 0.5,   t_2 = 0.9,   t_3 = 2.4
```

Branching ratio `n = alpha/beta = 0.4`; expected cluster size `1/(1-0.4) = 1.667`;
stationary rate `mu/(1-n) = 0.8333`/day. All comfortably subcritical.

### Form A: `sum log lambda(t_i) - integral lambda`

**Intensity at each event** (left limits — the event itself does not excite itself):

```
lambda(t_1) = 0.5                                                      = 0.5000000000
lambda(t_2) = 0.5 + 0.8 e^{-2(0.9-0.5)}  = 0.5 + 0.8 e^{-0.8}          = 0.8594631713
lambda(t_3) = 0.5 + 0.8 e^{-2(2.4-0.5)} + 0.8 e^{-2(2.4-0.9)}
            = 0.5 + 0.8 e^{-3.8} + 0.8 e^{-3.0}                        = 0.5577262722
```

Logs: `-0.6931471806`, `-0.1514473041`, `-0.5838869886`. Sum:

```
sum_i log lambda(t_i)  =  -1.4284814732
```

**The integral term, in closed form.** For the exponential kernel the compensator is
elementary:

```
integral_0^T lambda(u) du  =  mu T  +  sum_j  integral_{t_j}^{T} alpha e^{-beta(u - t_j)} du
                           =  mu T  +  (alpha/beta) sum_j [ 1 - e^{-beta (T - t_j)} ]
```

Note each parent contributes at most `alpha/beta = n = 0.4` — the branching ratio is
literally "the compensator mass one event eventually adds". Numerically:

```
mu T                          = 0.5 * 3            = 1.5
(alpha/beta) [1 - e^{-2*2.5}] = 0.4 * 0.993262053  = 0.397304821
(alpha/beta) [1 - e^{-2*2.1}] = 0.4 * 0.985004423  = 0.394001769
(alpha/beta) [1 - e^{-2*0.6}] = 0.4 * 0.698805788  = 0.279522315
                                                     -------------
integral_0^T lambda du                             = 2.5708289057
```

**Result:**

```
log L  =  -1.4284814732  -  2.5708289057  =  -3.9993103789
```

### Form B: `sum log f(tau_i | H) + log S(final gap)`

Now the interval representation. For each `i`, `log f(tau_i) = log lambda(t_i) -
xi_i` where `xi_i` is the compensator increment over `(t_{i-1}, t_i]`:

| `i` | `tau_i` | `xi_i = integral_{t_{i-1}}^{t_i} lambda` | `log lambda(t_i)` | `log f(tau_i)` |
|---|---|---|---|---|
| 1 | 0.5 | 0.2500000000 | −0.6931471806 | **−0.9431471806** |
| 2 | 0.4 | 0.4202684144 | −0.1514473041 | **−0.5717157184** |
| 3 | 1.5 | 1.3008684496 | −0.5838869886 | **−1.8847554382** |

and the censoring term over the final observed-but-empty gap `T - t_3 = 0.6`:

```
log S(0.6 | H_{t_3})  =  - integral_{2.4}^{3.0} lambda du  =  -0.5996920418
```

**Total:**

```
-0.9431471806 - 0.5717157184 - 1.8847554382 - 0.5996920418  =  -3.9993103789
```

**The two forms agree to 4.4e-16** — machine epsilon on the accumulated sums. Check
the telescoping by hand:
`0.2500000000 + 0.4202684144 + 1.3008684496 + 0.5996920418 = 2.5708289058`, which is
the integral term from Form A (`2.5708289057`; the last digit differs only because
the four increments were rounded to 10 decimals before adding). That is [§4.3](#43-proof-that-a-and-b-are-the-same-object)
made arithmetic.

### Reproduce it

```python
import numpy as np
mu, al, be = 0.5, 0.8, 2.0; ts = np.array([0.5, 0.9, 2.4]); T = 3.0
lam  = lambda t: mu + al*np.exp(-be*(t - ts[ts < t])).sum()
A = sum(np.log(lam(t)) for t in ts) - (mu*T + (al/be)*np.sum(1 - np.exp(-be*(T - ts))))
Lam = lambda a, b: mu*(b-a) + sum((al/be)*(np.exp(-be*(max(a,tj)-tj)) - np.exp(-be*(b-tj)))
                                  for tj in ts if tj < b)
B = sum(np.log(lam(t)) - Lam(p, t) for p, t in zip(np.r_[0.0, ts[:-1]], ts)) - Lam(ts[-1], T)
print(A, B, A - B)      # -3.9993103789 -3.9993103789 4.44e-16
```

### Three things to notice

1. **`log f(tau_i) = log lambda(t_i) - xi_i`** exactly. The likelihood term and the
   time-rescaling residual are the same computation. If you can compute one you have
   the other for free — which is precisely why FlowQuake, which computes *neither*
   (it computes `log f` directly from a flow), also has no residuals.
2. **The censoring term is not small here.** `-0.5997` out of `-3.9993` is 15% of
   the total log-likelihood, because there are only 3 events. At 21,889 events it is
   negligible — which is the quantitative justification for
   [§4.4](#44-what-flowquake-actually-reports-and-the-term-it-drops).
3. **`lambda(t_1) = mu` exactly**, because the process starts empty at `t = 0` with
   no pre-history. A real catalog has pre-history, which is exactly what the
   benchmark's *auxiliary window* (1971–1981 for ComCat_25) is for: it supplies the
   history that makes the first scored event's intensity correct without ever being
   a target ([data.py:216-218](../flowquake/data.py#L216-L218)).

---

## 12. Worked example B — time rescaling

Simulate a longer realization of the *same* process by thinning, rescale, and check
the gaps are `Exp(1)`.

### The simulation

Ogata thinning ([§6.2](#62-ogatas-thinning-algorithm)), using the fact that between
events the Hawkes intensity is non-increasing, so `lambda_bar = lambda(t)` dominates
until the next acceptance:

```python
def sim(T, rng, mu=0.5, al=0.8, be=2.0):
    ts, t = [], 0.0
    while True:
        arr = np.array(ts)
        lbar = mu + al*np.sum(np.exp(-be*(t - arr))) if ts else mu   # dominating rate
        t += rng.exponential(1.0/lbar)                               # candidate
        if t > T: break
        lt = mu + al*np.sum(np.exp(-be*(t - arr))) if ts else mu     # true rate
        if rng.random() <= lt/lbar: ts.append(t)                     # accept/reject
    return np.array(ts)
```

With `rng = np.random.default_rng(20260801)` and `T = 2000` days this produced
**1,686 events**, an empirical rate of **0.843/day** against the theoretical
`mu/(1-n) = 0.8333/day` from [§8.5](#85-the-stationary-rate) — 1.2% high. That is
pure Monte-Carlo noise, and you should be able to say so quantitatively: a
stationary Hawkes has Fano factor `1/(1-n)^2`, so
`sd[N(T)] ~= sqrt(E[N] / (1-n)^2) = sqrt(1667/0.36) = 68` events, and `1686` is
`0.3` standard deviations above `1667`. The one systematic effect here — the
simulation starts empty at `t = 0`, so clusters seeded before the window are missing
— pushes the count *down*, not up (30 independent seeds of this simulator average
1,650, i.e. slightly below the stationary 1,667).

### The rescaling

```
Lambda(t)  =  mu t  +  (alpha/beta) sum_{t_j < t} [ 1 - e^{-beta (t - t_j)} ]
xi_i       =  Lambda(t_i) - Lambda(t_{i-1})
```

First eight events:

| `i` | `t_i` | `Lambda(t_i)` | `xi_i` |
|---|---|---|---|
| 1 | 0.6802 | 0.3401 | 0.3401 |
| 2 | 14.0757 | 7.4378 | 7.0978 |
| 3 | 16.2881 | 8.9393 | 1.5014 |
| 4 | 18.0785 | 10.2280 | 1.2887 |
| 5 | 18.6610 | 10.8022 | 0.5742 |
| 6 | 25.0356 | 14.5178 | 3.7156 |
| 7 | 25.1923 | 14.7038 | 0.1860 |
| 8 | 29.5753 | 17.5876 | 2.8838 |

Note events 6 and 7 are 0.157 days apart in real time but 0.186 apart in
rescaled time — the rescaling has *stretched* the burst, because the intensity was
high there. That is the whole mechanism: rescaled time runs fast when the model says
events are likely.

### The test

Over all 1,686 rescaled gaps:

```
mean(xi)  =  0.9926      (Exp(1) has mean 1)
var(xi)   =  0.9965      (Exp(1) has variance 1)
KS against Exp(1):   D = 0.00878,   p = 0.9994
```

Two controls, to show the test has power (same realization, wrong compensator):

| compensator used | `mean(xi)` | KS `D` | KS `p` |
|---|---|---|---|
| **true** (`alpha = 0.8`) | 0.9926 | 0.00878 | 0.9994 |
| misspecified `alpha = 0.2` | 0.6928 | 0.19636 | 3.2e−57 |
| Poisson only (`alpha = 0`) | 0.5928 | 0.28256 | 4.8e−118 |

The under-excited models produce rescaled gaps that are far too *short* on average
(0.59, 0.69 vs 1.0), because they under-predict the total number of events. The KS
statistic detects this instantly. Reproduce with the script pattern above plus a
five-line KS implementation (asymptotic Kolmogorov distribution:
`p = 2 sum_k (-1)^{k-1} exp(-2 k^2 lam^2)` with
`lam = (sqrt(n) + 0.12 + 0.11/sqrt(n)) D`) — I used my own rather than SciPy's, and
the numbers above are from a single seeded run, so expect them to move with the
seed.

**The caveat from [§5.4](#54-the-three-caveats) in action:** here `alpha` was
*known*, not fitted, so the KS null is exact. Fit `alpha` to this same data and the
`p = 0.9994` becomes optimistic.

---

## How this shows up in FlowQuake

Deliberately short — the code walkthrough is [STACK.md](../STACK.md), and this section
only maps theory to place.

| theory | where in the repo |
|---|---|
| gaps `tau_i` in the interval representation | [data.py:206-210](../flowquake/data.py#L206-L210); `tau = diff(t_days)`, clipped at `TAU_FLOOR_DAYS = 1e-7` ([data.py:25](../flowquake/data.py#L25)) |
| non-simplicity / ties | the same clip, plus `"pairing_key": "time+duplicate_rank"` in [runs/total_win.json](../runs/total_win.json) |
| history `H_t` as engineered features | [data.py:154-165](../flowquake/data.py#L154-L165) `recency_matrix` — 7 lags × 4 features, all `H_t`-measurable, so §4's likelihood theory applies unchanged |
| filtration enlargement is legal | `SAFE_TOKEN_DIMS` ([model.py:32-35](../flowquake/model.py#L32-L35)) *restricts* the filtration (drops absolute `x, y`); restriction is also legal, and is the memorization control |
| modelling `f(tau\|H)` instead of `lambda` | [flowquake/flow.py](../flowquake/flow.py); the licence is [§4.3](#43-proof-that-a-and-b-are-the-same-object) |
| the mark factorization `f_t f_s f_m` | [model.py:1-16](../flowquake/model.py#L1-L16) docstring; assumptions spelled out in [§9.2](#92-the-chain-rule-and-what-flowquake-assumes) |
| change of variables to physical units | [model.py:233-235](../flowquake/model.py#L233-L235): `tll = log p(u) - log sigma - log tau`, the two Jacobians for `u = (log tau - mu)/sigma` and `tau = exp(log tau)` |
| the reported score is per-event, censoring term dropped | [evaluate.py:98-101](../flowquake/evaluate.py#L98-L101); see [§4.4](#44-what-flowquake-actually-reports-and-the-term-it-drops) |
| direct sampling instead of thinning/inversion | [model.py:247-269](../flowquake/model.py#L247-L269) `sample_next`; [§6.4](#64-what-flowquake-does) |
| the truncated first event | [ntest.py:88-104](../flowquake/ntest.py#L88-L104); derived in [§6.5](#65-the-truncated-first-event-and-why-the-naive-version-biases-every-forecast-day) |
| Hawkes branching ratio | reported as 0.968 in [MANUSCRIPT.md:364](../MANUSCRIPT.md) — **no committed artifact**, see [results/CLAIMS.md](../results/CLAIMS.md) rows N2/X8/P5 |
| time-rescaling residuals | **no KS/Q–Q test anywhere** — though ETAS's `xi_i` are computed and discarded at [etas_forward_eval.py:126](../scripts/etas_forward_eval.py#L126); [§5.5](#55-the-repo-does-not-do-this) |

---

## Common misconceptions

**1. "The conditional intensity is a probability."**
Actually it is a *rate*, with units of 1/time, and it is unbounded above.
`lambda dt` is (to first order) a probability; `lambda` alone is not. *Why it
matters:* a `tll` of `+2.6` (SCEDC_20, [runs/fullsuite_summary.json](../runs/fullsuite_summary.json))
looks impossible if you think it is a log-probability. It is a log-density in
`log(1/day)`; densities exceed 1 whenever the scale is small.

**2. "A neural net that predicts `lambda(t)` from time is a point-process model."**
Actually if the prediction does not change when you delete a past event, it is an
inhomogeneous Poisson process and it has no self-excitation at all. *Why it
matters:* this is the single most common failure mode in ML-for-TPP work; the test
is one line — perturb the history and see if `lambda` moves.

**3. "`sum log lambda(t_i) - integral lambda` and `sum log f(tau_i)` are two
different models you have to choose between."**
Actually they are two spellings of the same number ([§4.3](#43-proof-that-a-and-b-are-the-same-object)).
You choose between them for *computational* reasons, not statistical ones. *Why it
matters:* it is what licenses FlowQuake's whole design, and misunderstanding it
makes the repo look like it is cutting a corner when it is not.

**4. "Modelling `f(tau)` directly avoids normalization."**
Actually it *relocates* it. The point-process normalizer `exp(-integral lambda)` is
replaced by the requirement that `f(tau|H)` integrate to 1 over `tau` in `(0,
infinity)`. A normalizing flow satisfies that exactly; a softmax over binned `tau`
would too; an unnormalised energy would not. *Why it matters:* if the temporal head
were an MDN or an energy model, the "exact likelihood" claim would need re-examining
— and the same argument is what makes the spatial head's closed-form `Z_j` load-bearing
([STACK.md, Part V](../STACK.md#part-v--the-second-model-the-neural-etas-spatial-head)).

**5. "Self-exciting means nonlinear."**
Actually the Hawkes intensity is *linear* in the past counting measure. Clustering
comes from the feedback loop through the realized points, not from curvature. *Why
it matters:* it explains why the branching representation exists at all — linearity
is exactly what makes the superposition-of-Poisson-clusters argument work.

**6. "The branching ratio `n` is a probability, so `n < 1` is automatic."**
Actually `n` is an expected *count* of direct offspring and there is nothing stopping
a fitted model from returning `n > 1`. *Why it matters:* the ComCat refit is reported
at `n = 0.968` ([MANUSCRIPT.md:364](../MANUSCRIPT.md)) — 3% from a model that does not
have a stationary distribution and whose simulator would not terminate.

**7. "Time rescaling proves the model is right."**
Actually it tests one null (i.i.d. `Exp(1)` gaps) about one component (the ground
process) with one weak statistic. It is blind to marks, it is blind to ordering
unless you separately test independence, and its `p`-value is anti-conservative if
the parameters were fitted in-sample. *Why it matters:* "we passed KS" is not a
result; "we passed KS out of sample, with a Q–Q plot and an independence test, and
the competitor did not" is.

**8. "The `nll` in the artifacts is the negative log-likelihood of the catalog."**
Actually it is `-(mean tll + mean sll)`: a per-event mean, excluding `mll` by
benchmark convention, and excluding the censoring term
([evaluate.py:98-101](../flowquake/evaluate.py#L98-L101),
[STACK.md §3](../STACK.md#3-marks-factorization-and-the-three-scores)). *Why it
matters:* three separate people will otherwise "correct" your arithmetic by trying
to add `mll` back in.

**9. "Rejection sampling the first event is a hack."**
Actually it is exact — the accepted draw has density `f(tau)/S(a)` on `tau > a`,
which is the correct truncated conditional ([§6.5](#65-the-truncated-first-event-and-why-the-naive-version-biases-every-forecast-day)).
The hack is the 200-round cap, which silently drops `(1-S(a))^200` of lanes. *Why it
matters:* knowing which half is exact and which half is approximate is exactly the
kind of distinction a viva probes.

---

## Questions a professor will ask

### Q1. Define the conditional intensity and say what could go wrong with the definition.

`lambda(t|H_t) = lim_{dt->0} P(N(t+dt) - N(t) = 1 | H_t)/dt`. Three things can go
wrong. (i) The limit may not exist — the compensator may not be absolutely
continuous, in which case no intensity exists but the compensator still does
([§2.2](#22-existence-and-uniqueness)). (ii) `H_t` must be the *left-continuous*
filtration, so `lambda` is predictable; otherwise you could condition on the event
you are predicting. (iii) The "`= 1`" rather than "`>= 1`" matters only for
non-simple processes; for a simple process orderliness makes them agree to first
order.

### Q2. Why is the compensator, not the intensity, the fundamental object?

Doob–Meyer guarantees the compensator exists and is unique (the predictable
increasing part of the submartingale `N`) under conditions far weaker than those
needed for a density; `lambda` is only `dLambda/dt` when that derivative exists. And
the three central theorems — likelihood, time rescaling, inverse simulation — are
statements about `Lambda`. The martingale property
`E[N(t)-N(s)|H_s] = E[Lambda(t)-Lambda(s)|H_s]` makes the compensator the best
predictable forecast of future counts, which is the forecaster's object.

### Q3. Derive `f(tau|H) = lambda exp(-integral lambda)`, and identify the step people skip.

[§3.2](#32-the-subtle-step-the-hazard-is-the-intensity)–[§3.3](#33-from-s---h-s-to-the-density).
The skipped step is establishing that the hazard *equals* the intensity: on the
event "no points since `t_{i-1}`", the filtration `H_{t_{i-1}+tau}` coincides with
"`H_{t_{i-1}}` and nothing happened", which is precisely the hazard's conditioning
set. After that it is `S'/S = -h`, integrate, exponentiate. The step fails if the
filtration is enlarged by covariates that evolve *during* the gap.

### Q4. Show the two forms of the log-likelihood are the same.

Substitute `log f(tau_i) = log lambda(t_i) - integral_{t_{i-1}}^{t_i} lambda` and
`log S(T-t_n) = -integral_{t_n}^T lambda`; the `n+1` integrals partition `[0,T]`
exactly and telescope to `integral_0^T lambda`. Numerically:
[§11](#11-worked-example-a--a-three-event-hawkes-both-likelihood-forms), where both
routes give `-3.9993103789` and the four compensator increments
`0.25 + 0.4203 + 1.3009 + 0.5997` sum to the integral term `2.5708`.

### Q5. In the bin derivation, where does the `n log Delta` go?

It is the volume element converting a probability into a density. `P(data)` at
resolution `Delta` vanishes like `Delta^n`; the likelihood is the density with
respect to the natural reference measure (counting measure on `n` × Lebesgue on
ordered `n`-tuples), obtained by dividing by `Delta^n`. Anyone who says the
`log Delta` terms "cancel" is hand-waving — they are *absorbed into the choice of
dominating measure*. Every model compared here uses that same reference measure, so
the term is common to all of them and cancels exactly in any likelihood ratio; the
absolute per-event numbers the benchmark reports are densities with respect to it,
which is precisely why `tll` carries units of `log(1/day)`.

### Q6. State and derive the time-rescaling theorem.

[§5.1](#51-statement)–[§5.2](#52-derivation). The one-line version:
`P(xi_i > z | H) = S(Hc^{-1}(z)) = exp(-Hc(Hc^{-1}(z))) = e^{-z}`, and the answer
does not depend on `H`, so the `xi_i` are i.i.d. `Exp(1)` by induction. It is the
probability integral transform applied one gap at a time.

### Q7. Prove Ogata's thinning algorithm correct.

The candidate process is Poisson at rate `lambda_bar`; the retention probability
`lambda(s|H_s)/lambda_bar` is predictable (it depends only on accepted points
strictly before `s`); so the accepted process has conditional intensity
`lambda_bar × lambda(s|H_s)/lambda_bar = lambda(s|H_s)`. By uniqueness of the
conditional intensity for simple processes, it has the target law. The dominating
rate must be recomputed after every acceptance because `lambda` jumps up there; for
a Hawkes with non-increasing kernel, `lambda(t+)` after the last acceptance
dominates until the next one.

### Q8. Derive `mu/(1-n)` two ways.

Branching: immigrants at rate `mu`, each seeding a cluster of expected size
`sum_k n^k = 1/(1-n)`, and every event is in exactly one cluster. Intensity:
`E[lambda] = mu + integral g(t-u) E[lambda(u)] du = mu + n E[lambda]` using
`E[dN(u)] = E[lambda(u)]du` (the compensator/martingale property), then solve. Both
announce `n < 1` — the geometric series and the sign of `(1-n)`.

### Q9. What is the mark factorization assuming, and when is it false?

[§9.2](#92-the-chain-rule-and-what-flowquake-assumes)–[§9.3](#93-where-the-factorization-is-wrong-concretely).
It assumes `s ⫫ tau | H` and `m ⫫ (tau, s) | H` for the *same, next* event. Two
concrete failures: short-term aftershock incompleteness (observed `m` depends
directly on `tau`, because small events are undetectable in a mainshock's coda —
the subject of Mizrahi et al. 2021, the authors of the very `etas` package used
here); and magnitude-dependent location uncertainty (observed `s` scatter is wider
for small `m`, purely instrumentally). What is *not* a violation: parent magnitude
setting offspring spatial scale — both ETAS and FlowQuake model that via
`d_j = d exp(gamma(m_j - m_c))`.

### Q10. Why does FlowQuake need neither thinning nor the inverse method?

Because it models `f(tau|H)` directly and can sample from it by integrating the flow
ODE forward from a Gaussian draw ([model.py:256-257](../flowquake/model.py#L256-L257)).
No dominating rate, no acceptance loop, no root find. The costs are real: no rate
field on a grid, no compensator, hence no time-rescaling residuals, and no
closed-form truncated conditional — which is why the one place a conditional *is*
needed ([ntest.py:88-104](../flowquake/ntest.py#L88-L104)) falls back on rejection.

### Q11. HOSTILE — the repo has no goodness-of-fit test for the temporal model at all. Why should I believe the temporal win?

That is a fair hit and the honest answer is that the temporal claim rests on a
likelihood difference plus a bootstrap CI, and on no calibration diagnostic at the
per-event scale. What exists is the CSEP N-test, which checks daily *count*
calibration — necessary but coarser. What is missing is the standard tool,
time-rescaling residuals, and it is missing because FlowQuake never forms `Lambda`
([§5.5](#55-the-repo-does-not-do-this)). It is not unfixable: `xi_i = -log(1 -
F(tau_i|H))` and `F` is a 1-D quadrature of the flow density. I would compute it for
the 21,889 test events, report a KS test and a `u_i`-vs-`u_{i+1}` independence check
against ETAS's residuals on the same events, and treat that as a required addition
before submission.

### Q12. HOSTILE — you dropped the censoring term. So your "likelihood" is not a likelihood.

Correct, and the artifacts should say "mean per-event predictive log-density" rather
than "log-likelihood". Three mitigations. (i) The dropped term is a single
`log S(T - t_n)` over `n = 21,889` events, so its per-event contribution is
`O(1/n)`; in [§11](#11-worked-example-a--a-three-event-hawkes-both-likelihood-forms)
it was 15% of a 3-event total and it would be ~0.002% here. (ii) ETAS is scored the
same way from the same file, so the comparison is exactly paired
([evaluate.py:28-51](../flowquake/evaluate.py#L28-L51)). (iii) Arguably the per-event
one-step-ahead predictive score is the *better* object for forecast evaluation than
a retrospective window likelihood. But the naming is loose and should be tightened.

### Q13. HOSTILE: the tau floor

*"You clamp `tau` at `1e-7` days. `tll` includes a `-log tau` Jacobian
([model.py:235](../flowquake/model.py#L235)), which is `+16.12` for a floored event.
Your headline temporal gain is `+0.053` nats/event
([runs/total_win.json](../runs/total_win.json)). Isn't the win just tied timestamps?"*

This is the strongest available attack and it deserves a full answer.

The mechanism is real. `TAU_FLOOR_DAYS = 1e-7` ([data.py:25](../flowquake/data.py#L25))
and the comment on that line says the smallest nonzero catalog gap is `~5e-8` d, so
the floor binds. `-log(1e-7) = +16.118`, against a mean `tll` of `1.4876`
([runs/total_win.json](../runs/total_win.json)). Floored events are therefore high-`tll`
events.

But the magnitude is bounded, and here is the bound. Let `p` be the fraction of test
events at the floor. All of them map to the *same* normalized `u`, so the flow sees
an atom of mass `p`; `sigma_min = 0.02` ([runs/comcat25/config.yaml](../runs/comcat25/config.yaml))
convolves it with a Gaussian of that width in `u` units, capping the density at
roughly `p / (0.02 · sigma_logtau · sqrt(2 pi))` in `log tau` units. With
`sigma_logtau ~ 3` — an assumption; the fitted normalisation constants live in the
gitignored checkpoint, not in any committed artifact — that is `p/0.150`, so
`tll_floored ~= log(p/0.150) + 16.12`. At `p = 0.001` that is `+11.1`. Meanwhile
ETAS's temporal score at `tau -> 0` is `log lambda(t_{i-1}+)`, which is large but
finite — order `log(10^3 to 10^5) = 7` to `11.5`. So the per-event advantage on
floored events is plausibly `0` to `+3` nats, and the contribution to the mean gain
is `p × (0 to 3)`. **At `p <= 0.001` that is `<= 0.003` nats/event and cannot explain
`+0.053`. At `p = 0.02` it could explain the whole thing.**

So the answer is: *the concern is legitimate and the resolution is a single number I
cannot read from the committed tree*, because per-event score CSVs are gitignored
(`per_event*.csv` in [.gitignore](../.gitignore)) and `reference/` is not committed. The
check is three lines — count `tau <= TAU_FLOOR_DAYS` among test targets, and recompute
the paired mean gain with those events excluded — and it should be in the paper as a
robustness row. I would not defend the result without it.

> **Two thresholds circulate in this primer and they are not in conflict — know
> which you are quoting.** This answer says a floored fraction `p <= 0.001` is
> harmless and `p ~ 0.02` could explain the margin, because it credits ETAS with
> a large *finite* density at `tau -> c`, so the per-event *advantage* on floored
> events is only 0 to +3 nats.
> [Chapter 4 section 7.2](04-flows-and-density-estimation.md#72-what-sigma_min-actually-buys)
> and its hostile question H2 quote **0.29%**, which is `0.0533 / 18.2` — the
> per-event *ceiling* under `sigma_min = 0.02`, assuming ETAS scores those same
> events at roughly zero. That is deliberately the version most favourable to the
> objector. The honest range to state is therefore **0.3% to 2% of test events at
> the floor**, depending on what you grant ETAS, and the point that survives
> either way is that **the repository does not contain the count.**

### Q14. HOSTILE — you report a near-critical branching ratio of 0.968 for the fitted ETAS. Is your baseline even a valid model?

Two separate problems, and I will not conflate them. **First, the number is not
backed**: [results/CLAIMS.md](../results/CLAIMS.md) rows N2/X8/P5 record that no ETAS
parameter vector is committed anywhere under `runs/`; the file the manuscript cites
holds only `window/n/tll/sll/nll/etas_name/fit_window/params_frozen_from`. I verified
this. So `0.968` should be quoted as a manuscript assertion, not a checked fact.
**Second, if it is right**, the fitted ETAS is 3% from criticality: expected cluster
size `1/(1-0.968) = 31.25`, and `d/dn [1/(1-n)] = 977` there, so the model's
long-run behaviour is badly conditioned. That does not invalidate the *likelihood*
comparison — [§4](#4-the-likelihood-derived-twice) never assumed stationarity, and
both models are scored on the same finite window — but it does mean any statement
about simulated long-run rates or cluster statistics from that ETAS should be
treated as unstable. It also makes the *simulation-based* CSEP comparison more
delicate than the likelihood comparison, since near-critical simulation has heavy
count tails.

### Q15. HOSTILE — the auxiliary window. Isn't the first scored event's likelihood wrong?

Partly, and the repo handles the important part. The scored quantity is
`f(tau_i | H_{t_{i-1}})`, so the first *test* event's score needs the history before
it — which is supplied, because evaluation runs the whole catalog as one sequence
([data.py:358-371](../flowquake/data.py#L358-L371),
[evaluate.py:85](../flowquake/evaluate.py#L85)), not the test window in
isolation. The genuine edge case is the very first event of the *catalog*, whose
`log_tau` is filled with the median ([data.py:210](../flowquake/data.py#L210)) — and it
is never a target, by construction of the masks
([data.py:216-218](../flowquake/data.py#L216-L218)). What remains imperfect is that the
catalog itself starts at 1971 with an empty pre-history, so events early in the
auxiliary window have an under-estimated intensity. Since the aux window ends in
1981 and scoring starts in 2007, that error has 26 years of Omori decay to become
irrelevant.

### Q16. Give me a process that is not Poisson, not renewal, and not Hawkes.

A self-correcting (stress-release) process, `lambda(t) = exp(mu t - alpha N(t))`:
each event *lowers* the intensity, producing regular, anti-clustered,
under-dispersed sequences. It is the seismic-gap hypothesis made mathematical, and a
useful null precisely because real catalogs decisively reject it in favour of
clustering. Also: any nonlinear Hawkes, `lambda = phi(mu + sum g)` with `phi`
non-identity — used to allow inhibitory kernels, at the cost of losing the branching
representation entirely.

### Q17. Your model conditions on 30 engineered features, not the raw history. Does the likelihood theory still hold?

Yes. Those features are deterministic functions of the past event set, hence
`H_t`-measurable, so the conditioning sigma-algebra is a *sub*-sigma-algebra of the
internal history — and [§4](#4-the-likelihood-derived-twice) holds for any filtration
to which `N` is adapted and `lambda` predictable, including a coarser one.
Restricting costs *statistical efficiency* (if the truth depends on something the
features discard, you cannot represent it), never *validity*: the score is still
proper and the ETAS comparison still fair.
[model.py:32-35](../flowquake/model.py#L32-L35) restricts further by dropping absolute
`x, y`, and that restriction is the memorization control
([STACK.md §11](../STACK.md#11-modelpy--assembly-and-the-memorization-knob)).

### Q18. What breaks if the process is not simple?

The bin derivation of [§4.1](#41-derivation-a-infinitesimal-bins) immediately —
"each bin holds 0 or 1 event" *is* orderliness. Time rescaling breaks too: coincident
events give `xi_i = 0`, which is not `Exp(1)`. Ties are normally handled by treating
them as one event with a multiplicity mark, by dequantising (jittering within the
timestamp resolution), or — as here — by flooring the gap
([data.py:208](../flowquake/data.py#L208)). The floor is the least principled of the
three; its cost is quantified in [Q13](#q13-hostile-the-tau-floor).

### Q19. One sentence: why does any of this justify FlowQuake's architecture?

Because [§4.3](#43-proof-that-a-and-b-are-the-same-object) proves that an exactly
normalised conditional density over the next waiting time *is* an exactly normalised
point-process likelihood, so a 1-D normalizing flow — whose normalisation is free
and exact — can replace ETAS's hand-chosen, cheaply-integrable kernels without
giving up a single nat of rigour, and the price is paid entirely on the simulation
and diagnostic side rather than on the likelihood side.

---

## Further reading

1. **Daley & Vere-Jones, *An Introduction to the Theory of Point Processes*,
   Springer (Vol. I, 2nd ed. 2003; Vol. II, 2nd ed. 2008).** The reference. Vol. I
   Ch. 6–7 for conditional intensity, likelihood and cluster processes; Vol. II
   Ch. 14 for the martingale/compensator theory. Where almost everything unproved in
   this chapter is proved.

2. **Rasmussen, J. G. (2018). "Lecture Notes: Temporal Point Processes and the
   Conditional Intensity Function." arXiv:1806.00221.** ~30 free pages covering
   [§2](#2-the-conditional-intensity)–[§6](#6-simulation) in an afternoon. Best
   starting point.

3. **Ogata, Y. (1988). *J. Amer. Statist. Assoc.* 83(401), 9–27.** Introduced both
   the temporal ETAS model and time-rescaling residual analysis. The direct ancestor
   of everything this repo compares against.

4. **Ogata, Y. (1981). "On Lewis' simulation method for point processes." *IEEE
   Trans. Inform. Theory* 27(1).** Thinning, with its correctness proof. (Indexes
   disagree on the page range — dblp says 23–30, several bibliographies say 23–31 —
   so check before quoting one.)

5. **Hawkes, A. G. & Oakes, D. (1974). *J. Appl. Prob.* 11(3), 493–503.** The
   branching-equals-intensity equivalence of
   [§8.2](#82-the-branching-representation), properly proved. Pair with Hawkes
   (1971), *Biometrika* 58(1), 83–90, for the original definition.

6. **Brown, Barbieri, Ventura, Kass & Frank (2002). *Neural Computation* 14(2),
   325–346.** The elementary time-rescaling proof plus the practical goodness-of-fit
   recipe (KS bands, the `u_i`-vs-`u_{i+1}` independence check). Cite this for
   "standard TPP diagnostic".

7. **Reinhart, A. (2018). "A review of self-exciting spatio-temporal point processes
   and their applications." *Statistical Science* 33(3), 299–318.** The best survey bridging
   temporal Hawkes to the marked space–time setting of
   [§9](#9-marked-point-processes), including the separability assumptions made here.

8. **Shchur, Biloš & Günnemann (2020). "Intensity-free learning of temporal point
   processes." *ICLR*.** The clearest ML-side statement of the "model `f(tau|H)`
   directly, skip the intensity integral" argument FlowQuake instantiates. Read
   straight after [§4.3](#43-proof-that-a-and-b-are-the-same-object).

9. **Mizrahi, Nandan & Wiemer (2021). "Embracing data incompleteness for better
   earthquake forecasting." *J. Geophys. Res. Solid Earth* 126(12), e2021JB022379.**
   Methods paper for the `etas` package used here, and the source for the
   incompleteness counterexample in [§9.3](#93-where-the-factorization-is-wrong-concretely).

10. **Stockman, Lawson & Werner (2026). "EarthquakeNPP." *TMLR*; arXiv:2410.08226.**
    The benchmark contract: splits, `m_c` values, the `tll`/`sll`/`nll` definitions,
    and the ETAS baselines every number in `runs/` is measured against.

---

*Next: [Chapter 2 — seismology for the point-process modeller](02-seismology.md),
which supplies the marks this chapter has treated as abstract; then
[Chapter 3 — ETAS](03-etas.md), which is this chapter's machinery instantiated with
a century of seismology in the kernels. The scoring rules previewed in
[§10](#10-why-log-likelihood) are [Chapter 6](06-evaluation-and-csep.md).*
