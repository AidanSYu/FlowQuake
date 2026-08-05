# Chapter 4 — Neural density estimation, normalizing flows, and flow matching

The temporal head of FlowQuake is 22,657 parameters of MLP and it is the only part
of the production model that produces a likelihood by solving a differential
equation. This chapter builds the theory behind it from the definition of a
probability density up to the theorem that makes it trainable in three lines, and
then turns that theory on the repository: what the design buys, what it costs, and
which parts of the repo's own account of it do not survive contact with the code.

---

## What this chapter buys you

- You can **derive** the change-of-variables formula, the instantaneous
  change-of-variables theorem `d/dt_flow log p = -div(v)`, and Hutchinson's trace
  estimator, from scratch, on a whiteboard, without notes.
- You can **state and sketch the proof** of the conditional-flow-matching theorem
  (the gradients of the tractable conditional loss and the intractable marginal
  loss coincide), which is the single result that makes FlowQuake's temporal head
  trainable without ever solving an ODE during training.
- You can **read [flowquake/flow.py](../flowquake/flow.py) line by line** and say what
  each line is doing mathematically, including the sign convention on the
  backward integration and why `sigma_min` is there.
- You can **defend the architectural choice** — flow for time, closed-form
  structured heads for space and magnitude — on the merits, *and* state precisely
  where the repository's stated justification for it is wrong.
- You can **place FlowQuake in the neural-TPP taxonomy**: why "model `f`, not
  `lambda`" is a substantive choice about where the normalizer comes from, and what
  it costs you.
- You can **name the two open empirical checks** a hostile examiner will reach
  for — ODE step-count convergence and the density ceiling at the inter-event-time
  floor — and say exactly what the repo does and does not show about each.

## Prerequisites

- [Chapter 3 — ETAS](03-etas.md) for the power-law kernels, the Omori decay,
  and the Gutenberg–Richter law. The tail-behaviour argument in §11 below is a
  direct continuation of it.
- [Chapter 1 — point processes](01-point-processes.md) for the conditional
  intensity and the two equivalent forms of the point-process likelihood, and
  [Chapter 2 — seismology](02-seismology.md) for catalogs and completeness. If you
  have not read them, at minimum read
  [STACK.md §2](../STACK.md#2-the-likelihood-and-the-choice-that-shapes-this-repo),
  which states the `lambda` vs `f(tau | H)` fork this whole chapter lives on the
  right-hand side of.
- Comfort with multivariable calculus (Jacobians, the divergence theorem),
  measure-zero-level rigour about change of variables, and basic ODE existence
  theory. No prior exposure to flows, diffusion, or generative modelling is
  assumed.

**Notation used throughout** (fixed, no variants):

| symbol | meaning |
|---|---|
| `tau` | inter-event gap, in days |
| `t_flow` | the flow's integration variable, in `[0, 1]` — **never** event time |
| `z` | flow state; `z0 = z(t_flow=0)` is the latent, `z(1)` is data-side |
| `u` | the datum the flow models: normalized log inter-event time |
| `v(z, t_flow, c)` | the learned velocity field, conditioned on `c` |
| `q` | the true data distribution of `u` |
| `p_t` | the probability path; `p_0 = N(0, I)`, `p_1` is the model's density |
| `sigma_min` | the flow-matching bandwidth floor (repo: `sigma_min[0] = 0.02`) |
| `d` | dimension of the flow's state (1 for the temporal head) |
| `tll` | `log f_t(tau)`, in log(1/day) |

---

## 1. The density estimation problem: normalization is the entire difficulty

### 1.1 What makes a model a valid density

A model of a continuous random variable `X` on `R^d` is a *density model* if the
function `p_theta: R^d -> R` it produces satisfies

```
(i)   p_theta(x) >= 0            for all x
(ii)  int_{R^d} p_theta(x) dx = 1
```

Condition (i) is free: exponentiate, or softplus, or square. Condition (ii) is
the whole subject. Any neural network `g_theta(x) -> R` gives you an unnormalized
score. Turning it into a density requires

```
p_theta(x) = exp(g_theta(x)) / Z(theta),    Z(theta) = int exp(g_theta(x)) dx
```

and `Z(theta)` is a `d`-dimensional integral over a function you just made up. It
has no closed form, it changes every gradient step, and you need `log Z` — not
`Z` — to write down a log-likelihood. This is the *partition function problem*,
and every family of generative models is, at bottom, a different strategy for
dodging it.

Why this matters here and not just in the abstract: FlowQuake's headline metric is
a **per-event log-density in physical units**. `tll = 1.4876` means "the model
assigned density `e^1.4876 = 4.43` per day at the realized gap"
([runs/eval_test_N1.json](../runs/eval_test_N1.json)). That number is only meaningful
if the density integrates to 1 over `tau in (0, inf)`. A model that quietly
integrates to 1.5 would report `log 1.5 = 0.405` nats/event of free win — 7.6
times the entire measured margin over ETAS
(`+0.0533` nats/event, [runs/total_win.json](../runs/total_win.json)). Normalization
is not hygiene here; it is the result.

### 1.2 The taxonomy

Every generative family answers one question: *where does the normalizer come
from?*

| family | representative | normalizer comes from | is `log p(x)` … | cost of one `log p(x)` |
|---|---|---|---|---|
| **Autoregressive** | PixelCNN, MADE, WaveNet, GPT | chain rule: `p(x) = prod_i p(x_i \| x_<i)`, each 1-D factor normalized by construction | **exact** | 1 forward pass (parallel over dims); sampling costs `d` sequential passes |
| **Discrete normalizing flow** | NICE, RealNVP, Glow, MAF/IAF | change of variables from a normalized base; the model is a bijection so mass is conserved | **exact** | 1 pass + a Jacobian log-det chosen to be cheap |
| **Continuous normalizing flow** | FFJORD, rectified flow (**FlowQuake's time head**) | same, but the bijection is an ODE flow map | **exact**, up to ODE solver error | ODE solve + a divergence per step |
| **Diffusion / score-based** | DDPM, NCSN, DSTPP | variational bound; or exactly via the probability-flow ODE | **bounded** (ELBO) by default; exact only via the PF-ODE | 1 pass for the bound; full ODE solve for the exact value |
| **Latent-variable / VAE** | VAE | `log p(x) >= ELBO`; the true marginal integrates out `z` | **bounded** (lower) | 1 pass, bound only |
| **Energy-based** | EBM, Boltzmann machines | nowhere — `log Z` is estimated by MCMC/AIS | **unavailable** in closed form | MCMC or annealed importance sampling |
| **Implicit / GAN** | GAN | no density is ever defined | **unavailable** | n/a |
| **Mixture density network** | Bishop's MDN, LogNormMix | each component is normalized; convex combination of normalized things is normalized | **exact** | 1 pass, closed form |
| **Structured closed-form mixture** | **FlowQuake's `heads.py`, `neural_etas.py`** | the kernel's integral is known analytically and the weights do not depend on the query point | **exact** | 1 pass, closed form |

Two rows deserve comment because they are the two FlowQuake actually uses.

**Row "continuous normalizing flow"** is the only row where "exact" carries an
asterisk that is about *numerics* rather than *mathematics*. The model's density is
exactly defined; you compute it by numerical integration and inherit the solver's
error. §8 quantifies that error for this repo.

**Row "structured closed-form mixture"** is the trick that carries FlowQuake's
spatial win. A mixture `sum_j w_j K_j(s)` is normalized iff each `K_j` is and
`sum_j w_j = 1`. If, additionally, the `w_j` and the shape parameters of `K_j` are
computed **without looking at `s`**, then `int sum_j w_j K_j(s) ds = sum_j w_j
int K_j = 1` holds with the integral pulled inside. Let the weights see `s` and
you are back to an energy-based model with an intractable `Z`. That is the
"normalization argument" of
[STACK.md, Part V](../STACK.md#the-normalization-argument-this-is-the-whole-trick),
and it is why [flowquake/neural_etas.py:61-62](../flowquake/neural_etas.py#L61-L62)
is the load-bearing line of that file.

### 1.3 The one line that matters for point processes

From the likelihood identity (Chapter 2 / [STACK.md §2](../STACK.md#2-the-likelihood-and-the-choice-that-shapes-this-repo)),

```
log L = sum_i log lambda(t_i | H_{t_i}) - int_0^T lambda(u | H_u) du
      = sum_i log f(tau_i | H_{t_{i-1}}) + (censoring term)
```

Modelling `lambda` moves the normalization problem into `int lambda` — an integral
along a history-dependent path. Modelling `f(tau | H)` moves it into the
normalization of a **one-dimensional density**. That is a spectacular reduction:
`d = 1` is the regime where flows are cheapest and where exact likelihoods are
free. FlowQuake takes that road, and the rest of this chapter is what it costs and
what it buys.

---

## 2. Change of variables, and why the Jacobian is the whole design problem

### 2.1 The derivation

Let `f: R^d -> R^d` be a diffeomorphism (bijective, differentiable, with
differentiable inverse), let `Z = f(X)`, and let `p_Z` be a known density. Fix any
measurable set `A ⊆ R^d`. Because `f` is a bijection,

```
P(X in A) = P(Z in f(A)) = int_{f(A)} p_Z(z) dz
```

Substitute `z = f(x)`. The multivariate substitution rule gives
`dz = |det J_f(x)| dx` where `J_f(x) = df/dx` is the `d x d` Jacobian:

```
P(X in A) = int_A p_Z(f(x)) |det J_f(x)| dx
```

This holds for *every* measurable `A`, so the integrands agree almost everywhere:

```
p_X(x) = p_Z(f(x)) · |det J_f(x)|
```

and taking logs,

```
log p_X(x) = log p_Z(f(x)) + log |det J_f(x)|                       (CoV)
```

Read the direction carefully: `f` maps **data to latent** (the *normalizing*
direction). The generator is `g = f^{-1}`, mapping latent to data. If instead you
parameterize `g` and want the density of `x = g(z)`, the same argument with roles
swapped gives `log p_X(x) = log p_Z(g^{-1}(x)) - log|det J_g(g^{-1}(x))|`.
Sign errors here are the single most common bug in flow code, and they are not
detectable by eye — they are detectable by
`tests/test_flow.py::test_density_integrates_to_one`, which grid-integrates the
model and checks it hits 1.0.

### 2.2 The architectural constraint

`(CoV)` is a complete recipe: pick any diffeomorphism, get an exact density.
The catch is `log|det J_f|`. For a general `d x d` matrix the determinant costs
`O(d^3)`, and you need it at every forward pass, differentiably. So flow
architecture is entirely the art of building expressive bijections with cheap
determinants. The families:

**Planar flows** (Rezende & Mohamed 2015, ICML): `f(x) = x + w·h(a^T x + b)`, a
rank-one perturbation of the identity. By the matrix determinant lemma
`det(I + w a^T h') = 1 + h'(a^T x + b)·a^T w`, so the log-det is one scalar,
`O(d)`. Each layer moves mass along a single direction, so you need many, and
invertibility needs a constraint on `a^T w`. Historically important, obsolete.

**Coupling layers / RealNVP** (Dinh, Sohl-Dickstein & Bengio 2017, ICLR). Split
`x = (x_A, x_B)`; leave `x_A` alone; transform `x_B` with parameters from `x_A`:

```
z_A = x_A
z_B = x_B * exp(s(x_A)) + t(x_A)

        [ I                   0        ]
J   =   [ d z_B/d x_A     diag(e^s)    ]      log|det J| = sum_k s_k(x_A)
```

`O(d)` from one forward pass of `s`, and both directions are closed-form (invert
by `x_B = (z_B - t)·e^{-s}`), so sampling and density evaluation cost the same.
Expressivity comes from stacking layers with permuted splits.

**Autoregressive flows** (MAF: Papamakarios, Pavlakou & Murray 2017, NeurIPS;
IAF: Kingma et al. 2016, NeurIPS): `z_i = x_i·exp(s_i(x_<i)) + t_i(x_<i)`. The
Jacobian is triangular, `log|det J| = sum_i s_i(x_<i)`. MAF evaluates density in
one pass but samples in `d` sequential passes; IAF is the transpose. That
asymmetry is why IAF lives inside VAEs (you sample) and MAF is used for density
estimation (you evaluate).

The rest of the zoo — Glow's invertible 1x1 convolutions, neural spline flows,
sum-of-squares polynomial flows — is variations on "make the Jacobian triangular,
low-rank, or block-diagonal". Survey: Papamakarios et al. 2021, JMLR.

### 2.3 What the repo actually tried

Two committed artifacts record early experiments on the **spatial** head, both on
`SCEDC_30` (a `m_c = 3.0` cut, *not* one of the benchmark's five catalogs):

| artifact | spatial head | params | train time | test `sll` | ETAS `sll` | gap |
|---|---|---|---|---|---|---|
| [runs/chk_realnvp.json](../runs/chk_realnvp.json) | RealNVP coupling flow | 468,064 | 80.0 s | −11.4151 | −7.6206 | **−3.794** |
| [runs/chk_mdn.json](../runs/chk_mdn.json) | Gaussian mixture density net | 274,864 | 74.6 s | −11.5121 | −7.6206 | **−3.892** |

Both artifacts also record `"test_mll": NaN` (no working magnitude head at that
point) and `"steps": 400` — these were 75–80-second probes, not converged runs. Do
not over-read them: they establish that neither a coupling flow nor an MDN was
anywhere near ETAS *after 400 steps*, which is weak evidence about what either
could do after 20,000. Neither implementation survives in the committed tree
(`grep -ri "realnvp\|mdn" flowquake/ scripts/` returns nothing), so the exact
architectures cannot be inspected. The load-bearing argument against them is the
one in §10, not these two numbers.

---

## 3. Continuous normalizing flows

### 3.1 The object

Instead of composing `L` discrete bijections, take the limit: let the
transformation be the solution map of an ODE.

```
dz/dt_flow = v(z(t_flow), t_flow),      z(0) = z0 ~ p_0 = N(0, I)
```

`v: R^d x [0,1] -> R^d` is a neural network. Write `phi_t` for the **flow map**:
`phi_t(z0) = z(t_flow = t)`, with `phi_0 = id`. The model's density is the
pushforward `p_1 = (phi_1)_# p_0`.

This is Chen, Rubanova, Bettencourt & Duvenaud 2018 (NeurIPS, "Neural Ordinary
Differential Equations"), where the density result below is their Theorem 1.

### 3.2 When does this even make sense?

State the hypotheses, because a professor will ask and because they genuinely hold
for FlowQuake's network.

**(H1) Existence and uniqueness.** If `v(·, t)` is uniformly Lipschitz in `z` with
constant `L` for all `t`, and `v(z, ·)` is continuous in `t`, then Picard–Lindelöf
gives a unique solution on all of `[0, 1]` for every initial condition. Uniqueness
is what makes `phi_t` a *bijection*: two trajectories cannot merge (they would have
to agree at the merge point and hence agree everywhere backwards). The inverse map
is obtained by solving `dz/ds = -v(z, 1-s)`, which has the same Lipschitz constant.

**(H2) Differentiability.** For the Jacobian and its trace to exist pointwise we
want `v(·, t)` to be `C^1`.

**Does FlowQuake's `v` satisfy these?** Yes, and demonstrably.
[flow.py:53-59](../flowquake/flow.py#L53-L59) builds `v` as
`Linear -> SiLU -> Linear -> SiLU -> Linear -> SiLU -> Linear`.
SiLU is `x·sigmoid(x)`, which is `C^inf` with derivative bounded by
`max_x SiLU'(x) ≈ 1.0998` (attained near `x ≈ 2.4`). Linear maps are Lipschitz
with constant `||W||_op`. A composition of Lipschitz maps is Lipschitz with the
product of the constants, so

```
L <= (prod_l ||W_l||_op) · 1.0998^{n_layers}
```

globally, and `v` is `C^inf` in `z`. So (H1) and (H2) hold unconditionally — not
"in practice", but as a theorem about this architecture. That is a good thing to
be able to say out loud. (The time embedding
[flow.py:32-35](../flowquake/flow.py#L32-L35) is sinusoidal, hence `C^inf` and
bounded in `t_flow`; continuity in `t` is likewise unconditional.)

### 3.3 Deriving the instantaneous change of variables

**Claim.**

```
d/dt_flow  log p_t(z(t_flow))  =  - div v(z(t_flow), t_flow)
                               =  - tr( dv/dz )
```

where `z(t_flow)` is a trajectory of the ODE and `p_t` is the pushforward of `p_0`.

#### Route A — via the Jacobian and Jacobi's formula (recommended for the viva)

Let `J_t := d phi_t / d z0`, the `d x d` Jacobian of the flow map. Differentiate
the ODE with respect to `z0` and exchange the order of differentiation (valid by
(H2) and smooth dependence on initial conditions):

```
dJ_t/dt_flow = (dv/dz)|_{z(t_flow)} · J_t,       J_0 = I
```

This is a linear matrix ODE. Jacobi's formula says that for an invertible
differentiable `M(t)`,

```
d/dt log det M(t) = tr( M(t)^{-1} · dM/dt )
```

Apply it with `M = J_t`. (`J_t` is invertible: `det J_0 = 1`, `det J_t` is
continuous, and it can never hit 0 because that would break injectivity of
`phi_t`, contradicting uniqueness. So `det J_t > 0` throughout and the absolute
value in `(CoV)` can be dropped.)

```
d/dt_flow log det J_t = tr( J_t^{-1} · (dv/dz) · J_t ) = tr( dv/dz ) = div v
```

using cyclicity of the trace. Now apply `(CoV)` in the generator direction:
`p_t(phi_t(z0)) · det J_t = p_0(z0)`, i.e.

```
log p_t(z(t_flow)) = log p_0(z0) - log det J_t
```

Differentiate in `t_flow`:

```
d/dt_flow log p_t(z(t_flow)) = - d/dt_flow log det J_t = - div v      ∎
```

#### Route B — via the continuity equation (the physicist's derivation)

Probability mass is conserved and transported by the velocity field, so `p_t`
satisfies the **continuity (transport) equation**

```
d p_t / dt_flow  +  div( p_t · v )  =  0
```

*Where that comes from:* for any fixed region `Omega`, the rate of change of mass
inside equals minus the flux out through the boundary,

```
d/dt_flow int_Omega p_t dx = - int_{dOmega} p_t (v · n) dS
                           = - int_Omega div(p_t v) dx    (divergence theorem)
```

and since `Omega` is arbitrary the integrands agree.

Expand the divergence with the product rule:

```
d p_t/dt_flow + v · grad p_t + p_t · div v = 0
```

Now take the **material derivative** — the rate of change seen by an observer
riding a trajectory `z(t_flow)`:

```
d/dt_flow [ p_t(z(t_flow)) ] = (d p_t/dt_flow)(z) + grad p_t(z) · dz/dt_flow
                             = (d p_t/dt_flow)(z) + v · grad p_t(z)
                             = - p_t(z) · div v(z, t_flow)
```

Divide by `p_t(z) > 0`:

```
d/dt_flow log p_t(z(t_flow)) = - div v(z(t_flow), t_flow)             ∎
```

Two derivations, one line apart. Route A is more elementary and connects directly
to `(CoV)`; Route B is the one that generalizes (add a diffusion term and you get
Fokker–Planck, which is how diffusion models are analysed).

### 3.4 The likelihood formula, integrated

Integrate the theorem from `t_flow = 0` to `1` along the trajectory that ends at
the datum `u`:

```
log p_1(u) = log p_0(z(0))  -  int_0^1 div v(z(t_flow), t_flow) dt_flow
```

with `z(1) = u`. Two things to notice.

1. **This is exact.** No ELBO, no bound, no importance weights. That is the whole
   selling point over VAEs and diffusion models.
2. **You must solve the ODE to evaluate it.** Specifically, you must solve it
   *backward* from `u` to recover `z(0)`, accumulating the divergence integral as
   you go. That is exactly what
   [flow.py:109-134](../flowquake/flow.py#L109-L134) does.

**Reading `flow.py`'s sign convention.** The code substitutes `s = 1 - t_flow` and
integrates `s` forward from 0 to 1 with `dz/ds = -v` (line 124: `return -v, div`).
The `logdet` accumulator integrates `div` with respect to `s`:

```
logdet(s=1) = int_0^1 div v(z(1-s), 1-s) ds
            = int_0^1 div v(z(t_flow), t_flow) dt_flow      (t_flow = 1-s)
```

and line 134 returns `log_p0 - logdet`, matching the formula exactly. ✓

---

## 4. The divergence cost, Hutchinson's estimator, and why FlowQuake pays neither

### 4.1 The cost

`div v = tr(dv/dz) = sum_{i=1}^d dv_i/dz_i`. Reverse-mode autodiff gives you one
row of the Jacobian per backward pass (a vector-Jacobian product `e_i^T (dv/dz)`),
so the exact trace costs **`d` backward passes**. Inside an ODE solver with `S`
steps and `R` stages per step, that is `S · R · d` backward passes per sample. For
FFJORD-scale images (`d ≈ 3072`) this is fatal.

### 4.2 Hutchinson's estimator, derived

**Claim.** Let `A` be any `d x d` matrix and `eps` a random vector with
`E[eps] = 0` and `Cov(eps) = I`. Then

```
E[ eps^T A eps ] = tr(A)
```

*Proof.* Write it out in coordinates and use linearity of expectation:

```
E[ eps^T A eps ] = E[ sum_{i,j} eps_i A_ij eps_j ]
                 = sum_{i,j} A_ij E[eps_i eps_j]
                 = sum_{i,j} A_ij delta_ij            (Cov(eps) = I, E[eps]=0)
                 = sum_i A_ii = tr(A)                 ∎
```

The practical payoff: `eps^T (dv/dz) eps = (eps^T dv/dz) · eps` needs **one**
vector-Jacobian product, regardless of `d`. `O(d)` becomes `O(1)`. This is
Hutchinson's estimator (M. F. Hutchinson, "A stochastic estimator of the trace of
the influence matrix for Laplacian smoothing splines", *Communications in
Statistics — Simulation and Computation* **18**(3), 1059–1076, 1989 — the paper
also introduces the `{-1, +1}` variant on minimum-variance grounds), imported into
CNFs by Grathwohl et al. 2019 (ICLR, "FFJORD").

**Variance.** Take `eps` Rademacher (`eps_i` iid uniform on `{-1, +1}`). Since
`eps_i^2 = 1`,

```
eps^T A eps = sum_i A_ii + sum_{i != j} A_ij eps_i eps_j = tr(A) + S
```

so the estimator is unbiased with error `S`, and

```
Var(S) = E[S^2] = sum_{i!=j} sum_{k!=l} A_ij A_kl E[eps_i eps_j eps_k eps_l]
```

For `i != j` and `k != l`, `E[eps_i eps_j eps_k eps_l] = 1` iff `{i,j} = {k,l}`,
and 0 otherwise. Hence

```
Var(S) = sum_{i != j} ( A_ij^2 + A_ij A_ji )
```

and for symmetric `A`, `Var(S) = 2 · ( ||A||_F^2 - sum_i A_ii^2 )`. Gaussian
`eps` gives `2||A||_F^2` for symmetric `A` — larger by exactly `2 sum_i A_ii^2`,
because it also has to average over the fluctuations of `eps_i^2` (Rademacher pins
`eps_i^2 = 1`). So Rademacher is never worse and is strictly better whenever the
diagonal is non-zero; that is why it is the default, and it is the choice
Hutchinson's original paper argues for on minimum-variance grounds.

Note the structural fact that matters below: **the variance is a sum over
off-diagonal entries.** At `d = 1` there are none, so `Var(S) = 0` identically —
Hutchinson is exact but also pointless, because at `d = 1` the trace *is* the
single scalar `dv/dz`.

### 4.3 Why FlowQuake needs none of this

The temporal head is `CondFlow(dim=1, ...)`
([model.py:85-87](../flowquake/model.py#L85-L87)). At `d = 1`:

- The "Jacobian trace" is one partial derivative.
- [flow.py:94-107](../flowquake/flow.py#L94-L107) computes it with
  `torch.func.jacrev` under a `vmap`, giving a per-sample exact `1 x 1` Jacobian
  whose `torch.diagonal(...).sum()` is that derivative.
- Exact, deterministic, one VJP. Same cost as one Hutchinson sample, zero
  variance.

There is no stochastic estimator anywhere in this repository, and there does not
need to be.

### 4.4 Where the repo's own explanation is wrong

[STACK.md §9](../STACK.md#9-flowpy--the-temporal-head) states:

> In high dimensions the divergence needs a stochastic (Hutchinson) estimator;
> here `dim = 1`, so the "Jacobian trace" is a single derivative and can be
> computed exactly and cheaply. **This is why the temporal head is a flow and the
> spatial head is not** — 1-D exactness is free, 2-D is not.

The first sentence is correct. **The conclusion does not follow, and the repo's
own code contradicts it.**

- [flow.py:6-8](../flowquake/flow.py#L6-L8) — the module docstring, written by the
  same author — says: *"dimension is 1-2, so the full Jacobian trace is cheap via
  torch.func.jacrev."* Two dimensions is explicitly declared cheap.
- `tests/test_flow.py::test_log_prob_matches_gaussian_2d_conditional`
  ([tests/test_flow.py:35-57](../tests/test_flow.py#L35-L57)) trains and evaluates a
  **2-D conditional flow** and checks its exact log-density against the analytic
  value to 0.3 nats. It runs in the standard test suite.
- Arithmetically: exact trace at `d = 2` is 2 backward passes vs Hutchinson's 1.
  A factor of two. Hutchinson only earns its variance at `d` in the hundreds.

So "2-D is not free" is false, and the cost argument is not the reason the spatial
head is a mixture. **The actual reason is stated in
[model.py:12-15](../flowquake/model.py#L12-L15):**

> The spatial/magnitude heads were pivoted from flows after val-LL diagnostics:
> flows anchored structure in their weights (memorizing train geography), while
> mixture components anchored to observed recent events move with the data at
> evaluation time.

That is an *inductive-bias* argument, not a *cost* argument, and it is the one you
should give. §10 develops it, including its limits. Flagging this discrepancy is
not pedantry: if you offer the cost argument in a viva and the examiner opens
`test_flow.py`, you lose the room.

---

## 5. Training CNFs the old way, and why it hurt

Before flow matching, you trained a CNF by maximum likelihood on the formula in
§3.4. That means: for every minibatch, solve the ODE, accumulate the divergence,
compute `log p_1(u)`, and backpropagate **through the solve**.

Two options, both bad.

**(a) Backprop through the solver.** Store every intermediate state of every RK
stage and differentiate the unrolled graph. Memory is `O(S · R · batch · d)` and
grows with solver accuracy: wanting a more accurate likelihood costs you memory.

**(b) The adjoint sensitivity method** (Chen et al. 2018). Solve a second,
*backward* ODE for the adjoint `a(t_flow) = dL/dz(t_flow)`:

```
da/dt_flow = - a(t_flow)^T · dv/dz
dL/dtheta  = - int_1^0 a(t_flow)^T · dv/dtheta  dt_flow
```

Memory is `O(1)` in the number of steps because the forward trajectory is
re-derived by integrating backwards. But: that reconstruction is not exactly
reversible and error accumulates for stiff `v`; cost roughly doubles; adaptive
solvers make the number of function evaluations a *learned* quantity (nothing
stops the field becoming stiff, so wall-clock per step silently grows); and the
gradient you compute is the gradient of an estimate whose error depends on solver
tolerance, which is not the same as an estimate of the gradient.

This is why CNFs were an interesting-but-impractical family for five years. Flow
matching removed the ODE from training entirely.

---

## 6. Flow matching: the theorem that makes this cheap

This is Lipman, Chen, Ben-Hamu, Nickel & Le 2023 (ICLR, "Flow Matching for
Generative Modeling"), essentially simultaneous with Liu, Gong & Liu 2023 (ICLR,
"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified
Flow") and Albergo & Vanden-Eijnden 2023 (ICLR, "Building Normalizing Flows with
Stochastic Interpolants"). The three papers arrive at the same objective from
different directions; "rectified flow" is the name for the straight-line-path
special case that FlowQuake uses.

### 6.1 The three objects

Let `q` be the data distribution on `R^d` and `p_0 = N(0, I)` the base.

**Probability path.** A family `{p_t}_{t_flow in [0,1]}` of densities with
`p_0 = N(0,I)` and `p_1 ≈ q`.

**Marginal vector field.** A field `w_t(z)` that *generates* the path, i.e. whose
flow map pushes `p_0` to `p_t`. Equivalently, `(p_t, w_t)` satisfies the
continuity equation of §3.3.

**Conditional path and field.** Choose, for each datum `u`, a path
`p_t(z | u)` with `p_0(· | u) = N(0, I)` and `p_1(· | u)` concentrated at `u`, and
a field `w_t(z | u)` generating it. Define the marginal path by mixing:

```
p_t(z) = int p_t(z | u) q(u) du
```

The conditional objects are *chosen by us* and are trivially computable. The
marginal objects are what we need and are intractable.

### 6.2 The marginalization lemma

**Claim.** The field

```
                int w_t(z | u) p_t(z | u) q(u) du
w_t(z)  :=      ----------------------------------
                            p_t(z)
```

generates the marginal path `p_t`.

*Proof.* Each conditional pair satisfies the continuity equation:

```
d p_t(z|u)/dt_flow  =  - div_z ( w_t(z|u) p_t(z|u) )
```

Multiply by `q(u)` and integrate over `u`, exchanging `d/dt_flow` and `div_z` with
the `u`-integral (justified by dominated convergence given enough regularity —
this is the one place the argument needs technical assumptions, and Lipman et al.
state them):

```
d p_t(z)/dt_flow = int d p_t(z|u)/dt_flow q(u) du
                 = - int div_z( w_t(z|u) p_t(z|u) ) q(u) du
                 = - div_z ( int w_t(z|u) p_t(z|u) q(u) du )
                 = - div_z ( w_t(z) p_t(z) )
```

where the last line is just the definition of `w_t` rearranged. That is exactly
the continuity equation for `(p_t, w_t)`. ∎

Note what `w_t(z)` *is*: a posterior-weighted average of the conditional fields,
`w_t(z) = E[ w_t(z|u) | z_t = z ]`. It is a conditional expectation, and the
conditional expectation is the minimizer of a squared loss. That observation is
the whole next subsection.

### 6.3 The gradient-equivalence theorem

Define the two objectives:

```
L_FM(theta)  = E_{t_flow ~ U[0,1],  z ~ p_t}          || v_theta(z, t_flow) - w_t(z)     ||^2
L_CFM(theta) = E_{t_flow ~ U[0,1],  u ~ q,  z ~ p_t(·|u)} || v_theta(z, t_flow) - w_t(z|u) ||^2
```

`L_FM` is what we want (it is minimized when `v_theta = w_t`, which by §6.2
transports `p_0` to `p_1`). It is intractable. `L_CFM` is a three-line Monte Carlo
estimate.

**Theorem (Lipman et al. 2023).** `L_FM(theta) = L_CFM(theta) + C` where `C` does
not depend on `theta`. Hence `grad_theta L_FM = grad_theta L_CFM`, and minimizing
the tractable objective is exactly minimizing the intractable one.

*Proof sketch (this is the full argument; only regularity side-conditions are
elided).* Expand both squares. Fix `t_flow`; the expectation over it is common.

```
||v - w||^2 = ||v||^2 - 2 <v, w> + ||w||^2
```

**Quadratic term.** By the definition `p_t(z) = int p_t(z|u) q(u) du`:

```
E_{z ~ p_t} ||v_theta(z,t)||^2 = int ||v_theta(z,t)||^2 p_t(z) dz
                               = int int ||v_theta(z,t)||^2 p_t(z|u) q(u) du dz
                               = E_{u ~ q, z ~ p_t(·|u)} ||v_theta(z,t)||^2
```

Identical in both objectives. ✓

**Cross term.** Multiply the lemma's definition through by `p_t(z)`:

```
w_t(z) p_t(z) = int w_t(z|u) p_t(z|u) q(u) du
```

Then

```
E_{z ~ p_t} <v_theta(z,t), w_t(z)>
    = int <v_theta(z,t), w_t(z)> p_t(z) dz
    = int <v_theta(z,t), w_t(z) p_t(z)> dz
    = int <v_theta(z,t), int w_t(z|u) p_t(z|u) q(u) du> dz
    = int int <v_theta(z,t), w_t(z|u)> p_t(z|u) q(u) du dz
    = E_{u ~ q, z ~ p_t(·|u)} <v_theta(z,t), w_t(z|u)>
```

using bilinearity of the inner product to pull it inside the `u`-integral.
Identical in both objectives. ✓

**Remainder.** `E||w_t(z)||^2` and `E||w_t(z|u)||^2` differ, but neither depends
on `theta`. Their difference is the constant `C`. ∎

**Say this out loud in a viva:** *"The two losses differ by a `theta`-independent
constant because the marginal field is the conditional expectation of the
conditional field, and regressing on a random variable and regressing on its
conditional mean have the same minimizer and the same gradient. The gap is exactly
the conditional variance."* That is the one-sentence version, and it is correct:
`C = E[Var(w_t(z|u) | z_t = z)]`, averaged over `t_flow` and `z`.

**What this does not say.** It does *not* say the trained `v_theta` equals `w_t`.
It says the objective is right. A finite network trained for finite steps
approximates `w_t` with some error, and that error propagates into `p_1` in a way
the theorem is silent about. Bounds exist (relating a Wasserstein error in `p_1`
to the `L^2` velocity error via Grönwall's inequality) but this repo neither uses
nor needs them.

---

## 7. The specific path FlowQuake uses

### 7.1 The path, and verifying it by differentiation

[flow.py:70-78](../flowquake/flow.py#L70-L78), with `s := sigma_min` and `t :=
t_flow` (the `.unsqueeze(-1)` broadcasting calls are elided for readability;
nothing else is changed):

```python
t  = torch.rand(B)
z0 = torch.randn_like(u)
zt = (1.0 - (1.0 - s) * t) * z0 + t * u
v  = self.velocity(zt, t, cond)
return F.mse_loss(v, u - (1.0 - s) * z0)
```

In our notation, the conditional path is

```
z_t = ( 1 - (1 - sigma_min) t_flow ) · z0  +  t_flow · u,      z0 ~ N(0, I)
```

**Verify the velocity target by differentiating.** `z0` and `u` are fixed along a
conditional path, so

```
d z_t / d t_flow = -(1 - sigma_min) z0 + u = u - (1 - sigma_min) z0     ✓
```

which is precisely `u - (1.0 - s) * z0` on line 78. The velocity along a
conditional path is **constant in `t_flow`** — the path is a straight line in
`(z0, u)` space. That is the "rectified" in rectified flow.

**Check the endpoints.**

```
t_flow = 0:   z_0 = 1 · z0 + 0 = z0  ~  N(0, I)                     ✓ base
t_flow = 1:   z_1 = (1 - (1 - sigma_min)) z0 + u = u + sigma_min·z0
```

**Check the marginal conditional law.** `z_t | u` is a linear combination of a
constant and a standard Gaussian, hence Gaussian:

```
z_t | u  ~  N( t_flow · u ,  ( 1 - (1 - sigma_min) t_flow )^2 · I )
```

so `mu_t(u) = t_flow · u` and `sigma_t = 1 - (1 - sigma_min) t_flow`. This is
exactly Lipman et al.'s "optimal transport" conditional path.

**Cross-check against the paper's closed-form conditional field.** Lipman et al.
give the conditional field for this path as a function of the *state* `z_t`, not
of `z0`. Substitute `z0 = (z_t - t_flow · u) / (1 - (1-sigma_min) t_flow)` into
`u - (1-sigma_min) z0`:

```
u - (1-sm)·(z_t - t·u) / (1 - (1-sm)t)
  = [ u(1 - (1-sm)t) - (1-sm) z_t + (1-sm) t u ] / (1 - (1-sm)t)
  = [ u - (1-sm)t·u + (1-sm)t·u - (1-sm) z_t ] / (1 - (1-sm)t)
  = ( u - (1-sm) z_t ) / ( 1 - (1-sm) t_flow )
```

which is Lipman et al.'s equation for `u_t(z|x_1)` on the OT path. The code's
form and the paper's form are algebraically identical; the code's is cheaper
because `z0` is already in hand.

With `sigma_min = 0` this reduces to `z_t = (1-t)z0 + t·u` with target `u - z0` —
the plain linear interpolant of Liu et al. 2023.

### 7.2 What `sigma_min` actually buys

Because `z0` and `u` are independent and `z_1 = u + sigma_min · z0`,

```
p_1  =  q  ∗  N(0, sigma_min^2 I)
```

The model's target density is the **data distribution convolved with a Gaussian of
bandwidth `sigma_min`** — a KDE bandwidth floor, exactly as the class docstring at
[flow.py:41-43](../flowquake/flow.py#L41-L43) says.

**The failure this prevents, with the arithmetic.** Suppose `q` has an atom: a set
of measure zero carrying probability mass `w`. Then `q` has no density at all, and
the supremum of achievable log-likelihood is `+infinity` — a model that
concentrates arbitrarily hard on the atom reports arbitrarily large likelihood
while learning nothing. With the convolution the density is bounded **everywhere**,
because a convolution of a probability measure with a bounded kernel cannot exceed
the kernel's peak (`d = 1`):

```
p_1(x)  =  int N(x - y; 0, sigma_min^2) q(dy)  <=  1 / (sigma_min · sqrt(2 pi))
```

for every `x`, with the atom itself contributing at least `w / (sigma_min ·
sqrt(2 pi))` of that. At `sigma_min = 0.02` (the production value,
[configs/final_s1554.yaml:27](../configs/final_s1554.yaml#L27)):

```
1 / (0.02 · sqrt(2 pi)) = 19.947        log(19.947) = 2.9931 nats
```

So the *optimal* flow under this objective cannot report more than **2.993 nats**
of log-density at any single point in normalized units.

**Does FlowQuake's data have atoms?** Yes, two sources, and one of them is created
by the repo itself.

1. **Timestamp quantization.** Catalog origin times are reported to finite
   precision (for ComCat, milliseconds — `1e-3 s = 1.157e-8` days; `reference/` is
   not committed so this cannot be re-checked here), so `tau` lives on a grid.
   Near the bottom of that grid the *relative* spacing is large, and `log tau` is
   what the model sees: consecutive grid multiples `k` and `k+1` sit
   `log(1 + 1/k)` apart. Multiples below `k ≈ 9` are swallowed by the floor of
   item 2, so the first spacing that survives is `log(10/9) = 0.105`; in
   normalized units (divide by `sigma_LT ≈ 2–3`) that is `0.035–0.053`, i.e. two
   to three times `sigma_min = 0.02`. So the smallest surviving gaps are
   genuinely resolved spikes rather than a smooth cloud, and `sigma_min` is only
   marginally wide enough to smear them together.
2. **The floor, which is code.** [data.py:208](../flowquake/data.py#L208) does
   `tau = np.clip(tau, TAU_FLOOR_DAYS, None)` with
   `TAU_FLOOR_DAYS = 1e-7` days ≈ 8.6 ms. **Every gap below 8.6 ms is mapped to
   exactly the same value.** That is a hard atom at `log tau = -16.118`, with mass
   equal to `P(tau < 8.6 ms)`. That mass is not zero: the constant's own comment
   ([data.py:25](../flowquake/data.py#L25)) records the catalog's smallest nonzero
   gap as `~5e-8` days, i.e. *below* the floor, so the clip binds on at least the
   shortest gaps and on every exactly-duplicated timestamp.

**The size of the exposure.** Combine the ceiling with the unit conversion of
Worked Example 2 (`tll = log p(u) - log sigma_LT - log tau`). For an event sitting
on the floor:

```
tll  <=  2.9931  -  log(sigma_LT)  +  16.118
```

With `sigma_LT` (the train-era std of `log tau`) plausibly around 2.0–3.0, that is
a per-event ceiling of **+18.0 to +18.4 nats**. The measured margin over ETAS is
`+0.0533` nats/event. So a fraction

```
0.0533 / 18.2  =  0.0029  =  0.29 %
```

of test events sitting on the floor, scored at the ceiling, would *by itself*
account for the entire temporal win. Note what that arithmetic assumes: the
reported margin is a **paired** gain, so 0.29% is the answer only if ETAS scores
those same events at roughly zero. ETAS's Omori kernel also gives a large (finite)
density at `tau -> c`, so the true leverage is smaller and 0.29% is the version of
the objection most favourable to the objector — which is the version worth being
able to answer.

**Is that what happened?** Probably not — see the hostile question H2 in §16 for
the argument — but the repo does not contain the check. `sigma_LT` lives in
`ckpt["stats"]["log_tau_std"]`, and no checkpoint is committed
([.gitignore](../.gitignore) excludes `*.pt`); per-event score CSVs are also excluded.
So neither the floored fraction nor the tail of the per-event `tll` distribution
can be recovered from the committed tree. This is a real, checkable, currently
unchecked exposure, and you should say so before the examiner does.

### 7.3 A discrepancy in the repo's `sigma_min` accounting

[configs/final_s1554.yaml](../configs/final_s1554.yaml) documents `sigma_min` as:

```yaml
  # KDE-style bandwidth floors (normalized units) per head [time, space, mag]:
  # prevent density collapse onto (discretized) training targets.
  sigma_min: [0.02, 0.01, 0.05]
```

**Only the first entry is used.** [model.py:85-92](../flowquake/model.py#L85-L92)
passes `sigma_min=sigma_min[0]` to `CondFlow`; `KernelMixtureHead` and
`GRMagnitudeHead` take no `sigma_min` argument at all, and `grep -rn sigma_min
flowquake/` finds no other consumer: the remaining hits are the config default
([config.py:37](../flowquake/config.py#L37)) and the pass-through
([train.py:44](../flowquake/train.py#L44)), plus an unrelated adaptive-KDE
bandwidth argument at [data.py:120](../flowquake/data.py#L120).
`sigma_min[1] = 0.01` and `sigma_min[2] = 0.05` are dead config keys. This is
mostly benign — the other two heads have their own floors, the softplus floors
`d >= d_floor_km`, `q >= q_floor` ([heads.py:82-83](../flowquake/heads.py#L82-L83))
and the `+0.005` shift ([heads.py:173](../flowquake/heads.py#L173)) — but the config
comment claims a mechanism that does not exist, and "we floored all three heads"
is a claim you should not make. Three adjacent items:

- `mag_dequant: 0.01` is likewise dead: [model.py:61](../flowquake/model.py#L61)
  accepts it with the comment `# kept for config compat (GR head absorbs it)` and
  never uses it.
- Both [STACK.md](../STACK.md#10-headspy--the-spatial-and-magnitude-heads) and
  [MANUSCRIPT.md:242-243](../MANUSCRIPT.md) call the `+0.005` in
  `GRMagnitudeHead.log_prob` "a half-bin shift for the catalog's
  **0.1**-magnitude discretization". Half of 0.1 is 0.05. `0.005` is the half-bin
  of a **0.01** grid — which is exactly what `final_s1554.yaml`'s own comment on
  `mag_dequant` says (`# uniform dequantization over the raw 0.01-mag grid`). One
  of the two statements about the catalog's magnitude resolution is wrong; the
  code is consistent with the 0.01 story, and `reference/` is not committed so
  *that* half cannot be settled here. **The half that can:** `0.005` is a
  hardcoded literal shared by all eleven catalogs, and the ISC/INGV catalogs are
  certainly on a 0.1 grid, so no assignment of ComCat's precision makes one
  constant correct everywhere — see
  [Ch. 2 §5.5](02-seismology.md#55-the-0005-in-headspy--what-it-corrects-and-the-doc-bug),
  which is the version of this criticism to use in a viva, and
  [Ch. 8 §2.8](08-flowquake-synthesis.md#28-conditional-gutenbergrichter-with-a-half-bin-shift)
  for the two binning conventions and why both give `+Δ/2`.
- [flow.py:3-4](../flowquake/flow.py#L3-L4), the *module* docstring, gives the path
  as `z_t = (1-t) z0 + t u` with target `(u - z0)` — the `sigma_min = 0` case only.
  The class docstring seven lines later and the code both use the general form.

---

## 8. Sampling, likelihood integration, and the step-count question

### 8.1 RK4 and its error

Both `sample` ([flow.py:80-92](../flowquake/flow.py#L80-L92)) and `log_prob`
([flow.py:109-134](../flowquake/flow.py#L109-L134)) use classical fourth-order
Runge–Kutta with a **fixed** step `h = 1/steps`:

```
k1 = f(z,             t)
k2 = f(z + h/2 · k1,  t + h/2)
k3 = f(z + h/2 · k2,  t + h/2)
k4 = f(z + h   · k3,  t + h)
z <- z + (h/6)(k1 + 2 k2 + 2 k3 + k4)
```

Local truncation error is `O(h^5)`; accumulated over `1/h` steps the **global
error is `O(h^4)`**. Doubling `steps` should reduce the error by `2^4 = 16`.

`log_prob` integrates the augmented system `(z, logdet)` jointly, so the logdet
accumulator inherits the same fourth-order accuracy — provided you feed the
*stage-perturbed* state into the divergence, which
[flow.py:126-131](../flowquake/flow.py#L126-L131) does correctly (each `f(...)` call
gets both the perturbed `z` and the perturbed `logdet`; the latter is ignored by
`f` since the divergence does not depend on it, but the structure is right).

**Cost.** At `steps = 64`: 64 steps x 4 stages = **256 velocity-plus-Jacobian
evaluations per event**. Times 21,889 test events, that is 5.6 million evaluations
of a 22.7k-parameter MLP with a `jacrev` on each. This is why
[model.py:221](../flowquake/model.py#L221) defaults `event_chunk = 4096` and
[model.py:230](../flowquake/model.py#L230) loops the evaluation over those chunks.

### 8.2 What step count the repo actually uses

| context | steps | source |
|---|---|---|
| validation during training | 32 | `val_ode_steps: 32` in every config, e.g. [configs/final_s1554.yaml](../configs/final_s1554.yaml); [config.py:61](../flowquake/config.py#L61) |
| auto-eval at end of training | 64 | [train.py:185](../flowquake/train.py#L185) hard-codes `"--steps", "64"` |
| `evaluate.py` CLI default | 96 | [evaluate.py:57](../flowquake/evaluate.py#L57) |
| sampling in `sample_next` | 24 | [model.py:249](../flowquake/model.py#L249) |

Across the 68 committed `eval_test*.json` artifacts (`git ls-files | grep
eval_test`), **66 record `ode_steps: 64`, one records 96
([runs/comcat25_s1555](../runs/comcat25_s1555/eval_test.json)), and one records 32
(`runs/smoke`)**. In particular the headline temporal number,
`tll = 1.4876391887664795`, comes from a **64-step** evaluation
([runs/eval_test_N1.json](../runs/eval_test_N1.json),
[runs/n1_density/eval_test.json](../runs/n1_density/eval_test.json)). So the
practical answer is: the reported results are 64-step results.

### 8.3 Is the result step-count sensitive? What the repo shows

The repo does **not** contain a deliberate convergence sweep artifact. It does
contain an *accidental* one, and it is worth walking through because it is exactly
the analysis you should present when asked.

`configs/final_s1555.yaml` and `configs/comcat25_s1555.yaml` are identical under
`diff` except for `out_dir`, one comment line, and one key (`mix_hidden: 64`,
which is also the code default at [model.py:60](../flowquake/model.py#L60)); both
set `seed: 1555`. Their two eval artifacts differ only in `ode_steps`:

| field | `runs/final_s1555/eval_test.json` | `runs/comcat25_s1555/eval_test.json` |
|---|---|---|
| `ode_steps` | 64 | 96 |
| `tll` | 1.485485315322876 | 1.485485315322876 |
| `sll` | −9.088310241699219 | −9.088310241699219 |
| `nll` | 7.602825164794922 | 7.602825164794922 |
| `paired.spatial.mean_gain` | −0.3985395882531781 | −0.3985395882531781 |
| `paired.temporal.mean_gain` | 0.051142535875441514 | 0.051142530276869186 |

Read the table carefully:

1. **`sll` and the spatial paired gain are bit-identical.** They must be — the
   spatial head is closed-form and does not touch the ODE. This is a strong
   indication that the two artifacts really do come from the same weights (two
   runs of the same seed and config), which is what makes the comparison legitimate
   at all.
2. **`tll` is identical to all 16 stored digits**, because it is stored as a
   float32 mean.
3. **The float64 `temporal.mean_gain` differs by `5.5986e-9` nats/event**, and the
   `joint.mean_gain` differs by *exactly the same amount* (difference of
   differences: 0.0), confirming the change is entirely temporal.

Apply Richardson extrapolation. If the error is `I_h = I + C h^4`, then

```
I_{1/64} - I_{1/96} = C (1/64^4 - 1/96^4) = (C/64^4) (1 - (64/96)^4)
                    = (C/64^4) · (1 - 0.19753)
```

so the 64-step error is

```
err(64) = (I_64 - I_96) / (1 - 0.19753) = 5.5986e-9 / 0.80247 = 6.98e-9 nats/event
err(96) = 0.19753 · err(64)             = 1.38e-9 nats/event
```

**Seven nanonats per event, against a claimed margin of 0.0533 nats/event.** The
discretization error is smaller than the reported margin by a factor of `7.6e6`.

### 8.4 What this argument does and does not establish

**Honest limits:**

- It is a **two-point** check, not a sweep. Richardson assumes the leading error
  term dominates; with two points you cannot verify that.
- It rests on the two runs sharing weights, which is *inferred* from bit-identical
  closed-form scores, not verified — **no checkpoint is committed**
  (`git ls-files | grep '\.pt$'` is empty; see
  [WORKING.md](../WORKING.md), "What cannot be re-derived here at all").
- It says nothing about **32 steps**, which is what early stopping uses. Model
  *selection* therefore ran on a coarser integration than the reported score. Since
  the error at 64 is ~7e-9, the error at 32 is ~16x that, ~1.1e-7 — still six
  orders of magnitude below the margin, so this is fine, but it is an inference,
  not a measurement.
- It says nothing about **other catalogs**. Denser catalogs (WHITE_06, `m_c` 0.6)
  have a different `log tau` distribution and could in principle stress the solver
  differently.

**The machinery to do it properly already exists and is not used.**
[scripts/diag_ll.py:34-37](../scripts/diag_ll.py#L34-L37) loops over
`steps in [32, 128, 512]` and prints `tll` at each, on 512 subsampled events for
both train and val splits. **No artifact under `runs/` records its output.** The
fix is one script run and one committed JSON.

**What you should say when asked.** *"The reported likelihood is a 64-step RK4
evaluation. The repo's own artifacts contain an accidental 64-vs-96 comparison at
fixed weights that puts the discretization error at about 7e-9 nats/event by
Richardson extrapolation, seven million times smaller than the claimed margin. The
diagnostic that would settle it properly — `scripts/diag_ll.py`, which already
sweeps 32/128/512 — was written but its output was never committed. That is a
one-hour gap in the evidence, and it should be closed before submission."*

### 8.5 A subtlety worth raising unprompted

`sample` (forward, 24–32 steps) and `log_prob` (backward, 64 steps) use RK4, which
is **not** a symmetric integrator. The discrete forward map at step size `h1` is
not the exact inverse of the discrete backward map at step size `h2`. So the
distribution the simulator draws from (used for the CSEP N/S/M tests,
[STACK.md Part VI](../STACK.md#part-vi--generative-evaluation-simulation-and-csep))
and the distribution whose density is reported as `tll` are *strictly different
objects* — they coincide only in the `h -> 0` limit.

The magnitude of the discrepancy is bounded by the same convergence numbers as
above (~1e-7 nats at 32 steps), so it is numerically irrelevant here. But it is a
conceptually real gap and it is the kind of thing an examiner enjoys finding
before you do.

---

## 9. Alternative neural TPP designs, and why "model `f`, not `lambda`" is substantive

The taxonomy question for a neural TPP is not "what architecture" but **"where
does the normalizer come from, and how much history does the model see?"**

| model | models `lambda` or `f`? | how the normalizer is obtained | history reach |
|---|---|---|---|
| **Hawkes / ETAS** (Ogata 1988, JASA) | `lambda` | closed form: kernels chosen so `int lambda` integrates analytically | **full** — sums over all prior events |
| **RMTPP** (Du et al. 2016, KDD) | `lambda`, log-linear in elapsed time | closed form, *because* the functional form was chosen to integrate | RNN summary of full history |
| **Neural Hawkes / CT-LSTM** (Mei & Eisner 2017, NeurIPS) | `lambda` = softplus of a continuously-decaying LSTM state | **Monte Carlo** estimate of `int lambda` | full, via continuous-time LSTM |
| **Fully-neural intensity** (Omi, Ueda & Aihara 2019, NeurIPS) | the **cumulative hazard** `Lambda`, by a monotone net; `lambda = dLambda/dtau` | **exact by construction** — autodiff of the monotone net | RNN summary |
| **LogNormMix** (Shchur, Biloš & Günnemann 2020, ICLR) | **`f` directly**, a log-normal mixture | **exact, closed form** | RNN summary |
| **Transformer Hawkes** (Zuo, Jiang, Li, Zhao & Zha 2020, ICML) | `lambda` | **Monte Carlo** (their reference implementation also offers trapezoidal numerical integration) | attention over full history, `O(n^2)` |
| **Self-attentive Hawkes** (Zhang et al. 2020, ICML) | `lambda` | Monte Carlo | attention |
| **NSTPP** (Chen, Amos & Nickel 2021, ICLR) | `lambda`, with a CNF for the spatial part | ODE solve | full, via neural ODE |
| **DeepSTPP** (Zhou et al. 2022, L4DC) | `lambda` as a kernel mixture over a latent process | closed form for their chosen kernel | **fixed window** — 20 events in the EarthquakeNPP setup |
| **DSTPP** (Yuan et al. 2023, KDD) | diffusion over `(tau, s)` | ELBO / bound | transformer summary |
| **FlowQuake** | **`f` directly**: `f_t` by CNF, `f_s` and `f_m` closed form | `f_t` exact up to ODE error; `f_s`, `f_m` exact closed form | 30 relational features at 7 lags + 80 observation-anchored mixture components; **full history** in the neural-ETAS spatial head |

Four lessons this table teaches.

**(1) Modelling `lambda` mortgages you to an integral you did not choose.** ETAS
gets away with it because Ogata picked kernels whose integrals are elementary. A
neural `lambda` has no such guarantee, so you either estimate `int lambda` by
Monte Carlo (Neural Hawkes, THP) — injecting variance into the reported
likelihood, unacceptable when your claimed margin is 0.05 nats — or you build the
integral in structurally (Omi et al.'s monotone cumulative-hazard trick).

**(2) Modelling `f` moves the normalizer into 1-D density estimation**, where
*every* exact-likelihood family is cheap: mixtures, autoregressive, flows. That is
why FlowQuake's `tll` is exact. The published NPP baselines that lose to ETAS on
this benchmark largely lose on `sll`, not `tll`
([STACK.md Part III](../STACK.md#part-iii--the-benchmark-contract)).

**(3) What you give up by modelling `f`** — volunteer these. *No rate field over
continuous time*: you cannot ask "what is `lambda` at 3 p.m. tomorrow", only "what
is the density of the next gap", so every gridded forecast needs sequential
simulation ([flowquake/ntest.py](../flowquake/ntest.py)) with its own subtleties (the
truncated-first-event conditional at
[ntest.py:88-104](../flowquake/ntest.py#L88-L104)). *No additive decomposition*: ETAS
can say "this much rate came from Landers 1992"; `f_t` cannot. *No superposition*:
two Hawkes processes superpose by adding intensities; two `f`-models do not compose
at all.

**(4) FlowQuake did not ablate its own choice of temporal family.** With `d = 1`,
a conditional log-normal mixture (LogNormMix) would give an exact closed-form
`f_t` with no ODE, no RK4, and no step-count question — and
[scripts/baselines.py:39-52](../scripts/baselines.py#L39-L52) already computes
unconditional and AR(1) log-normal `tll` baselines, so the author was aware of the
family. There is **no committed artifact comparing a flow temporal head to a
conditional mixture temporal head.** That is an unjustified design choice, and it
is the cleanest single experiment a reviewer could demand. Do not pretend
otherwise; say "the flow is not shown to be necessary for the temporal head, only
sufficient."

---

## 10. Why the spatial and magnitude heads are not flows

### 10.1 The argument

Restate the mechanism precisely. A flow over `(x, y)` conditioned on a vector `c`
produces

```
f_s(x, y | c) = p_0(F_theta(x, y; c)) · |det J_{F_theta}(x, y; c)|
```

The density's shape *in absolute coordinates* is a function of `theta`, modulated
by `c`. Now impose FlowQuake's central constraint:
`c` excludes absolute `x, y` by construction
([model.py:32-35](../flowquake/model.py#L32-L35), `SAFE_TOKEN_DIMS`). The head still
has to place probability mass somewhere in California. It cannot get "where" from
`c`. Therefore **it must get "where" from `theta`** — the weights must encode the
fault map of the training era. That is memorization by construction, not by
accident.

Compare the kernel mixture ([heads.py:96-113](../flowquake/heads.py#L96-L113)):

```
f_s(s) = sum_j w_j · K_{d_j, q_j, rho_j, theta_j}( s - s_j )  +  w_unif/A  +  w_kde·kde(s)
```

The component centres `s_j` are **the locations of observed events supplied at
evaluation time** (`comp_xy` from `lastk`,
[model.py:99-121](../flowquake/model.py#L99-L121)). The learned MLP produces only
`(mixture logit, d, q, rho, theta)` from `(cond, log dt_j, m_j, log dist_j)` —
recency, magnitude, offset. Move the entire catalog 500 km east: every `s_j`
moves with it, every relational feature is unchanged, and the density moves too.
The learned parameters are **translation-equivariant** by construction.

The empirical shape of the failure is §4.3 of the manuscript, in
[runs/ablation_h/memorization_figure.json](../runs/ablation_h/memorization_figure.json):
with the whole-catalog embedding exposed at `h = 4`, train NLL reaches 4.14 (better
than ETAS's 7.26) while held-out NLL blows up to 19.65 (worse than the Poisson floor
of 13.26). And the kill shot: for every `h > 0`, the best held-out checkpoint is the
*first one ever evaluated*, at step 250
([runs/ablation_h/ablation_h.json](../runs/ablation_h/ablation_h.json)) — you cannot
early-stop your way out.

### 10.2 The limits of the argument — state these before you are asked

**(a) The dichotomy is equivariance vs. not, not flow vs. mixture.** Nothing stops
you from building a flow over the **displacement** `(x - x_last, y - y_last)`, or a
flow whose base distribution is itself an observation-anchored mixture. Such a flow
would be translation-equivariant and would not memorize geography. So "flows
memorize" is false as a general claim about flows; the true claim is "a flow over
*absolute coordinates* with non-positional conditioning must memorize." Whether
the repo's RealNVP probe was of the absolute or displacement kind **cannot be
determined** — the implementation is not in the committed tree, only
[runs/chk_realnvp.json](../runs/chk_realnvp.json).

**(b) FlowQuake's spatial density does contain absolute geography.** The `kde`
component is a train-era smoothed-seismicity grid built in
[data.py:255-273](../flowquake/data.py#L255-L273) and stored in `ckpt["stats"]`. It
is fitted, frozen, and pre-val-only — but it is absolutely positioned. The precise
claim is: *the learned parameters cannot fingerprint geography; a separately fitted,
frozen background map deliberately can.* Those are different statements and
conflating them is how you get caught. [STACK.md §7](../STACK.md#7-datapy--turning-a-catalog-into-tensors)
makes this distinction correctly; make sure you do too.

**(c) It is an inductive-bias argument, so it is contingent on data size.** With
`10^7` training events instead of `5.5 x 10^4`, a flow over absolute coordinates
might simply learn the true fault geometry and generalize, since the test-era fault
geometry is largely the same fault geometry. The argument's force comes from
**55,442 training events** and a heavy-tailed target, not from a theorem.
(`runs/mw_robustness.json` → `california.comcat_mc25_headline.train_events`. Do
not quote [STACK.md §6](../STACK.md#6-why-etas-is-hard-to-beat)'s "~70,000
training events" for this: that figure is `92,263 − 21,889 = 70,374`, everything
before `test_start`, and includes the auxiliary and validation eras —
[Ch. 5 §3.2](05-sequence-models-ssm.md#32-cost).)

**(d) Magnitude is a different case entirely.** `GRMagnitudeHead` is not a flow
because Gutenberg–Richter *says* the answer is an exponential
([heads.py:157-174](../flowquake/heads.py#L157-L174)). A one-parameter family with a
century of empirical support beats a flexible density model on 55k events, and
`log f(m) = log beta - beta(m - m_c)` is exact, one line, and differentiable. The
argument here is "the parametric form is known", not "flows memorize".

---

## 11. Mixture density networks, and why Gaussians are the wrong shape for aftershock distances

### 11.1 What an MDN is

Bishop's mixture density network (Bishop 1994, Aston University technical report):
a network outputs the parameters of a mixture,

```
f(y | x) = sum_k pi_k(x) · N( y ; mu_k(x), Sigma_k(x) ),    sum_k pi_k = 1
```

Exact likelihood, closed form, one forward pass. It is the obvious first thing to
try for a conditional density, and [runs/chk_mdn.json](../runs/chk_mdn.json) records
that it was tried.

### 11.2 The tail argument, and how far it actually goes

ETAS's spatial kernel (Chapter 3) is a power law:

```
K(r) = (q - 1) / (pi d^2) · ( 1 + r^2/d^2 )^{-q}
```

which is the form [heads.py:104-105](../flowquake/heads.py#L104-L105) implements.
Asymptotically `log K(r) ~ -2q log r`: **logarithmic in `log r`**. A Gaussian
component has `log N(r) ~ -r^2/(2 sigma^2)`: **quadratic in `r`**. Those are
qualitatively different tails, and the penalty for using the wrong one grows
without bound.

Concretely, take `d = 5` km, `q = 1.5` (near the repo's `q_init = 1.8` and inside
its `q_floor = 1.15` range), a two-component Gaussian mixture
`0.9 · N(0, 3^2) + 0.1 · N(0, 30^2)` on the plane, and the uniform background floor
that `heads.py` supplies. The benchmark's Poisson spatial baseline is
`sll = -13.7745`, which implies a region area `A = e^{13.7745} ≈ 9.6e5` km² —
this reads the "Poisson" baseline as uniform over the region, which is reasonable
but not verifiable from the committed tree. The head's `bg_frac_init = 0.35` is
split evenly between the
uniform and KDE background components at initialization
([heads.py:71-73](../flowquake/heads.py#L71-L73): *"each bg part starts at
bg_frac/2"*), so the uniform weight is `0.175` and the floor is
`log(0.175/A) = -15.52`.

| `r` (km) | `log K_powerlaw` | `log f_gauss-mix` | uniform floor | power law minus best-of-Gaussian |
|---|---|---|---|---|
| 0.5 | −5.07 | −4.15 | −15.52 | **−0.92** (Gaussian wins) |
| 5 | −6.10 | −5.52 | −15.52 | **−0.57** (Gaussian wins) |
| 10 | −7.47 | −9.46 | −15.52 | **+1.98** |
| 20 | −9.31 | −11.17 | −15.52 | **+1.86** |
| 50 | −11.98 | −12.33 | −15.52 | **+0.35** |
| 100 | −14.05 | −16.50 → floor −15.52 | −15.52 | **+1.47** |
| 150 | −15.26 | floor −15.52 | −15.52 | **+0.26** |
| 200 | −16.12 | floor −15.52 | −15.52 | −0.61 |
| 500 | −18.87 | floor −15.52 | −15.52 | −3.35 |

(Reproduce in ten lines: `logK(r) = log(q-1) - log pi - 2 log d - q·log1p(r²/d²)`,
`logG(r) = log( sum_k w_k/(2 pi s_k²)·exp(-r²/2s_k²) )`, floor `= log(0.175) -
13.7745`.)

Read this table honestly, because it does not say what the folk version says.

- **The far tail is not where the power law wins.** The Gaussian mixture falls
  below the uniform floor at `r ≈ 91` km and the power law does so at `r ≈ 163`
  km, so beyond ~163 km the background dominates *both* models. The uniform
  component in
  [heads.py:106](../flowquake/heads.py#L106) is precisely the device that stops the
  Gaussian's `e^{-r^2}` collapse from being catastrophic. So "Gaussian mixtures
  blow up on the tail" is *prevented by architecture*, not by kernel shape.
- **The power law wins in the 10–150 km band**, by 0.3–2.0 nats for an event that
  lands there. That is the real, and still large, effect. (The gap is not
  monotone — it dips near `r = 50` km, where the mixture's wide component happens
  to sit — which is itself a warning that a two-component illustration is a
  cartoon, not a theorem.)
- **At very short range a Gaussian is better.** Fitted MDNs will exploit that.

### 11.3 The stronger version: it is about components per centre

Here is the argument that actually holds. Fit `K` isotropic Gaussians (by EM, all
centred at the origin, free weights and scales) to samples from the truncated
power-law kernel (`d = 5` km, `q = 1.5`, truncated at 500 km), and measure the
excess cross-entropy in nats per event:

| `K` | mean `-log f_GMM` | **excess over the truth** | fitted scales (km) |
|---|---|---|---|
| 1 | 9.9698 | **+2.063** | 35 |
| 2 | 8.1354 | **+0.229** | 7, 80 |
| 3 | 7.9485 | **+0.042** | 5, 19, 118 |
| 5 | 7.9091 | **+0.002** | 3, 7, 18, 50, 167 |
| 8 | 7.9076 | **+0.001** | 2 … 178, log-spaced |

(True entropy of the truncated power law: 7.9067 nats, itself a Monte-Carlo
estimate on 4e5 draws, so the last digit of every row is noise. Reproduce with ~20
lines of numpy: sample `r` by inverse CDF `F(u) = 1 - (1+u^2)^{1-q}` with
`u = r/d`, rejecting/truncating at 500 km, then run EM with the 2-D isotropic
update `sigma_k^2 = sum_i g_ik r_i^2 / (2 sum_i g_ik)`.)

**So the naive tail argument is weak: three log-spaced Gaussians reproduce a
truncated power law to 0.04 nats.** If someone hands you a free `K`-component MDN
at a single centre, "Gaussians can't do power laws" is nearly vacuous.

**The argument that survives** is about how FlowQuake spends its components.
`heads.py` places **one kernel per observed parent** — 80 components at 80
*different* locations, not 80 components stacked at one location providing
multi-scale radial coverage. Each parent therefore operates in the `K = 1` regime,
where the excess is **+2.06 nats/event**. To match the power law with Gaussians you
would need ~3 Gaussians *per parent*, i.e. 240 components and a 3x larger
per-component MLP, to buy back a shape that two scalars `(d, q)` give you for free.
That is the honest form of "Gaussians are the wrong shape here", and it is a
statement about parameter efficiency under an anchored-mixture inductive bias, not
about tails per se.

### 11.4 What this implies about `chk_mdn.json` and `chk_realnvp.json`

The measured gaps in those artifacts are **−3.89 and −3.79 nats** versus ETAS. The
kernel-shape effect estimated above is ~2 nats at most in the `K = 1` regime, and
those probes ran for 400 steps. So the shape argument explains **at most half** of
the observed gap; the rest is training budget and, per
[model.py:12-15](../flowquake/model.py#L12-L15), mis-location from weight-anchored
geography. Do not present those two artifacts as evidence for the tail argument.
They are evidence that "obvious first thing, 80 seconds of training, badly behind"
— nothing more.

---

## 12. Worked example 1 — an exactly solvable 1-D flow-matching model

This mirrors `tests/test_flow.py::test_log_prob_matches_gaussian_1d`
([tests/test_flow.py:19-32](../tests/test_flow.py#L19-L32)) and gives you the
closed-form answer the test only checks numerically. Everything below is
reproducible with `math` alone — no torch required.

### 12.1 Setup

Target `q = N(m, s^2)` with `m = 1.5`, `s = 0.7` (the test's values). Base
`p_0 = N(0, 1)`. Path with `sigma_min = 0` (the test uses `CondFlow`'s default):

```
z_t = (1 - t_flow) z0 + t_flow · u,     z0 ~ N(0,1),  u ~ N(m, s^2),  independent
```

### 12.2 Derive the marginal velocity field in closed form

`z_t` is a linear combination of independent Gaussians, hence Gaussian:

```
E[z_t]   = t_flow · m
Var(z_t) = (1 - t_flow)^2 · 1 + t_flow^2 · s^2  =:  sigma_t^2
```

The marginal field is `w_t(z) = E[ u - z0 | z_t = z ]`. Because `(z_t, u - z0)` is
jointly Gaussian, that conditional expectation is the linear regression:

```
E[u - z0]        = m
Cov(z_t, u - z0) = Cov((1-t)z0 + t·u,  u - z0)  =  t·s^2 - (1-t)·1
```

so

```
              t_flow·s^2 - (1 - t_flow)
w_t(z) = m +  ------------------------- · ( z - t_flow·m )   =:  a(t_flow)·z + b(t_flow)
                     sigma_t^2
```

with `b(t) = m(1 - a(t)·t)`. The field is **affine in `z`**, so `div w_t = a(t)`.

### 12.3 The key cancellation

Differentiate `sigma_t^2 = (1-t)^2 + t^2 s^2`:

```
d(sigma_t^2)/dt = -2(1-t) + 2 t s^2 = 2 [ t s^2 - (1-t) ]
```

Therefore

```
a(t) = [ t s^2 - (1-t) ] / sigma_t^2 = (1/2) · (d sigma_t^2/dt) / sigma_t^2
     = d/dt log sigma_t
```

and the divergence integral collapses:

```
int_0^1 div w_t dt_flow = log sigma_1 - log sigma_0 = log s - log 1 = log s
```

With `s = 0.7`, that is `log 0.7 = -0.356675`.

### 12.4 Assemble the density and check

Because the flow is affine and the marginal is `N(t·m, sigma_t^2)`, the trajectory
is `z(t) = t·m + sigma_t · z(0)`, so at `t_flow = 1`, `z(0) = (u - m)/s`. Plug into
the CNF likelihood formula:

```
log p_1(u) = log p_0(z(0)) - int_0^1 div w_t dt_flow
           = -0.5·((u-m)/s)^2 - 0.5·log(2 pi)  -  log s
```

which is **exactly `log N(u; m, s^2)`**. The machinery reproduces the analytic
Gaussian density with no approximation.

Numerically, at `u = m = 1.5`:

```
-0.5·0^2 - 0.5·log(2 pi) - log(0.7) = 0 - 0.918939 + 0.356675 = -0.562264
```

and at `u = m + s = 2.2`:

```
-0.5·1^2 - 0.918939 + 0.356675 = -1.062264
```

### 12.5 Run the actual RK4 integrator on it

Now feed the closed-form `w_t` into exactly `flow.py`'s backward RK4 scheme (same
sign convention, same four stages) and watch the error:

| `steps` | `log p_1(2.2)` | error vs −1.0622635893 | ratio to previous |
|---|---|---|---|
| 4 | −1.0621494793 | +1.141e−04 | — |
| 8 | −1.0622593909 | +4.198e−06 | 27.2 |
| 16 | −1.0622634297 | +1.596e−07 | 26.3 |
| 32 | −1.0622635826 | +6.679e−09 | 23.9 |
| 64 | −1.0622635890 | +3.137e−10 | 21.3 |
| 96 | −1.0622635892 | +5.511e−11 | 5.7 |
| 128 | −1.0622635892 | +1.635e−11 | 3.4 |

and the recovered latent `z(0)` equals `(2.2 - 1.5)/0.7 = 1.000000` to six decimals
from 16 steps onward.

Read the ratios carefully, because the last two rows are **not doublings**. For a
step-count ratio `k` the expected error ratio is `k^4`: `2^4 = 16` for the
doublings, `(96/64)^4 = 5.06` for the 64→96 row, `(128/96)^4 = 3.16` for the
96→128 row. Observed: 27.2, 26.3, 23.9, 21.3 for the doublings — *faster* than
`h^4`, because a same-sign `h^5` term still inflates the coarse-step errors — then
5.7 and 3.4, both slightly above their `k^4` targets. So the whole table is
consistent with clean fourth-order convergence; there is no round-off
contamination here. (Continuing the sweep in float64, the doubling ratios drift
*down* toward 16 — 19.2 at 64→128, 17.7 at 128→256, 17.1 at 256→512 — and
round-off only takes over past ~1024 steps, where the error reaches `1e-14`.)

**At 64 steps the error is 3e−10 nats**, about 20x smaller than the ~7e−9
estimated from the repo's own artifacts in §8.3 — which is the direction you would
expect, since the repo integrates a trained MLP rather than an affine field. Two
different routes agreeing to within a factor of 20 at 1e−9 is the point; neither
number is anywhere near 0.0533.

### 12.6 What `tests/test_flow.py` actually guards

Four separate invariants, and it is worth knowing which is which:

| test | invariant it protects |
|---|---|
| `test_log_prob_matches_gaussian_1d` | **the whole pipeline is calibrated**: FM training + backward ODE + logdet accumulation + base-density term jointly reproduce a known density to 0.15 nats. Catches sign errors, missing `log 2pi`, and a mis-specified velocity target. |
| `test_log_prob_matches_gaussian_2d_conditional` | the same at `d = 2` *and* that the conditioning vector actually reaches the velocity field (the target mean depends on `c`). |
| `test_sample_moments` | the **forward** integrator is consistent with the **backward** one — trains on `N(-0.8, 1.3^2)` and checks sampled mean and std to 0.1. |
| `test_density_integrates_to_one` | **normalization, on an untrained net with deliberately non-zero random weights** (`std=0.3`). Grid-integrates `exp(log_prob)` over `[-8, 8]` with `trapezoid` and asserts `\|integral - 1\| < 0.02`. This is the one that would catch a Jacobian sign flip, because a flipped sign still gives plausible-looking numbers but does not integrate to 1. |

Note the fourth test does not depend on training at all — it is a pure statement
about the change-of-variables implementation. That is good test design and worth
pointing at.

*(I could not execute these tests in the environment this chapter was written in —
no `torch` is installed on the available interpreter. The closed-form derivation
and the RK4 convergence table above were computed in pure Python and stand on their
own; the test assertions are quoted from the file.)*

---

## 13. Worked example 2 — the unit conversion, term by term

This is the exercise [STACK.md Part X](../STACK.md#exercises-that-will-actually-teach-you-the-stack)
sets first, and it is the one that decides whether you understand what `tll` is.

### 13.1 The claim

The flow models `u = (log tau - mu) / sigma`, where `mu = log_tau_mean` and
`sigma = log_tau_std` are the train-era normalization constants
([data.py:220-225](../flowquake/data.py#L220-L225)). We need `f(tau)` in units of
1/day. The claim is

```
log f(tau)  =  log p_U(u)  -  log sigma  -  log tau
```

### 13.2 Derivation, one change of variables at a time

Write `l := log tau`, so `tau = e^l`, and `u = (l - mu)/sigma`.

**Step 1: `u -> l`.** The map `l |-> u` is affine with `du/dl = 1/sigma`. By the
1-D change of variables (`(CoV)` with `d = 1`):

```
p_L(l) = p_U(u) · |du/dl| = p_U(u) / sigma
log p_L(l) = log p_U(u) - log sigma
```

*Sanity:* `sigma > 1` spreads the standardized variable out, so the density per
unit of `l` must be *smaller*. `- log sigma` is negative. ✓

**Step 2: `l -> tau`.** The map `tau |-> l = log tau` has `dl/dtau = 1/tau`:

```
p_T(tau) = p_L(l) · |dl/dtau| = p_L(log tau) / tau
log f(tau) = log p_L(log tau) - log tau
```

*Sanity:* a fixed interval in `log tau` near `tau = 100` days spans 100x more
real time than the same interval near `tau = 1` day, so the density per day must
be 100x smaller there. `- log tau` supplies that. ✓

**Combine:**

```
log f(tau) = log p_U(u) - log sigma - log tau        ∎
```

### 13.3 Match it to the code

[model.py:233-235](../flowquake/model.py#L233-L235), verbatim:

```python
lp_t    = self.head_t.log_prob(u_t, cs, steps=steps)
log_tau = tgt[i:i+event_chunk, 0] * st["log_tau_std"] + st["log_tau_mean"]
tll.append(lp_t - math.log(st["log_tau_std"]) - log_tau)
```

Line 1 is `log p_U(u)`. Line 2 un-normalizes to recover `log tau` from the stored
normalized target. Line 3 is the formula, term for term. ✓

A second, independent implementation of the same conversion sits in
[scripts/baselines.py:39-40](../scripts/baselines.py#L39-L40), where the
unconditional-lognormal baseline computes

```python
tll_uncond = (-0.5*((lt_val - mu0)/s0)**2 - np.log(s0)
              - 0.5*np.log(2*np.pi) - lt_val).mean()
```

The first two terms plus `-0.5 log 2pi` are `log p_U(u)` for a standard normal;
`- np.log(s0)` is `- log sigma`; `- lt_val` is `- log tau`. Two files, same
formula, written years apart in different styles — that agreement is worth more
than either alone.

### 13.4 What each omission costs you

| omitted term | what you report instead | size of the error | why it is fatal |
|---|---|---|---|
| `- log sigma` | density per unit of *normalized* log-day | constant `+ log sigma`; for `sigma ≈ 2.5` that is **+0.916 nats/event** | 17x the entire margin over ETAS (`+0.0533`). You would "beat" ETAS by an accounting error and the number would not even be a density. |
| `- log tau` | density per unit of `log tau` | `+ log tau`, which is **event-dependent** — mean of order `-1.5` to `-2.1` nats on the ComCat test window | Worse than a constant: it *reweights events*. Short-gap events get a large positive bonus, long-gap events a penalty. A model that is good at short gaps would win for the wrong reason. |
| both | density in normalized-log units | `+ log sigma + log tau` | You are comparing apples to a different fruit entirely; the number is not commensurable with ETAS's `TLL` and the paired comparison in [evaluate.py:28-51](../flowquake/evaluate.py#L28-L51) is meaningless. |

The "mean of order −1.5 to −2.1" estimate: the ComCat_25 test window is
2007-01-01 → 2020-01-17 = **4,764 days** (the dates are `test_start`/`test_end` in
[configs/final_s1554.yaml](../configs/final_s1554.yaml), also stated in
[README.md](../README.md)) carrying **21,889 events** (`n_events` in
[runs/eval_test_N1.json](../runs/eval_test_N1.json)), so the mean rate is
`lambda = 4.5947` events/day and the mean gap is `0.2176` days.
`log(0.2176) = -1.525`. For a genuine exponential gap distribution
`E[log tau] = -log lambda - gamma_Euler = -1.525 - 0.577 = -2.102`.

### 13.5 A free consistency check on the whole apparatus

For a homogeneous Poisson process of rate `lambda`, `f(tau) = lambda e^{-lambda
tau}`, so

```
E[ log f(tau) ]  =  log lambda  -  lambda · E[tau]  =  log lambda - 1
```

With `lambda = 21889/4764 = 4.5947`:

```
log(4.5947) - 1 = 1.5249 - 1 = 0.5249
```

The benchmark's published Poisson temporal baseline is **`tll = 0.5126`**
([README.md](../README.md), from
`reference/Experiments/ETAS/output_data_ComCat_25/ll_scores.json`). Agreement to
**0.0123 nats** — the residual being that the benchmark's Poisson rate is fitted on
the training era, not the test era. If your unit conversion were wrong by any of
the terms in §13.4, this check would fail by 0.9 nats or more. It is the cheapest
end-to-end validation of `tll`'s units in the whole project, and it takes one line.

For scale, the same window: `ETAS tll = 1.4343`, `FlowQuake tll = 1.4876`.
So ETAS is `0.9217` nats/event above the Poisson floor and FlowQuake is `0.9750`
above it — i.e. **FlowQuake's win is 5.8% of the distance ETAS itself travelled
from the trivial baseline.** State the margin that way and nobody can accuse you of
overselling it.

---

## 14. How this shows up in FlowQuake

Do not re-read [STACK.md §9](../STACK.md#9-flowpy--the-temporal-head) — it is the code
walkthrough and this chapter is its theory. The map from theory to artifact:

| theory (this chapter) | code | artifact |
|---|---|---|
| §3.4 exact CNF likelihood | [flow.py:109-134](../flowquake/flow.py#L109-L134) `log_prob` | `tll` in every `runs/*/eval_test.json` |
| §4.3 exact `d = 1` divergence | [flow.py:94-107](../flowquake/flow.py#L94-L107) `_vel_and_div` | — |
| §6.3 CFM gradient equivalence | [flow.py:70-78](../flowquake/flow.py#L70-L78) `fm_loss` (3 lines, no ODE) | `loss_t` in training logs |
| §7.1 the `sigma_min` path | [flow.py:76-78](../flowquake/flow.py#L76-L78) | `sigma_min: [0.02, …]` in every config |
| §7.2 the bandwidth floor / atom ceiling | `sigma_min[0]` only — see §7.3 | **no artifact**; the check is not run |
| §8.1 RK4 | [flow.py:118-131](../flowquake/flow.py#L118-L131) (backward), [flow.py:85-91](../flowquake/flow.py#L85-L91) (forward) | `ode_steps` field in eval JSONs |
| §8.3 step-count convergence | [scripts/diag_ll.py:34-37](../scripts/diag_ll.py#L34-L37) sweeps 32/128/512 | **no artifact committed**; only the accidental 64-vs-96 pair |
| §10 anchored mixture instead of a flow | [heads.py:96-113](../flowquake/heads.py#L96-L113), [model.py:12-15](../flowquake/model.py#L12-L15) | [runs/ablation_h/](../runs/ablation_h/) |
| §11 MDN / RealNVP probes | *(implementations not in the tree)* | [runs/chk_mdn.json](../runs/chk_mdn.json), [runs/chk_realnvp.json](../runs/chk_realnvp.json) |
| §13 the unit conversion | [model.py:233-235](../flowquake/model.py#L233-L235); independently [scripts/baselines.py:39-40](../scripts/baselines.py#L39-L40) | every `tll` number in the repo |

**Sizes worth memorizing.** The temporal head is `CondFlow(dim=1, cond_dim=30,
hidden=96, n_layers=3)`. `SAFE_TOKEN_DIMS` has 30 entries and `h_bottleneck = 0`,
so `cond_dim = 30`; `TimeEmbed(n_freq=4)` contributes `2·4 + 1 = 9` dims; the input
width is `1 + 9 + 30 = 40`. Parameter count:

```
Linear(40, 96) : 40·96 + 96  =  3,936
Linear(96, 96) : 96·96 + 96  =  9,312     (x2)
Linear(96,  1) : 96· 1 +  1  =      97
                              ---------
                                 22,657
```

**22,657 parameters** produce the temporal win over a nine-parameter ETAS
inversion — `(mu, k0, a, c, omega, tau, d, gamma, rho)` — that took 3–4 CPU-hours
per region to fit ([STACK.md §5](../STACK.md)). That framing — a small head, a big
conditioning-feature design — is the honest description of where the work is.
Compare the discarded spatial probes: 468k parameters (RealNVP) and 275k (MDN),
both far worse. Capacity was never the bottleneck.

---

## 15. Common misconceptions

**1. "Normalizing flows are approximate because they use a neural network."**
*Actually:* the likelihood is exact by construction — the change of variables is
an identity, not an approximation. What is approximate is (a) how close the learned
density is to the truth, which is true of every model, and (b) for CNFs, the ODE
solver's discretization error, quantified in §8.3 at ~7e-9 nats/event here.
*Why it matters:* "exact likelihood" is the reason this family is used for
scientific density estimation at all, and the reason `tll` is comparable to ETAS's
`TLL` at all. Conceding exactness concedes the point of the design.

**2. "Flow matching is a way of training diffusion models."**
*Actually:* flow matching trains a **deterministic** ODE, with a straight-line
conditional path and no forward noising process. Diffusion's probability-flow ODE
and flow matching's ODE are related — both transport noise to data, both can be
written as continuity equations — but the objectives, the paths, and the sampling
procedures are different. Flow matching came *after* diffusion and is best
understood as the more general framework (diffusion corresponds to a particular
curved probability path).
*Why it matters:* a professor asking "how is this different from a diffusion
model?" is testing whether you have a mechanism or a vibe.

**3. "`t` in `z_t` is time since the last earthquake."**
*Actually:* `t_flow in [0,1]` is the flow's integration variable, entirely
internal to the density model. The earthquake gap is `tau`, and the flow models
`u = (log tau - mu)/sigma` as a *static* datum. There is no relationship between
the two clocks. This chapter writes `t_flow` everywhere for exactly this reason.
*Why it matters:* it is the single most common confusion when a TPP person meets a
flow person, and getting it wrong makes every subsequent sentence incoherent.

**4. "Hutchinson's estimator is why this is fast."**
*Actually:* Hutchinson appears nowhere in this repository. At `d = 1` the trace is
one derivative, computed exactly by `torch.func.jacrev` with zero variance
([flow.py:94-107](../flowquake/flow.py#L94-L107)).
*Why it matters:* claiming a technique you do not use is a gift to an examiner.
The correct statement is "we do not need it, because `d = 1`".

**5. "`sigma_min` is a numerical-stability hack."**
*Actually:* it changes the model. The optimum of the CFM objective with
`sigma_min > 0` is `q ∗ N(0, sigma_min^2)`, a *different distribution* from `q`.
It caps the reportable density at any point at `1/(sigma_min sqrt(2pi))`. That is
a modelling decision with a measurable effect on the score, not a stability tweak.
*Why it matters:* it is the honest answer to "couldn't you cheat on discretized
data?", and it also means every reported `tll` is a likelihood under a slightly
smoothed model — a caveat the manuscript does not state.

**6. "`sigma_min` guarantees the reported likelihood cannot blow up."**
*Actually:* it bounds the density of the **global minimizer** of the objective. A
finite network trained for finite steps is not that minimizer and is not
constrained by the bound; nothing in the code clamps `log_prob`'s output.
`sigma_min` moves the target, it does not clamp the model.
*Why it matters:* this is the precise form of the answer to hostile question H2.
Over-claiming here is worse than not raising it.

**7. "The spatial head isn't a flow because 2-D divergences are expensive."**
*Actually:* [flow.py:6-8](../flowquake/flow.py#L6-L8) says 1–2 dimensions are both
cheap, and `tests/test_flow.py` runs a 2-D conditional flow in the standard suite.
The real reason is the inductive-bias argument of §10, stated at
[model.py:12-15](../flowquake/model.py#L12-L15). [STACK.md §9](../STACK.md#9-flowpy--the-temporal-head)
gets this wrong.
*Why it matters:* a reviewer who opens the test file catches you in 30 seconds.

**8. "Gaussian mixtures can't represent power-law tails."**
*Actually:* three log-spaced Gaussians reproduce a truncated power-law kernel to
0.04 nats/event (§11.3). The correct claim is about *components per anchor*: at
`K = 1` per parent — which is what an observation-anchored mixture spends — the
excess is 2.06 nats, and the power law buys that back with two scalars.
*Why it matters:* the naive version is easy to refute in one line of numpy, and
refuting it makes your other tail arguments look sloppy.

**9. "Exact likelihood means the samples are exactly from the model."**
*Actually:* `sample` (forward RK4, 24–32 steps) and `log_prob` (backward RK4, 64
steps) are different discrete maps and are not exact inverses. The simulated
catalogs that feed the CSEP tests come from a slightly different distribution than
the one whose density is reported (§8.5). The gap is ~1e-7 nats here, but it is
non-zero.
*Why it matters:* it is the seam between the likelihood claim and the calibration
claim, and it is exactly where a careful examiner will push.

**10. "Modelling `f` instead of `lambda` is a presentational choice."**
*Actually:* it determines whether your normalizer is a 1-D density integral (free)
or a path integral of a history-dependent rate (hard), and it determines whether
you can produce a rate field without simulation (you cannot). It is why FlowQuake's
`tll` is exact and why [flowquake/ntest.py](../flowquake/ntest.py) exists at all.

---

## 16. Questions a professor will ask

### Q1. State the instantaneous change of variables theorem and prove it.

`d/dt_flow log p_t(z(t_flow)) = -div v(z(t_flow), t_flow)` along a trajectory of
`dz/dt_flow = v`. Proof: `J_t = d phi_t/d z0` satisfies `dJ_t/dt = (dv/dz) J_t`
with `J_0 = I`; Jacobi's formula gives `d/dt log det J_t = tr(J_t^{-1}(dv/dz)J_t)
= tr(dv/dz) = div v`; the change of variables gives `log p_t(z(t)) = log p_0(z0) -
log det J_t`; differentiate. Assumptions: `v` uniformly Lipschitz in `z` and
continuous in `t` (Picard–Lindelöf: unique solutions, so `phi_t` is a bijection and
`det J_t > 0`), and `C^1` in `z` so the trace exists. FlowQuake's `v` is a
SiLU MLP, hence `C^inf` and globally Lipschitz with constant
`prod_l ||W_l||_op · 1.0998^{n_layers}`, so the hypotheses hold unconditionally.

### Q2. Walk me through `log_prob`'s return line, including the direction of integration.

You are given the datum `u = z(1)` and need `z(0)` to evaluate the base density, so
the solve must run backwards; the forward map is the generator. Uniqueness of ODE
solutions makes the backward solve well posed with the same Lipschitz constant. The
code substitutes `s = 1 - t_flow` and integrates `s` forward with `dz/ds = -v`
([flow.py:124](../flowquake/flow.py#L124)), so `logdet` accumulates
`int_0^1 div v dt_flow` with the correct sign. Then
[flow.py:133](../flowquake/flow.py#L133) computes
`log_p0 = -0.5*(z^2.sum(-1) + dim·log 2pi) = log N(z(0); 0, I_d)`, and line 134
returns `log_p0 - logdet` — §3.4 verbatim.

### Q3. State the flow-matching theorem.

`grad_theta L_FM = grad_theta L_CFM`, where `L_FM` regresses onto the intractable
marginal field `w_t(z)` and `L_CFM` onto the tractable conditional field
`w_t(z|u)`. They differ by a `theta`-independent constant equal to the expected
conditional variance of `w_t(z|u)` given `z_t = z`. The mechanism is that
`w_t(z) = E[w_t(z|u) | z_t = z]` (the marginalization lemma), and regressing on a
random variable and on its conditional mean have the same minimizer and the same
gradient. See §6.3 for the two-term expansion.

### Q4. Verify the velocity target on FlowQuake's specific path.

`z_t = (1 - (1-sigma_min) t_flow) z0 + t_flow u`. Since `z0` and `u` are fixed
along a conditional path, `dz_t/dt_flow = u - (1-sigma_min) z0`, matching
[flow.py:78](../flowquake/flow.py#L78) exactly. Substituting `z0` out gives
`(u - (1-sigma_min) z_t)/(1 - (1-sigma_min) t_flow)`, Lipman et al.'s OT-path
conditional field. Endpoints: `z_0 = z0 ~ N(0,I)`; `z_1 = u + sigma_min z0`.

### Q5. What is the model's density at `t_flow = 1`, exactly?

`p_1 = q ∗ N(0, sigma_min^2 I)` — the data distribution convolved with an isotropic
Gaussian of width `sigma_min`. Not `q`. Every reported `tll` is a likelihood under
that smoothed model.

### Q6. Why is the temporal head a flow but the spatial head a closed-form mixture?

Not because of divergence cost — 2-D exact traces are two backward passes and
`tests/test_flow.py` runs a 2-D flow. The reason
([model.py:12-15](../flowquake/model.py#L12-L15)) is inductive bias: a flow over
absolute `(x,y)` with conditioning that excludes absolute coordinates must encode
"where earthquakes happen" in its weights, i.e. memorize training-era geography.
An anchored mixture places components at observed event locations supplied at
evaluation time, so it is translation-equivariant and its geography moves with the
data. Time has no such structure — there is no "where" to memorize in a scalar
gap — so the flow's flexibility is safe there.

### Q7. What is the effect of the ODE step count on the reported number?

Reported results use 64 steps; validation used 32; `evaluate.py`'s CLI default is
96 and one artifact used it. RK4's global error is `O(h^4)`. The repo's two
artifacts from the same seed and config at 64 and 96 steps differ by `5.5986e-9`
nats/event in the temporal paired gain, with the spatial (closed-form) score
bit-identical; Richardson extrapolation gives a 64-step error of ~`7e-9`
nats/event. Against a `0.0533` margin, that is a factor of `7.6e6`. But this is a
two-point check on inferred-identical weights, not a sweep;
[scripts/diag_ll.py](../scripts/diag_ll.py) already sweeps 32/128/512 and its output
was never committed.

### Q8. Why model `log tau` rather than `tau`?

Gaps span nine orders of magnitude (milliseconds to years). A power-law tail in
`tau` becomes a roughly-tractable shape in `log tau`, and a Gaussian-base flow with
a bounded-Lipschitz velocity would have to travel enormous distances to reach the
tail in linear `tau`. The cost is the `-log tau` Jacobian term (§13), which is
event-dependent and must not be forgotten.

### Q9. What breaks if the MLP in the spatial head sees the target location `s`?

The normalizer. `int sum_j w_j(s) K_j(s) ds != sum_j w_j int K_j` when `w_j`
depends on `s`. The closed-form denominator `Z_j = pi/(rho_j d_j^{rho_j})` in
[neural_etas.py](../flowquake/neural_etas.py) stops being the integral of the
numerator, and you are left with an unnormalized energy whose partition function
requires a 2-D quadrature per event. Retraining does not fix it — it is a
structural property of the expression, not a fitting problem.

### Q10. Why is Hutchinson's estimator unbiased, and what is its variance?

`E[eps^T A eps] = sum_{ij} A_ij E[eps_i eps_j] = sum_{ij} A_ij delta_ij = tr(A)`
whenever `E[eps]=0, Cov(eps)=I`. For Rademacher `eps`,
`Var = sum_{i!=j}(A_ij^2 + A_ij A_ji)`, which is `2(||A||_F^2 - sum_i A_ii^2)` for
symmetric `A` and **zero at `d = 1`**. Gaussian `eps` gives `2||A||_F^2` for
symmetric `A` — larger by `2 sum_i A_ii^2`, which is why Rademacher is the
default (and is what Hutchinson 1989 actually proposed).

### Q11. Why not train the CNF by maximum likelihood with the adjoint method?

Memory-efficient but slow, and the reverse-time reconstruction of the forward
trajectory is not exactly reversible, so gradients inherit solver error. Adaptive
solvers also let the number of function evaluations drift upward during training as
the field becomes stiffer. Flow matching removes the ODE from training entirely:
one MSE per minibatch ([flow.py:70-78](../flowquake/flow.py#L70-L78)), which is why
the whole model trains on one GPU in hours.

---

**Now the hostile ones.**

### H1. "You claim exact likelihoods, but you only report 64-step numbers and never showed convergence. How do I know your +0.053 nats isn't a solver artifact?"

You are right that the repo does not contain a convergence artifact, and that is a
gap. What it does contain is an accidental controlled comparison: two runs of the
same config and seed evaluated at 64 and 96 steps
([runs/final_s1555/eval_test.json](../runs/final_s1555/eval_test.json),
[runs/comcat25_s1555/eval_test.json](../runs/comcat25_s1555/eval_test.json)), whose
closed-form spatial scores are bit-identical (so the weights match) and whose
temporal paired gain differs by `5.6e-9` nats/event. Richardson extrapolation under
RK4's `O(h^4)` gives a 64-step error of `7e-9` — seven million times smaller than
the margin. Three caveats I will not hide: it is a two-point check, the weight
identity is inferred rather than verified because no checkpoint is committed, and
it does not cover the 32-step validation used for model selection.
[scripts/diag_ll.py:34-37](../scripts/diag_ll.py#L34-L37) already sweeps 32/128/512;
running it on the three production seeds and committing the JSON would close this
in an hour, and it should be done before submission.

### H2. "You floor `tau` at 1e-7 days. That creates an atom. A flexible density model spikes on atoms. Your entire temporal win is 0.29% of events sitting on that spike."

That is a genuinely dangerous mechanism and the arithmetic is exactly as you say.
The ceiling on a floored event is `log(1/(sigma_min sqrt(2pi))) - log sigma_LT -
log(1e-7) = 2.993 - log sigma_LT + 16.118`, roughly `+18.2` nats for plausible
`sigma_LT`, and `0.0533/18.2 = 0.29%` of events at the ceiling would reproduce the
whole margin. Three things push against it, none of them dispositive.

First, `sigma_min = 0.02` caps the *optimum's* density — though I should be precise:
it caps the CFM optimum, not the trained network, so it moves the target rather
than clamping the output. Second, the win replicates across five California
catalogs with `m_c` from 0.6 to 2.5 and therefore wildly different short-gap
populations ([runs/fullsuite_summary.json](../runs/fullsuite_summary.json), 3-seed
means, five for five), across three seeds, and out of time on a 2020–2026 window of
10,187 unseen events (`dT = +0.0574 [0.0376, 0.0819]`,
[runs/total_win.json](../runs/total_win.json)); a floor artifact would not be expected
to transport like that. Third, ETAS is scored on the same floored gaps, so any
purely mechanical bonus is at least partly shared.

But the direct check — the fraction of test events at `tau = TAU_FLOOR_DAYS`, and
the upper tail of the per-event `tll` histogram — is **not in the repository**,
because per-event score CSVs and checkpoints are both gitignored
([WORKING.md](../WORKING.md)). It is two lines of pandas on the catalog plus one
re-score with the floored events dropped. Until it is run, this objection stands
unanswered, and I would not defend the temporal claim in front of a referee without
it.

### H3. "Your justification for not using a flow in space is that flows memorize. But that's an argument about absolute-coordinate parameterization, not about flows. Build me a displacement flow and your argument evaporates."

Correct, and I would concede it immediately. The defensible claim is: *a flow over
absolute coordinates, conditioned on features that exclude absolute coordinates,
must place its mass using its weights and therefore memorizes training-era
geography.* A flow over `(x - x_last, y - y_last)`, or a flow whose base is itself
an observation-anchored mixture, would be translation-equivariant and would not be
subject to that argument. The repo does not test that architecture — the RealNVP
probe's implementation is not even in the committed tree, only
[runs/chk_realnvp.json](../runs/chk_realnvp.json), so I cannot tell you which kind it
was. What the repo does establish is that *some* channel carrying absolute
geography into the heads produces catastrophic memorization
([runs/ablation_h/](../runs/ablation_h/): train NLL 4.14 vs held-out 19.65 at `h=4`,
and the best held-out checkpoint at every `h > 0` is the first one evaluated). That
is evidence about the mechanism, not about flows. A displacement-flow spatial head
is the obvious missing ablation and I would run it.

### H4. "You use a flow for a one-dimensional density. A conditional log-normal mixture gives you an exact closed-form likelihood with no ODE, no RK4, no step-count question, and fewer parameters. Why is any of this machinery justified?"

It is not shown to be. There is no committed artifact comparing the flow temporal
head to a conditional mixture temporal head; the only mixture baselines in the repo
are the *unconditional* and AR(1) log-normal ones in
[scripts/baselines.py](../scripts/baselines.py), which are calibration references, not
architectural ablations. So the honest statement is that the flow is sufficient for
the temporal win, not that it is necessary. The arguments I *can* make are weak:
LogNormMix (Shchur et al. 2020, ICLR) has to fit a mixture shape, and the target
here is a very heavy-tailed multimodal `log tau` distribution mixing background,
aftershock cascade and quiescence regimes; a flow does not commit to a component
count. But "does not commit to a component count" is a story, not a measurement.
The experiment is cheap — swap `CondFlow` for a mixture head with the same
`cond_dim`, retrain the three production seeds — and it should be in the paper.

### H5. "Your `sigma_min` config comment claims bandwidth floors on all three heads. Only one is wired up. What else in this repo is documented but not implemented?"

The `sigma_min` case is real: `sigma_min[1]` and `sigma_min[2]` never reach any
head ([model.py:85-92](../flowquake/model.py#L85-L92)), and `mag_dequant` is
explicitly dead ([model.py:61](../flowquake/model.py#L61)). Neither affects a reported
number — the spatial head has its own `d_floor`/`q_floor`, the magnitude head has
the `+0.005` shift — but the config comment describes a mechanism that does not
exist and should be corrected. Separately, both STACK.md and MANUSCRIPT.md describe
that `+0.005` as a half-bin for a **0.1**-magnitude grid, which is arithmetically
wrong (half of 0.1 is 0.05); the value is the half-bin of a 0.01 grid, matching
`mag_dequant`'s own comment. On "what else": the repo audits itself in
[results/CLAIMS.md](../results/CLAIMS.md) — of 142 traced claim rows, 8 distinct claims
are contradicted by their artifact and 12 have no committed backing
([WORKING.md](../WORKING.md)). I would rather point you at that file than pretend the
count is zero.

### H6. "The likelihood you report and the samples you feed CSEP come from different models. Explain."

RK4 is not a symmetric integrator, so the discrete forward map (`sample`, 24–32
steps) is not the exact inverse of the discrete backward map (`log_prob`, 64
steps). The reported `tll` is the density of the exact-ODE model approximated at
`h = 1/64`; the simulated catalogs come from the `h = 1/24` forward discretization.
They agree as `h -> 0`. Numerically the gap is bounded by the same convergence
analysis — ~1e-7 nats at 32 steps — so it does not move any number in the paper.
Conceptually it is a real seam between the likelihood claim (§4.1/§4.4 of the
manuscript) and the calibration claim (§4.2, the CSEP tests), and the clean fix is
to use the same step count in both, or a symmetric integrator.

### Q12. Two implementation details: why zero-init the last layer, and why Fourier features on `t_flow`?

[flow.py:61-62](../flowquake/flow.py#L61-L62) zeroes the final `Linear`'s weight and
bias, so at step 0 `v ≡ 0`, the ODE is the identity map, `logdet = 0`, and
`p_1 = p_0 = N(0,1)` exactly — a well-defined, finite-likelihood starting point.
Without it a random velocity field gives a random diffeomorphism and the first
validation numbers are meaningless. `TimeEmbed`
([flow.py:24-35](../flowquake/flow.py#L24-L35)) maps the scalar `t_flow` to
`[t, sin(2^k pi t), cos(2^k pi t)]`, `k = 0..3`, nine dimensions: an MLP fed a raw
scalar alongside 30-dimensional conditioning underweights it, and multi-frequency
features let the network express sharp `t_flow`-dependence. Same construction as a
transformer's positional encoding, different purpose — here it is about spectral
bias, not position identification.

### Q13. Give the numbers. How big is the temporal head and how big is the win?

22,657 parameters (`Linear(40,96) + 2x Linear(96,96) + Linear(96,1)`). On
ComCat_25's 21,889 test events: FlowQuake `tll = 1.4876`, ETAS `tll = 1.4343`,
Poisson `tll = 0.5126`. The paired gain is `+0.0533 [0.0403, 0.0675]` nats/event
with a 60.8% win rate and a block-bootstrap `p` at the resolution floor
([runs/total_win.json](../runs/total_win.json)). Framed against the trivial baseline:
ETAS is 0.9217 nats above Poisson, FlowQuake is 0.9750 — so the win is 5.8% of the
distance ETAS itself covered.

### Q14. If you had one week, what would you run?

Three things, in order. (1) The ODE step-count sweep on all production seeds and
all five catalogs — `scripts/diag_ll.py` already does it; commit the JSON. (2) The
`tau`-floor exposure: the fraction of test events at `TAU_FLOOR_DAYS`, the upper
tail of the per-event `tll`, and a re-score with those events dropped. (3) The
temporal-head ablation: conditional log-normal mixture vs. rectified flow, same
conditioning, three seeds. The first two close audit holes; the third is the
experiment that decides whether the flow is doing anything a closed-form mixture
could not.

---

## 17. Further reading

Ordered by what will actually help you defend this material.

1. **Lipman, Chen, Ben-Hamu, Nickel & Le (2023), "Flow Matching for Generative
   Modeling", ICLR.** Source of the theorem in §6.3 and of the exact conditional
   path at [flow.py:76](../flowquake/flow.py#L76). Read §3–4 and reproduce the
   marginalization lemma yourself — the single most examinable result here. Cited
   in [MANUSCRIPT.md](../MANUSCRIPT.md).

2. **Liu, Gong & Liu (2023), "Flow Straight and Fast: ... Rectified Flow", ICLR.**
   The straight-line-path view, i.e. FlowQuake's `sigma_min = 0` case, derived from
   a transport-cost argument instead of a probability-path one. (Albergo &
   Vanden-Eijnden 2023, ICLR, "Building Normalizing Flows with Stochastic
   Interpolants", is the third simultaneous derivation of the same object.)

3. **Chen, Rubanova, Bettencourt & Duvenaud (2018), "Neural Ordinary Differential
   Equations", NeurIPS.** The instantaneous change of variables (their Theorem 1)
   and the adjoint method of §5 — the paper that made CNFs a thing and, by making
   their training painful, made flow matching necessary.

4. **Grathwohl, Chen, Bettencourt, Sutskever & Duvenaud (2019), "FFJORD", ICLR.**
   Where Hutchinson's estimator enters CNFs; read it to see exactly what problem
   FlowQuake's `d = 1` lets it skip.

5. **Papamakarios, Nalisnick, Rezende, Mohamed & Lakshminarayanan (2021),
   "Normalizing Flows for Probabilistic Modeling and Inference", JMLR.** The
   survey — §2's discrete-flow taxonomy and the precise validity conditions for
   change-of-variables arguments.

6. **Shchur, Biloš & Günnemann (2020), "Intensity-Free Learning of Temporal Point
   Processes", ICLR.** The strongest statement of "model `f`, not `lambda`", and
   the closed-form mixture that hostile question H4 is built on. Read it *because*
   it is the ablation FlowQuake did not run.

7. **Omi, Ueda & Aihara (2019), "Fully Neural Network based Model for General
   Temporal Point Processes", NeurIPS.** The other route to an exact neural-TPP
   normalizer: model the cumulative hazard with a monotone net and differentiate.
   Same exactness as a flow, entirely different mechanism.

8. **Ogata (1988), "Statistical models for earthquake occurrences and residual
   analysis for point processes", JASA.** The ETAS paper; you cannot argue about
   kernel shape (§11) without it. Covered in [Chapter 3](03-etas.md).

9. **Bishop (1994), "Mixture Density Networks", Aston University technical
   report.** Twelve pages, still the clearest account of what an MDN is and why
   its likelihood is exact. Read it before dismissing MDNs.

10. **Daley & Vere-Jones, *An Introduction to the Theory of Point Processes*
    (Springer, 2 vols).** Reference for the measure-theoretic statements this
    chapter uses informally — conditional intensity, compensator, the equivalence
    of the two likelihood forms. Consult, do not read through.

**Repository files that are part of the reading, not commentary on it:**
[flowquake/flow.py](../flowquake/flow.py) (134 lines — read all of them),
[tests/test_flow.py](../tests/test_flow.py) (four invariants, §12.6),
[flowquake/model.py:219-243](../flowquake/model.py#L219-L243) (the unit conversion in
situ), [scripts/diag_ll.py](../scripts/diag_ll.py) (the convergence check
that was written and never reported), and
[STACK.md §9](../STACK.md#9-flowpy--the-temporal-head) for the code walkthrough this
chapter is the theory for.
