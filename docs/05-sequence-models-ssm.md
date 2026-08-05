# Chapter 5 — Sequence models and selective state-space models

From `x' = Ax + Bu` to the 295 lines of [flowquake/ssm.py](../flowquake/ssm.py),
with every step derived. The chapter ends by proving to you that this beautiful
machine is switched off in every production run in the repository, and by giving
you both the defensible and the honest answer to why it is in the paper anyway.

---

## What this chapter buys you

- You can **derive state-space duality** on a whiteboard: unroll the SSD
  recurrence and show it equals masked linear attention with a decay mask. This
  is the single result a Mamba-2 examiner will ask for.
- You can **read `selective_scan_chunked` line by line** and say what every
  einsum index means, give its time and memory complexity, and explain why chunk
  size `Q` trades one against the other.
- You can **derive the ZOH discretization** `A_bar = exp(Delta A)`,
  `B_bar = (Delta A)^-1 (exp(Delta A) - I) Delta B`, say what `Delta` means
  physically, and point out that FlowQuake's code uses the *Euler* `B_bar`, not
  the exact one.
- You can **do the vanishing-gradient argument properly** — Jacobian products,
  spectral radius, Gelfand's formula — instead of hand-waving about "long-range
  dependencies", and connect it to why `a_t = exp(-Delta_t A_h)` is literally an
  LSTM forget gate.
- You can **defend or attack** FlowQuake's use of an SSM: you will know the exact
  count of production runs that instantiate it (three, all ablations), the actual
  receptive field of the production temporal head (64 events), and where the
  repo's own prose overstates the encoder's role.
- You can **redo the worked example by hand** — a four-step recurrence and its
  two-chunk decomposition, in exact binary fractions — and thereby prove to
  yourself that you understand the algorithm rather than the vocabulary.

---

## Prerequisites

Read first, in this order:

- **[Chapter 1](01-point-processes.md)**, point processes and the
  likelihood. You need `lambda(t | H_t)`, `f(tau | H)`, and the three scores
  `tll` / `sll` / `mll`. [STACK.md Part I](../STACK.md) is the short version.
- **[Chapter 3](03-etas.md)**, ETAS. You need to know that ETAS sums over
  the *entire* history with an Omori power-law decay, because "reaching the whole
  history" is the thing this chapter's machinery is supposed to buy.
- **[STACK.md §7](../STACK.md)** on [flowquake/data.py](../flowquake/data.py), for the
  token layout: `TOKEN_DIM = 32`, four core dims plus `4 x 7 = 28` relational
  lag features.

Linear algebra assumed: eigendecomposition, matrix norms, the matrix
exponential. Calculus assumed: variation of parameters for linear ODEs. Nothing
about seismology is assumed in this chapter — earthquakes appear only as the
running example (§1.4, §8.4) and in §15.

**Notation.** `t` is event time in days and `tau` an inter-event gap, per the
primer's shared notation. But in this chapter the sequence index is the *event
index*, and I write it `t` too when discussing generic sequence models — as the
literature does. Where confusion is possible I write `t` (index) vs `t_days`.
`L` is sequence length, `d` a model width, `N` the SSM state dimension, `P` the
per-head channel width, `Q` the chunk length, `H` the number of heads. `Delta_t`
is the SSM step size (the selectivity knob), never a physical duration.

---

## 1. The sequence-modelling problem

### 1.1 The object

You have a sequence `u_1, ..., u_L` in `R^d` and you want, for each position `t`,
a representation `h_t` that summarizes `u_1, ..., u_t` (causal) and is useful for
predicting something about position `t+1`. In FlowQuake, `u_t` is the 32-dim
event token and `h_t` would be the conditioning vector for the heads that predict
event `t+1`'s time, place, and magnitude.

Every architecture is a different answer to: *how does information from position
`s` reach position `t`?*

### 1.2 The four axes that matter

1. **Receptive field.** Which `s` can influence `h_t` *at all*? A CNN with kernel
   `k` and `n` layers reaches `n(k-1)+1`; a fixed-window model reaches `W`; an
   RNN or attention reaches all of `1..t` *in principle* — and "in principle"
   does a lot of work, see §2.
2. **Parameter sharing.** One weight set reused at every position (RNN, CNN,
   attention) versus position-specific weights (an MLP over a flattened window).
   Sharing is what lets the model work at a length it was not trained at.
3. **Train-time parallelism.** During training you have the whole sequence: can
   you compute all `L` outputs in `O(1)` or `O(log L)` sequential steps, or do
   you need `L` dependent ones? This is the axis on which RNNs lost to
   transformers, and it is not about FLOPs — it is about *depth of the dependency
   graph*.
4. **Inference-time cost and state.** During generation you get one token at a
   time. What does one step cost, and how much state must you carry? A
   transformer's KV cache grows as `O(L)`; an RNN or SSM state is fixed-size.

Axes 3 and 4 pull in opposite directions, and the entire SSM research programme
is an attempt to have both.

### 1.3 The comparison table

Costs are per layer, per batch element, per head, for a sequence of length `L`
and width `d`. "Train parallel?" means the sequential depth of the forward pass
as a function of `L`.

| model | train parallel? | seq. depth | train time | train memory | decode step cost | decode state | receptive field |
|---|---|---|---|---|---|---|---|
| RNN / LSTM | no | `O(L)` | `O(L d^2)` | `O(L d)` | `O(d^2)` | `O(d)` | all of `1..t`, *effectively* `O(1/log(1/rho))` (§2) |
| CNN / TCN (dilated) | yes | `O(1)` | `O(L d^2 k)` | `O(L d)` | `O(d^2 k)` | `O(k_eff d)` | fixed: `n(k-1)+1`, or `k^n` dilated |
| Transformer (softmax attn) | yes | `O(1)` | `O(L^2 d)` | `O(L^2)` naive, `O(L d)` with FlashAttention | `O(L d)` | `O(L d)` (KV cache) | all of `1..t`, exactly |
| Linear attention | yes (two forms) | `O(1)` quad. / `O(L)` rec. | `O(L^2 d)` or `O(L d d')` | `O(L^2)` or `O(d d')` | `O(d d')` | `O(d d')` | all of `1..t`, low-rank-compressed |
| LTI SSM (S4) | yes (FFT) | `O(log L)` | `O(L log L · d)` | `O(L d)` | `O(N d)` | `O(N d)` | all of `1..t`, decaying per `A`'s spectrum |
| Selective SSM (Mamba/SSD) | yes (chunked scan) | `O(L/Q)` | `O(L Q (N+P) + L N P)` | `O(L Q + L N P / Q)` | `O(N P)` | `O(N P)` | all of `1..t`, decaying per learned `Delta` |

Two entries deserve immediate comment.

- **S4's `O(log L)` depth** is the FFT's butterfly depth. It applies only because
  the system is time-*invariant*, so there is one kernel to convolve with. §8
  shows selectivity destroys this.
- **Mamba/SSD's `O(L/Q)` depth** is the sequential loop over chunks in
  [ssm.py:90-92](../flowquake/ssm.py#L90-L92). For `L = 2048, Q = 64` that is 32
  Python iterations, which is small enough that nobody cares — but it is not
  `O(1)`, and a professor who asks "is Mamba really parallel?" wants exactly this
  answer.

### 1.4 What an earthquake catalog demands

Three facts from [Chapter 3](03-etas.md) set the requirements:

- **Omori decay is a power law**, `(t + c)^-p` with `p ≈ 1`, so the influence of a
  mainshock decays *polynomially*, not exponentially. A 2011 M9 still matters in
  2019. Any model whose memory decays geometrically — every RNN, every SSM, every
  decay-masked linear attention — has the *wrong tail shape* by construction.
- **The relevant lag in events is enormous.** At ComCat's rate the last 64 events
  span about 70 days ([data.py:30-33](../flowquake/data.py#L30-L33), the code's own
  comment). Two years is thousands of events.
- **Exact recall matters.** The useful fact is often "there was an M7.1 at
  *these* coordinates 400 days ago" — a specific token, not a summary. A
  fixed-size state fundamentally cannot store an unbounded number of specific
  past tokens; attention can (at `O(L)` cost).

Hold onto that third point. It is the deepest reason FlowQuake's production model
does not use an SSM, and it is the honest content of §15.

---

## 2. RNNs, and why their gradients die

### 2.1 The model

```
h_t = phi(W h_{t-1} + U u_t + b),      h_0 = 0
y_t = V h_t
```

`W` in `R^{d x d}`, `phi` an elementwise nonlinearity (tanh, ReLU). The
receptive field is "everything", the parameter count is independent of `L`, and
the sequential depth is `L`. It is the minimal answer to the sequence problem and
it does not work at long range. Here is exactly why.

### 2.2 Backpropagation through time is a Jacobian product

Let the loss `Lo` depend on `h_T`. Then

```
dLo/dW  =  sum_{t=1..T}  (dLo/dh_T) · (dh_T/dh_t) · (dh_t/dW)|_explicit
```

where `(dh_t/dW)|_explicit` treats `h_{t-1}` as constant. The middle factor is a
product of one-step Jacobians:

```
dh_T/dh_t  =  prod_{k=t+1..T}  dh_k/dh_{k-1}
           =  prod_{k=t+1..T}  D_k W,        D_k = diag(phi'(a_k))
```

with `a_k = W h_{k-1} + U u_k + b`. *Everything* about long-range learning in an
RNN is a statement about this product of `T - t` matrices.

### 2.3 The spectral-radius dichotomy

**Linear case, exactly.** Set `phi = identity`, so `D_k = I` and
`dh_T/dh_t = W^{T-t}`. Gelfand's formula says

```
lim_{n -> inf}  ||W^n||^{1/n}  =  rho(W)
```

the spectral radius (largest modulus eigenvalue), for *any* submultiplicative
norm. So `||dh_T/dh_t|| ~ rho(W)^{T-t}` asymptotically. Three regimes, and only
three:

- `rho(W) < 1` — gradients vanish geometrically. Nothing beyond a horizon is
  learnable.
- `rho(W) > 1` — gradients explode geometrically. Training diverges unless you
  clip (and clipping destroys the direction information you were trying to
  propagate).
- `rho(W) = 1` — a measure-zero knife edge, and even there `||W^n||` can grow
  polynomially if `W` is not diagonalizable (Jordan blocks).

**Nonlinear case, as a bound.** Let `gamma = sup_x |phi'(x)|` (`gamma = 1` for
tanh, `1/4` for sigmoid, `1` for ReLU) and `sigma_1(W)` be the largest singular
value. Since `||D_k W|| <= ||D_k|| · ||W|| <= gamma · sigma_1(W)` in the spectral
norm,

```
|| dh_T/dh_t ||  <=  ( gamma · sigma_1(W) )^{T-t}
```

So `gamma · sigma_1(W) < 1` is a **sufficient condition for vanishing
gradients**; contrapositively `gamma · sigma_1(W) > 1` is a **necessary
condition for exploding gradients**. This is the argument in Pascanu, Mikolov &
Bengio (2013, ICML, "On the difficulty of training recurrent neural networks"),
and it is the one to reproduce under questioning.

**The arithmetic that makes it visceral.** Suppose you have tuned things so that
the effective per-step factor is `0.99`. Then the gradient contribution from lag
`n` is scaled by:

```
n = 100    0.99^100  = 0.366
n = 1000   0.99^1000 = 4.3e-5
n = 5000   0.99^5000 = 1.5e-22
```

At `n = 5000` the contribution is below fp32's relative resolution (`~1e-7`)
against gradient terms of order one: it is *numerically* zero, not just small.
And `0.99` per step corresponds to a memory time constant of
`1/|log 0.99| ≈ 99.5` steps. To get a 5000-step time constant you need a factor
of `exp(-1/5000) = 0.9998`, which is indistinguishable from the exploding regime.
**Long memory and stable optimization are in direct conflict in a vanilla RNN.**
That is the whole problem.

### 2.4 Why gating helps

The LSTM (Hochreiter & Schmidhuber 1997, *Neural Computation*; the forget gate
was added by Gers, Schmidhuber & Cummins 2000, *Neural Computation*) introduces
a **cell state** with an additive update:

```
f_t = sigmoid(W_f [h_{t-1}, u_t])        forget gate,  in (0,1)^d
i_t = sigmoid(W_i [h_{t-1}, u_t])        input gate
g_t = tanh(W_g [h_{t-1}, u_t])           candidate
c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t        the cell
h_t = o_t ⊙ tanh(c_t)
```

Along the cell path, ignoring the indirect route through `h` into the gates,

```
dc_T/dc_t  =  prod_{k=t+1..T} diag(f_k)
```

Three structural changes, and each one matters:

1. **It is elementwise, not a matrix product.** There is no single `rho(W)` to
   trade off. Channel 3 can hold `f ≈ 0.999` (1000-step horizon) while channel 7
   holds `f ≈ 0.5` (1.4-step horizon), *simultaneously*. A vanilla RNN's spectral
   radius is one number for the whole state.
2. **`f_k` is input-dependent.** The network can *decide*, per step and per
   channel, whether to preserve or overwrite. This is content-based forgetting,
   and it is the direct ancestor of Mamba's selectivity.
3. **The path is additive**, so no rotation or mixing accumulates along it — the
   "constant error carousel".

Be honest about the limits: `prod f_k` with `f_k = 1 - eps` still decays as
`exp(-eps · n)`. Gating makes the decay rate *learnable and per-channel*; it does
not make it zero. And the derivative above is only the dominant path — the gates
themselves depend on `h_{t-1}`, so there are additional (small) terms.

**The sentence to remember for §8:** *a forget gate is a learned per-channel
decay factor in `(0,1)`, chosen as a function of the current input.* Mamba's
`a_t = exp(-Delta_t A_h)` is exactly that object, arrived at from a completely
different direction.

---

## 3. Attention, transformers, and the `L^2` wall

### 3.1 The mechanism

Project the sequence three ways: `Q = U W_Q`, `K = U W_K`, `V = U W_V`, all
`L x d`. Then

```
O  =  softmax( Q K^T / sqrt(d)  +  M ) V,      M[t,s] = 0 if s <= t else -inf
```

Per position: `O_t = sum_{s<=t} alpha_{ts} v_s` with
`alpha_{ts} = exp(q_t·k_s/sqrt(d)) / sum_{s'<=t} exp(q_t·k_{s'}/sqrt(d))`.

What you buy: **exact, content-addressed recall at unbounded distance in one
layer**, and a sequential depth of `O(1)` (two matmuls and a softmax). What you
pay for it is the `L x L` score matrix.

### 3.2 Cost

- Time: `O(L^2 d)` to form `QK^T` and `O(L^2 d)` to apply it to `V`.
- Memory: `O(L^2)` for the scores if you materialize them. FlashAttention (Dao
  et al. 2022, NeurIPS) tiles the computation so the `L x L` matrix is never
  written to HBM, reducing *memory* to `O(L d)` — but the **time is still
  `O(L^2 d)`**. This distinction is a favourite exam trap.

For a catalog: the ComCat_25 *test window alone* has 21,889 events
([runs/n1_density/eval_test.json](../runs/n1_density/eval_test.json), `n_events`),
and the **training window carries 55,442**
([runs/mw_robustness.json](../runs/mw_robustness.json) →
`california.comcat_mc25_headline.train_events`). The full sequence FlowQuake
evaluates in one pass is the whole catalog from the `aux_start` of 1971
([config.py:15](../flowquake/config.py#L15)) — the 1971-1981 auxiliary era, the
55,442 training events, the 1998-2007 validation era and the 21,889 test events:
**92,263** in total ([README.md](../README.md), the ComCat catalog line), of
which **70,374 = 92,263 − 21,889** precede `test_start`.
`L^2 ≈ 8.5 × 10^9` score entries per head per layer.

> **The event-count trap, once, for the whole primer.**
> [STACK.md §6](../STACK.md#6-why-etas-is-hard-to-beat) says "~70,000 training
> events". That number is `92,263 − 21,889 = 70,374`, i.e. *everything before the
> test window* — aux + train + val — not the training window, which is 55,442.
> Both are correct about different windows; quote 55,442 when you mean "what the
> model was fitted on" and 70,374 when you mean "the prefix the evaluator runs
> before scoring the first test event". [Ch. 3 §11](03-etas.md#11-why-etas-is-hard-to-beat--argued-not-asserted)
> and [Ch. 9 Q39](09-viva-question-bank.md#3-tier-2--etas-q33q53) use 55,442.
Even with FlashAttention's memory fix, `10^10 · d` FLOPs per head per layer per
forward pass is not something you do inside a training loop that runs 20,000
steps.

### 3.3 The KV cache and why decoding is where it hurts

At generation time you have already computed `K_{1..t-1}, V_{1..t-1}`; you cache
them and append. Per new token: `O(t d)` work, `O(t d)` memory. The state
**grows linearly with the sequence**.

FlowQuake's generative path ([flowquake/ntest.py](../flowquake/ntest.py)) simulates
`n_sims` independent continuations per forecast day — the default is
`--n-sims 10000` ([csep_forecast.py:101](../flowquake/csep_forecast.py#L101)). Each
lane needs its own cache. A KV cache for a `d_model = 96`, `4`-layer model over
`L = 70,000` prior events (the pre-`test_start` prefix, `92,263 − 21,889 = 70,374`,
rounded — this is the length the evaluator actually carries when it reaches the
first test event, so it is the right `L` for a cache-size argument) is

```
2 (K and V) x 4 layers x 70,000 x 96 floats  =  53.76 M floats  =  215 MB (fp32)
```

per lane. Times 10,000 lanes: **2.15 TB**. That is the number that kills the
transformer for this task, and §12 gives the SSM's answer.

---

## 4. Linear attention: kill the softmax, then reassociate

### 4.1 The kernel view

Write attention with a general similarity `sim`:

```
O_t  =  ( sum_{s<=t} sim(q_t, k_s) v_s )  /  ( sum_{s<=t} sim(q_t, k_s) )
```

Softmax is the case `sim(q,k) = exp(q·k / sqrt(d))`. That exponential is the only
thing standing between you and an `O(L)` algorithm, because it does **not**
factorize into a function of `q` times a function of `k`.

So replace it with one that does. Pick a feature map `phi: R^d -> R^{d'}` with
`phi >= 0` elementwise and set

```
sim(q, k)  =  phi(q)^T phi(k)
```

Katharopoulos, Vyas, Pappas & Fleuret (2020, ICML, "Transformers are RNNs") use
`phi(x) = elu(x) + 1`. Performer (Choromanski et al. 2021, ICLR) picks a random
`phi` so that `E[phi(q)^T phi(k)] = exp(q·k)`, i.e. it *does* approximate softmax
— but the SSM lineage does not; it just uses a different similarity and accepts
that it is a different model.

### 4.2 The reassociation, derived

Numerator:

```
sum_{s<=t} ( phi(q_t)^T phi(k_s) ) v_s
  = sum_{s<=t} phi(q_t)^T ( phi(k_s) v_s^T )       [scalar times vector, regrouped]
  = phi(q_t)^T  sum_{s<=t} phi(k_s) v_s^T
  = phi(q_t)^T  S_t,          S_t := sum_{s<=t} phi(k_s) v_s^T   in R^{d' x d_v}
```

Denominator:

```
sum_{s<=t} phi(q_t)^T phi(k_s)  =  phi(q_t)^T  z_t,   z_t := sum_{s<=t} phi(k_s)
```

The only thing used is **associativity of matrix multiplication**: `(q K^T) V`
costs `O(L^2)`, `q (K^T V)` costs `O(d' d_v)`. The softmax was the obstruction
because it sits *between* `QK^T` and `V` and is nonlinear in the scores.

### 4.3 The recurrent form falls out

`S_t` and `z_t` are running sums, so

```
S_t = S_{t-1} + phi(k_t) v_t^T
z_t = z_{t-1} + phi(k_t)
O_t = phi(q_t)^T S_t / (phi(q_t)^T z_t)
```

Constant-size state `(d' x d_v) + d'`, `O(d' d_v)` per step, `O(L d' d_v)` total.
**This is an RNN.** That is the paper's title, and it is a theorem, not a
metaphor.

Note the two *modes* this gives you for the same mathematical object:

| mode | how | time | memory | parallel |
|---|---|---|---|---|
| quadratic | form the `L x L` masked score matrix, multiply by `V` | `O(L^2 d')` | `O(L^2)` | fully |
| recurrent | roll `S_t` forward | `O(L d' d_v)` | `O(d' d_v)` | no |

Two algorithms, one function. Everything in §10 is about interpolating between
them.

### 4.4 What you lose

`S_t` is a `d' x d_v` matrix accumulating `t` rank-one terms. Once `t > d'` the
sum is a lossy compression: you cannot recover an arbitrary `v_s` from `S_t`.
Softmax attention's ability to sharply select one past position (the softmax can
concentrate almost all mass on a single `s`) is genuinely lost. This is the
"associative recall" gap, and it is real: it is why hybrid attention/SSM stacks
exist.

### 4.5 Add a decay mask — and note what you have built

Drop the denominator (SSMs do; the gated RMSNorm downstream handles scale) and
insert a per-step decay `a_r` in `(0,1)`:

```
S_t  =  a_t S_{t-1}  +  phi(k_t) v_t^T
O_t  =  phi(q_t)^T S_t
```

Unrolling (this is done properly in §9),

```
O_t  =  sum_{s<=t}  ( prod_{r=s+1..t} a_r ) · ( phi(q_t)^T phi(k_s) ) · v_s
```

which is masked linear attention where the mask is not `{0,1}` but a decaying
`L(t,s) = prod_{r=s+1..t} a_r`. Constant `a` gives RetNet-style "retention" (Sun
et al. 2023, "Retentive network: a successor to Transformer for large language
models", arXiv:2307.08621 — an arXiv preprint; cite it as such, it was not
accepted at a major venue); input-dependent `a_t` gives
Gated Linear Attention (Yang et al. 2024, ICML) and, with the identification
`phi(q_t) -> C_t`, `phi(k_s) -> B_s`, `v_s -> Delta_s x_s`, **it gives exactly
the recurrence implemented in [ssm.py:7-9](../flowquake/ssm.py#L7-L9)**.

Park that. We now build the same object from the other end — differential
equations — and the two roads will meet in §9.

---

## 5. Linear time-invariant systems, from scratch

### 5.1 The object

A continuous-time single-input single-output LTI system:

```
x'(t) = A x(t) + B u(t)        state equation,   x in R^N, A in R^{NxN}, B in R^{Nx1}
y(t)  = C x(t) + D u(t)        output equation,  C in R^{1xN}, D scalar
```

`u` is the input signal, `x` the internal state, `y` the output. "Linear" because
the map `u -> y` is linear; "time-invariant" because `A, B, C, D` do not depend
on `t`.

### 5.2 Solution by variation of constants

The homogeneous equation `x' = Ax` has solution `x(t) = e^{At} x(0)`, where

```
e^{At}  =  sum_{k>=0} (At)^k / k!
```

(This series converges for every square `A`; differentiating termwise gives
`d/dt e^{At} = A e^{At}`.)

For the driven system, substitute `x(t) = e^{At} w(t)`:

```
x'  =  A e^{At} w  +  e^{At} w'   =   A x  +  e^{At} w'
```

Comparing with `x' = Ax + Bu` forces `e^{At} w'(t) = B u(t)`, i.e.
`w'(t) = e^{-At} B u(t)`. Integrating from `0` to `t`:

```
w(t) = w(0) + int_0^t e^{-As} B u(s) ds,     w(0) = x(0)
```

and therefore

```
x(t)  =  e^{At} x(0)  +  int_0^t e^{A(t-s)} B u(s) ds
```

### 5.3 The impulse response, and why this is a convolution

Take `x(0) = 0`. Then

```
y(t)  =  int_0^t  C e^{A(t-s)} B  u(s) ds  +  D u(t)
      =  int_0^t  h(t-s) u(s) ds  +  D u(t),        h(r) := C e^{Ar} B
```

`h` is the **impulse response**: put `u = delta` and you read out `y = h`
(plus `D delta`). The key structural fact:

> **An LTI system's output is a convolution of the input with a single fixed
> kernel.** The kernel depends on `t - s` only — never on `t` and `s`
> separately. That is what time-invariance *means*.

### 5.4 The discrete version and the SSM convolution kernel

Suppose you have discretized (§6) to

```
x_k = A_bar x_{k-1} + B_bar u_k
y_k = C x_k + D u_k
```

Unroll from `x_0 = 0`:

```
x_1 = B_bar u_1
x_2 = A_bar B_bar u_1 + B_bar u_2
...
x_k = sum_{j=1..k}  A_bar^{k-j} B_bar u_j
```

so

```
y_k  =  sum_{j=1..k}  ( C A_bar^{k-j} B_bar ) u_j  +  D u_k
     =  sum_{j=1..k}  K_{k-j} u_j  +  D u_k,        K_r := C A_bar^r B_bar
```

The vector `K = (K_0, K_1, ..., K_{L-1})` is the **SSM convolution kernel**, and
`y = K * u` is a causal discrete convolution.

### 5.5 Why that means `O(L log L)`

Zero-pad `K` and `u` to length `M >= 2L - 1` (in practice the next power of two).
By the convolution theorem, circular convolution is pointwise multiplication in
the Fourier domain:

```
y  =  IFFT( FFT(K_pad) ⊙ FFT(u_pad) )[0:L]
```

Two forward FFTs, `M` complex multiplications, one inverse FFT:
`O(M log M) = O(L log L)`. The sequential depth is the FFT's butterfly depth,
`O(log L)`.

**But you still have to produce `K`.** Naively, `K_r = C A_bar^r B_bar` for
`r = 0..L-1` costs `L` matrix-vector products with `A_bar` — `O(L N^2)` dense, or
`O(L N)` if `A_bar` is diagonal. For `L = 16384` and `N = 64` the dense version is
`6.7 x 10^7` per channel per layer *per forward pass*, and repeated powering of a
dense matrix is numerically miserable. Making `K` cheap and stable is the entire
technical content of S4 (§7).

---

## 6. Discretization: zero-order hold and bilinear

You have a continuous system and a discrete sequence. You need `A_bar, B_bar`.

### 6.1 Zero-order hold, derived

**Assumption:** the input is held constant, `u(s) = u_k` for
`s in [k·Delta, (k+1)·Delta)`. That is what "zero-order hold" means: a
piecewise-constant reconstruction of `u` from its samples.

Apply the exact solution from §5.2 over one step, writing `x_k := x(k·Delta)`:

```
x_{k+1}  =  e^{A·Delta} x_k  +  int_{kD}^{(k+1)D} e^{A((k+1)D - s)} B u_k ds
```

Substitute `r = (k+1)Delta - s`, so `ds = -dr`; `s = k·Delta` gives `r = Delta`
and `s = (k+1)Delta` gives `r = 0`. The integral becomes

```
( int_0^Delta e^{Ar} dr ) B u_k
```

Now evaluate `int_0^Delta e^{Ar} dr` termwise:

```
int_0^Delta sum_k A^k r^k / k! dr  =  sum_k A^k Delta^{k+1} / (k+1)!
                                   =  A^{-1} sum_k (A Delta)^{k+1} / (k+1)!
                                   =  A^{-1} ( e^{A Delta} - I )
```

Hence the **exact ZOH discretization**:

```
A_bar  =  exp(Delta A)
B_bar  =  A^{-1} ( exp(Delta A) - I ) B
       =  (Delta A)^{-1} ( exp(Delta A) - I ) · Delta B
```

The second form is the one the literature writes, and the reason is the limit:
as `Delta A -> 0`, `(Delta A)^{-1}(exp(Delta A) - I) -> I`, so `B_bar -> Delta B`
smoothly. Note also that the *series* `sum_k A^k Delta^{k+1}/(k+1)!` is defined
even when `A` is singular — the `A^{-1}` is notation for that series, not a
requirement that `A` be invertible.

### 6.2 Bilinear / Tustin

Integrate `x' = Ax + Bu` over one step and approximate the integral by the
trapezoid rule:

```
x_{k+1} - x_k  ≈  (Delta/2)[ (A x_k + B u_k) + (A x_{k+1} + B u_{k+1}) ]
```

Collecting terms,

```
(I - (Delta/2) A) x_{k+1}  =  (I + (Delta/2) A) x_k  +  (Delta/2) B (u_k + u_{k+1})
```

so, folding the input terms into a single `u_k` (the usual S4 convention),

```
A_bar  =  (I - (Delta/2) A)^{-1} (I + (Delta/2) A)
B_bar  =  (I - (Delta/2) A)^{-1} Delta B
```

**Why anyone uses it:** the map `s -> (1 + sDelta/2)/(1 - sDelta/2)` sends the
open left half-plane exactly onto the open unit disk, for any `Delta`. So a
stable continuous system discretizes to a stable discrete one at any step size
(A-stability), and you need only one matrix inverse rather than a matrix
exponential. ZOH is *exact* for piecewise-constant inputs; bilinear is a
second-order approximation that never destabilizes. S4 used bilinear; Mamba uses
ZOH.

### 6.3 What `Delta` means physically

Take `A` with an eigenvalue `-a`, `a > 0`. Then that mode of `A_bar = exp(Delta A)`
has eigenvalue `exp(-Delta a)`, and the state's memory of an input decays by
`e^{-1}` after `1/(Delta a)` steps.

```
Delta a small  ->  A_bar ≈ 1,     B_bar ≈ Delta B ≈ 0   -> "ignore this step; keep the state"
Delta a large  ->  A_bar ≈ 0,     B_bar large           -> "reset the state; write this input hard"
```

**`Delta` is a ratio: sampling interval divided by the system's natural
timescale.** It sets, in units of *steps*, how far back the state remembers. In a
classical LTI setting it is a fixed constant. Making it a function of the input
is the entirety of Mamba's contribution (§8).

### 6.4 What `ssm.py` actually uses — a checkable claim

[ssm.py:68](../flowquake/ssm.py#L68) computes `log_a = -dt * A`, i.e.
`a_t = exp(-Delta_t A_h)` — that is the exact ZOH `A_bar` with a scalar
`A = -A_h`. But the input term at [ssm.py:76,80](../flowquake/ssm.py#L76-L80) and in
the reference at [ssm.py:115](../flowquake/ssm.py#L115) is `dt * B * x`, i.e.

```
B_bar_t x_t  =  Delta_t B_t x_t          (Euler / first-order)
```

not the exact ZOH `B_bar = ((1 - exp(-Delta A))/A) B`. The two differ by the
factor

```
(1 - exp(-Delta A)) / (Delta A)   in (0, 1],   -> 1 as Delta A -> 0
```

This is not a bug — Mamba-2 does the same, and the factor is absorbed by the
learnable scale of `B` and the downstream norm. But if a professor asks "is your
discretization exact?", the correct answer is: *the state transition is exact
ZOH; the input map is the leading-order (Euler) approximation to ZOH, and the
discrepancy is a strictly-positive scalar per head per step that the model can
absorb into `B_t`.* Being able to say that is the difference between having read
the code and having read the abstract.

---

## 7. HiPPO and S4: why the initialization of `A` is the whole game

### 7.1 The problem HiPPO poses

You have a state `x(t) in R^N` and a stream `u(s)` for `s <= t`. The state must
be a *summary* of the whole stream. Which summary?

HiPPO (Gu, Dao, Ermon, Rudra & Ré 2020, NeurIPS, "HiPPO: Recurrent Memory with
Optimal Polynomial Projections") answers: let `x(t)` be the coefficients of the
best degree-`(N-1)` polynomial approximation to `u|_{[0,t]}` in a weighted `L^2`
sense, with respect to some measure on the past. Remarkably, for several natural
measures this *online projection* obeys a **linear** ODE in `x`, with a specific
`A` and `B` you can write down.

**Careful here — this is a standard slip.** *Linear* does not imply
*time-invariant*, and the two headline HiPPO measures land on opposite sides:

- **LegT** (translated Legendre: uniform weight on a *sliding window* of fixed
  length `theta`) gives `x' = -(1/theta) A x + (1/theta) B u` — constant
  coefficients, so a genuine **LTI** system.
- **LegS** (scaled Legendre: uniform weight on all of `[0,t]`) gives
  `x'(t) = -(1/t) A x(t) + (1/t) B u(t)` — a **linear time-varying** ODE. The
  `1/t` is not decoration: it is exactly what makes LegS *timescale invariant*
  (rescale `t` and the projection is unchanged, so there is no built-in
  forgetting horizon). Under `t = e^s` the `1/t` is absorbed and the system
  becomes LTI in `s`, which is the sense in which "LegS is an LTI system" is
  sometimes said — but in `t` it is not.

For LegS the matrix is

```
A_nk  =  (2n+1)^{1/2} (2k+1)^{1/2}     n > k
      =  n+1                            n = k
      =  0                              n < k
B_n   =  (2n+1)^{1/2}
```

with the minus sign carried outside, in the `-(1/t) A x` above. (Indices from 0;
papers differ on whether the sign is folded into `A` and on whether indexing
starts at 0 or 1 — check the convention against the source before quoting it in
writing.)

What S4 actually inherits is the *matrix*, frozen: S4 drops the `1/t` and uses
the LegS `A` as the static generator of an LTI system. That is an
initialization heuristic borrowed from the LegS derivation, not the LegS
operator itself, and saying so is the difference between a clean and a muddled
answer.

### 7.2 Why this is not a detail

A randomly initialized `A` gives a random linear system. Its memory horizon is
whatever its random spectrum happens to be — typically a few tens of steps, and
the modes that decay slowly are unlikely to be the ones carrying signal. The
empirical claim from the S4 paper (Gu, Goel & Ré 2022, ICLR, "Efficiently
Modeling Long Sequences with Structured State Spaces") is that HiPPO
initialization is the difference between near-chance and state-of-the-art on the
Long Range Arena benchmark, with everything else held fixed. This is one of the
few "initialization is the result" findings in modern deep learning that has held
up.

**Note against FlowQuake:** [ssm.py:156-158](../flowquake/ssm.py#L156-L158)
initializes `A_log = log(Uniform(A_min, A_max))` with `A_min = 1, A_max = 16`.
That is a *plain random* init of scalar per-head decay rates — there is no HiPPO
structure anywhere in this repository. Mamba-2 also drops most of the HiPPO
structure (scalar `A` per head cannot encode a polynomial basis), and the
justification is that selectivity now supplies the memory control that `A`'s
structure used to. But you should say this out loud rather than let a professor
discover you thought you had HiPPO.

### 7.3 What S4's machinery actually solves

S4's problem is §5.5's: compute `K_r = C A_bar^r B_bar` for all `r < L`, fast and
stably, for an `A` you are not allowed to choose freely (it must stay near
HiPPO). The route, sketched:

1. **Structure `A` as diagonal-plus-low-rank (DPLR)**, `A = Lambda - P Q*`. The
   HiPPO-LegS matrix is *normal* plus low rank, and a normal matrix is unitarily
   diagonalizable, so this is achievable.
2. **Work with the truncated generating function** `K_hat(z) = sum_{r<L} K_r z^r`
   rather than the kernel directly; it telescopes to a resolvent
   `C_tilde (I - A_bar z)^{-1} B_bar`.
3. **Apply Woodbury** to reduce the (diagonal + low rank) resolvent to the
   diagonal one plus small corrections; each term is then a **Cauchy kernel** sum
   `sum_n c_n / (w_n - z)`, which has stable fast algorithms.
4. Evaluate `K_hat` at the `L` roots of unity, one inverse FFT recovers `K`.

Net cost: roughly `O(N + L)` up to logs, instead of `O(L N^2)`.

**I am sketching this, not deriving it.** The full derivation (the exact
telescoping, the Woodbury bookkeeping, the conjugate-symmetry reduction) is in
the S4 paper's appendix and I will not reproduce it. What you must be able to say
is: *S4 is fast because time-invariance makes the whole layer a single
convolution, and DPLR structure plus a generating-function trick makes that
convolution's kernel computable in near-linear time.*

### 7.4 S4D: diagonal is enough

Gu, Goel, Gupta & Ré (2022, NeurIPS, "On the parameterization and initialization
of diagonal state space models") showed that a **purely diagonal** `A` with a
good initialization (e.g. S4D-Lin, `A_n = -1/2 + i·pi·n`) recovers nearly all of
S4's performance while removing all the DPLR machinery: with `A_bar` diagonal,
`K_r = sum_n C_n A_bar_nn^r B_bar_n` is a Vandermonde sum. This is the result
that licenses everything after it, including Mamba's diagonal `A` and Mamba-2's
scalar `A`.

---

## 8. Mamba / S6: selectivity, and the death of the convolution

### 8.1 The change

Mamba (Gu & Dao, "Mamba: linear-time sequence modeling with selective state
spaces"; arXiv:2312.00752, 2023, published at COLM 2024 — [MANUSCRIPT.md
line 943](../MANUSCRIPT.md) cites only the arXiv id, which is worth upgrading)
makes `Delta`, `B`, `C` functions of the current input:

```
Delta_t = softplus( s_Delta(u_t) + Delta_bias )     (positive by construction)
B_t     = s_B(u_t)          in R^N
C_t     = s_C(u_t)          in R^N
A                            still a learned static (diagonal) matrix
A_bar_t = exp(Delta_t A),   B_bar_t = Delta_t B_t
```

`s_Delta, s_B, s_C` are linear maps of the input. Note `A` stays static — the
*discretization step* is what varies, and that is enough to make `A_bar_t` vary.

### 8.2 Why the FFT dies

Unrolling as in §5.4 but with time-varying matrices:

```
y_t  =  sum_{s<=t}  C_t ( prod_{r=s+1..t} A_bar_r ) B_bar_s u_s
     =  sum_{s<=t}  K(t, s) u_s
```

`K(t,s)` depends on `t` and `s` **separately**, not on `t - s`. The map is still
linear (so it is still a matrix acting on `u`), but it is not Toeplitz, so it is
not a convolution, so there is no convolution theorem, so there is no FFT. This
is the price of selectivity, stated exactly.

What is left? Two options:
- Compute the `L x L` lower-triangular matrix `K(t,s)` — the quadratic mode.
- Roll the recurrence forward — the linear mode, `O(L)` sequential steps.

Mamba-1's contribution was a **hardware-aware parallel scan**: implement the
recurrence with a work-efficient associative scan (§11) in SRAM, fusing the
discretization, scan, and gating so the `(L, d, N)` intermediate never touches
HBM. Mamba-2's contribution was to restrict `A` further so the quadratic mode
becomes a *matmul* and the two modes can be blended (§9, §10).

### 8.3 Selectivity *is* gating — derived

Take the scalar case `N = 1`, `A = -1`, and the **exact** ZOH from §6.1:

```
A_bar = exp(-Delta),      B_bar = (1 - exp(-Delta)) · B
```

Set `B = 1` and define `g_t := 1 - exp(-Delta_t) in (0,1)`. The recurrence

```
h_t  =  A_bar_t h_{t-1} + B_bar_t u_t  =  (1 - g_t) h_{t-1}  +  g_t u_t
```

is **exactly** the classical gated RNN / leaky-integrator update, with the gate
`g_t` a monotone reparameterization of `Delta_t`. Mamba states this connection as
a theorem; the one-dimensional case is the two lines above.

So `Delta_t` is a forget gate and an input gate *tied together*: large `Delta` →
forget hard and write hard; small `Delta` → remember and ignore. Compare §2.4:
the LSTM has separate `f_t` and `i_t`; the SSM ties them because they both come
from one step size. That coupling is a modelling assumption, and it is a
defensible one — "this observation is important" and "the previous state is now
stale" are usually the same event.

### 8.4 What selectivity buys, on a catalog

Concretely, in an earthquake sequence indexed by event:

- **An M7.1 mainshock arrives.** Everything about the near-future rate field has
  just changed: the last five hundred M2.5 background events are now nearly
  irrelevant to what happens next. The right behaviour is `Delta_t` large —
  overwrite the state, write this event's location and magnitude hard.
- **An M2.5 background event arrives during a quiet period.** Nothing changed.
  The right behaviour is `Delta_t` small — leave the state alone, so that a
  mainshock 3,000 events ago is not diluted by 3,000 uninformative writes.

A *time-invariant* SSM cannot do this: its decay `exp(Delta A)` is the same at
every step, so 3,000 background events decay the mainshock's contribution by
`exp(3000 · Delta A)`, which for any decay fast enough to be useful is
numerically zero. **Content-based forgetting is exactly the property a
self-exciting point process needs**, because the informativeness of an event
varies over orders of magnitude with its magnitude (Utsu productivity says
triggering scales as `e^{a(m - m_c)}`; between M2.5 and M7.1 that is a factor of
`e^{4.6a}`, roughly `10^4` at `a ≈ 2`).

### 8.5 The caveat you must volunteer

In FlowQuake the SSM's sequence index is the **event index**, not physical time.
`Delta_t` is a learned function of the token; the token's dim 0 *is* `log tau`,
so the model *could* learn `Delta_t` proportional to elapsed physical time — but
nothing makes it. A genuinely continuous-time SSM would set
`Delta_t = tau_t` (the observed gap) and get ZOH decay
`exp(-tau_t A_h)` that is exactly an exponential-kernel Hawkes memory over
*physical* time. That is a one-line change to
[ssm.py:193](../flowquake/ssm.py#L193) and it is the most obvious unexplored
improvement in this file. Say so before you are asked.

(Even then, the memory would be *exponential* in physical time, whereas Omori
says *power law*. A mixture of heads with different `A_h` approximates a power
law as a sum of exponentials — that is a real and standard technique — but it is
an approximation, and ETAS gets the power law for free.)

---

## 9. Mamba-2 / SSD, and the duality proof

### 9.1 The restriction

Mamba-2 (Dao & Gu 2024, ICML, "Transformers are SSMs: Generalized Models and
Efficient Algorithms Through Structured State Space Duality") restricts `A` from
diagonal to **scalar times identity, per head**. So per head `h`, `A = A_h · I`
with `A_h > 0` a single learned number, and

```
a_t  =  exp(-Delta_t A_h)      a single scalar per (step, head)
```

This is a real loss of expressiveness — all `N` state dimensions of a head now
decay at the same rate — and it is bought back by having many heads with
different `A_h`. What it buys is that the decay factors *commute and factor out
of everything*, which is what makes §10 possible.

### 9.2 The recurrence in `ssm.py`

Per head, with state `H_t in R^{N x P}`, input `x_t in R^P`, projections
`B_t, C_t in R^N` ([ssm.py:7-9](../flowquake/ssm.py#L7-L9)):

```
H_t  =  a_t H_{t-1}  +  Delta_t B_t x_t^T          a_t = exp(-Delta_t A_h)
y_t  =  C_t^T H_t  +  D ⊙ x_t
```

`H_t` is an **outer-product memory**: each step adds a rank-one matrix
`B_t x_t^T`. This is the fast-weights / linear-attention state `S_t` from §4.3,
with `B_t` in the role of `phi(k_t)` and `Delta_t x_t` in the role of `v_t`.

Note the shapes in the code: `x` is `(B, L, H, P)` — per head — while `Bm` and
`Cm` are `(B, L, N)` — **shared across heads**
([ssm.py:37-41](../flowquake/ssm.py#L37-L41)). In attention terms this is one
key/query pair broadcast over many value heads; Mamba-2 names this head pattern
**multi-input SSM (MIS)** and identifies it with **multi-value attention (MVA)**
— `B` and `C` (the key/query analogues) shared across the channels of `X` (the
value analogue), with `A` remaining per-head.

### 9.3 The duality, derived

Unroll from an initial state `H_0`:

```
H_1 = a_1 H_0 + Delta_1 B_1 x_1^T
H_2 = a_2 H_1 + Delta_2 B_2 x_2^T = a_2 a_1 H_0 + a_2 Delta_1 B_1 x_1^T + Delta_2 B_2 x_2^T
...
H_t = ( prod_{r=1..t} a_r ) H_0  +  sum_{s=1..t} ( prod_{r=s+1..t} a_r ) Delta_s B_s x_s^T
```

Define the **decay** (empty product `= 1`, so `L(t,t) = 1`):

```
L(t, s)  :=  prod_{r=s+1..t} a_r  =  exp( sum_{r=s+1..t} log a_r )
```

Now read out. Using `C_t^T (B_s x_s^T) = (C_t^T B_s) x_s^T` — because
`B_s x_s^T` is rank one, so contracting on the left with `C_t` just produces the
scalar `C_t · B_s` times `x_s^T`:

```
y_t  =  C_t^T H_t
     =  ( prod_{r=1..t} a_r ) C_t^T H_0
        +  sum_{s<=t}  L(t,s) · (C_t · B_s) · Delta_s x_s
```

The second term is the whole story. Define the `L x L` matrix

```
M[t, s]  =  L(t,s) · (C_t · B_s)     if s <= t
         =  0                        if s >  t
```

Then, stacking the `x_s` into `X in R^{L x P}` and the `Delta_s` into a diagonal,

```
Y  =  M ( diag(Delta) X )     +   (initial-state term)
```

**This is masked attention.** `(C_t · B_s)` is an unnormalized attention score
with `C` playing query and `B` playing key; `L(t,s)` is a multiplicative causal
mask that decays instead of being `{0,1}`; `Delta_s x_s` is the value. Set
`a_r = 1` for all `r` and `L(t,s) = 1` on the lower triangle, recovering exactly
causal linear attention with no softmax.

> **State-space duality:** *the same function can be computed as a linear
> recurrence over a compressed state, or as a quadratic form with a structured
> (1-semiseparable) mask. Which one is faster depends on `L`, `N`, `P` and your
> hardware — not on the model.*

Cost of the two modes, per batch element per head:

| mode | time | memory | sequential depth |
|---|---|---|---|
| recurrent | `O(L N P)` | `O(N P)` | `O(L)` |
| quadratic | `O(L^2 (N + P))` | `O(L^2)` | `O(1)` |

The recurrent mode wins asymptotically in `L`; the quadratic mode wins on
hardware, because it is a matmul and a GPU's tensor cores do matmuls roughly an
order of magnitude faster per FLOP than the elementwise ops a scan is made of.
§10 takes both.

---

## 10. The chunked parallel scan, derived against `ssm.py`

### 10.1 The decomposition

Split `1..L` into `nc = L/Q` contiguous chunks of length `Q`. For a position `t`
in chunk `c`, split the sum in §9.3 at the chunk boundary:

```
y_t  =  sum_{s in chunk c, s<=t} L(t,s) (C_t·B_s) Delta_s x_s      <- INTRA
     +  C_t^T ( L(t, start_c - 1) · H_{start_c - 1} )              <- INTER
```

- **INTRA** is a `Q x Q` quadratic form — one small masked-attention block per
  chunk, all chunks in parallel.
- **INTER** needs only the *state at the chunk boundary*, `H_{start_c - 1}`.
  Those `nc` states satisfy their own short recurrence, which you run
  sequentially — but only `nc = L/Q` steps, not `L`.

That is the whole algorithm. Everything below is bookkeeping.

### 10.2 Line by line

All references to [flowquake/ssm.py](../flowquake/ssm.py).

**Padding, lines 54-61.**
```python
pad = (-L) % chunk
```
For `L = 200, Q = 64`: `(-200) % 64 = 56`, so `Lp = 256`, `nc = 4`. Zeros are
appended to `x, dt, Bm, Cm` along the length axis. (`F.pad(x, (0,0,0,0,0,pad))`
pads the *third*-from-last axis of `(B, L, H, P)`, which is `L`. Getting that arg
order wrong is the classic bug here.) Padding with `dt = 0` gives `log a = 0`,
i.e. `a = 1` (no decay) and update `Delta·B·x = 0` (no write), so the padded
steps are exact no-ops for the state — which is why the *returned final state* is
still correct after padding. That is not obvious and it is tested (§14).

**Chunk reshape, 63-66.** `(B, Lp, ...) -> (B, nc, Q, ...)`.

**Decay in log space, 68-70.**
```python
log_a = -dtb * A          # (B, nc, Q, H) <= 0
log_a = log_a.permute(...)# (B, nc, H, Q)
cs    = torch.cumsum(log_a, dim=-1)   # inclusive, within chunk
```
`cs[..., t] = sum_{r=0..t} log a_r` (local index). Never form `prod a_r`; form
`sum log a_r`. §13 explains why this is non-negotiable.

**The intra-chunk mask, 73 (and `segsum_decay`, 21-33).**
```python
diff = cs.unsqueeze(-1) - cs.unsqueeze(-2)    # diff[t,s] = cs_t - cs_s
mask = torch.tril(ones(Q, Q, bool))
return torch.where(mask, diff, -inf).exp()
```
`cs_t - cs_s = sum_{r=s+1..t} log a_r`, so `exp` of it is `L(t,s)` — including
`L(t,t) = exp(0) = 1`. Above the diagonal, `-inf` then `exp` gives exactly `0`.
Output `(B, nc, H, Q, Q)`.

**Scores and weights, 74-75.**
```python
G = einsum("bctn,bcsn->bcts", Cb, Bb)   # G[t,s] = C_t · B_s     (shared over heads)
W = M * G.unsqueeze(2)                  # (B, nc, H, Q, Q)
```
`G.unsqueeze(2)` inserts the head axis so the shared scores broadcast over `H`.
`W[h,t,s] = L_h(t,s) · (C_t·B_s)` — the matrix `M` of §9.3, per chunk per head.

**Intra output, 76.**
```python
y_intra = einsum("bchts,bcsh,bcshp->bcthp", W, dtb, xb)
```
Read the indices: `y_intra[b,c,t,h,p] = sum_s W[b,c,h,t,s] · dt[b,c,s,h] · x[b,c,s,h,p]`
= `sum_{s<=t} L(t,s)(C_t·B_s) Delta_s x_s[p]`. Exactly §9.3's INTRA.

**Chunk summary state, 79-80.**
```python
decay_to_end = (cs[..., -1:] - cs).exp()        # L(end, s), (B,nc,H,Q)
S = einsum("bchs,bcsh,bcsn,bcshp->bchnp", decay_to_end, dtb, Bb, xb)
```
`S[c] = sum_{s in c} L(end_c, s) · Delta_s B_s x_s^T` — the chunk's *own*
contribution to the state at its right edge, `(N, P)` per head. This is
"summarize the chunk into one state", and it costs `O(Q N P)` per chunk, i.e.
`O(L N P)` overall.

**Inter-chunk recurrence, 83-93.**
```python
chunk_decay = cs[..., -1].exp()      # prod of all a_r in the chunk
s_run = init_state or zeros
for c in range(nc):
    states.append(s_run)
    s_run = chunk_decay[:, c, ...] * s_run + S[:, c]
```
This is the *same* recurrence as the per-step one, at chunk granularity: total
decay across the chunk times the incoming state, plus the chunk's summary. `nc`
sequential steps. `states[c]` is the state *entering* chunk `c`, i.e.
`H_{start_c - 1}`. `s_run` at exit is `H_L` (or `H_Lp`, equal by the no-op
padding argument).

**Inter output, 96.**
```python
y_inter = einsum("bcht,bctn,bchnp->bcthp", cs.exp(), Cb, states)
```
`cs.exp()[t] = prod_{r=0..t} a_r` = the decay applied to the entering state after
`t+1` steps — correct, because the entering state has not yet been multiplied by
`a_0`. Contract with `C_t`, and add.

**Assemble, 98-99.** `y = (y_intra + y_inter)`, reshape to `(B, Lp, H, P)`, slice
`[:, :L]`, cast back, return with `s_run`.

### 10.3 Complexity, and the `Q` trade-off

Per batch element per head:

| term | time | memory |
|---|---|---|
| `segsum_decay` (`M`) | `O(nc · Q^2) = O(L Q)` | `O(L Q)` |
| `G` | `O(nc · Q^2 N) = O(L Q N)` | `O(L Q)` (shared over heads) |
| `W` | `O(L Q)` | `O(L Q)` |
| `y_intra` | `O(L Q P)` | `O(L P)` |
| `S` | `O(L N P)` | `O(nc N P) = O(L N P / Q)` |
| chunk loop | `O(nc N P)` | `O(L N P / Q)` |
| `y_inter` | `O(L N P)` | `O(L P)` |

```
TIME    =  O( L Q (N + P)  +  L N P )
MEMORY  =  O( L Q          +  L N P / Q )
DEPTH   =  O( L / Q )
```

The two time terms are of the same order when `Q ≈ N ≈ P`; that is the standard
SSD guidance. Be precise if pressed: with `N = P`, the terms are `2LQN` and
`LN^2`, which are *equal* at `Q = N/2` and within a factor of two at `Q = N`.
At [ssm.py](../flowquake/ssm.py)'s module defaults (`d_model = 256`, `expand = 2`,
`n_heads = 8`, `d_state = 64`, `chunk = 64`) you get
`P = d_inner/n_heads = 512/8 = 64`, `N = 64`, `Q = 64` — all three equal, i.e.
squarely in that regime.

**Memory is linear in `Q`, depth is inverse in `Q`.** Concretely, at the shape
the encoder is actually built at in this repo (`B = 8`, `L = 2048`, `H = 6`,
`Q = 64`, `N = P = 32`, from
[runs/ablation_h/h4.yaml](../runs/ablation_h/h4.yaml): `batch_size 8`,
`window 2048`, `n_heads 6`, `d_state 32`, `d_model 96` with `expand 2` so
`P = 192/6 = 32`, `chunk 64`):

```
nc        =  L / Q  =  2048 / 64  =  32
W tensor  =  B · nc · H · Q^2  =  8 · 32 · 6 · 4096  =  6,291,456 floats  =  25.2 MB
```

`segsum_decay` materializes `diff`, the `where` result, and its `exp` — three
tensors of that size — so a transient peak of roughly 75 MB per layer per
forward. What is *retained* for backward is less than that (`exp` saves only its
own output, `where` saves only the boolean mask), but `M` and `W` are both
`(B, nc, H, Q, Q)` and both live until backward, so budget tens of MB per layer
times 4 layers for the decay machinery alone. Double `Q` to 128 and it doubles;
halve it to 32 and it halves while the sequential loop grows from 32 to 64
Python iterations. That is the trade, in numbers you can quote.

(The `n1_density`-family production configs carry the same architecture fields,
but with `h_bottleneck: 0` no `SSDBlock` is ever constructed there — see §15.1 —
so these are the ablation runs' numbers, not the headline runs'.)

### 10.4 Gradients

Autograd differentiates straight through the Python loop at
[ssm.py:90-92](../flowquake/ssm.py#L90-L92), building a graph `nc` deep in the
chunk-state recurrence. For `nc = 32` this is fine. For `encode_full`
([model.py:202-209](../flowquake/model.py#L202-L209)) with `segment = 16384`,
`nc = 256` per segment — but that path is under `@torch.no_grad()`, so no graph
is built. Worth knowing which is which.

---

## 11. The alternative: Blelloch's associative scan

### 11.1 The recurrence is an associative scan

The chunk recurrence `s_c = g_c s_{c-1} + u_c` (and the per-step one, `H_t = a_t
H_{t-1} + v_t`) is a **first-order linear recurrence**. Represent each step by
its affine map `(g, u): s -> g s + u`, and define composition

```
(g1, u1) ⊗ (g2, u2)  :=  (g1 g2,  g2 u1 + u2)         "apply 1, then 2"
```

**Associativity, proved:**

```
((g1,u1) ⊗ (g2,u2)) ⊗ (g3,u3) = (g1g2, g2u1+u2) ⊗ (g3,u3)
                              = (g1g2g3,  g3(g2u1+u2) + u3)
                              = (g1g2g3,  g3g2u1 + g3u2 + u3)

(g1,u1) ⊗ ((g2,u2) ⊗ (g3,u3)) = (g1,u1) ⊗ (g2g3, g3u2+u3)
                              = (g1g2g3,  g2g3u1 + g3u2 + u3)
```

Equal. So the prefix "sums" under `⊗` can be computed by any parallel-scan
algorithm.

### 11.2 Blelloch's work-efficient scan

Blelloch (1990, CMU tech report, "Prefix sums and their applications") gives a
two-phase algorithm on a balanced binary tree over the `n` elements:

- **Up-sweep (reduce):** `log n` levels, each combining pairs, computing partial
  reductions bottom-up.
- **Down-sweep:** `log n` levels, pushing exclusive prefixes back down.

Total **work `O(n)`, depth `O(log n)`**. (The simpler Hillis–Steele scan has
depth `O(log n)` but work `O(n log n)`.) Martin & Cundy (2018, ICLR,
"Parallelizing linear recurrent neural nets over sequence length") is the paper
that brought this into deep learning for linear RNNs; S5 (Smith, Warrington &
Linderman 2023, ICLR) uses a parallel scan rather than the FFT, and Mamba-1's
CUDA kernel is a fused, memory-aware version of the same idea.

### 11.3 When you would prefer it

Prefer a Blelloch scan when:

- **`L/Q` is large and depth dominates.** Depth `O(log L)` beats `O(L/Q)` once
  `L/Q` is in the hundreds and the per-step work is too small to keep the device
  busy.
- **You cannot afford `O(LQ)` memory.** The scan never materializes a `Q x Q`
  block; its transient is `O(L · state_size)`.
- **`N` and `P` are large relative to `Q`.** The chunked form's advantage comes
  from the `Q x Q` blocks being matmul-shaped; if `Q` must be small, the
  advantage evaporates.

Prefer chunking (what Mamba-2 and [ssm.py](../flowquake/ssm.py) do) when:

- **You have matmul hardware.** The `Q x Q` intra-chunk block is a batched matmul
  that lands on tensor cores. A scan is elementwise: it is bandwidth-bound and
  uses a small fraction of a modern GPU's peak. This is the real reason Mamba-2
  moved from scan to chunking, and it is the answer to "why not just use
  Blelloch?"
- **You want a framework-portable implementation.** [ssm.py](../flowquake/ssm.py) is
  pure PyTorch einsums; a work-efficient scan with good memory behaviour
  essentially requires a custom kernel. The module docstring says exactly this:
  *"No custom CUDA kernels (runs on Windows)"*
  ([ssm.py:3-5](../flowquake/ssm.py#L3-L5)).

---

## 12. Streaming: prefill, step, and a state that does not grow

### 12.1 The two entry points

- **`prefill`** ([ssm.py:274-285](../flowquake/ssm.py#L274-L285)) runs a whole
  segment with an optional incoming `(scan state, conv cache)` per layer and
  returns the outgoing pair. This is the "absorb the observed catalog" phase.
- **`step`** ([SSDBlock.step, ssm.py:207-232](../flowquake/ssm.py#L207-L232)) is the
  single-event update. It is the naive recurrence for one `t`:
  `a = exp(-dt*A)`, `state = a*state + B ⊗ (dt*x)`, `y = C·state + D*x`. Cost
  `O(H N P)` per layer, **independent of how many events came before**.

The depthwise causal convolution needs its own cache, because a conv of width
`d_conv = 4` needs the previous 3 inputs. `step` maintains it manually
([ssm.py:214-217](../flowquake/ssm.py#L214-L217)): concatenate the cached window
with the new input, dot with the conv weights, slide the window. The `forward`
path returns the same cache when `return_state=True`
([ssm.py:188](../flowquake/ssm.py#L188)). If you forget the conv cache, the first 3
events after a segment boundary are silently wrong — which is precisely what
`test_encoder_prefill_matches_forward` catches.

### 12.2 Constant vs growing state, in numbers

For FlowQuake's production-size architecture (`d_model = 96`, `n_layers = 4`,
`d_state = 32`, `n_heads = 6`, `expand = 2` ⇒ `d_inner = 192`, `d_head = 32`,
`d_xbc = 192 + 64 = 256`):

| quantity | formula | value | per 10^4 lanes (fp32) |
|---|---|---|---|
| SSM state | `n_layers · H · N · P` = `4 · 6 · 32 · 32` | 24,576 floats = 98.3 KB | **983 MB** |
| conv cache | `n_layers · d_xbc · (d_conv-1)` = `4 · 256 · 3` | 3,072 floats = 12.3 KB | 123 MB |
| transformer KV cache at `L = 70,000` | `2 · n_layers · L · d_model` = `2 · 4 · 70000 · 96` | 53.8 M floats = 215 MB | **2.15 TB** |

A factor of roughly 2,000, and the SSM's number does not grow as the simulation
advances. [flowquake/ntest.py](../flowquake/ntest.py) broadcasts one prefilled state
across `n_sims` lanes and calls `encoder.step` once per simulated event
([ntest.py:124-125](../flowquake/ntest.py#L124-L125)). **This is the strongest
architectural argument for an SSM in this application**, and it survives the fact
that the encoder is currently disabled — it is an argument about what the design
*enables*.

### 12.3 The invariant that makes it legal

Streaming is correct iff

```
scan(u_{1..L})  ==  concat( scan(u_{1..k}),  scan(u_{k+1..L} | init_state = state_k) )
```

i.e. the returned `s_run` is *exactly* `H_k` and `init_state` is interpreted as
`H_0`. This is not a property you get for free from a parallel algorithm — it is
a claim about the algebra — and it is exactly what
`test_scan_init_state_continuation` asserts (§14).

---

## 13. Numerics: why fp32 and log-space are non-negotiable

[ssm.py:52](../flowquake/ssm.py#L52) casts every input to fp32 on entry:

```python
x, dt, A, Bm, Cm = (t.float() for t in (x, dt, A, Bm, Cm))
```

and restores the original dtype only at the end. Under `torch.autocast` those
einsums would otherwise run in bf16/fp16. Here is why that would be fatal.

### 13.1 The exponential-of-a-cumulative-sum problem

At the module's initialization ranges (`A_min = 1, A_max = 16`,
`dt_min = 1e-3, dt_max = 0.1`, [ssm.py:132-135](../flowquake/ssm.py#L132-L135)), a
single step can have `log a = -Delta A = -0.1 · 16 = -1.6`. Over one chunk of
`Q = 64`:

```
cs spans  64 · (-1.6)  =  -102.4
exp(-102.4)  =  3.4e-45      (fp32 min normal 1.2e-38 -> flushes to subnormal/zero)
exp(+102.4)  =  2.96e+44     (fp32 max 3.40e+38    -> overflows to inf)
```

And `Delta_t = softplus(...)` is unbounded above, so at *runtime* it can exceed
`dt_max`. In fp16 (max 65504, min normal 6.1e-5) it is far worse: `exp(-11)` is
already subnormal, so a decay accumulated over ~7 steps at `|log a| = 1.6` per
step is lost.

**Consequence 1: never form products of `a_r`.** Form `sum log a_r` and
exponentiate the *difference* `cs_t - cs_s`. The difference is bounded by the
decay over `t - s` steps, not by the decay over the whole chunk, so it stays in
range even when `cs` itself does not.

**Consequence 2: fp32 minimum.** `cs ≈ -100` in fp32 has absolute resolution
`7.6e-6` (24-bit mantissa; `numpy.spacing(np.float32(-100.0))`), so `cs_t - cs_s`
carries `~1e-5` absolute error in the exponent, i.e. `~1e-5` relative error in
the decay. Acceptable. In fp16, `cs` near `-100` has absolute resolution
`0.0625` — and `exp(0.0625) - 1 = 6.4%`, i.e. a ~6% error in every decay factor.

### 13.2 The `-inf` masking trick, and why the order matters

`segsum_decay` ([ssm.py:32-33](../flowquake/ssm.py#L32-L33)):

```python
mask = torch.tril(ones(Q, Q, bool))
return torch.where(mask, diff, -inf).exp()
```

The temptation is to write `torch.exp(diff) * mask.float()`. That is **wrong**,
not merely inelegant:

- `log a_r <= 0` so `cs` is non-increasing, so for `s > t` we have
  `diff[t,s] = cs_t - cs_s >= 0` — the strictly upper triangle contains
  `exp` of *positive* numbers, up to `exp(+102.4) = 2.96e44`, which **overflows
  fp32 to `inf`**.
- Then `inf * 0.0 = NaN`, and the NaN propagates through `W`, `y_intra`, the loss,
  and every parameter.

Masking *before* the exponential means the upper triangle is `exp(-inf) = 0`
exactly, with no intermediate infinity. Note that every *other* `exp` in
`selective_scan_chunked` is provably of a non-positive argument
(`decay_to_end = exp(cs_end - cs_s) <= 1` since `cs_end <= cs_s`;
`chunk_decay = exp(cs[-1]) <= 1`; `cs.exp() <= 1`). **`segsum_decay` is the one
place in the file where the sign can flip, and it is exactly where the mask is.**
That is not a coincidence; it is the invariant to state when asked.

### 13.3 What could still bite

- **Catastrophic cancellation** in `cs_t - cs_s` when both are large: bounded as
  above, fine at fp32 for `Q = 64`, would need care at `Q = 1024`.
- **`decay_to_end` underflow** for `s` near a chunk's start when the chunk decay
  is severe: those terms *should* be ~0 and flushing them to 0 is harmless.
- **The `D ⊙ x` skip** at [ssm.py:199](../flowquake/ssm.py#L199) is added outside
  the fp32 region's cast-back, in whatever dtype `x` was — a mixed-precision
  wrinkle worth knowing about, though `D` is initialized to ones and the term is
  well-scaled.

---

## 14. Testing a scan implementation

[tests/test_ssm.py](../tests/test_ssm.py) has **five** tests, not two. (STACK.md
describes two; the file has five. Both statements about *content* are correct —
STACK.md is summarizing, not miscounting the file's substance.) They are:

| test | what it pins down |
|---|---|
| `test_chunked_scan_matches_naive` | the chunked algorithm equals the serial recurrence, `y` **and** final state, at `L = 200`, `chunk = 64` |
| `test_scan_init_state_continuation` | `scan[0:64]` then `scan[64:128]` seeded with `init_state = s1` equals `scan[0:128]`, at `chunk = 32` |
| `test_block_causality` | perturbing `u[60:]` leaves `y[:60]` unchanged (and does change `y[60:]`), on an `SSDBlock` at `chunk = 16` |
| `test_encoder_prefill_matches_forward` | segmented `prefill` equals a single `forward` (conv cache included), `L = 90`, `chunk = 16` |
| `test_encoder_step_matches_forward` | 10 single-event `step`s after a prefill equal the batch forward's last position, `chunk = 16` |

### 14.1 Why the first two are the right two

**Test 1 pins down *exactness*.** The chunked decomposition is claimed to be an
algebraic identity, not an approximation. The reference `selective_scan_ref`
([ssm.py:102-118](../flowquake/ssm.py#L102-L118)) is a five-line `for` loop that any
reader can verify against §9.2 by eye, and it runs in **fp64** — so the reference
is not the thing under test, and the `atol/rtol = 1e-4` tolerance is measuring
fp32 accumulation error in the implementation, not disagreement about the maths.
It checks the final state as well as the outputs.

`L = 200` is chosen deliberately: `200 = 3·64 + 8`, so `pad = 56` and the last
chunk is 87.5% padding. This exercises (a) the padding branch, (b) a partial
final chunk, and (c) the claim from §10.2 that padded steps are state no-ops so
the *returned final state* is still `H_200` and not `H_256`. If padding were done
with `dt = 1` instead of `0`, or if the slice `[:, :L]` were off by one, this test
fails.

**Test 2 pins down *compositionality*.** Test 1 only ever calls the scan with
`init_state = None`, so it validates the output half of the state contract but
not the input half. Test 2 validates the round trip: what comes out as `s_run`
must be exactly what goes in as `init_state`, with identical semantics. That
invariant is what licenses (a) streaming simulation
([ntest.py](../flowquake/ntest.py)), (b) segmented full-catalog evaluation
([model.py:202-209](../flowquake/model.py#L202-L209)), and (c) in principle,
gradient checkpointing over segments. It is a *different* property from
exactness, and it does not follow from it.

Together: **one test says the fast algorithm computes the right function; the
other says the function composes over splits.** Those are the two theorems the
implementation claims. Everything else (causality, prefill, step) is a
consequence tested for defence in depth.

### 14.2 What is *not* tested — say this before you are asked

1. **No gradient test.** There is no `gradcheck` and no finite-difference
   comparison. A wrong backward — a transposed einsum in the vjp, a detached
   `states` list — would pass all five tests and silently train a different
   model. This is the single largest hole.
2. **The two scan-level tests never reach the numerical extreme regime.** The
   generator in `_rand_inputs` uses `dt = rand·0.1 + 1e-3 in [1e-3, 0.101]` and
   `A = rand·10 + 0.5 in [0.5, 10.5]`
   ([test_ssm.py:14-15](../tests/test_ssm.py#L14-L15)), so the worst per-step
   `|log a|` is `0.101 · 10.5 = 1.06` and the worst span over a `Q = 64` chunk
   is `-67.9`. `exp(+67.9) = 3.0e29`, which is **finite in fp32**. So replacing
   the `-inf` mask with a multiplicative `{0,1}` mask would still pass
   `test_chunked_scan_matches_naive` and `test_scan_init_state_continuation`,
   even though it produces `inf * 0 = NaN` at the module's own initialization
   ranges (`A_max = 16`, §13.2). The coverage that exists is *accidental*:
   `test_block_causality` perturbs `u[:, 60:] += 100.0`, which drives
   `dt = softplus(linear(u) + dt_bias)` to `O(10-100)` and hence chunk spans far
   past fp32's `exp` range, so it would plausibly go NaN and fail. I could not
   run it (`torch` is not installed in this checkout), so treat "the suite would
   still pass" as established for the two scan tests and *unresolved* for the
   block test. Either way there is no test that *deliberately* covers the
   overflow regime, and that is the gap to volunteer.
3. **Length is only tested at `L <= 200`** (the five tests use `L = 200, 128,
   100, 90, 40`). fp32 accumulation error growth at `L = 2048` (the training
   crop) or `L = 16384` (the evaluation segment) is never measured against fp64.
4. **`chunk` is only tested at 64, 32 and 16.** `chunk = 1` and `chunk >= L` are
   the degenerate cases that should also agree and are not exercised.

---

## 15. The honest part: this encoder is off in every production run

### 15.1 The counts, verified from this checkout

[model.py:76-84](../flowquake/model.py#L76-L84):

```python
if h_bottleneck > 0:
    self.encoder = SSMEncoder(...)
    self.h_proj  = nn.Linear(d_model, h_bottleneck)
else:
    self.encoder = None
cond_dim = len(SAFE_TOKEN_DIMS) + h_bottleneck
```

So `h_bottleneck = 0` means the encoder is **never constructed** — not
regularized, not down-weighted, absent. Counting `h_bottleneck` across every
git-tracked YAML in this repository (`git ls-files | grep -E '\.(ya?ml)$'`, 123
files):

| value of `h_bottleneck` | count of tracked YAMLs |
|---|---|
| `0` (explicit) | **114** |
| key absent → defaults to `0` via [config.py:41](../flowquake/config.py#L41) | 3 (`configs/comcat25_lr3.yaml`, `runs/comcat25_lr3/config.yaml`, `runs/smoke/config.yaml`) |
| `4` | 2 (`runs/ablation_h/h4.yaml` and `runs/ablation_h/h4/config.yaml` — one experiment, two copies) |
| `16` | 2 (same pattern) |
| `64` | 2 (same pattern) |

Restricted to committed *run directories*: **74** contain a `config.yaml`, and
exactly **3** of them (`runs/ablation_h/h4`, `.../h16`, `.../h64`) instantiate the
encoder. **71 do not** — 69 setting `h_bottleneck: 0` explicitly and 2
(`runs/comcat25_lr3`, `runs/smoke`) omitting the key and inheriting the `0`
default. Of the 33 files in `configs/` (all of them YAML), **zero** instantiate
it: 32 set `0` and `configs/comcat25_lr3.yaml` omits the key.

The three ComCat seeds behind the headline temporal number
(`runs/fullsuite_summary.json` names `n1_density`, `n1_s1553`, `n1_s1554`) all
carry `h_bottleneck: 0` — verified in
[runs/n1_density/config.yaml:30](../runs/n1_density/config.yaml#L30) and its two
siblings.

**A discrepancy to report.** [STACK.md](../STACK.md) Part 0 states: *"Across all
tracked YAMLs, 181 set `h_bottleneck: 0` and exactly three set it higher."* The
"exactly three" is right if you count distinct ablation *experiments*; **181 does
not match this checkout** — the count is 114, and the tracked-YAML total is 123.
Use the numbers in the table above.

### 15.2 What production uses instead

With `h = 0`, [`_cond`](../flowquake/model.py#L160-L171) returns
`tokens[mask][:, SAFE_TOKEN_DIMS]` and nothing else: 30 dimensions, being
`log tau`, `m`, and the 28 relational lag features
([model.py:35](../flowquake/model.py#L35)). The long-history mechanisms are:

1. **Exponentially spaced relational lags.** `RECENCY_LAGS = (1,2,4,8,16,32,64)`
   ([data.py:26](../flowquake/data.py#L26)); for each lag `k`, the four features
   `log(t_i - t_{i-k})`, `x_i - x_{i-k}`, `y_i - y_{i-k}`, `m_{i-k}` — seven
   lags × four features = the 28 relational dims. Seven *lags* cover 64 events
   the way a dilated conv covers a long receptive field cheaply.
2. **Long-lived big-trigger mixture components.** `BIG_M = 16` slots holding the
   largest `m >= 4.5` events in the trailing 730 days
   ([data.py:34-37](../flowquake/data.py#L34-L37)).
3. **Optional nearest-prior-event tier** (`n_near > 0`), the `n_near` spatially
   nearest strictly-prior events over *all* history, via a KD-tree
   ([data.py:76-117](../flowquake/data.py#L76-L117)). **Not enabled in the ComCat
   headline runs** — `n_near` is absent from
   [runs/n1_density/config.yaml](../runs/n1_density/config.yaml) and defaults to `0`
   ([config.py:20](../flowquake/config.py#L20)).

### 15.3 Is that a fair substitute? Honest assessment

**Where it is genuinely fine.** Mechanisms 2 and 3 feed the `lastk` tensor, which
reaches the **spatial** head via `_comp_inputs`
([model.py:99-121](../flowquake/model.py#L99-L121)). So for space, "the model can
condition on a mainshock from two years ago" is *true and structural*: that
mainshock is literally a mixture component with its own kernel. That is a good
design and arguably better than an SSM for this purpose, because it gives
**exact** recall of a specific past event's coordinates — exactly the capability a
fixed-size state cannot have (§4.4).

**Where it is not fine, and you must say so.** `lastk` never reaches the temporal
head. `head_t` and `head_m` see only `cond`
([model.py:181-192](../flowquake/model.py#L181-L192)), and with `h = 0`, `cond` is
the 30 safe token dims. The largest lag in those dims is
`RECENCY_LAGS[-1] = 64`.

> **The production temporal head's receptive field reaches back exactly 64
> events** — the conditioning row for event `i` is a function of events
> `i-64 … i` and of nothing older.

At ComCat's rate that is about 70 days ([data.py:30-33](../flowquake/data.py#L30-L33)).
It is a *fixed window* — a bigger one than DeepSTPP's 20 events, and enriched
with `log`-spaced order statistics rather than raw events, but a fixed window
nonetheless. That is the very limitation
[MANUSCRIPT.md](../MANUSCRIPT.md) §1 names as one of the "two reasons NPPs
underperform". The headline temporal win (`tll` 1.4868 vs ETAS 1.4343,
[runs/fullsuite_summary.json](../runs/fullsuite_summary.json)) is therefore produced
by a **64-event-window** model, not a whole-catalog one.

Is the win therefore fake? No — it is a real, multi-seed, block-bootstrapped win
against a full-history ETAS. But the *mechanism story attached to it* in
[WORKING.md:18-20](../WORKING.md#L18-L20) ("A Mamba-style selective state-space
encoder reads the entire history instead of a fixed window, and a rectified-flow
head gives exact likelihoods for the next event's time") is not what produced it.
The defensible mechanism story is: **a flexible exact-likelihood density over
`log tau`, conditioned on well-chosen multi-scale relational statistics, beats a
parametric Omori intensity on next-gap prediction.** That is a good result. It is
just a different result.

### 15.4 "So why is the Mamba encoder in the paper?"

**The defensible answer.** The encoder is the **independent variable of §4.3**,
the paper's mechanism result. The question "what happens when the heads see a
learned whole-catalog embedding?" cannot be asked without building one. From
[runs/ablation_h/memorization_figure.json](../runs/ablation_h/memorization_figure.json)
at `ckpt_last`:

| h | train nll | held-out nll | gap |
|---|---|---|---|
| 0 | 7.281167 | 7.621031 | **0.339864** |
| 4 | 4.143447 | 19.645805 | **15.502358** |
| 16 | 4.182444 | 18.731383 | 14.548939 |
| 64 | 4.272901 | 18.330903 | 14.058002 |

and from [runs/ablation_h/ablation_h.json](../runs/ablation_h/ablation_h.json), the
best held-out checkpoint for `h = 4, 16, 64` is `step: 250` in all three cases —
the *first* validation ever run. Memorization is immediate, not a late-training
pathology you can early-stop out of. The encoder is the **negative control that
establishes why the structured heads are necessary**, and a negative control that
works is worth its code.

Secondarily: the streaming state (§12) is what would make 10^4-lane simulation
feasible if the channel were ever opened, and `ntest.py` already branches on
`model.encoder is not None` ([ntest.py:46,124](../flowquake/ntest.py#L46)) so the
capability is wired, not aspirational.

**The honest answer.** In every production number in this repository,
`self.encoder is None`. [README.md:3](../README.md#L3) opens with *"Selective-SSM
(Mamba-style) whole-catalog encoder + flow-matching marked point process"* —
leading with a component that is not in the model being measured. The README does
correct itself twenty lines later (*"Production runs with `h_bottleneck=0`"*), and
MANUSCRIPT §4.3 is explicit; but a reviewer who reads the first sentence and then
greps for `h_bottleneck` will feel misled, and will be right to. The accurate
one-line framing is:

> FlowQuake is a structured, observation-anchored marked temporal point process.
> The selective-SSM encoder is the *ablation arm* that demonstrates why the
> structure is necessary.

**A second honest point, and the one a sharp examiner will actually press.** The
`h > 0` arm changes *three* things at once relative to `h = 0`: it adds (i) an
SSM encoder, (ii) a learned linear projection to `h` dims, and (iii) a channel
that has seen **absolute `x, y`** — because `enc_in = tokens`, the full 32-dim
token ([model.py:165](../flowquake/model.py#L165)), whereas `SAFE_TOKEN_DIMS`
excludes dims 1 and 2. The measured collapse (train `sll` rising to `-7.27` at
`h = 4`, i.e. mass pinned on training epicentres) is diagnostic of (iii). So the
experiment supports *"any learned global channel carrying absolute geography
causes memorization"* — which is what MANUSCRIPT §4.3's own sentence says ("Any
learned global embedding lets the heads memorize the training catalog") — and
**not** a claim about selective state-space models specifically. The missing
control is an `h > 0` arm whose encoder input is also restricted to
`SAFE_TOKEN_DIMS`. That control does not exist in `runs/`. If it were run and
memorization vanished, §4.3's finding would be about *coordinates*, not about
*flexibility*. Have that answer ready; it is the sharpest attack on the paper's
central mechanism claim.

---

## Worked example: unroll `L = 4` by hand, then chunk it with `Q = 2`

One head, `N = P = 1`, so the state is a scalar and every quantity fits on a
line. Everything below is exact in binary fractions — no rounding.

### Setup

Choose `A_h = ln 2` so the decays come out as powers of one half:

```
Delta = [ 1,  2,  1,  3 ]
a_t   = exp(-Delta_t · ln 2) = 2^{-Delta_t}
a     = [ 1/2, 1/4, 1/2, 1/8 ]

B     = [ 1,  1,  2,  1 ]
C     = [ 1,  2,  1,  1 ]
x     = [ 1,  2, -1,  1 ]
```

Precompute the per-step write `u_s = Delta_s · B_s · x_s`:

```
u_1 = 1·1·( 1) =  1
u_2 = 2·1·( 2) =  4
u_3 = 1·2·(-1) = -2
u_4 = 3·1·( 1) =  3
```

### Part A — the serial recurrence (`selective_scan_ref`)

`H_t = a_t H_{t-1} + u_t`, `H_0 = 0`, `y_t = C_t H_t`:

```
H_1 = (1/2)(0)      +  1  =  1
H_2 = (1/4)(1)      +  4  =  0.25 + 4      = 4.25
H_3 = (1/2)(4.25)   + (-2)=  2.125 - 2     = 0.125
H_4 = (1/8)(0.125)  +  3  =  0.015625 + 3  = 3.015625

y_1 = 1 · 1        = 1
y_2 = 2 · 4.25     = 8.5
y_3 = 1 · 0.125    = 0.125
y_4 = 1 · 3.015625 = 3.015625
```

### Part B — the chunked decomposition, `Q = 2`

**Chunk 0 = steps {1,2}.**

`log a = [-ln2, -2 ln2]`, `cs = [-ln2, -3 ln2]`, so `exp(cs) = [1/2, 1/8]`.

`segsum_decay`: `M[t,s] = exp(cs_t - cs_s)` on the lower triangle.

```
M = [ 1     0 ]        M[2,1] = exp(-3ln2 + ln2) = 2^-2 = 1/4 = a_2   ✓
    [ 1/4   1 ]
```

`G[t,s] = C_t B_s`:

```
G = [ 1·1  1·1 ]  = [ 1  1 ]
    [ 2·1  2·1 ]    [ 2  2 ]
```

`W = M ⊙ G`:

```
W = [ 1     0 ]
    [ 1/2   2 ]
```

`Delta_s x_s = [1·1, 2·2] = [1, 4]`. So

```
y_intra[1] = 1·1              = 1
y_intra[2] = (1/2)(1) + 2·(4) = 0.5 + 8 = 8.5
```

The state entering chunk 0 is `0`, so `y_inter = 0` and `y_1 = 1`, `y_2 = 8.5`. ✓

Chunk summary, `decay_to_end[s] = exp(cs_end - cs_s) = [1/4, 1]`:

```
S_0 = (1/4)(u_1) + (1)(u_2) = 0.25 + 4 = 4.25
chunk_decay_0 = exp(cs[-1]) = 2^-3 = 1/8
s_run = (1/8)(0) + 4.25 = 4.25          (= H_2 ✓)
```

**Chunk 1 = steps {3,4}.**

`log a = [-ln2, -3 ln2]`, `cs = [-ln2, -4 ln2]`, `exp(cs) = [1/2, 1/16]`.

```
M = [ 1     0 ]        M[4,3] = exp(-4ln2 + ln2) = 2^-3 = 1/8 = a_4   ✓
    [ 1/8   1 ]

G = [ C_3 B_3  C_3 B_4 ]  = [ 1·2  1·1 ]  = [ 2  1 ]
    [ C_4 B_3  C_4 B_4 ]    [ 1·2  1·1 ]    [ 2  1 ]

W = [ 2     0 ]
    [ 1/4   1 ]
```

`Delta_s x_s = [1·(-1), 3·1] = [-1, 3]`:

```
y_intra[3] = 2·(-1)                 = -2
y_intra[4] = (1/4)(-1) + (1)(3)     = -0.25 + 3 = 2.75
```

Inter-chunk, with `state_in = 4.25` and `y_inter[t] = C_t · exp(cs_t) · state_in`:

```
y_inter[3] = 1 · (1/2)  · 4.25 = 2.125
y_inter[4] = 1 · (1/16) · 4.25 = 0.265625
```

Add:

```
y_3 = -2   + 2.125    = 0.125       ✓ matches Part A
y_4 =  2.75 + 0.265625 = 3.015625   ✓ matches Part A
```

And the carried state:

```
S_1 = exp(cs_4 - cs_3)·u_3 + 1·u_4 = (1/8)(-2) + 3 = 2.75
chunk_decay_1 = exp(cs_4) = 2^-4 = 1/16
s_run = (1/16)(4.25) + 2.75 = 0.265625 + 2.75 = 3.015625   (= H_4 ✓)
```

### Redo it in five lines

```python
import numpy as np
A = np.log(2); dt = np.array([1.,2.,1.,3.]); a = np.exp(-dt*A)
B = np.array([1.,1.,2.,1.]); C = np.array([1.,2.,1.,1.]); x = np.array([1.,2.,-1.,1.])
h = 0.; y = []
for t in range(4): h = a[t]*h + dt[t]*B[t]*x[t]; y.append(C[t]*h)
print(y)   # [1.0, 8.5, 0.125, 3.015625]
```

Then reproduce Part B chunk by chunk and check `np.allclose`. If your chunked
version disagrees, the three usual culprits are: an off-by-one in `cs` (exclusive
vs inclusive cumsum), forgetting that `L(t,t) = 1`, and applying `chunk_decay`
*after* rather than *before* adding the chunk summary.

---

## How this shows up in FlowQuake

Rather than restate the code walkthrough, here is the theory-to-artifact map;
[STACK.md §8](../STACK.md) is the code-level companion.

| theory in this chapter | where it lives |
|---|---|
| §6 discretization, `a_t = exp(-Delta_t A_h)` | [ssm.py:68](../flowquake/ssm.py#L68) (`log_a = -dtb * A`); Euler `B_bar` at [76, 80, 115](../flowquake/ssm.py#L76) |
| §9 SSD recurrence | module docstring [ssm.py:7-9](../flowquake/ssm.py#L7-L9); reference implementation [102-118](../flowquake/ssm.py#L102-L118) |
| §9.3 duality — the masked quadratic form | [ssm.py:73-76](../flowquake/ssm.py#L73-L76) (`M`, `G`, `W`, `y_intra`) |
| §10 chunked scan | [ssm.py:36-99](../flowquake/ssm.py#L36-L99) in full |
| §10.2 padding path | [ssm.py:54-61, 98](../flowquake/ssm.py#L54-L61) |
| §12 streaming | [SSDBlock.step 207-232](../flowquake/ssm.py#L207-L232), [SSMEncoder.prefill/step 274-295](../flowquake/ssm.py#L274-L295); consumed by [ntest.py:46-50,124-125](../flowquake/ntest.py#L46-L50) |
| §13 numerics | fp32 cast [ssm.py:52](../flowquake/ssm.py#L52); `-inf` mask [ssm.py:32-33](../flowquake/ssm.py#L32-L33) |
| §14 tests | [tests/test_ssm.py](../tests/test_ssm.py), five tests |
| §15 the disabled encoder | [model.py:76-84](../flowquake/model.py#L76-L84), [model.py:160-171](../flowquake/model.py#L160-L171), [config.py:41](../flowquake/config.py#L41) |
| §15 the memorization result | [runs/ablation_h/memorization_figure.json](../runs/ablation_h/memorization_figure.json), [runs/ablation_h/ablation_h.json](../runs/ablation_h/ablation_h.json) |

**Parameter count.** At the ablation configs' size (`d_model = 96`,
`n_layers = 4`, `d_state = 32`, `n_heads = 6`, `expand = 2`, `d_conv = 4`;
[runs/ablation_h/h4.yaml](../runs/ablation_h/h4.yaml)) the encoder is, per block:

```
in_proj  Linear(96 -> 192 + 256 + 6 = 454)   96·454 + 454 = 44,038
conv     Conv1d(256, 256, k=4, groups=256)   256·4 + 256  =  1,280
dt_bias 6  +  A_log 6  +  D 192  +  RMSNorm 192           =    396
out_proj Linear(192 -> 96)                   192·96 + 96  = 18,528
                                                     block = 64,242
```

plus `embed` `Linear(32->96) = 3,168`, four pre-norms `= 384`, `norm_f = 96`:

```
SSMEncoder total  =  3,168 + 384 + 4·64,242 + 96  =  260,616 parameters
```

(Counted by hand from the constructor, twice, independently — `torch` is not
installed in this checkout, so I could not execute it. Two assumptions worth
naming: `nn.Conv1d(256, 256, 4, groups=256)` has weight shape `(256, 1, 4)`
plus a 256-bias, and `nn.RMSNorm` defaults to `elementwise_affine=True`, i.e. a
weight and no bias. Verify with
`sum(p.numel() for p in SSMEncoder(d_in=32, d_model=96, n_layers=4, d_state=32,
n_heads=6).parameters())`.) For comparison, `h_proj` at `h = 4` is
`96·4 + 4 = 388` parameters — the entire memorization channel of §4.3 is 388
weights wide, which is itself a striking fact: it does not take much capacity to
leak geography.

---

## Common misconceptions

1. **People think** SSMs are `O(L)` and transformers `O(L^2)`, so SSMs are
   strictly better. **Actually** the chunked SSD form is
   `O(L Q (N+P) + L N P)` — for short `L` and small `N, P` a fused attention
   kernel is faster, and the constant-size state is a hard information bottleneck
   (§4.4). **Why it matters:** for a catalog you often need *exact* recall of a
   specific old event, which is what attention and FlowQuake's mixture
   components give you and what an SSM state cannot.

2. **People think** "state space model" means the same thing as in Kalman
   filtering / classical state-space time series. **Actually** the equations are
   the same but deep SSMs are deterministic: no process noise, no observation
   noise, no posterior, no uncertainty propagation; trained by SGD on a
   downstream loss rather than by EM/MLE on a state-space likelihood.
   **Why it matters:** a statistician on your committee will ask, and "it's a
   Kalman filter" is wrong in exactly the part they care about.

3. **People think** `Delta` is the data's sampling interval or a hyperparameter.
   **Actually** in S6/SSD it is a learned, input-dependent gate
   (`Delta_t = softplus(linear(u_t) + bias)`, [ssm.py:193](../flowquake/ssm.py#L193)),
   and in FlowQuake it is **not** the physical inter-event time — the SSM indexes
   events, not seconds. **Why it matters:** the obvious improvement (§8.5) is to
   *make* it the physical gap, and you should know that has not been tried here.

4. **People think** selectivity means the model attends to important past tokens.
   **Actually** it is a scalar gate per head per step controlling *how much state
   to keep and how hard to write* — there is no content-based retrieval of a
   *specific* past position. **Why it matters:** "Mamba can look back at the
   mainshock" is false; "Mamba can avoid overwriting whatever it stored about the
   mainshock" is true.

5. **People think** the chunked scan is a fast approximation. **Actually** it is
   an exact algebraic identity; the test asserts agreement with an fp64 serial
   recurrence to `1e-4`, and the residual is fp32 accumulation, not method error.
   **Why it matters:** if someone reports a "chunking error" they have a bug, not
   a tolerance.

6. **People think** you need custom CUDA kernels for Mamba. **Actually**
   [ssm.py](../flowquake/ssm.py) is ~300 lines of pure PyTorch einsums. **Why it
   matters:** kernels buy memory (never materializing the `Q x Q` blocks) and
   wall-clock, not correctness — and the pure version is what makes this
   auditable.

7. **People think** linear attention is a lossy approximation of softmax
   attention. **Actually** it is a *different* similarity function. Performer's
   random features do approximate softmax; `elu(x)+1` and the identity map used in
   SSMs approximate nothing. **Why it matters:** "linear attention loses accuracy
   because the approximation is bad" is the wrong diagnosis; the right one is the
   rank bottleneck of §4.4.

8. **People think** FlowQuake's SSM inherits HiPPO's principled memory.
   **Actually** [ssm.py:156-158](../flowquake/ssm.py#L156-L158) initializes
   `A_log = log(Uniform(1, 16))` — plain random scalar decays per head, no HiPPO
   anywhere in the repo. **Why it matters:** the memory horizon here comes from
   `Delta` selectivity and the spread of `A_h` across 6 heads, not from an
   orthogonal-polynomial projection, and you should not claim otherwise.

9. **People think** the encoder gives FlowQuake its whole-catalog reach.
   **Actually** it is switched off in 71 of 74 committed runs (§15.1); the
   spatial head's reach comes from big-trigger mixture components and the
   temporal head's receptive field is exactly 64 events. **Why it matters:** this
   is the single most attackable sentence in the repo's public documentation.

10. **People think** mixed-precision autocast is fine here. **Actually** the
    decays are exponentials of cumulative sums, which overflow *and* underflow
    fp16 within a chunk; [ssm.py:52](../flowquake/ssm.py#L52) casts to fp32
    explicitly for exactly this reason. **Why it matters:** the failure mode is a
    silent NaN in the loss thousands of steps in, not an error at line 52.

---

## Questions a professor will ask

**1. Derive state-space duality.**
Unroll `H_t = a_t H_{t-1} + Delta_t B_t x_t^T` to get
`H_t = (prod_{r<=t} a_r) H_0 + sum_{s<=t} L(t,s) Delta_s B_s x_s^T` with
`L(t,s) = prod_{r=s+1..t} a_r`. Contract with `C_t^T`; since `B_s x_s^T` is rank
one, `C_t^T B_s x_s^T = (C_t·B_s) x_s^T`. So
`y_t = sum_{s<=t} L(t,s)(C_t·B_s) Delta_s x_s` — a lower-triangular matrix
`M[t,s] = L(t,s)(C_t·B_s)` applied to `diag(Delta)X`. That is masked linear
attention with `C` as query, `B` as key, `Delta x` as value, and a decaying
instead of binary mask. Full derivation in §9.3.

**2. Why does selectivity kill the FFT?**
Because `y_t = sum_s K(t,s) u_s` with
`K(t,s) = C_t (prod_{r=s+1..t} A_bar_r) B_bar_s`, which depends on `t` and `s`
separately, not on `t - s`. Non-Toeplitz ⇒ not a convolution ⇒ no convolution
theorem ⇒ no FFT. You are left with a scan or a quadratic form.

**3. Derive ZOH.**
Hold `u` constant over one step; apply the variation-of-constants solution over
`[kΔ, (k+1)Δ]`; substitute `r = (k+1)Δ - s`; evaluate
`int_0^Δ e^{Ar} dr = A^{-1}(e^{AΔ} - I)` termwise from the exponential series.
Gives `A_bar = exp(ΔA)`, `B_bar = (ΔA)^{-1}(exp(ΔA) - I) ΔB`. §6.1.

**4. Is your discretization exact?**
The state transition is exact ZOH. The input map is the **Euler** approximation
`B_bar = Δ B`, not the exact `((1 - e^{-ΔA})/A) B`; they differ by the scalar
`(1 - e^{-ΔA})/(ΔA) in (0,1]`, which the learnable `B_t` and the downstream
gated RMSNorm absorb. Mamba-2 does the same. §6.4.

**5. Do the vanishing-gradient argument properly.**
`dh_T/dh_t = prod_{k} D_k W`. Linear case: `= W^{T-t}`, and by Gelfand
`||W^n||^{1/n} -> rho(W)`, so the scale is `rho^{T-t}` — vanish below 1, explode
above 1, knife edge at 1. Nonlinear: `||D_k W|| <= gamma·sigma_1(W)`, so
`gamma·sigma_1(W) < 1` is sufficient for vanishing and
`gamma·sigma_1(W) > 1` necessary for exploding (Pascanu et al. 2013). §2.3.

**6. Why do gates help, and is `a_t = exp(-Delta_t A_h)` a gate?**
Gates make the decay elementwise (per channel, so many horizons coexist),
input-dependent (content-based forgetting), and additive (no accumulated
rotation). And yes: with `A = -1` and exact ZOH, `h_t = (1-g_t)h_{t-1} + g_t u_t`
with `g_t = 1 - exp(-Delta_t)`, which is the classical gated update exactly. §8.3.

**7. Prove the scan operator is associative.**
`(g1,u1)⊗(g2,u2) := (g1g2, g2u1+u2)`. Both bracketings give
`(g1g2g3, g2g3u1 + g3u2 + u3)`. §11.1. This is what licenses any parallel-scan
implementation.

**8. Why chunking rather than a Blelloch scan?**
Because the intra-chunk block is a batched matmul that runs on tensor cores,
while a scan is elementwise and bandwidth-bound — roughly an order of magnitude
in effective FLOPs on modern GPUs. Blelloch wins when depth `O(L/Q)` dominates,
when you cannot afford `O(LQ)` memory, or when `Q` must be small. §11.3.

**9. Give me the complexity of `selective_scan_chunked` and explain `Q`.**
`TIME = O(L Q (N+P) + L N P)`, `MEMORY = O(LQ + LNP/Q)`, `DEPTH = O(L/Q)`.
The two time terms balance at `Q ≈ N ≈ P`. Memory grows linearly in `Q` (the
`(B, nc, H, Q, Q)` decay tensors — 25 MB each at FlowQuake's training shape, and
`segsum_decay` transiently makes three of them), while sequential depth falls as
`1/Q`. §10.3.

**10. Why fp32, and why is the mask applied before the `exp`?**
Because `cs` spans `-102` over a chunk at the module's own init ranges:
`exp(-102.4) = 3.4e-45` (underflow) and `exp(+102.4) = 2.96e44` (overflows fp32's
`3.4e38` to `inf`). Multiplying an `inf` by a `0` mask gives `NaN`. Masking to
`-inf` *before* exponentiating gives exact zeros with no intermediate infinity.
Every other `exp` in the function is provably of a non-positive argument. §13.

**11. Why `L = 200` with `chunk = 64` in the test?**
`200 = 3·64 + 8`, so `pad = 56` and the final chunk is mostly padding. It
exercises the padding branch, a partial final chunk, and the claim that padded
steps (`dt = 0` ⇒ `a = 1`, write `= 0`) are exact state no-ops so the *returned
final state* is `H_200` and not `H_256`. §14.1.

**12. Why is the reference implementation in fp64?**
So the reference is not the thing under test. With both in fp32 you could not
tell an algorithmic error from accumulated rounding; with the reference in fp64,
the `1e-4` tolerance is measuring the fast path's fp32 error against ground
truth.

--- *hostile from here* ---

**13. (hostile) Your README's first line calls this a "Selective-SSM
(Mamba-style) whole-catalog encoder". Show me one production run where the
encoder object exists.**
There isn't one. [model.py:82-83](../flowquake/model.py#L82-L83) sets
`self.encoder = None` when `h_bottleneck = 0`, and 71 of the 74 committed run
configs leave it at 0 (69 explicitly, 2 by omission) — including all three
ComCat seeds behind the headline
temporal number. The three exceptions are the §4.3 ablations. The framing in
README.md's opening sentence and WORKING.md's claim 1 overstates the encoder's
role; the accurate framing is that FlowQuake is a structured, observation-anchored
marked TPP and the SSM encoder is the ablation arm establishing why the structure
is necessary. That is a documentation defect, and the fix is a rewritten abstract,
not a rerun. §15.1, §15.4.

**14. (hostile) You claim to fix fixed-window encoders. What is the receptive
field of your production temporal head?**
Exactly 64 events. With `h = 0`, `head_t` sees only
`tokens[:, SAFE_TOKEN_DIMS]` — 30 dims whose largest lag is
`RECENCY_LAGS[-1] = 64` — and the `lastk` tensor with its 730-day big triggers
goes only to the *spatial* head. About 70 days at ComCat's rate. It is a bigger
and better-featurized window than DeepSTPP's 20 events, but it is a fixed window,
and the mechanism story attached to the temporal win in WORKING.md is not the
mechanism that produced it. The correct story is: an exact-likelihood flow over
`log tau` conditioned on multi-scale relational statistics beats a parametric
Omori intensity at next-gap prediction. The fix I would run: feed the `lastk`
summary (or the big-trigger block) into `cond`, and re-measure. §15.3.

**15. (hostile) Your §4.3 concludes that flexibility causes memorization. But
what is the treatment, exactly?**
The `h > 0` arm changes three things at once: it adds an SSM encoder, a learned
projection, and — decisively — a channel that has seen **absolute `x, y`**
(`enc_in = tokens`, the full 32-dim token,
[model.py:165](../flowquake/model.py#L165)), whereas `SAFE_TOKEN_DIMS` excludes dims
1 and 2. The signature of the collapse is spatial (train `sll` rises to `-7.27`
at `h = 4`, mass pinned on training epicentres), which points at the coordinate
channel, not at the SSM or at flexibility per se. The experiment therefore
supports "any learned global channel carrying absolute geography lets the heads
memorize" — which is exactly what MANUSCRIPT §4.3's own sentence claims — but not
a claim about state-space models. The missing control is `h > 0` with the encoder
restricted to `SAFE_TOKEN_DIMS`; it does not exist in `runs/`, and it is the
first experiment I would run. §15.4.

**16. (hostile) Your scan has no gradient test. Convince me the backward is
right.**
I can't, from the tests. All five tests in
[tests/test_ssm.py](../tests/test_ssm.py) are forward-only; a transposed einsum in
the vjp or a detached tensor in the chunk-state list would pass every one of them
and silently train a different model. The mitigations are weak (autograd derives
the backward from the forward, so the failure mode requires an autograd bug or a
`detach`, not a hand-written kernel) and the fix is cheap: a
`torch.autograd.gradcheck` on `selective_scan_chunked` in fp64 at small
`(B,L,H,P,N)`, plus a comparison of `d loss/d params` between chunked and
reference. That is a genuine hole and it should be closed before submission.

**17. (hostile) Show me a test that would fail if someone replaced the `-inf`
mask with a multiplicative `{0,1}` mask.**
There is no test that *targets* it. The scan-level generator uses `dt <= 0.101`
and `A <= 10.5` ([test_ssm.py:14-15](../tests/test_ssm.py#L14-L15)), so the worst
`Q = 64` chunk span is `-67.9` and `exp(+67.9) = 3.0e29` — finite in fp32, so
`inf * 0 = NaN` never triggers in either scan test. At the module's own init
ranges (`A_max = 16`, `dt_max = 0.1`) the span is `-102.4` and
`exp(+102.4) = 2.96e44` overflows. The one place the suite might catch it by
accident is `test_block_causality`, which adds `+100.0` to the inputs and so
pushes `softplus`'d `dt` to `O(10-100)`; I did not execute it (`torch` is not
installed here) so I will not claim more than "plausibly". Either way the suite
does not cover the regime the module itself initializes into by design. The fix
is a new test with `A = 16, dt = 0.1, Q = 64` asserting `torch.isfinite`. §14.2.

**18. (hostile) Why is a Mamba-2 scan the right inductive bias for a
self-exciting point process at all?**
Partly it isn't, and that is worth conceding. The state decays *geometrically in
event index* — `exp(-sum Delta_r A_h)` — while Omori says influence decays as a
*power law in physical time*. A mixture of heads with different `A_h` can
approximate a power law as a sum of exponentials (a standard and old technique),
but it is an approximation that ETAS gets exactly for free, and the SSM's clock
is the wrong one: it counts events, not days. The defensible claims are narrower
and I would make only these: (i) selectivity is the right *shape* of mechanism,
because a self-exciting process's per-event informativeness varies by orders of
magnitude with magnitude (`e^{a(m-m_c)}`, a factor of ~`10^4` between M2.5 and
M7.1); (ii) the constant-size state is what makes 10^4-lane catalog simulation
feasible where a KV cache would need terabytes (§12.2). The single most obvious
improvement is to set `Delta_t` from the observed physical gap `tau_t` rather
than from a learned linear map, which turns the head into an exponential-kernel
Hawkes memory over real time. §8.5.

**19. What is the memory of an SSM state vs a KV cache, in numbers, at your
simulation budget?**
At `d_model = 96`, 4 layers, `H = 6`, `N = P = 32`: SSM state `4·6·32·32 =
24,576` floats `= 98.3 KB` per lane, constant. KV cache at `L = 70,000`:
`2·4·70,000·96 = 53.8 M` floats `= 215 MB` per lane. At the default
`--n-sims 10000` ([csep_forecast.py:101](../flowquake/csep_forecast.py#L101)) that is
983 MB versus 2.15 TB. §12.2.

**20. What does HiPPO buy, and does this code use it?**
HiPPO chooses `A, B` so the state is the coefficient vector of the best
degree-`(N-1)` polynomial approximation to the history under a chosen measure —
for LegS, timescale-invariantly over all of `[0,t]`. Empirically it is the
difference between chance and SOTA on Long Range Arena (S4, 2022). This code does
**not** use it: `A_log = log(Uniform(1,16))`
([ssm.py:156-158](../flowquake/ssm.py#L156-L158)), plain random scalar decays. That
is consistent with Mamba-2, whose scalar-per-head `A` cannot carry a polynomial
basis anyway, and whose position is that selectivity replaces structure — but it
means no HiPPO guarantee applies here. §7.2.

---

## Further reading

1. **Hochreiter & Schmidhuber (1997), *Neural Computation*, "Long short-term
   memory."** The constant-error-carousel argument. Read it for the gradient
   analysis, not the architecture — the analysis is the part that transfers to
   §8.3.
2. **Pascanu, Mikolov & Bengio (2013), ICML, "On the difficulty of training
   recurrent neural networks."** The rigorous vanishing/exploding conditions
   (§2.3) and gradient clipping. This is the paper to cite when you make the
   spectral-radius claim.
3. **Vaswani et al. (2017), NeurIPS, "Attention is all you need."** For the
   baseline everything is measured against, and for the `L^2` cost you must be
   able to state precisely.
4. **Katharopoulos, Vyas, Pappas & Fleuret (2020), ICML, "Transformers are RNNs:
   fast autoregressive transformers with linear attention."** The reassociation
   of §4.2 and the RNN form of §4.3, in four pages.
5. **Gu, Dao, Ermon, Rudra & Ré (2020), NeurIPS, "HiPPO: recurrent memory with
   optimal polynomial projections."** Why `A`'s initialization is a modelling
   decision rather than a hyperparameter.
6. **Gu, Goel & Ré (2022), ICLR, "Efficiently modeling long sequences with
   structured state spaces" (S4).** The DPLR kernel algorithm sketched in §7.3.
   Read the appendix only if you must derive it; the main text is enough to
   explain what problem it solves.
   Its sequel — Gu, Goel, Gupta & Ré (2022), NeurIPS, "On the parameterization
   and initialization of diagonal state space models" (S4D) — establishes that a
   diagonal `A` is enough, the result that licenses Mamba's and Mamba-2's
   parameterizations.
7. **Gu & Dao, "Mamba: linear-time sequence modeling with selective state
   spaces," arXiv:2312.00752 (2023), published at COLM 2024.** Selectivity, the
   gating connection (§8.3), and the hardware-aware scan.
   [MANUSCRIPT.md line 943](../MANUSCRIPT.md) cites only the arXiv id; adding the
   COLM venue is a free improvement.
8. **Dao & Gu (2024), ICML, "Transformers are SSMs: generalized models and
   efficient algorithms through structured state space duality" (Mamba-2).** The
   direct source for §9 and §10. If you read one paper from this list, read this
   one — [flowquake/ssm.py](../flowquake/ssm.py) is an implementation of its
   Algorithm 1.
9. **Blelloch (1990), CMU tech report, "Prefix sums and their applications."**
   The work-efficient scan of §11.2, still the clearest statement of work vs
   depth. Smith, Warrington & Linderman (2023, ICLR, "Simplified state space
   layers for sequence modeling", S5) is the deep-learning application: parallel
   scan instead of FFT, and a good contrast with the chunked approach.
10. **Dascher-Cousineau, Shchur, Brodsky & Günnemann (2023), *GRL*, RECAST**
    (doi:10.1029/2023GL103909, as cited in [MANUSCRIPT.md](../MANUSCRIPT.md)). A
    recurrent neural point process applied to aftershock forecasting — the
    closest published thing to "use a sequence model on a catalog", and useful
    for calibrating what the sequence-modelling literature has actually
    delivered in seismology.
