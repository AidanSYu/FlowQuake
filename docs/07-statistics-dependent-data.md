# Statistics for dependent data: bootstraps, families, and equivalence

Every FlowQuake result is a statement about a **paired per-event gain series** — one number per test earthquake, "how many more nats did FlowQuake assign to what actually happened than ETAS did" — and every headline claim is a claim about that series' mean. This chapter is the machinery between "here are 21,889 numbers" and "FlowQuake beats ETAS, p < 0.05", plus every place that machinery can be attacked.

The short version: those 21,889 numbers are nowhere near 21,889 independent observations. An aftershock sequence is, statistically, close to one observation repeated a few thousand times. Divide by `sqrt(n)` and you are wrong by a factor of three — enough to turn a tie into a win.

## What this chapter buys you

- **Derive** `Var(mean)` under autocorrelation, define `n_eff`, and say with a number how badly an i.i.d. SE lies on FlowQuake's own catalogs. (A nominal 95% i.i.d. interval has real coverage of 39–65% depending on catalog; §2.3, computed from `runs/replacement_readiness.json`.)
- **Prove** the i.i.d. bootstrap SE for a mean is *exactly* the naive SE — not a partial fix, no fix — and explain what each member of the block-bootstrap family repairs.
- **Prove** the stationary bootstrap's resample is stationary (a Markov-chain argument) and that fixed-length blocks are only *periodically* stationary — the reason Politis & Romano 1994 exists — then use the same chain to **derive** what its variance estimates: a geometric lag window on the long-run variance, which for AR(1) is the exact closed form `VIF_SB(L) = (1+q)/(1-q)`, `q = phi(1-1/L)` (§4.5, §5.4).
- **Prove** Holm controls FWER under arbitrary dependence, reproduce all six stored `p_holm` values by hand, and defend a specific answer to "what is the family?", including the sensitivity analysis the repo does not run (§9.3: pooling flips exactly one of twelve verdicts).
- **Derive** TOST from the intersection–union principle — including why 90% and not 95% — and assess whether the repo's 0.05/0.10 nats/event margins hold up. (Partly; §10.4 has the criticism.)
- **Derive** McNemar, compute its power, and explain why `McNemar exact p = 1.00` in [MANUSCRIPT.md](../MANUSCRIPT.md) §4.2 is *not* evidence of equivalence — 38% power against an 80/20 alternative on 10 discordant days.
- **Explain** why FlowQuake's spatial head wins on the mean while winning only 47.85% of events, and what that shape implies.

## Prerequisites

- Chapter 1, point processes and the likelihood — [01-point-processes.md](01-point-processes.md). You need `lambda(t | H_t)`, the compensator `Lambda(t)`, the factorization `f(tau, x, y, m | H) = f_t(tau|H) f_s(x,y|H) f_m(m|H)`, and `tll` / `sll` / `mll` / `nll`. [STACK.md](../STACK.md) Part I §1–3 is the shorter authoritative version of the same material.
- Chapter 3, ETAS — [03-etas.md](03-etas.md), or [STACK.md](../STACK.md) Part II. You need only that ETAS is the comparison model and is fitted per region.
- Undergraduate probability: expectation, covariance, CLT, binomial, what a p-value is. No seismology beyond "aftershocks cluster".

Notation is the shared convention. Two disambiguations: ETAS's Omori taper timescale (called `tau` in the repo's code) is written **`tau_tap`** here because `tau` is the inter-event gap; and `L` always means a bootstrap **block length**, never a likelihood.

---

## 1. The object under study

Fix a region and a test window. Both models score every test event. For the `i`-th event define `g_i = score_FlowQuake(i) - score_ETAS(i)`, where `score` is `tll`, `sll`, or `tll + sll`. The repo calls the three series `dT`, `dS`, `dTot`, in nats/event.

Three load-bearing properties:

1. **Paired.** Both models see the same event, history and window, so whatever is hard about an event is hard for both. Pairing removes enormous common variance; the unpaired comparison of two mean `tll`s would give far wider intervals on the same data. This is why [flowquake/stats.py](../flowquake/stats.py) works on one series of differences, not two series of scores.
2. **Serially ordered.** `g_1, g_2, ...` are in event-time order — which is what makes every block method here applicable, and what makes i.i.d. false.
3. **`dTot = dT + dS` exactly**, event by event. Three metrics, two degrees of freedom. This matters in §9.

The hypothesis is always `H0: E[g] = 0`, two-sided. Because log-likelihood is a strictly proper score (Gneiting & Raftery 2007, *JASA*), `E[g] > 0` means FlowQuake's predictive distribution is genuinely closer to the truth, not merely different.

Note that `g_i` is a **forecast loss differential**, the object econometrics has studied since Diebold & Mariano (1995, *JBES*). Their point is ours: loss differentials are serially correlated and need a variance estimator that says so. FlowQuake uses a block bootstrap where Diebold–Mariano uses a Newey–West HAC estimator; §4.5 shows these are two roads to the same destination.

---

## 2. Why `n` is a lie

### 2.1 The derivation

Let `X_1..X_n` be weakly stationary: `E[X_i] = m`, `Var(X_i) = sigma^2`, `Cov(X_i, X_j) = gamma_{|i-j|}`. Write `rho_k = gamma_k / sigma^2`, so `rho_0 = 1`. For `Xbar = (1/n) sum X_i`:

```
Var(Xbar) = (1/n^2) sum_{i=1}^n sum_{j=1}^n Cov(X_i, X_j)
          = (1/n^2) sum_i sum_j gamma_{|i-j|}
```

Count pairs at each lag: `k = 0` occurs `n` times, each `k >= 1` occurs `2(n-k)` times (once with `i > j`, once with `i < j`). So

```
Var(Xbar) = (1/n^2) [ n*gamma_0 + 2 sum_{k=1}^{n-1} (n-k) gamma_k ]
          = (sigma^2 / n) [ 1 + 2 sum_{k=1}^{n-1} (1 - k/n) rho_k ]
```

Define the **variance inflation factor** and the **effective sample size**:

```
VIF_n = 1 + 2 sum_{k=1}^{n-1} (1 - k/n) rho_k        n_eff = n / VIF_n        Var(Xbar) = sigma^2 / n_eff
```

`n_eff` is the number of independent observations giving the same precision. If the `rho_k` are absolutely summable and `n` is large, the triangular weight tends to 1 for each fixed `k` and `VIF -> 1 + 2 sum_{k>=1} rho_k`. Then `sigma^2 * VIF = sum_{k=-inf}^{inf} gamma_k` is exactly the **long-run variance** `sigma_LR^2`, i.e. `2*pi` times the spectral density at zero. Every method in §4–§5 is, underneath, an estimator of `sigma_LR^2`. Say that in a viva and you have shown you know what a block bootstrap is doing.

Two facts a professor will check: nothing requires `rho_k > 0` — negative autocorrelation gives `VIF < 1` and `n_eff > n`, making the i.i.d. SE conservative (earthquake gains are in the bad regime, but the formula is agnostic); and `VIF_n >= 0` always, because `n*Var(Xbar) >= 0`, so the `(1-k/n)` taper is not decoration, it is what keeps the quantity a variance.

### 2.2 Two closed forms

**AR(1).** `X_i - m = phi(X_{i-1} - m) + e_i`, `|phi| < 1`, gives `rho_k = phi^k` and

```
VIF = 1 + 2 sum_{k>=1} phi^k = 1 + 2phi/(1-phi) = (1 + phi)/(1 - phi)
```

Invert to interpret any observed VIF: `phi = (VIF - 1)/(VIF + 1)`.

**Block-constant (the aftershock caricature).** `K` sequences of `B` events, gain identical within a sequence, sequences i.i.d., `n = KB`. Directly, `Xbar = (1/K) sum_j Y_j` with `Y_j` the sequence value, so `Var(Xbar) = sigma^2/K = sigma^2 B/n`, i.e. **`VIF = B`, `n_eff = K`**. Cross-check via the autocorrelation formula: randomising the block phase makes the process stationary, and `X_i`, `X_{i+k}` share a block with probability `(1-k/B)_+`, in which case they are equal — so `rho_k = (1-k/B)_+` and

```
sum_{k=1}^{B-1} rho_k = sum_{k=1}^{B-1} (1 - k/B) = (B-1) - (1/B)(B-1)B/2 = (B-1)/2
=>   VIF = 1 + 2 sum_{k>=1} rho_k = 1 + 2(B-1)/2 = B
```

Two independent derivations, same answer — with one honest caveat: the first is exact for a window aligned to block boundaries (`n = KB`), the second uses the large-`n` form `VIF = 1 + 2 sum_{k>=1} rho_k` rather than the finite-`n` triangular-weighted `VIF_n`. They agree to `O(B/n)`, which is all anyone needs when `B` is tens and `n` is tens of thousands. This gives the second reading of any VIF: **"my `n` events behave as if they came in identical clumps of size VIF."**

### 2.3 How badly the i.i.d. SE lies — from FlowQuake's own artifacts

`runs/replacement_readiness.json` stores, per California catalog, the `dT` mean, the plain i.i.d. `stderr` (computed at [stats.py:176](../flowquake/stats.py#L176)), and the stationary-block-bootstrap 95% CI. Since `half-width_boot / (1.96 * SE_iid) = sqrt(VIF)`, the two together give an empirical VIF:

| catalog | n | mean dT | SE_iid | i.i.d. 95% CI | bootstrap 95% CI | ratio | VIF | n_eff | equiv. phi | true coverage of the i.i.d. CI |
|---|---|---|---|---|---|---|---|---|---|---|
| ComCat_25 | 21889 | 0.053296 | 0.002424 | [0.04854, 0.05805] | [0.03969, 0.06798] | 2.98 | 8.86 | 2469 | 0.797 | 49.0% |
| WHITE_06 | 24080 | 0.046559 | 0.002458 | [0.04174, 0.05138] | [0.03046, 0.06462] | 3.55 | 12.57 | 1915 | 0.853 | 42.0% |
| SanJac_10 | 4399 | 0.029182 | 0.005438 | [0.01852, 0.03984] | [-0.00569, 0.07593] | 3.83 | 14.66 | 300 | 0.872 | 39.1% |
| SaltonSea_10 | 4103 | 0.104031 | 0.008982 | [0.08643, 0.12164] | [0.06967, 0.14436] | 2.12 | 4.50 | 912 | 0.636 | 64.5% |
| SCEDC_20 | 13062 | 0.078121 | 0.003606 | [0.07105, 0.08519] | [0.06028, 0.09728] | 2.62 | 6.85 | 1906 | 0.745 | 54.6% |

Source: `runs/replacement_readiness.json` → `checks[california_block_bootstrap_temporal].evidence.<catalog>.{n, mean, stderr, ci}`. `n`, `mean`, `stderr` and `ci` are read verbatim; the last five columns are computed from them. "True coverage" is `2*Phi(1.96/ratio) - 1`, i.e. the real probability that the nominal-95% i.i.d. interval covers the truth, taking the bootstrap interval as correct.

Read the SanJac_10 row carefully. **The i.i.d. interval `[0.0185, 0.0398]` is entirely positive; the block-bootstrap interval `[-0.0057, 0.0759]` contains zero.** Win flips to tie purely on the variance estimator, and the repo records `"decision": "tie"` for exactly this reason. That row is the single strongest argument in this repository for why any of this matters.

Calibration on the AR(1) scale:

| phi | VIF | SE understated by | n_eff/n | real coverage of a nominal 95% i.i.d. CI |
|---|---|---|---|---|
| 0.3 | 1.86 | 1.36x | 53.9% | 85.0% |
| 0.5 | 3.00 | 1.73x | 33.3% | 74.2% |
| 0.7 | 5.67 | 2.38x | 17.6% | 59.0% |
| 0.8 | 9.00 | 3.00x | 11.1% | 48.6% |
| 0.9 | 19.00 | 4.36x | 5.3% | 34.7% |
| 0.95 | 39.00 | 6.24x | 2.6% | 24.6% |

`phi = 0.5` — stickiness you would call "mild" from the plot — already inflates the SE by 73% and drops a 95% interval to 74% coverage.

---

## 3. The bootstrap: plug-in, and why it fails on dependent data

### 3.1 The plug-in principle

Efron (1979, *Annals of Statistics*). You want the law of `theta_hat - theta = T(F_hat_n) - T(F)` under the unknown `F`. You cannot have it. The bootstrap substitutes:

```
Law_F [ T(F_hat_n) - T(F) ]     approximated by     Law_{F_hat_n} [ T(F_hat_n*) - T(F_hat_n) ]
   (unknown; what you want)                    (computable: resample from F_hat_n, repeat B times)
```

On the right you know the generating distribution — it is `F_hat_n`, sitting in memory — so you can draw as many replicates as you like. Consistency needs (1) `F_hat_n -> F` in a metric strong enough for the plug-in to converge, and (2) `T` smooth at `F` (Hadamard differentiability). Condition 2 is where the bootstrap famously breaks: the sample maximum (Bickel & Freedman 1981, *Annals of Statistics*), the mean under infinite variance (Athreya 1987, *Annals of Statistics*), parameters on a boundary. A finite-variance mean is as smooth as it gets, so smoothness is not our problem. **Condition 1 is**, fatally.

### 3.2 The i.i.d. bootstrap SE is *exactly* the naive SE

Draw `n` indices uniformly with replacement, so `X*_1..X*_n` are conditionally i.i.d. from the empirical *marginal*. Then

```
E*[X*_i]    = (1/n) sum_j X_j = Xbar
Var*(X*_i)  = (1/n) sum_j (X_j - Xbar)^2 =: s_n^2
X*_i conditionally independent

=>  Var*(Xbar*) = (1/n^2) * n * s_n^2 = s_n^2 / n
```

That is an **identity**, not an approximation. The i.i.d. bootstrap standard error of a mean *is* `s_n/sqrt(n)` — the naive SE, up to `n` vs `n-1`. It does not partly correct for dependence; it reproduces the number you were trying to escape at a thousand times the cost.

The reason: `F_hat_n` here is the empirical measure of the *marginal*. Resampling from it discards the joint law of `(X_i, X_{i+1}, ...)` entirely. Condition 1 fails not through slow convergence but because the plug-in estimates the wrong object — the marginal, when the parameter of interest `sigma_LR^2 = sum_k gamma_k` lives in the joint. **The fix must preserve local joint structure: resample segments, not points.**

---

## 4. The block bootstrap family

All four members cut the series into segments, draw segments with replacement, concatenate to length `n`, recompute, repeat `B` times. They differ in how segments are chosen, and each difference repairs a named defect.

### 4.1 Non-overlapping blocks (Carlstein 1986)

Carlstein (1986, *Annals of Statistics*). Split into `K = n/L` disjoint consecutive blocks; draw `K` of them i.i.d. uniformly; concatenate. **Fixes:** within-block dependence is preserved. **Exact property:** `E*[Xbar*] = Xbar` when `L | n`. **Defect:** only `K = n/L` distinct blocks exist — with `n = 4399` and `L = 50` that is 87 — so the *variance of the variance estimate* is high.

### 4.2 Moving blocks (Künsch 1989)

Künsch (1989, *Annals of Statistics*, "The jackknife and the bootstrap for general stationary observations"); independently Liu & Singh (1992). Use every window `B_i = (X_i..X_{i+L-1})`, `i = 1..n-L+1`. **Fixes:** `n-L+1` distinct blocks instead of `n/L`, dramatically reducing the variance of the variance estimate. MBB is the standard workhorse.

**New defect — an edge bias.** Observation `X_t` appears in `min(t, L, n-L+1, n-t+1)` of the windows: interior points in `L` of them, `X_1` in exactly 1, `X_2` in 2, and so on at both ends. So

```
E*[X*_j] = sum_t w_t X_t     with     w_t = min(t, L, n-L+1, n-t+1) / ((n-L+1) L)
```

and the `w_t` are not uniform — the first and last `L-1` observations are underweighted, so `E*[Xbar*] != Xbar`. The bias is `O(L/n)`, small but real: the bootstrap distribution is not centred on the statistic you are reporting, which contaminates percentile intervals.

### 4.3 Circular blocks (Politis & Romano 1992)

Politis & Romano (1992), "A circular block-resampling procedure for stationary data", in LePage & Billard (eds), *Exploring the Limits of Bootstrap*, Wiley. Wrap the series onto a circle (`X_{n+i} := X_i`). Now there are exactly `n` blocks, one per start, and **every observation appears in exactly `L` of them** — so `w_t = 1/n` and `E*[Xbar*] = Xbar` exactly. The MBB edge bias is gone.

**Cost:** you manufacture a spurious dependence at the seam between `X_n` and `X_1`. Irrelevant for a mean (one join in `n/L`); not irrelevant for a statistic sensitive to trend, and a series with drift becomes discontinuous there. **Remaining defect:** the resample is still not stationary.

### 4.4 The stationary bootstrap (Politis & Romano 1994), with proof

Politis & Romano (1994, *JASA*, "The stationary bootstrap"). Same circular wrap, but block lengths are **i.i.d. Geometric(p)** with mean `L = 1/p`. As implemented at [stats.py:66-79](../flowquake/stats.py#L66-L79):

```
p_new = 1 / L
pos = 0
while pos < n:
    start  ~ Uniform{0..n-1}
    length ~ Geometric(p_new)              # support {1,2,...}, mean 1/p_new = L
    take   = min(length, n - pos)          # truncate the last block to fill exactly n
    idx    = (start + 0..take-1) mod n     # circular wrap
    append X[idx];  pos += take
```

**Theorem.** Conditionally on the data, `X*_1..X*_n` is *stationary*: the joint law of `(X*_j, ..., X*_{j+k})` does not depend on `j`.

**Proof.** Write `X*_j = X_{K_j}`, `K_j` the circular index position `j` was drawn from. Because the geometric is **memoryless**, the block construction is equivalent to a one-step construction: `K_1 ~ Uniform{1..n}`; given `K_j`, with probability `1-p` continue (`K_{j+1} = K_j + 1 mod n`), with probability `p` start fresh (`K_{j+1} ~ Uniform{1..n}`, independent). (Equivalence is exactly memorylessness: given a Geometric block has run `r` steps, it ends next step with probability `p`, independent of `r`.) So `(K_j)` is a Markov chain on `Z_n` with kernel

```
P(K_{j+1} = k' | K_j = k) = (1-p) 1{k' = k+1 mod n} + p/n
```

The uniform `pi` on `Z_n` is invariant: if `K_j ~ pi` then `K_j + 1 mod n ~ pi` (shifting a uniform on a cyclic group is uniform), and a mixture of two uniforms is uniform. Since `K_1 ~ pi`, the chain starts in its invariant law and is a stationary Markov chain. Any measurable function of a stationary process is stationary, and `X*_j = X_{K_j}` is one. **QED.**

Both ingredients are visible in the code: memorylessness (`rng.geometric`) gives the Markov chain; the circular wrap (`% n`) makes the shift preserve uniformity — without it, `K_j = n` would have nowhere to go and uniformity would break at the boundary.

**Why fixed lengths fail.** Take CBB with fixed `L` and let `r = ((j-1) mod L) + 1` be `j`'s position inside its block. If `r < L`, positions `j` and `j+1` are in the same block, so `K_{j+1} = K_j + 1` with probability 1 and `Cov*(X*_j, X*_{j+1}) = gamma_hat_1`. If `r = L`, position `j+1` starts a freshly drawn block, so `X*_j` and `X*_{j+1}` are conditionally independent and `Cov*(X*_j, X*_{j+1}) = 0`. The covariance depends on `j mod L`: the resample is **periodically stationary with period `L`, not stationary**. Geometric lengths smear the boundary uniformly — "am I at a boundary?" has probability `p` at every `j` — which is exactly what restores stationarity.

**Does it matter for a mean?** Not much: CBB and SB variance estimators are both consistent for `sigma_LR^2`. Honestly, the SB's stationarity buys cleaner theory, validity for statistics depending on the joint law beyond the mean, and freedom from `L | n`. Lahiri (1999, *Annals of Statistics*, "Theoretical comparisons of block bootstrap methods") found the SB to have the *largest* asymptotic variance of the block bootstraps — but Nordman (2009, *Annals of Statistics*, "A note on the stationary bootstrap's variance") corrected that calculation and showed the SB's variance in fact matches the *non-overlapping* block bootstrap's, so the SB is worse than MBB/CBB but not as much worse as first reported. If a professor raises this, concede the ranking and know the correction: the SB is the safe easy-to-implement choice, not the minimum-variance one.

### 4.5 What the stationary bootstrap actually estimates — derived

The §4.4 Markov chain gives the conditional variance in two lines. Write `xbar` for the sample mean, `ghat_c(k) = (1/n) sum_{i=1}^n (X_i - xbar)(X_{i+k mod n} - xbar)` for the **circular** sample autocovariance. Since `K_j ~ Uniform`, `E*[X*_j] = xbar`. For the covariance at lag `k`, the chain either ran `k` steps without a restart — probability `(1-p)^k`, and then `K_{j+k} = K_j + k mod n` — or it restarted at least once, after which `K_{j+k}` is uniform and independent of `K_j`, contributing 0. Hence

```
Cov*(X*_j, X*_{j+k}) = (1-p)^k * ghat_c(k)
```

and plugging that into the §2.1 pair-counting identity (which needs only stationarity, and §4.4 proved the resample is stationary):

```
n * Var*(Xbar*) = ghat_c(0) + 2 sum_{k=1}^{n-1} ((n-k)/n) (1-p)^k ghat_c(k)          (*)
```

**QED** — no citation needed. (Politis & Romano 1994 state the same result in terms of the *non-circular* estimator `ghat(k) = (1/n) sum_{i=1}^{n-k}(X_i - xbar)(X_{i+k} - xbar)`, where it reads `n Var* = ghat(0) + 2 sum_k b_k ghat(k)` with `b_k = ((n-k)/n)(1-p)^k + (k/n)(1-p)^{n-k}`. The two are algebraically identical, because `ghat_c(k) = ghat(k) + ghat(n-k)`; the extra term in `b_k` is exactly the wrap-around half. Both forms were checked against a 10^5-replicate Monte Carlo of the repo's own `_bootstrap_means` on an AR(1) series and agree to Monte Carlo error at `L = 1, 5, 20, 50`.) For `n >> L` the taper `(n-k)/n` is 1 over the range where `(1-p)^k` is non-negligible, so:

> The stationary bootstrap variance is a **lag-window estimator of the long-run variance with geometric weights `(1 - 1/L)^k`.**

Three consequences follow immediately. It is a cousin of the Newey–West / Bartlett HAC estimator (Newey & West 1987, *Econometrica*), which uses triangular weights `(1-k/L)_+` for the same job — so a Diebold–Mariano test with a Newey–West variance and a stationary-bootstrap CI on the same series should agree to within Monte Carlo noise, and if they do not something is wrong. The weights decay on scale `L`, so autocorrelation at lags `>> L` is effectively truncated: **`L` must be a few multiples of the true decorrelation length or the variance is understated** (that is the bias term in §5.1). And the weights are positive, so the estimator is guaranteed non-negative, which fixed-lag truncation is not.

### 4.6 Does the repo's implementation match the theory?

Checking [flowquake/stats.py:42-81](../flowquake/stats.py#L42-L81) against the requirements:

| requirement | code | verdict |
|---|---|---|
| geometric lengths, mean `L` | `p_new = 1/max(L,1)`; `rng.geometric(p_new)` (numpy: support `{1,2,...}`, mean `1/p`) | correct |
| starts uniform over all `n` | `rng.integers(0, n)` | correct |
| circular wrap | `idx = (start + arange(take)) % n` | correct |
| resample length exactly `n` | `take = min(length, n - pos)`, `while pos < n` | correct |
| mean block 50 | `mean_block: int = 50` at [:45](../flowquake/stats.py#L45), [:87](../flowquake/stats.py#L87), [:116](../flowquake/stats.py#L116), [:145](../flowquake/stats.py#L145), [:164](../flowquake/stats.py#L164); **no `scripts/` call site overrides it** (verified by grep; also [WORKING.md](../WORKING.md) item 16) | correct |
| percentile CI of replicate means | `np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])` | correct, but see §6 |

A textbook stationary bootstrap. Three things a careful reader should still flag:

1. **`_bootstrap_means` ([stats.py:84-110](../flowquake/stats.py#L84-L110)) duplicates the inner loop of `stationary_block_bootstrap_ci` without its degenerate-input guards.** The CI function returns early for `n == 0` and for a constant series; `_bootstrap_means` does not, so `block_bootstrap_pvalue` and `tost_equivalence` would raise on an empty series where the CI function returns `nan`. Not a live bug on any committed artifact, but duplicated code that has already diverged once.
2. **`x = x[np.isfinite(x)]` silently drops non-finite gains and records no count.** Dropping also *splices* the series: previously non-adjacent events become adjacent, and the block bootstrap treats them as correlated neighbours. Harmless if the count is tiny — which is unverifiable, because the count is not stored.
3. **Blocks are counted in events, not time.** Fifty consecutive events during Ridgecrest span minutes; fifty in a quiet interval span months. A fixed event-count block therefore corresponds to a wildly varying decorrelation *time*. A time-indexed block bootstrap (blocks of, say, 30 days) is the natural alternative and nothing in the repo tries it. This choice is never stated or defended — see §5.3 and Q6.

---

## 5. Choosing the block length

### 5.1 The bias–variance tradeoff, derived

Treat the estimator as a lag window of scale `L` on `sigma_LR^2 = sum_k gamma_k`.

**Bias.** The window down-weights lag `k` by `(1-1/L)^k ≈ exp(-k/L)`, so it misses about `sum_k gamma_k (1 - exp(-|k|/L)) ≈ (1/L) sum_k |k| gamma_k`. Write `G = sum_k |k| gamma_k`; then `Bias ≈ -G/L`, order `1/L` — **larger `L`, smaller bias.** At `L = 1` you are back to the i.i.d. bootstrap and `Var* = s_n^2/n` (§3.2).

**Variance.** The estimator averages roughly `n/L` effectively independent blocks, so its own sampling variance grows linearly in `L/n`: `Var ≈ c (L/n) sigma_LR^4`, with `c = 4/3` for MBB and `c = 2` for SB (Politis & White 2004; we do not derive `c`, and Nordman (2009) revises the SB constant — treat it as an order-of-magnitude input, which is all the `n^{1/3}` conclusion needs). The `L -> n` end is a hard degeneracy for the *fixed*-block methods: with CBB at `L = n` every resample is a rotation of the original, every replicate mean equals `Xbar`, and the estimator returns `Var* = 0` — no sampling variability left and nothing but bias. The SB approaches the same wall more gently, because geometric lengths keep some blocks short.

**Optimum.**

```
MSE(L) ≈ G^2/L^2 + c sigma_LR^4 L/n
d/dL:   -2G^2/L^3 + c sigma_LR^4/n = 0
=>      L_opt = ( 2G^2 / (c sigma_LR^4) )^{1/3} * n^{1/3}
```

**The rate is `n^{1/3}` for variance estimation.** (For a two-sided *distribution function* the MSE-optimal rate is `n^{1/5}`, one-sided `n^{1/4}` — Hall, Horowitz & Jing 1995, *Biometrika*; not derived here. The point worth making is that the optimal block length depends on *what you are estimating*.)

### 5.2 Rules of thumb and automatic selection

`n^{1/3}` is the starting point: `21889^{1/3} = 27.97`, `4399^{1/3} = 16.39`, `1121^{1/3} = 10.39`. So `L = 50` is **1.8x** the naive rule at California scale and **4.8x** at Iran scale — same order everywhere, which is reassuring but is not a justification.

The defensible answer is Politis & White (2004, *Econometric Reviews*, "Automatic block-length selection for the dependent bootstrap"), with the correction in Patton, Politis & White (2009, same journal): estimate `G` and `sigma_LR^2` with a flat-top lag window and automatic bandwidth, plug into the formula above. It is implemented as `arch.bootstrap.optimal_block_length` in Python. **FlowQuake does not use it.** The pragmatic alternative referees accept is to **report the sensitivity curve** and show the CI half-width is flat over a broad range of `L`.

### 5.3 The hostile question: `mean_block = 50` is hard-coded. Does it matter?

Facts verified in this repository:

- `50` is the default in all five signatures in [flowquake/stats.py](../flowquake/stats.py) (lines 45, 87, 116, 145, 164), and `grep -rn "mean_block" scripts/ configs/ runs/` returns **nothing** — no override, no config, no artifact record.
- **There is no block-length sensitivity check anywhere in the repository** — not a script, not a test, not a stored result. [tests/test_stats.py](../tests/test_stats.py) is three tests long: the one that names the bootstrap directly uses a *constant* series, which the early-return guard at [stats.py:62-64](../flowquake/stats.py#L62-L64) short-circuits before any resampling happens; the one that does reach the resampling loop (`test_paired_gain_summary_classifies_clear_win_and_loss`) checks only the `win`/`loss` label on a monotone `linspace` ramp, and never varies `mean_block`. No test would notice if the block length changed.
- [WORKING.md](../WORKING.md) item 16 finds the same independently: the block length "lives only in code ... No result JSON records it, so a reviewer working from the evidence pack cannot confirm the resampling scheme."
- **The check cannot be re-run from a clean clone**, because the per-event gain CSVs are excluded by `.gitignore` ([WORKING.md](../WORKING.md), "What cannot be re-derived here at all"). Every bootstrap CI under `runs/` summarises a file that is not present.

So the honest answer is: **the sensitivity is unknown and cannot be established from the public artifacts.** Do not claim it is fine. Claim instead that theory says the curve should have a broad plateau, and show what that looks like.

### 5.4 What a sensitivity curve looks like — exactly, for AR(1)

The real series is unavailable (gitignored), but the *shape* of the curve is not a matter of opinion: for an AR(1) it has a closed form, straight out of identity (*) in §4.5. With `rho_k = phi^k` and weights `(1-p)^k`, `p = 1/L` (dropping the `(n-k)/n` taper, which is negligible whenever `L << n`), the geometric series collapses:

```
VIF_SB(L) = 1 + 2 sum_{k>=1} (1-1/L)^k phi^k = (1 + q)/(1 - q),      q = phi (1 - 1/L)
```

a one-line result worth having in your head: **the stationary bootstrap with mean block `L` behaves exactly like an AR(1) whose coefficient has been shrunk from `phi` to `phi(1-1/L)`.** `L = 1` gives `q = 0` and `VIF = 1` (the i.i.d. bootstrap, §3.2); `L -> inf` gives the true `(1+phi)/(1-phi)`.

Evaluated at ComCat's implied `phi = 0.797` (§2.3), where the true SE inflation is `sqrt(8.85) = 2.975`:

| mean block L | q | VIF_SB(L) | SE inflation `sqrt(VIF)` | % of the true inflation |
|---|---|---|---|---|
| 1 | 0.000 | 1.00 | 1.000 | 33.6% |
| 2 | 0.399 | 2.33 | 1.525 | 51.2% |
| 5 | 0.638 | 4.52 | 2.126 | 71.4% |
| 10 | 0.717 | 6.07 | 2.465 | 82.8% |
| 25 | 0.765 | 7.52 | 2.741 | 92.1% |
| **50** | **0.781** | **8.13** | **2.852** | **95.9%** |
| 100 | 0.789 | 8.48 | 2.912 | 97.9% |
| 200 | 0.793 | 8.66 | 2.943 | 98.9% |
| 400 | 0.795 | 8.76 | 2.959 | 99.5% |
| inf | 0.797 | 8.85 | 2.975 | 100% |

This is the **bias** side of §5.1 and nothing else: it is a deterministic function, no seeds, no Monte Carlo. The reading is that `L = 50` recovers 96% of the true SE inflation for a series this sticky, `L = 10` recovers only 83%, and everything above `L ≈ 100` is buying under half a percent. Since bias falls as `1/L` while estimator variance grows as `L/n` (§5.1), the useful window is exactly the region where the last two columns have flattened but `n/L` is still large — here roughly `L in [25, 400]`.

**Monte Carlo on top of that.** Running the repo's own `stationary_block_bootstrap_ci` (`n_boot = 2000`) on one AR(1) realisation with `phi = 0.797`, `n = 21889`, at three bootstrap seeds, gives half-width / (`1.96 * SE_iid`):

| L | 1 | 2 | 5 | 10 | 25 | **50** | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 3 | 1.03 | 1.59 | 2.24 | 2.50 | 2.91 | 3.03 | 2.97 | 2.85 | 3.01 | 3.03 | 2.82 |
| seed 4 | 1.01 | 1.52 | 2.09 | 2.49 | 2.90 | 2.96 | 3.04 | 2.97 | 3.02 | 2.94 | 2.93 |
| seed 5 | 0.99 | 1.44 | 2.12 | 2.49 | 2.79 | 2.92 | 2.89 | 2.87 | 2.94 | 2.88 | 2.82 |
| **mean** | **1.01** | **1.52** | **2.15** | **2.49** | **2.86** | **2.97** | **2.97** | **2.90** | **2.99** | **2.95** | **2.86** |

Reproduce it with:

```python
import numpy as np, math
from flowquake.stats import stationary_block_bootstrap_ci
phi, n = 0.797, 21889
rng = np.random.default_rng(0)
s = 0.359                                    # any scale: the table is a ratio
x = np.empty(n); x[0] = rng.normal(0, s)
e = rng.normal(0, s * math.sqrt(1 - phi**2), n)
for i in range(1, n): x[i] = phi * x[i-1] + e[i]
se = x.std(ddof=1) / math.sqrt(n)
for L in (1, 2, 5, 10, 25, 50, 100, 200, 400, 800, 1600):
    for sd in (3, 4, 5):
        lo, hi = stationary_block_bootstrap_ci(x, mean_block=L, n_boot=2000, seed=sd)
        print(L, sd, round(((hi - lo) / 2) / (1.96 * se), 2))
```

Two readings, and one non-reading. **`L = 1` recovers the i.i.d. bootstrap** (ratio 1.01), confirming §3.2 empirically. **The curve is flat from `L ≈ 25` upward** at 2.86–2.99, with `L = 50` inside it — the seed-to-seed scatter of ±0.06 at fixed `L` is the same size as the movement across a factor of 60 in `L`, which *is* the point: at this `n`, Monte Carlo noise dominates block-length choice above `L ≈ 25`. The non-reading: nothing here degrades at large `L`, because `n = 21889` is far from the `L -> n` wall of §5.1; a curve on 1121 Iran events would look different and should be run separately.

**This is a synthetic series and says nothing about FlowQuake's real gains.** The AR(1) was *built* to have ComCat's implied `VIF`, so its agreement with §2.3's 2.98 at `L = 50` is construction, not evidence. What the section establishes is only that a curve of this shape is what a defensible `mean_block` justification looks like, and that the repository does not contain one.

**How to run the real check** (needs `runs/n1_density/per_event_test.csv`, gitignored):

```python
import pandas as pd
from flowquake.stats import stationary_block_bootstrap_ci
m = pd.read_csv("runs/n1_density/per_event_test.csv")      # + merge ETAS per-event scores
g = (m["tll"] - m["TLL"]).to_numpy()
for L in (1, 5, 10, 25, 50, 100, 200, 400, 800):
    lo, hi = stationary_block_bootstrap_ci(g, mean_block=L, n_boot=4000, seed=0)
    print(L, round(lo, 5), round(hi, 5), round((hi - lo) / 2, 6))
```

If the half-width is flat over `L in [25, 400]` and the win/tie/loss decision never changes there, say so in Methods and the criticism is answered. If SanJac_10 flips inside the plateau, that is a finding and it must be reported.

---

## 6. Confidence intervals from a bootstrap

Let `theta_hat` be the observed statistic, `theta_hat*_1..theta_hat*_B` the replicates, `q*_a` the `a`-quantile of the replicates.

### 6.1 The four constructions

**Percentile:** `CI = [q*_{alpha/2}, q*_{1-alpha/2}]`. Justification: if there exists a monotone `g` with `g(theta_hat) - g(theta) ~ H` for some `H` symmetric about 0 and free of `theta`, the percentile interval is exact — and you never need to know `g`. That is the "transformation-respecting" property, which is why a percentile interval for a variance is always positive. **First-order accurate**, coverage error `O(n^{-1/2})`; no correction for median bias or skewness. This is [stats.py:80](../flowquake/stats.py#L80).

**Basic (reverse-percentile):** pivot on `D = theta_hat - theta`, approximated by `D* = theta_hat* - theta_hat`, giving `CI = [2 theta_hat - q*_{1-alpha/2}, 2 theta_hat - q*_{alpha/2}]`. Percentile and basic are **reflections of each other about `theta_hat`** and coincide iff the bootstrap distribution is symmetric there. When they differ materially, that is a diagnostic that a first-order method is not good enough. Also first-order accurate.

**BCa** (Efron 1987, *JASA*, "Better bootstrap confidence intervals"): the percentile interval at adjusted levels

```
alpha_lo = Phi( z0 + (z0 + z_{alpha/2}) / (1 - a(z0 + z_{alpha/2})) )      (similarly alpha_hi)
```

with `z0 = Phi^{-1}(#{theta_hat*_b < theta_hat}/B)` correcting **median bias** and the acceleration `a` correcting **skewness**. Second-order accurate, `O(n^{-1})`, still transformation-respecting. **The catch for us:** `a` is conventionally jackknife-estimated, and the delete-one jackknife is *invalid under dependence* — deleting one observation destroys exactly the local structure the block bootstrap preserves. A block-bootstrap BCa needs a **delete-block jackknife** (Künsch 1989, same paper as MBB). Doable, not free, not done here.

**Studentized (bootstrap-t):** pivot on `t* = (theta_hat* - theta_hat)/se*`, requiring an SE inside every replicate. Second-order accurate and usually best for a mean — but each replicate of a dependent series needs its own long-run-variance estimate, multiplying cost and introducing a second block-length choice. Not transformation-respecting, and unstable when `se*` is near zero.

### 6.2 When percentile suffices, and when it does not

Percentile is adequate when the statistic is a smooth mean-like functional, `n_eff` is large, the bootstrap distribution is roughly symmetric and centred near `theta_hat`, and you want a *decision* rather than exact endpoints. It is inadequate when `n_eff` is small, the distribution is skewed, or you quote endpoints to several significant figures.

**Both failure conditions are live here.** `n_eff` is about 300 for SanJac_10 and 912 for SaltonSea_10 (§2.3) — 300 is not "large `n`". And the `dS` gain distribution is strongly right-skewed: the head wins on only 47.85% of forward-window events yet has mean `+0.0666` (§12), and a positive mean with a sub-50% win rate is the signature of a heavy right tail. Skew is precisely what BCa is for.

The honest position: percentile is defensible for most of the decisions actually made, but three intervals sit close enough to zero for a second-order correction to matter — SanJac_10 `dT` at `[-0.0057, 0.0759]`, Japan `dT` at `[-0.0319, 0.0049]`, and the nearest *win*, Iran's `dTot` at `[0.0098, 0.1711]`, whose lower endpoint is one percent of its upper. **BCa with a delete-block jackknife acceleration is the upgrade a referee can legitimately demand**, and those are exactly the claims it could move.

**Monte Carlo noise in the endpoints.** The same California series appears twice, bootstrapped with different seeds:

| artifact | dT CI | dTot CI |
|---|---|---|
| `runs/total_win.json` → `test_2007_2020` (seeds 11 / 15; [total_win_summary.py:78-84](../scripts/total_win_summary.py#L78-L84) starts at 11 and steps by 2 per metric) | [0.0403, 0.0675] | [0.1006, 0.1268] |
| `runs/stats_hardening.json` → `per_region.California` / `total_with_head_family.California` (seeds 100 / 600) | [0.0402, 0.0674] | [0.1006, 0.1261] |

The `dTot` upper endpoints differ by `7e-4`. **The intervals are printed to four decimal places but reproducible only to about three.** That is `B = 2000` replicate noise, not an error — but "0.1268" carries a false fourth digit, and since no artifact records `n_boot`, `mean_block` or `seed` ([WORKING.md](../WORKING.md) item 16), a referee recomputing sees an unexplained mismatch.

---

## 7. Bootstrap p-values and the resolution floor

### 7.1 The construction

[stats.py:113-126](../flowquake/stats.py#L113-L126):

```python
means = _bootstrap_means(values, mean_block=50, n_boot=4000, seed=seed)
b = len(means)
p_lo = (np.sum(means <= 0.0) + 1.0) / (b + 1.0)
p_hi = (np.sum(means >= 0.0) + 1.0) / (b + 1.0)
return float(min(1.0, 2.0 * min(p_lo, p_hi)))
```

This is a **percentile-interval-inversion p-value**: `p = 2 min(F*(0), 1-F*(0))`, the smallest `alpha` at which the two-sided percentile interval would exclude zero. It is *not* a p-value from resampling under `H0`. The more standard framing recentres the bootstrap distribution at zero and computes `P*(|theta_hat* - theta_hat| >= |theta_hat|)`; the two agree exactly when the bootstrap distribution is symmetric about `theta_hat` and differ in the tails otherwise — so for the skewed `dS` series they will differ. Neither is wrong; the repo's choice inherits the first-order accuracy and the skew-blindness of the percentile interval.

**Add-one smoothing.** `(count+1)/(B+1)` rather than `count/B`, for two reasons. A Monte-Carlo p-value must never be exactly 0, which would assert impossibility from finitely many draws. And validity: under `H0` the observed statistic is exchangeable with the `B` replicates, so its rank among `B+1` values is uniform on `{1..B+1}`, giving `P(p <= alpha) <= alpha` exactly; without the `+1` the Monte-Carlo p-value is anti-conservative. (Standard for permutation tests — Davison & Hinkley 1997, *Bootstrap Methods and their Application*; the bootstrap case approximates it, since bootstrap replicates are not exactly exchangeable with the observed value.)

### 7.2 The floor, verified against the artifacts

With `B = 4000` and all bootstrap means strictly positive (`count = 0`):

```
p_lo = (0+1)/(4000+1) = 1/4001        p = 2/4001 = 0.000499875...   ->  rounds to 0.0005
```

**`p_boot: 0.0005` means "at the resolution floor", i.e. `p <= 0.0005`. It is not a measured 0.0005** — the true p could be `1e-30`. Every `0.0005` in `runs/stats_hardening.json` and `runs/total_win.json` is this floor. [STACK.md](../STACK.md) Part VII says so correctly and it is worth repeating, because it is the most misread number in the artifacts.

**The lattice check.** Achievable p-values are exactly `2(k+1)/4001`, integer multiples of `2/4001 ≈ 0.00049988`. Every stored p-value in the repository lies on this lattice, which independently confirms `B = 4000` — a number recorded nowhere in the artifacts:

| stored p | multiple | exact value | implied count of replicate means `<= 0` |
|---|---|---|---|
| 0.0005 | 1 | 2/4001 = 0.0004999 | 0 |
| 0.0015 | 3 | 6/4001 = 0.0014996 | 2 |
| 0.0055 | 11 | 22/4001 = 0.0054986 | 10 |
| 0.009 | 18 | 36/4001 = 0.0089978 | 17 |
| 0.0185 | 37 | 74/4001 = 0.0184954 | 36 |
| 0.09098 | 182 | 364/4001 = 0.0909773 | 181 |
| 0.13697 | 274 | 548/4001 = 0.1369658 | 273 |
| 0.64784 | 1296 | 2592/4001 = 0.6478380 | 1295 |

Stored values from `runs/stats_hardening.json` → `family_dT_holm.*.p_raw` and `total_with_head_family.*.p_boot`. All eight land on the lattice; Worked Example 1 uses this to reproduce the Holm adjustments to five decimals.

### 7.3 A consistency trap

`stationary_block_bootstrap_ci` defaults to `n_boot = 2000`; `block_bootstrap_pvalue` and `tost_equivalence` default to `4000`; and [stats_hardening.py:146-157](../scripts/stats_hardening.py#L146-L157) passes seeds `100+i`, `200+i`, `300+i`. So within a single artifact entry the 95% CI, the p-value and the 90% TOST interval come from **three different bootstrap runs**. They are mutually consistent on every committed row — Japan `dT` has CI `[-0.0319, 0.0049]` and `p_boot = 0.13697` — but nothing enforces it. A CI excluding zero alongside `p_boot > 0.05` is possible and would look like a bug. One bootstrap draw feeding all three would cost nothing and remove the risk.

---

## 8. Multiple comparisons: FWER, FDR, and a proof of Holm

### 8.1 The two error rates

With `V` = true nulls rejected and `R` = total rejections: `FWER = P(V >= 1)` ("at least one false claim"); `FDR = E[V/max(R,1)]` ("expected fraction of claims that are false"). FWER is right when a **single** false claim damages the work; FDR is right when the output is a **screen** and you tolerate a known dud fraction among the hits. FlowQuake's regional claims are quoted individually ("FlowQuake beats ETAS in Japan"), so **FWER is the correct target** and the repo chooses correctly.

### 8.2 Bonferroni

Reject `H_i` iff `p_i <= alpha/m`. Let `I_0` be the true nulls, `m_0 = |I_0|`. A valid p-value has `P(p_i <= u) <= u` under its null, so

```
FWER = P( union_{i in I_0} {p_i <= alpha/m} )
    <= sum_{i in I_0} P(p_i <= alpha/m)       [Boole]
    <= m_0 (alpha/m) <= alpha
```

The union bound needs **no assumption whatsoever** about dependence. That is Bonferroni's whole selling point, and its conservatism when `m_0 < m` or tests are strongly positively dependent.

### 8.3 Holm's step-down procedure

Holm (1979, *Scandinavian Journal of Statistics*). Order `p_(1) <= ... <= p_(m)` with hypotheses `H_(1)..H_(m)`; let `k = min{ j : p_(j) > alpha/(m-j+1) }` (`k = m+1` if none); reject `H_(1)..H_(k-1)`. **Step-down**: walk upward from the smallest p-value and stop at the first failure. The intuition: at step 1 there are `m` live hypotheses so the threshold is `alpha/m`; if you reject `H_(1)` you have *learned* it was false, so at most `m-1` can still be true nulls and the threshold relaxes to `alpha/(m-1)`.

Adjusted p-values: `p~_(j) = max_{i <= j} min(1, (m-i+1) p_(i))`, so `H_(j)` is rejected at level `alpha` iff `p~_(j) <= alpha`. The outer `max` enforces monotonicity — an adjusted p-value never decreases as you move to a larger raw p-value — which is what makes the step-down stop coherent.

**The repo's implementation** ([stats.py:129-138](../flowquake/stats.py#L129-L138)):

```python
items = sorted(pvals.items(), key=lambda kv: kv[1])
m = len(items)
adjusted, running = {}, 0.0
for rank, (label, p) in enumerate(items):          # rank = 0..m-1
    running = max(running, (m - rank) * p)         # multiplier m - rank = m - (j-1) = m - j + 1
    adjusted[label] = float(min(1.0, running))
```

With `rank = j-1` the multiplier is exactly `m-j+1`. The `min(1, ·)` is applied after the running max rather than inside it, but since `running` is non-decreasing and `min(1, ·)` is monotone, `min(1, max_i x_i) = max_i min(1, x_i)` — identical. **The function is a correct Holm.** (The name `holm_bonferroni` is standard usage; the procedure is Holm, not Bonferroni.)

### 8.4 Proof that Holm controls FWER under arbitrary dependence

Let `I_0` be the true nulls, `m_0 = |I_0| >= 1` (nothing to prove if `m_0 = 0`), and `A` = "at least one true null is rejected". Suppose `A` occurs and let `r` be the **smallest rank at which a true null is rejected**. Then:

1. Holm is step-down and rejects a prefix, so all of `H_(1)..H_(r-1)` are rejected; by minimality of `r`, none of them is a true null.
2. Hence every one of the `m_0` true nulls sits at rank `>= r`. There are `m-r+1` such ranks, so `m - r + 1 >= m_0`, i.e. `1/(m-r+1) <= 1/m_0`.

Rejecting `H_(r)` requires `p_(r) <= alpha/(m-r+1) <= alpha/m_0`, and `p_(r)` is the p-value of some true null. So `A` implies `min_{i in I_0} p_i <= alpha/m_0`, and

```
FWER = P(A) <= P( min_{i in I_0} p_i <= alpha/m_0 )
            <= sum_{i in I_0} P( p_i <= alpha/m_0 )     [Boole]
            <= m_0 (alpha/m_0) = alpha
```

**QED.** The only assumptions are p-value validity under each null and the union bound. **No independence, no positive dependence, nothing.**

**Holm uniformly dominates Bonferroni:** the first threshold is identical (`alpha/m`) and every later one is strictly larger, so Holm rejects a superset. There is never a power reason to use Bonferroni. The one thing Bonferroni gives that Holm does not is a set of **simultaneous confidence intervals** (invert each `alpha/m` test); Holm's step-down structure does not invert as cleanly. If asked "why not Bonferroni?", answer "Holm dominates it at no cost"; if asked "why no Holm-adjusted intervals?", answer honestly that step-down procedures do not straightforwardly invert to simultaneous intervals.

### 8.5 Hochberg and Benjamini–Hochberg

Hochberg (1988, *Biometrika*) reverses direction: **step-up**, `k = max{ j : p_(j) <= alpha/(m-j+1) }`, reject `H_(1)..H_(k)`. Same thresholds walked from the largest p-value downward, stopping at the first success. It rejects everything Holm does and sometimes more — uniformly more powerful. **The price:** validity rests on the **Simes inequality** (Simes 1986, *Biometrika*), which holds under independence and positive regression dependence (PRDS; Sarkar 1998, *Annals of Statistics*) but **can fail under general dependence**, in which case Hochberg's FWER exceeds alpha.

FlowQuake's six regional p-values come from six catalogs but **one architecture, one codebase, one training recipe and overlapping seeds**. A systematic bug or a systematically favourable modelling choice would move all six the same way — positive dependence, the *benign* case for Simes. But the dependence structure is not characterised, and Holm needs no such argument. **Holm is the right choice**, and the reason to give is "we did not want to assert a dependence structure across regions."

Benjamini & Hochberg (1995, *JRSS-B*) reject `H_(1)..H_(k)` with `k = max{ j : p_(j) <= j alpha/m }`, controlling FDR under independence and PRDS. Under arbitrary dependence, Benjamini & Yekutieli (2001, *Annals of Statistics*) deflate the threshold by `sum_{i=1}^m 1/i` — for `m = 6` that is `2.45`, so BY is 2.45x stricter than BH. FDR would be right if the six regions were a screen whose *set* was reported; [MANUSCRIPT.md](../MANUSCRIPT.md) §4.4 quotes each region individually as a standalone claim, so FWER is right.

---

## 9. What is the family? The judgement call you must defend

No theorem settles this. The family is a *scientific* choice about which set of claims must be simultaneously true for the paper's conclusion to hold — and it is where a professor will push hardest.

### 9.1 What the repo does

[scripts/stats_hardening.py](../scripts/stats_hardening.py) builds **two separate families of six**: `family_dT_holm` (one temporal-gain claim per region, [:167-172](../scripts/stats_hardening.py#L167-L172)) and `total_with_head_family` (one total-gain-with-head claim per region, [:242-247](../scripts/stats_hardening.py#L242-L247)). Its docstring states the rationale — *"the family = one headline dT claim per region"* ([:4-5](../scripts/stats_hardening.py#L4-L5)) — and the artifact's `notes` field repeats it. A pre-stated, documented choice is worth a great deal.

### 9.2 Should the seeds be in the family?

**Against:** if the reported number is the mean over seeds, seeds are a variance-reduction device *inside one estimator*, not `k` hypotheses. **For:** if anyone ever looked at per-seed results and chose which to report, seeds become a selection step the family must absorb.

**What the repo can prove.** [stats_hardening.py:121-135](../scripts/stats_hardening.py#L121-L135) now globs `per_event_full_s*.csv` and averages the per-event series over every seed found ([:196-215](../scripts/stats_hardening.py#L196-L215)), and [:223-231](../scripts/stats_hardening.py#L223-L231) records `head_seeds`, `n_head_seeds`, `dTot_seed_means`, `dTot_seed_std`, a `seed_aggregation` string and a `single_seed_warning` flag — exactly the right design, making a one-seed run visible rather than indistinguishable from a three-seed one. It also prints a stderr warning naming the offending regions ([:238-241](../scripts/stats_hardening.py#L238-L241)).

**But the committed artifact predates it.** `runs/stats_hardening.json` → `total_with_head_family.California` has keys `[n, dTot_mean, dTot_ci, decision, p_boot, dTot_abs_below_0.05, temporal_variant, pairing, p_holm, significant_05_holm]` — **no `head_seeds`, no `n_head_seeds`, no `single_seed_warning`.** Same for `runs/total_win.json`, which lacks the `head_seed_files` / `n_head_seeds` / `single_seed_warning` keys that [total_win_summary.py:69-71](../scripts/total_win_summary.py#L69-L71) now writes. Both were produced by the pre-`2e8fa8a` code ("Stop the headline statistics resting on a single training seed").

So **the six committed total-likelihood numbers are a single training seed of a stochastically trained head, while [MANUSCRIPT.md](../MANUSCRIPT.md) describes 3-seed means.** [WORKING.md](../WORKING.md) item 3 says the same and proves it arithmetically: Italy `0.2095 - 0.0712 = 0.1383`, the seed-0 `dS` exactly, not the 3-seed `0.1373`. The correct viva answer is: *the code now averages over seeds and flags single-seed runs; the committed artifact predates that fix and is seed 0; the fix is a re-run, listed as an open item in the repository's own working document.*

### 9.3 Should the metrics be in the family? — with the sensitivity analysis

`dTot = dT + dS` exactly, so there are two degrees of freedom, not three.

**For pooling into one family of 12:** the headline is "FlowQuake beats ETAS", and any of `dT`, `dS`, `dTot` in any region could support it — twelve-plus chances to get lucky, and two separate six-member families let a claim ride on whichever family flatters it. **Against:** the two families answer different pre-stated questions in different paper sections, and — the strongest defence — **the repo reports its losses.** `runs/stats_hardening.json` → `per_region.California.dTot_mean` is `-0.3107` with `"decision": "loss"` (the base kernel-mixture model), sitting in the same file as the wins. A group that reports losses is not selecting on outcome.

**The analysis nobody ran.** Pool all twelve raw p-values into one Holm correction, using the exact `2(k+1)/4001` values from §7.2:

| claim | p_raw | p_holm (family of 6, stored) | p_holm (pooled 12) | sig @ 0.05 pooled? |
|---|---|---|---|---|
| dT/California | 0.000500 | 0.003 | 0.00600 | yes |
| dT/Italy | 0.000500 | 0.003 | 0.00600 | yes |
| dTot/California | 0.000500 | 0.003 | 0.00600 | yes |
| dTot/Italy | 0.000500 | 0.003 | 0.00600 | yes |
| dTot/Chile | 0.000500 | 0.003 | 0.00600 | yes |
| dTot/Japan | 0.001500 | 0.0045 | 0.01050 | yes |
| dTot/Greece | 0.005499 | 0.011 | 0.03299 | yes |
| dT/Chile | 0.008998 | 0.03599 | 0.04499 | yes |
| **dTot/Iran** | **0.018495** | **0.0185** | **0.07398** | **NO** |
| dT/Iran | 0.090977 | 0.27293 | 0.27293 | no |
| dT/Japan | 0.136966 | 0.27393 | 0.27393 | no |
| dT/Greece | 0.647838 | 0.64784 | 0.64784 | no |

**Eleven of twelve verdicts are unchanged; one flips.** Iran's total-likelihood claim goes from `p_holm = 0.0185` to `0.0740` and stops being significant. So [MANUSCRIPT.md](../MANUSCRIPT.md) §4.4's "Holm-adjusted p ≤ 0.019 across the family" is true for the six-region family and false for a pooled twelve. That is the precise answer to "what if I disagree with your family?" — and it is a *good* answer, because it shows five of six total-likelihood wins are robust to the family definition and names the one that is not.

Note what protects the rest: the resolution floor `2/4001 = 0.0005` is below `alpha/m` even at `m = 30` (`0.05/30 = 0.00167`), so any claim at the floor survives almost any family you can construct. The claims at risk are those with genuinely measured p-values in the 0.005–0.02 range.

### 9.4 Should the CSEP tests be in the family?

The CSEP N/S/M consistency tests are **goodness-of-fit tests of one forecast**, not head-to-head claims, and the repo reports pass *rates* rather than per-day significance — no multiplicity to correct. But the **paired McNemar comparison** in §4.2 *is* a hypothesis test supporting a headline claim, and it is in **no family at all**.

The counter-argument is stronger than it first looks: McNemar is used here to support a **null** conclusion, and multiple-comparison corrections make it *easier* to fail to reject — they push in the authors' favour. Adding a non-rejection to an FWER family is not a conservative act. The right remedy for that claim is an **equivalence bound**, not a correction (§11.4).

---

## 10. Equivalence testing: TOST

### 10.1 Absence of evidence

`p > 0.05` says the data are compatible with `H0`, not that `H0` is true. An infinitely noisy experiment gives `p > 0.05` every time. A CI of `[-5.0, +5.0]` and one of `[-0.001, +0.001]` both "fail to reject" and mean opposite things.

Not pedantry here: **Greece's native `dT` CI is `[-0.163, -0.045]`, and its few-shot `dT` CI is `[-0.056, +0.035]`** (the first from `runs/multiregion_master.json` → `Greece.native.paired.dT_ci`, the second from `runs/stats_hardening.json` → `per_region.Greece.dT_tost_0.1.ci90`). Both summarise as "not a win". Only the second supports "ties ETAS".

### 10.2 TOST, derived

To affirm `|theta| < delta` for a pre-stated margin `delta > 0`, set up the *composite* null

```
H0 : |theta| >= delta       i.e.   H0 = H0- ∪ H0+ ,   H0- : theta <= -delta ,   H0+ : theta >= +delta
H1 : |theta| <  delta
```

Schuirmann (1987, *Journal of Pharmacokinetics and Biopharmaceutics*): run **two one-sided tests**, each at level `alpha`, and declare equivalence iff **both** reject. Test 1 rejects `H0-` iff the one-sided level-`alpha` lower confidence bound exceeds `-delta`; Test 2 rejects `H0+` iff the one-sided upper bound is below `+delta`.

**Why the size is `alpha`, not `alpha^2` and not `2 alpha`.** This is the **intersection–union principle** (Berger 1982; Berger & Hsu 1996, *Statistical Science*). The null is a *union* and we reject only when both components are rejected. Fix any `theta` in `H0`; it lies in `H0-` or in `H0+`. Say `theta <= -delta`. Rejecting the overall null requires, in particular, rejecting `H0-`, and Test 1 has level `alpha` there:

```
P_theta(reject H0) <= P_theta(reject H0-) <= alpha
```

Symmetrically for `theta >= +delta`. Taking the sup over `H0`, the size is at most `alpha`. **No multiplicity correction is needed** — indeed the test is conservative, since at the boundary only one of the two tests is ever binding.

**Why a 90% interval for a 5% test.** The one-sided level-`alpha` lower bound is precisely the lower endpoint of a two-sided `(1-2alpha)` interval, and likewise the upper. With `CI_{1-2alpha} = [q_alpha, q_{1-alpha}]`, Test 1 rejects iff `q_alpha > -delta` and Test 2 rejects iff `q_{1-alpha} < +delta`; both reject iff `[q_alpha, q_{1-alpha}]` lies inside `(-delta, +delta)`. With `alpha = 0.05`, `1 - 2alpha = 0.90`. Hence:

```
Equivalence at margin delta, level 0.05   <=>   the 90% two-sided CI lies strictly inside (-delta, +delta)
```

Using the 95% interval would make it a level-0.025 test — valid but less powerful, and your stated alpha would be wrong. This is the most commonly botched detail in equivalence testing and a professor will ask.

### 10.3 The repo's implementation

[stats.py:141-158](../flowquake/stats.py#L141-L158):

```python
means = _bootstrap_means(values, mean_block=50, n_boot=4000, seed=seed)
lo, hi = np.percentile(means, [5.0, 95.0])          # <- 90% two-sided interval
return {"margin": float(margin), "ci90": [float(lo), float(hi)],
        "equivalent": bool(lo > -margin and hi < margin)}
```

Percentiles 5 and 95, strict containment: **a correct level-0.05 TOST** on the stationary-block-bootstrap distribution of the mean. The key name `ci90` is honest about what it is.

### 10.4 Is the 0.05 / 0.10 nats/event margin defensible?

The repo tests both ([stats_hardening.py:156](../scripts/stats_hardening.py#L156)). Three readings:

- **As a likelihood ratio.** `exp(0.05) = 1.0513`, `exp(0.10) = 1.1052` — "equivalent at 0.05" means the two models' per-event densities at the realised outcome differ by under 5.1% on average.
- **In aggregate.** `0.05 * 21889 = 1094` nats of total log-likelihood ratio on the California test set — a Bayes factor of `e^1094`. Per-event and aggregate intuitions point in opposite directions; you must say which you mean.
- **As a fraction of the available skill.** ETAS's temporal edge over the Poisson floor is `1.4343428344882627 - 0.5126406686259881 = 0.9217` nats/event (`runs/n1_density/eval_test.json` → `baselines.ETAS.tll`, `baselines.Poisson.tll`). So `0.05` is **5.4%** and `0.10` is **10.8%** of everything ETAS's temporal model buys over "you learned nothing". A defensible anchor.

**Now the criticism.** The same margins are applied to `dTot` (Italy's `dTot_tost_0.05`, [stats_hardening.py:162-164](../scripts/stats_hardening.py#L162-L164)). But ETAS's **total** edge over Poisson is `13.261863460288378 - 7.2554275527505645 = 6.0064` nats/event (same file, `baselines.*.nll`). On that scale `0.05` is **0.83%** and `0.10` is **1.66%** of the available skill — as a *fraction*, six and a half times stricter, or read the other way, six and a half times laxer as an absolute standard for "the same". A margin of 0.05 nats is a meaningful bar on the temporal scale and close to a free pass on the total scale.

**Defensible position:** pre-register the margin *per metric*, scaled to that metric's dynamic range — e.g. 5% and 10% of the ETAS-minus-Poisson gap, giving roughly `(0.046, 0.092)` for `dT` and `(0.30, 0.60)` for `dTot`. The repo does none of this and states no derivation for `0.05`/`0.10` at all. That is a real, specific, correctable weakness. It does, however, cut in the direction that matters least: it makes `dTot` equivalence *easier* to declare, and the repo mostly uses `dTot` TOST to support a tie for Italy's base model, not to support a win.

### 10.5 Power

For a mean with standard error `se`, TOST power at true effect `theta` is approximately

```
Phi( (delta - theta)/se - z_{1-alpha} ) + Phi( (delta + theta)/se - z_{1-alpha} ) - 1
```

with `z_{0.95} = 1.645`. At `theta = 0` this is `2 Phi(delta/se - 1.645) - 1`, which reaches 0.80 only when `delta/se >= 1.645 + 1.282 = 2.927` — **you need `delta` to be about 2.9 standard errors.** For Japan's `dT`, `se ≈ (0.0011 + 0.0289)/(2 × 1.645) = 0.0091`, so `delta/se = 5.5` — comfortably powered. For Greece, `se ≈ (0.0352 + 0.0560)/(2 × 1.645) = 0.0277`, so `delta/se = 1.80` at `delta = 0.05` — **underpowered**, which is exactly why Greece fails TOST at 0.05 and passes at 0.10 (`0.10/0.0277 = 3.6`). "Greece is equivalent at 0.10" partly means "Greece is too noisy to say anything tighter", and the write-up should say both.

---

## 11. McNemar's test

### 11.1 Derivation

`n` paired binary outcomes — here, per CSEP forecast day, whether each model passes:

```
                B pass   B fail
      A pass     n11      n10
      A fail     n01      n00
```

Model `(n11, n10, n01, n00) ~ Mult(n; p11, p10, p01, p00)`. The question is **marginal homogeneity**:

```
P(A passes) = p11 + p10 ,  P(B passes) = p11 + p01     =>     H0: p10 = p01
```

The concordant cells cancel: `p11` and `p00` are pure nuisance. **Eliminate them by conditioning.** Let `d = n10 + n01`. Conditionally on `d`,

```
n10 | d  ~  Binomial( d, p10/(p10 + p01) )        which under H0 is    Binomial(d, 1/2)
```

free of *all* nuisance parameters — a similar test. The **exact two-sided p-value** is twice the smaller binomial tail, capped at 1:

```
p_exact = min( 1, 2 * sum_{i=0}^{min(n10,n01)} C(d,i) / 2^d )
```

The large-sample form `(n10 - n01)^2/(n10 + n01) ~ chi^2_1` is a poor approximation for small `d`; use the exact form whenever `d < 25`.

**Why only discordant pairs carry information.** Algebraically, `H0` constrains only `p10` and `p01`; the concordant counts inform `p11` and `p00`, which are unconstrained under both hypotheses, so conditioning on them loses nothing. Intuitively, a day on which both models pass (or both fail) tells you the day was easy (or hard) — it cannot tell you which model is better. Only disagreement is a vote. This is also McNemar's weakness: `n` can be 10,000, but if only 6 pairs are discordant, the test has 6 observations.

### 11.2 The FlowQuake numbers, recomputed first-hand

Recomputed here from the raw per-day S-test quantiles in `runs/n1_density/csep_head/csep_results.json` and `runs/csep_h2h_etas/csep_results.json` — a day passes when its S-test quantile (or the minimum of a two-sided pair) is `>= 0.025`, with NaN and the `(-1,-1)` not-evaluable sentinel excluded, matching `csep_summary` in [flowquake/csep_forecast.py](../flowquake/csep_forecast.py):

```
head standalone summary   : N 95/100, S 79/85 (0.9294), M 90/92
shared evaluable days     : 83
head passes / ETAS passes : 77 / 77
head-only passes (n10)    : 5
ETAS-only passes (n01)    : 5
discordant d              : 10
concordant agreements     : 73        <- NOT 77
```

Exact McNemar with `d = 10`, `min(n10, n01) = 5`:

```
sum_{i=0}^{5} C(10,i) = 1 + 10 + 45 + 120 + 210 + 252 = 638
one tail = 638/1024 = 0.623047        p = min(1, 2 × 0.623047) = 1.0000
```

confirming [results/CLAIMS.md](../results/CLAIMS.md) C19 and [MANUSCRIPT.md](../MANUSCRIPT.md) §4.2 (`:458`). The second comparison (head vs the production kernel-mixture head, `runs/csep_h2h_fq/csep_results.json`) has 81 shared days, 75 vs 78 passes, `n10 = 3`, `n01 = 6`, `d = 9`:

```
sum_{i=0}^{3} C(9,i) = 1 + 9 + 36 + 84 = 130        p = 2 × 130/512 = 0.5078125
```

confirming C20.

**A wording correction the repo already knows about.** [MANUSCRIPT.md:457-458](../MANUSCRIPT.md) says the models "agree on **77/83 evaluable days each**". That is each model's *pass count*, not an agreement count; true concordance is **73/83** (= 83 − 10 discordant), verified above. [WORKING.md](../WORKING.md) item 15 lists it among seven strings to fix.

**A provenance point.** [STACK.md](../STACK.md) Part VII lists "McNemar's exact test (in `scripts/audit_readiness.py`)" among the repo's tools. [audit_readiness.py:314](../scripts/audit_readiness.py#L314) only *counts* discordant days and emits the literal string `"(McNemar p~1.0)"`; it computes no p-value, and `grep -rn "binom\|scipy.stats" --include="*.py" .` over the whole repository returns nothing (SciPy *is* used elsewhere — `scipy.spatial`, `scipy.optimize`, `scipy.ndimage` — but `scipy.stats` is never imported and no binomial tail is computed). **No McNemar p-value is computed anywhere in the codebase** — the value is asserted in prose, and it happens to be right. Say this before a referee does.

### 11.3 The power problem

With `d = 10` the complete set of achievable two-sided exact p-values is:

| split (n10/n01) | 0/10 | 1/9 | 2/8 | 3/7 | 4/6 | **5/5** |
|---|---|---|---|---|---|---|
| exact p | 0.001953 | 0.021484 | 0.109375 | 0.343750 | 0.753906 | **1.000000** |

To reach `p < 0.05` you need at least a **9-1 split**; the observed 5-5 is the least informative outcome available. Power, computed exactly as `sum_{k: p(k) <= 0.05} C(d,k) theta^k (1-theta)^{d-k}` with `theta = p10/(p10+p01)`:

| true theta | 0.50 (null) | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|
| power at d = 10 | 0.022 | 0.048 | 0.150 | **0.376** | 0.736 |
| power at d = 40 | 0.039 | 0.212 | 0.703 | 0.981 | 1.000 |
| power at d = 100 | 0.035 | 0.462 | 0.979 | 1.000 | 1.000 |

**At `d = 10` the test has 38% power against a true 80/20 discordance split and 15% against 70/30.** So `p = 1.00` is compatible with a genuinely large difference in S-test pass propensity. It is *absence of evidence*, and after §10 you know what to do about that.

### 11.4 What the claim can and cannot say

[REPLACEMENT_READINESS.md:27](../REPLACEMENT_READINESS.md) and [MANUSCRIPT.md:592](../MANUSCRIPT.md) both call the head's S-test "statistically indistinguishable from ETAS's". Strictly, `p = 1.00` licenses only *"we found no evidence of a difference, on 83 days with 10 discordant"*.

Two things save the claim. The **direction of the argument**: what is being defended is not "the head is better on consistency" but "the +0.06 nats/event spatial gain **cost nothing** in calibration", and a null result is the right *shape* of evidence for a "cost nothing" claim provided the standalone pass rates are also reported. [STACK.md](../STACK.md) Part VI states this reading explicitly and correctly. And the **standalone rates are strong independently**: S 79/85 = 92.9% against a 5% nominal rejection rate does not depend on any comparison.

Two things do not save it. **No equivalence test was run for this claim** — the repo demands TOST for every "ties ETAS" *likelihood* statement ([stats_hardening.py:7-11](../scripts/stats_hardening.py#L7-L11)) and then makes a "statistically indistinguishable" *CSEP* statement from a bare non-rejection. That is internally inconsistent. The fix is an exact binomial equivalence bound on `theta = p10/(p10+p01)`: with `n10 = 5`, `d = 10`, the Clopper–Pearson 90% interval is `[Beta^{-1}_{0.05}(5,6), Beta^{-1}_{0.95}(6,5)] = [0.222, 0.778]` (computed here; the repo does not compute it), so all you can honestly assert is "the discordance is no more lopsided than about 78/22" — which does not exclude a materially worse model. And **"indistinguishable" implies equivalence**; prefer "we found no significant difference (McNemar exact p = 1.00 on 10 discordant days; ~38% power against an 80/20 alternative)". That sentence survives a viva; the current one invites the attack.

---

## 12. Paired means versus win rates

### 12.1 A constructed example

Take 1000 events: on 520 the model loses 0.02 nats, on 480 it wins 0.10.

```
win rate = 480/1000 = 48.0%                                  < 50%
mean     = 0.520(-0.02) + 0.480(+0.10) = -0.0104 + 0.0480 = +0.0376 nats/event      > 0
```

A model that loses more often than it wins can still be much better on average, because log-likelihood is unbounded below: one event where the baseline assigns near-zero density contributes an arbitrarily large gain, while an event the baseline already handles well can only be beaten by a little. The reverse is equally constructible: win 70% by 0.001 and lose 30% by 0.01 gives a 70% win rate and mean `-0.0023`.

**The two statistics answer different questions.** The mean is the proper-scoring-rule comparison — the quantity determining which model to bet on. The win rate is a sign-test statistic about the *median* gain, insensitive to magnitude by construction.

### 12.2 FlowQuake's actual numbers

From `runs/total_win.json` (single-seed head; see §9.2):

| window | metric | mean (nats/event) | 95% CI | win rate | p_boot |
|---|---|---|---|---|---|
| test 2007–2020 (n = 21889) | dT | 0.0533 | [0.0403, 0.0675] | **60.80%** | 0.0005 |
| | dS | 0.0600 | [0.0510, 0.0688] | **49.72%** | 0.0005 |
| | dTot | 0.1133 | [0.1006, 0.1268] | 56.54% | 0.0005 |
| forward 2020–2026 (n = 10187) | dT | 0.0574 | [0.0376, 0.0819] | **60.51%** | 0.0005 |
| | dS | 0.0666 | [0.0553, 0.0784] | **47.85%** | 0.0005 |
| | dTot | 0.1241 | [0.1035, 0.1455] | 55.16% | 0.0005 |

Source: `runs/total_win.json` → `test_2007_2020.{dT,dS,dTot}`, `forward_2020_2026.{dT,dS,dTot}`.

**The forward-window `dS` row is exactly the constructed pattern**: mean `+0.0666`, strictly positive CI, win rate **47.85%** — below half. In-window `dS` is `49.72%`, also below half. The temporal gain has the *opposite* shape: 60.5% win rate at a similar mean.

### 12.3 What the shape implies

Solve for implied magnitudes. If losers lose `L` on average and winners win `W`, then for forward `dS`: `0.4785 W - 0.5215 L = 0.0666`. With `L = 0.05` (a small routine loss), `W = 0.1937` — **the average winning margin is 3.9x the average losing margin.** The gain is *concentrated*.

Physically this is the expected signature of the neural-ETAS head. In the dense aftershock cloud — the majority of events — ETAS's power-law spatial kernel is already near-optimal and the head can only match it or lose slightly. Where the head wins is the minority ETAS handles badly: background events far from any recent parent, where the head's causal multi-scale smoothed-seismicity background contributes real density and ETAS's near-uniform background contributes almost none. Losing 0.05 nats on a well-modelled aftershock and gaining 2 nats on a background event ETAS essentially did not anticipate produces exactly this pattern.

**Defend it** with: log-likelihood is the proper score, the mean is the right summary, the CI excludes zero, and the pattern replicates out of time (49.72% → 47.85% win rate, +0.0600 → +0.0666 mean) — a stable model property, not a fluke. **Attack it** with: an operational forecaster who cares about the *typical* day may prefer the model that wins more often, and a mean driven by a minority of events is more fragile to a handful of mis-located catalog entries. The right supplementary evidence is a decomposition of the mean gain by event type and distance-to-nearest-parent — and `runs/n1_density/spatial_gap_decomp.json` holds only a background-vs-triggered split with no distance strata, which [WORKING.md](../WORKING.md) lists under "Needs hardware or data" (N6–N8). **The natural robustness check is stated as missing in the repo's own working document.** Say so before you are asked.

---

## 13. Model selection and test-set hygiene

**Mechanics.** Train `< 1998-01-01`, validate `[1998-01-01, 2007-01-01)`, test `[2007-01-01, 2020-01-17)` = 21,889 events at `m_c = 2.5` ([README.md](../README.md)). Early stopping uses validation only — [scripts/train_neural_etas.py](../scripts/train_neural_etas.py)'s docstring: "`[val_start, test_start)` validate (early stopping)". Correct hygiene. The forward window 2020-01-17 → 2026 (10,187 events) was used for neither fitting nor early stopping and is the strongest evidential object in the repository.

**The garden of forking paths.** Gelman & Loken (2013 working paper; 2014, *American Scientist*, "The statistical crisis in science"): even with no deliberate p-hacking, a researcher who *would have* made different analysis choices had the data looked different has an effective multiple-comparisons problem that no correction over the *reported* tests can fix. The relevant count is not "how many tests did I report" but "how many analyses were reachable". Reachable choices visible in this repository: which of five California catalogs to headline; native vs few-shot temporal variant per region (`temporal_variant` is a per-region field); the `h_bottleneck` value; the pairing key (`index_from_zero` vs `time+duplicate_rank`); `mean_block`; `n_boot`; the TOST margins; the family definition; which seeds exist.

**How many times has the test set been looked at?** A lower bound from committed artifacts alone: `find runs -name "eval_test.json"` returns **65** files (all 65 git-tracked) across **78** top-level run directories, and the `h`-ablation alone scores test at `h in {0,4,16,64}` × `{best,last}` = 8 evaluations (`runs/ablation_h/memorization_figure.json`, every row of which carries a `test` block). So the test window has been scored on the order of **70+ times** in the committed evidence, and more in runs that were not kept. **The repo says so itself**, verbatim in [scripts/train_neural_etas.py](../scripts/train_neural_etas.py)'s docstring:

> "The reported grids include ablations and multiple seeds; **do not describe these runs as a test-scored-once protocol.**"

Quote that rather than wait to be caught by it. The defensible framing: *the 2007–2020 window functioned as a development set for ablation and selection; the 2020–2026 forward window is the only genuinely held-out evaluation, and it was scored with a frozen model.* "We evaluated once on test" is not available.

**Seeds.** Two axes: the flow model's training seed (`_s1553/_s1554/_s1555`) and the head's (`_s0/_s1/_s2`). [MANUSCRIPT.md:5-7](../MANUSCRIPT.md) advertises "3-seed (mean ± std)". Two live issues: the committed headline total-likelihood artifacts are **single-seed** (§9.2, code fixed, artifacts not regenerated); and the per-seed spread bound is misstated — `MANUSCRIPT.md:570` claims ≤0.003 and `:972` claims ≤0.006, while the actual max−min over `runs/neural_etas/<region>/summary_full_s{0,1,2}.json` → `dS_mean` reaches **0.0070** for Chile ([WORKING.md](../WORKING.md) item 2). The surviving conclusion — "all six clear zero at every seed" — is true, since every per-seed `dS_ci` is strictly positive. Neither issue is seed *selection*, the serious version of the offence; but neither is verifiable from the artifacts either, because the artifacts do not record which seeds existed.

**Pre-registration.** [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md)'s ladder has rungs 1–3 `[DONE]` and **rung 4 — "freeze a checkpoint and run rolling forecasts on a future catalog window that was not used for model selection" — not done.** `runs/total_win.json`'s own `notes` opens: *"forward_2020_2026 is a retrospective out-of-time/pseudo-prospective replication, not a registered prospective forecast."* That is the statistical point: the objection pre-registration answers is not "is the model good" but "did anyone see the answers first", and external custody of the model and of a not-yet-existing catalog is the only thing that answers it. [WORKING.md](../WORKING.md)'s "The collaboration hole" makes this argument at length and correctly.

**The claim-tracing audit.** [WORKING.md](../WORKING.md) reports an independent read of every reported number against the artifacts:

| quantity | value |
|---|---|
| traced claim rows | 142 (134 distinct claims) |
| match exactly or to rounding | 114 — of which **63 exact** at the artifact's own precision, **51** round to the printed value |
| ambiguous between two committed artifacts | 2 |
| **distinct claims contradicted by their artifact** | **8** |
| **distinct claims with no committed backing at all** | **12** |
| committed run files | 226 (136 summary JSONs, 90 run configs) |
| readiness verdict | `RESEARCH_PREVIEW_READY`; 15 checks, 11 PASS / 4 WARN |

Sources: [WORKING.md](../WORKING.md) "Current state"; per-claim map [results/CLAIMS.md](../results/CLAIMS.md); readiness tally `runs/replacement_readiness.json` → `overall`, `checks[].level`.

Eight contradicted claims out of 134 is a 6% error rate — better than most papers, and *knowable* only because someone did the audit. The right viva posture is to name the number, name the worst case — [MANUSCRIPT.md:328-331](../MANUSCRIPT.md)'s "individually significant in every era", which `runs/prospective.json` contradicts (it stores no per-window CI or p-value at all, and Chile is positive in only 10 of 19 180-day windows, `bins_dT_positive_frac = 0.5263`) — and say it is scheduled for correction.

---

## Worked example 1 — Holm by hand on the six regional `dT` p-values

**Goal:** reproduce `runs/stats_hardening.json` → `family_dT_holm.*.p_holm` using only arithmetic.

**Step 1 — recover the exact raw p-values.** Stored `p_raw` is rounded to 5 dp, but §7.2 says every bootstrap p is an exact multiple of `2/4001`. Invert: `k+1 = round(p_stored × 4001/2)`.

| region | `p_raw` stored | `p × 4001/2` | count `k+1` | exact `p = 2(k+1)/4001` |
|---|---|---|---|---|
| California | 0.0005 | 1.00025 | 1 | 2/4001 = 0.000499875 |
| Italy | 0.0005 | 1.00025 | 1 | 2/4001 = 0.000499875 |
| Chile | 0.009 | 18.0045 | 18 | 36/4001 = 0.008997751 |
| Iran | 0.09098 | 182.0055 | 182 | 364/4001 = 0.090977256 |
| Japan | 0.13697 | 274.0085 | 274 | 548/4001 = 0.136965759 |
| Greece | 0.64784 | 1296.0039 | 1296 | 2592/4001 = 0.647838040 |

**Step 2 — sort ascending, apply multiplier `m - j + 1` with `m = 6`, then take the running maximum and cap at 1.**

| rank `j` | region | exact `p_(j)` | `× (6-j+1)` | product | running max | `p_holm` | stored | match |
|---|---|---|---|---|---|---|---|---|
| 1 | California | 0.000499875 | ×6 | 0.002999250 | 0.002999250 | **0.00300** | 0.003 | yes |
| 2 | Italy | 0.000499875 | ×5 | 0.002499375 | max(0.0029993, 0.0024994) = 0.002999250 | **0.00300** | 0.003 | yes |
| 3 | Chile | 0.008997751 | ×4 | 0.035991002 | 0.035991002 | **0.03599** | 0.03599 | yes |
| 4 | Iran | 0.090977256 | ×3 | 0.272931767 | 0.272931767 | **0.27293** | 0.27293 | yes |
| 5 | Japan | 0.136965759 | ×2 | 0.273931517 | 0.273931517 | **0.27393** | 0.27393 | yes |
| 6 | Greece | 0.647838040 | ×1 | 0.647838040 | 0.647838040 | **0.64784** | 0.64784 | yes |

**All six reproduce to the last stored digit.** Note Italy's raw product (0.002499) is *smaller* than California's yet its adjusted p-value is the same — the running max doing its job. And Chile at `0.03599` is significant at 0.05 while Iran at `0.27293` is not, matching the stored `significant_05` flags `true, true, false, true, false, false` for California, Italy, Japan, Chile, Greece, Iran.

**Sanity checks that would have caught an error:** the adjusted values are monotone (`0.003 <= 0.003 <= 0.03599 <= 0.27293 <= 0.27393 <= 0.64784`); the last multiplier is 1 so `p_holm(Greece) = p_raw(Greece)`; every adjusted p exceeds its raw p. And critically — had you used the *rounded* 0.0005 you would get `6 × 0.0005 = 0.0030` (agrees) but `4 × 0.009 = 0.036` where the stored value is `0.03599` (**disagrees in the fifth digit**). That is how you know the code passed the unrounded float to `holm_bonferroni` and rounded only on output ([stats_hardening.py:166-171](../scripts/stats_hardening.py#L166-L171)).

Five lines of Python if you prefer:

```python
from flowquake.stats import holm_bonferroni
raw = {"California": 2/4001, "Italy": 2/4001, "Japan": 548/4001,
       "Chile": 36/4001, "Greece": 2592/4001, "Iran": 364/4001}
print({k: round(v, 5) for k, v in holm_bonferroni(raw).items()})
# {'California': 0.003, 'Italy': 0.003, 'Japan': 0.27393, 'Chile': 0.03599,
#  'Greece': 0.64784, 'Iran': 0.27293}
```

## Worked example 2 — a TOST decision by hand

The rule (§10.2): equivalence at margin `delta`, level 0.05, holds iff the **90%** two-sided CI lies strictly inside `(-delta, +delta)`.

**Case A — Japan `dT`, margin 0.05.** From `runs/stats_hardening.json` → `per_region.Japan.dT_tost_0.05`:

```
ci90 = [-0.028939975305366163, +0.0010742098537419409],  delta = 0.05

Lower test:  ci90_lo > -delta ?     -0.028940 > -0.050000    TRUE   (0.021060 to spare)
Upper test:  ci90_hi < +delta ?     +0.001074 < +0.050000    TRUE   (0.048926 to spare)
Both reject at 5%  =>  EQUIVALENT at 0.05
```

Stored `"equivalent": true`. Correct. Japan's temporal gain is `-0.0139` nats/event and we can now *affirm* that whatever it is, it is smaller in magnitude than 0.05 — a per-event density difference under `exp(0.05) = 5.1%`. That is a positive statement about a tie, which `dT_decision_raw: "tie"` alone would not have licensed.

**Case B — Iran `dT`, margin 0.10.** From `per_region.Iran.dT_tost_0.1`:

```
ci90 = [-0.12210308721050286, -0.0013525685331943467],  delta = 0.10

Lower test:  -0.122103 > -0.100000    FALSE  (overshoots by 0.022103)
Upper test:  -0.001353 < +0.100000    TRUE
Not both  =>  NOT EQUIVALENT even at the loose 0.10 margin
```

Stored `"equivalent": false` at both margins. Correct — and this is the part to get right in a viva. Iran's `dT` has `p_boot = 0.09098` and a 95% CI `[-0.1335, 0.0112]` crossing zero, so the naive reading is "tie". **TOST refuses to certify it.** The data are compatible with FlowQuake being up to 0.122 nats/event *worse* in Iran — a 13% lower per-event predictive density. The honest sentence is "Iran's few-shot temporal result is inconclusive: neither a win nor a demonstrable tie, at `n = 1121` events."

**Case C — construct one yourself.** With a 90% CI of `[-0.041, +0.038]`: at `delta = 0.05`, `-0.041 > -0.05` and `+0.038 < +0.05`, both TRUE → equivalent. At `delta = 0.04`, `-0.041 > -0.04` is FALSE → not equivalent. The decision hinges on the *pre-stated* margin, which is why it must be fixed before the data are seen. A margin chosen after seeing the CI is not a test.

## Worked example 3 — `n_eff` by hand

The §2.2 caricature with `n = 12`, `B = 4`, `K = 3`: gains `+0.20 ×4, -0.10 ×4, +0.05 ×4`.

```
Xbar = (0.20 - 0.10 + 0.05) × 4/12 = 0.15 × 4/12 = 0.05
sigma^2 = [(0.15)^2 + (-0.15)^2 + 0^2] × 4/12 = 0.045 × 4/12 = 0.015

i.i.d. answer :  Var(Xbar) = sigma^2/n = 0.015/12 = 0.00125   ->  SE = 0.0354
correct answer:  Var(Xbar) = sigma^2/K = 0.015/3  = 0.00500   ->  SE = 0.0707
VIF = 0.00500/0.00125 = 4 = B          n_eff = 12/4 = 3 = K
```

The i.i.d. SE is understated by `sqrt(4) = 2x`. Confirm via autocorrelations: `rho_k = (1-k/4)_+` gives `0.75, 0.50, 0.25, 0`, so `VIF = 1 + 2(0.75+0.50+0.25) = 4`. Agrees. Scale up: ComCat's implied `VIF = 8.86` says its 21,889 events behave like 2,469 independent observations — as if events arrived in identical clumps of about nine.

---

## How this shows up in FlowQuake

[STACK.md](../STACK.md) Part VII is the code walkthrough for this material; this section only maps theory to location.

| theory | code / artifact | note |
|---|---|---|
| §2 autocorrelated mean | `stderr` at [stats.py:176](../flowquake/stats.py#L176) is the **i.i.d.** SE, stored beside the bootstrap CI | keeping both is what makes the §2.3 VIF table computable — a good decision |
| §4.4 stationary bootstrap | [stats.py:42-81](../flowquake/stats.py#L42-L81) | geometric lengths, uniform starts, `% n` wrap, mean 50 — matches theory (§4.6) |
| §5.3 block length | `mean_block: int = 50` in five signatures; **no override anywhere; no sensitivity check anywhere** | [WORKING.md](../WORKING.md) item 16 flags that no artifact records it |
| §6 percentile CI | [stats.py:80](../flowquake/stats.py#L80) | first-order accurate; BCa is the upgrade |
| §7 p-value + floor | [stats.py:113-126](../flowquake/stats.py#L113-L126) | `p_boot: 0.0005` = `2/4001` = the floor, not a measurement |
| §8.3 Holm | [stats.py:129-138](../flowquake/stats.py#L129-L138) | correct Holm; Worked Example 1 reproduces all six stored values |
| §9.1 the family | [stats_hardening.py:167-172](../scripts/stats_hardening.py#L167-L172), [:242-247](../scripts/stats_hardening.py#L242-L247) | two families of six, pre-stated in the docstring and the artifact `notes` |
| §10 TOST | [stats.py:141-158](../flowquake/stats.py#L141-L158); margins at [:156](../scripts/stats_hardening.py#L156) | percentiles 5/95 = 90% CI = level-0.05 TOST, correct |
| §11 McNemar | **not implemented**; [audit_readiness.py:314](../scripts/audit_readiness.py#L314) emits the string `"(McNemar p~1.0)"` | value asserted in prose; recomputed correctly here and in `results/CLAIMS.md` C19/C20 |
| §12 win rate vs mean | [total_win_summary.py:82](../scripts/total_win_summary.py#L82), [score_neural_etas.py:49](../scripts/score_neural_etas.py#L49) | `runs/total_win.json` holds the 47.85% / +0.0666 pattern |
| §13 hygiene | [scripts/train_neural_etas.py](../scripts/train_neural_etas.py) docstring | "do not describe these runs as a test-scored-once protocol" |
| decision recording | `ci_decision`, [stats.py:32-39](../flowquake/stats.py#L32-L39) | the win/loss/tie string is written into the artifact, so the rule is recorded with the number rather than applied later by a human |

**Where the repo contradicts its own documentation** — found while writing this chapter; report these before a referee does:

1. [STACK.md](../STACK.md) Part VII lists "McNemar's exact test (in `scripts/audit_readiness.py`)" among the repo's tools. No McNemar p-value is computed anywhere in the codebase; `audit_readiness.py` counts discordant days and prints a hard-coded `"p~1.0"`. The asserted value is nonetheless correct (§11.2).
2. `runs/stats_hardening.json` and `runs/total_win.json` were both produced by the pre-`2e8fa8a` single-seed code and lack the `head_seeds` / `n_head_seeds` / `single_seed_warning` keys the current scripts write. **The committed total-likelihood headline is one training seed**, while [MANUSCRIPT.md](../MANUSCRIPT.md) describes 3-seed means ([WORKING.md](../WORKING.md) item 3 proves it arithmetically).
3. [WORKING.md](../WORKING.md) states "One per-event file is tracked repo-wide (`runs/neural_etas/ComCat_25/per_event_forward_full.json`)". That file is 288 bytes and is a *summary* (`n`, `dS_mean`, `dS_ci`, `dS_win_rate`, `decision`, `ckpt`, `feat`) — despite its name it contains no per-event data. **No per-event series is recoverable from the repository**, which is why §5.3's block-length check cannot be run from a clean clone.
4. [MANUSCRIPT.md:457-458](../MANUSCRIPT.md)'s "they agree on 77/83 evaluable days each" is each model's pass count; true concordance is 73/83 (§11.2). [WORKING.md](../WORKING.md) item 15 already lists this.

---

## Common misconceptions

1. **"n = 21,889, so the standard error is tiny."** Actually `n_eff` is about 2,469 on ComCat and about **300** on San Jacinto (§2.3). *Why it matters:* on San Jacinto the i.i.d. interval excludes zero and the block-bootstrap interval does not — the entire win/tie verdict turns on this.
2. **"The bootstrap fixes dependence, it's non-parametric."** Actually the *i.i.d.* bootstrap's SE for a mean is **exactly** `s_n/sqrt(n)`, an algebraic identity (§3.2). *Why it matters:* 4,000 i.i.d. replicates on a clustered series buy nothing but a false sense of rigour.
3. **"Blocks preserve correlation, so any block scheme will do."** Actually MBB is mean-biased at the edges, CBB fixes that but leaves the resample periodically non-stationary, and only geometric lengths make it stationary (§4.4). *Why it matters:* the differences are second-order for a mean, but you will be asked which you used and why, and "they're all the same" is a wrong answer.
4. **"`p_boot: 0.0005` means p = 0.0005."** Actually it means `p <= 2/4001`, the smallest value `B = 4000` can express (§7.2); the true p could be `1e-30`. *Why it matters:* Holm-correcting a floor value is conservative and fine, but *comparing* two claims both at 0.0005 compares two censored values and learns nothing.
5. **"The CI crosses zero, so the models tie."** Actually that is absence of evidence; equivalence needs TOST (§10). *Why it matters:* Iran's `dT` CI crosses zero and **fails** TOST even at 0.10 — the data are compatible with FlowQuake being 0.12 nats/event worse. "Tie" would be a false claim.
6. **"TOST at alpha = 0.05 uses a 95% interval."** Actually the **90%** interval, because two one-sided 5% tests correspond to a `(1-2×0.05)` two-sided interval (§10.2). *Why it matters:* 95% silently makes it level-0.025 — valid but underpowered, and your stated alpha is wrong.
7. **"Two one-sided tests need a multiplicity correction."** Actually intersection–union gives size `alpha`, not `2 alpha`, with no correction (§10.2), because the null is a *union* and you reject only when both components are rejected. *Why it matters:* people "fix" TOST by Bonferroni-ing to 0.025 each and end up with a test whose true size is far below 0.025.
8. **"McNemar p = 1.00 proves the models are equivalent."** Actually with 10 discordant days the test has ~38% power against 80/20 and ~15% against 70/30 (§11.3), and 5-5 is the least informative possible outcome. *Why it matters:* it is the same error as #5, made where the repo elsewhere insists on TOST.
9. **"A positive mean gain means winning most of the time."** Actually FlowQuake's spatial head wins **47.85%** of forward-window events with mean **+0.0666** (§12.2). *Why it matters:* the mean is the proper-score answer and is the right headline, but a reader who assumes "wins on average" implies "wins usually" will be surprised, and an operational forecaster may legitimately care about the typical day.
10. **"Bonferroni is the conservative default; Holm is an optional refinement."** Actually Holm rejects a strict superset under the same (zero) assumptions (§8.4). *Why it matters:* there is no scenario where Bonferroni is more defensible, only less powerful. The one genuine difference is that Bonferroni inverts to simultaneous intervals and Holm does not.
11. **"Holm needs independence."** Actually the proof uses only p-value validity and the union bound (§8.4). *Hochberg* needs Simes, hence positive dependence; Holm needs nothing. *Why it matters:* this question separates someone who memorised the procedures from someone who understands them.
12. **"The bootstrap CI is the truth, so four decimals is fine."** Actually the same series with different seeds gives `dTot` CIs of `[0.1006, 0.1268]` and `[0.1006, 0.1261]` (§6.2). *Why it matters:* the fourth digit is Monte Carlo noise, and since no artifact records `n_boot`, `seed` or `mean_block`, a referee recomputing sees an unexplained mismatch.

---

## Questions a professor will ask

**Q1. Derive `Var(mean)` under stationarity and define `n_eff`.** §2.1: `Var(Xbar) = (sigma^2/n)[1 + 2 sum_{k=1}^{n-1}(1-k/n) rho_k]`, from counting the `2(n-k)` pairs at each lag `k >= 1`; `n_eff = n/VIF`. For AR(1), `VIF = (1+phi)/(1-phi)`; for clumps of size `B`, `VIF = B`, provable two ways (§2.2).

**Q2. Why does the i.i.d. bootstrap fail here? Be precise.** Not "it underestimates" — it reproduces the naive SE *exactly*: `Var*(Xbar*) = s_n^2/n` conditionally on the data, because the resampled points are conditionally independent draws from the empirical *marginal* (§3.2). The plug-in needs `F_hat_n` to capture the joint law; the marginal empirical measure does not.

**Q3. Prove the stationary bootstrap's resample is stationary, and why the circular block bootstrap's is not.** Memorylessness turns the block construction into a Markov chain on `Z_n` with kernel `(1-p)·shift + p·uniform`; uniform is invariant under both components hence under the mixture, and the chain starts uniform — stationary (§4.4). With fixed `L`, `Cov*(X*_j, X*_{j+1})` is `gamma_hat_1` inside a block and 0 at a boundary, so the resample is periodically stationary with period `L`.

**Q4. What is the stationary bootstrap variance actually estimating?** A lag-window estimate of the long-run variance `sum_k gamma_k` with geometric weights `(1-p)^k` — derivable in two lines from the §4.4 Markov chain, since `Cov*(X*_j, X*_{j+k}) = (1-p)^k ghat_c(k)` (§4.5). Same target as a Newey–West HAC estimator with a different taper, which is why a Diebold–Mariano test and this bootstrap should agree. For AR(1) it collapses to `VIF_SB(L) = (1+q)/(1-q)` with `q = phi(1-1/L)`: the bootstrap sees an AR(1) with the coefficient shrunk from `phi` to `phi(1-1/L)`.

**Q5. Justify `mean_block = 50`.** *(hostile)* Honestly: it is not justified in the repository. It is a hard-coded default in five signatures, no call site overrides it, no artifact records it, and **there is no sensitivity check anywhere** (§5.3; [WORKING.md](../WORKING.md) item 16 agrees). What I can say: the MSE-optimal rate is `n^{1/3}` (derived §5.1), which is 28 at California scale, so 50 is within a factor of 2; the SB variance is a geometric lag window of scale `L` (derived §4.5), so at `L = 50` correlation beyond a few hundred events is truncated; and the closed form `VIF_SB(L) = (1+q)/(1-q)`, `q = phi(1-1/L)`, says that at ComCat's implied `phi = 0.797` a block of 50 recovers **96%** of the true SE inflation while 10 recovers only 83% (§5.4). What I *cannot* do is run the real check, because the per-event CSVs are gitignored — so I have shown the shape of the curve on a synthetic AR(1), not on FlowQuake's gains. The fix is (a) Politis & White (2004) automatic selection, (b) publish the sensitivity curve, (c) record `mean_block`, `n_boot` and `seed` in every artifact. Until then the correct claim is "we used `L = 50` and have not demonstrated insensitivity".

**Q6. Blocks of 50 *events* — minutes during an aftershock sequence, months in a quiet period. Why is an event-count block the right unit?** *(hostile)* A genuine weakness the repository never addresses. The defence: dependence in the gain series is driven by *sequence membership*, an event-count phenomenon (a sequence is `k` events regardless of duration), not a wall-clock one. The counter: background events — where the spatial head does its winning (§12.3) — are separated in time, not in event count, so a time-based block would treat them differently. The right answer is to run both, with 30-day blocks as the comparator, and report whether any decision changes. Not done.

**Q7. Your CI is a percentile interval. Why not BCa?** Percentile is first-order accurate and transformation-respecting, enough when the bootstrap distribution is roughly symmetric and `n_eff` is large. Neither condition is comfortable: `n_eff ≈ 300` for San Jacinto, and `dS` is strongly right-skewed (sub-50% win rate, positive mean). BCa is the right upgrade, but its acceleration is normally jackknife-estimated and the delete-one jackknife is invalid under dependence — it needs a **delete-block** jackknife (Künsch 1989). That is the work item, and the claims it could move are San Jacinto `dT`, Japan `dT` and Iran's `dTot`, whose CI lower endpoint is 0.0098 (§6.2).

**Q8. Compute Holm for six p-values by hand and tell me what the running max is for.** Worked Example 1: multipliers `6,5,4,3,2,1` down the sorted list; the running max enforces monotonicity so a hypothesis is never rejected while one with a smaller p-value is retained — which is what makes the step-down stopping rule coherent. All six reproduce to the stored digit.

**Q9. Prove Holm controls FWER under arbitrary dependence.** §8.4. Let `r` be the first rank at which a true null is rejected; step-down means all earlier rejections are false nulls, so all `m_0` true nulls sit at ranks `>= r`, giving `m - r + 1 >= m_0`. Rejection needs `p_(r) <= alpha/(m-r+1) <= alpha/m_0`, so a type-I error implies `min_{i in I_0} p_i <= alpha/m_0`, and the union bound closes it at `alpha`. Only p-value validity and Boole's inequality are used.

**Q10. Why Holm and not Hochberg, which is more powerful?** Hochberg needs the Simes inequality, which holds under independence and PRDS but can fail under general dependence. The six regional tests share one architecture, one codebase and overlapping seeds; I have not characterised that dependence, and Holm requires me not to (§8.5).

**Q11. Your family is six regions. Why not six regions × three metrics × three seeds?** *(hostile)* Seeds are not separate hypotheses when you report the mean over them — though I concede the committed artifacts are single-seed while the manuscript says 3-seed (§9.2), and that must be regenerated. Metrics are the substantive question: `dTot = dT + dS` exactly, so two degrees of freedom, and the two families answer two pre-stated questions in two paper sections. The strongest defence is that losses are reported: `per_region.California.dTot_mean = -0.3107, "loss"` sits in the same file as the wins. And here is the sensitivity analysis (§9.3): pooling all twelve into one Holm family changes **one** of twelve verdicts — Iran's total gain goes from `p_holm = 0.0185` to `0.0740` and stops being significant. So [MANUSCRIPT.md](../MANUSCRIPT.md) §4.4's "Holm-adjusted p ≤ 0.019 across the family" is family-dependent, and I would report the pooled-12 column alongside.

**Q12. Should the CSEP McNemar test be in the family?** Arguably yes — it is a hypothesis test supporting a headline claim and is currently in no family at all. But corrections make it *easier* to fail to reject, and it supports a **null** conclusion, so adding it to an FWER family pushes in the authors' favour, not against. The right remedy is an equivalence bound, not a correction (§9.4, §11.4).

**Q13. Derive TOST and explain the 90% interval.** §10.2. The null is the union `{theta <= -delta} ∪ {theta >= +delta}`; reject only if both one-sided level-alpha tests reject; by intersection–union the size is `alpha`, because any `theta` in the null lies in one component and rejection requires rejecting that component. The one-sided level-alpha bounds are the endpoints of a `(1-2alpha)` two-sided interval, so `alpha = 0.05` gives 90%. [stats.py:153](../flowquake/stats.py#L153) uses percentiles 5 and 95 — correct.

**Q14. Defend the 0.05 nats/event margin.** *(hostile)* Partly defensible. In favour: `exp(0.05) = 1.051`, a 5% per-event density difference, and 5.4% of ETAS's entire temporal edge over the Poisson floor (0.9217 nats/event, from `runs/n1_density/eval_test.json` → `baselines`). Against: the same margin is applied to `dTot`, where ETAS's edge over Poisson is 6.0064 nats/event, making 0.05 only 0.83% of the available skill — six and a half times laxer in relative terms on that scale. A margin should be pre-registered per metric and scaled to its dynamic range; the repo states no derivation for 0.05/0.10 at all. I would replace them with 5%/10% of the ETAS-minus-Poisson gap per metric (§10.4). Also worth stating: `0.05 × 21889 = 1094` nats aggregate, so "equivalent at 0.05" is about per-event sharpening, not about the models being indistinguishable in aggregate.

**Q15. Derive McNemar and explain the conditioning.** §11.1. `H0: p10 = p01` after the concordant cells cancel from the marginals; conditioning on `d = n10 + n01` gives `n10 | d ~ Bin(d, p10/(p10+p01))`, which under `H0` is `Bin(d, 1/2)` **free of all nuisance parameters** — a similar test. Exact two-sided p is twice the smaller binomial tail, capped at 1. Concordant pairs inform only `p11` and `p00`, unconstrained under both hypotheses.

**Q16. You report McNemar p = 1.00 and conclude "statistically indistinguishable". Justify that.** *(hostile)* I cannot, as stated. With `d = 10` the test has 37.6% power against a true 80/20 discordance split and 15.0% against 70/30; 5-5 is the least informative of the six possible outcomes; and reaching `p < 0.05` at all requires a 9-1 split (§11.3). The defensible statements are that the head's *standalone* CSEP pass rates are strong on their own (N 95/100, S 79/85, M 90/92), and that we found no evidence the spatial-likelihood gain degraded calibration. The claim I should be making is "the gain cost nothing detectable in consistency", not "indistinguishable". It is also inconsistent that the repo demands TOST for every likelihood tie and accepts a bare non-rejection here; the fix is an exact equivalence bound on the discordance ratio, which at 5/10 gives a Clopper–Pearson 90% interval of `[0.222, 0.778]` — it cannot exclude a materially worse model.

**Q17. Your spatial head wins on 47.85% of events. Isn't that a losing model?** No — log-likelihood is a proper score, the mean determines which model to bet on, and the mean gain is `+0.0666` with CI `[0.0553, 0.0784]` excluding zero. The sub-50% win rate says the gain is *concentrated*: solving `0.4785 W - 0.5215 L = 0.0666` with `L = 0.05` gives `W = 0.194`, so the average win is 3.9x the average loss (§12.3). The mechanism is that ETAS's power-law kernel is already near-optimal in the dense aftershock cloud and near-blind for background events far from any recent parent, which is where the head's smoothed-seismicity background pays. **The legitimate criticism** is that this needs a per-stratum decomposition to confirm, and `runs/n1_density/spatial_gap_decomp.json` holds only a background-vs-triggered split with no distance strata — [WORKING.md](../WORKING.md) lists it under "needs hardware or data".

**Q18. How many times has the test set been looked at?** *(hostile)* Order 70+, as a lower bound: 65 committed `eval_test.json` files across 78 run directories, plus 8 test evaluations in the `h`-ablation alone. The repository says so itself — [scripts/train_neural_etas.py](../scripts/train_neural_etas.py)'s docstring reads "do not describe these runs as a test-scored-once protocol". The honest framing is that 2007–2020 functioned as a development set for ablation and selection, and that the 2020–2026 forward window (10,187 events, frozen model) is the only genuinely held-out evaluation. That window replicates all three gains (`dT +0.0574`, `dS +0.0666`, `dTot +0.1241 [0.1035, 0.1455]`), which is the real answer to the forking-paths objection — and it is still retrospective, not registered.

**Q19. What fraction of the paper's numbers actually trace to an artifact?** Of 142 traced rows covering 134 distinct claims: 114 match exactly or to rounding (63 exact, 51 to rounding), 2 are ambiguous between two committed artifacts, 8 distinct claims are **contradicted** by their artifact, and 12 have **no committed backing** ([WORKING.md](../WORKING.md) "Current state"; per-claim map in [results/CLAIMS.md](../results/CLAIMS.md)). The worst case is [MANUSCRIPT.md:328-331](../MANUSCRIPT.md)'s "individually significant in every era", which `runs/prospective.json` contradicts — it stores no per-window CI or p-value at all, and Chile is positive in only 10 of 19 180-day windows (`bins_dT_positive_frac = 0.5263`).

**Q20. What single statistical change would most strengthen this work?** Not a better test — a registered prospective forecast (rung 4 of [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md)). Every device in this chapter answers "given these data, what can I say?"; none answers "did anyone see the answers first". Second: record `mean_block`, `n_boot` and `seed` in every artifact and publish the block-length sensitivity curve — a day of work that closes the largest methodologically *available* hole.

---

## Further reading

- **Efron, B. (1979), "Bootstrap methods: another look at the jackknife", *Annals of Statistics*.** The origin. Read it for the plug-in principle stated cleanly, before anyone built machinery on top.
- **Künsch, H. R. (1989), "The jackknife and the bootstrap for general stationary observations", *Annals of Statistics*.** The moving-block bootstrap *and* the delete-block jackknife in one paper — the latter is what you need for BCa intervals on dependent data (§6.1).
- **Politis, D. N. & Romano, J. P. (1994), "The stationary bootstrap", *JASA*.** The method this repository uses. Read it for the stationarity theorem and the variance formula of §4.5 (which §4.5 re-derives from scratch).
- **Lahiri, S. N. (1999), "Theoretical comparisons of block bootstrap methods", *Annals of Statistics*, and Nordman, D. J. (2009), "A note on the stationary bootstrap's variance", *Annals of Statistics*.** The efficiency ranking of the block bootstraps, and the correction to it. Lahiri put the SB last; Nordman showed its variance actually matches the non-overlapping block bootstrap's. Knowing both is the difference between reciting a ranking and understanding it.
- **Politis, D. N. & Romano, J. P. (1992), "A circular block-resampling procedure for stationary data", in LePage & Billard (eds), *Exploring the Limits of Bootstrap*, Wiley.** Short; the wrap-around fix for the MBB mean bias, which the stationary bootstrap inherits.
- **Politis, D. N. & White, H. (2004), "Automatic block-length selection for the dependent bootstrap", *Econometric Reviews*, with the correction in Patton, Politis & White (2009), same journal.** The answer to "how did you choose the block length?". Implemented as `arch.bootstrap.optimal_block_length`. Cite both — the 2009 correction matters.
- **Lahiri, S. N. (2003), *Resampling Methods for Dependent Data*, Springer.** The reference book: block bootstrap consistency, the `n^{1/3}` rate, and the relative efficiency of MBB vs SB — the honest counterweight to "the stationary bootstrap is strictly better".
- **Hall, P., Horowitz, J. L. & Jing, B.-Y. (1995), "On blocking rules for the bootstrap with dependent data", *Biometrika*.** The source of the three different optimal rates — `n^{1/3}` for variance or bias, `n^{1/4}` for a one-sided distribution function, `n^{1/5}` for a two-sided one. The lesson is that "the" optimal block length does not exist independently of what you are estimating.
- **Diebold, F. X. & Mariano, R. S. (1995), "Comparing predictive accuracy", *JBES*.** The forecast-comparison framing of exactly this problem — a serially correlated loss differential — with a HAC variance instead of a bootstrap. Read it to see that FlowQuake's `dT` series is a well-studied object with a well-studied test, and to have an independent cross-check available.
- **Holm, S. (1979), "A simple sequentially rejective multiple test procedure", *Scandinavian Journal of Statistics*.** Four pages; the §8.4 proof is essentially the paper's. Worth the original to see how little is assumed.
- **Benjamini, Y. & Hochberg, Y. (1995), "Controlling the false discovery rate", *JRSS-B*.** For knowing when FDR is the right target — and therefore being able to explain why it is not the right target here.
- **Berger, R. L. & Hsu, J. C. (1996), "Bioequivalence trials, intersection-union tests and equivalence confidence sets", *Statistical Science*.** The rigorous treatment of why TOST has size alpha with no correction. Schuirmann (1987, *Journal of Pharmacokinetics and Biopharmaceutics*) is the original proposal; Lakens (2017, *Social Psychological and Personality Science*) is the readable modern tutorial.
- **Gelman, A. & Loken, E. (2013/2014), "The garden of forking paths" (working paper) / "The statistical crisis in science", *American Scientist*.** Why correcting over the *reported* tests does not fix the problem, and why pre-registration is a different kind of evidence.
- **Gneiting, T. & Raftery, A. E. (2007), "Strictly proper scoring rules, prediction, and estimation", *JASA*.** Why the mean log-likelihood gain is the right thing to test at all, and why the win rate is not.

---

*Repo sources read for this chapter:* [flowquake/stats.py](../flowquake/stats.py), [scripts/stats_hardening.py](../scripts/stats_hardening.py), [scripts/total_win_summary.py](../scripts/total_win_summary.py), [scripts/audit_readiness.py](../scripts/audit_readiness.py), [scripts/train_neural_etas.py](../scripts/train_neural_etas.py), [tests/test_stats.py](../tests/test_stats.py), `runs/stats_hardening.json`, `runs/total_win.json`, `runs/replacement_readiness.json`, `runs/n1_density/eval_test.json`, `runs/n1_density/csep_head/csep_results.json`, `runs/csep_h2h_etas/csep_results.json`, `runs/csep_h2h_fq/csep_results.json`, [flowquake/csep_forecast.py](../flowquake/csep_forecast.py), `runs/multiregion_master.json`, `runs/prospective.json`, `runs/n1_density/spatial_gap_decomp.json`, `runs/ablation_h/memorization_figure.json`, `runs/neural_etas/ComCat_25/per_event_forward_full.json`, [WORKING.md](../WORKING.md), [MANUSCRIPT.md](../MANUSCRIPT.md), [README.md](../README.md), [REPLACEMENT_READINESS.md](../REPLACEMENT_READINESS.md), [STACK.md](../STACK.md), [results/CLAIMS.md](../results/CLAIMS.md).
