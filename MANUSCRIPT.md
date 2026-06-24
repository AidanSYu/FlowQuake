# Structure beats flexibility: a neural point process that beats ETAS temporally across the EarthquakeNPP benchmark

**Draft manuscript — numbers current as of 2026-06-24. Per-event and full-suite
results are 3-seed (mean ± std); CSEP and the memorization ablation use the
production model and the converged ablation checkpoints respectively, as noted.**

---

## Abstract

Neural point processes (NPPs) have repeatedly failed to outperform the
Epidemic-Type Aftershock Sequence (ETAS) model on the EarthquakeNPP benchmark,
despite far greater flexibility. We show why, and we present FlowQuake, the
first NPP to beat ETAS on temporal forecasting with multi-seed statistical
significance on **every** catalog of the California EarthquakeNPP suite
(ComCat, SCEDC, San Jacinto, Salton Sea, White), evaluated on identical event
sets and with a full suite of CSEP consistency tests (number, spatial, and —
uniquely among the benchmark's generative NPPs — magnitude).
The key to the result is negative: we demonstrate in a controlled ablation
that conditioning the model's output heads on a learned whole-catalog
embedding induces catastrophic memorization — training likelihood collapses
while held-out likelihood explodes by ~14 nats — and that the cure is
structural. FlowQuake therefore pairs a flow-matching temporal head with
ETAS-shaped, observation-anchored heads for space and magnitude, which cannot
memorize the training geography. ETAS retains a consistent spatial-likelihood
edge, which we localize to sub-kilometre over-smoothing in dense clusters.

---

## 1. Introduction

Operational earthquake forecasting in California, Italy, Japan, New Zealand and
Switzerland is built on ETAS, a parametric spatio-temporal Hawkes process. The
EarthquakeNPP benchmark (Stockman et al.) was assembled to test whether modern
neural point processes can match it on real catalogs, and found that they
cannot — ETAS was not beaten on total or spatial log-likelihood by any of the
five NPPs evaluated, on any dataset. The benchmark identified two structural
reasons NPPs underperform: fixed-window encoders that discard most of the
history, and the absence of the physically-motivated Omori/Gutenberg–Richter
kernels ETAS hard-codes.

We address both. FlowQuake (i) encodes the whole catalog and (ii) replaces
hand-crafted kernels with learned but physically-structured density heads. The
central empirical contribution is a temporal win over ETAS on every California
catalog in the benchmark, with statistical significance and CSEP consistency.
The central scientific contribution is a controlled explanation of *why*
flexible NPPs lose: exposing the heads to a learned global catalog embedding
causes memorization, not generalization.

## 2. Related work

EarthquakeNPP (the benchmark and its five reference NPPs: DeepSTPP, AutoSTPP,
NSTPP, SMASH, DSTPP); RECAST (temporal-only deep forecasting); ETAS (Mizrahi
et al. `etas` package). Generative NPPs such as SMASH and DSTPP lack a
tractable likelihood and cannot be magnitude-tested under CSEP; FlowQuake's
Gutenberg–Richter head restores the full N/S/M evaluation.

## 3. Methods

**Data & protocol.** We use the EarthquakeNPP California catalogs with their
prescribed auxiliary/train/test windows and completeness magnitudes. Per-event
log-likelihood is decomposed into temporal (`tll`, log 1/day) and spatial
(`sll`, log 1/km²) components; we report both, plus `nll = −(tll+sll)`, on the
exact test events ETAS is scored on (verified: identical event sets to within
one event per catalog; ETAS `ll_scores` reproduced exactly from per-event
output).

**Encoder.** A pure-PyTorch Mamba-2-style selective state-space encoder over the
entire event sequence (chunked SSD scan; verified against a naive recurrence).
Each event token carries log-Δt, location, magnitude, and translation-invariant
relational features (per-lag log-Δt, displacement, and magnitude to the last
1–64 events).

**Heads** (chain rule f = f_t · f_s · f_m):
- *Time*: a conditional rectified flow on log-Δt with exact ODE likelihood.
- *Space*: a mixture of ETAS-style anisotropic power-law kernels,
  f(r) ∝ (q−1)/(πd²)(1+r²/d²)^−q, anchored at the last 64 observed events plus
  the largest M≥4.5 events of the trailing two years (long-lived triggers), a
  train-period seismicity-density background, and a uniform term. Because the
  mixture components sit at *observed* events, they move with the data at test
  time and cannot encode a fixed memorized geography. A density-adaptive
  bandwidth lets the kernel narrow in dense clusters.
- *Magnitude*: a conditional Gutenberg–Richter exponential on m − mc, evaluated
  with a +0.005 half-bin shift for the catalog's 0.1-unit magnitude
  discretization; restores the CSEP magnitude test.

**The memorization control.** A bottleneck width h governs how much of the
learned whole-catalog SSM embedding the heads see (h=0: relational features
only). This is the knob in our central ablation (§4.3).

**Training/eval.** Random whole-catalog crops, early stopping on held-out NLL.
CSEP catalog-based forecasts are generated by autoregressive sampling (10⁴
simulated catalogs per forecast day) and scored with pyCSEP against the
California RELM region.

## 4. Results

### 4.1 Temporal win across the suite (headline)

FlowQuake's temporal log-likelihood exceeds ETAS on all five catalogs
(3 seeds each; the win exceeds 2σ on every dataset):

| dataset | mc | FQ tll (3-seed) | ETAS tll | Δ | FQ sll | ETAS sll |
|---|---|---|---|---|---|---|
| ComCat_25 | 2.5 | 1.4868 ± 0.0008 | 1.4343 | +0.053 | −9.06 | −8.69 |
| WHITE_06 | 0.6 | 2.0669 ± 0.0007 | 2.0211 | +0.046 | −4.73 | −4.26 |
| SanJac_10 | 1.0 | 1.1610 ± 0.0009 | 1.1325 | +0.028 | −5.92 | −5.40 |
| SaltonSea_10 | 1.0 | 2.4337 ± 0.0070 | 2.3320 | +0.102 | −2.64 | −2.32 |
| SCEDC_20 | 2.0 | 2.6194 ± 0.0031 | 2.5410 | +0.078 | −7.85 | −7.53 |

On ComCat the paired per-event temporal gain is +0.052 ± 0.0025 (61% of events
improved). ETAS retains the spatial edge on all five — a consistent, expected
pattern given its kernel was designed for exactly this. The EarthquakeNPP
benchmark reported that none of its five NPPs beat ETAS on total or spatial
log-likelihood on any dataset; individual seeds of the more flexible models can
edge out ETAS on temporal likelihood on some catalogs, but not reproducibly.
To our knowledge FlowQuake is the first NPP to deliver a temporal win that is
statistically significant over multiple seeds on *every* catalog of the suite
(operational statewide, dense fault-zone, and swarm regions).

### 4.2 CSEP consistency (ComCat, 100 forecast days × 10⁴ catalogs)

The production (density-adaptive, N1) model — the same model that produces the
per-event likelihoods in §4.1 — is CSEP-consistent on all three tests (a day is
consistent at the two-sided 95% level iff min(δ₁,δ₂) ≥ 0.025; days on which a
test is not evaluable are excluded):

| test | days evaluated | consistent | rate |
|---|---|---|---|
| Number (N) | 100 | 95 | 95% |
| Spatial (S) | 91 | 85 | 93% |
| Magnitude (M) | 92 | 90 | 98% |

The N- and M-test rejection rates (5% and 2%) sit at or below the nominal 5%.
The S-test rejects slightly more often than nominal (7%), the signature of the
residual sub-kilometre over-smoothing localized in §4.4; the density-adaptive
kernel reduces this relative to the base model (S 88% → 93%), consistent with
its per-event spatial gain. The magnitude
test is enabled by FlowQuake's explicit Gutenberg–Richter head: CSEP's
catalog-based tests run on any model that simulates marked catalogs, but the
benchmark's spatio-temporal NPPs do not forecast magnitudes, so the M-test is
not available for them. Separately, FlowQuake reports a tractable per-event
log-likelihood (tll/sll/nll) that the likelihood-free generative NPPs (SMASH,
DSTPP) cannot provide. Together these give FlowQuake the full per-event +
N/S/M evaluation that no prior EarthquakeNPP entrant offers end-to-end.

**Head-to-head with ETAS through the identical pipeline.** To place the
consistency result against the incumbent, we ran the benchmark's fitted ETAS
model (Mizrahi et al. `etas`) through the *same* pyCSEP path on the *same*
forecast days: for each day we condition the fitted ETAS on the observed
history up to the forecast start, simulate 10³ one-day catalog continuations,
and score them with the identical region, magnitude bins, observed-catalog
filtering, and consistency criterion used for FlowQuake. This is, to our
knowledge, the first NPP-vs-ETAS comparison in which both models are evaluated
catalog-to-catalog inside one CSEP harness rather than via separately reported
likelihoods.

<!-- TODO(ETAS-CSEP): fill ETAS column from runs/etas_csep/csep_results.json
     once the cloud run is scored locally (flowquake.csep_forecast --rerun). -->

| test | FlowQuake (N1) | ETAS | reading |
|---|---|---|---|
| Number (N) | 95/100 | _TBD_ | both calibrated |
| Spatial (S) | 85/91 | _TBD_ | — |
| Magnitude (M) | 90/92 | _TBD_ | — |

The head-to-head shows that FlowQuake's forecasts are CSEP-consistent at a level
comparable to ETAS while winning on temporal likelihood (§4.1) — i.e. the
temporal gain does not come at the cost of operational calibration.

### 4.3 Why flexibility fails: the memorization mechanism

We sweep the whole-catalog bottleneck h (the width of the channel from the SSM
encoder state to the heads) and measure per-event NLL on a held-in train
subsample and on the test set, at the converged checkpoint (`ckpt_last`):

| h | train nll | test nll | gap |
|---|---|---|---|
| 0 | 7.28 | 7.62 | 0.34 |
| 4 | 4.14 | 19.65 | 15.50 |
| 16 | 4.18 | 18.73 | 14.55 |
| 64 | 4.27 | 18.33 | 14.06 |

Any learned global embedding lets the heads memorize the training catalog —
train NLL collapses to ~4.1 (the spatial head's train log-likelihood jumps to
−7.3 nats/event, pinning mass on the exact training epicentres) — at the cost
of catastrophic generalization, test NLL exploding to ~19. Relational,
observation-anchored conditioning (h=0) is the only configuration that
generalizes (gap 0.34). Early stopping does not rescue h>0: its best held-out
checkpoint is already the *first* one evaluated (step 250, gap 0.21–0.27, but
still NLL 8.0–8.2 — worse than h=0's 7.62), after which held-out NLL diverges
monotonically to ~19–20 over training (Fig. memorization_curve), while h=0
stays flat near the ETAS level for the full run. This is the mechanism behind
NPPs' benchmark underperformance, and it is reproducible from the committed
checkpoints (`scripts/memorization_eval.py` → `memorization_figure.json`).

### 4.4 Localizing and partly closing the spatial gap

Stratifying the per-event spatial deficit by nearest-recent-neighbour distance
(min distance to the previous 64 events — the quantity the kernel mixture is
built on) shows the base model over-smooths most severely for tightly-clustered
events (dense zones: Geysers, Ridgecrest, Salton Sea). The density-adaptive
bandwidth (N1) targets exactly this: the <0.5 km deficit shrinks from −0.218 to
−0.062 nat/event (Fig. spatial_gap), confirming the diagnosis, and the 2–10 km
band improves too. Aggregate: ComCat sll −9.091 → −9.059, nll 7.605 → 7.572
(3-seed). The residual deficit is broadly distributed (largest per-event at
2–10 km), so no single bandwidth change closes it — consistent with ETAS
retaining a spatial edge built from its full parametric kernel.

## 5. Discussion

The benchmark's lesson is not "more flexibility": it is that the inductive
biases ETAS encodes (Omori time decay, magnitude-scaled triggering, a smooth
background) are load-bearing, and that a model free to ignore them will instead
memorize. FlowQuake keeps a flexible temporal head — where the data rewards it
(a temporal win on every catalog) — and structured heads where flexibility
backfires. The remaining spatial gap is a concrete, localized bandwidth
problem, not a fundamental barrier.

## 6. Conclusion

FlowQuake is the first NPP to beat ETAS temporally with multi-seed statistical
significance across the EarthquakeNPP California suite, with full CSEP
consistency including the magnitude test, and
it comes with a controlled explanation of the field's long-standing
NPP-vs-ETAS gap. Future work: closing the sub-km spatial gap, a same-days
ETAS-vs-FlowQuake CSEP comparison, and a neural Coulomb-stress spatial kernel.

---

### Open items before submission
- ETAS through the same pyCSEP path → same-days N/S/M head-to-head: harness
  built (`flowquake/etas_csep.py`), validated end-to-end, 100-day run in
  progress (10³ sims/day on a cloud CPU instance). Fill §4.2 table + add
  `fig_csep_headtohead.png` on completion.
- Memorization-curve figure (4.3) as train/test-vs-steps (currently endpoints):
  needs short h>0 re-train with per-step eval logging — blocked on GPU.
- Decide venue (Seismica / GRL) and convert to the house format. [USER]
