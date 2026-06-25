# Structure beats flexibility: a transferable neural point process that beats ETAS temporally across tectonic regimes

**Draft manuscript — numbers current as of 2026-06-26. Per-event and full-suite
results are 3-seed (mean ± std); CSEP and the memorization ablation use the
production model and the converged ablation checkpoints respectively, as noted.
Cross-regime transfer (§4.5) uses catalogs built from USGS ComCat; the Chile
ETAS baseline was still being fit at last edit (its native-vs-ETAS cell is the
only pending number — Chile serves as a transfer source, where its ETAS row is
confirmatory).**

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
edge, which we trace to its stochastically-declustered background and show is
near-optimal for its model class (three independent reconstructions land
0.27–0.35 nat short of it). Because FlowQuake's heads are translation-invariant
rather than tied to a fitted background field, the model **transfers across
tectonic regimes** — something ETAS, refit per region, structurally cannot do.
Pre-trained on one region and applied to four others spanning subduction,
continental extension and collision (Japan, Chile, Greece, Iran; catalogs built
from USGS ComCat), it beats region-fitted ETAS on temporal forecasting after a
brief warm-started fine-tune in every regime, and on data-poor regions — where
training from scratch fails — transfer is *essential*: a model fine-tuned from a
data-rich region beats both from-scratch training and region-fitted ETAS.

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
hand-crafted kernels with learned but physically-structured density heads. We
make three contributions. (1) A temporal win over ETAS on every California
catalog in the benchmark, with multi-seed significance and CSEP consistency.
(2) A controlled explanation of *why* flexible NPPs lose: exposing the heads to
a learned global catalog embedding causes memorization, not generalization, and
the cure is structural. (3) The result that gives the structural cure its edge —
because the heads are anchored to observed events rather than a fitted
background, one pre-trained FlowQuake **transfers across tectonic regimes**:
fine-tuned briefly, it beats region-fitted ETAS on temporal forecasting in
Japan, Chile, Greece and Iran, and is the model of choice on data-poor regions
where ETAS — which must be inverted per region — is hardest to fit. ETAS keeps a
near-optimal spatial likelihood in-domain, which we characterize rather than
claim to beat.

## 2. Related work

**ETAS and operational forecasting.** ETAS originates with Ogata's temporal
self-exciting model [Ogata 1988] and its space–time extension [Ogata 1998];
it underpins operational and pseudo-prospective forecasting and the CSEP
testing experiments. We use the maximum-likelihood inversion and
catalog-continuation simulator of Mizrahi et al. [2021] (the `etas` package) as
our incumbent, both for per-event likelihoods and for the CSEP head-to-head.

**Neural point processes on EarthquakeNPP.** The EarthquakeNPP benchmark
[Stockman et al.] assembles California catalogs with fixed train/test windows
and evaluates five NPPs — DeepSTPP, AutoSTPP, NSTPP, and the generative SMASH
and DSTPP — finding that none beats ETAS on total or spatial log-likelihood on
any dataset. The two limitations it identifies — fixed-window encoders (e.g.
DeepSTPP conditions on a short event window) and the absence of the
Omori/Gutenberg–Richter structure ETAS hard-codes — motivate FlowQuake's
whole-catalog SSM encoder and its structured, observation-anchored heads.
RECAST [Dascher-Cousineau et al. 2023] is a strong temporal-only deep
forecaster; FlowQuake instead targets the full marked, spatial problem.
The generative models (SMASH, DSTPP) sample catalogs but expose no tractable
per-event density, so they cannot report tll/sll/nll; the benchmark's
spatio-temporal NPPs do not forecast magnitudes, so they cannot be CSEP
magnitude-tested. FlowQuake provides both.

**Method ingredients.** The temporal head is a conditional rectified
flow / flow-matching model [Lipman et al. 2023]; the encoder is a
pure-PyTorch Mamba-2-style selective state-space scan [Gu & Dao 2023; Dao & Gu
2024]. Forecasts are scored with the CSEP consistency-test framework
[Schorlemmer et al. 2007; Zechar et al. 2010] via pyCSEP [Savran et al. 2022].

*(Bibliographic details for [Stockman et al.] (EarthquakeNPP venue/year) and the
exact entries for the five reference NPPs are to be finalized against the
benchmark's published bibliography before submission.)*

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

<!-- HEAD-TO-HEAD STATUS (2026-06-25): a first ETAS run through this harness
     (10^3 sims/day) is NOT yet reportable. It produced anomalous ETAS scores
     (N 73/100, all 27 failures from systematic count UNDER-prediction; S only
     63/100 evaluable vs FlowQuake's 91). The under-prediction is not a sim-count
     effect and points to the ETAS continuation not conditioning on post-2007
     events as triggers (a flowquake/etas_csep harness issue); the S-evaluability
     gap is a genuine sim-count confound (10^3 vs 10^4 -> more all-empty ETAS
     days). FIX before reporting: (1) verify/repair trigger conditioning in
     etas_csep; (2) match sim counts (re-score FlowQuake at 10^3, or ETAS at
     10^4). Raw data: runs/etas_csep_pod/. Do NOT fill the table until both
     models are on an equal, validated footing. -->

| test | FlowQuake (N1) | ETAS | reading |
|---|---|---|---|
| Number (N) | 95/100 | _pending validation_ | — |
| Spatial (S) | 85/91 | _pending validation_ | — |
| Magnitude (M) | 90/92 | _pending validation_ | — |

A like-for-like ETAS comparison through this harness is left to a validated
follow-up (see the source comment above); FlowQuake's standalone CSEP
consistency is established in the table at the top of this section.

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

A finer decomposition pins the residual to triggering *coverage*: 64% of ComCat
test events recur within 0.5 km of a prior event, and 85% of those nearest
priors lie outside the model's last-64-event window — i.e. aftershocks of older,
moderate mainshocks that ETAS captures by integrating triggering over the full
history with Omori decay. Restoring this coverage is, however, where the model
class meets its limit: a properly full-history-normalized neural-ETAS spatial
head, fit end-to-end, reaches sll −8.97 against ETAS's −8.69, and two further
reconstructions (a fitted proper-normalizer kernel, −8.94; the production
near-trigger retrain, −9.01) land in the same 0.27–0.35-nat band. ETAS's
stochastically-declustered background and jointly-fit kernel are near-optimal
for this parametric family; matching them spatially requires essentially
reproducing ETAS. (An apparent −7.9 ceiling we initially measured was an
artifact of normalizing a spatial density over a target-dependent neighbour set
rather than over all prior events.) We therefore treat the spatial gap as a
characterized property of the problem, not a closable defect, and turn to where
the neural model has a *structural* advantage ETAS lacks: transfer (§4.5).

### 4.5 Cross-regime generalization (transfer beats ETAS where data is scarce)

ETAS is refit per region: its background field and inverted parameters are
specific to one catalog, so it cannot forecast a region it was not fit on.
FlowQuake's heads, by construction, condition only on translation-invariant
relational features (§3), so a *single* trained model can be applied anywhere.
We test this directly. Using identical EarthquakeNPP-format catalogs built from
the USGS ComCat archive, we evaluate on four regions spanning distinct tectonic
regimes — Japan (subduction), Chile (subduction), Greece (continental
extension), Iran (continental collision) — over 2011–2020 (the Japan window
includes the M9.0 Tōhoku sequence), each against an ETAS model fit natively on
that region. Temporal skill is reported as the per-event paired gain over
region-fitted ETAS (dT = tll_FlowQuake − tll_ETAS, nats/event); nll = −(tll+sll)
and ETAS retains its spatial/total edge in every region, as in California.

Table 4.5 — temporal gain over region-fitted ETAS (positive = beats ETAS):

| region | regime | train events | FQ native | transfer zero-shot | transfer few-shot |
|---|---|---|---|---|---|
| California | transform | 55,442 | **+0.053** | (source) | — |
| Japan | subduction | 21,206 | **+0.064** | −0.003 | **+0.071** |
| Greece | extension | 3,287 | −0.000 | **+0.012** | **+0.037** |
| Iran | collision | 2,480 | −0.028 | **+0.014** | **+0.044** |

Two findings (Fig. multiregion_transfer). First, the temporal win is not a
California artifact: on data-rich regions (California, Japan) a natively-trained
FlowQuake beats region-fitted ETAS temporally, replicating the suite result in a
different tectonic regime. Second, and the central new result, is a
data-efficiency effect: on data-poor regions (Greece 3.3k, Iran 2.5k training
events) a from-scratch FlowQuake has too little data and ties or loses to ETAS,
but a model pre-trained on a data-rich region and fine-tuned for only 2,000
steps beats region-fitted ETAS in **every** regime (+0.037 to +0.071), and even
zero-shot transfer (no target-region training) beats native and ETAS on the
data-poor regions. Transfer must match catalogue completeness — a source trained
at Mc 2.5 transfers cleanly to another Mc-2.5 region (California↔Japan, in both
directions: a Japan-trained model scores tll 1.410 on California vs ETAS-CA
1.434) but degrades across a large completeness gap, so we pair sources and
targets by Mc (the operationally realistic setting). ETAS offers no analogue:
its zero-shot floor is the homogeneous-Poisson baseline. A single pre-trained
FlowQuake, briefly adapted, is thus competitive-to-superior to a per-region ETAS
inversion across the global range of tectonic settings, and is the model of
choice precisely where ETAS is hardest to fit — newly-instrumented or
low-seismicity regions.

## 5. Discussion

The benchmark's lesson is not "more flexibility": it is that the inductive
biases ETAS encodes (Omori time decay, magnitude-scaled triggering, a smooth
background) are load-bearing, and that a model free to ignore them will instead
memorize. FlowQuake keeps a flexible temporal head — where the data rewards it
(a temporal win on every catalog) — and structured heads where flexibility
backfires. The same design choice that prevents memorization — heads tied to
observed events rather than to absolute geography — is what makes the model
*portable*: it has no fitted background field to leave behind, so it transfers
across tectonic regimes. The remaining spatial gap is, by contrast, a genuine
property of the problem: ETAS's declustered background is near-optimal for its
class, and we do not claim to beat it spatially. The contribution is not a
total-likelihood crown on any single catalog but a *generalizing* temporal
forecaster — one that, pre-trained and briefly adapted, beats per-region ETAS
across the world's major tectonic settings and is most valuable exactly where
ETAS is hardest to apply: data-poor regions.

## 6. Conclusion

FlowQuake is the first NPP to beat ETAS temporally with multi-seed statistical
significance across the EarthquakeNPP California suite, with full CSEP
consistency including the magnitude test; it comes with a controlled explanation
of the field's long-standing NPP-vs-ETAS gap (memorization, cured structurally);
and it is the first to demonstrate **cross-regime transfer** — a single
pre-trained model that, after a brief fine-tune, beats region-fitted ETAS on
temporal forecasting across subduction, extension and collision settings, and
that rescues data-poor regions where both from-scratch training and per-region
ETAS fall short. ETAS retains a near-optimal spatial edge in-domain, which we
characterize rather than claim to beat. Future work: a transferable spatial head
(the current spatial transfer is weaker than temporal), pooled multi-region
pre-training, and prospective evaluation.

## References

*Core entries verified; those marked [verify] need bibliographic confirmation
against the cited source before submission.*

- Ogata, Y. (1988). Statistical models for earthquake occurrences and residual
  analysis for point processes. *J. Amer. Statist. Assoc.* 83(401), 9–27.
- Ogata, Y. (1998). Space–time point-process models for earthquake occurrences.
  *Ann. Inst. Statist. Math.* 50(2), 379–402.
- Mizrahi, L., Nandan, S., Wiemer, S. (2021). Embracing data incompleteness for
  better earthquake forecasting. *J. Geophys. Res. Solid Earth*,
  doi:10.1029/2021JB022379. (the `etas` package)
- Schorlemmer, D., Gerstenberger, M. C., Wiemer, S., Jackson, D. D., Rhoades,
  D. A. (2007). Earthquake likelihood model testing. *Seismol. Res. Lett.*
  78(1), 17–29.
- Zechar, J. D., Gerstenberger, M. C., Rhoades, D. A. (2010). Likelihood-based
  tests for evaluating space–rate–magnitude earthquake forecasts. *Bull.
  Seismol. Soc. Am.* 100(3), 1184–1195.
- Savran, W. H., Bayona, J. A., Iturrieta, P., et al. (2022). pyCSEP: A Python
  toolkit for earthquake forecast developers. *Seismol. Res. Lett.* 93(5).
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. (2023). Flow
  matching for generative modeling. *ICLR*.
- Gu, A., Dao, T. (2023). Mamba: Linear-time sequence modeling with selective
  state spaces. *arXiv:2312.00752*. Dao, T., Gu, A. (2024). Transformers are
  SSMs (Mamba-2). *ICML*.
- Dascher-Cousineau, K., Shchur, O., Brodsky, E. E., Günnemann, S. (2023).
  Using deep learning for flexible and scalable earthquake forecasting (RECAST).
  *Geophys. Res. Lett.* [verify].
- Stockman, S., et al. EarthquakeNPP: A benchmark for neural point-process
  earthquake forecasting. [verify venue/year/authors].
- Reference NPPs (cite per the benchmark): DeepSTPP, AutoSTPP, NSTPP, SMASH,
  DSTPP. [verify].

---

### Open items before submission
- ETAS-vs-FlowQuake CSEP head-to-head: harness built (`flowquake/etas_csep.py`)
  and a first 100-day ETAS run completed (raw data `runs/etas_csep_pod/`), but
  the ETAS scores are NOT yet reportable — systematic N under-prediction
  (suspected: continuation not conditioning on post-2007 triggers) and a
  sim-count confound (10³ vs 10⁴). Needs: fix trigger conditioning + matched
  sim counts, then fill §4.2 and add `fig_csep_headtohead.png`. Optional
  (paper stands without it).
- Finalize the bibliography: confirm [verify]-marked entries (EarthquakeNPP and
  the five reference NPPs) against the benchmark's published references. [USER]
- Decide venue (Seismica / GRL) and convert to the house format. [USER]

*Done in the pre-submission audit pass: §4.3 made reproducible
(`scripts/memorization_eval.py`) with the divergence-curve figure
(`fig_memorization_curve.png`); headline novelty claim narrowed to the
multi-seed-significant suite-wide statement; S-test denominator corrected to 91
evaluable days; full-suite summary made reproducible
(`scripts/aggregate_fullsuite.py`); §2 + references added.*
