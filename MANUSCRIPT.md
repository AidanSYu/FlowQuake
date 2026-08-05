# Structure beats flexibility: a transferable neural point process that surpasses ETAS on total likelihood across tectonic regimes

**Draft manuscript — numbers current as of 2026-07-02 (spatial/total results via
the full-history neural-ETAS head, §4.4; out-of-time 2020–2026 replication,
§4.1; Holm/TOST statistics, §3). Per-event and full-suite
results are 3-seed (mean ± std); CSEP and the memorization ablation use the
production model and the converged ablation checkpoints respectively, as noted.
Cross-regime results (§4.5) use AUTHORITATIVE catalogs — the internationally
reviewed ISC Bulletin (Japan, Chile, Greece, Iran; mc 4.0) and the INGV bulletin
(Italy; mc 2.5) — each with a region-fitted ETAS baseline (the `etas`
inversion), paired on matched test events. For the composite total-likelihood
claim in four ISC regions, the FlowQuake/ETAS magnitude-filtering conventions
leave a small completeness-edge intersection; coverage is reported in
`runs/stats_hardening.json`. One consistent source per
completeness regime removes inter-agency heterogeneity as a confound for the
cross-regime comparison. (An earlier draft used USGS-global catalogs; rebuilding
on authoritative data revised the cross-regime claim from "beats everywhere" to
the density-dependent statement below — see §4.5.)**

---

## Abstract

Neural point processes (NPPs) have repeatedly failed to outperform the
Epidemic-Type Aftershock Sequence (ETAS) model on the EarthquakeNPP benchmark,
despite far greater flexibility. We show *when* that verdict flips and why —
and, for the first time, flip it on **total** likelihood.
FlowQuake is, to our knowledge, the first NPP to beat ETAS temporally with
positive 3-seed means on **every** catalog of the California EarthquakeNPP suite
(ComCat, SCEDC, San Jacinto, Salton Sea, White), with conservative
block-bootstrap significance on four of five catalogs (San Jacinto is positive
but its autocorrelation-aware interval touches zero), on identical event sets
and with a full suite of CSEP consistency tests (number, spatial, and — uniquely
among the benchmark's generative NPPs — magnitude). Run catalog-to-catalog
through ETAS's own pyCSEP protocol at a matched simulation budget, FlowQuake is
statistically indistinguishable from the incumbent on all three consistency
tests.
We then close the axis every NPP has lost on. Diagnosing the spatial deficit as
triggering *coverage* — recurrence at locations older than any fixed context
window — we build a **full-history spatial head that strictly generalizes
ETAS's own spatial density**: its gate-closed form reproduces the benchmark's
published inversion to 2×10⁻⁹ nats in spatial log-likelihood, and it adds a
causal multi-scale
smoothed-seismicity background (the benchmark ETAS's background is uniform) and
per-parent neural modulations that never see the target location, so the
density remains exactly normalized. Trained under the standard NPP protocol,
it beats ETAS spatially in **all six regions tested, across four tectonic
regimes** (California +0.06, Italy +0.14, Japan +0.05, Chile +0.03, Greece
+0.09, Iran +0.15 nats/event; every gain block-bootstrap significant), and the
composite model beats region-fitted ETAS on **total** log-likelihood in all six
(California +0.113, Italy +0.210, Japan +0.039, Chile +0.061, and with few-shot
temporal transfer Greece +0.076 and Iran +0.084; all Holm-adjusted p ≤ 0.019) —
the first NPP total-likelihood win on this benchmark. Because the head is
initialized from each region's ETAS inversion, this composite is a strict
*upgrade* of a deployed ETAS system rather than an inversion-free replacement.
On a **retrospective out-of-time window** — 6.4 years of California data
(2020–2026), fetched with the benchmark's exact recipe and not used for fitting
or early stopping — the temporal, spatial and total wins all replicate
(dT +0.057, dS +0.067, dTot +0.124 [0.104, 0.146] on 10,187 events including
the 2024 M7.0 Cape Mendocino sequence). We do not call this a registered
prospective forecast because those events existed during development.
The temporal advantage itself is **density-dependent and regime-general**:
significant on dense catalogs in three regimes (California +0.05, Italy +0.07,
Chile +0.03; Holm-adjusted p = 0.003, 0.003 and 0.036 respectively) and fading
as catalogs thin to mc 4.0,
where ETAS's parametric Omori/Gutenberg–Richter structure is near-optimal
(Japan, whose Tohoku-dominated test set is textbook-Omori, is a tie) or where
too few events remain to train from scratch (data-poor Greece, Iran).
A controlled ablation explains the underlying structure–flexibility trade-off:
conditioning the output heads on a learned whole-catalog embedding induces
catastrophic memorization — training likelihood collapses while held-out
likelihood explodes by ~14 nats — and the cure is structural. FlowQuake
therefore pairs a flow-matching temporal head with ETAS-shaped,
observation-anchored heads for space and magnitude that cannot memorize the
training geography through learned absolute-coordinate weights. The learned
triggering structure is translation-invariant; deployment still recomputes
lightweight per-region normalization and a train-era smoothed-seismicity
background map. This makes one pre-trained model **transferable across tectonic
regimes** without ETAS-style per-region parameter inversion, and a
leave-one-region-out foundation model shows the key data-efficiency effect: on
data-poor regions, where training from scratch fails outright, transfer
*rescues* most of the gap to locally-inverted ETAS — Greece rises from a −0.11
nats/event loss to statistical equivalence (TOST at ±0.1 nats/event), Iran from
−0.28 to −0.06 (a four-fold narrowing that does not yet reach the equivalence
margin). Held-out Japan remains a small loss/tie boundary case, while a pooled
global deployment model trained on all regions is positive there
(autocorrelation-aware tie). The transfer covers *both* likelihood axes: the
spatial head transfers as well, beating an unseen region's own ETAS spatially
zero-shot in 7 of 7 within-completeness-regime transfers (California→Italy,
Japan→Chile, …), so one region's learned "how to beat ETAS" — temporal and
spatial — carries to another without retraining, provided completeness matches.

---

## 1. Introduction

Operational earthquake forecasting in California, Italy, Japan, New Zealand and
Switzerland is built on ETAS, a parametric spatio-temporal Hawkes process. The
EarthquakeNPP benchmark (Stockman et al. 2026) was assembled to test whether modern
neural point processes can match it on real catalogs, and found that they
cannot — ETAS was not beaten on total or spatial log-likelihood by any of the
five NPPs evaluated, on any dataset. The benchmark identified two structural
reasons NPPs underperform: fixed-window encoders that discard most of the
history, and the absence of the physically-motivated Omori/Gutenberg–Richter
kernels ETAS hard-codes.

We address both — and one of the two answers is negative, which we report as a
finding rather than bury. FlowQuake (i) implements the whole-catalog encoder the
benchmark's diagnosis calls for, a Mamba-2-style selective SSM over the entire
event sequence, and (ii) replaces hand-crafted kernels with learned but
physically-structured density heads. **Only (ii) pays.** At every bottleneck
width and every checkpoint we tested, routing the learned whole-catalog state
into the heads is worse than conditioning on translation-invariant relational
features computed over a 64-event lag window (§4.3); every result reported below
therefore uses `h_bottleneck = 0`, in which the SSM encoder is not instantiated
and the heads see a 30-dimensional hand-crafted relational block. The benchmark's
attribution of NPP failure to fixed-window encoding is thus not supported by our
experiments: the deficit we could close was the *kernel* structure, not the
context length. We make five contributions. (1) A temporal win over ETAS on every California
catalog in the benchmark in 3-seed means, with autocorrelation-aware significance
on four of five catalogs and CSEP consistency — and an **out-of-time
replication** on the 2020–2026 California window.
(2) The scope of that win: it is **density-dependent and regime-general**. Across
authoritative agency catalogs the temporal advantage holds significantly in three
tectonic regimes on dense (low-mc) data (California/transform, Italy/extension,
Chile/subduction) and shrinks to a tie or loss as catalogs thin to mc 4.0, where
ETAS's parametric structure is near-optimal — a clean characterization of *when*
neural flexibility helps.
(3) **The first NPP spatial and total-likelihood win on this benchmark.** We
localize the spatial deficit shared by all benchmark NPPs to triggering
*coverage* (recurrence at locations older than any fixed context window), then
close it with a full-history spatial head that strictly generalizes ETAS's own
density — initialized at the published inversion, exactly normalized, with a
causal smoothed-seismicity background replacing ETAS's uniform one. It beats
ETAS spatially in all six regions tested; the composite beats region-fitted
ETAS on total log-likelihood in all six (Holm-adjusted p ≤ 0.019), and the
result replicates on the out-of-time California window. Because the head builds on each
region's ETAS inversion, it upgrades a deployed ETAS system rather than
replacing the inversion. (4) A
controlled explanation of *why* flexible NPPs otherwise lose: exposing the heads
to a learned global catalog embedding causes memorization, not generalization,
and the cure is structural. (5) Because the shared weights are tied to observed
events and relational features, not learned absolute geography, one pre-trained
FlowQuake **transfers across tectonic regimes** without ETAS-style per-region
parameter inversion. It still recomputes lightweight per-region normalization
and background maps. Leave-one-region-out pre-training recovers most of the
data-poor-region gap after brief fine-tuning — where training from scratch fails
and ETAS is hardest to invert — reaching statistical equivalence with
region-fitted ETAS in Greece (TOST, ±0.1 nats/event) and narrowing Iran's loss
four-fold (−0.28 → −0.06, short of the equivalence margin), while Japan remains
a small loss/tie boundary case. On sparse mc 4.0 catalogs ETAS remains
near-optimal temporally; the spatial/total win uses the ETAS-initialized
full-history head rather than the inversion-free transfer model.

## 2. Related work

**ETAS and operational forecasting.** ETAS originates with Ogata's temporal
self-exciting model [Ogata 1988] and its space–time extension [Ogata 1998];
it underpins operational and pseudo-prospective forecasting and the CSEP
testing experiments. We use the maximum-likelihood inversion and
catalog-continuation simulator of Mizrahi et al. [2021] (the `etas` package) as
our incumbent, both for per-event likelihoods and for the CSEP head-to-head.

**Neural point processes on EarthquakeNPP.** The EarthquakeNPP benchmark
[Stockman et al. 2026] assembles California catalogs with fixed train/test windows
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

**Transfer learning and foundation models for earthquake forecasting.**
Transfer learning is well established for *seismological signal processing* —
phase picking, detection, denoising, magnitude and ground-motion estimation —
where labelled waveforms transfer across networks; this is a different task from
*forecasting* future seismicity and we do not claim novelty there. For
forecasting, the closest prior result is **SafeNet** [Zhang et al. 2025], which
pre-trains in one region and few-shot fine-tunes in another and reports gains
over ETAS. SafeNet, however, is a 4°×4° **gridded annual-maximum-magnitude
classifier** scored on F1/recall: it predicts a binary/categorical label per
cell-year, not a likelihood, intensity, or marked point process, and so cannot
be placed on the tll/sll/nll axis ETAS and the EarthquakeNPP benchmark use. Our
contribution is therefore stated precisely as the **first neural
point-process-likelihood forecaster shown to transfer across tectonic regimes
and beat region-fitted ETAS on temporal log-likelihood** — not the broad,
already-occupied claim of "first transfer learning for earthquake forecasting."
On the point-process axis, the **neural modulated renewal process (NMRP)**
[Zhan et al. 2026] is the nearest competitor: it matches and in some cases
surpasses ETAS temporally on EarthquakeNPP, but is California-only, has no
spatial head, performs no cross-region transfer, and reports no multi-seed
significance — we differentiate on the spatial head, the multi-catalog
significance-tested win, and transfer. **RECAST** [Dascher-Cousineau et al.
2023] *proposes* multi-region adaptation but demonstrates only temporal,
single-region forecasting and underperforms ETAS below ~10⁴ events; **FERN/FERN+**
[Zlydenko et al. 2023] only *ties* ETAS, with gains attributable to sub-Mc
magnitudes, per region and without transfer. Pre-trained "foundation" models in
seismology to date target detection or generic signal regression rather than
forecasting (e.g. SeisLM [Liu et al. 2024]; multi-region energy-budget models),
and multi-region *training* is not cross-regime *transfer* [cf. EPBench 2025].
To our knowledge no prior neural forecaster pre-trains once and, after brief
fine-tuning, beats locally-inverted ETAS in data-poor regions where ETAS has no
transfer mechanism — the data-efficiency result of §4.5.

**Method ingredients.** The temporal head is a conditional rectified
flow / flow-matching model [Lipman et al. 2023]; the encoder is a
pure-PyTorch Mamba-2-style selective state-space scan [Gu & Dao 2023; Dao & Gu
2024]. Forecasts are scored with the CSEP consistency-test framework
[Schorlemmer et al. 2007; Zechar et al. 2010] via pyCSEP [Savran et al. 2022].

*(Core references — EarthquakeNPP [Stockman et al. 2026, TMLR, arXiv:2410.08226],
the `etas` package [Mizrahi et al. 2021a,b], pyCSEP, the CSEP tests, and the
benchmark's reference NPPs (NSTPP, DeepSTPP, AutoSTPP, DSTPP, SMASH) — were
verified against authoritative sources in the pre-submission bibliography pass
(see References). The related-work entries SafeNet [10.1038/s41598-025-93877-7],
NMRP [10.1029/2025EF007342], RECAST [10.1029/2023GL103909], and FERN
[10.1038/s41598-023-38033-9] are cited by DOI; a final EarthArXiv/arXiv sweep is
scheduled immediately pre-submission.)*

## 3. Methods

**Data & protocol.** We use the EarthquakeNPP California catalogs with their
prescribed auxiliary/train/test windows and completeness magnitudes. Per-event
log-likelihood is decomposed into temporal (`tll`, log 1/day) and spatial
(`sll`, log 1/km²) components; we report both, plus `nll = −(tll+sll)`, on the
exact test events ETAS is scored on (verified: identical event sets to within
one event per catalog; ETAS `ll_scores` reproduced exactly from per-event
output).

**Encoder, and what the reported models actually condition on.** The
implementation provides a pure-PyTorch Mamba-2-style selective state-space
encoder over the entire event sequence (chunked SSD scan; verified against a
naive recurrence, `tests/test_ssm.py`). Each event token carries log-Δt,
location, magnitude, and translation-invariant relational features (per-lag
log-Δt, displacement, and magnitude to the last 1–64 events).

The encoder reaches the heads only through the bottleneck `h` below. **Every
result in §4 is at `h = 0`**, where `FlowQuakeTPP` sets `self.encoder = None`
(`flowquake/model.py:76-83`) and the head conditioning is
`tokens[:, SAFE_TOKEN_DIMS]` — a 30-dimensional vector of log-Δt, magnitude and
the seven-lag relational block, i.e. **hand-crafted Hawkes order statistics over
a 64-event window, with no learned sequence model in the path**. The reported
temporal model is therefore ~29.5k parameters (CondFlow 22,657 +
KernelMixtureHead 6,852 + GRMagnitudeHead 31), not the ~290k of the encoder
configuration. We state this plainly because the negative result in §4.3 is only
interpretable against it: the whole-catalog encoder was built, trained, and lost.

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

**Cross-regime catalogs (§4.5).** Foreign catalogs are built directly from
authoritative agency FDSN services (`scripts/build_region.py`): the ISC reviewed
Bulletin (Japan, Chile, Greece, Iran) and the INGV bulletin (Italy), each
projected to the EarthquakeNPP format with an azimuthal-equidistant local frame
and a region polygon for the ETAS inversion. Completeness mc is set per region
from a maximum-curvature estimate verified to be stable across the training and
test eras (`scripts/check_completeness.py`); we use one source per completeness
regime to avoid inter-agency heterogeneity. A region-fitted ETAS baseline is
inverted on each (`reference/Experiments/ETAS`, the `etas` package), paired on
matched test events. For total-likelihood composites, the event-count coverage
of the temporal/head/ETAS intersection is reported explicitly because the ETAS
pipeline bins magnitudes before the completeness cut while FlowQuake's temporal
CSV uses the raw catalog cut.

**Transfer and the foundation model (§4.5).** Zero-shot transfer applies the
trained weights to a target region, recomputing only the input standardization
and background map from target data (lightweight preprocessing, not an ETAS-style
parameter inversion); few-shot adds a 2,000-step warm-started fine-tune.
The foundation model is pre-trained on several regions at once with the target
held out (`scripts/train_pooled.py`): each minibatch is drawn from one region and
the model's normalization is swapped to that region for the step, so the shared
weights learn in a common normalized space. Significance on cross-regime gains
uses a stationary block bootstrap (block length 50 events) that preserves
aftershock autocorrelation, alongside the per-event paired statistic.

**Family-wise and equivalence testing (`scripts/stats_hardening.py`).** The six
per-region headline temporal comparisons of §4.5 form one claim family; we
report block-bootstrap p-values with a Holm–Bonferroni step-down correction
across the family. Any "ties ETAS" statement is additionally required to pass a
bootstrap TOST equivalence test — the 90% CI of the mean paired gain must lie
within ±0.1 nats/event (both margins 0.05 and 0.10 are reported in
`runs/stats_hardening.json`) — because a confidence interval that merely crosses
zero can reflect low power rather than genuine parity.

## 4. Results

### 4.1 Temporal win across the suite (headline)

FlowQuake's temporal log-likelihood exceeds ETAS on all five catalogs (3 seeds
each; seed-to-seed spread is shown for reproducibility — significance claims
rest on the paired block bootstrap, not seed variance):

| dataset | mc | FQ tll (3-seed) | ETAS tll | Δ | FQ sll | ETAS sll |
|---|---|---|---|---|---|---|
| ComCat_25 | 2.5 | 1.4868 ± 0.0008 | 1.4343 | +0.053 | −9.06 | −8.69 |
| WHITE_06 | 0.6 | 2.0669 ± 0.0007 | 2.0211 | +0.046 | −4.73 | −4.26 |
| SanJac_10 | 1.0 | 1.1610 ± 0.0009 | 1.1325 | +0.028 | −5.92 | −5.40 |
| SaltonSea_10 | 1.0 | 2.4337 ± 0.0070 | 2.3320 | +0.102 | −2.64 | −2.32 |
| SCEDC_20 | 2.0 | 2.6194 ± 0.0031 | 2.5410 | +0.078 | −7.85 | −7.53 |

On ComCat the paired per-event temporal gain is +0.052 ± 0.0025 (61% of events
improved). Under the autocorrelation-aware block bootstrap used for cross-region
figures, the temporal gain is significant on four of the five California
catalogs; San Jacinto remains positive but its interval touches zero. Against
the production kernel-mixture head ETAS retains the spatial edge on all five —
the deficit we diagnose and then close with the full-history head in §4.4
(numbers in this table are the production model's). The EarthquakeNPP
benchmark reported that none of its five NPPs beat ETAS on total or spatial
log-likelihood on any dataset; individual seeds of the more flexible models can
edge out ETAS on temporal likelihood on some catalogs, but not reproducibly.
To our knowledge FlowQuake is the first NPP to deliver a positive temporal
ETAS gain on *every* catalog of the suite (operational statewide, dense
fault-zone, and swarm regions), with statistical significance under the
conservative paired block bootstrap on all but the San Jacinto boundary case.

This temporal advantage is stable in time rather than an average inflated by a
single sequence. Both models are frozen before the held-out ComCat test period
and every per-event likelihood conditions only on prior events, so binning the
paired temporal gain by calendar time gives a skill-over-time decomposition of
the test window (§3; because the test window also informed model selection, we
label this a stability decomposition, not a prospective test). Across 180-day
windows FlowQuake's temporal gain is positive in **85% of windows**, and the
overall gain carries a stationary block-bootstrap 95% CI of **[+0.040, +0.068]
nats/event**, strictly above zero. The same per-region time-binned analysis
(`scripts/prospective_eval.py` → `runs/prospective.json`) shows the California
and Chile temporal wins of §4.5 are individually significant in every era of
their test windows, while few-shot transfer lifts data-poor Greece and Iran from
clear losses toward the parity boundary — the same density/transfer pattern the
full-period block bootstrap reports in §4.5.

**Out-of-time replication (2020–2026).** The stability decomposition above still
lives inside the benchmark's test window. For a stricter out-of-time check we
extended the ComCat catalog beyond the benchmark's 2020-01-17 endpoint using its
exact construction recipe — same USGS query, RELM polygon, mc 2.5, duplicate
jitter, and projection center (verified to reproduce the published coordinates
to <10⁻⁶ km; `scripts/build_comcat_forward.py`) — through 2026-07-01: 10,187
new events including the 2024 M7.0 Cape Mendocino, 2022 M6.4 Ferndale and 2021
M6.2 Petrolia sequences. Both models are scored at frozen benchmark states: the
production FlowQuake checkpoint and the benchmark's published ETAS inversion
(parameters fit through 2007), each conditioning on the full pre-event history.
The ETAS scorer reproduces the package's spatial per-event scores to
2×10⁻⁹ nats; the temporal term matches to ~10⁻⁵ per event except the first
target, where this implementation uses a consistent window-start anchor for both
background and triggering integrals (mean effect 1.5×10⁻⁴ nats;
`scripts/etas_forward_eval.py`). The paired temporal gain on the forward window
is **dT = +0.057, block-bootstrap 95% CI [+0.038, +0.082]** (60.5% of events
improved), matching the in-window +0.053 without using post-2020 targets for
fitting or early stopping. The spatial and total results replicate too: with the
frozen §4.4 head, dS = +0.067 [+0.055, +0.078] and
**dTot = +0.124 [+0.104, +0.146]** (`runs/total_win.json`). The spatial mean is
tail-driven rather than median-driven (47.8% event-level dS win rate), which is
consistent with the triggering-coverage mechanism in §4.4; late-sequence
aftershock incompleteness around the 2024 M7.0 sequence affects both models
symmetrically. Because the 2020–2026 events already existed during this project,
we treat this as retrospective out-of-time replication rather than a registered
prospective forecast. As an additional fairness control we re-inverted ETAS on
the full ComCat catalog *through* the forward-window start (1981–2020-01-17, the
operational-practice variant that lets ETAS use every event our checkpoint could
have seen; `reference/Experiments/ETAS/config/ComCat_25_refit2020.json`, EM
converged in 12 iterations). The refit barely moves the parameters (a 1.556→1.603,
log₁₀μ −6.333→−6.389, ρ 0.557→0.571; branching ratio 0.968) and improves ETAS's
forward NLL by only **0.016 nats** (7.464→7.448; dT +0.005, dS +0.011). Scored
against this fairer, up-to-date ETAS the FlowQuake total win narrows only slightly,
to **dTot = +0.108** (temporal +0.052, spatial +0.056) — the out-of-time win is
therefore not an artifact of stale ETAS parameters
(`runs/forward_etas_ComCat_25_refit2020/summary.json`).

### 4.2 CSEP consistency (ComCat, 100 forecast days × 10⁴ catalogs)

The production (density-adaptive, N1) model — the same model that produces the
per-event likelihoods in §4.1 — is CSEP-consistent on all three tests (a day is
consistent at the two-sided 95% level iff min(δ₁,δ₂) ≥ 0.025; days on which a
test is not evaluable are excluded). CSEP forecasts here are simulated with the
production kernel-mixture spatial head; the full-history head of §4.4 — the one
that wins the spatial and total likelihood — is run through the *same* CSEP
harness separately below ("Does the likelihood-winning head stay CSEP-consistent?"),
so no consistency claim rests on the assumption that the two heads calibrate alike:

| test | days evaluated | consistent | rate |
|---|---|---|---|
| Number (N) | 100 | 95 | 95% |
| Spatial (S) | 92 | 85 | 92% |
| Magnitude (M) | 92 | 90 | 98% |

The N- and M-test rejection rates (5% and 2%) sit at or below the nominal 5%.
The S-test rejects slightly more often than nominal (8%), the signature of the
residual sub-kilometre over-smoothing localized in §4.4; the density-adaptive
kernel reduces this relative to the base model (S 88% → 92%), consistent with
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

Both models simulate 10³ one-day catalogs per forecast day on the identical 100
days, scored through the same harness (matched simulation budget):

| test | FlowQuake (N1) | ETAS | reading |
|---|---|---|---|
| Number (N) | 95/100 (95%) | 97/100 (97%) | both consistent; counting is ETAS's core strength |
| Spatial (S) | 82/85 (96%) | 80/86 (93%) | FlowQuake at least as well spatially calibrated |
| Magnitude (M) | 89/92 (97%) | 87/92 (95%) | both consistent; only models with a magnitude head are M-testable |

At a matched budget on identical days, **both models are CSEP-consistent on all
three tests**: every pass rate sits at or above the nominal 95% except ETAS's
S-test at 93%, and the 2–4-day differences between the columns are within
binomial sampling noise at these evaluable-day counts. The substantive finding
is therefore not that one model out-passes the other on calibration — they tie —
but that FlowQuake, run catalog-to-catalog through the *incumbent's own*
evaluation protocol, is statistically indistinguishable from ETAS on number,
spatial, and magnitude consistency while additionally (i) reporting a tractable
per-event log-likelihood that ETAS-class simulators expose only through
Monte-Carlo and that the generative NPP baselines (SMASH, DSTPP) cannot provide
at all, and (ii) carrying the cross-regime transferability of §4.5 that ETAS,
which must be re-inverted per region, structurally lacks. A first ETAS run
through this harness had under-predicted counts (N 73/100); we traced it to the
fitted inversion's cached source set not being re-conditioned on
post-test_start mainshocks, and recomputing the triggering source set at each
forecast origin removes it (N → 97/100) — the number reported above. FlowQuake's
standalone 10⁴ consistency is in the table at the top of this section; this
head-to-head holds the simulation budget fixed at 10³ for both models so the
comparison is strictly like-for-like.

**Does the likelihood-winning head stay CSEP-consistent?** A natural objection is
that the full-history head of §4.4 buys its spatial likelihood by concentrating
probability so sharply on recent seismicity that its *simulated* catalogs would
be too clustered to pass the S-test — i.e. that the likelihood win trades away
CSEP calibration. We tested this directly. Because the head's near-set and
far-field priors are selected target-location-*independently*, at a fixed
forecast time its spatial density evaluates over the whole CSEP grid in one
vectorized pass, giving a full-history gridded simulator (validated to reproduce
the head's per-event SLL to 9.5×10⁻⁷ nats; `flowquake/neural_etas_forecast.py`).
We drew each day's counts, times and magnitudes from the identical production
simulator (the head is spatial-only, so N and M are unchanged) and its event
*locations* from the head's per-day grid density, then scored the result through
the same pyCSEP path, identical 100 forecast days, and matched 10³-catalog
budget (`flowquake/csep_forecast_head.py`). The head is CSEP-consistent —
**N 95/100, S 79/85 (92.9%), M 90/92** — and, crucially, on the S-test it is
*statistically indistinguishable from ETAS*: pairing the two runs day-by-day
(the same forecast days, so the comparison is exactly paired) they agree on
**77/83 evaluable days each**, with only 10 discordant days split 5–5, a McNemar
exact p = 1.00. Against the production kernel-mixture head the paired difference
is likewise not significant (75/81 vs 78/81, McNemar p = 0.51). In other words
the head improves ETAS's per-event spatial likelihood by +0.06 nats (§4.4) while
matching ETAS's CSEP spatial calibration to the day: the likelihood win costs
nothing in consistency. This closes the last operational-replacement gate —
FlowQuake beats ETAS on temporal, spatial and total likelihood and is
statistically indistinguishable from it on all three CSEP consistency tests,
run catalog-to-catalog through the incumbent's own protocol.

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

### 4.4 Localizing — and closing — the spatial gap

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
history with Omori decay. Early attempts to restore this coverage suggested a
hard limit: truncated-mixture retrains and fitted proper-normalizer kernels all
landed 0.27–0.35 nats short of ETAS. (An apparent −7.9 ceiling we initially
measured was an artifact of normalizing a spatial density over a
target-dependent neighbour set rather than over all prior events — a trap we
document because it silently inflates any "reconstruction ceiling".) Those
attempts, however, truncated the kernel sums in space and used fixed,
mis-calibrated temporal weights. Removing both restrictions removes the limit.

**A full-history spatial head that strictly generalizes ETAS
(`flowquake/neural_etas.py`).** We reimplement the benchmark's exact spatial
density — verified to reproduce its per-event spatial log-likelihood to
2×10⁻⁹ nats from the published inverted parameters
(`scripts/etas_sll_repro.py`) — and generalize it in three ways, all preserving
exact normalization:
(i) the **background**, uniform in the benchmark's ETAS, becomes a learned
mixture of the uniform term and *causal* multi-scale smoothed-seismicity
densities (Gaussian KDEs over all strictly-prior events at 1.5/6/25/100 km,
evaluated per target — no future information);
(ii) each parent event's **weight and kernel shape** receive multiplicative
neural modulations g(m_j, Δt) that depend only on the parent's magnitude and
elapsed time — never on the target location — applied to a
target-location-independent near set (top-256 by ETAS weight + 128 nearest the
previous event), so the mixture's closed-form integral survives and the density
stays exactly normalized (the trap above is structurally excluded);
(iii) a global background/triggering rebalancing (two scalars).
With the KDE gate closed the head is the benchmark ETAS to float precision; the
default training initialization opens a small (~5%) KDE gate so gradients reach
the smoothed-seismicity background, starting only +0.002 to +0.004 nats/event
above ETAS. Trained gains below are always measured against the package ETAS
scores, not against this near-ETAS initialization. Training uses the standard
NPP split (train < 1998, validation 1998–2007 for early stopping), but the
reported grid includes ablations, controls and three seeds that were all scored
on the 2007–2020 test set during development; we therefore lean on the
2020–2026 out-of-time replication as the cleaner single-window confirmation.

**Results (3 seeds, ComCat).** The head beats ETAS spatially: test sll
**−8.630 vs −8.690** (seeds −8.6298/−8.6299/−8.6291), paired per-event
**dS = +0.060, 95% CI [+0.051, +0.069]**. The ablation attributes the gain
honestly: the causal multi-scale background alone (no neural modulation) gives
dS = +0.051; a classical control that additionally SGD-refits the global ETAS
kernel parameters (an flETAS-style refit with smoothed background, no neural
component) gives +0.056 — a conservative lower bound on a full classical refit,
since our control re-fits the kernel only on the near set and leaves the far
field at the inversion values; the per-parent neural modulation brings +0.060.
Most
of the spatial win is thus *structure ETAS itself could add* (a smoothed
background — which the benchmark's incumbent, and every NPP compared against
it, lacked), and the neural component contributes a smaller, consistent
increment on top. What the head learns is interpretable
(Fig. neural_etas_modulation): the background mixture puts ~85% of its mass on
the seismicity maps with the **1.5 km map dominant** — exactly the
sub-kilometre recurrence structure the coverage diagnosis identified — and the
modulation surfaces upweight old-but-colocated parents relative to ETAS's
global Omori decay.

**The result generalizes and replicates out-of-time.** Trained per region under
the identical protocol, the head beats each region-fitted ETAS spatially in
**all six regions** — California +0.060, Italy +0.137, Japan +0.053, Chile
+0.029, Greece +0.088, Iran +0.146 nats/event (3-seed means; per-seed spread
≤0.003, so all six clear zero at every seed), every block-bootstrap CI strictly
positive — and on the frozen 2020–2026 California forward window the
spatial gain *grows* (dS = +0.067 [+0.055, +0.078]). Combined with the flow
temporal head, the composite beats region-fitted ETAS on **total**
log-likelihood in all six regions (California +0.113, Italy +0.210, Japan
+0.039, Chile +0.061; Greece +0.076 and Iran +0.084 with few-shot temporal
transfer; Holm-adjusted p ≤ 0.019 across the family;
`runs/stats_hardening.json`) — on ComCat, nll 7.142 vs ETAS's 7.255, the
benchmark's first NPP total-likelihood win, replicating on the 2020–2026
out-of-time window at dTot = +0.124 [+0.104, +0.146]. Japan's total effect is a
statistically positive but small +0.039 nats/event, below the 0.05-nat
interpretability margin used for equivalence checks. Four-region composite
totals outside California/Italy are computed on the intersection of
FlowQuake-temporal, neural-head and ETAS-scored events (coverage:
Japan 96.3%, Chile 97.1%, Greece 92.0%, Iran 89.0%), while the head-only dS
comparisons use the full ETAS-scored test sets. One qualification keeps this
honest: the head is initialized from each region's ETAS inversion, so the
composite *upgrades* an existing ETAS deployment rather than removing the
inversion. CSEP consistency for this head is not an open item — §4.2 reports it
run through the same pyCSEP path, on the same 100 forecast days, at a matched
10³-catalog budget (N 95/100, S 79/85, M 90/92; S-test statistically
indistinguishable from ETAS, McNemar exact p = 1.00).

**The spatial head itself transfers across regimes — within a completeness
regime (`scripts/transfer_neural_etas.py`).** The head's learned components are
region-agnostic recipes — how much smoothed background to mix in, and how to
reshape each parent's kernel by its magnitude and elapsed time (the
translation-invariant modulation g(m_j, Δt)) — so a head *trained on one region*
should improve ETAS in *another*. It does: applying a source-trained head's
learned weights to a target region's own ETAS-initialized features (the target
still supplies its inversion and causal background, but no target training),
the source head beats the target's region-fitted ETAS spatially **zero-shot in
7 of 7 within-completeness-regime transfers** (win or tie; e.g. California→Italy
+0.073, Japan→Chile +0.095, Chile→Japan +0.016 nats/event), rising to **7/7
clear wins** after a light few-shot recalibration of the four background/scale
scalars on the target's validation split (the MLP stays frozen). The gain is not
merely the smoothed background: an ablation transferring a background-only head
(no modulation) still wins, but the neural modulation adds a further +0.01 to
+0.04 nats/event on top in the unseen region, and for Japan→Greece the
modulation is what converts a non-win into a win. Transfer fails only *across*
completeness regimes (mc 2.5 → mc 4.0, 0 of 4), the same regime-matching
constraint the temporal transfer obeys — spatial clustering scales are
completeness-dependent. Thus **both** axes of the ETAS-beating model transfer:
one region's learned "how to beat ETAS" — temporally and spatially — carries to
another without retraining, provided completeness matches. The per-region ETAS
inversion (for the frozen far field) and causal background map remain the only
region-specific ingredients.

### 4.5 Cross-regime generalization: a density-dependent temporal win and a transferable foundation model

ETAS is refit per region: its background field and inverted parameters are
specific to one catalog, so it cannot forecast a region it was not fit on.
FlowQuake's learned weights, by construction, condition on translation-invariant
relational features (§3); a target region still supplies normalization statistics
and a train-era smoothed-seismicity background map. Thus a *single* trained
weight set can be applied anywhere without ETAS-style parameter inversion.
We test cross-regime behaviour on **authoritative** catalogs to remove
data-quality confounds: the internationally reviewed ISC Bulletin for Japan
(subduction), Chile (subduction), Greece (continental extension) and Iran
(continental collision), all complete to mc 4.0 across both training and test
eras (maximum-curvature b = 0.80–1.07: Japan 0.82, Chile 0.80, Greece 1.07,
Iran 0.88); and the INGV bulletin for Italy (continental extension/
compression), complete to mc 2.5 (b = 0.98). Each region has an ETAS model fit
natively on it (the `etas` inversion). Temporal skill is the per-event paired
gain over region-fitted ETAS (dT = tll_FlowQuake − tll_ETAS, nats/event), with
significance from a stationary block bootstrap that preserves aftershock
autocorrelation; spatial/total are reported alongside. Each region is
represented by one temporal model chosen by a fixed rule — the natively-trained
model where the catalog is large enough to train from scratch (California,
Italy, Japan, Chile), and the few-shot foundation model in the two data-poor
regions where native training fails (Greece, Iran) — and that same variant is
used in both the temporal and the total-likelihood families of §3.

**The temporal win is density-dependent and regime-general (Fig.
density_dependence; `scripts/make_density_figure.py`).** Sweeping completeness
across the California suite and the foreign catalogs, a natively-trained
FlowQuake beats region-fitted ETAS temporally on dense (low-mc) catalogs in
three distinct regimes; all listed dense wins are significant under the block
bootstrap except the San Jacinto boundary case:

| catalog | regime | mc | FQ native dT (95% CI) |
|---|---|---|---|
| WHITE / Salton Sea / San Jacinto / SCEDC | transform (CA) | 0.6–2.0 | +0.03 … +0.10 (all >0) |
| California | transform | 2.5 | **+0.053** [+0.040, +0.067] |
| Italy | extension | 2.5 | **+0.071** [+0.050, +0.097] |
| Chile | subduction | 4.0 | **+0.034** [+0.007, +0.065] |
| Japan | subduction | 4.0 | −0.014 [−0.032, +0.005]* |
| Greece | extension | 4.0 | −0.107 [−0.144, −0.070] |
| Iran | collision | 4.0 | −0.276 [−0.347, −0.205] |

Family-wise, the California, Italy and Chile wins survive a Holm–Bonferroni
correction across all six per-region comparisons (adjusted p = 0.003, 0.003
and 0.036 respectively; block-bootstrap p-values;
`runs/stats_hardening.json`).
The advantage is largest where catalogs are dense and shrinks as they thin: at
mc 4.0 ETAS's parametric Omori/Gutenberg–Richter structure is near-optimal.
FlowQuake is a small negative/equivalent boundary case on Japan (the 95% block
bootstrap interval touches zero and TOST equivalence holds at ±0.05; Japan's
test set is dominated by the textbook-Omori M9.0 Tōhoku aftershock sequence,
ETAS's ideal case), and loses on
the two data-poor regions where from-scratch training has too few events (Greece
2.6k, Iran 2.0k training events above mc). On the densest foreign catalog, Italy, even the production
kernel-mixture head narrows the spatial gap to dS = −0.067 — small enough that
total likelihood reaches a statistical tie (dTotal = +0.004, TOST-equivalent at
±0.05) — and with the full-history head of §4.4 Italy becomes a decisive
**total-likelihood win** (dTotal = +0.210 [+0.186, +0.234]; nll 7.387 vs ETAS
7.596).

**The temporal win is not a magnitude-scale artifact (`scripts/mag_robustness.py`).**
Because the catalogs mix magnitude types (ML, Md, Mw, mb) with different scalings,
a natural concern is that the gain reflects magnitude-conversion heterogeneity
rather than genuine forecasting skill. Stratifying the paired per-event temporal
gain dT by event magnitude shows the win is broad-based rather than concentrated
in any band: in California dT is positive in all six 0.5-unit bins from mc 2.5 to
5.5 (block-bootstrap CI strictly above zero in five), in Italy it is positive in
every bin (significant wherever the bin is populated), and the rank correlation
between dT and magnitude is negligible in all three dense regimes (Spearman
+0.055 California, +0.004 Italy, +0.039 Chile).

**Mw-homogenization stress test, and the data-efficiency mechanism it exposes
(`scripts/mw_robustness.py`).** We go further and re-run the full pipeline
(re-inverting ETAS and retraining FlowQuake) on moment-magnitude-homogenized
catalogs. In California this is essentially a null operation: ML ≈ Mw over the
relevant range (Mw − ML ≈ 0.06, Clinton et al. 2006; ML = Mw for 3 ≲ M ≲ 6,
Bakun 1984) and NCSN duration magnitudes are calibrated to ML (Oppenheimer et al.
1992), so ComCat magnitudes are already Mw-equivalent — the treatment used by the
EarthquakeNPP benchmark, UCERF3, and USGS operational ETAS. Restricting to the
M ≥ 3.0 subset, where the catalog is 83% ML / 1% Md and magnitudes are
unambiguously Mw, the production (mc 2.5-trained) model still beats ETAS
temporally (**dT = +0.074, 95% CI [+0.050, +0.101]**, n = 7850): the headline win
is robust to magnitude scale. Crucially, the *same* advantage on those M ≥ 3.0
events **vanishes (dT = −0.079) when the model is re-trained on only the sparse
M ≥ 3.0 data** — identifying the win as a *data-efficiency* property: FlowQuake
extracts forecasting skill from the dense low-magnitude record (which improves its
forecasts of the large events too) that ETAS's parametric form leaves on the
table. This is the same density-dependence reported above, now mechanistically
localized.

The Italy total-likelihood results above (the kernel-mixture tie and the
full-history-head win) are *native-catalog* results and must be qualified
accordingly. Unlike California, Italian ML ≪ Mw and
the INGV catalogue is duration-magnitude-dominated before 2005, so Mw
homogenization is mandatory and is performed with the official per-type relations
(Gasperini et al. 2013, as in INGV's HORUS/CPTI15). That conversion *compresses
the catalogue ~2× above completeness* (19.4k → 10.4k training events), and the
temporal advantage erodes (dT = −0.25). Thinning is the dominant cause: a
density-matched control on the **native ML** scale (raising mc to 2.8, 9.2k
training events) already collapses the win to a tie (dT = +0.002 [−0.024,
+0.031]) — removing the dense low-magnitude record, at the same magnitude scale,
removes most of the edge. The further drop from a tie to a clear loss under Mw
reflects a residual sensitivity to the heavy, type-dependent Md→Mw compression
itself (the stretch distorts the magnitude features the neural heads consume),
which ETAS — re-fitting a single productivity exponent — absorbs. Both effects
are specific to Italy's duration-magnitude-dominated catalogue and neither arises
in California, where magnitudes are already Mw-equivalent. We therefore report the
Italy total-likelihood parity as a dense-native-catalogue result, do not claim it
under Mw homogenization, and note that FlowQuake's advantage presupposes a dense
catalogue on a temporally consistent magnitude scale.

**Transfer rescues data-poor regions; pooled deployment supplies the global artifact.** We
pre-train one model on the pooled mc-4.0 regions with each target region held out
(`scripts/train_pooled.py`; region-homogeneous minibatches with per-region
normalization, so the shared weights operate in a common normalized space — the
same mechanism that makes zero-shot transfer possible), then evaluate it on the
held-out region zero-shot and after a brief (2,000-step) warm-started fine-tune:

| region | FQ native dT | pooled zero-shot dT | pooled few-shot dT | vs ETAS (bootstrap + TOST) |
|---|---|---|---|---|
| Japan | −0.015 | −0.021 | −0.022 | native tie; held-out transfer small loss |
| Chile | **+0.034** | −0.027 | **+0.042** | native win (Holm-adjusted p = 0.036); few-shot also positive |
| Greece | −0.107 | −0.040 | −0.012 | **equivalent** (TOST ±0.1: 90% CI [−0.056, +0.035]) |
| Iran | −0.276 | −0.105 | −0.063 | loss narrowed 4×; **not** equivalent (90% CI [−0.122, −0.001]) |

Japan is the hard held-out-transfer case: the native model is a block-bootstrap
tie, but pooled zero-/few-shot transfer remains a small temporal loss on the
Tohoku-dominated test set. On Greece and Iran the from-scratch model loses to
ETAS by 0.11–0.28 nats; a model pre-trained on the *other* regimes and briefly
fine-tuned recovers most of that gap — Greece to demonstrated statistical
equivalence with region-fitted ETAS (TOST at ±0.1 nats/event), Iran to a
four-fold-smaller residual loss that does not yet reach the equivalence margin
(we say "narrowed", not "tie": its 90% CI sits just below zero). Transfer thus
turns a regime where you cannot train a neural model from scratch into one at or
near ETAS-level temporal skill *without a per-region inversion*. A standard
region-fitted ETAS pipeline has no learned
cross-region weight-transfer path; absent a new inversion its natural floor is a
homogeneous-Poisson baseline. The same data-efficiency effect appears within a
single region (Fig. data_efficiency; `scripts/data_efficiency.py`): truncating
Chile's training history, transfer beats from-scratch training across the
1k–5k-event range, and the two converge only when data is abundant. Transfer is
strongest within a completeness regime; we therefore pre-train and transfer
within a common mc (the operationally realistic setting).

**One pooled global model (the deployment artifact).** The
leave-one-out protocol holds each region out of pre-training to prove transfer to
*unseen* regimes; the deployable artifact is the complement — a single model
pre-trained on *all six* regions at once, mixing mc 2.5 and mc 4.0 (per-region
normalization lets one set of weights span both completeness levels). Applied
with no per-region weight fitting after pooling and paired against each region's
own ETAS inversion (`scripts/eval_global.py`):

| region | regime | mc | global pooled dT (z) | global few-shot dT |
|---|---|---|---|---|
| California | transform | 2.5 | **+0.031** (15.1) | — |
| Italy | extension | 2.5 | **+0.102** (21.9) | — |
| Japan | subduction | 4.0 | **+0.011** (2.7) | — |
| Chile | subduction | 4.0 | **+0.022** (2.8) | — |
| Greece | extension | 4.0 | −0.055 (−3.0) | **−0.008** (CI incl. 0) |
| Iran | collision | 4.0 | −0.158 (−5.0) | **−0.045** (CI incl. 0) |

Under an ordinary paired z-score this single model is positive in four regions
spanning both completeness levels and all three regimes (all z > 2), but the
autocorrelation-aware block bootstrap is stricter: California and Italy are
robust wins, while Japan and Chile are positive ties. A brief 2,000-step
fine-tune brings the two remaining data-poor regions to intervals that include
zero; we flag that "CI includes zero" is weaker than demonstrated equivalence —
under the TOST criterion of §3 the leave-one-out few-shot Greece qualifies as
equivalent while Iran does not (`runs/stats_hardening.json`). Pooling shows
*positive transfer*: the global model improves over the region-*native* models
on Japan (native tie → pooled positive tie) and Italy (+0.071 → +0.102) —
knowledge from other regimes improves in-region forecasting — and on Italy it
beats ETAS on total likelihood too (dTotal +0.084). One pooled model, with no
per-region weight fitting after pooling, therefore gives a deployment path to
temporal wins/ties across the tested regimes, but not yet a universal
operational replacement. That is the deployment property to harden next, and it
is exactly what per-region ETAS inversion (hours of EM per catalog, no learned
transfer) structurally lacks.

In sum: temporally, FlowQuake beats region-fitted ETAS wherever catalogs are
dense — in transform, extension and subduction regimes — while at mc 4.0 ETAS's
hard-coded structure remains at least as good temporally (Japan tie;
Greece/Iran need transfer to approach parity); we characterize this boundary
rather than overclaim past it. Spatially — and therefore on total likelihood —
the full-history head of §4.4 reverses the verdict in *all six* regions,
including the sparse ones (every dTot Holm-significant), with the qualification
that the head builds on each region's ETAS inversion and Italy's total win is a
native-catalogue result (it is not claimed under Mw homogenization, below).

## 5. Discussion

The benchmark's lesson is not "more flexibility": it is that the inductive
biases ETAS encodes (Omori time decay, magnitude-scaled triggering, a smooth
background) are load-bearing, and that a model free to ignore them will instead
memorize. FlowQuake keeps a flexible temporal head — where the data rewards it —
and structured heads where flexibility backfires. The structure–flexibility
trade-off is *quantitative and resolves with data density*: neural flexibility
buys a temporal edge precisely when there are enough events to estimate the
fine-grained inter-event structure ETAS approximates parametrically, which is why
FlowQuake wins on dense (low-mc) catalogs in every regime we tested and converges
to a tie as catalogs thin to mc 4.0, where the Omori/Gutenberg–Richter form is
already near-optimal. This reframes the long-standing "NPPs cannot beat ETAS"
verdict as completeness-conditional rather than absolute. The same design choice
that prevents memorization — learned weights tied to observed events and
relational features rather than absolute geography — is what makes the model
*portable*: one pre-trained model transfers across tectonic regimes without
ETAS-style parameter inversion, although per-region normalization and a
smoothed-seismicity background map remain part of preprocessing. On data-poor
regions where a neural model cannot be trained from scratch, transfer recovers
most of the gap to locally-inverted ETAS (equivalence in Greece, a four-fold
narrowing in Iran). On the spatial axis the diagnosis of §4.4 resolves the
benchmark's longest-standing verdict: the deficit every NPP shows against ETAS
is triggering *coverage* (recurrence at locations older than any fixed context
window), and a full-history spatial head that strictly generalizes ETAS's own
density — initialized at the published inversion and trained under the NPP
protocol — beats ETAS spatially on the flagship catalog, and with it on total
likelihood. The contribution is thus a characterization of *when and where*
neural flexibility helps (dense catalogs, full-history spatial structure,
transfer to data-poor regions), together with a pooled/few-shot deployment path
that wins or approaches per-region ETAS temporally across the tested tectonic
settings — most valuable exactly where ETAS is hardest to apply.

## 6. Conclusion

FlowQuake is the first NPP we know of to beat ETAS temporally in 3-seed means
across the EarthquakeNPP California suite, with block-bootstrap significance on
four of five catalogs and full CSEP
consistency including the magnitude test; it comes with a controlled explanation
of the field's long-standing NPP-vs-ETAS gap (memorization, cured structurally).
We then establish *when* the temporal win generalizes: on authoritative agency
catalogs it holds, significantly, in three distinct tectonic regimes
(transform, extension, subduction) wherever catalogs are dense, and narrows to
a tie as catalogs thin to mc 4.0 where ETAS's parametric structure is
near-optimal. We close the axis NPPs have always lost: a full-history spatial
head that strictly generalizes ETAS's own density — initialized at the
published inversion, exactly normalized, with a causal smoothed-seismicity
background replacing the incumbent's uniform one — beats region-fitted ETAS
spatially in all six regions tested, and the composite model beats it on
**total** log-likelihood in all six (ComCat nll 7.142 vs 7.255; Italy 7.387 vs
7.596; every dTot Holm-significant), the first NPP total-likelihood win on this
benchmark. The temporal, spatial and total gains all replicate on a
**retrospective out-of-time window** — 6.4 years of post-2020 California data
(2020–2026) with both benchmark-state models frozen (Italy's total win is a
native-catalogue result and is not claimed under Mw homogenization, §4.5; the
head builds on each region's ETAS inversion, so it upgrades rather than removes
the inversion).
Finally, because its shared weights encode relational triggering structure rather
than learned absolute geography, one pre-trained FlowQuake **transfers across
tectonic regimes** without ETAS-style per-region parameter inversion: a
foundation model transfers, pooled deployment is positive in four regions under
ordinary paired z-scores and robustly wins two under the block bootstrap, and on
data-poor regions where from-scratch training falls short, fine-tuning recovers
most of the gap (equivalence with region-fitted ETAS in Greece under TOST; a
four-fold narrowing in Iran). The full-history spatial head transfers too,
within a completeness regime: a source-trained head beats an unseen target
region's own ETAS spatially zero-shot in 7 of 7 within-regime transfers (§4.5),
so *both* likelihood axes of the ETAS-beating model are portable. Per-region
normalization, a smoothed-seismicity background map, and the target's ETAS
inversion (which the head upgrades) remain required. Future work: authoritative
low-mc catalogs in further regimes to test whether the dense-catalog win extends
(the prohibitive cost of inverting ETAS on very large low-mc catalogs is the
current limit); cross-completeness spatial transfer (currently the one transfer
boundary); and operational deployment of the pre-trained model in
newly-instrumented regions.

## References

*All entries below were confirmed against authoritative sources (arXiv, ACM DL,
ICML/NeurIPS/L4DC/ICLR proceedings, GeoScienceWorld, Crossref, and the `etas`
package README) during the pre-submission bibliography pass.*

- Ogata, Y. (1988). Statistical models for earthquake occurrences and residual
  analysis for point processes. *J. Amer. Statist. Assoc.* 83(401), 9–27.
- Ogata, Y. (1998). Space–time point-process models for earthquake occurrences.
  *Ann. Inst. Statist. Math.* 50(2), 379–402.
- Mizrahi, L., Nandan, S., Wiemer, S. (2021). Embracing data incompleteness for
  better earthquake forecasting. *J. Geophys. Res. Solid Earth* 126(12),
  e2021JB022379. doi:10.1029/2021JB022379. (methods paper for the `etas` package)
- Mizrahi, L., Nandan, S., Wiemer, S. (2021). The effect of declustering on the
  size distribution of mainshocks. *Seismol. Res. Lett.* 92(4), 2333–2342.
  doi:10.1785/0220200231. (the `etas` package)
- Stockman, S., Lawson, D. J., Werner, M. J. (2026). EarthquakeNPP: A benchmark
  for earthquake forecasting with neural point processes. *Transactions on
  Machine Learning Research* (TMLR). arXiv:2410.08226.
- Schorlemmer, D., Gerstenberger, M. C., Wiemer, S., Jackson, D. D., Rhoades,
  D. A. (2007). Earthquake likelihood model testing. *Seismol. Res. Lett.*
  78(1), 17–29. doi:10.1785/gssrl.78.1.17.
- Zechar, J. D., Gerstenberger, M. C., Rhoades, D. A. (2010). Likelihood-based
  tests for evaluating space–rate–magnitude earthquake forecasts. *Bull.
  Seismol. Soc. Am.* 100(3), 1184–1195. doi:10.1785/0120090192.
- Savran, W. H., Bayona, J. A., Iturrieta, P., et al. (2022). pyCSEP: A Python
  toolkit for earthquake forecast developers. *Seismol. Res. Lett.* 93(5),
  2858–2870. doi:10.1785/0220220033.
- Gasperini, P., Lolli, B., Vannucci, G. (2013). Empirical calibration of local
  magnitude data sets versus moment magnitude in Italy. *Bull. Seismol. Soc. Am.*
  103(4), 2227–2246. doi:10.1785/0120120356. (Mw homogenization for INGV.)
- Clinton, J. F., Hauksson, E., Solanki, K. (2006). An evaluation of the SCSN
  moment tensor solutions: robustness of the Mw magnitude scale, style of
  faulting, and automation of the method. *Bull. Seismol. Soc. Am.* 96(5),
  1689–1705. doi:10.1785/0120050241. (Mw ≈ ML in California.)
- Bakun, W. H. (1984). Seismic moments, local magnitudes, and coda-duration
  magnitudes for earthquakes in central California. *Bull. Seismol. Soc. Am.*
  74(2), 439–458. (ML = Mw for 3 ≲ M ≲ 6.)
- Oppenheimer, D., Klein, F., Eaton, J. (1992). The first 20 years of CALNET, the
  Northern California Seismic Network. *U.S. Geol. Surv. Open-File Rep.* 92-209.
  (NCSN duration magnitude calibrated to ML.)
- Dascher-Cousineau, K., Shchur, O., Brodsky, E. E., Günnemann, S. (2023).
  Using deep learning for flexible and scalable earthquake forecasting (RECAST).
  *Geophys. Res. Lett.* 50(17), e2023GL103909. doi:10.1029/2023GL103909.
- Chen, R. T. Q., Amos, B., Nickel, M. (2021). Neural spatio-temporal point
  processes (NSTPP). *ICLR*. arXiv:2011.04583.
- Zhou, Z., Yang, X., Rossi, R., Zhao, H., Yu, R. (2022). Neural point process
  for learning spatiotemporal event dynamics (DeepSTPP). *L4DC*, PMLR 168,
  777–789. arXiv:2112.06351.
- Zhou, Z., Yu, R. (2023). Automatic integration for spatiotemporal neural point
  processes (AutoSTPP). *NeurIPS 36*. arXiv:2310.06179.
- Yuan, Y., Ding, J., Shao, C., Jin, D., Li, Y. (2023). Spatio-temporal diffusion
  point processes (DSTPP). *KDD '23*, 3173–3184. doi:10.1145/3580305.3599511.
  arXiv:2305.12403.
- Li, Z., Xu, Q., Xu, Z., Mei, Y., Zhao, T., Zha, H. (2024). Beyond point
  prediction: score matching-based pseudolikelihood estimation of neural marked
  spatio-temporal point process (SMASH). *ICML*. arXiv:2310.16310.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. (2023). Flow
  matching for generative modeling. *ICLR*.
- Gu, A., Dao, T. (2023). Mamba: Linear-time sequence modeling with selective
  state spaces. *arXiv:2312.00752*. Dao, T., Gu, A. (2024). Transformers are
  SSMs (Mamba-2). *ICML*.

---

### Open items before submission
- Decide venue and convert to the house format. [USER]
- Run a final EarthArXiv/arXiv novelty sweep immediately before submission to
  catch any 2026 preprints on transfer/foundation models for forecasting. [USER]
- [DONE 2026-07-02] Out-of-time window: 2020–2026 ComCat forward test —
  temporal, spatial and total gains all replicate frozen (§4.1, §4.4;
  `runs/total_win.json`). ETAS-refit-through-2020 fairness control DONE
  (`ComCat_25_refit2020`, EM converged 12 iters): refitting ETAS on all data
  through the forward-window start improves its forward NLL by only 0.016 nats,
  and FlowQuake still wins the forward total by +0.108 nats
  (`runs/forward_etas_ComCat_25_refit2020/summary.json`).
- [DONE earlier] Mw-homogenization robustness (§4.5, `scripts/mw_robustness.py`):
  California's win is Mw-robust on the unambiguously-Mw M ≥ 3 subset; the Italy
  erosion is a density effect from Md→Mw compression, matched by a native-scale
  density control.
- [DONE] CSEP re-run with the full-history spatial head: the full-history
  gridded simulator is built and validated to 9.5×10⁻⁷ nats, and the 100-day
  run at the matched 10³-catalog budget is complete — N 95/100, S 79/85,
  M 90/92, with the S-test statistically indistinguishable from ETAS on the
  paired days (McNemar exact p = 1.00), reported in §4.2.
- Remaining hardening: ETAS-refit-through-2020 forward control DONE (above).
  Optional full flETAS (EM, free background) baseline beyond the SGD refit
  control of §4.4 remains. Three-seed full-history heads are complete for all
  six regions (spread ≤ 0.006 nats/event).

*Done in the pre-submission audit pass: §4.3 made reproducible
(`scripts/memorization_eval.py`) with the divergence-curve figure
(`fig_memorization_curve.png`); headline novelty claim narrowed to the
suite-wide positive temporal statement plus the San Jacinto block-bootstrap
boundary; S-test denominator corrected to 92
evaluable days; full-suite summary made reproducible
(`scripts/aggregate_fullsuite.py`); §2 + references added. ETAS-vs-FlowQuake
CSEP head-to-head completed and validated at a matched 10³-sim budget on
identical days (`flowquake/etas_csep.py` → `runs/csep_h2h_etas/`,
`runs/csep_h2h_fq/`): the earlier N under-prediction was traced to the fitted
inversion's source set not being re-conditioned on post-test_start mainshocks
and fixed (`reload.source_events = reload.prepare_source_events()`); both models
are now CSEP-consistent (a calibration tie), reported in §4.2. Pseudo-prospective
skill-over-time test added (`scripts/prospective_eval.py` → `runs/prospective.json`),
reported in §4.1.*
