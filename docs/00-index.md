# docs/ — table of contents

Navigation for the FlowQuake first-principles primer. Start at
[`../PRIMER.md`](../PRIMER.md) for the framing, the dependency graph, the study plans,
the master notation table, the glossary, the memorizable numbers, the honest summary,
and [what this primer does not cover](../PRIMER.md#what-this-primer-does-not-cover).
This file is the map: every chapter, every H2 section.

Companion documents at the repository root: [`../STACK.md`](../STACK.md) (code
walkthrough), [`../MANUSCRIPT.md`](../MANUSCRIPT.md) (the paper),
[`../results/CLAIMS.md`](../results/CLAIMS.md) (claim-to-artifact audit),
[`../WORKING.md`](../WORKING.md) (open items).

**Read order.** 1 and 2 are the roots. 3 needs 1 + 2. 4 needs 1 + 3. 5 needs 1 + 3.
6 needs 1–3 and forward-references 7. 7 needs 1 (+ 3 lightly). 8 needs all of 1–7.
9 needs 8 immediately before Tier 5.

---

## Chapters

### [1. Point processes from first principles](01-point-processes.md)

*Conditional intensity, the compensator, the likelihood derived twice, time rescaling,
simulation, Hawkes branching, marked processes. The root of everything.*

- [What this chapter buys you](01-point-processes.md#what-this-chapter-buys-you)
- [Prerequisites](01-point-processes.md#prerequisites)
- [1. The object: counting processes, histories, and simplicity](01-point-processes.md#1-the-object-counting-processes-histories-and-simplicity)
- [2. The conditional intensity](01-point-processes.md#2-the-conditional-intensity)
- [3. The hazard view](01-point-processes.md#3-the-hazard-view)
- [4. The likelihood, derived twice](01-point-processes.md#4-the-likelihood-derived-twice)
- [5. Time rescaling](01-point-processes.md#5-time-rescaling)
- [6. Simulation](01-point-processes.md#6-simulation)
- [7. A ladder of processes](01-point-processes.md#7-a-ladder-of-processes)
- [8. Hawkes processes](01-point-processes.md#8-hawkes-processes)
- [9. Marked point processes](01-point-processes.md#9-marked-point-processes)
- [10. Why log-likelihood](01-point-processes.md#10-why-log-likelihood)
- [11. Worked example A — a three-event Hawkes, both likelihood forms](01-point-processes.md#11-worked-example-a--a-three-event-hawkes-both-likelihood-forms)
- [12. Worked example B — time rescaling](01-point-processes.md#12-worked-example-b--time-rescaling)
- [How this shows up in FlowQuake](01-point-processes.md#how-this-shows-up-in-flowquake)
- [Common misconceptions](01-point-processes.md#common-misconceptions)
- [Questions a professor will ask](01-point-processes.md#questions-a-professor-will-ask)
- [Further reading](01-point-processes.md#further-reading)

### [2. Seismology for the point-process modeller](02-seismology.md)

*What an earthquake and a catalog physically are; magnitude scales; completeness;
Gutenberg–Richter, Omori–Utsu, productivity and Båth; triggering; the eleven catalogs.*

- [What this chapter buys you](02-seismology.md#what-this-chapter-buys-you)
- [Prerequisites](02-seismology.md#prerequisites)
- [1. What an earthquake physically is](02-seismology.md#1-what-an-earthquake-physically-is)
- [2. Magnitude scales, saturation, and why mixing them is dangerous](02-seismology.md#2-magnitude-scales-saturation-and-why-mixing-them-is-dangerous)
- [3. How a catalog is made](02-seismology.md#3-how-a-catalog-is-made)
- [4. Completeness magnitude m_c](02-seismology.md#4-completeness-magnitude-m_c)
- [5. Gutenberg–Richter, and the b-value done properly](02-seismology.md#5-gutenbergrichter-and-the-b-value-done-properly)
- [6. Omori–Utsu, and why the decay is a power law](02-seismology.md#6-omoriutsu-and-why-the-decay-is-a-power-law)
- [7. Productivity scaling and Båth's law](02-seismology.md#7-productivity-scaling-and-båths-law)
- [8. Foreshocks, aftershocks, swarms — and declustering](02-seismology.md#8-foreshocks-aftershocks-swarms--and-declustering)
- [9. Triggering: static, dynamic, and why space is anisotropic](02-seismology.md#9-triggering-static-dynamic-and-why-space-is-anisotropic)
- [10. Spatial structure and background models](02-seismology.md#10-spatial-structure-and-background-models)
- [11. Tectonic regimes, and whether one model should transfer](02-seismology.md#11-tectonic-regimes-and-whether-one-model-should-transfer)
- [12. The catalogs FlowQuake actually uses](02-seismology.md#12-the-catalogs-flowquake-actually-uses)
- [Worked example A — a b-value by hand, with and without the binning correction](02-seismology.md#worked-example-a--a-b-value-by-hand-with-and-without-the-binning-correction)
- [Worked example B — how many M ≥ 4 aftershocks does an M6.5 produce?](02-seismology.md#worked-example-b--how-many-m--4-aftershocks-does-an-m65-produce)
- [Worked example C — moment to magnitude, both conventions](02-seismology.md#worked-example-c--moment-to-magnitude-both-conventions)
- [How this shows up in FlowQuake](02-seismology.md#how-this-shows-up-in-flowquake)
- [Common misconceptions](02-seismology.md#common-misconceptions)
- [Questions a professor will ask](02-seismology.md#questions-a-professor-will-ask)
- [Further reading](02-seismology.md#further-reading)

### [3. ETAS, derived and dissected](03-etas.md)

*The incumbent, transcribed from this repository's code: the nine parameters, the two
normalizers, the branching ratio, the EM inversion, six lines of attack, and the
strict-superset proof.*

- [What this chapter buys you](03-etas.md#what-this-chapter-buys-you)
- [Prerequisites](03-etas.md#prerequisites)
- [1. The problem ETAS was invented to solve](03-etas.md#1-the-problem-etas-was-invented-to-solve)
- [2. Lineage](03-etas.md#2-lineage)
- [3. Building ETAS from three laws](03-etas.md#3-building-etas-from-three-laws)
- [4. The exact form used by this benchmark](03-etas.md#4-the-exact-form-used-by-this-benchmark)
- [5. Normalization, in full](03-etas.md#5-normalization-in-full)
- [6. The branching ratio for this parameterization](03-etas.md#6-the-branching-ratio-for-this-parameterization)
- [7. Fitting: the EM / stochastic-declustering inversion](03-etas.md#7-fitting-the-em--stochastic-declustering-inversion)
- [8. Practical pathologies a professor will probe](03-etas.md#8-practical-pathologies-a-professor-will-probe)
- [9. Known failure modes](03-etas.md#9-known-failure-modes)
- [10. Variants and competitors](03-etas.md#10-variants-and-competitors)
- [11. Why ETAS is hard to beat — argued, not asserted](03-etas.md#11-why-etas-is-hard-to-beat--argued-not-asserted)
- [12. How FlowQuake's neural-ETAS head generalizes ETAS](03-etas.md#12-how-flowquakes-neural-etas-head-generalizes-etas)
- [13. Worked example](03-etas.md#13-worked-example)
- [14. How this shows up in FlowQuake](03-etas.md#14-how-this-shows-up-in-flowquake)
- [15. Common misconceptions](03-etas.md#15-common-misconceptions)
- [16. Questions a professor will ask](03-etas.md#16-questions-a-professor-will-ask)
- [17. Further reading](03-etas.md#17-further-reading)

### [4. Neural density estimation, normalizing flows, and flow matching](04-flows-and-density-estimation.md)

*Where the normalizer comes from; change of variables; CNFs; the flow-matching theorem;
`sigma_min`; RK4 step-count convergence; why the spatial and magnitude heads are not
flows.*

- [What this chapter buys you](04-flows-and-density-estimation.md#what-this-chapter-buys-you)
- [Prerequisites](04-flows-and-density-estimation.md#prerequisites)
- [1. The density estimation problem: normalization is the entire difficulty](04-flows-and-density-estimation.md#1-the-density-estimation-problem-normalization-is-the-entire-difficulty)
- [2. Change of variables, and why the Jacobian is the whole design problem](04-flows-and-density-estimation.md#2-change-of-variables-and-why-the-jacobian-is-the-whole-design-problem)
- [3. Continuous normalizing flows](04-flows-and-density-estimation.md#3-continuous-normalizing-flows)
- [4. The divergence cost, Hutchinson's estimator, and why FlowQuake pays neither](04-flows-and-density-estimation.md#4-the-divergence-cost-hutchinsons-estimator-and-why-flowquake-pays-neither)
- [5. Training CNFs the old way, and why it hurt](04-flows-and-density-estimation.md#5-training-cnfs-the-old-way-and-why-it-hurt)
- [6. Flow matching: the theorem that makes this cheap](04-flows-and-density-estimation.md#6-flow-matching-the-theorem-that-makes-this-cheap)
- [7. The specific path FlowQuake uses](04-flows-and-density-estimation.md#7-the-specific-path-flowquake-uses)
- [8. Sampling, likelihood integration, and the step-count question](04-flows-and-density-estimation.md#8-sampling-likelihood-integration-and-the-step-count-question)
- [9. Alternative neural TPP designs, and why "model `f`, not `lambda`" is substantive](04-flows-and-density-estimation.md#9-alternative-neural-tpp-designs-and-why-model-f-not-lambda-is-substantive)
- [10. Why the spatial and magnitude heads are not flows](04-flows-and-density-estimation.md#10-why-the-spatial-and-magnitude-heads-are-not-flows)
- [11. Mixture density networks, and why Gaussians are the wrong shape for aftershock distances](04-flows-and-density-estimation.md#11-mixture-density-networks-and-why-gaussians-are-the-wrong-shape-for-aftershock-distances)
- [12. Worked example 1 — an exactly solvable 1-D flow-matching model](04-flows-and-density-estimation.md#12-worked-example-1--an-exactly-solvable-1-d-flow-matching-model)
- [13. Worked example 2 — the unit conversion, term by term](04-flows-and-density-estimation.md#13-worked-example-2--the-unit-conversion-term-by-term)
- [14. How this shows up in FlowQuake](04-flows-and-density-estimation.md#14-how-this-shows-up-in-flowquake)
- [15. Common misconceptions](04-flows-and-density-estimation.md#15-common-misconceptions)
- [16. Questions a professor will ask](04-flows-and-density-estimation.md#16-questions-a-professor-will-ask)
- [17. Further reading](04-flows-and-density-estimation.md#17-further-reading)

### [5. Sequence models and selective state-space models](05-sequence-models-ssm.md)

*From `x' = Ax + Bu` to `ssm.py`: RNN gradients, linear attention, LTI systems, ZOH,
HiPPO/S4, Mamba selectivity, SSD duality, the chunked scan — and the proof that the
encoder is switched off in every production run.*

- [What this chapter buys you](05-sequence-models-ssm.md#what-this-chapter-buys-you)
- [Prerequisites](05-sequence-models-ssm.md#prerequisites)
- [1. The sequence-modelling problem](05-sequence-models-ssm.md#1-the-sequence-modelling-problem)
- [2. RNNs, and why their gradients die](05-sequence-models-ssm.md#2-rnns-and-why-their-gradients-die)
- [3. Attention, transformers, and the `L^2` wall](05-sequence-models-ssm.md#3-attention-transformers-and-the-l2-wall)
- [4. Linear attention: kill the softmax, then reassociate](05-sequence-models-ssm.md#4-linear-attention-kill-the-softmax-then-reassociate)
- [5. Linear time-invariant systems, from scratch](05-sequence-models-ssm.md#5-linear-time-invariant-systems-from-scratch)
- [6. Discretization: zero-order hold and bilinear](05-sequence-models-ssm.md#6-discretization-zero-order-hold-and-bilinear)
- [7. HiPPO and S4: why the initialization of `A` is the whole game](05-sequence-models-ssm.md#7-hippo-and-s4-why-the-initialization-of-a-is-the-whole-game)
- [8. Mamba / S6: selectivity, and the death of the convolution](05-sequence-models-ssm.md#8-mamba--s6-selectivity-and-the-death-of-the-convolution)
- [9. Mamba-2 / SSD, and the duality proof](05-sequence-models-ssm.md#9-mamba-2--ssd-and-the-duality-proof)
- [10. The chunked parallel scan, derived against `ssm.py`](05-sequence-models-ssm.md#10-the-chunked-parallel-scan-derived-against-ssmpy)
- [11. The alternative: Blelloch's associative scan](05-sequence-models-ssm.md#11-the-alternative-blellochs-associative-scan)
- [12. Streaming: prefill, step, and a state that does not grow](05-sequence-models-ssm.md#12-streaming-prefill-step-and-a-state-that-does-not-grow)
- [13. Numerics: why fp32 and log-space are non-negotiable](05-sequence-models-ssm.md#13-numerics-why-fp32-and-log-space-are-non-negotiable)
- [14. Testing a scan implementation](05-sequence-models-ssm.md#14-testing-a-scan-implementation)
- [15. The honest part: this encoder is off in every production run](05-sequence-models-ssm.md#15-the-honest-part-this-encoder-is-off-in-every-production-run)
- [Worked example: unroll `L = 4` by hand, then chunk it with `Q = 2`](05-sequence-models-ssm.md#worked-example-unroll-l--4-by-hand-then-chunk-it-with-q--2)
- [How this shows up in FlowQuake](05-sequence-models-ssm.md#how-this-shows-up-in-flowquake)
- [Common misconceptions](05-sequence-models-ssm.md#common-misconceptions)
- [Questions a professor will ask](05-sequence-models-ssm.md#questions-a-professor-will-ask)
- [Further reading](05-sequence-models-ssm.md#further-reading)

### [6. Forecast evaluation: scoring rules, information gain, and CSEP](06-evaluation-and-csep.md)

*Propriety and the Bernardo uniqueness theorem; nats into sentences; paired comparison;
calibration vs sharpness; the CSEP programme; the N/S/M tests and the repo's nonstandard
pass rule; McNemar and its power.*

- [What this chapter buys you](06-evaluation-and-csep.md#what-this-chapter-buys-you)
- [Prerequisites](06-evaluation-and-csep.md#prerequisites)
- [1. What is being scored, and by whom](06-evaluation-and-csep.md#1-what-is-being-scored-and-by-whom)
- [2. Scoring rules from first principles](06-evaluation-and-csep.md#2-scoring-rules-from-first-principles)
- [3. Information gain: turning nats into a sentence](06-evaluation-and-csep.md#3-information-gain-turning-nats-into-a-sentence)
- [4. Why paired comparison, and what the estimand is](06-evaluation-and-csep.md#4-why-paired-comparison-and-what-the-estimand-is)
- [5. Calibration and sharpness](06-evaluation-and-csep.md#5-calibration-and-sharpness)
- [6. The CSEP programme](06-evaluation-and-csep.md#6-the-csep-programme)
- [7. Gridded vs catalog-based forecasts, and why FlowQuake must simulate](06-evaluation-and-csep.md#7-gridded-vs-catalog-based-forecasts-and-why-flowquake-must-simulate)
- [8. The consistency tests, defined](06-evaluation-and-csep.md#8-the-consistency-tests-defined)
- [9. Comparative tests: R, T, W — and McNemar](06-evaluation-and-csep.md#9-comparative-tests-r-t-w--and-mcnemar)
- [10. Catalog-based CSEP: simulated catalogs as the null](06-evaluation-and-csep.md#10-catalog-based-csep-simulated-catalogs-as-the-null)
- [11. The fine print a professor will attack](06-evaluation-and-csep.md#11-the-fine-print-a-professor-will-attack)
- [12. Alarm-based evaluation: Molchan, ROC, and why likelihood won](06-evaluation-and-csep.md#12-alarm-based-evaluation-molchan-roc-and-why-likelihood-won)
- [13. Worked examples](06-evaluation-and-csep.md#13-worked-examples)
- [14. How this shows up in FlowQuake](06-evaluation-and-csep.md#14-how-this-shows-up-in-flowquake)
- [15. Common misconceptions](06-evaluation-and-csep.md#15-common-misconceptions)
- [16. Questions a professor will ask](06-evaluation-and-csep.md#16-questions-a-professor-will-ask)
- [17. Further reading](06-evaluation-and-csep.md#17-further-reading)

### [7. Statistics for dependent data: bootstraps, families, and equivalence](07-statistics-dependent-data.md)

*Why 21,889 events are ~2,469 independent ones; the block-bootstrap family and what the
stationary bootstrap estimates; block length; percentile vs BCa; the p-value floor; Holm;
family definition; TOST; McNemar; win rates; test-set hygiene.*

- [What this chapter buys you](07-statistics-dependent-data.md#what-this-chapter-buys-you)
- [Prerequisites](07-statistics-dependent-data.md#prerequisites)
- [1. The object under study](07-statistics-dependent-data.md#1-the-object-under-study)
- [2. Why `n` is a lie](07-statistics-dependent-data.md#2-why-n-is-a-lie)
- [3. The bootstrap: plug-in, and why it fails on dependent data](07-statistics-dependent-data.md#3-the-bootstrap-plug-in-and-why-it-fails-on-dependent-data)
- [4. The block bootstrap family](07-statistics-dependent-data.md#4-the-block-bootstrap-family)
- [5. Choosing the block length](07-statistics-dependent-data.md#5-choosing-the-block-length)
- [6. Confidence intervals from a bootstrap](07-statistics-dependent-data.md#6-confidence-intervals-from-a-bootstrap)
- [7. Bootstrap p-values and the resolution floor](07-statistics-dependent-data.md#7-bootstrap-p-values-and-the-resolution-floor)
- [8. Multiple comparisons: FWER, FDR, and a proof of Holm](07-statistics-dependent-data.md#8-multiple-comparisons-fwer-fdr-and-a-proof-of-holm)
- [9. What is the family? The judgement call you must defend](07-statistics-dependent-data.md#9-what-is-the-family-the-judgement-call-you-must-defend)
- [10. Equivalence testing: TOST](07-statistics-dependent-data.md#10-equivalence-testing-tost)
- [11. McNemar's test](07-statistics-dependent-data.md#11-mcnemars-test)
- [12. Paired means versus win rates](07-statistics-dependent-data.md#12-paired-means-versus-win-rates)
- [13. Model selection and test-set hygiene](07-statistics-dependent-data.md#13-model-selection-and-test-set-hygiene)
- [Worked example 1 — Holm by hand on the six regional `dT` p-values](07-statistics-dependent-data.md#worked-example-1--holm-by-hand-on-the-six-regional-dt-p-values)
- [Worked example 2 — a TOST decision by hand](07-statistics-dependent-data.md#worked-example-2--a-tost-decision-by-hand)
- [Worked example 3 — `n_eff` by hand](07-statistics-dependent-data.md#worked-example-3--n_eff-by-hand)
- [How this shows up in FlowQuake](07-statistics-dependent-data.md#how-this-shows-up-in-flowquake)
- [Common misconceptions](07-statistics-dependent-data.md#common-misconceptions)
- [Questions a professor will ask](07-statistics-dependent-data.md#questions-a-professor-will-ask)
- [Further reading](07-statistics-dependent-data.md#further-reading)

### [8. FlowQuake: the whole argument, and every joint where it can be attacked](08-flowquake-synthesis.md)

*The claim in three sentences with its qualifiers; every design decision and the
alternative not taken; the two-model structure; the memorization result in depth; the
claim inventory; the two disagreeing Holm families; out-of-time replication; transfer;
reproducibility; the ranked attack surface.*

- [What this chapter buys you](08-flowquake-synthesis.md#what-this-chapter-buys-you)
- [Prerequisites](08-flowquake-synthesis.md#prerequisites)
- [1. The claim, in three sentences](08-flowquake-synthesis.md#1-the-claim-in-three-sentences)
- [2. The design as a chain of decisions](08-flowquake-synthesis.md#2-the-design-as-a-chain-of-decisions)
- [3. The two-model structure, in full](08-flowquake-synthesis.md#3-the-two-model-structure-in-full)
- [4. The memorization result, in depth](08-flowquake-synthesis.md#4-the-memorization-result-in-depth)
- [5. The full claim inventory](08-flowquake-synthesis.md#5-the-full-claim-inventory)
- [6. Where the temporal family and the total family diverge](08-flowquake-synthesis.md#6-where-the-temporal-family-and-the-total-family-diverge)
- [7. The out-of-time 2020–2026 replication](08-flowquake-synthesis.md#7-the-out-of-time-20202026-replication)
- [8. Transfer and the "foundation model" framing](08-flowquake-synthesis.md#8-transfer-and-the-foundation-model-framing)
- [9. External dependencies and reproducibility](08-flowquake-synthesis.md#9-external-dependencies-and-reproducibility)
- [10. The attack surface, ranked](08-flowquake-synthesis.md#10-the-attack-surface-ranked)
- [11. What would have to be true for an operational replacement](08-flowquake-synthesis.md#11-what-would-have-to-be-true-for-an-operational-replacement)
- [12. Worked examples](08-flowquake-synthesis.md#12-worked-examples)
- [13. How this shows up in FlowQuake](08-flowquake-synthesis.md#13-how-this-shows-up-in-flowquake)
- [14. Common misconceptions](08-flowquake-synthesis.md#14-common-misconceptions)
- [15. Questions a professor will ask](08-flowquake-synthesis.md#15-questions-a-professor-will-ask)
- [16. Further reading](08-flowquake-synthesis.md#16-further-reading)

### [9. The question bank: 157 questions with model answers](09-viva-question-bank.md)

*Seven tiers from foundations to "what would you do next", plus five board-level worked
examples, the ten questions most likely to sink you, and a one-page cheat sheet.*

- [What this chapter buys you](09-viva-question-bank.md#what-this-chapter-buys-you)
- [Prerequisites](09-viva-question-bank.md#prerequisites)
- [1. How to use this bank](09-viva-question-bank.md#1-how-to-use-this-bank)
- [2. Tier 1 — Foundations (Q1–Q32)](09-viva-question-bank.md#2-tier-1--foundations-q1q32)
- [3. Tier 2 — ETAS (Q33–Q53)](09-viva-question-bank.md#3-tier-2--etas-q33q53)
- [4. Tier 3 — Method (Q54–Q81)](09-viva-question-bank.md#4-tier-3--method-q54q81)
- [5. Tier 4 — Evaluation and statistics (Q82–Q103)](09-viva-question-bank.md#5-tier-4--evaluation-and-statistics-q82q103)
- [6. Tier 5 — Hostile / defence (Q104–Q129)](09-viva-question-bank.md#6-tier-5--hostile--defence-q104q129)
- [7. Tier 6 — Code level (Q130–Q145)](09-viva-question-bank.md#7-tier-6--code-level-q130q145)
- [8. Tier 7 — What would you do next (Q146–Q157)](09-viva-question-bank.md#8-tier-7--what-would-you-do-next-q146q157)
- [9. Worked examples](09-viva-question-bank.md#9-worked-examples)
- [10. How this shows up in FlowQuake](09-viva-question-bank.md#10-how-this-shows-up-in-flowquake)
- [11. Common misconceptions](09-viva-question-bank.md#11-common-misconceptions)
- [12. The ten questions most likely to sink you](09-viva-question-bank.md#12-the-ten-questions-most-likely-to-sink-you)
- [13. One-page cheat sheet](09-viva-question-bank.md#13-one-page-cheat-sheet)
- [14. Further reading](09-viva-question-bank.md#14-further-reading)

---

## Quick jumps

**The derivations you must be able to produce**

- [The likelihood, derived twice](01-point-processes.md#4-the-likelihood-derived-twice) · [Time rescaling](01-point-processes.md#5-time-rescaling) · [`mu/(1−n)` two ways](01-point-processes.md#8-hawkes-processes)
- [The Aki b-value MLE and the binning correction](02-seismology.md#5-gutenbergrichter-and-the-b-value-done-properly) · [Rate-and-state → Omori with `p = 1`](02-seismology.md#6-omoriutsu-and-why-the-decay-is-a-power-law)
- [`Z_j = pi/(rho·d_j^rho)` and the branching ratio](03-etas.md#5-normalization-in-full) · [`a_eff = a − rho·gamma`](03-etas.md#6-the-branching-ratio-for-this-parameterization) · [the EM E- and M-steps](03-etas.md#7-fitting-the-em--stochastic-declustering-inversion)
- [Instantaneous change of variables](04-flows-and-density-estimation.md#3-continuous-normalizing-flows) · [the flow-matching gradient-equivalence theorem](04-flows-and-density-estimation.md#6-flow-matching-the-theorem-that-makes-this-cheap) · [`log f(tau) = log p(u) − log sigma − log tau`](04-flows-and-density-estimation.md#13-worked-example-2--the-unit-conversion-term-by-term)
- [State-space duality](05-sequence-models-ssm.md#9-mamba-2--ssd-and-the-duality-proof) · [zero-order hold](05-sequence-models-ssm.md#6-discretization-zero-order-hold-and-bilinear)
- [Log score strictly proper](06-evaluation-and-csep.md#2-scoring-rules-from-first-principles) · [information gain](06-evaluation-and-csep.md#3-information-gain-turning-nats-into-a-sentence)
- [`Var(mean)` under autocorrelation](07-statistics-dependent-data.md#2-why-n-is-a-lie) · [the stationary bootstrap is stationary](07-statistics-dependent-data.md#4-the-block-bootstrap-family) · [Holm controls FWER under arbitrary dependence](07-statistics-dependent-data.md#8-multiple-comparisons-fwer-fdr-and-a-proof-of-holm) · [TOST from intersection–union](07-statistics-dependent-data.md#10-equivalence-testing-tost)

**The uncomfortable sections — read these before an examiner does**

- [The repo runs no time-rescaling residuals](01-point-processes.md#5-time-rescaling) (§5.5) and [the `tau` floor](01-point-processes.md#questions-a-professor-will-ask) (Q13)
- [The `+0.005` half-bin constant and the doc bug](02-seismology.md#5-gutenbergrichter-and-the-b-value-done-properly) (§5.5); [Italy under Mw homogenization](02-seismology.md#2-magnitude-scales-saturation-and-why-mixing-them-is-dangerous) (§2.4)
- [ETAS's uniform background](03-etas.md#8-practical-pathologies-a-professor-will-probe) (§8.7) and [what the head's ablation actually shows](03-etas.md#12-how-flowquakes-neural-etas-head-generalizes-etas) (§12.4)
- [The ODE step-count evidence, and its limits](04-flows-and-density-estimation.md#8-sampling-likelihood-integration-and-the-step-count-question) (§8.3–8.4); [`sigma_min` accounting](04-flows-and-density-estimation.md#7-the-specific-path-flowquake-uses) (§7.3)
- [The encoder is off in every production run](05-sequence-models-ssm.md#15-the-honest-part-this-encoder-is-off-in-every-production-run) (§15)
- [The nonstandard CSEP pass criterion](06-evaluation-and-csep.md#8-the-consistency-tests-defined) (§8.5) and [the McNemar power problem](06-evaluation-and-csep.md#11-the-fine-print-a-professor-will-attack) (§11.5)
- [`mean_block = 50` is undefended](07-statistics-dependent-data.md#5-choosing-the-block-length) (§5.3); [test-set hygiene](07-statistics-dependent-data.md#13-model-selection-and-test-set-hygiene) (§13)
- [The full claim inventory](08-flowquake-synthesis.md#5-the-full-claim-inventory) (§5) and [the ranked attack surface](08-flowquake-synthesis.md#10-the-attack-surface-ranked) (§10)

---

## A note on links inside the chapters

**Fixed.** Every chapter now uses `../`-prefixed paths for repository files
(`../STACK.md`, `../flowquake/model.py`, `../runs/total_win.json`, …) and bare
filenames for sibling chapters (`03-etas.md`), so every link resolves when the file is
viewed inside `docs/`. Chapters 2, 5, 8 and 9 were originally written with
repository-root-relative paths — 433 link instances — and have been rewritten. The
*text* of every reference was always correct; only the path prefix was wrong.

If you edit a chapter, the invariant to preserve is: **repository files get `../`,
sibling chapters get no prefix, `PRIMER.md` at the root gets `../PRIMER.md`.** Every
in-page and cross-chapter anchor in `docs/` and `PRIMER.md` has been verified against
the GitHub heading-slug rule; a rename of any H2 breaks this file and the Quick-jumps
section below, so re-check after renaming a section.
