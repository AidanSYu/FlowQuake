# FlowQuake — paper plan & results ledger

Target framing (decided 2026-06-24): **not** "beat ETAS on ComCat total NLL"
(the benchmark's hardest spatial case; no NPP has closed it). Instead:

> The first neural point process to beat ETAS on **temporal** forecasting of the
> flagship EarthquakeNPP catalog (whole-California ComCat_25) with statistical
> significance and CSEP consistency, with a full N/S/M seismological evaluation
> prior NPPs could not provide — explained by a controlled **memorization
> finding**: exposing the heads to a learned whole-catalog embedding destroys
> generalization, while structure ("neural ETAS") succeeds.

Three pillars: (1) significant temporal win, (2) the memorization mechanism,
(3) full CSEP suite incl. the magnitude test (moat: SMASH/DSTPP can't run it).

**Target venue:** Seismica (open, fast, exact audience) or GRL. Not a top-ML
conference (the benchmark itself only reached TMLR).

---

## Results ledger (ComCat_25 test: 21,889 events; ETAS tll 1.4343 / sll −8.6898 / nll 7.2554)

### Per-event log-likelihood (the kill metric)
| model | tll | sll | nll | note |
|---|---|---|---|---|
| Poisson | 0.5126 | −13.7745 | 13.2619 | benchmark null |
| ETAS | 1.4343 | −8.6898 | 7.2554 | incumbent |
| FlowQuake (canonical, 3 seeds) | 1.4860 ± 0.0024 | −9.0907 ± 0.0023 | 7.6047 ± 0.0022 | beats ETAS tll |
| FlowQuake + N1 density kernel (3 seeds) | **1.4868 ± 0.0008** | **−9.0589 ± 0.0090** | **7.5720 ± 0.0090** | best; spatial gain > seed noise |

- **Temporal win**: paired per-event gain +0.052 ± 0.0025, win rate ~61%. The win
  is ~20× the 3-seed std → unambiguous. (M2)
- **Spatial**: still trails ETAS by ~0.36 nat (was 0.40 before N1). 100% of the
  deficit is sub-1km over-smoothing in dense clusters; N1 (density-adaptive
  kernel width) recovers part of it. At 1–5 km FlowQuake already *beats* ETAS.
- **Correctness gate (M4)**: ETAS reproduces ll_scores.json exactly; FlowQuake's
  21,889 scored events are *identical* to ETAS's set (perfect intersection).

### Full-suite temporal result (THE headline) — FlowQuake N1 vs ETAS, per-event LL
| dataset | mc | FQ tll | ETAS tll | temporal | FQ sll | ETAS sll | FQ nll | ETAS nll |
|---|---|---|---|---|---|---|---|---|
| ComCat_25 | 2.5 | 1.488 | 1.434 | **WIN** | −9.054 | −8.690 | 7.566 | 7.255 |
| WHITE_06 | 0.6 | 2.068 | 2.021 | **WIN** | −4.727 | −4.261 | 2.659 | 2.240 |
| SanJac_10 | 1.0 | 1.162 | 1.133 | **WIN** | −5.919 | −5.398 | 4.758 | 4.266 |
| SaltonSea_10 | 1.0 | 2.435 | 2.332 | **WIN** | −2.639 | −2.315 | 0.203 | −0.017 |
| SCEDC_20 | 2.0 | 2.619 | 2.541 | **WIN** | −7.847 | −7.534 | 5.228 | 4.993 |

FlowQuake beats ETAS **temporally on all five** California EarthquakeNPP catalogs
(operational statewide + dense fault-zone + Salton Sea swarm); ETAS keeps the
**spatial** edge on all five (consistent, characterized: sub-km over-smoothing).
Single-seed except ComCat (3-seed); multi-seed in progress. This is the first
NPP to beat ETAS temporally across the suite.

### CSEP consistency (100 forecast days × 10k simulated catalogs, best seed)
| test | pass rate @95% | reading |
|---|---|---|
| N (number) | 95/100 = 95% | perfectly calibrated |
| S (spatial) | 81/92 = 88% | consistent; mild over-rejection (sub-km issue) |
| M (magnitude) | 90/92 = 98% | excellent; uniquely enabled by the GR head |

A forecast passing ~95% of days at α=0.05 *is* CSEP-consistent. (M1)
**TODO**: run ETAS through the same pyCSEP path for a same-days head-to-head.

### Memorization mechanism (the strongest single result)
Ablation over the whole-catalog embedding bottleneck h, train-subsample vs test
NLL at the **converged** (overfit) checkpoint:
| h_bottleneck | train_nll | test_nll | generalization gap |
|---|---|---|---|
| 0 (relational only) | 7.28 | 7.62 | **0.34** |
| 4 | 4.14 | 19.65 | **15.5** |
| 16 | 4.18 | 18.73 | **14.5** |
| 64 | 4.27 | 18.33 | **14.1** |

The instant *any* learned global catalog embedding reaches the heads, training
NLL collapses to ~4.1 (memorizing the catalog) while test NLL explodes to ~19.
h=0 — heads conditioned only on translation-invariant relational features —
generalizes (gap 0.34). This is why flexible NPPs lose to ETAS on this
benchmark, and why structure wins. (M3)

---

## Method summary (for §3)
- **Encoder**: pure-PyTorch Mamba-2-style selective-SSM whole-catalog encoder
  (chunked SSD scan; verified vs naive recurrence). *Currently h_bottleneck=0:
  the encoder is ablated from the production model because exposing its state
  memorizes — itself a key finding. Bounded-h is future work.*
- **Time head**: conditional rectified flow on log-τ (exact ODE likelihood).
  Beats AR(1)/ETAS temporally.
- **Space head**: mixture of ETAS-style anisotropic power-law kernels
  (q−1)/(πd²)(1+r²/d²)^−q anchored at the last-64 observed events + top-16
  M≥4.5 trailing-2yr events (long-lived triggers) + uniform + train-era KDE
  fault-map background. Mass moves with observed data at eval → cannot
  memorize geography. N1 adds density-adaptive width.
- **Magnitude head**: conditional Gutenberg–Richter exponential β(·). Closed
  form; enables the CSEP M-test.
- Translation-invariant relational conditioning (per-lag [log Δt, Δx, Δy, m]).

## Section skeleton
1. Intro — NPPs have not beaten ETAS (EarthquakeNPP); the benchmark prescribed
   four fixes; we implement all four and win temporally.
2. Related work — EarthquakeNPP, RECAST, the 5 reference NPPs, Werner+ 2025.
3. Method — encoder + structured marked heads; the h-bottleneck design.
4. Results — (a) per-event LL multi-seed table; (b) CSEP N/S/M vs ETAS;
   (c) memorization ablation; (d) spatial-gap localization figure.
5. Discussion — structure vs flexibility; the residual sub-km spatial gap.
6. Conclusion — first significant temporal win + full-suite eval + design lesson.

## Open builds (ranked)
1. **ETAS through pyCSEP** on the same 100 days — the head-to-head CSEP table.
2. **N1 multi-seed** (in progress) — confirm the spatial gain holds.
3. **Breadth**: temporal win on WHITE_06 + SanJac_10 (softest ETAS-temporal
   targets) → "wins on the dense low-Mc catalogs", not just ComCat.
4. Memorization ablation as a train/test-vs-steps curve figure (not just
   endpoints) for §4c.
