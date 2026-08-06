# MOONSHOT — the information limit of catalog-based earthquake forecasting

This file exists to stop scope collapse. `WORKING.md` tracks the current paper;
this tracks the one worth writing. If a decision looks reasonable in isolation
but is forbidden below, the decision is wrong.

---

## THE ANSWER, 2026-08-05

> **A decade of magnitude below M3 is worth +0.22 [+0.02, +0.40] nats to a
> fitted ETAS and -0.71 [-0.95, -0.46] nats to a flexible learned model, scored
> on identical target events. They differ in SIGN with probability 0.985.**

Measured on a bit-identical frame (1,673 windows, 123 scored M>=3 targets, same
grid, verified by hash), the corrected neural curve -- the checkpoint-surface
plateau, not the early-stopping artefact -- against the ETAS control:

| mc | FlowQuake | ETAS (uniform) | margin |
|---|---|---|---|
| 2.5 | -4.0427 | -5.6003 | **+1.5576** |
| 2.0 | -4.0301 | -5.4308 | +1.4007 |
| 1.5 | -4.3676 | -5.3370 | +0.9694 |
| 1.0 | -5.1194 | -5.2712 | **+0.1518** |

| | nats/decade | 95% CI | total 2.5 -> 1.0 |
|---|---|---|---|
| ETAS | **+0.2162** | [+0.0209, +0.3973] | +0.3291 [+0.0573, +0.5799] |
| FlowQuake | **-0.7135** | [-0.9512, -0.4555] | -1.0767 [-1.4264, -0.6695] |
| difference | **+0.9297** | [+0.6449, +1.1879] | |

**What this settles, and why it is neither branch of the original either/or.**
The moonshot was framed as "still rising -> capability claim; saturates ->
limits claim". The actual answer is a THIRD thing: the two model classes
disagree about the sign. The information is demonstrably present -- ETAS
converts catalog depth into forecast skill, significantly, on these very
targets. A high-capacity density model does not merely fail to convert it; it
gets worse, and it does so about 3.3x faster than ETAS improves.

So **the quantity this experiment bounds is not the information content of
catalogs. It is the ability of a model class to use it.** Any "information
limit of catalog-based forecasting" measured with a learned model alone would
have reported a limit that belongs to the model, and would have had the wrong
sign.

**The crossover is the operationally interesting number.** FlowQuake beats ETAS
by 1.5576 nats at mc 2.5 and by 0.1518 at mc 1.0: the entire advantage of the
learned model is spent within 1.5 decades of catalog depth. Linear extrapolation
puts the crossing just below mc 1.0 -- i.e. on a sufficiently deep catalog the
physics baseline would win outright. That is a falsifiable prediction and it is
the natural next experiment.

**The mechanism, recomputed -- because the published one was the SAME artefact.**
The old story was "`n_eff_cells` falls 167.8 -> 56.0 -> 32.4, the model sharpens
5x while accuracy peaks: deeper catalogs make it more confident without making
it more correct." That is **wrong, and wrong for the same reason the +0.7500
was**: 167.8 was the barely-trained mc 2.5 checkpoint being diffuse, so the
apparent sharpening was the model finishing training, not responding to catalog
depth. On the corrected plateau checkpoints the sign reverses:

| mc | FlowQuake `n_eff` | ETAS `n_eff` |
|---|---|---|
| 2.5 | 26.2 | 75.2 |
| 2.0 | 34.7 | 49.0 |
| 1.5 | 31.3 | 32.6 |
| 1.0 | 35.2 | 31.8 |

FlowQuake gets **0.75x sharper, i.e. slightly BROADER** (26.2 -> 35.2), while
ETAS sharpens **2.37x** (75.2 -> 31.8). So the real mechanism is not
overconfidence. **The learned model's spatial resolution is saturated**: adding
smaller earthquakes does not sharpen its forecast at all, while it does move it
in a direction the likelihood dislikes. ETAS meanwhile converts the same extra
events into a forecast that is both sharper AND better -- which is precisely why
the two curves separate.

1x. **The ETAS baseline is weakly identified, and the headline sentence rests
   on it.** The moonshot says "the information is demonstrably present -- ETAS
   extracts it". That reading assumes ETAS's +0.2162/decade measures
   INFORMATION. A competing explanation is ESTIMATION: ETAS gets 520 training
   events at mc 2.5 and 10,601 at mc 1.0, a **20x larger estimation sample**, so
   its forecasts could improve for reasons having nothing to do with small
   earthquakes carrying signal about large ones.

   Tested by refitting at a fixed 520-event budget
   (`scripts/etas_matched_n.py`, `runs/panel_white/etas_matched_n_uniform.json`).
   **The test is inconclusive as designed, and the reason is itself the finding:**

   | mc | arm | N | a | K | n |
   |---|---|---|---|---|---|
   | 2.5 | both (identical) | 520 | 0.344 | 0.649 | 0.99 |
   | 2.0 | matched_n | 520 | 0.587 | 0.409 | 0.99 |
   | 1.5 | matched_n | 520 | **4.3e-142** | 0.99 | 0.99 |
   | 1.0 | matched_n | 520 | **0.011** | 0.979 | 0.99 |

   At 520 events the low-mc fits COLLAPSE: the productivity exponent `a` goes to
   numerical zero -- aftershock productivity independent of magnitude, which is
   physically false -- while `K` pins at its bound. And `n = 0.99`, the branching
   barrier, in **all eight fits**. Truncating to the most recent 520 events also
   shortens the time window, so sample size and span are confounded and no clean
   attribution is possible from this design.

   What it does establish: **the control is fragile.** An ETAS that degenerates
   at 520 events is a weak foundation for a claim about what information exists,
   and a referee will say so.

   **The correct test, not yet run**, avoids starving any fit. ETAS's forecast
   uses two separable channels: (a) parameters estimated from the training
   catalog at that mc, and (b) the conditioning history fed to the intensity at
   forecast time. Gain via (b) is information; gain via (a) is estimation. A 2x2
   cross-over -- theta fitted at mc 1.0 used with mc 2.5 history, and the
   reverse -- separates them with every fit fully identified.

1w. **When a confound is removed, EVERY number derived from the confounded run
   is suspect, not just the headline.** The `n_eff` mechanism story survived the
   correction unexamined and was written into this file, minutes after the
   headline it was supporting had been overturned, because it "already made
   sense". It was caught only because the regenerated figure visibly
   contradicted its own caption. A supporting number that still fits the old
   narrative is the easiest thing in the world to carry forward.

**Scope, stated once and not softened elsewhere.** One region (WHITE / San
Jacinto), one target magnitude (M>=3), one horizon (1 day), one architecture,
ONE SEED. G1 already established that the ETAS slope is heterogeneous across
regions (I^2 ~ 70%, Q p = 0.016; 2 of 4 regions significant), so the +0.22 is
this region's number and not a universal constant. The sign DIVERGENCE is the
claim that travels; its magnitude is regional.

Artifacts: `runs/surface_white/moonshot_answer.json`,
`runs/surface_white/intervals.json`, `scripts/moonshot_answer.py`.

---

## The claim we are trying to earn the right to make

> **Each decade of magnitude below M4 contributes X ± Y nats of forecast skill
> for M≥4 earthquakes. The curve [has not saturated by the completeness limit of
> the catalog | saturates at mc Z], implying [N× more forecast information about
> large earthquakes is recoverable from microseismicity that operational
> catalogs discard | a hard information limit on catalog-based forecasting].**

Not "we beat ETAS." Beating ETAS is a *control condition* in this experiment,
not the result. The result is a **measured quantity with an error bar**.

The question underneath it, in plain language: *how much of a large earthquake
is knowable from the small earthquakes around it, and are we anywhere near that
limit?* Nobody has answered this on likelihood terms.

**It publishes either way.** Still rising at the left edge → capability claim
(forecastability is limited by catalog completeness, not by physics). Saturates
→ limits claim (here is the ceiling, measured). High-profile venues publish
bounds; they do not publish 2% method increments.

---

## The killer figure

Four region panels plus a forest plot of their slopes and the pooled estimate
(`scripts/make_pooled_figure.py`). It is multi-panel by necessity, not
preference: no catalog available gives both magnitude range and target count,
so each region carries only ~1 decade and the headline number is a pool. See
"The data plan".

- **x**: catalog completeness `mc`, descending left-to-right, from the sparsest
  usable threshold down to that region's measured Mc **plus the safety
  margin** — never below it. Points below Mc measure detection, not information
  (invariants 1 and 1e). Grids: WHITE {2.5, 2.0, 1.5, 1.0}; QTM SanJac
  {2.3, 1.8, 1.3}; QTM SaltonSea {2.5, 2.1, 1.7}; ComCat {3.5, 3.0, 2.8}.
- **y**: the **shape (S-test) score** on a fixed, mc-invariant set of target
  events — invariant to rescaling λ, so the magnitude tail cannot move it
  (invariant 1c). Level is a companion panel, never silently folded in.
- **series**: FlowQuake, and ETAS **re-inverted at that same mc**
- **bands**: 3-seed spread, and a **paired block bootstrap over windows** —
  never over events (invariant: targets arrive in sequences)
- **two arms**, both plotted: matched-N and matched-window (see Design below)
- **forest panel**: per-region slopes with CIs, the DerSimonian-Laird pooled
  estimate, and `I²` / `p_Q`. If the regions disagree, say so on the figure —
  a region-dependent limit is a different claim from a universal one, and it
  is the more interesting of the two.
- **M_TARGET is 3.0 for the dense panels and 4.0 for ComCat.** Legal, since
  invariant 1 fixes the target set within a curve rather than across panels,
  and it doubles as a check on target-magnitude dependence. Label it.

If that figure does not exist, there is no paper. Everything in the run ledger
is either producing that figure or defending it.

---

## Design invariants — violating any of these voids the result

1. **The target set never changes.** Every point on every curve is scored on the
   *same* events: `magnitude >= M_TARGET` in the *same* test window. `mc` changes
   only what the model *sees* (history and training), never what it is *scored
   on*. Scoring on mc-dependent target sets measures nothing and is the single
   easiest way to fake this result.

1b. **The metric must be the M≥M_TARGET sub-process likelihood, NOT per-event
   next-event density.** This is the invariant that is easiest to get wrong and
   fatal if missed. `tll` as currently computed is the log density of the
   waiting time to the next event *above mc* — its units and magnitude depend on
   mc (at mc 0.5 events are minutes apart, at mc 2.5 hours apart), so **nats/event
   is not comparable across the x-axis of our own figure.** Comparing them
   directly would manufacture an enormous spurious slope.

   The comparable quantity is the intensity of the **thinned target process**:

   ```
   λ_tgt(t, x, y)  =  λ_total(t, x, y | H^mc_<t)  ×  P(m ≥ M_TARGET | H^mc_<t)
   log L_tgt       =  Σ_targets log λ_tgt(t_i, x_i, y_i)  −  ∫∫ λ_tgt dt dx dy
   ```

   Both an mc-0.5 model and an mc-2.5 model are then forecasting **the same
   physical quantity** — the rate of M≥M_TARGET events per day per km² — and their
   likelihoods are on one scale. This is also exactly what CSEP scores, and it
   yields `P(M≥6 in 30d)` for free.

   Implementation is simulation-based (`flowquake/target_process.py`), reusing the
   validated forward simulator rather than hand-rolling a compensator.
   `runs/mw_robustness.json` already fell into the naive version of this trap —
   it compares `comcat_mc25_production_on_Mge3` against `comcat_mc30_retrained`
   across different test sets and different ETAS baselines. Do not cite that file
   as a scaling result until it is recomputed on the target process.
1c. **Decompose the score, or the magnitude tail will masquerade as the result.**
   Found empirically in the synthetic validation (`curve2`, 2026-08-02): with
   under-trained models the target-process score *degraded* monotonically as mc
   fell — **and it did so in the matched-N arm too**, where N was pinned at 2432.
   Sample size cannot explain that. The cause is that
   `λ_tgt = λ_total × P(m ≥ M_TARGET)`, and at lower mc the GR head must
   extrapolate further up the magnitude tail to reach M_TARGET, so tail
   mis-calibration enters the score with an mc-dependent weight.

   This is a real confound that will be present at full training, just smaller.
   Every point must therefore report the decomposition:

   ```
   ll_tgt = [rate term]   how many M>=M_TARGET events, when
          + [spatial term] where
          + [tail term]    P(m >= M_TARGET | history), the mc-sensitive one
   ```

   And each curve must be run a second time with the magnitude tail **held
   fixed** at the empirical training-era GR b-value instead of the learned head.
   If the slope survives that substitution, it is forecast information. If it
   does not, the curve is measuring magnitude-head calibration and the claim
   collapses. Treat the fixed-tail variant as the primary result and the
   learned-tail variant as the ablation, not the reverse.

   **Fixing the tail is not sufficient — the fix has to be structural.** The v2
   validation held b fixed across the grid and still produced a fake slope,
   because *any* residual error in b enters with mc-dependent weight:
   `d ln λ_tgt / d b = −ln10 · (M_TARGET − mc)`. A b error of 0.13 (which v2 had,
   see the run ledger) over-forecast by 1.16× at mc 2.5 and 1.83× at mc 1.0 —
   monotone in mc, indistinguishable from signal. b always carries sampling
   error, so this is a permanent hazard, not a one-off bug.

   The metric is therefore split into its CSEP **N-test / S-test** components,
   `poisson_ll_parts`:

   ```
   ll = [−Λ + N log Λ]          level  (N-test: total count)
      + [Σ n_c log(λ_c/Λ)]      shape  (S-test: normalised density)
      − log-factorial
   ```

   **`shape` is the PRIMARY curve metric**, because it is invariant to
   `λ → cλ` and therefore cannot be moved by a mis-specified tail *at all* —
   not "less", exactly zero. `level` is reported alongside it: a level slope
   is the diagnostic for a tail problem and must never be quietly dropped.
   Measured on v2, the level slope was −0.19 (informative) and −0.14 (null),
   i.e. common-mode, exactly as this predicts.

1d. **Equalise the rate field's Monte-Carlo precision across mc, and always run
   the null control.** Found by the ground-truth validation, which is exactly
   what it was for. The null catalog — whose sub-threshold events are pure
   Poisson noise and which MUST give a flat curve — instead sloped at **+0.32 to
   +0.37 nats/decade**, 45% of the informative slope, and it did so in the
   matched-N arm where N is pinned. So it is not sample size.

   The cause is the estimator. A noisier λ scores worse under a Poisson
   likelihood (Jensen), so skill *appears* to improve as mc falls with no
   information involved.

   **The first fix matched the wrong quantity, and this is the correction.** It
   equalised the ABSOLUTE variance of λ. What biases a Poisson log-likelihood is
   the gap between `E[log λ]` and `log E[λ]`, and that is set by the RELATIVE
   variance. Writing `T` for the total simulated events behind the field and
   `p_c` for a cell's share:

   ```
   λ_c = w · C_c / n_sims,        C_c ~ Poisson(T p_c)
   E[λ_c]   = w T p_c / n_sims
   Var(λ_c) = w² T p_c / n_sims²
   Var/E²   = 1 / (T p_c)                    ← w and n_sims both cancel
   ```

   Resolution depends on **T alone**. And `T = n_sims × (events per simulation)`
   grows directly with the catalog rate above mc, so under a flat `n_sims` it
   varied 14-fold across a single grid. Measured on the real WHITE catalog:

   | mc | 2.5 | 2.0 | 1.5 | 1.0 |
   |---|---|---|---|---|
   | T | 93 | 234 | 734 | 1319 |
   | shape/target | −11.669 | −10.822 | −9.133 | −7.204 |

   That is **+3.02 nats/decade on the informative arm and +2.62 on a surrogate
   null whose true slope is zero by construction** — by far the largest bias
   found in this pipeline. A controlled experiment (same observations, same true
   density, only T varied) reproduced a **+4.54** nat shift across that T range
   against a measured **+4.53**. The entire slope was Monte-Carlo resolution.

   Residual bias against the exact density, in nats per target event:

   | T | 93 | 234 | 734 | 1319 | 5,000 | 20,000 | 100,000 |
   |---|---|---|---|---|---|---|---|
   | bias | −4.98 | −3.02 | −1.04 | −0.44 | −0.10 | −0.03 | −0.003 |

   So `scaling_curve.sims_for_matched_resolution` fixes **T = 20,000 simulated
   events per window** at every point, choosing `n_sims` inversely to the event
   rate. Cost becomes identical across the curve rather than exploding at low
   mc. `scripts/etas_by_mc.py` does the same, so the ETAS control is matched to
   the neural curve. **On by default; `--no-match-precision` reproduces the
   biased estimator.**

   The general lesson: **a Monte-Carlo estimate inside a non-linear score is
   biased by its own sample size.** Any quantity that changes the effective
   sample size across the x-axis — event counts, simulation counts, grid
   occupancy — will masquerade as signal. Verify by scoring the SAME truth at
   two sample sizes and confirming the difference vanishes.

   Even so, the null control is not optional. Report the null slope alongside
   every curve and treat the *excess over null* as the finding. A curve without
   its null is uninterpretable.

   **And the null must be footprint-matched, or it is not a null.** The v2 null
   replaced sub-threshold events with events drawn uniformly over the bounding
   box. That is not zero information, it is *negative* information: the small
   events of a branching catalog are aftershocks and sit on the same structures
   as the targets (measured spatial spread 121 km, against 119 km for the
   targets), while uniform noise spreads over 231 km. Training on it drags the
   spatial density away from where targets actually occur, and the arm lost
   0.47 nats/decade of shape score from that alone — 3.5× the real effect,
   with the wrong sign.

   **And the null is not expected to be FLAT.** This took three attempts to get
   right. Any forecaster that *uses* the small events must be harmed when their
   coupling to the targets is destroyed — a triggering model reads decoupled
   events as real triggers and forecasts on them. Measured on the WHITE
   surrogate, the fitted magnitude-productivity exponent collapses from ~0.35 to
   ~0.02: the model correctly learns that parent magnitude no longer predicts
   offspring and goes magnitude-blind. So a "zero-information null" in the sense
   of *the model is unaffected* does not exist for a model that conditions on
   these events, and demanding `|null slope| < 0.10` would reject a null that is
   working perfectly.

   The null supplies a **counterfactual**, not a zero: *what if these events
   carried no coupling?* The finding is the **difference**, with its interval
   from a paired block bootstrap over windows — resampled once per replicate and
   applied to both arms, so the difference is a within-window contrast.

   One trap inside the trap: the surrogate must preserve the per-era event
   counts. A single circular shift across the whole span preserves the total but
   redistributes it, because real small events cluster in time. That handed the
   control **26% more training data** at mc 1.5 (4,871 against 3,864). Pass
   `era_bounds` so each era keeps exactly the count it started with.

   The correct control is `make_decoupled_null`: graft the targets onto an
   **independent realisation's** small events. Same generator, same parameters,
   different seed, so count, b-value, Omori clustering and spatial footprint
   all match (measured 120.9 km against 120.9 km), and the *only* thing removed
   is the coupling to this particular target set. The uniform null is retained
   as a separate, clearly-labelled robustness arm — "how badly does the pipeline
   degrade on actively misleading events?" is a real question, just not this one.

1i. **The ETAS simulator was double-thinning the first generation.** Found by
   building an analytic rate field and disagreeing with the simulator by a
   factor of 2.7 that did not shrink with `n_sims` — so not Monte-Carlo noise.

   `simulate_etas` drew `k ~ Poisson(w_i)` for each historical parent, then
   passed each parent **repeated k times** into `offspring()`, which drew its
   own Poisson. The expected first generation was therefore `Σ wᵢ²` rather than
   `Σ wᵢ`.

   **That is not a uniform error, and the way it is non-uniform is the worst
   possible one here.** Squaring *suppresses* parents with `wᵢ < 1` and
   *inflates* those with `wᵢ > 1`. Small events over short windows have
   `wᵢ ≪ 1` and large events have `wᵢ > 1`, so the bug systematically crushed
   the contribution of small earthquakes and exaggerated that of large ones —
   a direct distortion of the magnitude dependence of triggering, which is the
   single quantity this whole project exists to measure. Measured on real
   history: 0.102 against an analytic 0.277.

   It also survived a test. `test_history_drives_aftershocks` asserted
   `loud > 3 * quiet` using one M6 trigger, whose `w = 8.1` the bug inflated to
   ~65 — so the assertion passed *because* of the defect, in the one regime
   where the defect happens to help. A loose assertion in the wrong regime is
   worse than no test: it certifies the opposite of what it appears to.

   The consequence is the one that cost the most time. With triggering crushed,
   the constant background dominated, the forecast came out nearly flat across
   windows, and `corr(n_expected, n_observed)` sat at about zero on real data.
   That reads as *"ETAS has no forecast skill in this region"* — a statement
   about seismology — when it was a statement about a `np.repeat`. It was only
   separable by computing the same quantity a second way.

   Fixed, and now checked against the closed form: the simulator must come in
   slightly ABOVE the first-generation analytic value, since it adds later
   generations. Measured ratio 1.21, converging in `n_sims`.

   **Every ETAS number produced before this is void**, including the G1
   comparison and the `n = 0.990` branching-ratio observation.

   `flowquake.etas_fit.etas_rate_field` now computes the ETAS forecast in
   closed form, which is both exact and far cheaper than the ~165M simulations
   per curve point that matched resolution would otherwise demand at a 1-day
   horizon. It omits generations triggered *inside* the window — the standard
   first-generation approximation, measured at ~21% here, and stated rather
   than assumed.

1j. **The ETAS control's own fit quality improves with mc, and that is a
   confound.** Every inversion on WHITE returns the branching ratio pinned at
   the `max_branching` barrier. Raising the cap does not release it — the
   unconstrained fit runs to 5.0 and keeps improving — so the barrier is not
   distorting anything; the likelihood genuinely wants a runaway.

   The cause is weak identifiability, not a bug. At mc 2.5 there are 520
   training events, and the likelihood surface is nearly flat along a ridge in
   `(K, a)`:

   | background | cap | n | K | a | ll |
   |---|---|---|---|---|---|
   | uniform | 0.99 | 0.990 | 0.642 | 0.351 | −5221.4 |
   | uniform | 5.0 | 5.000 | 3.790 | 0.242 | −5158.1 |
   | smoothed | 0.99 | 0.989 | 0.972 | 0.017 | −4882.8 |
   | smoothed | 5.0 | 5.000 | 2.847 | 0.431 | −4866.9 |

   A 5× change in branching ratio buys 16 nats. The smoothed background is
   worth **338 nats** over uniform, which is a large result for gate G1 on its
   own — a uniform background on a narrow fault zone forces the model to
   explain spatial clustering as triggering, because it has no other way.

   The confound for the curve: **ETAS is better identified at low mc simply
   because there are more events**, so its skill would rise with mc even if the
   added events carried no new information. That is the same sample-size effect
   the matched-N arm isolates for the neural model, and the ETAS control
   currently has no matched-N equivalent. Either give it one, or report the
   ETAS curve as confounded and lean on the neural matched-N arm for the
   information claim. Report the fitted `n` at every point and flag any that
   sit on the barrier.

1k. **A single "nats per decade" slope is the wrong summary if the curve
   saturates — and the first clean measurement says it does.** WHITE, ETAS
   probe, 1-day horizon, paired block bootstrap over 1,673 windows:

   | comparison | Δ shape/target | 95% CI | |
   |---|---|---|---|
   | mc 2.5 → 2.0 | **+0.140** | [+0.018, +0.263] | **significant** |
   | mc 2.0 → 1.0 | −0.063 | [−0.292, +0.165] | not significant |
   | linear slope | +0.042 | [−0.153, +0.235] | spans zero |

   Deepening the catalog from mc 2.5 to 2.0 measurably improves the M≥3
   forecast. Going further, all the way to mc 1.0, adds nothing detectable.
   Fitting one line through that shape averages a real gain together with a
   flat tail and reports approximately zero — so the headline number would be
   **"no effect"** when the data actually say **"a real effect that saturates
   by mc 2.0."** Those are different papers, and the second is the more
   interesting one.

   So report the CURVE and its saturation point, not a single slope. Keep the
   per-decade figure only for the range where the curve is still rising, and
   say where it stops.

   Two methodological notes that generalise:

   - **Use paired differences, not per-point intervals.** The individual points
     have CIs about 0.9 nats wide, which would make everything here look
     indistinguishable. The paired bootstrap resamples windows ONCE per
     replicate and applies them to every mc, cancelling the shared window
     effects, and the differences come out 7× tighter. The slope is a
     within-window contrast; scoring it with between-point uncertainty throws
     away most of the power.
   - The saturation is exactly the outcome the charter said it would accept.
     Gate G3 exists to make that publishable rather than something to be
     engineered away.

   **Caveats, stated because this is one measurement:** one region, one probe
   (ETAS, not FlowQuake), one seed, and the ETAS control carries the
   identifiability confound of invariant 1j. It is a preliminary result, not
   the paper.

1l. **Integrate the kernel over the cell; do not sample it at the centre.**
   Found while working through the bias audit's unadjudicated candidates. The
   analytic ETAS field evaluated the spatial density at each cell centre and
   multiplied by the cell area. That is midpoint quadrature, and it is only
   valid when the kernel is smooth across a cell. It is not: the spatial scale
   fitted on WHITE is **d = 1.0 km against 2 km cells**.

   | parent position | midpoint / exact |
   |---|---|
   | at the cell centre | **1.94** |
   | 0.7 km off centre | 1.26 |
   | adjacent cell | 0.84 |
   | 4 km away | 0.92 |
   | 10 km away | 0.99 |

   So the field was over-concentrated into each parent's own cell and depleted
   around it — a distortion of the SHAPE, which is the primary metric, not of
   the level where it could be argued away. `_subgrid_correction` replaces
   midpoint with a 4x4 sub-grid over the 7x7 block around each parent, which
   brings the parent's own cell from 1.94 to 1.015 and the whole field to
   within 0.1% of fine quadrature (total-variation distance < 0.02). Cost is
   parents x 49 x 16, negligible beside the full cells x parents product.

   The general rule: **whenever the model's spatial scale is comparable to or
   smaller than the grid cell, the discretisation is part of the model.** Check
   `d` against `bin_km` before trusting any gridded intensity.

1m. **The water level is not neutral — report the curve's sensitivity to it.**
   The floor exists to keep `log(0)` out of the score and is standard CSEP
   practice, but it adds a term that depends on how CONCENTRATED the forecast
   is. Forecast concentration changes with mc, so the floor becomes an
   mc-dependent contribution. Measured on synthetic fields at MATCHED
   simulation budget, the gap between a diffuse and a sharp field:

   | floor_frac | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
   |---|---|---|---|---|
   | sharp − diffuse | 0.608 | 0.830 | 1.053 | 0.835 |

   A 0.45-nat swing from a parameter with no physical meaning — the same order
   as the signal. Note this survives matched `T`: equalising the simulation
   budget equalises resolution, not concentration.

   `score_window` therefore carries `ll_shape_by_floor` for
   {1e-4, 1e-3, 1e-2} on every window. **A slope is only reportable if it
   survives that sweep**, and the sweep goes in the paper rather than a single
   chosen floor.

1n. **A frame must record the absolute time its windows are measured from.**
   `start_days` is a number of days since *something*, and the frame never said
   what. Every consumer recomputed `t_days` from its own catalog's first event —
   which is fine until two catalogs being compared do not share one.

   They do not. The surrogate null shifts the earliest small event, so its
   first-event time sits **1.7 hours** after the informative arm's. The frame is
   copied between arms (correctly — the target set and grid must be identical),
   so the null arm was being forecast **7% of a 1-day window out of step** while
   the informative arm stayed correctly aligned. An asymmetric misalignment
   between the two arms of a comparison biases the comparison itself, not merely
   its noise.

   `build_frame` now records `t0`, and `etas_by_mc` uses it rather than its own
   catalog's first event, warning loudly if an old frame lacks it. **RV7 and RV8
   null-arm numbers are affected and must be recomputed.**

   The general form: **any quantity expressed as an offset must travel with its
   origin.** A relative coordinate shared between two datasets is a bug waiting
   for the datasets to disagree.

1o. **Absolute times live in float64; only differences may be narrowed.** The
   time axis is days since the catalog epoch, so by the test era its values are
   3000–6600. float32 has a **21–42 second quantum** there — 2400× coarser than
   the `TAU_FLOOR_DAYS = 1e-7` (~9 ms) floor the rest of the code carefully
   clamps to. The simulator's `t_last` was float32 while everything around it
   (`t_buf`, `start_t`, `end_t`, `t_ref`) was float64.

   Two consequences. `t_next = t_last + tau` silently discards any `tau` below
   half a quantum, so an event drawn 10 s after its predecessor advances
   simulated time by exactly zero. And `lastk_from_bufs` differences float32
   `t_last` against float64 `t_buf`: at the first sampling step of a window both
   hold the *same* event, so the lag must be identically zero, but rounding makes
   it ±21 s at random (50.4% negative, clamped back to the floor; the rest
   fictitious). `log(21 s)` vs `log(9 ms)` is a **7.8-log-unit swing** in a
   feature normalised by `log_tau_std ≈ 2.8` — about **2.8σ of pure noise**
   injected into the single most informative short-term-triggering input, on the
   one step that decides whether the window produces any events at all.

   This one is **noise, not bias**: the rounding is a property of the day number,
   not of mc, so it cannot manufacture a slope. That is why it is recorded
   separately from 1c/1d/1i.

   **Measured consequence — and it is far smaller than the 2.8σ suggests.** A
   paired comparison on the same checkpoint, same 30 target-bearing windows,
   same torch seed, 400 sims, float64 against a reverted float32 copy, with a
   same-code/different-seed run as the Monte-Carlo noise floor:

   | | float32 (defect) | different seed (noise floor) |
   |---|---|---|
   | bit-identical fields | 19/30 | 0/30 |
   | total expected count | **−1.005%** | −0.713% |
   | spatial shape, mean TV | **0.105** | 0.853 |

   The defect moves the forecast **1.4× the noise floor on count and 0.12× on
   shape** — *below* the sampling scatter the estimator already tolerates. The
   fix is right because the code was wrong and the fix is free, **not** because
   it moved a result. Committed CSEP/N-test artifacts shift within Monte-Carlo
   noise, so pass rates like 95/100 are unaffected in substance even though they
   are no longer bit-reproducible.

   **The lesson is the error I made writing this up.** I measured 2.8σ in a
   *normalised input feature* and reported it as the impact. Output sensitivity
   is a different quantity: the rejection sampler already conditions the first
   event on landing after the window opens — typically hours after `t_last` — so
   a ±21 s perturbation of an already-long gap barely moves the conditional. *An
   input-space σ is not an output-space effect; size the output or say nothing
   about size.* This is the exact mirror of the mistake in 1p, where I
   under-scoped a defect by looking only at the inter-arm difference. Both come
   from reasoning about a mechanism instead of measuring its consequence.

   The **accumulated-drift** version of this concern was measured and
   **refuted**: 3e-6 relative over 200 steps and non-monotone in mc, because the
   elapsed-time sum is dominated by long gaps and to-nearest rounding cancels.
   The damage is entirely in the first-step lag, not in the accumulation.

   General form: **difference in the wide type, then narrow.** `data.py` already
   does this correctly — it computes `t_days[i] - t_days[cand]` in float64 and
   stores only the small-magnitude `log` in float32. A swept check confirmed
   `ntest.py`'s `t_last` was the sole place an absolute time was carried narrow.

1p. **Nothing shared between two arms may be recomputed from each arm's own
   data.** `prep_real_validation` already copies the frame rather than rebuilding
   it, for exactly this reason, and says so: *the frame is a property of the
   EXPERIMENT, not of the arm.* But the **ETAS fit region** was left out of the
   frame, so `etas_by_mc` derived it per-arm as the catalog's bounding box plus
   10 km — and derived it **after the mc cut**, making it a function of both the
   arm and mc.

   Measured null/informative area ratio: **1.121, 1.079, 1.058, 1.051** at mc
   2.5, 2.0, 1.5, 1.0. The surrogate rotates the footprint, so the null's box is
   always bigger; a uniform background fits μ ≈ N_bg/(area·T), so a bigger box
   means a smaller μ and a systematically under-forecast null. The asymmetry
   **shrinks** as mc drops, so it penalises the null hardest at high mc — it
   *suppresses* the measured mc-trend rather than creating one. Correcting it
   makes the curve steeper, not flatter.

   Sizing it as an *inter-arm* asymmetry understates it, and I made that mistake
   first. Against genuine inter-arm μ differences of 55–200% (the surrogate
   destroys the time–magnitude coupling, so the null's productivity exponent α
   collapses to 0.013–0.069 against 0.33–0.38 and the fit compensates with
   background), a 5–12% area error looks second-order. It is not, because the
   damage is not confined to the *difference* between arms.

   μ is fitted as a rate **density**: EM pins μ·area·T to the background event
   count, so μ ∝ 1/area_region. The rate field is then evaluated on the **fixed**
   scoring grid, so the background mass actually placed on the grid is off by
   `grid_area / region_area`. For the informative arm alone:

   | mc | fit region (km²) | grid/region |
   |---|---|---|
   | 2.5 | 10112.5 | **1.1748** |
   | 2.0 | 10603.6 | 1.1204 |
   | 1.5 | 11416.5 | 1.0406 |
   | 1.0 | 11758.3 | **1.0104** |

   The informative arm over-forecasts background by **17.5% at mc 2.5 and 1.0% at
   mc 1.0** — a 16% monotone swing across the very axis the claim is measured
   along, in the arm whose curve *is* the result. Smooth, monotone, mc-dependent:
   the same signature as 1c, 1d and 1i. The observed saturation (per-half-decade
   increments +0.217, +0.116, +0.022 nats) rests on this curve, so **the region
   fix is a precondition for interpreting it, not cleanup afterwards.**

   **Measured outcome (RV9 → RV10): this one moved the science.** Same data, same
   frame, same code except the region:

   | | RV9 (region per-arm, per-mc) | RV10 (region = scoring grid) |
   |---|---|---|
   | informative slope | +0.1501 **[−0.0425, +0.3329]** — spans zero | **+0.2162 [+0.0439, +0.3845]** — excludes zero |
   | shape curve | −5.5511, −5.4264, −5.3052, **−5.3414** | −5.6003, −5.4308, −5.3370, **−5.2712** |
   | last step (1.5→1.0) | **−0.0362** — turns down | **+0.0658** — still rising |
   | null per-step | −0.355, −0.269, −0.423 (scattered) | −0.365, −0.359, −0.348 (uniform) |

   The fix flipped the headline slope from non-significant to significant, and
   the apparent **downturn at the lowest mc was itself the artifact** — the
   region was over-forecasting background by 17.5% at mc 2.5 against 1.0% at mc
   1.0, so removing it cost 0.049 nats at the top of the range and gained 0.070
   at the bottom, tilting the whole curve. The null's per-step increments
   collapsing from scattered to near-uniform is independent evidence that what
   was removed was structure, not noise.

   The lesson to carry: *a defect found while comparing two arms need not live in
   the comparison.* I nearly filed this as a small inter-arm asymmetry because
   that is where it surfaced. Always ask what the same mechanism does to each arm
   on its own.

   And read this against **1o**, found the same day by the same reasoning and
   equally real as a code defect — yet measured at 0.12× the Monte-Carlo noise
   floor, i.e. irrelevant. Two defects, identical provenance, opposite
   consequence. *Nothing about how a bug is discovered predicts whether it
   matters; only measuring its effect on the reported quantity does.* Fix both,
   but spend the reruns where the measurement says to.

1q. **The informative-minus-null difference validates the instrument; it is NOT
   the quantity the claim is about.** Both arms move with mc, and they move for
   different reasons. From RV9 (per target event, uniform background):

   | mc | informative | Δ | null | Δ | difference |
   |---|---|---|---|---|---|
   | 2.5 | −9.2796 | — | −9.9242 | — | +0.6446 |
   | 2.0 | −9.0622 | **+0.2174** | −10.5147 | −0.5905 | +1.4525 |
   | 1.5 | −8.9458 | **+0.1164** | −10.7588 | −0.2440 | +1.8130 |
   | 1.0 | −8.9240 | **+0.0218** | — | — | — |

   The informative arm gains **+0.3556 nats/target across 1.5 decades**, in
   increments that roughly halve each half-decade — saturation, independently
   reproducing the finding in 1k. The null arm meanwhile *degrades* by −0.83
   nats: handed decoupled data, ETAS fits it more confidently and forecasts
   worse. So the difference grows mostly because **the null is falling, not
   because the informative arm is rising.**

   Quantified on the **shape** term, which is the primary metric (1c) — the
   table above is the total, and the two disagree in an important way. Per-step
   increments with paired block-bootstrap CIs
   (`scripts/saturation_diagnostic.py`, RV9 informative):

   | step | RV10 increment (current) | 95% CI | RV9 (pre-1p) |
   |---|---|---|---|
   | 2.5 → 2.0 | **+0.1695** | **[+0.0165, +0.3209]** significant | +0.1248 |
   | 2.0 → 1.5 | +0.0938 | [−0.0812, +0.2626] | +0.1211 |
   | 1.5 → 1.0 | +0.0658 | [−0.0402, +0.1765] | −0.0362 |

   Only the first half-decade is individually significant; the increments then
   decelerate by roughly half each step without reversing. RV9's apparent
   **downturn** in the last step was the 1p region artifact, not the physics —
   which is precisely why a per-step view is worth having: a linear slope would
   have absorbed the reversal silently. Over the full range the difference
   implies **+0.8375 nats/decade** against the informative
   arm's own **+0.1398** — a **6.0× overstatement**, worse than the 3.5× the
   totals suggested. (On RV10, which corrects 1p, the same comparison gives
   +0.9335 against +0.2194 — still a **4.3×** overstatement. The factor moves
   with the fixes; the structural point does not.) Reporting the difference as the headline would have inflated
   the claim six-fold while looking like the *more* rigorous choice, because it
   is null-corrected.

   Rule: use the difference to answer *"can the probe detect information at all?"*
   (RV's actual job) and the informative arm's own curve to answer *"how much does
   each decade add?"* (the claim). Never let a control statistic become the
   reported effect.

   These specific numbers are **provisional pending the 1p region fix**, which
   perturbs the informative curve by a monotone 16% in background mass. The
   structural point stands regardless of what the corrected numbers are.

   The fix is not "share the bounding box" but **use the frame's scoring grid as
   the region**. It is already shared, already mc-independent, and it makes the
   normalising area equal to the area actually scored, so no background mass
   falls outside the grid. Verified: the grid (11880 km², fixed) contains
   **100.0000%** of both arms' events at every mc.

   General form: **if two arms must agree on a quantity, compute it once and
   record it; never recompute it downstream.** The frame-copy comment had the
   principle right and the list incomplete — a shared-property list is a
   liability unless something enforces it.

1f. **The validation needs a POSITIVE control, not only a null.** A null that
   comes back flat proves nothing unless the informative arm is known to have
   something to detect, and that condition failed silently for three
   consecutive validation runs.

   Measured on the v3 informative arm, per scored point:

   | quantity | value |
   |---|---|
   | `corr(n_expected, n_observed)` across windows | **−0.09 to +0.15** |
   | `cv(n_expected)` across windows | 0.007–0.03 |
   | observed target counts per window | 3 to 17 (5.7×) |

   The forecast is flat to within 1–3% while the thing being forecast varies
   by nearly a factor of six, and the two are uncorrelated. The model was
   emitting the marginal rate — **no forecast skill at all**. A probe with no
   skill cannot respond to *any* change in its input, so its null arm is flat
   for a reason that has nothing to do with the estimator being sound. Every
   "PASS" that setup could produce would be vacuous.

   And it was not stopped early. The training log shows val nll at
   11.6536 / 11.6538 / 11.6542 for steps 2000 / 2500 / 3000, with the cosine
   schedule fully decayed to lr = 0 — it **converged**, on a 3,000-step budget
   against the production config's 20,000.

   **Do not read this as "the neural model cannot condition on history."** It
   was measured on the catalog that invariant 1g shows to be near-Poisson, where
   the optimal forecast really is close to constant — so a flat output was the
   *correct* answer and says nothing about the model's capacity. Whether
   FlowQuake conditions on history is **untested**; the first honest test is the
   G3 pilot on WHITE at a 1-day horizon, where the signal is known to exist.
   What the log does establish is narrower and still useful: a short schedule
   can converge cleanly to a constant forecaster, so "loss plateaued" is not by
   itself evidence of a trained model. Worth watching that `loss_s ≈ 11.49`
   against `loss_t ≈ 1.45` means the spatial term supplies ~8× the gradient
   magnitude despite equal loss weights — a candidate explanation *if* the
   temporal pathway turns out to be weak on real data.

   So the ground-truth validation must run a forecaster whose skill is
   guaranteed by construction. ETAS is the natural choice: Omori decay and
   magnitude-scaled productivity make it condition on history by definition,
   and `flowquake/etas_fit.py` already fits it per mc against a saved frame.
   The estimator then faces a real signal — lowering mc genuinely hands ETAS
   more triggering history — and the null genuinely removes it.

   Check `corr(n_expected, n_observed)` on every validation run before reading
   its verdict. If it is not significantly positive, the run says nothing,
   whatever the verdict line prints.

   **Gate on significance, not on an absolute value.** The attainable
   correlation is capped by Poisson counting noise: `Var(N) = Var(λ) + E[N]`, so
   even a perfect forecaster reaches only `sqrt(Var(λ)/Var(N))` — about 0.57 at
   a 1-day horizon on WHITE and 0.80 at 30 days. An absolute threshold like
   "corr > 0.20" therefore means different things at different horizons and
   would have rejected a working probe. Report the **share of the attainable
   ceiling** captured, and gate on `corr > 0` with `p < 0.05`.

   With the simulator bug (1i) fixed and the horizon at 1 day, the ETAS probe
   went from `corr = −0.084` (p = 0.54) to **`+0.135` (p = 3.2e-8)**, with the
   forecast's coefficient of variation rising from 0.19 to 0.88. It forecasts.

1g. **The validation catalog must be as clustered as a real one.** Swapping in
   ETAS did not fix the flat correlation — a properly fitted ETAS scored
   `corr = −0.073` on the same frame, against the neural model's `+0.154`.
   Neither is broken. The **catalog** was:

   | catalog | variance/mean, 30-day M-target counts | predictable |
   |---|---|---|
   | ComCat M≥4 | 38.92× | 97% |
   | QTM SaltonSea | 5.36× | 81% |
   | WHITE | 2.90× | 66% |
   | QTM SanJac | 2.03× | 51% |
   | **synthetic (branching 0.45, bg 6.0/day)** | **1.21×** | **17%** |

   Poisson is 1.0. The synthetic validation catalog was 2.4–32× less clustered
   than every real catalog — a background-dominated process in which the
   optimal forecast really is close to constant. Both probes were *right*; the
   experiment had nothing to detect. Three validation runs (RV1–RV3) were spent
   before this was checked.

   Generating parameters now default to `branching 0.75, bg_per_day 0.10`
   (measured 2.44×, between QTM SanJac and WHITE), and
   `scripts/synthetic_validation.py` **aborts** when the target series falls
   below `--min-overdispersion` (default 1.8) before spending any compute.

   The general lesson, and it is not confined to synthetic data: **check that
   the quantity you intend to forecast is forecastable at all before measuring
   how well anything forecasts it.** Variance/mean over the scoring windows
   costs one line and bounds every skill number that follows.

   **Why the generator did this, which is worth knowing in general.** The
   branching ratio of the *target sub-process* is not the branching ratio of
   the catalog. For productivity `10^(α(m−m₀))` and Gutenberg-Richter `b`, a
   parent at the target threshold produces on the order of

       n_sub  ≈  br · 10^(α·Δ) · 10^(−b·Δ)  =  br · 10^(−(b−α)·Δ)

   offspring that also clear the threshold, with `Δ = M_TARGET − m₀`. The
   generator used **α = 0.5, b = 1.0, Δ = 2.5**, giving `n_sub ≈ 0.025` — the
   M≥3 sub-process was Poisson *by construction*, even though the full catalog
   had a branching ratio of 0.9 and looked properly clustered. Raising `br`
   cannot fix that; it only pushes the full process toward supercriticality
   (at α = 0.5, b = 1.0 the mean branching ratio is `2·br`, so the `br = 0.75`
   trialled here was already supercritical at 1.5 and finite only because of
   the productivity clip).

   Clustering *at the threshold you score* requires **α ≈ b**, the
   self-similar case, which also makes the branching integral diverge unless
   magnitudes are truncated. Real catalogs satisfy both: α ≈ 0.8–1.0 against
   b ≈ 1.0, with a finite maximum magnitude. Tuning toward that got the
   synthetic to 1.6×, still short of the 2.0–5.4× the real regional catalogs
   show.

   **So the ground-truth validation moves to real data with a surrogate null.**
   Matching a simulator to reality is the harder way to obtain realistic
   clustering when the real catalogs are already on disk. Keep a real
   catalog's targets and its own small events as the informative arm — its
   clustering is real by definition, and the probe demonstrably forecasts it —
   and build the null by decoupling the small events from the target set
   (time-shift or an independent region) while preserving their marginals.
   The null's true slope is still zero by construction, which is the only
   ground truth the test actually needs. `branching_catalog` stays for
   controlled experiments where the generating parameters must be known, with
   the α relationship above documented at its definition.

1h. **Forecast at a horizon where the target series is actually predictable.**
   The design specified 30-day windows, inherited from the operational
   `P(M≥5 in 30 days)` readout. Measured on all four panels, the lag-1
   correlation of target counts between consecutive windows — the between-window
   signal ANY forecaster must capture — is:

   | panel | 1 d | 3 d | 7 d | 14 d | 30 d |
   |---|---|---|---|---|---|
   | WHITE | **+0.116\*** | +0.107\* | +0.189\* | +0.001 | −0.133 |
   | QTM SanJac | **+0.066\*** | +0.090 | −0.059 | −0.081 | −0.062 |
   | QTM SaltonSea | **+0.252\*** | +0.244\* | +0.074 | +0.002 | +0.196 |
   | ComCat M≥4 | **+0.097\*** | +0.021 | +0.131\* | +0.162\* | +0.004 |

   (\* = p < 0.05.) **1 day is significant in all four panels. 30 days in none.**

   The reason is physical, not statistical. Aftershock sequences in these
   regions start and finish *inside* a 30-day window, so their burstiness never
   carries across a window boundary. That shows up clearly on WHITE: the M≥3
   counts are 2.90× overdispersed at 30-day windows — plenty of excess variance
   — but the excess is **within-window** burstiness, and the between-window
   correlation is −0.133 (p = 0.33). A forecaster is asked to predict the next
   window from the previous one, and at 30 days there is nothing there to use.

   Confirmed on real forecasts: on WHITE at 30 days, the fitted ETAS scored
   `corr(n_expected, n_observed) = −0.084`, and so did the crudest possible
   baselines — persistence −0.150, trailing 2-window mean −0.145. When a
   one-line rule and a fitted point process agree at the same wrong sign, the
   horizon is wrong, not the model.

   **So the curve is scored at a 1-day horizon** (3-day as the robustness
   check). This also multiplies the window count by ~30 (WHITE: 56 → 1,675),
   so signal and statistical power both improve. Keep `P(M≥M_LARGE in 30 days)`
   as the operational readout it was always meant to be — it is a
   communication device, not the likelihood the curve is built on.

   Check the lag-1 correlation of the target counts before committing compute
   to any new region or horizon. It is a two-line calculation and it bounds
   what every model in the comparison can possibly achieve.

1e. **Score only where the catalog is complete, and choose that region
   causally.** The catalog build reported Mc = 1.30 flat for California using
   maximum curvature, and that number licensed the {3.0, 2.5, 2.0, 1.5} grid.
   It does not survive the b-value stability check:

   | mc | 1.0 | 1.5 | 2.0 | 2.5 | 2.6 | 3.0 |
   |---|---|---|---|---|---|---|
   | b | 0.740 | 0.837 | 0.914 | 0.968 | 0.977 | 0.978 |

   Above true completeness b is flat; a monotone downward drift as the threshold
   falls is missing small events. Statewide, b only stabilises at **mc ≈ 2.6** —
   which would put three of the four planned points below completeness.

   That statewide figure is itself misleading, and the resolution is the point
   of this invariant. Per 1° cell, b-stability gives **Mc 1.2–1.8** across the
   well-instrumented interior. A superposition of GR laws with different Mc is
   not a GR law, and the aggregate rolls over at the *worst* contributing cell,
   not the typical one — the offshore and border cells (Mendocino, the Baja
   margin, offshore Central Coast) hold ~430k of the 658k events and drag the
   aggregate down with them.

   So: build a `flowquake.completeness.CompletenessMask` and score inside it.

   - **Causal.** Estimate per-cell Mc from the **training era only**. Picking
     the testing region with knowledge of where the test-period earthquakes
     fell is look-ahead, and it is the flattering kind that does not look like
     cheating.
   - **Fixed.** ONE mask, built at the *lowest* mc on the grid, reused for every
     point and every control model. A mask that moves with mc puts an
     mc-dependent quantity back into the comparison.
   - **Verified.** The mask reports `mc_union`, the b-stability Mc of the pooled
     accepted cells. It must come out at or below the threshold. Individually
     complete cells whose *mixture* is not complete is the failure above,
     one level down.

   Measured for California with the mask from 1995–2011 and a 1.5 threshold:
   7 cells, 229,207 events, `mc_union = 1.3`, and **126 M≥4 targets** in the
   test window against 293 statewide. That is the honest cost — roughly 57% of
   the targets — and it buys the validity of every point on the x-axis. Do not
   trade it back for statistical power.

   Never set the grid from maximum curvature. On simulated catalogs with a known
   Ogata–Katsura detection curve, MAXC is optimistic in every case tested; on
   the real catalog it was optimistic by 1.3 magnitude units.

   **Use MBS, and leave a margin.** Two further measurements changed how this is
   estimated. First, the single-reference first-crossing rule (Cao & Gao 2002)
   fires on noise: across nested estimation eras for one 1° cell it returned
   1.3, 2.1, 1.1, 1.8 — impossible as physics, since restricting to a more
   recent era can only improve completeness. The Woessner & Wiemer MBS variant
   (compare `b(mc)` to the *mean* of b over `[mc, mc+0.5]`, require 3
   consecutive thresholds) gave 2.1, 2.0, 1.9 on the same data. MBS is the
   default. Second, **both** estimators are biased *optimistic* on ground truth
   (−0.08 and −0.11, sd ≈ 0.23–0.27), and optimistic is the dangerous
   direction. Cells are therefore accepted only when
   `Mc_cell + MC_SAFETY_MARGIN ≤ threshold`, margin 0.3.

   Finally, `mc_union` is necessary but **not sufficient**. Pooling can make the
   union look *better* than its worst member — abundant events from good cells
   swamp a sparse incomplete one — the same mixture artifact running backwards.
   The binding constraint is `mc_worst_cell`, because the score is evaluated
   cell by cell and one incomplete cell contributes a biased rate however good
   the average looks.

2. **Both arms, always.**
   - **matched-N** — same number of training events at every mc, achieved by
     extending the training window backwards as mc rises. Isolates *information
     per event*. Confounded by time span; that is the price.
   - **matched-window** — same calendar window at every mc, N varies. This is the
     operationally real question ("what do I gain by lowering my detection
     threshold?"). Confounded by sample size; that is the price.
   Arm B is the headline. Arm A is what proves Arm B is not just sample size.
   Reporting one without the other is the failure mode `runs/mw_robustness.json`
   already fell into.
3. **ETAS is re-inverted at every mc.** A curve against a fixed-mc ETAS is a
   curve about ETAS's mc sensitivity, not about information.
4. **≥3 tectonic regimes.** California alone is a JGR paper. The generality is
   the claim.
5. **Two metrics, both reported.** Per-target nats *and* probability gain on
   `P(M ≥ M_LARGE within 30 days)`, with M_LARGE chosen for POWER, not for
   headline value. In the California test window M≥6.0 occurs in only 3 of
   73 windows and cannot support a reliability diagram; M≥5.0 occurs in 13.
   Report M≥5.0 as the decision metric and M≥6.0 as an explicitly
   underpowered secondary. Nats alone will not carry a high-profile venue; the
   probability is what a general reader and an agency can both read.
6. **Every point is ≥3 seeds.** The current manuscript reports seed-0 numbers as
   3-seed (`WORKING.md` item 3). Do not repeat that here.

---

## The data plan — settled by measurement, 2026-08-02

Applying invariant 1e honestly kills the single-catalog design. **ComCat
California cannot support this claim**, and the numbers are not close:

| threshold | cells passing | events | M≥4 test targets |
|---|---|---|---|
| mc 1.5 | **0** | — | — |
| mc 2.0 | 5 | 97,832 | 6 |
| mc 2.5 | 16 | 377,131 | 143 |
| mc 3.0 | 20 | 482,600 | 163 |

No part of ComCat California is reliably complete at mc 1.5. The usable grid
with a workable target count is {3.0, 2.5} — **half a decade**, which cannot
carry a slope in nats *per decade*. The original plan rested on the
max-curvature Mc of 1.30, and that number was wrong.

Worse, masking concentrates the targets. Inside the mc≤1.5 mask, **87% of the
M≥4 targets are the 2019 Ridgecrest sequence** and one 30-day window holds 85%
of them, with only 12 of 74 windows non-empty. The effective sample size is
nearer one independent sequence than 126 events.

The dense catalogs already in `reference/Datasets/` resolve the completeness
problem and reintroduce the count problem, in the opposite direction:

| catalog | span | Mc | M_TARGET | decades | test targets | occupancy | top window |
|---|---|---|---|---|---|---|---|
| ComCat statewide | 25 y | 2.5 | 4.0 | 0.7 | 300 | 67/77 | 34% |
| **WHITE** (San Jacinto) | 13 y | **0.6** | 3.0 | **1.5** | 132 | 45/56 | **11%** |
| QTM SanJac | 10 y | 1.0 | 3.0 | 1.0 | 58 | 27/44 | 14% |
| QTM SaltonSea | 10 y | 1.4 | 3.0 | 0.8 | 71 | 28/44 | 24% |

(Range is quoted after applying the `MC_SAFETY_MARGIN` floor, so it is what the
grid can legally use rather than the raw Mc-to-M_TARGET distance. Counts are
for the splits actually written into `configs/panel_*.yaml`.)

**No single catalog gives both range and targets.** Range comes from dense
template-matched catalogs over small areas; targets come from large areas over
long times; and the two are in direct tension because a dense catalog is dense
precisely because it is small and recent.

**So the design is multi-region, and that is a strengthening, not a
concession.** Each region contributes a panel of ~1–2 decades; together they
span mc 0.6 → 3.5, roughly 3 decades. For a claim about a *limit*, four
independent regions agreeing on a slope is better evidence than one region's
longer curve, because a single curve cannot distinguish a real information
limit from one catalog's idiosyncrasy. This is what G4's "≥3 regimes" already
asked for; the measurement makes it mandatory rather than optional.

Consequences to honour:

- **M_TARGET differs by panel** (3.0 for the dense regions, 4.0 for ComCat).
  Legal — invariant 1 fixes the target set *within* a curve, not across panels
  — and it doubles as a check on whether the answer depends on target
  magnitude. It must be stated on the figure, not buried.
- **Pool with a random-effects model, never naively.** Regions differ in
  tectonics, network, and span; a fixed-effect pool would understate the
  uncertainty by pretending they are replicates of one experiment.
- **WHITE at M_TARGET 3.0 is the lead panel**: the widest range with the least
  concentrated targets (top window 14%).
- **Report the Ridgecrest concentration for the ComCat panel explicitly.** With
  34% of statewide targets in one window, its CI must come from a block
  bootstrap over sequences, not events.

---

## G1 — the ETAS control across four regions (COMPLETE, 2026-08-04)

All four panels scored at both backgrounds. Slope per decade of the SHAPE term
(1c), paired block bootstrap over forecast windows, DerSimonian-Laird pool
(`scripts/pool_etas_panels.py`):

| panel | targets | mc grid | uniform | smoothed |
|---|---|---|---|---|
| San Jacinto (WHITE) | 123 | 2.5–1.0 | **+0.2162** [+0.0455, +0.3832] | **+0.2602** [+0.1202, +0.4030] |
| San Jacinto (QTM) | 58 | 2.3–1.3 | −0.0945 [−0.3228, +0.1338] | +0.1138 [−0.1110, +0.3612] |
| Salton Sea (QTM) | 70 | 2.5–1.7 | **+0.5379** [+0.1200, +0.9858] | **+0.5868** [+0.2331, +1.0229] |
| California (ComCat) | 259 | 3.5–2.8 | −0.1249 [−0.3629, +0.2659] | −0.0423 [−0.2063, +0.2085] |
| **POOLED** | | | **+0.1072 [−0.1414, +0.3558]** | **+0.1933 [−0.0123, +0.3988]** |

**I² = 70.9% / 70.0%; Cochran's Q = 10.31 / 10.01 (df 3, p = 0.016 / 0.019).**
The regions are not measuring one common slope, and *that is the result*, not a
nuisance to average away.

Figure: `figures/g1_etas_regions_uniform.png` and `..._smoothed.png`
(`scripts/make_g1_figure.py`) — per-region curves on the left, forest plot with
the random-effects diamond on the right. It is deliberately laid out as a
meta-analysis rather than as one pooled number with an interval, because a lone
pooled number would read as "+0.19, not quite significant" when the truth is
"two regions gain substantially, two do not, and averaging them is the wrong
operation."

Per-step increments say the same thing in more detail — where the effect exists
it is spent almost entirely in the FIRST half-decade below the reference
completeness, then flattens:

| panel | step 1 | step 2 | step 3 |
|---|---|---|---|
| WHITE (uniform) | **+0.1695** [+0.0165, +0.3209] | +0.0938 | +0.0658 |
| Salton Sea (smoothed) | **+0.3678** [+0.1640, +0.5953] | +0.1016 | — |
| San Jacinto QTM (uniform) | +0.0096 | −0.1041 | — |
| ComCat (smoothed) | −0.0517 | +0.0323 | — |

**The honest reading.** For the best-fit physics baseline at a 1-day horizon:
a decade of magnitude buys real, significant forecast skill in **two of four
regions**, nothing measurable in the other two, and where it buys anything it is
nearly all spent in the first half-decade. Pooled across regions the slope is
**+0.11 to +0.19 nats/decade with an interval that includes zero**. There is no
universal "X nats per decade" constant to report.

This is the limits branch of the moonshot, and it is a real result rather than a
failure to find one — but it must be stated as *heterogeneous*, not as a null.
Two regions genuinely gain; the claim that fails is universality.

Cautions that stay attached to these numbers:
* WHITE and QTM San Jacinto cover the **same fault system** with different
  catalogs and differ in sign under a uniform background (−0.09 vs +0.22).
  Catalog construction is not a detail here. QTM's 58 targets are also the
  fewest of the four.
* ComCat spans only **0.7 decades** at M_TARGET 4.0 and was already reframed as
  the high-mc anchor after the completeness work (1e). Its flat result bounds
  the effect at high magnitude; it does not test the deep-catalog regime.
* The smoothed background is **not** uniformly better: it gains +0.35…+0.50
  nats/target on QTM San Jacinto but is neutral-to-worse on WHITE (−0.049,
  −0.137, +0.038, −0.038). Which background wins is itself region-dependent.
* Every fit but one sits at the branching-ratio barrier (1j); the exception is
  QTM San Jacinto at mc 2.3 uniform, n = 0.9713.
* **This is ETAS, not FlowQuake.** G1 bounds what the physics baseline extracts.
  Whether a learned model extracts more is G3, and G3 is still running.

## G3 — first scored point (ONE POINT, not a curve), 2026-08-04

`matched_window` at mc 2.0, scored against the ETAS control on the **same frame**
— same grid, same active mask, same 1,673 windows, same 123 targets, same fixed
GR tail. SHAPE term, nats per target event:

| forecaster | shape | level | total | corr | share of ceiling |
|---|---|---|---|---|---|
| ETAS uniform | −5.4308 | −3.6437 | −9.1463 | +0.155 | 23.7% |
| ETAS smoothed | −5.5455 | −3.6662 | −9.2836 | +0.144 | 21.9% |
| **FlowQuake** | **−4.1428** | **−3.3997** | **−7.6144** | **+0.169** | **25.8%** |

Invariant 1f **PASSES** (corr +0.169, p = 3.4e-12), so the score is interpretable.

Margin over the better ETAS background: **+1.288 nats/target on shape**, +1.53
total. But note *where* it comes from — the count correlation is barely different
(25.8% vs 23.7% of the attainable ceiling), so essentially the whole advantage is
in **where the model puts probability mass, not how many events it predicts.**

**Do not read a claim into this.** The moonshot question is whether the margin
*grows as mc drops*; one point cannot answer it, and a constant margin would be a
capability claim, not an information-in-small-events claim (invariant 3).
`scripts/compare_g3.py` computes the margin slope with both arms as soon as the
curve completes, and refuses to interpret any point failing 1f.

### The complete `matched_window` arm: an INTERIOR OPTIMUM at mc 2.0

**Correction first.** With only mc 2.0 / 1.5 / 1.0 scored, the neural slope read
**−0.9390 [−1.2978, −0.5948]** and I recorded "skill falls as the catalog
deepens." Adding the mc 2.5 anchor changed it to **−0.1676 [−0.4173, +0.0848] —
not significant.** The three-point read was an artifact of omitting the endpoint;
the curve is not monotone at all. *A slope through an incomplete grid is not a
weak version of the answer, it is a different answer.*

All four points, `matched_window`:

| mc | neural shape | ETAS shape | margin | corr | share of ceiling | 1f |
|---|---|---|---|---|---|---|
| 2.5 | −4.8928 | −5.6003 | +0.7075 | 0.126 | 19.2% | PASS |
| 2.0 | **−4.1428** | −5.4308 | **+1.2880** | 0.169 | 25.8% | PASS |
| 1.5 | −4.4141 | −5.3370 | +0.9229 | **0.484** | **73.8%** | PASS |
| 1.0 | −5.0818 | −5.2712 | +0.1894 | 0.208 | 31.8% | PASS |

* neural slope **−0.1676 [−0.4173, +0.0848]** — consistent with zero
* ETAS slope **+0.2162 [+0.0439, +0.3845]** — significant, and exactly the G1 value
* margin slope **−0.3839 [−0.6913, −0.0764]** — **ETAS closes the gap, significantly**

The neural curve is an **inverted U peaking at mc 2.0**, half a decade below the
reference completeness, and falling away at both ends. Read against the failure
modes below, the two ends fail for opposite reasons: mc 2.5 has only 965 training
events (underfit, best-val at step 200), mc 1.0 has 24,337 and goes sharp in the
wrong places (best-val at step 8,200, `n_eff_cells` 35.5 against 56.0).

**What survives as a claim.** FlowQuake beats the fitted ETAS baseline at *every*
completeness (+0.19 to +1.29 nats/target), so the capability claim holds. But the
*margin shrinks significantly* as the catalog deepens, because ETAS improves
monotonically while the neural model peaks and turns over. **On this panel,
deeper catalogs do not buy the learned model more skill — they buy the physics
baseline more.** That is close to the opposite of the moonshot's hoped-for
result, and it is the honest reading of the completed arm.

Robustness already checked: the ordering is **stable at every water level**
(1e-4 / 1e-3 / 1e-2 differ by ≤0.08 nats, invariant 1m), and it appears in the
**total** as well as the shape (−7.614 / −7.733 / −8.668). It is not a floor
artifact and not a shape-only artifact.

**A dissociation worth its own line.** Count skill and placement skill move in
opposite directions. `corr(n_expected, n_observed)` goes 0.169 → **0.484** →
0.208 — at mc 1.5 the model captures **74% of the attainable Poisson ceiling**,
three times anything ETAS reaches — while the shape score falls monotonically.
The model gets much better at *how many* and worse at *where*.

`n_eff_cells` on target-bearing windows says why: the mc 1.0 model is **more**
concentrated than the mc 2.0 one (35.5 vs 56.0) yet scores worse, i.e. it is
sharp in the wrong places. ETAS, by contrast, concentrates as mc drops (75.2 →
31.8) and improves — its concentration is earned. Meanwhile `matched_n` at mc 1.0
is diffuse (69.5), uninformative, and **fails 1f outright** (corr 0.016,
p = 0.53): 520 events cannot support learning the densest process. Two opposite
failure modes — overfit-sharp and undertrained-diffuse — both scoring badly.

**The interpretation this does NOT license.** The turnover at mc 1.0 is *not* by
itself the information limit. Best-validation steps run 200 / 400 / 1200 / 8200 as
mc drops, so the deep-catalog models train longest before overfitting — they are
using their data — but every point shares one capacity and one 12,000-step budget
while the catalog grows 25x (965 → 24,337 events). "This architecture at this
capacity cannot exploit deeper catalogs" and "deeper catalogs carry no more
forecastable information" are different claims, and only the second is the
moonshot.

### G2 capacity probe at mc 1.0 — capacity is NOT what binds

Same config, same data, same 12,000-step budget, **3.35x the parameters**
(15,432 → 51,752: `d_model` 64→128, `d_state` 24→48, `flow_hidden` 64→128,
`mix_hidden` 48→96). Scored on the same frame:

| | shape | total | corr | share | n_eff_cells |
|---|---|---|---|---|---|
| 1x, mc 1.0 | −5.0818 | −8.6684 | +0.208 | 31.8% | 35.5 |
| **2x, mc 1.0** | **−5.1237** | **−8.7296** | +0.216 | 32.9% | **29.9** |
| 1x, mc 2.0 (peak) | −4.1428 | −7.6144 | +0.169 | 25.8% | 56.0 |

The extra capacity closes **none** of the 0.94-nat gap to the mc 2.0 peak — it
very slightly widens it, and makes the field *sharper still* (n_eff 29.9 vs 35.5),
i.e. more confidently wrong. Validation agrees and was 60x cheaper to get: the
2x model's best val is worse (nll 5.5208 vs 5.4676) and arrives at step **1,400
instead of 8,200** — it overfits sooner, which is the opposite of a
capacity-starved model.

**So the deep-catalog turnover is not "the model is too small."** What remains on
the table is the architecture's inductive bias, the fixed hyperparameters (one
lr/dropout for every point), or a genuine property of the deeper catalog. Scope
of this probe, stated plainly: **one capacity step, one seed, one mc, one
hyperparameter setting.** It rules out the cheapest explanation, not every
explanation.

### Seed robustness — the inverted U is not seed noise

Every curve point is a single training seed, which was the one weakness that
could have explained the whole shape. Retrained at mc 1.0 with seed 1556:

Every interior point of the curve was retrained with seed 1556 and rescored,
**including the peak** — which was the single point the whole inverted-U claim
rested on:

| mc | seed 1555 | seed 1556 | spread | 2-seed mean |
|---|---|---|---|---|
| 2.5 | −4.8928 | — | — | −4.893 |
| **2.0 (peak)** | −4.1428 | **−4.0883** | **0.055** | **−4.116** |
| 1.5 | −4.4141 | −4.3051 | 0.109 | −4.360 |
| 1.0 | −5.0818 | −5.0410 | 0.041 | −5.061 |
| 1.0, 2x capacity | −5.1237 | — | — | — |

Seed spread is **0.041–0.109 nats**. The peak-to-mc-1.0 drop is **0.95 nats** and
the peak-to-mc-2.5 drop is **0.78 nats**, i.e. **7–23x the seed noise**. The
second seed makes the peak slightly *more* pronounced, not less. Validation
agrees independently (mc 2.0 nll 9.5593 vs 9.5140; mc 1.5 7.2206 vs 7.2166;
mc 1.0 5.4676 vs 5.4712).

Averaging seeds leaves the shape intact — **−4.893 / −4.116 / −4.360 / −5.061** —
so **the inverted U is a property of the model-and-data, not of the seed.**

### The mechanism: sharpness rises monotonically, accuracy does not

`n_eff_cells` = 1/Σp², the effective number of cells the forecast spreads over —
lower means sharper:

| mc | n_eff_cells | shape |
|---|---|---|
| 2.5 | **167.8** | −4.8928 |
| 2.0 | 56.0 | **−4.1428** |
| 1.5 | 32.4 | −4.4141 |
| 1.0 | 35.5 | −5.0818 |

The model gets **5x sharper** from mc 2.5 to mc 1.5 while accuracy peaks at mc
2.0 and then falls. Both ends are miscalibrated in opposite directions: at mc 2.5
it is diffuse and underfit (965 events, best-val at step 200); below mc 2.0 it
becomes **overconfident** — concentrating probability faster than its skill
justifies. ETAS over the same range does the opposite: it sharpens (75.2 → 31.8)
*and* improves, because its concentration is earned.

That is the cleanest statement of the result available so far: **deeper catalogs
make this model more confident without making it more correct.**

### GATE G3 COMPLETE — both arms, and the answer is a PEAK, not a slope

All 8 points scored (`runs/panel_white/curve.json`). Per-step increments in the
SHAPE term, paired block bootstrap over windows — the only honest summary of a
curve that bends (invariant 1k):

Final intervals use the **corrected circular moving-block bootstrap** (L = 30
windows, 6,000 replicates — see 1r; the pre-fix i.i.d. intervals were too narrow):

| step | FlowQuake `matched_window` | ETAS uniform |
|---|---|---|
| 2.5 → 2.0 | **+0.7500** [+0.4692, +1.0144] * | +0.1695 [+0.0044, +0.3284] marginal |
| 2.0 → 1.5 | −0.2713 [−0.6001, +0.0314] | +0.0938 [−0.0982, +0.2748] |
| 1.5 → 1.0 | **−0.6677** [−0.9721, −0.3580] * | +0.0658 [−0.0481, +0.1822] |
| **P(rise > 0 AND fall < 0)** | **1.0000** | **0.0100** |

That last row is the result in its cleanest form, and it needs no multiplicity
correction because it is a single intersection hypothesis: **the learned model
has an interior optimum with certainty; the physics baseline has none.** ETAS's
first step is marginal even here (lower bound +0.0044) and fails under gap-based
clustering, which is why it must not be called significant.

`matched_n` is deliberately absent from this table — it shares the mc 2.5 model
bit-for-bit with `matched_window` and its mc 1.0 endpoint fails invariant 1f. Its
per-point values are recorded below for completeness, not as confirmation.

> **NARROWED after adversarial review, 2026-08-04.** Five hostile reviewers
> attacked this result on independent lenses (metric, statistics, training
> confounds, pipeline artifacts, alternative interpretations) and an adjudicator
> weighed them. They reproduced every published increment to four decimals, and
> they broke four things. What follows is the surviving claim; the strikethrough
> version is kept because the overstatement is instructive. Verdict: **CLAIM
> NEEDS NARROWING**, not refuted.

**What stands: an interior optimum EXISTS.** FlowQuake's shape score is
significantly higher at mc 2.0 than at *both* mc 2.5 (+0.7500) and mc 1.0
(+0.9390). The intersection-union test P(rise > 0 AND fall < 0) is
**0.9998–1.0000 under every resampling scheme tried** — i.i.d., moving-block
L = 30 and L = 90, and cluster bootstraps at gaps > 30 d and > 60 d — and needs
no multiplicity correction because it is an intersection hypothesis.

**What had to be narrowed:**

1. ~~"peaks at mc 2.0"~~ → **the optimum lies in the mc 2.0–1.5 region, point
   estimate mc 2.0.** The 2.0→1.5 step is −0.2713 [−0.6228, +0.0591], not
   significant, and P(argmax = mc 2.0) is only **0.899** under the cluster
   bootstrap.
2. ~~"+0.75 to +0.83, significant in both arms"~~ → **+0.7500 [+0.5474, +0.9582],
   one arm, and an UPPER BOUND.** The two arms' mc 2.5 checkpoints are
   **bit-identical** (max absolute weight difference exactly 0.0 — at mc 2.5,
   `matched_n`'s N = 520 *is* the whole catalog, so the two arms share their left
   endpoint and are one contrast with two right endpoints, not two measurements).
   And the mc 2.5 anchor is the earliest checkpoint the run ever saved — step 200
   with `val_every` 200 and `warmup` 400, i.e. mid-warmup at half peak LR, with
   val NLL rising monotonically thereafter (11.6084 → 11.6276 → 12.1037 → 13.47).
   **Its true optimum lies inside (0, 200) and was never stored**, so the mc 2.5
   score is a lower bound and the step is an upper bound. This is the one
   confound nobody could bound numerically.
3. ~~"−0.6677 and −0.7463, both significant"~~ → **−0.6677 [−1.0262, −0.3040], one
   arm.** Delete the `matched_n` member: that endpoint fails the repo's own
   invariant 1f (corr 0.016, p = 0.53) *and* its run trained with **18 of 77
   logged batches carrying no data gradient** (sliding `train_start` to 2014-08-21
   while leaving `aux_start` at 2008-01-01 produced empty crops). Corrected for
   the simulation design effect the step is about **−0.585**. It survives
   leave-one-aftershock-sequence-out over all 15 sequences ([−0.9024, −0.4914],
   **zero sign flips**), the second seed (−0.7359), and moving-block L = 90
   (Holm p = 0.0007); it is marginal only under a cluster-jackknife t with K = 15.
4. ~~"the physics baseline gains +0.17, significant"~~ → **ETAS shows no robustly
   significant change at any step.** Its +0.1695 is [−0.0233, +0.3110] under the
   cluster bootstrap (p = 0.059) and fails Holm by every method, even though it
   survives moving-block. Correct wording: **"ETAS is flat where the learned
   model falls"** — not "ETAS is helped". What IS supportable is stronger than
   what was claimed: the **neural-minus-ETAS** increment is **+0.5805
   [+0.3536, +0.8877]** at 2.5→2.0 and **−0.7335 [−1.2484, −0.3655]** at 1.5→1.0,
   significant for both ETAS backgrounds, both neural arms, and within both the
   clustered and background target subgroups.
5. ~~"therefore there is an optimal catalog completeness"~~ → **not licensed.**
   One region, one target magnitude, one horizon, one architecture, two seeds.

**Invariant 2, restated honestly.** I claimed "both arms agree, so the turnover
is not a sample-size effect." The arms are *not* independent — they share the
mc 2.5 model exactly, and their mc 2.0 endpoints differ by 0.061 nats, inside the
0.055-nat seed spread. What survives is weaker and still worth having: the
turnover appears when N grows 520 → 10,601 **and** when N is held near 520, so it
is not driven by sample size *alone* — but the two arms are one experiment with a
shared anchor, not a replication.

**Why no slope is quoted as the headline.** A linear fit through all four points
gives −0.1676 [−0.4172, +0.0823], not significant — it averages a large
significant gain against a large significant loss and reports approximately
nothing. Restricting to mc 2.5/2.0/1.5 flips it to **+0.4787 [+0.0979, +0.8351]**,
significant and positive. Both numbers are arithmetically correct and both are
misleading. The 3-point restriction is defensible for `matched_n`, whose mc 1.0
point fails invariant 1f (corr 0.016, p = 0.53), but **not** for
`matched_window`, whose mc 1.0 point passes cleanly (corr 0.208, p = 7e-18) —
dropping it there would be selecting a valid measurement out of the record to
obtain a preferred sign. *This is exactly the failure mode 1k exists to prevent,
and it very nearly produced a headline with the wrong sign.*

**Internal noise control.** At mc 2.5 the two arms train on identical data and,
as the review established, produce **bit-identical weights**. So the −4.8928 vs
−4.9111 gap is **0.018 nats of pure SCORING Monte-Carlo noise** — I originally
called it "training/scoring", which was wrong: no training variation is present.
Every increment above is 15–46x it.

1s. **That 0.018 is ONE observation at ONE mc, and it is not a global tolerance.**
   It is a single |difference| of two runs at mc 2.5, where `n_sims` = 98,155 —
   the most heavily simulated point on the curve. Scoring noise falls with
   `n_sims`, so quoting 0.018 anywhere else understates it. Measured properly at
   the other end of the grid (mc 1.0, `n_sims` = 3,902), **6 replicates per
   device** on an RTX 5090 (`runs/panel_white/gpu_cpu_noise_mc1.json`):

   | device | n | mean `ll_shape_per_target_event` | sd |
   |---|---|---|---|
   | CUDA | 6 | −5.100912 | **0.035706** |
   | CPU  | 6 | −5.120513 | **0.038308** |

   So the per-run sd at mc 1.0 is **~0.036, twice the 0.018 figure**, and the
   difference of two independent runs has sd ≈ 0.051. Scoring is UNSEEDED (no
   `manual_seed` anywhere in the path), so every run is an independent draw and
   this spread is unavoidable, not a bug.

   *How this was found, because the lesson generalises:* I set a GPU-vs-CPU
   acceptance gate at "within 0.018 nats", ran it, and got 0.0353 — a FAIL. The
   defect was the threshold, not the GPU: a 0.018 gate would reject ~72% of
   honest CPU-vs-CPU comparisons. **A tolerance measured at one operating point
   is not a tolerance at another**, and importing it silently converts sampling
   noise into a false finding. The fix was to measure the noise where the gate
   actually runs rather than argue about the number.

1t. **CUDA and CPU score the same estimand — verified, not assumed.**
   Two-sample over the replicates above: **GPU − CPU = +0.0196 nats, SE 0.0214,
   t = 0.92, 95% CI [−0.0223, +0.0615]**. The interval contains zero and the two
   devices show near-identical spread (0.036 vs 0.038). Device choice is
   therefore free for future points *on noise grounds*.

   It is NOT free on provenance grounds: a curve must still be produced
   end-to-end on one device, because these runs bound the MEAN difference at
   ±0.06 nats, which is comparable to the smaller increments being claimed.
   Mixing devices within a curve would put a ±0.06 systematic where a 0.27-nat
   step is being tested. Between curves, compare only whole curves.

1r. **A bootstrap named "block" was resampling single days.**
   `pooling.block_bootstrap_slope` drew `rng.integers(0, n_win, n_win)` — an
   i.i.d. resample of individual forecast windows — while its docstring claimed
   to handle the fact that targets arrive in aftershock sequences. The windows
   are **consecutive 1-day forecasts** (every `start_days` difference is exactly
   1.0, verified) and a sequence stays correlated for weeks, so the resampled
   unit was ~30x smaller than the unit of dependence. **Every interval this
   project reported was too narrow**, on the G1 pooled slopes and the G3
   increments alike.

   Fixed: `_resample` now draws contiguous runs of `DEFAULT_BLOCK_WINDOWS = 30`,
   with `block_len=1` retained only to reproduce pre-fix numbers.
   `tests/test_block_bootstrap.py` pins the property that separates the two — a
   block draw retains lag-1 correlation an i.i.d. draw destroys (0.6+ vs <0.15 on
   an AR(1) fixture) — and asserts the fix can never *narrow* an interval on
   correlated data.

   Recomputed under the fix, the neural result **survives**: 2.5→2.0 becomes
   [+0.4513, +1.0082] at L=30 and [+0.4061, +1.0131] at L=90; 1.5→1.0 becomes
   [−0.9923, −0.3908] and [−1.0614, −0.3923]. ETAS's +0.1695 is the casualty —
   it survives moving-block but not gap-based clustering, hence the restatement
   in point 4 above.

   *Why no test caught it:* an i.i.d. bootstrap is perfectly well-behaved. It
   simply answers a different question, and nothing in a suite that checks
   "does the estimator run and return sane numbers" can tell the two apart. The
   general form: **a test that the code works is not a test that the code
   computes the intended estimand.**

### THE INTERIOR OPTIMUM DOES NOT SURVIVE — (mc, step) surface, 2026-08-05

The adjudicator's recommended experiment ran: all four `matched_window` points
retrained with **early stopping disabled**, every validation checkpoint kept
(240 each), and `ll_shape` scored on a 33-point grid dense below step 500.
132 scored points, `runs/surface_white/surface.json`.

**The +0.7500 rise was an artefact of the checkpoint-selection rule.**

| mc | best step | best `ll_shape` | plateau (steps >= 1000) | sd |
|---|---|---|---|---|
| 2.5 | 5500 | -3.8332 | -4.0427 | 0.1362 |
| 2.0 | 9500 | -3.8624 | -4.0301 | 0.0930 |
| 1.5 | 3500 | -4.2969 | -4.3676 | 0.0386 |
| 1.0 | 5500 | -5.0350 | -5.1194 | 0.0396 |

| step | published | argmax | plateau |
|---|---|---|---|
| **2.5 -> 2.0** | **+0.7500** | **-0.0292** | **+0.0126** |
| 2.0 -> 1.5 | -0.2713 | -0.4345 | -0.3375 |
| 1.5 -> 1.0 | -0.6677 | -0.7381 | -0.7518 |

The published mc 2.5 anchor scored -4.9272 at step 200. That model's own
optimum is -3.8332 at step 5500 -- the anchor sat **1.0940 nats below its own
best**, mid-warmup, and was being compared against fully-trained models at every
other mc. Removing the selection rule removes the entire rise.

**Consequences, stated plainly:**

1. **`P(rise > 0 AND fall < 0) = 1.0000` is DEAD.** There is no rise to
   intersect with the fall. The intersection-union test was measuring a
   training artefact at one end of the curve.
2. **The shape is not an inverted U.** It is FLAT over the first half-decade
   and then declines monotonically: **-1.077 nats total from mc 2.5 to 1.0**.
3. **The falls got STRONGER.** 2.0 -> 1.5 was reported as not significant
   (-0.2713, interval spanning zero) and is -0.3375 here; 1.5 -> 1.0 goes from
   -0.6677 to -0.7518.
4. **The surviving claim is cleaner than the one it replaces:** deepening the
   catalog never helps this model, and below mc 2.0 it actively and
   monotonically hurts. That is still a limits result, and it no longer depends
   on an "optimal completeness" that does not exist.

**What is NOT yet established, and must not be quoted as if it were:**

* The per-step spread is treated above as if the 23 plateau checkpoints were
  independent. **They are not** -- they come from one training run and are
  serially correlated, so any standard error computed that way is too narrow.
  This is invariant 1r's error in a new costume. Proper intervals need the
  block bootstrap over WINDOWS (`pooling.block_bootstrap_slope`) applied to the
  per-window arrays, which are committed alongside the aggregates.
* **One seed, one fresh training run.** Not bit-comparable to the published
  curve (the code changed); internally consistent across all four points, which
  is the comparison that matters, but seed variation is unmeasured here and the
  mc 2.5 plateau sd is 0.1362.
* `argmax` over 33 noisy checkpoints is **itself a selection rule** and biases
  the maximum upward. That is why the plateau mean is quoted as the headline
  and both are shown.

**HONEST INTERVALS (two-axis circular block bootstrap, 6000 reps).** The
standard errors above treat 23 plateau checkpoints as independent; they are
serially correlated, so those were too narrow -- by **10x**, as it turned out.
`scripts/surface_intervals.py` resamples BOTH axes: windows in contiguous
30-day blocks (targets arrive in sequences) and checkpoints in blocks of 3
(adjacent checkpoints share almost all their weights), paired across every mc so
each increment stays a within-window, within-stage contrast.

| step | increment | 95% CI | verdict |
|---|---|---|---|
| 2.5 -> 2.0 | +0.0126 | [-0.2814, +0.3707] | nothing |
| 2.0 -> 1.5 | -0.3375 | [-0.6942, -0.0302] | marginal |
| 1.5 -> 1.0 | -0.7518 | [-1.0557, -0.4402] | solid |
| **TOTAL 2.5 -> 1.0** | **-1.0767** | **[-1.4544, -0.6613]** | **P(decline) = 0.9997** |

**P(rise > 0 AND fall < 0) = 0.5337**, against the published 1.0000. But note
also **P(monotone non-increasing) = 0.4580** -- also a coin flip. With 123 target
events this dataset CANNOT distinguish "flat then falling" from "slight rise
then falling". Neither shape is established, and the claim must not be restated
as monotonicity just because the inverted U died.

**What IS established is the endpoint comparison**, and it should carry the
paper: going from mc 2.5 to mc 1.0 costs **1.08 nats [0.66, 1.45]** of forecast
skill per target event.

1v. **The binding constraint is target events, not compute.** Variance
   decomposition of the total decline: window/sequence sampling sd = 0.1956,
   checkpoint sd = 0.0327 -- windows are **95% of the variance**. More training
   seeds, more checkpoints and more GPU time all attack the 2.7% component. Only
   more target events narrow this interval, and those come from longer catalogs
   or more regions, not from a bigger machine. Worth checking BEFORE buying
   compute to sharpen a result: a seed sweep here would have cost ~$20 and moved
   the interval by almost nothing.

1u. **An unstored optimum is not a small error, and "it is only the endpoint"
   is not a defence.** The confound was known, described precisely, and
   correctly identified as unbounded -- and it still turned out to carry the
   entire headline effect. The lesson generalises past this experiment: when a
   selection rule depends on the axis under study, the honest move is to
   ELIMINATE the rule and remeasure, not to bound its influence by argument.
   Bounding it by argument is what produced +0.7500.

### `matched_n` (invariant 2) — per-point detail

`matched_n` fixes the training event count at N = 520 (the count at mc 2.5) and
varies only the magnitude threshold, so it separates "more events" from "more
information per event". Scored points:

| mc | `matched_window` | `matched_n` | 1f on `matched_n` |
|---|---|---|---|
| 2.0 | −4.1428 | **−4.0817** | PASS (corr 0.142) |
| 1.5 | −4.4141 | −4.4710 | PASS (corr 0.074, only 11% of ceiling) |
| 1.0 | −5.0818 | −5.2173 | **FAIL** (corr 0.016, p = 0.53) |

Two observations that are already safe to make:

* At mc 2.0 the **matched-N model scores slightly better on 520 events than the
  full-window model does on 2,816** (−4.0817 vs −4.1428). Five times the training
  data bought nothing — this model saturates in sample size well below 2,816
  events, which is a data-efficiency result in its own right.
* The decline with depth appears **with N held fixed**, so it is not purely a
  sample-size effect. `matched_window` shrinks nothing and declines;
  `matched_n` shrinks the time span instead and declines too.

**The slope is deliberately not quoted here.** This arm is missing its mc 2.5
anchor, and that is precisely the configuration in which `matched_window` read
−0.9390 (significant) and then became −0.1676 (not significant) once the anchor
landed. The same correction may well apply. Also, the mc 1.0 point **fails
invariant 1f**, so no slope through it means anything regardless —
`scripts/compare_g3.py` says so and refuses to interpret it.

Figure: `figures/g3_panel_white_matched_window_uniform.png`
(`scripts/make_g3_figure.py`) — three panels, one per thing that had to be
believed. Left: both curves on the same frame with the margin shaded and the
seed replicates overplotted as open markers. Middle: the margin alone, peaking at
mc 2.0 and collapsing to +0.19 by mc 1.0, annotated with its bootstrapped slope.
Right: `n_eff_cells` on a log axis — FlowQuake sharpens 5x and converges onto
ETAS's concentration, while its accuracy does not follow.

## What the neural arm actually costs — measured, 2026-08-03

Matched resolution (1d) fixes `T = 20,000` simulated events per target-bearing
window **regardless of mc**. That is the whole point: resolution must not vary
along the axis being measured. But it also means the cost per curve point is
roughly **constant**, not proportional to `n_sims`, because `n_sims` falls
exactly as fast as events-per-simulation rises:

| mc | n_sims | events/sim | T per window |
|---|---|---|---|
| 2.5 | 98,155 | low | 20,000 |
| 2.0 | 33,681 | | 20,000 |
| 1.5 | 11,423 | | 20,000 |
| 1.0 | 3,902 | high | 20,000 |

I first concluded from constant T that cost is constant per point. **That was
wrong, and the measurement corrected it.** Cost tracks `n_sims`, not `T`, because
every lane pays a full sampling-loop pass whether or not it emits an event — the
loop is over lanes and steps, and events are only what falls out of it:

| point | n_sims | wall clock |
|---|---|---|
| `matched_window` mc 2.0 | 33,681 | ~8 h |
| `matched_n` mc 1.0 | 3,902 | ~1 h |

Ratio 8.6 against a simulation ratio of 8.6 — proportional, not constant. So the
mc 2.5 points (98,155 sims) are the long pole at **~23 h each**, and the cheap
end of the curve is nearly free. Sampling is through a 16-step flow ODE,
single-threaded (`at::TensorIteratorBase::serial_for_each`, confirmed with
`sample`).

* **~1 h to ~23 h per curve point, scaling with `n_sims`**
* **~26–35 h per panel** (8 points, concurrency 2), dominated entirely by the
  two mc 2.5 points
* **~5 days for all four panels**

The practical consequence: the expensive end of the curve is the **most
complete** catalog, not the deepest one. That is counter-intuitive and worth
remembering when planning — matched resolution needs ~25x more simulations at
mc 2.5 than at mc 1.0 to reach the same T.

This is a design constraint, not a bug, and it does not justify lowering `T`:
the measured residual bias is 0.023 nats at T = 20,000 against 0.44 at T = 2,000,
and the informative slope being measured is ~0.2. Cutting T to buy speed would
reintroduce the largest artifact in the pipeline (1d) to chase the smallest one.

The honest options, in preference order: (a) run the lead panel fully and let the
ETAS control carry the other three, since G1 is ~200x cheaper per panel and is
already done; (b) raise concurrency — the machine has 18 cores and only 4 are
busy, but each scorer holds ~4 GB and this laptop is shared, so 3 is the ceiling
worth taking; (c) accept a longer wall clock. **Do not** narrow to one arm:
invariant 2 requires both `matched_window` and `matched_n`, and a single-arm
result cannot separate information content from sample size.

## Gates — do not pass one without the result in hand

| gate | question | artifact | if it fails |
|---|---|---|---|
| **G0** | Is the encoder framing honest? | `h_bottleneck` described accurately in README/MANUSCRIPT | fix before anything else; it contaminates every downstream claim |
| **G1** | Does a classical ETAS with a free smoothed background reach +0.060? | `runs/fletas_free_bg/*.json` | §4.4's neural framing collapses. Retitle honestly, and the moonshot becomes *more* important, not less — it no longer depends on a neural win |
| **G2** | Does the memorization result survive the three missing controls? | `runs/ablation_h_controls/*.json` | rewrite §4.3 as "absolute-coordinate conditioning causes memorization"; the encoder may be salvageable, which changes the architecture available to the moonshot |
| **G3** | Is there a measurable slope at all? 3-point pilot, California, mc {2.5, 1.5, 0.5} | `runs/scaling_pilot/*.json` | if the slope is flat and tight at the pilot, the moonshot is dead — say so, publish the null in the JGR paper, stop |
| **G4** | Does the slope hold in ≥2 more regimes? | `runs/scaling_curve/*.json` | descope to a single-region GRL/JGR result. Do not pretend one region is a general claim |
| **G5** | Does it show up in `P(M≥6 in 30d)`? | `runs/large_event_skill/*.json` | keep the nats result, drop the operational framing, aim Nature Geoscience not Science |

**G3 is the honest kill switch.** It is ~6 runs and a day. Run it before
committing to anything downstream.

---

## Scope tripwires — if you catch yourself doing these, scope has collapsed

- Reporting **any** new number as a win over ETAS without the matched-arm control.
  → The comparison is the control, not the finding.
- Adding a **model improvement** because it would raise the headline. Architecture
  work bought +0.004 nats; data elasticity bought +0.153. Architecture is not
  where this paper lives.
- Extending the **region count** before G3 passes. Breadth before slope is how
  this becomes a 40-month programme with a JGR outcome.
- Letting `MANUSCRIPT.md` absorb this work. **The moonshot is a second paper.**
  The current manuscript ships to JGR/Seismica with honest framing; it is not
  upgraded into a Science submission by accretion.
- Scoring on an mc-dependent target set "just to see." You will see a huge
  spurious effect and it will be hard to unsee.
- Starting the Coulomb / neural-operator / foundation-model / prospective-CSEP
  work. All four scored DESCOPE. They are not this paper. See "Explicitly out".
- Spending GPU time before the code runs end-to-end on CPU at `--steps 200`.

---

## Explicitly out of scope

Named so they stop resurfacing as good ideas:

- **Deterministic earthquake prediction.** Not a goal anyone credible holds.
- **Mainshock precursors from catalogs.** The evidence says the signal, if it
  exists, is in deformation data (GNSS), not catalog statistics. That is a
  different instrument and a different project (tier 3).
- **Neural-operator Coulomb-stress kernel** (`SEED.md`). Needs depth, focal
  mechanisms and finite-fault models the pipeline does not ingest; walks into
  the DeVries/Mignan graveyard. Revisit only after depth is a model input.
- **Global foundation model / zero-shot everywhere.** P(total-likelihood win
  with no ETAS inversion) ≈ 15%. Harmonisation, not compute, is the blocker.
- **Registered prospective CSEP.** Upgrades the engineering claim, not the
  scientific one. 42 months, external custody. Worth doing eventually; it is
  not the moonshot and must not become the excuse for not doing the readouts.
- **Cross-domain "general law of point processes."** The law's functional form
  is 1974 statistics.

---

## Venue ladder, tied to evidence

| evidence in hand | honest venue |
|---|---|
| current result set, framing fixed, G1 answered | JGR: Solid Earth / Seismica / GJI; TMLR for §4.3 |
| + physics readouts (θ vs rupture strike, β(t), P(background)) | GRL / JGR, better if θ converges |
| + scaling curve, 1 region | GRL / JGR |
| + scaling curve, ≥3 regimes, both arms, ETAS refit throughout | **Nature Geoscience / PNAS** |
| + `P(M≥6 in 30d)` skill moves with it | **Science / Nature** (~15%) |

Aiming at the top costs nothing extra: the experiment is identical either way.

---

## Prior art to clear before committing

Read these first; the framing is live and we may have competition.

- arXiv:2607.26918 (Jul 2026) — *How to quantify earthquake predictability?
  Advances in earthquake forecasting and predictability limits*
- Seismica — *Large earthquakes are more predictable than smaller ones*
- Ross et al. 2019 (Science) — QTM, 1.8M events, the low-mc catalog this depends on
- Helmstetter et al. 2007 — variable-bandwidth smoothed seismicity (the G1 baseline)

---

## Run ledger

Status is updated as runs land. Every row names the artifact that settles it.

| id | run | gate | artifact | status |
|---|---|---|---|---|
| R0 | encoder framing audit | G0 | README + MANUSCRIPT §1/§3 | **DONE** |
| RC | low-mc California catalog | pre-G3 | `reference/Datasets/ComCat_lowmc/` | **DONE** — 658,351 events, projection verified 5.1e-12 km. ⚠️ its Mc 1.30 is max-curvature and **superseded**; see RM |
| RM | completeness re-measurement | pre-G3 | `flowquake/completeness.py` | **DONE** — statewide b-stability Mc = **2.6**, not 1.30. Per-cell 1.2–1.8. Causal mask (1995–2011, Mc≤1.5): 7 cells, `mc_union` 1.3, 126 M≥4 targets (was 293). Invariant 1e |
| RV1 | ground-truth validation, unmatched precision | pre-G3 | `verdict_UNMATCHED_PRECISION.json` | **DONE** — PASS, but null slope +0.32…+0.37 exposed invariant 1d |
| RV2 | ground-truth validation, matched precision | pre-G3 | `verdict_v2_UNIFORM_NULL_BIASED_B.json` | **DONE — FAIL, and correctly so.** Two independent defects: `aki_utsu_b` assumed 0.1 binning → b 0.869 vs true 0.993, amplified to a −0.19/decade level slope; and the uniform null had 2× the footprint → −0.47/decade shape slope. Both fixed |
| RV3 | ground-truth validation, 3 arms, shape metric | pre-G3 | `curve_informative_v3_NEURAL_PROBE_NO_SKILL.json` | **STOPPED — the probe had no skill.** `corr(n_expected, n_observed)` = −0.09…+0.15 and `cv(n_expected)` = 0.007–0.03 while observed counts varied 5.7×. Confirmed the b fix landed (level slope −0.19 → **−0.024**), but could certify nothing. Invariant 1f |
| RV4 | ETAS probe, single point, same frame | pre-G3 | scratchpad | **DONE — exonerates both probes.** A fitted ETAS scored `corr = −0.073` on the same frame. Not the model, not the estimator: the **catalog** was near-Poisson (variance/mean 1.21 vs 2.0–5.4 real). Invariant 1g |
| RV5 | ground-truth validation on a REAL catalog with a surrogate null | pre-G3 | `runs/real_validation/` | staged and verified on WHITE: `time_shift` drops small-events-per-target from **480.7 to 361.1** against a decoupled baseline of 368.5; count (58,328), footprint (22.3 km) and all 308 targets preserved exactly; both arms share a bit-identical frame. First pass ran at the 30-day horizon and was **uninterpretable** — see RH |
| RT | Monte-Carlo resolution bias | pre-G3 | invariant 1d (corrected) | **DONE — the largest bias in the pipeline.** The shape score depends on `T`, the total simulated events behind the rate field, through a relative variance of `1/(T p_c)`. `T` scales with the catalog rate above mc, so it ran 93→1319 across one grid and produced **+3.02 nats/decade on the informative arm and +2.62 on a null whose true slope is zero**. A controlled experiment (same truth, same observations, only `T` varied) reproduced +4.54 against a measured +4.53. The earlier fix equalised *absolute* variance, the wrong quantity |
| RE | ETAS simulator double-thinning | pre-G3 | invariant 1i | **DONE — voids every prior ETAS number.** The first generation from observed history was drawn twice, giving `Σ wᵢ²` instead of `Σ wᵢ`: suppressing small parents and inflating large ones, i.e. distorting exactly the magnitude dependence this project measures. Found by computing the same field in closed form and disagreeing by 2.7× in a way that did not shrink with `n_sims`. Fixed; simulator now sits at 1.21× the first-generation analytic value as it should |
| **RV7** | **ETAS-probe validation on WHITE, 1-day horizon, era-bounded surrogate null** | **pre-G3** | `runs/real_validation_h1/etas/verdict.json` | **PASS — the first validated result in this project.** Probe check passed (corr +0.154, p = 2.1e-10, 24% of the 0.656 Poisson ceiling). informative slope +0.042 [−0.153, +0.235]; surrogate null −0.839 [−1.091, −0.604]; **difference +0.880 [+0.572, +1.189]**, excluding zero. The null's skill decays monotonically (12.4% → 3.8% of ceiling) as decoupled events accumulate, and its fitted α collapses to ~0.02 — both signatures of a control that is working. **SUPERSEDED: the null-arm numbers are void** — the frame carried no time origin, so the surrogate arm was forecast 1.7 h out of step (1n), and the ETAS fit region was derived per-arm (1p). The `+0.880` difference must not be quoted |
| RV8 | RV7 rerun with corrected sub-grid quadrature | pre-G3 | — | **KILLED mid-run.** The time-origin bug (1n) was found while it was in flight, which voided its null arm before it finished. Superseded by RV9 |
| RV9 | RV7 rerun: sub-grid quadrature **and** shared time origin | pre-G3 | `runs/real_validation_h1/etas_subgrid/` | **7/8 fits complete** (null mc=1.0 still scoring), informative arm complete. Informative own curve −9.2796 / −9.0622 / −8.9458 / −8.9240 → **+0.3556 nats/target across mc 2.5→1.0**, in increments **+0.2174, +0.1164, +0.0218** that roughly halve each half-decade — saturation, independently reproducing 1k. Null degrades −0.83 as decoupled events accumulate, so the *difference* (+0.645, +1.453, +1.813) grows mostly because the null falls: quoting it as the headline overstates by **3.5×** (1q). Branching ratio pinned at **0.990 in all 7 completed fits**, both arms, every mc (1j). **Still carries the 1p region artifact** — numbers provisional |
| RF | float32 absolute time in the neural simulator | pre-G3 | invariant 1o, `tests/test_ntest_time_precision.py` | **DONE.** `t_last` was float32 while `t_buf`/`start_t`/`end_t`/`t_ref` were all float64. At test-era day numbers that is a **21–42 s quantum**, 2400× the `TAU_FLOOR_DAYS` floor the code clamps to: sub-quantum `tau` advanced time by exactly zero, and the first-step encoder lag — which must be identically zero — became ±21 s of coin-flip noise, **2.8σ** in a feature normalised by `log_tau_std`. **Noise, not bias**, so no slope-based check could have caught it. The accumulated-drift form of the same worry was measured and **refuted** (3e-6, non-monotone). **Measured impact, paired same-seed over 30 windows: −1.005% on total expected count and 0.105 mean spatial TV, against a Monte-Carlo noise floor of −0.713% and 0.853 — i.e. 1.4x the noise on count, 0.12x on shape.** Below the scatter the estimator already tolerates. Fixed because it was wrong and free, NOT because it moved a result; CLAIMS.md family-3 artifacts shift within MC noise |
| **RV10** | **RV9 rerun with the region taken from the scoring grid** | **pre-G3** | `runs/real_validation_h1/etas_gridregion/verdict.json` | **PASS, and it supersedes RV9.** Probe check +0.156 (p=1.4e-10, 24% of the 0.656 ceiling). Informative shape curve −5.6003 / −5.4308 / −5.3370 / −5.2712 — **monotone, no reversal**. Per-step increments **+0.1695 [+0.0165, +0.3209]** (significant), +0.0938 [−0.0812, +0.2626], +0.0658 [−0.0402, +0.1765]: decelerating by roughly half each half-decade. Linear slope **+0.2162 [+0.0439, +0.3845]**, now excluding zero where RV9's spanned it. Null slope −0.7145, difference +0.9307 [+0.6984, +1.1631]. Quoting the difference would overstate the claim **4.3×** (1q). Branching still pinned at 0.990 throughout (1j) |
| RR | ETAS fit region derived per-arm and per-mc | pre-G3 | invariant 1p | **DONE — and it moved the result** (see RV10). μ ∝ 1/area_region while the field is evaluated on the fixed grid, so background mass was off by `grid_area/region_area`: **1.175 at mc 2.5 falling to 1.010 at mc 1.0**, a 16% monotone swing along the claim's own axis. Correcting it flipped the informative slope from non-significant to significant and removed a spurious downturn at the lowest mc. Original scoping below, kept because under-scoping it was the mistake |
| ~~RR (initial scoping)~~ | ~~second-order inter-arm asymmetry~~ | — | — | **PENDING.** μ ∝ 1/area_region while the rate field is evaluated on the fixed grid, so background mass on the grid is off by `grid_area/region_area` — **1.175 at mc 2.5 falling to 1.010 at mc 1.0** for the informative arm, a 16% monotone swing along the claim's own axis. Fix: take the region from the frame's scoring grid (11880 km², mc-independent, verified to contain **100.0000%** of both arms' events). Requires RV10 |
| RA | adversarial bias audit, 5 dimensions | pre-G3 | workflow `wf_fb00dc61-c11` | **PARTIAL** — 20 candidates, 12 refuted, 1 confirmed, **7 never adjudicated** (the run hit a usage limit and the refuters plus the synthesis step died). The confirmed one was the lane-budget recursion, already fixed mid-audit. Two further gaps it surfaced are now closed: curve.json silently accepted `None` rows for failed points, and nothing checked that the neural and ETAS arms cover the same mc grid. **The 7 unadjudicated candidates are outstanding work, not a clean bill of health** |
| RH | horizon diagnostic across all four panels | pre-G3 | invariant 1h | **DONE, and it changes the design.** Lag-1 correlation of target counts is significant at **1 day in all four panels** and at **30 days in none**. Sequences begin and end inside a 30-day window, so burstiness never crosses a boundary: WHITE is 2.90× overdispersed at 30 days yet its between-window correlation is −0.133 (p = 0.33). Confirmed on real forecasts — fitted ETAS −0.084, persistence −0.150, trailing mean −0.145, all the same wrong sign. **Curve moves to a 1-day horizon**; RV5 and G3 rerun there |
| RD | data feasibility across all catalogs on hand | pre-G3 | "The data plan" above | **DONE** — ComCat cannot carry the claim (0 cells complete at mc 1.5; usable grid {3.0, 2.5}). Design pivoted to 4 panels spanning mc 0.6–3.5 |
| R3a | scaling pilot, **WHITE**, mc {2.5, 2.0, 1.5, 1.0} | G3 | `runs/panel_white/` | blocked on RV3 — lead panel |
| R3b | scaling pilot, QTM SanJac, mc {2.3, 1.8, 1.3} | G3 | `runs/panel_qtm_sanjac/` | blocked on RV3 |
| R3c | scaling pilot, QTM SaltonSea, mc {2.5, 2.1, 1.7} | G3 | `runs/panel_qtm_saltonsea/` | blocked on RV3 |
| R3d | scaling pilot, ComCat statewide, mc {3.5, 3.0, 2.8} | G3 | `runs/panel_comcat/` | blocked on RV3 — high-mc anchor, many targets |
| R8 | cross-region random-effects pool | G4 | `flowquake/pooling.py` | tooling **DONE**, blocked on R3a–d |
| R1 | ETAS free smoothed background per mc | G1 | `runs/etas_by_mc/` | ready |
| R2 | h-ablation × 3 controls | G2 | `runs/ablation_h_controls/` | ready |
| R4 | scaling curve, ≥3 regimes × mc × 2 arms × 3 seeds | G4 | `runs/scaling_curve/` | blocked on G3 |
| R5 | ETAS re-inversion at every mc | G4 | `runs/etas_by_mc/` | ready |
| R6 | P(M≥5 in 30d) skill vs mc | G5 | in `target_process.json` per point | ready |
| R7 | physics readouts (θ, β, P(bg)) | — | `runs/readouts/` | ready |

**Every gate now has working, smoke-tested tooling.** What remains is compute and
the RV3 verdict — not engineering.

**What RV2 bought.** It failed, and that is the return on building it. Both
defects it caught were bias generators, not crashes: each produced a smooth,
monotone, entirely plausible slope of the kind that would have gone straight
into the figure. Neither is visible without a null — the informative arm alone
looked fine in both cases. The uniform-null defect in particular would have
made the pipeline look *robust* (informative flat, null strongly negative =
"our model extracts signal where there is none to extract"), which is the most
seductive possible failure mode. Cost of the null: one extra arm. Cost of
skipping it: a retraction.

### Tooling built (2026-08-02)

| module | role |
|---|---|
| `flowquake/target_process.py` | the mc-invariant metric (invariants 1b, 1c); N-test/S-test split, magnitude-quantum inference |
| `flowquake/completeness.py` | b-stability Mc (MBS), causal per-cell mask, safety margin (invariant 1e) |
| `flowquake/pooling.py` | paired block-bootstrap slope CI + DerSimonian-Laird cross-region pool |
| `flowquake/surrogate.py` | surrogate nulls on REAL catalogs: circular time shift / rotation (invariant 1g) |
| `flowquake/proc.py` | child-process reaping, so a stopped run actually stops |
| `scripts/validate_with_etas.py` | ground-truth validation with a probe that provably forecasts (invariant 1f) |
| `scripts/build_completeness_mask.py` | builds and verifies the mask, writes the masked catalog |
| `configs/panel_white.yaml` | lead panel (Mc 0.6, 1.5 decades, 132 targets) |
| `configs/panel_qtm_saltonsea.yaml` | swarm-dominated contrast panel (Mc 1.4) |
| `configs/panel_qtm_sanjac.yaml` | same fault as WHITE, different catalog (Mc 1.0) |
| `flowquake/etas_fit.py` | self-contained ETAS: EM, smoothed background, simulator. Also removes the unpinned-fork provenance hole (`WORKING.md` item 8) |
| `scripts/scaling_curve.py` | the sweep; invariants enforced in code |
| `scripts/etas_by_mc.py` | invariant 3 + gate G1 |
| `scripts/ablation_h_controls.py` | gate G2, with the gap **decomposed** into temporal/spatial |
| `scripts/build_comcat_lowmc.py` | low-mc catalog + per-era completeness |
| `scripts/synthetic_validation.py` | the ground-truth precondition |
| `scripts/readouts.py` | θ vs rupture strike, β(t), P(background) |
| `scripts/bootstrap_remote.sh` | fresh-box provisioning, gate-ordered stages, hard cost guard |
| `configs/moonshot_lowmc.yaml` | the sweep's fixed frame (splits/architecture); only `mc` varies |
| `scripts/make_moonshot_figure.py` | ONE panel in detail: both arms, ETAS controls, sign-aware interpretation |
| `scripts/make_pooled_figure.py` | **the killer figure**: all panels + forest plot + random-effects pooled slope |

**Frame convention, verified.** EarthquakeNPP stores `(x, y) = (NORTHING,
EASTING)`, not the usual easting-northing. Solved by fitting the transform that
reproduces the shipped `ComCat_catalog.csv`: with the swap and centre = catalog
mean lat/lon it matches to **5.1e-12 km**; without it, **1,926 km**. Two
consequences: any newly built catalog must use the same convention or it lands
in a mirrored frame, and the head's `theta` is *already* a geographic azimuth
(clockwise from North), so the θ→strike readout needs no 90° rotation — an
earlier version applied one and would have reported every rupture azimuth
rotated by 90°.

**Resource incident, 2026-08-02.** Running the sweep at concurrency 6 with
matched-precision `n_sims` took a 48 GB machine to **0 GB free and 34 GB of
swap**. Two compounding causes, both now fixed:

- `ntest.simulate_windows` appended a **full lane-width array per step**. With
  matched precision the lane count reaches ~66k (52 windows × 1277 sims) and the
  horizon cap is 18,000 steps, so the records alone ran to tens of GB. Records
  now hold only live entries; the same 66k-lane configuration peaks at
  **2.48 GB**, and `tests/test_target_process.py` asserts the bound.
- The worker pools never checked memory. Each worker holds the full catalog
  tensors — `lastk` is ~400 MB at mc 1.5 and ~2.5 GB at template-catalog scale —
  so concurrency alone can exhaust a machine. `wait_for_memory` now blocks a new
  worker until `--mem-per-worker` GB looks free, and `max_lanes` caps the
  simulator's working set regardless of what the caller asks for.

Lesson for the cloud runs: memory, not compute, is the binding constraint at low
mc. Size the instance by `lastk` × concurrency, not by core count.

**Orphan incident, same day, and worse in kind.** Stopping the sweep mid-run
left **three `flowquake.train` children alive holding 11.7 GB**, with free
memory at 1.0 GB and swap at 8.4 of 9.2 GB. `pkill -f scaling_curve` had
reported success — it matches the *parent's* command line, and the children's
does not contain it. So the machine was fully occupied by a run that every
obvious check said had stopped, which is worse than a leak: the standard
diagnostic actively lies.

Fixed in `flowquake/proc.py`. Children are spawned via `spawn()` into their own
process group and registered; SIGTERM/SIGINT/SIGHUP and `atexit` terminate the
whole set, with SIGKILL after a grace period. `tests/test_proc.py` reproduces
the incident end to end — parent spawns a long-lived child, parent is killed,
the child must not outlive it.

Generalise it: **any long-running job here must be verified stopped by checking
the machine, not by checking that the kill command returned 0.**

**And measure memory with the right counter.** A later watchdog killed a
perfectly healthy RV5 run because it thresholded on `vm_stat`'s *Pages free*.
On this 48 GB machine that counter read **0.1 GB** while 21 GB sat *inactive*
and macOS itself reported **89% free** — macOS keeps free pages near zero by
design and uses the remainder as reclaimable cache. Available memory is
`free + inactive`, which `scripts/scaling_curve.py:free_gb` already computes
correctly; only the ad-hoc shell watchdog was wrong.

The counters that actually distinguished the two events:

| | genuine incident | false alarm |
|---|---|---|
| Pages free | 0 GB | 0.1 GB |
| free + inactive | ~0 GB | **21.6 GB** |
| compressor | **29.5 GB** | 1.2 GB |
| swap in use | 34 / 41 GB | 5.2 / 6.1 GB (sticky, not live) |

So: trigger on **`free + inactive`** and on **compressor size**. Swap *level*
is a poor signal on macOS because it does not drain once allocated; swap
*growth* would be usable, level is not.

**Performance work, measured not assumed.** MPS is 1.6× *slower* than CPU at
production model size (0.03M params — dispatch-bound, not FLOP-bound); 12
threads on one run buys 2.24× while six 1-thread processes buy 3.75×; leaving
BLAS threads unpinned at 16-way measured 0.3×. So the sweep is a 6-way process
pool with threads pinned to 1, and `ntest.simulate_windows` batches all forecast
windows into the lane dimension for a further **3.9×** (52 windows × 100 sims:
35.5s → 9.0s). Native/SIMD work was considered and rejected: the hot path is
per-op dispatch on tiny tensors, not arithmetic. The one place native code will
pay is QTM scale, where `lastk` is ~2.5 GB/process — that needs an mmap'd shared
array and is not yet done.

**Bugs caught by building the harness, each of which would have produced a
plausible wrong answer:** a 974k-cell grid that made the Poisson score measure
the water level; the magnitude-tail confound (invariant 1c); an O(E²) pair
builder that hung; an ETAS EM that ran away to branching ratio 469.9.
