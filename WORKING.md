# FlowQuake — working document

Current state, what is left, and who has to do it. Numbers here were read from
the committed artifacts under `runs/`; the full claim-to-artifact map, including
every claim that failed to trace, is `results/CLAIMS.md`.

## What this project claims, in plain language

Earthquake catalogs are the standard test bed for point-process forecasting, and
the standard model is ETAS: a hand-built recipe in which each earthquake raises
the rate of nearby future earthquakes by an Omori decay in time, a power-law
kernel in space, and a Gutenberg–Richter law in magnitude. Neural point
processes are far more flexible than ETAS and have repeatedly lost to it on the
EarthquakeNPP benchmark.

FlowQuake claims three things.

1. **The temporal loss flips when the model sees the whole catalog.** A
   Mamba-style selective state-space encoder reads the entire history instead of
   a fixed window, and a rectified-flow head gives exact likelihoods for the
   next event's time. On all five California catalogs of the benchmark the
   3-seed mean temporal log-likelihood beats region-fitted ETAS: ComCat
   1.486833 vs 1.434343, White 2.066893 vs 2.021097, San Jacinto 1.160957 vs
   1.132527, Salton Sea 2.433720 vs 2.332039, SCEDC 2.619408 vs 2.540983
   (`runs/fullsuite_summary.json`). Four of five are significant under a
   stationary block bootstrap; San Jacinto's interval crosses zero and is
   recorded as a tie.

2. **The flexibility is what kills the spatial head, and the fix is structure.**
   Letting the heads see the learned whole-catalog embedding causes catastrophic
   memorization: train NLL drops to 4.14 while held-out NLL blows up to 19.65,
   a gap of 15.50, versus 7.28 / 7.62 / 0.34 with the bottleneck closed
   (`runs/ablation_h/memorization_figure.json`, `h=4` and `h=0` at `ckpt_last`).
   Early stopping does not rescue it — the best held-out checkpoint for every
   `h>0` is the first one evaluated, at step 250. So production runs
   `h_bottleneck=0` and the spatial head is built from structure instead: a
   full-history neural-ETAS density that, with its gate closed, reproduces the
   benchmark's published ETAS spatial log-likelihood to 1.77e-9 nats
   (`runs/etas_sll_repro.json`), then adds a causal multi-scale
   smoothed-seismicity background and per-parent neural modulations that never
   see the target location.

3. **With that head the composite beats ETAS on total likelihood, in six
   regions and out of time.** Per-event total gain vs each region's own ETAS
   inversion: California +0.1133, Italy +0.2095, Japan +0.0390, Chile +0.0608,
   Greece +0.0756, Iran +0.0844 nats/event, all Holm-adjusted p ≤ 0.0185
   (`runs/stats_hardening.json` → `total_with_head_family`). On ComCat that is
   nll 7.142122 against ETAS's 7.255428. On a 2020–2026 forward window of 10,187
   events never used for fitting or early stopping, all three gains replicate:
   dT +0.0574, dS +0.0666, dTot +0.1241 [0.1035, 0.1455]
   (`runs/total_win.json`).

The bound on all of this: the head is **initialized from each region's ETAS
inversion**, and per-region normalization plus a train-era smoothed-seismicity
background map are still required. So this is an upgrade path for a deployed
ETAS system, not an inversion-free replacement for one. `REPLACEMENT_READINESS.md`
states the same boundary and is the file to quote from.

## Current state

`runs/` is committed: 226 files, 136 summary JSONs and 90 run configs. The
evidence trail behind essentially every reported number is now in the
repository, and `results/CLAIMS.md` maps each claim to its key. Across 142 traced
claim rows (134 distinct claims), 114 match the artifact exactly or to rounding,
2 are ambiguous between two committed artifacts, 8 distinct claims are
contradicted by their artifact, and 12 distinct claims have no committed backing
at all. Note that "matches to rounding" is the larger half of that 114: 63 rows
match at the artifact's own precision, 51 round to the printed value. Both are
sound, but only the first is exact.

**Aggregation is clean.** All 30 values in `runs/fullsuite_summary.json` (3-seed
mean and sample sd of `tll`/`sll`/`nll` for five datasets) recompute from the 15
per-seed `eval_test.json` files: 29 to a difference of exactly zero, and
`SaltonSea_10.sll_sd` to 8.7e-19 — one unit in the last place, i.e. float
summation-order noise, not a discrepancy in the stored value. The three
paired CSEP statistics were re-derived from raw per-day S-test quantiles rather
than trusted as stored summaries: head vs ETAS on 83 shared days is 77 passes
each with 10 discordant days split 5–5 and McNemar exact p = 1.0000; head vs the
production kernel-mixture head is 75/81 vs 78/81 with p = 0.5078125. The four
CSEP runs' `results[].day` lists are element-for-element identical, so the
"identical 100 forecast days" claim holds literally.

**What is still external.** `reference/` is a clone of the EarthquakeNPP
benchmark and is not committed; nothing trains or evaluates without it. All 90
committed run configs under `runs/` have `catalog_path` under it, as do all 33 in
`configs/` — 123 tracked YAMLs, not one of which resolves without
`reference/`. Required and *not* listed in
`README.md:132-148`: `Datasets/NewZealand/`, `Datasets/Italy_Mw/` plus
`Italy_mw_raw/`, `Datasets/ComCat_forward/`, `Datasets/ComCat_extended/`,
`Experiments/ETAS/pycsep_tests_parallel.py`, and
`output_data_<Cfg>/parameters_0.json` (which `flowquake/etas_csep.py` needs).

Worse, six ETAS configs the manuscript depends on — `Japan_25`, `Chile_25`,
`Greece_25`, `Iran_25`, `Italy_25` and `ComCat_25_refit2020` — are not shipped
by the benchmark, are written by no script in this repo, and live only inside
the gitignored `reference/` tree. Nobody but the author can regenerate the §4.5
region baselines or the §4.1 refit control.

**What cannot be re-derived here at all.** Per-event score CSVs are excluded by
`.gitignore`, so every block-bootstrap CI and everything in
`runs/stats_hardening.json` is a summary. One per-event file is tracked
repo-wide (`runs/neural_etas/ComCat_25/per_event_forward_full.json`). No
checkpoints are tracked (`git ls-files | grep '\.pt$'` is empty), which makes
§4.3's "reproducible from the committed checkpoints" wrong as written — the
numbers are committed, the checkpoints are not.

**Readiness verdict.** `runs/replacement_readiness.json` reports `overall:
"RESEARCH_PREVIEW_READY"` across 15 checks, 11 PASS and 4 WARN. Rungs 1–3 of
`REPLACEMENT_READINESS.md`'s ladder are done; rung 4, a prospective deployment
on a window not used for model selection, is not.

## Target venue

Primary: **JGR: Solid Earth**. The load-bearing claims are seismological — CSEP
consistency, completeness-regime transfer, per-region ETAS inversions — and need
reviewers who can judge whether a pyCSEP N/S/M pass rate and a matched
simulation budget mean what the paper says they mean. The paper is also long
(988 lines with ablations, controls and a claim-boundary section), which fits a
full research article rather than a letter.

Fallback: **TMLR**. EarthquakeNPP itself is TMLR 2026 (Stockman, Lawson &
Werner), so the benchmark, the split protocol and the reference NPPs are already
familiar to that reviewer pool; there is no page limit; and the memorization
result in §4.3 is a machine-learning finding that would land well there. The
cost is that CSEP consistency and the ETAS-inversion provenance question get
less scrutiny than they deserve.

Not recommended: *Seismological Research Letters*. The short format would force
cutting either the ablation ladder or the claim-boundary section, and both are
what make the headline defensible.

`MANUSCRIPT.md:950` flags the venue decision as the author's. Treat the above as
a recommendation, not a settled choice — but decide before the format
conversion in the laptop list, because the two are the same piece of work.

## Laptop work

Ordered. Everything here is doable on a laptop with no GPU and no `reference/`
tree. Each item is checkable against a file already in the repo.

1. **Reconcile the ComCat Δ cell.** `MANUSCRIPT.md:298` prints Δ +0.053 in a
   column whose other four cells are 3-seed differences (WHITE 0.045796,
   SanJac 0.028430, SaltonSea 0.101681, SCEDC 0.078426, all rounded correctly).
   ComCat's 3-seed difference is 0.05248974448868626 → +0.052, which is what the
   body says three lines later at `MANUSCRIPT.md:304`. The printed +0.053 is the
   single-seed `n1_density` paired bootstrap mean 0.053296262845673396 — a
   legitimate number that §4.5 and the abstract quote correctly, but it does not
   belong in this column. Either print +0.052 or add a footnote saying the
   column mixes bases. Check: the table cell and `:304` agree.

2. **Fix the per-seed spread bound.** `MANUSCRIPT.md:570` claims "per-seed
   spread ≤0.003"; `MANUSCRIPT.md:972` claims ≤0.006. Actual max−min over
   `runs/neural_etas/<region>/summary_full_s{0,1,2}.json` → `dS_mean`: ComCat
   0.0008, Italy 0.0019, Japan 0.0003, Chile 0.0070 (0.0267/0.0337/0.0277), Iran
   0.0056, Greece 0.0045. Write ≤0.007 in both places. The clause that matters —
   "all six clear zero at every seed" — is true; every per-seed `dS_ci` is
   strictly positive, so the conclusion survives untouched. Check: no stated
   bound below 0.0070.

3. **Relabel the composite totals as seed 0, or recompute them.**
   `MANUSCRIPT.md:5-7` advertises "Per-event and full-suite results are 3-seed
   (mean ± std)", and §4.4 reports the six totals in the same paragraph as
   3-seed head means, but `scripts/stats_hardening.py:167` reads
   `runs/neural_etas/<head>/per_event_full_s0.csv` — one seed. The arithmetic
   proves it: Italy 0.2095 − 0.0712 = 0.1383, which is the s0 `dS` exactly, not
   the 3-seed 0.1373. Material only for Chile: stated +0.061, 3-seed would be
   +0.064 (0.0343 + 0.029367). Italy would move 0.210 → 0.209; California and
   Japan are unchanged at 3 d.p. Relabelling is the laptop fix; recomputing is
   item 1 of "Needs hardware or data". Check: no sentence claims 3-seed totals
   while `stats_hardening.py:167` reads `_s0`.

4. **Correct the Japan→Greece transfer sentence.** `MANUSCRIPT.md:606-610` says
   a transferred background-only head "still wins" and that for Japan→Greece
   "the modulation is what converts a non-win into a win". The artifacts say
   background-only is **−0.015**, a loss, and the full transferred head is
   dS 0.0282 with ci [−0.0121, 0.0676] and `decision: "tie"`
   (`runs/neural_etas/spatial_transfer_summary.json` →
   `mlp_decomposition_japan_source.Greece`;
   `runs/neural_etas/transfer_from_Japan_25.json` →
   `targets.Greece_25.zero_shot`). Loss → tie, not non-win → win. The "+0.01 to
   +0.04" modulation increment is fine (0.0101–0.0432). Check: the sentence says
   "tie" and the "still wins" generalization is qualified.

5. **Replace the Greece and Iran native CIs.** `MANUSCRIPT.md:658-659` print
   [−0.144, −0.070] and [−0.347, −0.205]. Nothing in `runs/` produces either.
   Both committed sources agree with each other:
   `runs/multiregion_master.json` → `Greece.native.paired.dT_ci`
   [−0.16308742268956303, −0.04480017107870762] and `Iran.native.paired.dT_ci`
   [−0.36975851467862897, −0.173887497485534]; `runs/prospective.json` gives
   [−0.16285, −0.04604] and [−0.36926, −0.17691]. The stated intervals are
   roughly 35% narrower than the real ones — they read like leftovers from an
   earlier bootstrap configuration. The point estimates −0.107 and −0.276 are
   exact. Note also that the table cites `runs/stats_hardening.json`, whose
   Greece and Iran `per_region` rows are the **few-shot** variant, so the cited
   artifact cannot supply these two rows at all; fix the citation too. Check:
   both intervals appear verbatim in `multiregion_master.json`.

6. **Cut or downgrade the "every era" claim.** `MANUSCRIPT.md:328-331` says the
   time-binned analysis shows the California and Chile temporal wins are
   "individually significant in every era of their test windows".
   `runs/prospective.json` stores no per-window CI or p-value of any kind, and
   Chile has only 10 of 19 180-day windows positive
   (`Chile.native.bins_dT_positive_frac` 0.5263), with negatives to −0.0557.
   California is 23 of 27 (0.8519). This is the one claim in the manuscript that
   the artifact actively contradicts rather than merely fails to support.
   Replace it with the window fractions, which are backed. Check: no sentence
   asserts per-era significance.

7. **Re-aggregate the two 10⁴ CSEP summaries.** `runs/n1_density/csep/csep_results.json`
   day 2982 records `S.quantile [-1.0, -1.0]` with `observed` NaN — the harness's
   own not-evaluable sentinel, which `flowquake/csep_forecast.py:246` excludes.
   Re-running the committed `csep_summary()` over that file's own `results[]`
   gives S 85/91 = 0.9341, not the stored 85/92 = 0.9239 that
   `MANUSCRIPT.md:385` quotes. Same off-by-one in `runs/final_s1555/csep`
   (stored 81/92 = 88.0%, recomputed 81/91 = 89.0%), which is the source of the
   "S 88% → 92%" comparison at `MANUSCRIPT.md:391`. The correction runs *in the
   author's favour*: `MANUSCRIPT.md:389`'s "rejects slightly more often than
   nominal (8%)" is really 6.6%. The four 10³ runs are already self-consistent
   with current code. `MANUSCRIPT.md:977-979` claims this denominator was
   already "corrected to 92 evaluable days" — that fix landed in prose only.
   Check: stored `summary.S.n_eval` equals what `csep_summary(results)` returns.

8. **Settle the ETAS fork.** `pyproject.toml:25-38` records two candidate
   implementations, `lmizrahi/etas` and `ss15859/etas`, states they are
   different code, and marks the choice `TODO [USER, blocks release]`. Nothing
   in the repository can decide it: no version, commit, sha, package, env,
   provenance, repo or git key exists in any of the 136 committed run JSONs or
   90 committed YAMLs, and the invert/predict logs are gitignored and absent
   from disk. `git show 95e6b92^:requirements.txt` shows the `ss15859` line was
   **commented out**, so it is weaker evidence than it looks. The decisive
   artifact is `etas-*.dist-info/direct_url.json` in the training environment,
   which records the git URL and resolved commit for a VCS install; the full
   command list is in `results/CLAIMS.md` under "ETAS provenance". This matters
   because the fork is what every reported gain is measured *against*: it
   affects the five §4.5 region inversions, `ComCat_25_refit2020`, and the §4.2
   ETAS CSEP column, which imports the installed package at runtime
   (`flowquake/etas_csep.py:70-71`). California is safe either way —
   `runs/forward_etas/summary.json` → `params_frozen_from` is
   `"ComCat_25 inversion (train<=2007, published with benchmark)"`, i.e. the
   benchmark's own shipped output. Pin with a commit SHA once known and cite it
   in Methods. Check: `pyproject.toml` has one pinned fork and no `TODO`.

9. **Close out the Poisson baseline row.** It is now fully backed and needs no
   new run. `flowquake/evaluate.py:104-110` copies `ll_scores.json` verbatim
   into every eval JSON, so `runs/n1_density/eval_test.json` →
   `baselines.Poisson` holds `tll` 0.5126406686259881, `sll` −13.774504128914366,
   `nll` 13.261863460288378, matching `README.md:51` on all three; the companion
   ETAS row at `README.md:50` is `baselines.ETAS` = 1.4343428344882627 /
   −8.689770387238827 / 7.2554275527505645. Change `README.md:46` to cite the
   committed `runs/n1_density/eval_test.json` key rather than the absent
   `reference/` path, keeping the original provenance as a parenthetical. **Do
   not** cite the `baselines` block of a foreign-region run: `runs/{chile,greece,
   italy,japan}_n1/eval_test.json` all carry the California ComCat numbers
   because `--etas_dir` defaults to `output_data_ComCat_25`. Check:
   `README.md:46` names a committed file.

10. **Commit the six author-authored ETAS configs.** `Japan_25`, `Chile_25`,
    `Greece_25`, `Iran_25`, `Italy_25` and `ComCat_25_refit2020` are not shipped
    by the benchmark and are generated by no script here, so
    `README.md:126-129` and `REPRODUCE.md:23-28` describe a reproduction path
    that only works on the author's machine. Copy them to `configs/etas/` and
    have `scripts/run_etas_regions.py` install them into
    `reference/Experiments/ETAS/config/`. Check: `REPRODUCE.md` §2 runs on a
    fresh clone plus a bare benchmark clone.

11. **Convert `MANUSCRIPT.md` to the chosen house format with numbered
    figures.** The manuscript currently carries bare slugs, not captioned
    numbered figures: `MANUSCRIPT.md:489` "(Fig. memorization_curve)", `:501`
    "(Fig. spatial_gap)", `:561` "(Fig. neural_etas_modulation)", `:644` "(Fig.
    density_dependence)", `:757` "(Fig. data_efficiency)". Twelve PNGs are
    tracked in `figures/`: `fig_fullsuite.png`, `fig_csep.png`,
    `fig_csep_headtohead.png`, `fig_memorization.png`,
    `fig_memorization_curve.png`, `fig_neural_etas_modulation.png`,
    `fig_spatial_gap.png`, `density_dependence.png`,
    `data_efficiency_chile.png`, `global_vs_etas.png`,
    `multiregion_transfer.png`, `transfer_japan.png` — so seven tracked figures
    are currently cited by no slug at all. Assign Figure 1…n in order of first
    reference, write a caption per figure stating the artifact it was drawn
    from, and replace each slug with the number. Also number the tables (§4.1
    suite, §4.2 standalone and head-to-head, §4.4 six-region, §4.5 density and
    LOO). Do this after item 8, so the Methods ETAS citation is final before the
    format freeze. Check: no "(Fig. <slug>)" remains, and every tracked PNG is
    either numbered or deleted.

12. **Capture or cut the twelve unbacked numbers.** They are listed as N1–N12 in
    `results/CLAIMS.md`. The ones that are pure laptop work: delete the "2–10 km
    band" phrasing at `MANUSCRIPT.md:501` and `:503`, since no script computes
    that band (`scripts/make_figures.py` uses [<0.5, 0.5–1, 1–2, 2–5, 5–10,
    10–30, >30] and `scripts/spatial_gap_decomp.py` uses [<0.5, 0.5–2, 2–5,
    5–15, 15–50, >50]); and correct `MANUSCRIPT.md:507`'s recurrence figure,
    which `scripts/trigger_coverage.py` computes for a **<5 km** neighbour, not
    the <0.5 km the text states. The rest need a re-run and are in the next
    section. Check: every remaining number in §4.4's localization passage has a
    `results/CLAIMS.md` row that is not NO ARTIFACT.

13. **Remove the stale duplicate artifacts.** `runs/neural_etas/ComCat_25/`
    holds two generations of the same seed-0 runs: `summary_full.json`
    (`dS` 0.0578) beside `summary_full_s0.json` (0.06), and
    `summary_bg_only.json` (0.0497) beside `summary_bg_only_s0.json` (0.0513).
    The manuscript uses the `_s0/_s1/_s2` generation — its quoted per-seed slls
    −8.6298/−8.6299/−8.6291 are exactly those three files — so §4.4's "+0.051"
    resolves to `bg_only_s0`. A future checker reading the unsuffixed file gets
    +0.050 and reports a mismatch. Delete or rename the two unsuffixed files.
    Also `runs/csep_results_s1555.json` is byte-identical to
    `runs/final_s1555/csep/csep_results.json` (verified with `cmp`); drop the
    loose copy. Check: one file per (config, seed) under
    `runs/neural_etas/ComCat_25/`.

14. **Fix the stale `baselines` blocks.** `runs/n1_density/eval_forward.json`
    carries `baselines.ETAS.tll` 1.4343428344882627 — the in-window value — where
    the forward window's is 1.0102738057926097
    (`runs/forward_etas/summary.json`); its `split` field says `"test"`; and its
    `sll`/`nll` are the production kernel head, not the §4.4 head. Only
    `paired_vs_ETAS.temporal` in that file is safe to quote, and it is correct
    (1.0677136 − 1.0102738 = 0.0574398). The foreign-region `eval_test.json`
    files have the same problem (item 9). Either overwrite the block or add an
    explicit `baselines_note`. Check: no committed eval JSON carries a
    `baselines` block that does not belong to its own window and region.

15. **Repair the remaining documentation inaccuracies.**
    `README.md:132-148` expected-tree omissions (listed in "Current state"
    above); `README.md:150-151` still says on-disk size is `PENDING — measure
    with du -sh`; `README.md:212-215` still says `results/CLAIMS.md` values are
    `PENDING`, which this revision made false; `MANUSCRIPT.md:874` cites the
    spatial transfer as "(§4.5)" when it is reported in §4.4;
    `MANUSCRIPT.md:307` says San Jacinto's interval "touches zero" when
    `runs/replacement_readiness.json` gives [−0.005686476386749143,
    0.07592596149130082] and `decision: "tie"`, i.e. it crosses zero;
    `MANUSCRIPT.md:457-458`'s "they agree on 77/83 evaluable days each" reads as
    an agreement count but is each model's pass count — true concordance is
    73/83; §4.3's "reproducible from the committed checkpoints" should say
    reproducible from the committed *metrics*, since no `.pt` is tracked. Check:
    each of these seven strings is gone or corrected.

16. **Record the bootstrap configuration in the artifacts.** The block length of
    50 events stated at `MANUSCRIPT.md:276` lives only in code
    (`flowquake/stats.py:45`, `:87`, `:116`, `:145`, `:164`, all defaulting
    `mean_block: int = 50`; no `scripts/` call site overrides it). No result JSON
    records it, so a reviewer working from the evidence pack cannot confirm the
    resampling scheme. Have `flowquake/stats.py` return `mean_block`, `n_boot`
    and `seed` alongside every CI and write them into the summaries. Check: at
    least one committed JSON states the block length.

17. **Reconcile the Japan event count.** 14886 in `runs/stats_hardening.json`
    and `runs/completeness.json`, 14888 in `runs/multiregion_master.json`
    (`Japan.native.paired.n`). Japan's native `dT` is quoted as −0.014 in the
    §4.5 density table (from `stats_hardening`, n=14886) and −0.015 in the LOO
    table (from `multiregion_master`, n=14888) — both trace, to different
    pairings of the same comparison. Pick one pairing and note the two-event
    difference. Same class of issue: the bootstrap `n` is one lower than the
    eval `n` on SanJac_10 (4399 vs 4400) and SaltonSea_10 (4103 vs 4104), one
    event lost in the FlowQuake/ETAS time merge. Check: one `n` per region per
    table, with the merge loss stated.

18. **Soften the N 73/100 → 97/100 causal story.** `MANUSCRIPT.md:433-436`
    attributes the earlier ETAS N under-prediction solely to the fitted
    inversion's source set not being re-conditioned on post-`test_start`
    mainshocks. But `runs/etas_csep_pod/csep_results.json` records `n_sims`
    10000 while `runs/csep_h2h_etas/csep_results.json` records 1000, and
    `n_sims` is passed straight through as pyCSEP's `n_cat`
    (`flowquake/csep_forecast.py:180`). If the pod run scored ~1000 real
    catalogs with `n_cat` 10000, pyCSEP pads with ~9000 empty catalogs, which by
    itself produces the observed under-prediction — and that run's M rate is
    also depressed (73/92), which padding explains and a source-set bug does
    not. The two endpoints are backed; the single stated cause is not. Check:
    the passage reports both differences between the runs.

## Needs hardware or data

- **3-seed composite totals.** Item 3's recompute needs
  `runs/neural_etas/<head>/per_event_full_s{1,2}.csv` and the six FlowQuake
  temporal `per_event_test.csv` files, all excluded by `.gitignore`. If they
  survive in the original run tree this is one CPU re-run of
  `scripts/stats_hardening.py` with `HEAD_COMBOS` pointing at each seed; if not,
  it is six head evaluations on GPU.
- **The §4.4 localization artifacts** (N6–N8): the <0.5 km deficit shrinking
  from −0.218 to −0.062, the 64% recurrence figure and the 85% out-of-window
  figure. `runs/n1_density/spatial_gap_decomp.json` contains only a
  background-vs-triggered split (`gap` 0.3640368794995337, entirely attributed
  to `gap_from_background_bg>0.5`, with `gap_from_triggered_bg<0.5` 0.0) and no
  distance strata. `scripts/trigger_coverage.py` prints and writes nothing.
  Needs the per-event CSVs plus a script change to persist the strata.
- **The 9.5e-7-nat simulator validation** (N5). Nothing in `runs/**/*.json`
  holds any value in 1e-8…1e-5. `flowquake/neural_etas_forecast.py:130` prints a
  max-abs-err over 40 randomly sampled events with a 1e-3 threshold, which is
  weaker than what `MANUSCRIPT.md:448-449` states. Either re-run the validation
  over the full test set and write it to
  `runs/n1_density/csep_head/csep_results.json`, or restate the claim as the
  40-event spot check it is.
- **The ETAS temporal-term reproduction** (N4): ~1e-5/event agreement and the
  1.5e-4 anchor effect. `runs/etas_sll_repro.json` covers the spatial term only.
  Needs a temporal counterpart to `scripts/etas_sll_repro.py`, plus
  `reference/`.
- **The refit2020 parameter table** (N2, N3): `a`, log10 μ, ρ, branching ratio
  and the 12 EM iterations. Recoverable by dumping
  `reference/Experiments/ETAS/config/ComCat_25_refit2020.json` and the inversion
  log into `runs/forward_etas_ComCat_25_refit2020/summary.json`. Needs the
  gitignored tree, not new compute.
- **The optional full flETAS baseline** (`MANUSCRIPT.md:970-971`): EM with a free
  background, beyond the committed SGD refit control
  (`runs/neural_etas/ComCat_25/summary_refit_globals_s0.json` → `dS_mean`
  0.0564). CPU, hours per region. The manuscript already frames the SGD control
  as a conservative lower bound, so this is strengthening, not repair.
- **`reference/` itself**: the five California catalogs plus
  `california_shape.npy` and `Datasets/plot_utils.py`; the five agency catalogs
  (regenerable via `scripts/build_region.py`, `REPRODUCE.md` §1); the
  benchmark's `invert_etas.py`, `predict_etas.py` and
  `pycsep_tests_parallel.py`; and `output_data_<Cfg>/{ll_scores.json,
  augmented_catalog.csv, parameters_0.json}` per catalog.
- **A future catalog window.** Not obtainable by any amount of compute — see
  below.

## The collaboration hole

Every result in this repository is retrospective. The 2020–2026 window is the
strongest thing here — 10,187 events, frozen model, dT +0.0574, dS +0.0666,
dTot +0.1241 [0.1035, 0.1455] (`runs/total_win.json`) — and it is still a
retrospective out-of-time replication. `runs/total_win.json`'s own `notes` field
says so in the first line, and `REPLACEMENT_READINESS.md:40-41` lists it under
"Holes That Still Matter" for exactly this reason.

The gap is rung 4 of the ladder at `REPLACEMENT_READINESS.md:59-61`: "freeze a
checkpoint and run rolling forecasts on a future catalog window that was not
used for model selection." Rungs 1–3 are marked `[DONE]`. Rung 4 is the only
remaining rung before the operational-artifact work, and `README.md:13` and
`:198-200` both say the same thing — what remains is registering a genuinely
future rolling forecast.

**This is structurally impossible to do alone, and that is what makes
co-authorship legitimate.** A registered prospective CSEP forecast requires a
third party to hold two things the author cannot self-certify: the frozen model,
deposited and hashed before the forecast period opens, and the future catalog,
which does not exist yet. The whole evidential value of the exercise comes from
the custody being external. No amount of additional retrospective work — more
regions, more seeds, a full flETAS baseline — substitutes for it, because the
objection being answered is not "is the model good" but "did anyone see the
answers first". A CSEP testing centre, or any group that operates one, supplies
precisely that and nothing else in this repository can.

Concretely, the contribution to ask for is: register the frozen
`runs/n1_density` checkpoint plus the full-history spatial head as a forecast
model in a CSEP testing centre, on the same 1-day-ahead N/S/M protocol already
implemented in `flowquake/csep_forecast_head.py`, against ETAS as the
comparison model, for a declared prospective period. The harness is done and
validated at a matched 10³-catalog budget on 100 days; what is missing is a
custodian and a clock.

Second-order and much weaker: an independent group re-inverting ETAS for the
five §4.5 regions with a named, pinned `etas` build would settle N1 from the
outside. Useful, but this is a favour, not a co-authorship.

## Do not do

- Do not push, and do not change repository visibility.
- Do not quote the §4.1 table's ComCat Δ of +0.053 as a 3-seed number; the
  3-seed value is +0.052 (item 1).
- Do not quote the six composite totals as 3-seed until item 3 is resolved.
- Do not cite `runs/stats_hardening.json` → `per_region.California.dTot_mean`
  (−0.3107, `"loss"`) as the total-likelihood headline; that is the base
  kernel-mixture model. The headline is
  `total_with_head_family.California.dTot_mean` (+0.1133, `"win"`). Two keys with
  the same name and opposite verdicts.
- Do not cite the `baselines` block of `runs/n1_density/eval_forward.json` or of
  any foreign-region `eval_test.json`; both are stale (item 14).
- Do not pin the `etas` fork by guessing which of the two candidates it was.
  Resolve it from the training environment or say in Methods that it is
  unresolved.
- Do not describe the 2020–2026 window as prospective, registered, or
  operational. It is a retrospective out-of-time replication.
- Do not fill `results/CLAIMS.md` by copying from `MANUSCRIPT.md`. Its only value
  is being an independent read of the artifacts; a mismatch is a finding.
- Do not force-add checkpoints, per-event CSVs or CSEP catalog grids wholesale
  to make a claim checkable. Add the single file a specific claim needs, with
  `git add -f`, and say in `results/CLAIMS.md` why.
- Do not delete a NO ARTIFACT row from `results/CLAIMS.md` without either
  capturing the number or cutting the claim from the manuscript.
