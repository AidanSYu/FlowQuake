# Claim → evidence map

One row per reported claim, tracing it to the run directory the artifact lives
in, the exact JSON key to read, and the value that key actually holds.

Values below were read from the committed artifacts, not copied from
`MANUSCRIPT.md` — the point of this file is to be an independent check on the
manuscript, so a mismatch is a finding, not a typo.

`.gitignore` admits the summary JSONs named here. Checkpoints (`*.pt`),
precomputed features (`*_trigfeat.npz`), simulated CSEP catalogs
(`CSEP_day_*.csv`) and per-event score CSVs stay out; force-add an individual
one with `git add -f <path>` if a claim needs the raw pairing. Consequence: the
per-event pairings behind every block-bootstrap CI and behind
`runs/stats_hardening.json` exist as summaries only and cannot be re-derived
from this repository.

Config paths are relative to `configs/`, run paths to `runs/`, script paths to
`scripts/` unless a `flowquake.` module is named.

**Status legend.** `MATCH` — artifact equals the stated value at the artifact's
own precision. `ROUNDING` — artifact carries more digits and rounds exactly to
the stated value. `AMBIGUOUS` — two committed artifacts could back the stated
value and they disagree. `MISMATCH` — the artifact contradicts the manuscript.
`NO ARTIFACT` — nothing committed backs the number.

Coverage: **142 claim rows** across the five family tables below, counted by
status exactly as written:

| status | rows |
|---|---|
| `MATCH` | 63 |
| `ROUNDING` | 51 |
| `NO ARTIFACT` | 13 |
| `MISMATCH` | 11 |
| `AMBIGUOUS` | 2 |
| `MATCH` with a `NO ARTIFACT` caveat (T23 only) | 1 |
| MISMATCH-adjacent wording (X31 only) | 1 |
| **total** | **142** |

Those 142 rows cover **134 distinct claims**: several `MISMATCH`/`NO ARTIFACT`
findings are cited from more than one family, so the 11 `MISMATCH` rows are
**8 distinct** contradictions (M1–M8), and the 13 `NO ARTIFACT` rows plus T23
cover **12 distinct** unbacked claims (N1–N12) — S28 alone carries three of them
(N6, N7, N8) and T23 carries N11. The two READ FIRST sections below enumerate
those 8 + 12 + 2 findings once each, and every one of them is cross-referenced
from at least one family table.

`ROUNDING` is the second-largest bucket and is not a weaker form of `MATCH`
being hidden: it means the artifact carries more digits than the manuscript
prints and rounds to the printed value exactly. The distinction is recorded per
row so a reader can re-derive either.

---

## READ FIRST — every MISMATCH

Eight claims are contradicted by the artifacts they rest on. None of them
inverts a headline result; all of them are wrong as written.

| # | location | stated | artifact says | artifact |
|---|---|---|---|---|
| M1 | `MANUSCRIPT.md:570` (and the tail note at `:972`, which says ≤0.006) | per-seed spread of the six head `dS` values is ≤0.003 | Chile range 0.0070 (0.0267/0.0337/0.0277, sample sd 0.0038), Iran 0.0056, Greece 0.0045 — 3 of 6 regions exceed the bound, Chile by 2.3×. The companion clause "all six clear zero at every seed" **is** true: every per-seed `dS_ci` is strictly positive | `neural_etas/{Chile,Iran,Greece}_25/summary_full_s{0,1,2}.json` → `dS_mean` |
| M2 | `MANUSCRIPT.md:5-7` vs `:574-577` | front matter: "Per-event and full-suite results are 3-seed (mean ± std)", and §4.4 reports the six composite totals in the same paragraph as 3-seed head means | the six totals are **seed 0 only**. `scripts/stats_hardening.py:167` reads `runs/neural_etas/<head>/per_event_full_s0.csv`. Arithmetic confirms it: Italy 0.2095 − 0.0712 = 0.1383 = the s0 `dS` exactly, not the 3-seed 0.1373. A 3-seed Chile total would be **+0.064**, not the stated +0.061; Italy would be +0.209 | `stats_hardening.json` → `total_with_head_family.*.dTot_mean` |
| M3 | `MANUSCRIPT.md:606-610` | a transferred background-only head "still wins", and "for Japan→Greece the modulation is what converts a non-win into a win" | Japan→Greece background-only is **−0.015**, a loss, not a win; the full transferred head is `dS` 0.0282 with `ci` [−0.0121, 0.0676] and `decision: "tie"`. The modulation converts a loss into a **tie**. (The "+0.01 to +0.04" modulation increment is fine: 0.0101–0.0432) | `neural_etas/spatial_transfer_summary.json` → `mlp_decomposition_japan_source.Greece`; `neural_etas/transfer_from_Japan_25.json` → `targets.Greece_25.zero_shot` |
| M4 | `MANUSCRIPT.md:658` (§4.5 density table) | Greece native `dT` 95% CI **[−0.144, −0.070]** | no committed artifact produces that interval. Both sources agree with each other and disagree with the manuscript: **[−0.16308742268956303, −0.04480017107870762]** (`multiregion_master.json`) and [−0.16285061695258432, −0.04604435366769163] (`prospective.json`). Stated interval is ~35% narrower. The point estimate −0.107 is exactly right | `multiregion_master.json` → `Greece.native.paired.dT_ci` |
| M5 | `MANUSCRIPT.md:659` (§4.5 density table) | Iran native `dT` 95% CI **[−0.347, −0.205]** | **[−0.36975851467862897, −0.173887497485534]** (`multiregion_master.json`); [−0.36925978106898144, −0.17691360058207323] (`prospective.json`). Substantially wider than stated. Point estimate −0.276 is exact | `multiregion_master.json` → `Iran.native.paired.dT_ci` |
| M6 | `MANUSCRIPT.md:328-331` | the time-binned analysis "shows the California and Chile temporal wins are individually significant in every era of their test windows" | `prospective.json` stores **no per-window CI or p-value of any kind**, so no era-level significance exists anywhere. Chile has only 10 of 19 180-day windows positive (`bins_dT_positive_frac` 0.5263), with negatives to −0.0557; California is 23 of 27. The claim is contradicted by the artifact it cites | `prospective.json` → `Chile.native.series[].dT`, `.bins_dT_positive_frac` |
| M7 | this file's own former "Open items" (`Figures are gitignored`) | figures not tracked | **12 figures are tracked** (`figures/*.png`, committed in 9507356); `.gitignore` says "Paper figures are tracked". Corrected in this revision | `git ls-files figures/` |
| M8 | `README.md:132-148` (expected `reference/` tree) | the tree lists everything a stranger must supply | incomplete for the committed runs. Also required: `Datasets/NewZealand/NewZealand_catalog.csv` (`runs/newzealand_n1{,_s1553,_s1554}`), `Datasets/Italy_Mw/` + `Italy_mw_raw/` (`runs/italy_mw_n1`, the §4.5 Mw control), `Datasets/ComCat_forward/` (the §4.1 out-of-time claim), `Datasets/ComCat_extended/`, `Experiments/ETAS/pycsep_tests_parallel.py`, and `output_data_<Cfg>/parameters_0.json` (required by `flowquake/etas_csep.py`). The "nothing runs without it" statement itself is accurate — all **90** committed run configs under `runs/` have `catalog_path` under `reference/`, as do all 33 in `configs/` (123 tracked YAMLs, 123 `catalog_path` values, every one under `reference/`) | `runs/*/config.yaml` → `data.catalog_path` |

### Two AMBIGUOUS rows

| # | location | issue |
|---|---|---|
| A1 | `MANUSCRIPT.md:298` (Δ column) | ComCat Δ is printed **+0.053**. The other four Δ cells are the 3-seed differences rounded correctly (WHITE 0.045796→+0.046, SanJac 0.028430→+0.028, SaltonSea 0.101681→+0.102, SCEDC 0.078426→+0.078). ComCat's 3-seed difference is **0.05248974448868626**, which rounds to +0.052 — and the body three lines later (`:304`) says +0.052 for exactly this quantity. The printed +0.053 equals a *different* artifact: the single-seed (`n1_density`) paired block-bootstrap mean 0.053296262845673396, which §4.5 and the abstract legitimately quote. So one cell of an otherwise-3-seed column is a single-seed bootstrap number, and the table disagrees with its own paragraph by 0.001 |
| A2 | `MANUSCRIPT.md:385`, `:391` | the stored S-test denominator is one day too high and the manuscript inherits it. `n1_density/csep/csep_results.json` day 2982 records `S.quantile [-1.0, -1.0]` with `observed` NaN — the harness's own not-evaluable sentinel, which `flowquake/csep_forecast.py:246` excludes. Re-running the committed `csep_summary()` over that file's own `results[]` yields S **85/91 = 0.9341**, not the stored **85/92 = 0.9239** quoted as "92 \| 85 \| 92%". Same off-by-one in `final_s1555/csep` (stored 81/92 = 88.0%; recomputed 81/91 = 89.0%), which is the source of the "S 88% → 92%" comparison at `:391`. Direction is *against* the author — corrected rates are better, and `:389`'s "rejects slightly more often than nominal (8%)" is really 6.6%. The four 10³ runs are all self-consistent with current code; only the two 10⁴ runs are stale. `MANUSCRIPT.md:977-979` claims the denominator was "corrected to 92 evaluable days", so this is a fix that landed in prose but not in the aggregation |

---

## READ FIRST — every NO ARTIFACT

Twelve reported numbers or attributions have no committed backing. They are
listed here so they cannot be dropped silently; each needs to be either
captured to JSON or cut from the manuscript.

| # | location | unbacked number / claim | why |
|---|---|---|---|
| N1 | front matter `:10-11`, `§3:262`, `§4.2:403` | the `etas` implementation that produced every ETAS baseline ("the `etas` inversion", "Mizrahi et al. `etas`") | **no version, commit, sha, package, env, provenance, repo or git key exists in any of the 136 committed run JSONs or 90 committed YAMLs.** The only provenance-shaped keys anywhere are `params_frozen_from`, `etas_name`, `fit_window`, `source`, `seed_dirs`. `pyproject.toml:25-38` records two candidate forks and states outright that the repo cannot decide between them. `scripts/run_etas_regions.py` logs to `runs/etas_{invert,predict}_<Cfg>.log`, excluded by `.gitignore`'s `*.log` and absent from disk. See "ETAS provenance" below for the exact commands that would settle it |
| N2 | `MANUSCRIPT.md:362-363` | refit2020 ETAS parameters: `a` 1.556→1.603, log10 μ −6.333→−6.389, ρ 0.557→0.571, branching ratio 0.968 | `forward_etas_ComCat_25_refit2020/summary.json` holds only `window`, `n`, `tll`, `sll`, `nll`, `etas_name`, `fit_window`, `params_frozen_from`. No ETAS parameter vector is committed anywhere in `runs/`. The cited `reference/Experiments/ETAS/config/ComCat_25_refit2020.json` is inside the gitignored `reference/` tree |
| N3 | `MANUSCRIPT.md:362` | refit2020 EM "converged in 12 iterations" | no iteration/convergence key in that summary |
| N4 | `MANUSCRIPT.md:344-347` | ETAS **temporal** term reproduces the package to ~10⁻⁵ per event; first-target anchor effect mean 1.5×10⁻⁴ nats | `etas_sll_repro.json` covers the **spatial** term only. No temporal-reproduction artifact is committed |
| N5 | `MANUSCRIPT.md:448-449`, `:965` | the full-history gridded simulator reproduces the head's per-event SLL to 9.5×10⁻⁷ nats | exhaustive numeric scan of `runs/**/*.json` finds no value in 1e-8…1e-5. `flowquake/neural_etas_forecast.py:130` only *prints* a max-abs-err, and does so over 40 randomly sampled events with a 1e-3 pass threshold — a weaker statement than the manuscript's phrasing |
| N6 | `MANUSCRIPT.md:500-503` | the <0.5 km deficit shrinks from −0.218 to −0.062 nat/event | `n1_density/spatial_gap_decomp.json` holds only `n` 21889, `fq_sll` −9.053807266738362, `etas_sll` −8.689770387238829, `gap` 0.3640368794995337, `frac_events_etas_wins` 0.5646214993832519, `gap_from_background_bg>0.5` 0.3640368794995337, `gap_from_triggered_bg<0.5` 0.0 — **no distance strata at all**. `scripts/trigger_coverage.py` prints its numbers and writes nothing |
| N7 | `MANUSCRIPT.md:501`, `:503` | the "2–10 km band" improves; residual largest per-event at 2–10 km | no script computes that band. `scripts/make_figures.py` uses [<0.5, 0.5–1, 1–2, 2–5, 5–10, 10–30, >30]; `scripts/spatial_gap_decomp.py` uses [<0.5, 0.5–2, 2–5, 5–15, 15–50, >50] |
| N8 | `MANUSCRIPT.md:507-508` | 64% of ComCat test events recur within 0.5 km of a prior event; 85% of those nearest priors lie outside the last-64-event window | printed by `scripts/trigger_coverage.py`, written nowhere. Secondary problem: the script's printed coverage stat is for a **<5 km** neighbour, not <0.5 km as the manuscript states |
| N9 | `MANUSCRIPT.md:539` | default training initialization starts "+0.002 to +0.004 nats/event above ETAS" | no init-config summary is committed |
| N10 | `README.md:126-129`, `REPRODUCE.md:22-28` | the §4.5 region baselines and the §4.1 refit2020 control are regenerable from this repo | **no script in the repo writes `reference/Experiments/ETAS/config/*.json`.** The six non-benchmark ETAS configs the manuscript depends on — `Japan_25`, `Chile_25`, `Greece_25`, `Iran_25`, `Italy_25`, `ComCat_25_refit2020` — are author-authored, live only inside the gitignored `reference/` tree, and are committed nowhere. A stranger cannot regenerate them |
| N11 | `MANUSCRIPT.md:276` | stationary block bootstrap block length 50 events | correct in code (`flowquake/stats.py:45`, `:87`, `:116`, `:145`, `:164` all default `mean_block: int = 50`; no call site in `scripts/` overrides it), but **no result JSON records the bootstrap configuration**, so the block length is not recoverable from the evidence pack alone |
| N12 | `MANUSCRIPT.md:970-971` | optional full flETAS (EM, free background) baseline | not run; the committed control is the SGD refit (`summary_refit_globals_s0.json`, `dS_mean` 0.0564). Correctly flagged as remaining in the manuscript's own open items |

---

## Family 1 — Temporal (§4.1)

Seed directories are `SEED_DIRS` in `scripts/aggregate_fullsuite.py`; each holds
one `eval_test.json` with `tll`/`sll`/`nll` and `baselines.ETAS.{tll,sll,nll}`.
The 3-seed means are aggregated into `runs/fullsuite_summary.json`.

**Aggregation verified.** All 30 values in `fullsuite_summary.json` (mean and
sample sd, ddof=1, of `tll`/`sll`/`nll` for five datasets) were recomputed from
the 15 committed per-seed `eval_test.json` files. 29 of the 30 reproduce with a
difference of exactly 0.0; the thirtieth (`SaltonSea_10.sll_sd`) differs by
8.7e-19, one unit in the last place, which is summation-order float noise and
not a discrepancy in the stored value. `n_events` is identical across seeds
within each dataset (21889 / 24080 / 4400 / 4104 / 13062). Seeds are genuinely
distinct (1553/1554/1555 per each run's `config.yaml`).

### §4.1 table (`MANUSCRIPT.md:296-302`)

Artifact for every row: `runs/fullsuite_summary.json`. Δ column source:
`runs/replacement_readiness.json` → `checks[california_temporal_suite].evidence.dT`.

| # | claim | stated | artifact value | JSON key | status |
|---|---|---|---|---|---|
| T1 | ComCat_25 FQ tll, 3-seed mean ± sd | 1.4868 ± 0.0008 | 1.486832578976949 / 0.0008107134678745433 (recomputed from `n1_density` 1.4876391887664795, `n1_s1553` 1.4868407249450684, `n1_s1554` 1.4860178232192993 — exact) | `ComCat_25.tll`, `.tll_sd` | MATCH |
| T2 | ComCat_25 region-fitted ETAS tll | 1.4343 | 1.4343428344882627 (identical in `baselines.ETAS.tll` of all three seed `eval_test.json`) | `ComCat_25.etas_tll` | MATCH |
| T3 | ComCat_25 Δ over ETAS | +0.053 | 3-seed gain 0.05248974448868626 (→ +0.052); single-seed `n1_density` paired bootstrap mean 0.053296262845673396 (→ +0.053) | `checks[california_temporal_suite].evidence.dT.ComCat_25` vs `checks[california_block_bootstrap_temporal].evidence.ComCat_25.mean` | **AMBIGUOUS (A1)** |
| T4 | ComCat_25 FQ sll / ETAS sll | −9.06 / −8.69 | −9.058865229288736 (`sll_sd` 0.009024574369430258) / −8.689770387238827 | `ComCat_25.sll`, `.etas_sll` | MATCH |
| T5 | WHITE_06 FQ tll, 3-seed mean ± sd | 2.0669 ± 0.0007 | 2.0668934186299643 / 0.0006617227810225719 (from 2.0676558017730713, 2.066556453704834, 2.0664680004119873 — exact) | `WHITE_06.tll`, `.tll_sd` | MATCH |
| T6 | WHITE_06 ETAS tll | 2.0211 | 2.0210970061274423 | `WHITE_06.etas_tll` | MATCH |
| T7 | WHITE_06 Δ | +0.046 | 0.04579641250252209 | `...evidence.dT.WHITE_06` | ROUNDING |
| T8 | WHITE_06 FQ sll / ETAS sll | −4.73 / −4.26 | −4.725900491078694 / −4.2610686365574395 | `WHITE_06.sll`, `.etas_sll` | MATCH |
| T9 | SanJac_10 FQ tll, 3-seed mean ± sd | 1.1610 ± 0.0009 | 1.1609567801157634 / 0.0008829548198669929 (from 1.161832332611084, 1.1609714031219482, 1.1600666046142578 — exact) | `SanJac_10.tll`, `.tll_sd` | MATCH |
| T10 | SanJac_10 ETAS tll | 1.1325 | 1.1325267069430716 | `SanJac_10.etas_tll` | MATCH |
| T11 | SanJac_10 Δ | +0.028 | 0.02843007317269186 | `...evidence.dT.SanJac_10` | ROUNDING |
| T12 | SanJac_10 FQ sll / ETAS sll | −5.92 / −5.40 | −5.923290252685547 / −5.398118234811221 | `SanJac_10.sll`, `.etas_sll` | MATCH |
| T13 | SaltonSea_10 FQ tll, 3-seed mean ± sd | 2.4337 ± 0.0070 | 2.433719793955485 / 0.00704049643577668 (from 2.4354193210601807, 2.425985097885132, 2.4397549629211426 — exact) | `SaltonSea_10.tll`, `.tll_sd` | MATCH |
| T14 | SaltonSea_10 ETAS tll | 2.3320 | 2.332039202380453 | `SaltonSea_10.etas_tll` | MATCH |
| T15 | SaltonSea_10 Δ | +0.102 | 0.10168059157503206 | `...evidence.dT.SaltonSea_10` | ROUNDING |
| T16 | SaltonSea_10 FQ sll / ETAS sll | −2.64 / −2.32 | −2.637502113978068 / −2.3150835316085487 | `SaltonSea_10.sll`, `.etas_sll` | MATCH |
| T17 | SCEDC_20 FQ tll, 3-seed mean ± sd | 2.6194 ± 0.0031 | 2.619408051172892 / 0.003079546775498436 (from 2.619103193283081, 2.61649227142334, 2.622628688812256 — exact) | `SCEDC_20.tll`, `.tll_sd` | MATCH |
| T18 | SCEDC_20 ETAS tll | 2.5410 | 2.5409825345527426 | `SCEDC_20.etas_tll` | MATCH |
| T19 | SCEDC_20 Δ | +0.078 | 0.07842551662014952 | `...evidence.dT.SCEDC_20` | ROUNDING |
| T20 | SCEDC_20 FQ sll / ETAS sll | −7.85 / −7.53 | −7.848306496938069 / −7.534222208042888 | `SCEDC_20.sll`, `.etas_sll` | MATCH |

### §4.1 body

| # | claim (location) | stated | artifact value | artifact → key | status |
|---|---|---|---|---|---|
| T21 | ComCat paired per-event temporal gain and win rate (`:304`) | +0.052 ± 0.0025, 61% improved | 3-seed mean gain 0.05248967962494191, mean stderr 0.0024462216119013197, mean win rate 0.6103522317145598 (per seed 0.05329626 / 0.05249777 / 0.05167500) | `n1_density`, `n1_s1553`, `n1_s1554` `eval_test.json` → `paired_vs_ETAS.temporal.{mean_gain,stderr,win_rate}` | MATCH |
| T22 | block bootstrap significant on 4 of 5; SanJac positive, interval touches zero (`:305-307`, `:315-317`, abstract `:30-32`) | 4 of 5; SanJac tie | ComCat win, `ci` [0.03968943039040959, 0.06798247572732614]; WHITE_06 win [0.030456473572587874, 0.06461784566084337]; SanJac_10 **tie**, mean 0.029182325735982863, `ci` [−0.005686476386749143, 0.07592596149130082]; SaltonSea_10 win [0.06967317591468951, 0.144360564134678]; SCEDC_20 win [0.06028236923397232, 0.09728371647260345] | `replacement_readiness.json` → `checks[california_block_bootstrap_temporal].evidence.<cat>.{mean,ci,decision}` | MATCH (wording note below) |
| T23 | block length 50 events (`:276`) | 50 | `flowquake/stats.py:45` default `mean_block: int = 50`; no `scripts/` call site overrides it | code only — see **N11** | MATCH / NO ARTIFACT |
| T24 | temporal gain positive in 85% of 180-day windows (`:328-331`) | 85% | 0.8518518518518519 (23 of 27 windows) | `prospective.json` → `California.native.bins_dT_positive_frac` | MATCH |
| T25 | overall ComCat temporal gain, block-bootstrap 95% CI (`:330-331`) | [+0.040, +0.068] | `dT` 0.053296262845673396, `dT_ci` [0.04044003357906552, 0.06782420487674817] | `prospective.json` → `California.native.dT`, `.dT_ci` | MATCH |
| T26 | California headline temporal gain with CI (`:654`, abstract) | +0.053 [+0.040, +0.067] | `dT_mean` 0.0533, `dT_ci` [0.0402, 0.0674], `dT_p_boot` 0.0005, `n` 21889 | `stats_hardening.json` → `per_region.California.dT_mean`, `.dT_ci` | MATCH |
| T27 | California/Chile wins "individually significant in every era" (`:328-331`) | every era | no per-window CI or p-value stored; Chile 10/19 windows positive | `prospective.json` → `Chile.native.series[].dT` | **MISMATCH (M6)** |
| T28 | positive 3-seed temporal mean on all five California catalogs (abstract `:27-30`, `:292`, `:314-317`) | all five positive | all five > 0 (0.05248974448868626, 0.04579641250252209, 0.02843007317269186, 0.10168059157503206, 0.07842551662014952); check level PASS, `all_2_seed_sd` true | `replacement_readiness.json` → `checks[california_temporal_suite].evidence.dT` | MATCH |

**Wording note on T22.** "San Jacinto remains positive but its interval touches
zero" is generous: the interval is [−0.005686, 0.075926], i.e. it *crosses*
zero, and the stored `decision` is `"tie"`.

**Competing ComCat seed triples (provenance trap, not a mismatch).** Three
3-seed ComCat triples are committed and they disagree: `n1_density`/`n1_s1553`/
`n1_s1554` → mean tll 1.486832578976949 (Δ +0.0525), which is what the
manuscript reports; `final_s1553`/`final_s1554`/`final_s1555` → 1.4859981934229534
(Δ +0.0517); `comcat25`/`comcat25_s1554`/`comcat25_s1555` → 1.4840261538823445
(Δ +0.0497). The canonical set is documented (`SEED_DIRS` at
`scripts/aggregate_fullsuite.py:23`, and `README.md:182-189`), and the config
diff shows `n1_density` carries `spatial_density_feat` / `density_radius_km` /
`d_floor_km` — the density-adaptive N1 production model of §4.2 — which the
other two triples lack. `final_s1555` and `comcat25_s1555` report an identical
tll (1.485485315322876): duplicate runs of one config differing only in
`mix_hidden`.

**Bootstrap `n` differs from eval `n` on two catalogs** — SanJac_10 4399 vs
4400, SaltonSea_10 4103 vs 4104 — one event lost in the FlowQuake/ETAS time
merge.

**Stale block in `n1_density/eval_forward.json`.** Its `baselines.ETAS.tll` is
1.4343428344882627 (the in-window value) rather than the forward window's
1.0102738057926097 from `forward_etas/summary.json`; its `split` field says
`"test"`; its `sll`/`nll` are the production kernel head, not the §4.4 head.
Only `paired_vs_ETAS.temporal` in that file is safe to quote — and it is correct
(1.0677136 − 1.0102738 = 0.0574398).

### Out-of-time and fairness control

| # | claim (location) | stated | artifact value | artifact → key | status |
|---|---|---|---|---|---|
| T29 | forward 2020–2026 paired temporal gain, CI, win rate (`:347-350`) | +0.057, [+0.038, +0.082], 60.5%, 10,187 events | `mean` 0.0574, `ci` [0.0376, 0.0819], `win_rate` 0.6051, `n` 10187; corroborated by `eval_forward.json` `mean_gain` 0.05743982591990415, `win_rate` 0.6050849121429273 | `total_win.json` → `forward_2020_2026.dT.{mean,ci,win_rate}` | MATCH |
| T30 | ETAS refit through 2020 improves forward NLL by 0.016 nats (7.464→7.448); `dT` +0.005, `dS` +0.011 (`:363-369`) | as stated | frozen `nll` 7.464320811779553, refit 7.448446148714125 (diff 0.01587466306542762); `tll` 1.0154842540955407 vs 1.0102738057926097 (0.0052104483); `sll` −8.463930402809666 vs −8.474594617572162 (0.010664214762496016) | `forward_etas/summary.json` + `forward_etas_ComCat_25_refit2020/summary.json` → `nll`/`tll`/`sll` | MATCH |
| T31 | temporal component of the forward win vs the refit ETAS (`:369`) | +0.052 | 1.0677136182785034 − 1.0154842540955407 = 0.052229364182962756 | `eval_forward.json` `tll` minus refit `summary.json` `tll` | MATCH |

---

## Family 2 — Spatial and total likelihood (§4.4, §6)

| # | claim (location) | stated | artifact value | artifact → key | status |
|---|---|---|---|---|---|
| S1 | California total-likelihood win (`:575`, abstract `:51`) | +0.113 | 0.1133 | `stats_hardening.json` → `total_with_head_family.California.dTot_mean` | ROUNDING |
| S2 | Italy total-likelihood win | +0.210 | 0.2095 | `...Italy.dTot_mean` | ROUNDING |
| S3 | Japan total-likelihood win (`:575-576`, `:581`) | +0.039 | 0.039, `dTot_ci` [0.0163, 0.062], `p_holm` 0.0045, `p_boot` 0.0015 | `...Japan.dTot_mean`, `.dTot_ci` | MATCH |
| S4 | Chile total-likelihood win (`:576`) | +0.061 | 0.0608 | `...Chile.dTot_mean` | ROUNDING |
| S5 | Greece total-likelihood win, few-shot temporal (`:576`) | +0.076 | 0.0756, `temporal_variant` `"fewshot"` | `...Greece.dTot_mean` | ROUNDING |
| S6 | Iran total-likelihood win, few-shot temporal (`:576`) | +0.084 | 0.0844, `temporal_variant` `"fewshot"` | `...Iran.dTot_mean` | ROUNDING |
| S7 | Holm-adjusted p across the six-region total family (`:577`) | all ≤ 0.019 | 0.003, 0.003, 0.0045, 0.003, 0.011, 0.0185 (max = Iran 0.0185); all `significant_05_holm` true | `total_with_head_family.*.p_holm` | MATCH |
| S8 | seed basis of the six totals (front matter `:5-7` vs `:574-577`) | 3-seed | seed 0 only (`scripts/stats_hardening.py:167`) | `total_with_head_family.*.dTot_mean` | **MISMATCH (M2)** |
| S9 | ComCat composite total NLL vs ETAS (`:578`, `:857`) | 7.142 vs 7.255 | `fq_nll` 7.142121886887271, `etas_nll` 7.255427552750566 (diff 0.113306) | `total_win.json` → `test_2007_2020.fq_nll`, `.etas_nll` | ROUNDING |
| S10 | Italy composite total NLL vs ETAS (`:857`) | 7.387 vs 7.596 | derived −(1.322468721459759 + −8.709388732910156) = 7.386920; ETAS `nll` 7.596423745029326 | `multiregion_master.json` `Italy.native.paired.tll` + `neural_etas/Italy_25/summary_full_s0.json` `test_sll_neural`; `Italy.ETAS.nll` | ROUNDING |
| S11 | ComCat head spatial SLL vs ETAS and the three per-seed values (`:547-548`) | −8.630 vs −8.690 (−8.6298/−8.6299/−8.6291) | −8.6297607421875 / −8.62991714477539 / −8.629073143005371; 3-seed mean −8.629584; `test_sll_etas` −8.68977038723882 | `neural_etas/ComCat_25/summary_full_s{0,1,2}.json` → `test_sll_neural`, `test_sll_etas` | ROUNDING |
| S12 | ComCat paired per-event spatial gain (`:549`) | +0.060 | 0.06 (`summary_full_s0` `dS_mean` 0.06; 3-seed mean 0.060200) | `total_win.json` → `test_2007_2020.dS.mean` | MATCH |
| S13 | ComCat `dS` 95% block-bootstrap CI (`:549`) | [+0.051, +0.069] | [0.051, 0.0688] | `total_win.json` → `test_2007_2020.dS.ci` | ROUNDING |
| S14 | six-region spatial win: California (`:569`, abstract `:48`) | +0.060 (+0.06) | seeds 0.06 / 0.0599 / 0.0607 → 0.060200 | `neural_etas/ComCat_25/summary_full_s{0,1,2}.json` → `dS_mean` | ROUNDING |
| S15 | six-region spatial win: Italy | +0.137 (+0.14) | 0.1383 / 0.1373 / 0.1364 → 0.137333 | `neural_etas/Italy_25/...` | ROUNDING |
| S16 | six-region spatial win: Japan | +0.053 (+0.05) | 0.0525 / 0.0526 / 0.0528 → 0.052633 | `neural_etas/Japan_25/...` | ROUNDING |
| S17 | six-region spatial win: Chile | +0.029 (+0.03) | 0.0267 / 0.0337 / 0.0277 → 0.029367 | `neural_etas/Chile_25/...` | ROUNDING |
| S18 | six-region spatial win: Greece | +0.088 (+0.09) | 0.0857 / 0.087 / 0.0902 → 0.087633 | `neural_etas/Greece_25/...` | ROUNDING |
| S19 | six-region spatial win: Iran | +0.146 (+0.15) | 0.1465 / 0.1489 / 0.1433 → 0.146233 | `neural_etas/Iran_25/...` | ROUNDING |
| S20 | per-seed spread ≤0.003 (`:570-571`) | ≤0.003 | Chile 0.0070, Iran 0.0056, Greece 0.0045 exceed it | `summary_full_s{0,1,2}.json` → `dS_mean` | **MISMATCH (M1)** |
| S21 | ablation: causal multi-scale background alone gives `dS` +0.051 of the +0.060 (`:549-551`) | +0.051 | 0.0513, `ci` [0.0434, 0.0595], win. **Trap:** a second file `summary_bg_only.json` (same config `"bg_only"`, seed 0) gives 0.0497 | `neural_etas/ComCat_25/summary_bg_only_s0.json` → `dS_mean` | ROUNDING |
| S22 | ablation: classical flETAS-style SGD refit control (`:551-555`) | +0.056 | 0.0564, `ci` [0.0477, 0.0654], win | `neural_etas/ComCat_25/summary_refit_globals_s0.json` → `dS_mean` | ROUNDING |
| S23 | gate-closed head reproduces the published ETAS spatial LL (`:522-524`, abstract `:42`) | 2×10⁻⁹ nats | `max_abs_sll_err` 1.7655796824556091e-09; `mean_sll_ours` −8.689770387238818 vs `mean_sll_ref` −8.689770387238829; `match` true; `n_test` 21889 | `etas_sll_repro.json` → `max_abs_sll_err` | ROUNDING |
| S24 | pairing coverage for the four non-CA/Italy totals (`:584-585`) | Japan 96.3%, Chile 97.1%, Greece 92.0%, Iran 89.0% | 0.9626, 0.9709, 0.9195, 0.8904 | `total_with_head_family.{Japan,Chile,Greece,Iran}.pairing.coverage_vs_etas` | ROUNDING |
| S25 | out-of-time 2020–2026 total replication (`:580`, `:353`, abstract `:59`) | +0.124 [+0.104, +0.146], 10,187 events | `mean` 0.1241, `ci` [0.1035, 0.1455], `n` 10187, `p_boot` 0.0005 | `total_win.json` → `forward_2020_2026.dTot.{mean,ci}`, `.n` | ROUNDING |
| S26 | out-of-time spatial gain grows (`:573`, `:352`, abstract `:59`) | +0.067 [+0.055, +0.078] | `dS_mean` 0.0666, `dS_ci` [0.0552, 0.0784] (`total_win.json` forward `dS.ci` is [0.0553, 0.0784]) | `neural_etas/ComCat_25/per_event_forward_full.json` → `dS_mean`, `dS_ci` | ROUNDING |
| S27 | N1 density-adaptive aggregate improvement on ComCat, 3-seed (`:502`) | sll −9.091 → −9.059, nll 7.605 → 7.572 | baseline 3-seed mean sll −9.090745, nll 7.604747; N1 sll −9.058865 (sd 0.009025), nll 7.572033 (sd 0.009040) | `final_s155{3,4,5}/eval_test.json`; `fullsuite_summary.json` → `ComCat_25.sll`, `.nll` | ROUNDING |
| S28 | distance-band localization and the triggering-coverage diagnosis (`:500-508`) | <0.5 km deficit −0.218 → −0.062; 2–10 km band improves; 64% recur within 0.5 km; 85% of nearest priors outside the last-64 window | no distance strata anywhere | `n1_density/spatial_gap_decomp.json` (background/triggered split only) | **NO ARTIFACT (N6, N7, N8)** |
| S29 | zero-shot spatial-head transfer, 7 of 7 within-regime (`:603-604`) | 7 of 7 (win or tie); CA→Italy +0.073, Japan→Chile +0.095, Chile→Japan +0.016 | 7 `within_regime` rows: 6 `"win"` + 1 `"tie"` (Japan→Greece); ComCat→Italy 0.0726, Japan→Chile 0.0949, Chile→Japan 0.016; `cross_completeness` 0 of 4 | `neural_etas/spatial_transfer_summary.json` → `within_regime[].zero_shot_dS`, `.zero_shot` | ROUNDING |
| S30 | transfer ablation: background-only "still wins"; Japan→Greece modulation converts a non-win into a win (`:606-610`) | as stated | Japan→Greece `bg_only` −0.015 (a loss); full head 0.0282, `decision` `"tie"`. Modulation increments 0.0101–0.0432 do match "+0.01 to +0.04" | `spatial_transfer_summary.json` → `mlp_decomposition_japan_source.Greece`; `within_regime[Japan→Greece].zero_shot` | **MISMATCH (M3)** |
| S31 | default initialization starts +0.002 to +0.004 nats/event above ETAS (`:539`) | +0.002…+0.004 | not committed | — | **NO ARTIFACT (N9)** |
| S32 | optional full flETAS (EM, free background) spatial baseline (`:970-971`) | listed as not yet run | not run; nothing committed. The baseline that *is* committed is the SGD refit control, `dS_mean` 0.0564, `dS_ci` [0.0477, 0.0654], `decision` `"win"` (row S22). The manuscript's own open-items list already flags this, so no sentence overclaims it | `neural_etas/ComCat_25/summary_refit_globals_s0.json` (the committed substitute) | **NO ARTIFACT (N12)** |

**What the head learned** (illustrative, backed):
`neural_etas/ComCat_25/summary_full_s0.json` → `bg_weights[unif,kde...]`
[0.12905, 0.3984900116920471, 0.24100999534130096, 0.1752299964427948,
0.05621999874711037], `alpha_far` 0.9639248847961426, `mu_adj`
2.6997389793395996 — supports `MANUSCRIPT.md:561`'s "~85% of its mass" reading
of the smoothed-seismicity components.

**Stale duplicate artifacts (ambiguity trap).** `runs/neural_etas/ComCat_25/`
holds two generations of the same seed-0 runs: `summary_full.json` (config
`"full"`, seed 0) gives `dS` 0.0578 while `summary_full_s0.json` gives 0.06;
`summary_bg_only.json` gives 0.0497 while `summary_bg_only_s0.json` gives
0.0513. The manuscript demonstrably uses the `_s0/_s1/_s2` generation (its quoted
per-seed slls are exactly those three files), so S21's +0.051 resolves to
`bg_only_s0`. Deleting or renaming the two unsuffixed files removes the trap.

**Key collision in `stats_hardening.json`.** `dTot_mean` appears in two blocks
with opposite verdicts: `per_region.California.dTot_mean` = −0.3107 (`"loss"`,
the base kernel-mixture model) vs
`total_with_head_family.California.dTot_mean` = +0.1133 (`"win"`, with the
full-history head). A checker reading the wrong block concludes the headline is
inverted.

**Japan event count** is 14886 in `stats_hardening.json` and `completeness.json`
but 14888 in `multiregion_master.json` (`Japan.native.paired.n`). Does not change
the 96.3% coverage figure.

`total_win.json` `fq_sll` −8.629760984 for the test window is seed 0, not the
3-seed mean −8.629584; both round to the stated −8.630.

`MANUSCRIPT.md:874` (§6) cites the spatial transfer as "(§4.5)"; it is reported
in §4.4.

---

## Family 3 — CSEP consistency (§4.2)

Nine `csep_results.json` are committed: `csep_h2h_etas`, `csep_h2h_fq`,
`etas_csep_pod`, `final_s1555/csep`, `final_s1555/csep_smoke`,
`n1_density/csep`, `n1_density/csep_head`, `n1_density/csep_head_smoke`, plus
the loose `runs/csep_results_s1555.json`. The loose file is byte-identical to
`final_s1555/csep/csep_results.json` (verified with `cmp`) — a duplicate, not a
ninth run. The two `*_smoke` files are 2–3 day debug runs (`n_sims` 500 and 200)
and back no manuscript number.

**Provenance caveat — these artifacts predate a simulator fix.** Every file in
this family was produced through `flowquake.ntest.simulate_day_events`, which
carried the absolute event time in float32 (`t_last`). At test-era day numbers
that is a 21–42 second quantum; see `MOONSHOT.md` invariant 1o. The fix widens
it to float64, so **these artifacts are no longer bit-reproducible against
current code.**

The substance is unaffected, and that is measured rather than assumed: a paired
comparison on one checkpoint over 30 target-bearing windows at 400 sims, same
torch seed, float64 against a reverted float32 copy, moves the total expected
count by **−1.005%** and the mean spatial TV by **0.105**. The Monte-Carlo noise
floor — the same code with a different seed — is **−0.713%** and **0.853**. The
defect is therefore 1.4× the noise on count and 0.12× on shape, i.e. below the
scatter these pass rates already carry. A 95/100 does not become a different
number in any meaningful sense; it simply will not reproduce byte-for-byte.

| # | claim (location) | stated | artifact value | artifact → key | status |
|---|---|---|---|---|---|
| C1 | production N1 standalone N-test, 100 days × 10⁴ sims (`:384`) | 100 / 95 / 95% | `n_eval` 100, `n_pass` 95, `pass_rate` 0.95 | `n1_density/csep/csep_results.json` → `summary.N` | MATCH |
| C2 | production N1 standalone S-test (`:385`) | 92 / 85 / 92% | stored `n_eval` 92, `n_pass` 85, 0.9239130434782609; recomputed with the committed criterion: 91 / 85 / 0.9340659340659341 (day 2982 is the `[-1.0,-1.0]` sentinel) | `summary.S` vs recomputed from `results[].S.quantile` | **AMBIGUOUS (A2)** |
| C3 | production N1 standalone M-test (`:386`) | 92 / 90 / 98% | `n_eval` 92, `n_pass` 90, 0.9782608695652174 | `summary.M` | ROUNDING |
| C4 | density-adaptive kernel improves the S-test, 88% → 92% (`:391`) | 88% → 92% | base 81/92 = 0.8804347826086957; density 85/92 = 0.9239130434782609. Both carry the same +1 denominator inflation: recomputed 81/91 = 0.8901 and 85/91 = 0.9341 | `final_s1555/csep` and `n1_density/csep` → `summary.S.pass_rate` | ROUNDING (see A2) |
| C5 | FlowQuake N1 N-test at matched 10³ (`:417`) | 95/100 (95%) | `n_eval` 100, `n_pass` 95, 0.95 | `csep_h2h_fq/csep_results.json` → `summary.N` | MATCH |
| C6 | FlowQuake N1 S-test at matched 10³ (`:418`) | 82/85 (96%) | 85 / 82 / 0.9647058823529412 | `csep_h2h_fq` → `summary.S` | ROUNDING |
| C7 | FlowQuake N1 M-test at matched 10³ (`:419`) | 89/92 (97%) | 92 / 89 / 0.967391304347826 | `csep_h2h_fq` → `summary.M` | ROUNDING |
| C8 | ETAS N-test through the same harness (`:417`) | 97/100 (97%) | 100 / 97 / 0.97 | `csep_h2h_etas` → `summary.N` | MATCH |
| C9 | ETAS S-test through the same harness (`:418`) | 80/86 (93%) | 86 / 80 / 0.9302325581395349 | `csep_h2h_etas` → `summary.S` | ROUNDING |
| C10 | ETAS M-test through the same harness (`:419`) | 87/92 (95%) | 92 / 87 / 0.9456521739130435 (94.6%) | `csep_h2h_etas` → `summary.M` | ROUNDING |
| C11 | a first ETAS run under-predicted counts, N 73/100, before the source-set fix (`:433`) | 73/100 | 100 / 73 / 0.73; this file also records `n_sims` 10000 vs 1000 in the fixed run | `etas_csep_pod` → `summary.N` | MATCH |
| C12 | both models simulate 10³ one-day catalogs per forecast day (`:412-413`, `:438-439`) | 10³ both | `n_sims` 1000 in both. FlowQuake per-day corroboration present (`n_nonempty` 320–1000, median 927; `sim_mean` finite). ETAS per-day records are placeholders: `n_nonempty` 1 and `sim_mean` NaN on all 100 days, hard-coded by `flowquake/csep_forecast.py:159` in `--rerun` mode; the only in-file support is the free-text `n_sims_note` | `csep_h2h_fq`, `csep_h2h_etas` → `n_sims`, `results[].n_nonempty`, `.sim_mean` | MATCH (weakly corroborated) |
| C13 | both models and the head are scored on the identical 100 forecast days (`:412-413`, `:453`) | identical 100 days | `n_days` 100 in all four; the `results[].day` lists are element-for-element identical across all four files (day indices 0…4763) | `csep_h2h_fq`, `csep_h2h_etas`, `n1_density/csep_head`, `n1_density/csep` → `results[].day` | MATCH |
| C14 | full-history head N-test at matched 10³ (`:455`) | 95/100 | 100 / 95 / 0.95 (`n_sims` 1000, head `ComCat_25`) | `n1_density/csep_head` → `summary.N` | MATCH |
| C15 | full-history head S-test (`:455`) | 79/85 (92.9%) | 85 / 79 / 0.9294117647058824 | `n1_density/csep_head` → `summary.S` | MATCH |
| C16 | full-history head M-test (`:455`) | 90/92 | 92 / 90 / 0.9782608695652174 | `n1_density/csep_head` → `summary.M` | MATCH |
| C17 | paired S-test, head vs ETAS: they agree on 77/83 evaluable days each (`:457-458`) | 77/83 each | recomputed pairing: 83 commonly evaluable days, head passes 77, ETAS passes 77. Stored as `shared_days` 83, `head_S_pass` 77, `etas_S_pass` 77 | `n1_density/csep_head` + `csep_h2h_etas` `results[].S.quantile`; `replacement_readiness.json` → `checks[full_history_head_csep].evidence` | MATCH (wording note below) |
| C18 | only 10 discordant days, split 5–5 (`:458`) | 10, 5–5 | recomputed: 10 discordant, 5 head-only and 5 ETAS-only (73 concordant of 83) | recomputed from `results[].S.quantile`; not stored as a key | MATCH |
| C19 | head vs ETAS paired S-test McNemar exact p = 1.00 (`:458-459`, `:592`, `:968`) | 1.00 | recomputed two-sided exact binomial on b01=5, b10=5: p = 1.0000. `replacement_readiness.json` states "McNemar p~1.0" in prose but stores no p-value key | recomputed; no p-value key committed | MATCH |
| C20 | head vs production kernel-mixture head paired S-test (`:459-460`) | 75/81 vs 78/81, p = 0.51 | recomputed: 81 commonly evaluable days, head 75, production 78, 9 discordant (3 head-only, 6 production-only), McNemar exact p = 0.5078125 | `n1_density/csep_head` + `csep_h2h_fq` `results[].S.quantile` | ROUNDING |
| C21 | the full-history gridded simulator reproduces the head's per-event SLL to 9.5×10⁻⁷ nats (`:448-449`, `:965`) | 9.5×10⁻⁷ | nothing in 1e-8…1e-5 in any committed JSON | — | **NO ARTIFACT (N5)** |

**Wording note on C17.** "they agree on 77/83 evaluable days each" reads as an
agreement count but is each model's *pass* count out of 83 shared days; true
concordance is 73/83 (73 + 10 discordant = 83). The numbers are right; the
sentence invites misreading.

**The N 73/100 → 97/100 attribution cannot be isolated.** `MANUSCRIPT.md:433-436`
attributes the earlier under-prediction solely to the fitted inversion's source
set not being re-conditioned on post-`test_start` mainshocks. But `etas_csep_pod`
records `n_sims` 10000 while `csep_h2h_etas` records 1000, and `n_sims` is
passed straight through as pyCSEP's `n_cat` (`flowquake/csep_forecast.py:180`).
If the pod run scored ~1000 real catalogs with `n_cat` 10000, pyCSEP pads with
~9000 empty catalogs, which alone produces the observed N under-prediction — and
the pod file's M rate is also depressed (73/92), consistent with padding rather
than a source-set bug. The artifacts support the two endpoints, not the single
stated cause. The author's own `note` field in `csep_h2h_etas` acknowledges the
10000-vs-1000 `--rerun` default confusion.

---

## Family 4 — Transfer and memorization (§4.1 forward, §4.3, §4.4 transfer, §4.5)

| # | claim (location) | stated | artifact value | artifact → key | status |
|---|---|---|---|---|---|
| X1 | forward window contains 10,187 new events (`:339`) | 10,187 | 10187 (also `forward_etas/summary.json` `n`, `eval_forward.json` `n_events`, `per_event_forward_full.json` `n`) | `total_win.json` → `forward_2020_2026.n` | MATCH |
| X2 | frozen out-of-time `dT`, CI, fraction improved (`:348-350`) | +0.057, [+0.038, +0.082], 60.5% | 0.0574, [0.0376, 0.0819], 0.6051 | `forward_2020_2026.dT.{mean,ci,win_rate}` | ROUNDING |
| X3 | frozen §4.4 head forward `dS` and event-level win rate (`:352-354`) | +0.067 [+0.055, +0.078], 47.8% | 0.0666, [0.0553, 0.0784], 0.4785 (`per_event_forward_full.json`: 0.0666, [0.0552, 0.0784], 0.4785) | `forward_2020_2026.dS.{mean,ci,win_rate}` | ROUNDING |
| X4 | forward total gain (`:353`, `:580`) | +0.124 [+0.104, +0.146] | 0.1241, [0.1035, 0.1455] | `forward_2020_2026.dTot.{mean,ci}` | ROUNDING |
| X5 | in-window ComCat total the forward window replicates (`:578-579`) | 7.142 vs 7.255, +0.113 | `fq_nll` 7.142121886887271, `etas_nll` 7.255427552750566, `dTot.mean` 0.1133, `dTot.ci` [0.1006, 0.1268] | `test_2007_2020.*` | ROUNDING (same claim as S9) |
| X6 | refit improves ETAS's own forward NLL marginally (`:364-365`) | 0.016 (7.464→7.448); `dT` +0.005, `dS` +0.011 | 0.015875; 0.005210; 0.010664 | `forward_etas_ComCat_25_refit2020/summary.json` + `forward_etas/summary.json` → `nll`/`tll`/`sll` | MATCH |
| X7 | vs the fairness-control refit ETAS the total win narrows slightly (`:366-367`) | +0.108 (temporal +0.052, spatial +0.056) | 7.340256443481511 − 7.448446148714125 = 0.108190; `tll` 1.0677136320078393 − 1.0154842540955407 = 0.052229; `sll` −8.40797007548935 − (−8.463930402809666) = 0.055960 | `total_win.json` forward `fq_nll`/`fq_tll`/`fq_sll` minus refit `nll`/`tll`/`sll`. **No artifact stores +0.108 or a CI for it** | MATCH (derived) |
| X8 | refit ETAS parameter values, branching ratio, EM iterations (`:362-364`) | `a` 1.556→1.603, log10μ −6.333→−6.389, ρ 0.557→0.571, branching 0.968, 12 iterations | not committed | — | **NO ARTIFACT (N2, N3)** |
| X9 | forward ETAS scorer agrees with the package on the temporal term (`:344-347`) | ~10⁻⁵/event; anchor effect 1.5×10⁻⁴ nats | not committed | — | **NO ARTIFACT (N4)** |
| X10 | ETAS spatial per-event scores reproduced to 2e-9 nats (`:344-345`, `:522-523`) | 2×10⁻⁹ | 1.7655796824556091e-09, `match` true | `etas_sll_repro.json` → `max_abs_sll_err` | MATCH |
| X11 | skill-over-time stability (`:325-327`) | 85% of windows positive; [+0.040, +0.068] | `bins_dT_positive_frac` 0.8518518518518519 (23/27), `dT_ci` [0.04044003357906552, 0.06782420487674817] | `prospective.json` → `California.native` | MATCH |
| X12 | California and Chile wins significant in every era (`:328-330`) | every era | Chile 10/19 positive, no per-era statistic stored | `prospective.json` | **MISMATCH (M6)** |
| X13 | Greece from-scratch native temporal loss (`:658`, `:741`) | −0.107 | −0.10691914737781416 (identical at `prospective.json` `Greece.native.dT`) | `multiregion_master.json` → `Greece.native.paired.dT` | MATCH |
| X14 | 95% CI on Greece's native temporal loss (`:658`) | [−0.144, −0.070] | [−0.16308742268956303, −0.04480017107870762] | `Greece.native.paired.dT_ci` | **MISMATCH (M4)** |
| X15 | Iran from-scratch native temporal loss (`:659`, `:742`) | −0.276 | −0.2759649781920259 | `Iran.native.paired.dT` | MATCH |
| X16 | 95% CI on Iran's native temporal loss (`:659`) | [−0.347, −0.205] | [−0.36975851467862897, −0.173887497485534] | `Iran.native.paired.dT_ci` | **MISMATCH (M5)** |
| X17 | LOO few-shot lifts Greece to TOST equivalence (`:741`) | −0.012; equivalent, 90% CI [−0.056, +0.035] | `dT_mean` −0.0125, `dT_tost_0.1.ci90` [−0.05598692646015908, 0.035200303940858824], `equivalent` true, `temporal_variant` `"fewshot"` (point estimate corroborated at `multiregion_master.json` `Greece.fewshot.paired.dT` −0.012454249352756935) | `stats_hardening.json` → `per_region.Greece` | MATCH |
| X18 | LOO few-shot narrows Iran ~4× without equivalence (`:742`) | −0.063; not equivalent, 90% CI [−0.122, −0.001] | `dT_mean` −0.0634, `ci90` [−0.12210308721050286, −0.0013525685331943467], `equivalent` false (corroborated: `Iran.fewshot.paired.dT` −0.06342791631011692) | `stats_hardening.json` → `per_region.Iran` | MATCH |
| X19 | pooled zero-shot `dT` on the held-out data-poor regions (`:741-742`) | Greece −0.040, Iran −0.105 | −0.03951952813479355, −0.10490337351091977 (source tlls match `transfer_Greece.json` −1.0539535284042358 and `transfer_Iran.json` −1.2875525951385498) | `multiregion_master.json` → `{Greece,Iran}.zeroshot.paired.dT` | MATCH |
| X20 | Japan LOO row (`:739`) | native −0.015, zero-shot −0.021, few-shot −0.022 | −0.015218170444920626, −0.020583855702877164, −0.021952589724434323 | `Japan.{native,zeroshot,fewshot}.paired.dT` | MATCH |
| X21 | Chile LOO row (`:740`) | native +0.034, zero-shot −0.027, few-shot +0.042 | 0.03425662245105383, −0.02706920651977874, 0.04178263792370491 | `Chile.{native,zeroshot,fewshot}.paired.dT` | MATCH |
| X22 | spatial head transfers zero-shot 7/7 within regime (`:602-604`) | 7 of 7 win-or-tie; CA→Italy +0.073, Japan→Chile +0.095, Chile→Japan +0.016 | 7 rows: win, win, tie, win, win, win, win (6 + 1 = 7/7); 0.0726, 0.0949, 0.016 | `spatial_transfer_summary.json` → `within_regime[].zero_shot`, `.zero_shot_dS`; detail in `transfer_from_{ComCat_25,Japan_25,Chile_25}.json` | MATCH |
| X23 | light few-shot recalibration of the four scalars turns all seven into clear wins (`:604-606`) | 7/7 wins | `few_shot` = `"win"` for all 7 (`few_shot_dS` 0.1063, 0.1131, 0.1044, 0.1353, 0.0196, 0.037, 0.0457) | `within_regime[].few_shot`, `.few_shot_dS` | MATCH |
| X24 | transfer fails across completeness regimes (`:610-611`) | 0 of 4 | 4 rows, `zero_shot` = `"loss"` in all four (−0.0672, −0.2292, −0.1464, −0.328) | `cross_completeness[].zero_shot`, `.zero_shot_dS` | MATCH |
| X25 | neural modulation adds this much on a background-only transferred head (`:607-609`) | +0.01 to +0.04 | `mlp_adds` Chile 0.0308, Greece 0.0432, Iran 0.0101 — actual range 0.0101–0.0432, upper end slightly above the stated +0.04 | `mlp_decomposition_japan_source` | ROUNDING |
| X26 | for Japan→Greece the modulation converts a non-win into a win (`:609-610`) | non-win → win | `dS` 0.0282, `ci` [−0.0121, 0.0676], `decision` `"tie"`; `bg_only` −0.015 | `transfer_from_Japan_25.json` → `targets.Greece_25.zero_shot`; `spatial_transfer_summary.json` | **MISMATCH (M3)** |
| X27 | memorization ablation train/test NLL and gap at `ckpt_last` for h = 0, 4, 16, 64 (`:474-479`) | h0 7.28/7.62/0.34; h4 4.14/19.65/15.50; h16 4.18/18.73/14.55; h64 4.27/18.33/14.06 | h0 7.281167030334473 / 7.621030569076538 / 0.33986353874206543; h4 4.143446922302246 / 19.64580488204956 / 15.502357959747314; h16 4.182443857192993 / 18.731383323669434 / 14.54893946647644; h64 4.2729010581970215 / 18.33090305328369 / 14.05800199508667 | `ablation_h/memorization_figure.json` → rows with `ckpt == "last"` | MATCH |
| X28 | early stopping does not rescue h>0 (`:486-489`) | best at step 250, gap 0.21–0.27, NLL 8.0–8.2 (worse than h=0's 7.62); held-out then diverges to ~19–20 | `best` rows for h=4/16/64 all at step 250; `gap_nll` 0.20925915241241455 / 0.2444249391555786 / 0.27436113357543945; `test.nll` 8.21579110622406 / 8.124865412712097 / 8.063421964645386 (span 8.06–8.22, upper end above the stated 8.2); `last` `test.nll` 19.646 / 18.731 / 18.331 (span 18.3–19.6, not 19–20) | `memorization_figure.json` → rows with `ckpt == "best"` | ROUNDING |
| X29 | per-region head spatial gains, 3-seed means (`:569-570`) | +0.060, +0.137, +0.053, +0.029, +0.088, +0.146 | recomputed 0.060200, 0.137333, 0.052633, 0.029367, 0.087633, 0.146233; every seed's `dS_ci` strictly positive | `neural_etas/<region>/summary_full_s{0,1,2}.json` → `dS_mean` | MATCH |
| X30 | per-seed spread of those six (`:570-571`) | ≤0.003 | ranges 0.0008, 0.0019, 0.0003, 0.0070, 0.0056, 0.0045 | as above | **MISMATCH (M1)** |
| X31 | §4.3 ablation is "reproducible from the committed checkpoints" | committed checkpoints | **no `.pt` files are tracked** (`git ls-files \| grep -c '\.pt$'` = 0; `*.pt` is gitignored). The numbers are committed, the checkpoints are not | — | MISMATCH-adjacent wording; see the laptop list in `WORKING.md` |

**Ambiguity traps in this family (manuscript disambiguates, but barely).**

- `runs/ablation_h/ablation_h.json` is the file a reader would open for the §4.3
  table, and it reports the **best**-checkpoint numbers (h=4: 8.007/8.216/gap
  0.209) — wholly different from the manuscript's table. The manuscript is
  correct because it says "at the converged checkpoint (`ckpt_last`)" and cites
  `memorization_figure.json`, but the two sibling artifacts invite a misread.
- Japan's `n` differs across artifacts: 14888 (`multiregion_master`,
  `prospective`, `replacement_readiness`) vs 14886 (`stats_hardening`,
  `transfer_japan`). Japan's native `dT` is quoted as −0.014 in the density table
  (`stats_hardening`, n=14886) and −0.015 in the LOO table
  (`multiregion_master`, n=14888) — both trace, to different pairings of the same
  comparison.
- `runs/{greece,iran,japan,chile}_*/eval_test.json` all carry `baselines.ETAS` =
  the **California** ComCat inversion (`tll` 1.4343428344882627, `nll`
  7.2554275527505645), not the region's own ETAS, because `--etas_dir` defaults
  to `output_data_ComCat_25`. Nothing in the manuscript uses those keys, but
  anyone spot-checking a foreign-region claim from its `eval_test.json` will
  compute a badly wrong `dT`.
- The ComCat "3 seeds" `dS` of +0.060 with CI [+0.051, +0.069] pairs a 3-seed
  mean (0.060200) with **seed 0's** bootstrap CI ([0.0509, 0.0692]); no artifact
  contains a pooled-across-seeds CI.

---

## Family 5 — Provenance, baselines and reproducibility

Inventory: `git ls-files runs/` = 226 files (136 JSON + 90 YAML). No CSVs.
Working tree clean, nothing ignored under `runs/`, `reference/` absent.

| # | claim (location) | stated | artifact value | artifact → key | status |
|---|---|---|---|---|---|
| P1 | each cross-regime region has a region-fitted ETAS baseline from "the `etas` inversion" (front matter `:10-11`) | unpinned | no `etas` package name, version, URL or commit in any of the 136 committed run JSONs | — | **NO ARTIFACT (N1)** |
| P2 | "a region-fitted ETAS baseline is inverted on each (`reference/Experiments/ETAS`, the `etas` package)" (`:262`) | implementation unspecified | `scripts/run_etas_regions.py` shells out to `reference/Experiments/ETAS/invert_etas.py` and logs to `runs/etas_invert_<Cfg>.log`, but `*.log` is gitignored and no such log exists on disk (`find . -name "*.log"` is empty) | — | **NO ARTIFACT (N1)** |
| P3 | "the benchmark's fitted ETAS model (Mizrahi et al. `etas`)" was run through the same pyCSEP path (`:403`) | upstream implied | `flowquake/etas_csep.py:70-71` imports the installed package at runtime (`from etas.inversion import ETASParameterCalculation`, `from etas.simulation import ETASSimulation`). The artifact records `ckpt` `"runs/n1_density/ckpt_best.pt"` (the FlowQuake checkpoint — an artifact of the shared `--rerun` scorer) and `n_sims` 1000; no etas package name, version or commit. `replacement_readiness.json`'s `etas_csep_head_to_head` check is PASS with `"evidence": {}` — empty | `csep_h2h_etas/csep_results.json` → `ckpt`, `n_sims`, `n_sims_note` | **NO ARTIFACT (N1)** |
| P4 | California ETAS is the benchmark's published inversion, params fit through 2007, not re-inverted locally (`:342`) | published inversion | `"ComCat_25 inversion (train<=2007, published with benchmark)"` | `forward_etas/summary.json` → `params_frozen_from` | MATCH |
| P5 | refit2020 parameter shift — the only fork-fingerprinting numbers in the manuscript (`:363`) | `a` 1.556→1.603, log10μ −6.333→−6.389, ρ 0.557→0.571, branching 0.968 | file holds only `window`, `n`, `tll`, `sll`, `nll`, `etas_name`, `fit_window`, `params_frozen_from`. No parameter vector committed anywhere; grepping `runs/` for 1.603 / −6.389 / 0.571 / 0.968 returns only unrelated CSEP quantiles | `forward_etas_ComCat_25_refit2020/summary.json` | **NO ARTIFACT (N2)** |
| P6 | refit2020 EM "converged in 12 iterations" (`:362-363`) | 12 | no iteration key | same file | **NO ARTIFACT (N3)** |
| P7 | refit improves ETAS forward NLL by 0.016 nats (`:363-364`) | 7.464 → 7.448 | 7.464320811779553 and 7.448446148714125; difference 0.015874663065428 | `forward_etas/summary.json` + refit `summary.json` → `nll` | ROUNDING |
| P8 | the repo's ETAS spatial reimplementation reproduces the package per-event SLL to 2e-9 from published inverted parameters (`:344-345`, `:523`) | 2×10⁻⁹ | `max_abs_sll_err` 1.7655796824556091e-09; `mean_sll_ours` −8.689770387238818; `mean_sll_ref` −8.689770387238829; `n_test` 21889; `match` true | `etas_sll_repro.json` | ROUNDING |
| P9 | ETAS temporal term matches the package to ~1e-5/event; anchor effect 1.5e-4 (`:345-347`) | as stated | spatial term only in the repro artifact | — | **NO ARTIFACT (N4)** |
| P10 | ETAS `ll_scores` reproduced exactly from per-event output (`:222-224`) | reproduced exactly | `match` true, `max_abs_sll_err` 1.77e-9 (spatial only) | `etas_sll_repro.json` → `match` | MATCH |
| P11 | Poisson baseline tll on ComCat_25 (`README.md:51`) | 0.5126 | **0.5126406686259881** | `n1_density/eval_test.json` → `baselines.Poisson.tll` | ROUNDING |
| P12 | Poisson baseline sll (`README.md:51`) | −13.7745 | **−13.774504128914366** | `baselines.Poisson.sll` | ROUNDING |
| P13 | Poisson baseline nll (`README.md:51`) | 13.2619 | **13.261863460288378** | `baselines.Poisson.nll` | ROUNDING |
| P14 | ETAS targets to beat on ComCat_25 (`README.md:50`) | 1.4343 \| −8.6898 \| 7.2554 | 1.4343428344882627 \| −8.689770387238827 \| 7.2554275527505645 | `baselines.ETAS.{tll,sll,nll}` | ROUNDING |
| P15 | both baseline rows are cited to `reference/.../output_data_ComCat_25/ll_scores.json` (`README.md:46`) | that file | the whole `ll_scores.json` is copied verbatim into every eval JSON — `flowquake/evaluate.py:104-110` does `res["baselines"] = json.load(...)` — so the cited file's full content **is** in the repo, embedded, even though `reference/` is absent | `n1_density/eval_test.json` → `baselines` | MATCH |
| P16 | ComCat_25 3-seed tll (`:298`) | 1.4868 ± 0.0008 | 1.486832578976949 ± 0.0008107134678745433; recomputed from the three per-seed files to all 16 digits | `fullsuite_summary.json` → `ComCat_25.tll`, `.tll_sd` | ROUNDING |
| P17 | WHITE_06 3-seed tll (`:299`) | 2.0669 ± 0.0007 | 2.0668934186299643 ± 0.0006617227810225719; recomputed identical to 16 digits | `WHITE_06.tll`, `.tll_sd` | ROUNDING |
| P18 | SanJac_10 3-seed tll (`:300`) | 1.1610 ± 0.0009 | 1.1609567801157634 ± 0.0008829548198669929; recomputed identical | `SanJac_10.tll`, `.tll_sd` | ROUNDING |
| P19 | SaltonSea_10 3-seed tll (`:301`) | 2.4337 ± 0.0070 | 2.433719793955485 ± 0.00704049643577668; recomputed identical | `SaltonSea_10.tll`, `.tll_sd` | ROUNDING |
| P20 | SCEDC_20 3-seed tll (`:302`) | 2.6194 ± 0.0031 | 2.619408051172892 ± 0.003079546775498436; recomputed identical | `SCEDC_20.tll`, `.tll_sd` | ROUNDING |
| P21 | seed-config naming (`README.md:182-189`) | `n1_density` / `n1_s1553` / `n1_s1554` for ComCat_25; `final_s{1553,1554,1555}` are the earlier canonical non-density seeds | `SEED_DIRS["ComCat_25"]` = `[n1_density, n1_s1553, n1_s1554]` at `scripts/aggregate_fullsuite.py:23`; those three configs' `out_dir`s are exactly `runs/n1_density` (seed 1555), `runs/n1_s1553` (1553), `runs/n1_s1554` (1554). All 15 `SEED_DIRS` directories exist with `eval_test.json`. `runs/final_s{1553,1554,1555}/` exist separately | `SEED_DIRS`; `fullsuite_summary.json` → `<ds>.seed_dirs` | MATCH |
| P22 | `results/CLAIMS.md` traces each claim; values `PENDING` until artifacts are committed (`README.md:212-215`) | PENDING | was true; **superseded by this revision** — every value above is now read from a committed artifact or explicitly marked NO ARTIFACT. `README.md:212-215` needs the `PENDING` sentence removed | this file | MATCH (now stale in README) |
| P23 | "Figures are gitignored" (former open item in this file) | gitignored | 12 figures tracked | `git ls-files figures/` | **MISMATCH (M7)** |
| P24 | the expected `reference/` tree lists everything a stranger must supply (`README.md:132-148`) | that tree | incomplete — see M8 | `runs/*/config.yaml` → `data.catalog_path` | **MISMATCH (M8)** |
| P25 | regenerate the ETAS baselines by driving the benchmark's `invert_etas.py`/`predict_etas.py`; "needs `reference/Experiments/ETAS/config/ComCat_25.json`" (`README.md:126-129`, `REPRODUCE.md:22-28`) | only the benchmark's own config is needed | no script writes `reference/Experiments/ETAS/config/*.json`; the six non-benchmark configs are author-authored and committed nowhere | — | **NO ARTIFACT (N10)** |
| P26 | pytest: 16 pass, data-alignment tests skip until `reference/` exists (`README.md:88`) | 16 pass | 22 test functions total (`test_data.py` 6, `test_heads.py` 4, `test_flow.py` 4, `test_ssm.py` 5, `test_stats.py` 3); 22 − 6 data-alignment = 16. Consistent; not executed here | `tests/*.py` | MATCH |
| P27 | overall readiness verdict, per-check levels, strongest defensible claim | — | `overall` `"RESEARCH_PREVIEW_READY"`; 15 checks, 11 PASS / 4 WARN (`california_spatial_total_gap`, `california_block_bootstrap_temporal`, `pooled_global_temporal`, `legacy_package_surface`); `strongest_defensible_claim` present and consistent with `REPLACEMENT_READINESS.md:66-70` | `replacement_readiness.json` → `overall`, `checks[].level`, `strongest_defensible_claim` | MATCH |

### ETAS provenance — what would settle N1

The fork only affects ETAS **inversions**, not scoring: `flowquake/evaluate.py`
reads the baseline off disk (`ll_scores.json` / `augmented_catalog.csv`), and
`scripts/etas_sll_repro.py` and `scripts/etas_forward_eval.py` are independent
reimplementations. Nothing in `flowquake/` or `scripts/` imports the `etas`
package except `flowquake/etas_csep.py`. So the fork matters for exactly two
things: the five §4.5 region inversions plus `ComCat_25_refit2020`, and the §4.2
ETAS CSEP column.

`git show 95e6b92^:requirements.txt` shows the `ss15859` line was **commented
out**, never an active install line — weaker evidence than it looks. No `.venv`,
lockfile or pip-freeze capture exists anywhere in the repo.

Run these in the training environment, not here:

1. `pip show -f etas` → Location, Version, whether the dist-info exists.
2. `python -c "import etas,inspect;print(etas.__file__);print(getattr(etas,'__version__','none'))"`.
3. `cat .../etas-*.dist-info/direct_url.json` — for a VCS install this records
   the exact git URL and resolved commit. **This is the single decisive file.**
4. If installed editable or from a clone:
   `git -C <clone> remote -v && git -C <clone> rev-parse HEAD`.
5. API fingerprint if the install is gone:
   `inspect.signature(etas.simulation.ETASSimulation.simulate)`.
   `flowquake/etas_csep.py` calls `simulate(forecast_n_days=1, n_simulations=,
   m_threshold=, info_cols=[], chunksize=)` as a lazy generator and calls
   `reload.prepare_source_events()`; whichever fork supports that streaming
   `chunksize` kwarg and that method is the one that ran.
6. Shell history for `pip install ... etas`, and the `runs/etas_invert_*.log`
   files if they survive in the original run directory — they contain the import
   path.
7. `reference/Experiments/ETAS/`'s own requirements: `invert_etas.py`'s imports
   may themselves force one fork.

Until then `pyproject.toml:38`'s
`etas = ["etas @ git+https://github.com/lmizrahi/etas.git", ...]` is an
unverified guess. The pin should carry a commit SHA once known.

---

## Open items for this file

- N1 (the `etas` fork) is the highest-priority unresolved row and the only one
  that touches what every reported gain is measured *against*.
- N5's 9.5e-7 simulator validation is printed rather than written. Either
  capture it into `csep_head/csep_results.json` or cite the console log
  explicitly in the manuscript.
- N6–N8: the §4.4 distance-band and triggering-coverage numbers need
  `scripts/trigger_coverage.py` and `scripts/spatial_gap_decomp.py` to write
  their strata to JSON, and the stated bands need to match a band some script
  actually computes.
- A2: the two 10⁴ CSEP runs' stored `summary` blocks predate the current
  `csep_summary()` sentinel handling. Re-aggregating them from their own
  `results[]` changes the S denominator from 92 to 91.
- Several run directories are named by CLI override rather than by a committed
  config (region seeds, `--no-mlp` / `--refit-globals` ablations,
  `de_<region>_*`). Those rows name the invocation instead; if a reviewer needs
  to reproduce them exactly, the invocation should be recorded alongside the
  artifact.
- Per-event score CSVs are excluded by `.gitignore`, so every block-bootstrap
  CI and everything in `runs/stats_hardening.json` is a summary that cannot be
  independently re-derived from this repository. Only one per-event file is
  tracked repo-wide (`runs/neural_etas/ComCat_25/per_event_forward_full.json`).
