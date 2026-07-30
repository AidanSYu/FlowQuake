# Claim → evidence map

One row per reported claim, tracing it to the config that produced the run, the
run directory the artifact lives in, the script that writes that artifact, and
the exact JSON key to read.

**Every value below is `PENDING`.** Fill each one by reading the committed
artifact, not by copying from `MANUSCRIPT.md` — the point of this file is to be
an independent check on the manuscript, so a mismatch is a finding, not a typo.
Anything that cannot be filled from a committed artifact should be marked
`NO ARTIFACT` rather than left blank.

`.gitignore` admits the summary JSONs named here. Checkpoints (`*.pt`),
precomputed features (`*_trigfeat.npz`), simulated CSEP catalogs
(`CSEP_day_*.csv`) and per-event score CSVs stay out; force-add an individual
one with `git add -f <path>` if a claim needs the raw pairing.

Config paths are relative to `configs/`, run paths to `runs/`, script paths to
`scripts/` unless a `flowquake.` module is named.

## §4.1 Temporal win across the California suite

Seed directories are `SEED_DIRS` in `scripts/aggregate_fullsuite.py`; each holds
one `eval_test.json` with `tll`/`sll`/`nll` and `baselines.ETAS.{tll,sll,nll}`.
The 3-seed means are aggregated into `runs/fullsuite_summary.json`.

| dataset | configs (3 seeds) | run dirs | script | JSON key | value |
|---|---|---|---|---|---|
| ComCat_25 | `n1_density.yaml`, `n1_s1553.yaml`, `n1_s1554.yaml` | `n1_density`, `n1_s1553`, `n1_s1554` | `aggregate_fullsuite.py` | `fullsuite_summary.json` → `ComCat_25.tll`, `.tll_sd`, `.etas_tll` | PENDING |
| WHITE_06 | `WHITE_06_n1.yaml`, `WHITE_06_n1_s1554.yaml`, `WHITE_06_n1_s1555.yaml` | `WHITE_06_n1`, `WHITE_06_n1_s1554`, `WHITE_06_n1_s1555` | `aggregate_fullsuite.py` | `WHITE_06.tll`, `.tll_sd`, `.etas_tll` | PENDING |
| SanJac_10 | `SanJac_10_n1.yaml`, `SanJac_10_n1_s1554.yaml`, `SanJac_10_n1_s1555.yaml` | `SanJac_10_n1`, `SanJac_10_n1_s1554`, `SanJac_10_n1_s1555` | `aggregate_fullsuite.py` | `SanJac_10.tll`, `.tll_sd`, `.etas_tll` | PENDING |
| SaltonSea_10 | `SaltonSea_10_n1.yaml`, `SaltonSea_10_n1_s1554.yaml`, `SaltonSea_10_n1_s1555.yaml` | `SaltonSea_10_n1`, `SaltonSea_10_n1_s1554`, `SaltonSea_10_n1_s1555` | `aggregate_fullsuite.py` | `SaltonSea_10.tll`, `.tll_sd`, `.etas_tll` | PENDING |
| SCEDC_20 | `SCEDC_20_n1.yaml`, `SCEDC_20_n1_s1554.yaml`, `SCEDC_20_n1_s1555.yaml` | `SCEDC_20_n1`, `SCEDC_20_n1_s1554`, `SCEDC_20_n1_s1555` | `aggregate_fullsuite.py` | `SCEDC_20.tll`, `.tll_sd`, `.etas_tll` | PENDING |

| claim | run dir | script | JSON key | value |
|---|---|---|---|---|
| ComCat paired per-event temporal gain, block-bootstrap CI | `n1_density` | `stats_hardening.py` | `stats_hardening.json` → `per_region.California.dT_mean`, `.dT_ci`, `.dT_p_boot` | PENDING |
| Skill-over-time: fraction of 180-day windows with positive dT | — | `prospective_eval.py --bin 180D` | `prospective.json` → `California.native.bins_dT_positive_frac`, `.dT`, `.dT_ci` | PENDING |
| Out-of-time 2020–2026: frozen FQ vs frozen ETAS, temporal | `n1_density` | `flowquake.evaluate --tag forward` | `n1_density/eval_forward.json` → `paired_vs_ETAS.temporal.mean_gain`, `.stderr`, `.win_rate`, `paired_vs_ETAS.n_matched` | PENDING |
| Frozen ETAS on the forward window | `forward_etas` | `etas_forward_eval.py` | `forward_etas/summary.json` → `window`, `n`, `tll`, `sll` | PENDING |
| ETAS-refit-through-2020 fairness control | `forward_etas_ComCat_25_refit2020` | `etas_forward_eval.py --name ComCat_25_refit2020` | `forward_etas_ComCat_25_refit2020/summary.json` → `tll`, `sll` | PENDING |

## §4.2 CSEP consistency

| claim | run dir | script | JSON key | value |
|---|---|---|---|---|
| Production N1 standalone N/S/M (100 days x 1e4) | `n1_density/csep` | `flowquake.csep_forecast` | `csep_results.json` → `summary.N.n_pass`/`.n_eval`, `summary.S.*`, `summary.M.*` | PENDING |
| ETAS through the same harness, matched 1e3 budget | `csep_h2h_etas` | `flowquake.etas_csep` then `flowquake.csep_forecast --rerun` | `csep_results.json` → `summary.{N,S,M}.n_pass`/`.n_eval` | PENDING |
| FlowQuake at the same matched 1e3 budget | `csep_h2h_fq` | `flowquake.csep_forecast` | `csep_results.json` → `summary.{N,S,M}.n_pass`/`.n_eval` | PENDING |
| Full-history head N/S/M, matched 1e3 budget | `n1_density/csep_head` | `flowquake.csep_forecast_head` | `csep_results.json` → `summary.{N,S,M}.n_pass`/`.n_eval` | PENDING |
| Paired S-test, head vs ETAS on shared days (McNemar) | `n1_density/csep_head` + `csep_h2h_etas` | `audit_readiness.py` (`full_history_head_csep` check) | `replacement_readiness.json` → the check's `evidence.shared_days`, `.head_S_pass`, `.etas_S_pass` | PENDING |
| Full-history simulator reproduces the head's per-event SLL | `neural_etas/ComCat_25` | `flowquake.neural_etas_forecast` validation path | NO ARTIFACT recorded — the 9.5e-7 nats figure is printed, not written to JSON. Decide whether to capture it. | PENDING |

## §4.3 Memorization mechanism

| claim | config | run dir | script | JSON key | value |
|---|---|---|---|---|---|
| h ∈ {0,4,16,64} train/test NLL at the converged checkpoint | `ablation_h/h{h}.yaml` (written by `ablation_h.py`) | `ablation_h/h{h}` | `ablation_h.py` then `memorization_eval.py` | `ablation_h/memorization_figure.json` → rows with `h`, `ckpt`, `step`, `train.nll`, `test.nll`, `gap_nll` | PENDING |
| Early stopping does not rescue h>0 (best held-out is the first checkpoint) | as above | `ablation_h/h{h}` | `memorization_eval.py` (`ckpt == "best"` rows) + `ablation_h/h*/metrics.jsonl` | `memorization_figure.json` → `step`, `test.nll` for `ckpt: "best"` | PENDING |

## §4.4 Spatial gap and the full-history head

| claim | run dir | script | JSON key | value |
|---|---|---|---|---|
| Stage-0: head reproduces the package ETAS SLL (gate closed) | — | `etas_sll_repro.py` | `etas_sll_repro.json` → `max_abs_sll_err`, `mean_sll_ours` | PENDING |
| Spatial gap localization by nearest-recent-neighbour distance | `n1_density` | `spatial_gap_decomp.py` | `n1_density/spatial_gap_decomp.json` | PENDING |
| ComCat head vs ETAS spatial, seed 0 | `neural_etas/ComCat_25` | `train_neural_etas.py ComCat_25 --seed 0` | `summary_full_s0.json` → `dS_mean`, `dS_ci`, `decision`, `test_sll_neural`, `test_sll_etas` | PENDING |
| Same, seeds 1 and 2 (3-seed spread) | `neural_etas/ComCat_25` | `train_neural_etas.py --seed 1 / 2` | `summary_full_s1.json`, `summary_full_s2.json` → `dS_mean` | PENDING |
| Background-only ablation (no neural modulation) | `neural_etas/ComCat_25` | `train_neural_etas.py --no-mlp` | the `--no-mlp` tag's `summary_*.json` → `dS_mean` | PENDING |
| Classical (flETAS-style) refit control | `neural_etas/ComCat_25` | `train_neural_etas.py --refit-globals` | that tag's `summary_*.json` → `dS_mean` | PENDING |
| What the head learned (background mixture weights) | `neural_etas/ComCat_25` | `train_neural_etas.py` / `make_modulation_figure.py` | `summary_full_s0.json` → `bg_weights[unif,kde...]`, `alpha_far`, `mu_adj` | PENDING |
| ComCat total likelihood, test window | — | `total_win_summary.py` | `total_win.json` → `test_2007_2020.dTot.mean`, `.ci`, `.decision`, `.p_boot`, and `.fq_nll` / `.etas_nll` | PENDING |
| ComCat total likelihood, 2020–2026 forward window | — | `total_win_summary.py` | `total_win.json` → `forward_2020_2026.dTot.mean`, `.ci` | PENDING |

### Six-region total-likelihood table (the README table)

All six rows come from one artifact: `runs/stats_hardening.json` →
`total_with_head_family.<Region>`. Read `dTot_mean` for the gain, `p_holm` for
the family-adjusted p-value, `pairing.coverage_vs_etas` for the coverage figure,
and `temporal_variant` to confirm native vs few-shot.

| region | FQ temporal run dir | head run dir | JSON key | value |
|---|---|---|---|---|
| California | `n1_density` | `neural_etas/ComCat_25` | `total_with_head_family.California.dTot_mean` | PENDING |
| Italy | `italy_n1` | `neural_etas/Italy_25` | `total_with_head_family.Italy.dTot_mean` | PENDING |
| Japan | `japan_n1` | `neural_etas/Japan_25` | `total_with_head_family.Japan.dTot_mean` | PENDING |
| Chile | `chile_n1` | `neural_etas/Chile_25` | `total_with_head_family.Chile.dTot_mean` | PENDING |
| Greece | `greece_fewshot` | `neural_etas/Greece_25` | `total_with_head_family.Greece.dTot_mean` | PENDING |
| Iran | `iran_fewshot` | `neural_etas/Iran_25` | `total_with_head_family.Iran.dTot_mean` | PENDING |

Region → run-dir mapping is `HEAD_COMBOS` in `scripts/stats_hardening.py`.

### Spatial-head transfer (7/7 within-regime)

| claim | run dir | script | JSON key | value |
|---|---|---|---|---|
| Source-trained head applied to an unseen target region | `neural_etas` | `transfer_neural_etas.py` | `neural_etas/transfer_from_<source>.json` | PENDING |

## §4.5 Cross-regime generalization and transfer

| claim | config | run dir | script | JSON key | value |
|---|---|---|---|---|---|
| Per-region native temporal vs region-fitted ETAS | `{japan,chile,greece,iran,italy}_n1.yaml`, one per region, seeds passed as `--seed 1553/1554/1555 --out ...` (REPRODUCE.md §3) | `{japan,chile,greece,iran,italy}_n1[_sSEED]` | `multiregion_table.py` | `multiregion_master.json` → `<Region>.native.paired.dT`, `.dT_decision` | PENDING |
| Holm-adjusted family across the six regions | — | `stats_hardening.py` | `stats_hardening.json` → `family_dT_holm.<Region>.p_holm`, `.significant_05` | PENDING |
| Greece TOST equivalence / Iran non-equivalence | — | `stats_hardening.py` | `stats_hardening.json` → `per_region.Greece.dT_tost_0.1`, `per_region.Iran.dT_tost_0.1` | PENDING |
| Leave-one-region-out pooled pre-training, zero-shot | `chile_n1.yaml`, `greece_n1.yaml`, `iran_n1.yaml` (pool) | `pool_loo_<region>` | `train_pooled.py` then `transfer_eval.py` | `transfer_<Region>.json` | PENDING |
| Few-shot recovery on the held-out region | `<region>_n1.yaml --init-from` | `<region>_fewshot` | `flowquake.train --init-from` | `<region>_fewshot/eval_test.json` → `tll`, `baselines.ETAS.tll` | PENDING |
| One pooled deployment checkpoint across regions | — | `pool_global` | `train_pooled.py`, `eval_global.py` | `global_eval.json` | PENDING |
| Data-efficiency curve | `chile_n1.yaml` | `de_chile_*` | `data_efficiency.py --region chile` | `data_efficiency_chile.json` | PENDING |
| Mw-homogenization robustness | `italy_mw_n1.yaml`, `italy_mc28_n1.yaml` | `italy_mw_n1`, `italy_mc28_n1` | `mw_robustness.py` | `mw_robustness.json` → `italy.mw_uniform_mc26`, the California M≥3 subset entry, and `interpretation` | PENDING |

## Claim-boundary audit

| claim | script | JSON key | value |
|---|---|---|---|
| Overall readiness verdict and per-check levels | `audit_readiness.py` | `replacement_readiness.json` → `overall`, `checks[].level`, `checks[].name` | PENDING |
| Strongest defensible claim sentence | `audit_readiness.py` | `replacement_readiness.json` → `strongest_defensible_claim` | PENDING |

## Open items for this file

- The 9.5e-7-nat simulator validation in §4.2 is printed rather than written to
  JSON. Either capture it in `csep_head/csep_results.json` or cite the console
  log explicitly in the manuscript.
- Figures are gitignored. If any manuscript figure is cited as evidence rather
  than illustration, decide whether to commit it.
- Several run directories are named by CLI override rather than by a committed
  config (region seeds, `--no-mlp` / `--refit-globals` ablations,
  `de_<region>_*`). Those rows name the invocation instead; if a reviewer needs
  to reproduce them exactly, the invocation should be recorded alongside the
  artifact.
