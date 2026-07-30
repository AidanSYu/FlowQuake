# Reproducing FlowQuake (cross-regime study)

End-to-end pipeline for the results in `MANUSCRIPT.md`. California-suite results
(§4.1–4.4) use the EarthquakeNPP catalogs as shipped; the cross-regime study
(§4.5) is built from authoritative agency data below. GPU: one job at a time
(single shared RTX 4090; run `nvidia-smi` before launching a GPU job). Run from
the repo root with `PYTHONPATH=.`.

## 1. Authoritative catalogs (CPU/network)
ISC reviewed Bulletin (mc 4.0 group) and INGV (Italy, mc 2.5):
```
python scripts/build_region.py Japan  22  46  122 150 --source isc  --dl-mag 3.5
python scripts/build_region.py Chile  -40 -17 -76 -66 --source isc  --dl-mag 3.5
python scripts/build_region.py Greece 34  42  19  29  --source isc  --dl-mag 3.5
python scripts/build_region.py Iran   25  40  44  63  --source isc  --dl-mag 3.5
python scripts/build_region.py Italy  36  47  6   19  --source ingv --dl-mag 2.0 --start 1992-01-01 --window-days 90
python scripts/check_completeness.py        # per-era Mc; confirms mc 4.0 (ISC) / 2.5 (Italy)
```
The downloader handles FDSN quirks: ISC dense-window splitting, INGV's silent
100-row cap (explicit `&limit`), GeoNet's 413 (recursive halving). Each region
writes `<name>_catalog.csv`, `<name>_shape.npy`, `<name>_meta.json`.

## 2. Region-fitted ETAS baselines (CPU, ~3–4 h each)
```
python scripts/run_etas_regions.py Japan_25 Chile_25 Greece_25 Iran_25 Italy_25 --invert-jobs 2
```
Inverts + predicts each (configs in `reference/Experiments/ETAS/config/`), writing
`output_data_<cfg>/augmented_catalog.csv` (per-event TLL/SLL) and `ll_scores.json`.

## 3. Native FlowQuake, multi-seed (GPU)
```
for cfg in japan chile greece iran italy; do
  for s in 1553 1554 1555; do
    python -m flowquake.train configs/${cfg}_n1.yaml --seed $s --out runs/${cfg}_n1_s$s --eval-after
  done
done
```
(California-suite configs: `configs/{comcat25,SCEDC_20,SanJac_10,SaltonSea_10,WHITE_06}*`.)

## 4. Foundation model — leave-one-region-out (GPU)
Pre-train on the other mc-4.0 regions, then zero-/few-shot the held-out one:
```
python scripts/train_pooled.py --configs configs/chile_n1.yaml configs/greece_n1.yaml configs/iran_n1.yaml \
    --out runs/pool_loo_japan --steps 30000
python scripts/transfer_eval.py --ckpt runs/pool_loo_japan/ckpt_best.pt \
    --catalog reference/Datasets/Japan/Japan_catalog.csv --mcut 4.0 ...   # zero-shot
python -m flowquake.train configs/japan_n1.yaml --init-from runs/pool_loo_japan/ckpt_best.pt --steps 2000 \
    --out runs/japan_fewshot --eval-after                                  # few-shot
# repeat holding out chile / greece / iran
```

## 5. Analyses and figures
```
python scripts/multiregion_table.py            # master vs-ETAS table -> runs/multiregion_master.json
python scripts/make_density_figure.py          # Fig. density_dependence (the central result)
python scripts/prospective_eval.py --bin 180D  # block-bootstrap info-gain over the test decade
python scripts/data_efficiency.py --region chile --pool-ckpt runs/pool_loo_chile/ckpt_best.pt --base configs/chile_n1.yaml
python scripts/make_data_efficiency_figure.py --region chile
python scripts/make_multiregion_figure.py
```

## 6. Full-history neural-ETAS spatial head (CPU — no GPU needed)
Closes/reverses the spatial gap. The gate-closed head is a strict superset of
the benchmark ETAS spatial density and reproduces the package per-event SLL to
2e-9 first; the default training init opens a small KDE gate so gradients reach
the smoothed-seismicity background:
```
python scripts/etas_sll_repro.py                        # Stage-0 harness check (must pass)
python scripts/precompute_trigger_features.py ComCat_25 # ~30 min; verifies vs package SLL
python scripts/train_neural_etas.py ComCat_25 --epochs 200 --patience 12 --seed 0   # + seeds 1,2
python scripts/train_neural_etas.py ComCat_25 --no-mlp ...          # background-only ablation
python scripts/train_neural_etas.py ComCat_25 --refit-globals ...   # classical (flETAS-style) control
python scripts/make_modulation_figure.py                # what the head learned
# other regions: python scripts/precompute_trigger_features.py Italy_25   (etc.)
```

Then re-run CSEP with that head, so the likelihood win and the consistency
result are measured on the same model (§4.2; matched 10^3-catalog budget on the
same 100 days as the production-head and ETAS runs):
```
python -m flowquake.csep_forecast_head runs/n1_density/ckpt_best.pt \
    --head ComCat_25 --n-days 100 --n-sims 1000   # -> runs/n1_density/csep_head/
python scripts/make_csep_h2h_figure.py                  # Fig. csep_headtohead
```

## 7. Out-of-time 2020-2026 window (CPU/network)
```
python scripts/build_comcat_forward.py                  # exact benchmark recipe, USGS fetch
python scripts/etas_forward_eval.py --validate          # scorer must reproduce package TLL/SLL
python scripts/etas_forward_eval.py                     # frozen 2007-fit ETAS on 2020-2026
python scripts/etas_forward_eval.py --name ComCat_25_refit2020 \
    --out runs/forward_etas_ComCat_25_refit2020          # after the refit-through-2020 inversion finishes
python -m flowquake.evaluate runs/n1_density/ckpt_best.pt --catalog \
    reference/Datasets/ComCat_extended/ComCat_extended_catalog.csv \
    --test-start 2020-01-17 --test-end 2026-07-01 --tag forward --device cpu \
    --etas-aug runs/forward_etas/per_event.csv           # frozen FlowQuake, same window
python scripts/precompute_trigger_features.py ComCat_25 --catalog \
    reference/Datasets/ComCat_extended/ComCat_extended_catalog.csv \
    --targets-from 2020-01-17 --out ComCat_25_forward
python scripts/score_neural_etas.py runs/neural_etas/ComCat_25/head_full_s0.pt \
    ComCat_25_forward --out runs/neural_etas/ComCat_25/per_event_forward_full.csv
python scripts/total_win_summary.py                      # paired dT/dS/dTot, test + forward
```

## 8. Statistics hardening (CPU)
```
python scripts/stats_hardening.py    # Holm-Bonferroni family + TOST equivalence -> runs/stats_hardening.json
```

## 9. Replacement-readiness audit (CPU)
```
python scripts/audit_readiness.py   # -> runs/replacement_readiness.json
```
This is the gate for public wording. With the run artifacts present it should
reach `RESEARCH_PREVIEW_READY`: the research-preview checks, the full-head CSEP
check and the ETAS-refit forward control all pass. It does not certify an
operational system — the head is still initialised on the target region's ETAS
inversion plus a causal KDE background, and no prospective forecast has been
registered.

## Key results (significance-tested, paired on reported event sets)
- Temporal, native FlowQuake vs region-fitted ETAS (dT, nats/event): WIN on
  every dense/low-mc catalog across 3 regimes — transform (California +0.053,
  SCEDC +0.078, …), extension (Italy +0.071), subduction (Chile +0.034);
  California/Italy survive Holm–Bonferroni at adj p = 0.003 and Chile at
  adj p = 0.036. Japan is a small negative/equivalent boundary case at mc 4.0
  (Tōhoku/Omori). Data-poor Greece/Iran lose natively; transfer recovers Greece
  to TOST equivalence (±0.1) and narrows Iran 4× (not equivalent).
- **Spatial/total (new)**: the full-history neural-ETAS head beats the
  benchmark ETAS spatially in all six tested regions. Combined with FlowQuake's
  temporal head (few-shot temporal transfer for Greece/Iran), total likelihood
  is positive in all six; Japan's +0.039 is a statistically positive but small
  effect below the 0.05-nat interpretability margin. Non-California/Italy totals
  are paired on the intersection of temporal/head/ETAS-scored events with
  coverage reported in `runs/stats_hardening.json`.
- **Out-of-time 2020-2026 replication (new)**: frozen production FlowQuake beats
  the frozen benchmark ETAS temporally on the post-2020 ComCat window (dT +0.057,
  10,187 events); with the frozen spatial head, dTot is +0.124. This is
  retrospective out-of-time scoring, not a registered prospective forecast.
The temporal advantage is density-dependent (shrinks as mc rises).

## Notes / caveats
- Magnitudes are agency-preferred (ISC/INGV); types mixed (documented in
  `<name>_meta.json`). At the analysis mc, b ≈ 0.9–1.0; an Mw-consistent
  robustness check is a supplement item.
- New Zealand (GeoNet) was attempted at mc 3.5 but its ETAS inversion was
  prohibitively slow at catalog scale and is omitted; the 3-regime result does
  not depend on it.
- The pooled global model is one shared deployment checkpoint with no
  per-region weight fitting after pooling; it is not leave-one-region-out
  zero-shot transfer because each region's training window participates in the
  pooled pre-training run.
- FlowQuake still recomputes per-region normalization and a train-era
  smoothed-seismicity background map. This is much lighter than ETAS inversion,
  but it is not zero target-catalog preprocessing.
- The production kernel-mixture spatial head still trails ETAS; the spatial/total
  win uses the full-history neural-ETAS head initialized from each region's ETAS
  inversion. CSEP consistency has been re-run with that upgraded head through the
  same pyCSEP path (100 identical forecast days, matched 10^3-catalog budget):
  N 95/100, S 79/85, M 90/92, S-test statistically indistinguishable from ETAS
  (McNemar exact p = 1.00). See MANUSCRIPT.md §4.2.
- RunPod / external compute not required; all runs above are local.
