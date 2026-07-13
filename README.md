# FlowQuake — ETAS-upgrade candidate for EarthquakeNPP

Selective-SSM (Mamba-style) **whole-catalog encoder** + **flow-matching marked
point process**, evaluated on the [EarthquakeNPP](https://github.com/ss15859/EarthquakeNPP)
benchmark against the operational ETAS model. The strongest current claim is a
multi-seed temporal log-likelihood win, transferable deployment evidence, and a
full-history neural-ETAS spatial head that flips total likelihood against
region-fitted ETAS in the six tested regions. This is an ETAS upgrade path, not
yet an operational replacement: CSEP must still be re-run with the upgraded
spatial head and a future prospective forecast must be registered before making
deployment claims. See `SEED.md` for the original research bet.

The two reasons neural point processes have lost to ETAS — fixed-window
encoders (DeepSTPP sees 20 events) and hand-crafted Omori/Gutenberg-Richter
kernels — are exactly what this stack replaces:

- **Encoder** (`flowquake/ssm.py`): Mamba-2-style selective SSM in pure
  PyTorch (chunked SSD scan, no CUDA kernels — runs on Windows). Encodes the
  *entire* catalog history, with magnitude marks (DeepSTPP drops them).
- **Heads** (`flowquake/flow.py`, `flowquake/heads.py`, `flowquake/model.py`):
  a conditional **rectified flow on log-τ** for time (exact ODE likelihood),
  plus **structured, observation-anchored** heads for space (a mixture of
  ETAS-style anisotropic power-law kernels at the last-64 events + long-lived
  big triggers + a train-era KDE background + uniform) and magnitude (a
  conditional Gutenberg–Richter exponential). Closed-form `sll`/`mll`; ODE only
  for `tll`. **Production runs with `h_bottleneck=0`**: exposing the heads to
  the learned whole-catalog embedding causes catastrophic memorization (§4.3 of
  `MANUSCRIPT.md`), so the learned conditioning excludes absolute coordinates
  and uses relational features. The spatial density still includes lightweight
  per-region normalization and a train-era smoothed-seismicity background map.

## Benchmark protocol (ComCat_25, grounded from the harness)

| | dates |
|---|---|
| auxiliary | 1971-01-01 → 1981-01-01 |
| train | → 1998-01-01 (NPP training uses aux+train) |
| val | 1998-01-01 → 2007-01-01 |
| test | 2007-01-01 → 2020-01-17 (21,889 events ≥ M2.5) |

Scores (per test event, higher better): `tll` = log conditional density of
event time (log 1/day), `sll` = log f(x,y|t) (log 1/km²), `nll = −(tll+sll)`.

**Targets to beat** (from `reference/Experiments/ETAS/output_data_ComCat_25/ll_scores.json`):

| model | tll | sll | nll |
|---|---|---|---|
| ETAS | 1.4343 | −8.6898 | **7.2554** |
| Poisson | 0.5126 | −13.7745 | 13.2619 |

Per-event ETAS scores (`augmented_catalog.csv`) are used for paired
per-event comparison (mean gain ± stderr, win rate).

## Run

```bash
# train the production (N1, density-adaptive) model (RTX 4090, ~45 min)
python -m flowquake.train configs/n1_density.yaml      # -> runs/n1_density/

# evaluate on the test window vs ETAS (exact ODE likelihoods)
python -m flowquake.evaluate runs/n1_density/ckpt_best.pt --steps 96

# CSEP N/S/M consistency (100 forecast days x 1e4 simulated catalogs)
python -m flowquake.csep_forecast runs/n1_density/ckpt_best.pt --n-days 100 --n-sims 10000

# memorization ablation (the mechanism result, §4.3) and its eval
python scripts/ablation_h.py            # trains h in {0,4,16,64}
python scripts/memorization_eval.py     # -> runs/ablation_h/memorization_figure.json

# manuscript figures
python scripts/make_figures.py          # -> figures/

# replacement-readiness / claim-boundary audit
python scripts/audit_readiness.py       # -> runs/replacement_readiness.json

# tests (scan exactness, causality, flow log-prob vs analytic, data alignment)
python -m pytest tests/ -q
```

The 3-seed canonical/full-suite numbers use `configs/final_s{1553,1554,1555}.yaml`
and the per-dataset `configs/{WHITE_06,SanJac_10,SaltonSea_10,SCEDC_20}_n1*.yaml`.

`reference/` is a clone of the EarthquakeNPP repo (data + ETAS baseline
outputs); it is not part of this package.

## Internal chain (from SEED)

1. **Temporal kill**: beat ETAS tll / pass temporal N-test. *(Known to
   underdeliver per benchmark authors — necessary, not sufficient.)*
2. **Spatial/total likelihood win**: the full-history neural-ETAS head closes
   the ETAS spatial gap and flips total likelihood on the reported regions.
3. **Operational replacement gate**: re-run matched ETAS-vs-FlowQuake CSEP with
   the full-history head, validate the ETAS-refit forward control, and register
   a genuinely future rolling forecast.

## Notes

- Likelihood conventions were verified against the harness: mean per-event
  TLL/SLL in `augmented_catalog.csv` reproduce the published `ll_scores.json`
  to all printed digits.
- Training uses random contiguous crops (window 2048, burn-in 256) of the
  whole catalog; evaluation encodes the full 92k-event sequence with carried
  streaming state, so test events condition on the *entire* history.
- The flow heads are zero-initialized (identity-ish velocity), which makes
  early FM training stable.
