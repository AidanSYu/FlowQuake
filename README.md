# FlowQuake — ETAS-upgrade candidate for EarthquakeNPP

Selective-SSM (Mamba-style) **whole-catalog encoder** + **flow-matching marked
point process**, evaluated on the [EarthquakeNPP](https://github.com/ss15859/EarthquakeNPP)
benchmark against the operational ETAS model. The strongest current claim is a
multi-seed temporal log-likelihood win, transferable deployment evidence, and a
full-history neural-ETAS spatial head that flips total likelihood against
region-fitted ETAS in the six tested regions. CSEP consistency has been re-run
with that upgraded head at a matched simulation budget and is statistically
indistinguishable from ETAS on all three tests (§4.2 of `MANUSCRIPT.md`). This
is still an ETAS upgrade path rather than an operational replacement: the head
is initialized from each region's ETAS inversion, and a future prospective
forecast must be registered before making deployment claims.

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

## Total likelihood vs region-fitted ETAS

The full-history neural-ETAS spatial head (`flowquake/neural_etas.py`) combined
with the flow temporal head, each region scored against *that region's own* ETAS
inversion. Paired per-event total-likelihood gain, nats/event, higher is better
(§4.4 of `MANUSCRIPT.md`; source `runs/stats_hardening.json`):

| region | ΔTotal (nats/event) | note |
|---|---|---|
| California (ComCat_25) | +0.113 | nll 7.142 vs ETAS 7.255 |
| Italy | +0.210 | native-catalogue scale, not claimed under Mw homogenization |
| Japan | +0.039 | positive, but below the 0.05-nat interpretability margin |
| Chile | +0.061 | |
| Greece | +0.076 | few-shot temporal transfer |
| Iran | +0.084 | few-shot temporal transfer |

Holm-adjusted p ≤ 0.019 across the six-region family. Totals outside California
and Italy are paired on the intersection of FlowQuake-temporal, neural-head and
ETAS-scored events (coverage 89.0–97.1%; per-region figures in §4.4).

**The qualifier that bounds this result:** the head is initialized from each
region's ETAS inversion, so the composite is an *upgrade of a deployed ETAS
system*, not an inversion-free replacement for one. Per-region normalization and
a train-era smoothed-seismicity background map are also still required, so this
is lighter than an ETAS inversion but not zero target-catalog preprocessing.

## Setup and data

```bash
git clone https://github.com/AidanSYu/FlowQuake.git
cd FlowQuake
pip install -e ".[dev]"     # core deps + pytest; Python >= 3.11
python -m pytest tests/ -q  # 16 pass; the data-alignment tests skip until reference/ exists
```

Optional extras, all declared in `pyproject.toml`:

| extra | unlocks |
|---|---|
| `dev` | `pytest` — the test suite |
| `figures` | `matplotlib` — `scripts/make_*_figure.py` |
| `csep` | `pycsep` — the CSEP N/S/M consistency tests |
| `etas` | the `etas` package + `seismostats` — re-inverting ETAS baselines |

**`reference/` must be fetched separately, and nothing runs without it.** Every
config's `catalog_path` is relative to it, so `flowquake.train` fails immediately
on a fresh clone. It is a clone of the benchmark, not part of this package:

```bash
git clone https://github.com/ss15859/EarthquakeNPP.git reference
```

Then populate the catalogs with the benchmark's own dataset step — for ComCat
that is a USGS ComCat query, the RELM/CSEP California polygon filter, the mc 2.5
cut, duplicate jitter, and an azimuthal-equidistant projection, all documented in
`reference/Datasets/ComCat/README.md` (`scripts/build_comcat_forward.py` in this
repo replicates the same recipe for the post-2020 window, and is a readable
statement of it). The cross-regime catalogs are built from agency FDSN services by
`scripts/build_region.py` — see `REPRODUCE.md` §1 for the exact invocations.

The ETAS baseline is the incumbent every number in the manuscript is measured
against, and it is **read from disk, never recomputed at eval time**:

```
reference/Experiments/ETAS/output_data_<Cfg>/ll_scores.json        # aggregate tll/sll/nll
reference/Experiments/ETAS/output_data_<Cfg>/augmented_catalog.csv # per-event TLL/SLL
```

If the clone does not already carry those outputs, regenerate them by driving the
benchmark's own `invert_etas.py` / `predict_etas.py` from here (CPU only, roughly
3–4 h of EM inversion per large region):

```bash
python scripts/run_etas_regions.py ComCat_25          # needs reference/Experiments/ETAS/config/ComCat_25.json
```

Expected tree once populated:

```
reference/
├── Datasets/
│   ├── plot_utils.py                        # projection helpers the build scripts import
│   ├── ComCat/ComCat_catalog.csv            # 92,263 events >= M2.5, 1971 -> 2020-01-17
│   ├── ComCat/california_shape.npy          # RELM polygon, [lat, lon] vertices
│   ├── WHITE/WHITE_catalog.csv              # WHITE_06
│   ├── QTM/{SanJac,SaltonSea}_catalog.csv   # SanJac_10, SaltonSea_10
│   ├── SCEDC/SCEDC_catalog.csv              # SCEDC_20
│   └── {Japan,Chile,Greece,Iran,Italy}/     # written by scripts/build_region.py (§4.5)
└── Experiments/ETAS/
    ├── config/<Cfg>.json                    # one per catalog
    ├── invert_etas.py  predict_etas.py      # the benchmark's inversion/prediction drivers
    └── output_data_<Cfg>/                   # ll_scores.json + augmented_catalog.csv
```

On-disk size of a fully populated `reference/`: PENDING — measure with
`du -sh reference/` and fill in.

## Run

```bash
# train the production (N1, density-adaptive) model (RTX 4090, ~45 min)
python -m flowquake.train configs/n1_density.yaml      # -> runs/n1_density/

# evaluate on the test window vs ETAS (exact ODE likelihoods)
python -m flowquake.evaluate runs/n1_density/ckpt_best.pt --steps 96

# CSEP N/S/M consistency (100 forecast days x 1e4 simulated catalogs)
python -m flowquake.csep_forecast runs/n1_density/ckpt_best.pt --n-days 100 --n-sims 10000

# the same tests with the full-history spatial head, at the matched 1e3 budget (§4.2)
python -m flowquake.csep_forecast_head runs/n1_density/ckpt_best.pt --n-days 100 --n-sims 1000

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

The 3-seed full-suite numbers (`scripts/aggregate_fullsuite.py` →
`runs/fullsuite_summary.json`) come from the **N1** density-adaptive configs:
`configs/{n1_density,n1_s1553,n1_s1554}.yaml` for ComCat_25, and
`configs/{WHITE_06,SanJac_10,SaltonSea_10,SCEDC_20}_n1{,_s1554,_s1555}.yaml` for
the other four catalogs — the run directories `aggregate_fullsuite.py` reads are
exactly these configs' `out_dir`s. `configs/final_s{1553,1554,1555}.yaml` are the
earlier *canonical* (non-density-adaptive) ComCat seeds, kept for the
canonical-vs-N1 spatial comparison in §4.4.

## Claim ladder

1. **Temporal kill**: beat ETAS tll / pass temporal N-test. *(Known to
   underdeliver per benchmark authors — necessary, not sufficient.)*
2. **Spatial/total likelihood win**: the full-history neural-ETAS head closes
   the ETAS spatial gap and flips total likelihood on the reported regions.
3. **Operational replacement gate**: matched ETAS-vs-FlowQuake CSEP with the
   full-history head is done (N 95/100, S 79/85, M 90/92; S-test McNemar exact
   p = 1.00 vs ETAS) and so is the ETAS-refit forward control. What remains is
   registering a genuinely future rolling forecast.

## Notes

- Likelihood conventions were verified against the harness: mean per-event
  TLL/SLL in `augmented_catalog.csv` reproduce the published `ll_scores.json`
  to all printed digits.
- Training uses random contiguous crops (window 2048, burn-in 256) of the
  whole catalog; evaluation encodes the full 92k-event sequence with carried
  streaming state, so test events condition on the *entire* history.
- The flow heads are zero-initialized (identity-ish velocity), which makes
  early FM training stable.

## License and data

Code in this repository is licensed under Apache-2.0 (see `LICENSE`).

The catalogs and ETAS baseline outputs under `reference/` are **not** covered by
that licence. They derive from the [EarthquakeNPP](https://github.com/ss15859/EarthquakeNPP)
benchmark (Stockman, Lawson & Werner, TMLR 2026) and from the agency catalogs it
and `scripts/build_region.py` draw on (USGS ComCat, SCEDC, QTM, ISC, INGV); each
carries its own licence and attribution terms, which apply to any redistribution.
Nothing under `reference/` is committed here.
