# FlowQuake — beat ETAS on EarthquakeNPP

Selective-SSM (Mamba-style) **whole-catalog encoder** + **flow-matching marked
point process**, evaluated on the [EarthquakeNPP](https://github.com/ss15859/EarthquakeNPP)
benchmark against the operational ETAS model. See `SEED.md` for the research bet.

The two reasons neural point processes have lost to ETAS — fixed-window
encoders (DeepSTPP sees 20 events) and hand-crafted Omori/Gutenberg-Richter
kernels — are exactly what this stack replaces:

- **Encoder** (`flowquake/ssm.py`): Mamba-2-style selective SSM in pure
  PyTorch (chunked SSD scan, no CUDA kernels — runs on Windows). Encodes the
  *entire* catalog history, with magnitude marks (DeepSTPP drops them).
- **Decoder** (`flowquake/flow.py`, `flowquake/model.py`): conditional
  rectified flows for `f(τ|h) · f(x,y|τ,h) · f(m|τ,x,y,h)`. Simulation-free
  FM training; **exact** log-likelihood at eval via backward ODE with exact
  divergence (targets are 1–2 dim, so the Jacobian trace is cheap).

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
# train (RTX 4090, ~45 min for 20k steps)
python -m flowquake.train configs/comcat25.yaml

# evaluate on the test window vs ETAS (exact ODE likelihoods)
python -m flowquake.evaluate runs/comcat25/ckpt_best.pt --steps 96

# temporal N-test (daily forecast count distributions via simulation)
python -m flowquake.ntest runs/comcat25/ckpt_best.pt --n-days 20 --n-sims 1000

# tests (scan exactness, causality, flow log-prob vs analytic, data alignment)
python -m pytest tests/ -q
```

`reference/` is a clone of the EarthquakeNPP repo (data + ETAS baseline
outputs); it is not part of this package.

## Internal chain (from SEED)

1. **Temporal kill**: beat ETAS tll / pass temporal N-test. *(Known to
   underdeliver per benchmark authors — necessary, not sufficient.)*
2. **CSEP spatial/magnitude win**: the actual bar. `flowquake/ntest.py`
   simulation machinery extends to daily CSEP catalog forecasts (S/M tests
   via pycsep, harness in `reference/Experiments/ETAS/`).
3. **Coulomb-stress kernel**: neural-operator anisotropic spatial intensity —
   the one axis ETAS still wins.

## Notes

- Likelihood conventions were verified against the harness: mean per-event
  TLL/SLL in `augmented_catalog.csv` reproduce the published `ll_scores.json`
  to all printed digits.
- Training uses random contiguous crops (window 2048, burn-in 256) of the
  whole catalog; evaluation encodes the full 92k-event sequence with carried
  streaming state, so test events condition on the *entire* history.
- The flow heads are zero-initialized (identity-ish velocity), which makes
  early FM training stable.
