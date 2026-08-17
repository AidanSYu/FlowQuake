# FlowQuake

Measuring where the forecast information in small earthquakes actually lives.

Dense seismic monitoring promises better earthquake forecasts through deeper
detection. This repository measures that promise directly: how much forecast
skill each band of small earthquakes adds, band by band, and where the value
stops.

## The two results

**Detection depth is nearly spent.** With a standard ETAS model fitted once on
larger events and then frozen, re-scoring the identical forecast with
progressively deeper conditioning history shows each magnitude band adding less
than the one before it, with the contribution indistinguishable from zero near
M1. Of all the forecast information a deeper catalog could supply at San
Jacinto, 85% (interval 45 to 96) is already present in events above M0.75. A
saturating curve beats a scale-free one decisively (delta AIC +19.6).

**Location precision is not.** Adding 2 km of epicentre error to the same
catalog erases about a third of the total information, with no sign of leveling
off. After removing a constant offset, two published catalogs of the same fault
already disagree about the same earthquakes by 0.6 km typically and 1.3 km at
the 90th percentile.

Checks: catalog completeness verified to M0.65 (Poisson test against the
Gutenberg-Richter extrapolation), the shape survives four different frozen model
fits and grid coarsening, and magnitude-dependent location error is bounded to
at most 9 to 17% of the observed decline.

## The instrument

- `scripts/information_ladder.py` fits ETAS once at an anchor magnitude,
  freezes it, and re-scores the same frame with deeper history per rung.
  Supports epicentre jitter (uniform or magnitude-dependent) and a decoupled
  fit anchor for model-specificity tests.
- `scripts/floor_magnitude.py` contests saturating against scale-free
  accumulation on the per-window matrix, with bootstrap intervals.
- `scripts/completeness_check.py` finds the magnitude where a catalog stops
  being complete, requiring deficits to be both statistically decisive and
  sustained.
- `scripts/artifact_bound.py` bounds how much of the measured saturation
  magnitude-dependent location error could account for.

Everything runs on CPU. Result summaries live in `runs/` as JSON next to the
per-window matrices that support the bootstraps.

## Reproducing

```bash
python -m pytest tests/            # 196 tests
python scripts/information_ladder.py --panel runs/panel_white \
    --base configs/panel_white.yaml --anchor 2.5 --floor 0.75 --step 0.25
python scripts/floor_magnitude.py \
    --ladder runs/panel_white/information_ladder_uniform_deep.json
```

See `REPRODUCE.md` for the full protocol, including the earlier neural-model
comparison (a flexible learned model fails to extract the information that ETAS
extracts from small events, which is a finding about model structure rather
than a tooling problem).

## Data

Catalogs under `reference/Datasets/` retain their original documentation:
WHITE (San Jacinto), QTM (Salton Sea and San Jacinto), SCEDC, and ComCat
derivatives. The deep-catalog measurements use WHITE, the only catalog tested
that is complete below M1.0.
