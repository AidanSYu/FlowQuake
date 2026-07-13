# FlowQuake Replacement Readiness

## Scope

FlowQuake is a candidate replacement for ETAS-like operational seismicity
forecasting pipelines where the target is marked point-process forecasting:
next-event time, location, and magnitude likelihoods or catalog simulations.

It is not a replacement for earthquake early warning, waveform phase picking,
deterministic earthquake prediction, ground-motion simulation, building-code
hazard maps, or emergency-response systems.

## What Is Strong Now

- Temporal likelihood: FlowQuake beats ETAS on every California EarthquakeNPP
  catalog in 3-seed means.
- Mechanism: allowing learned whole-catalog embeddings into the heads causes
  catastrophic memorization; relational/observation-anchored heads avoid it.
- CSEP: the production ComCat model has standalone N/S/M consistency, including
  magnitude.
- Spatial/total likelihood: the full-history neural-ETAS head, initialized from
  each region's ETAS inversion, beats ETAS spatially in all six tested regions
  and flips total likelihood for the composite model.
- Transfer: leave-one-region-out pooled pre-training plus brief fine-tuning
  rescues data-poor regions to ETAS-level temporal skill in the tested set.
- Deployment shape: a single pooled checkpoint can be evaluated across all
  tested regions without per-region weight fitting after pooling.

## Holes That Still Matter

- The production kernel-mixture spatial head still trails ETAS; the spatial win
  is from the newer full-history head.
- The ETAS-vs-FlowQuake CSEP head-to-head is built for the production head, but
  CSEP has not been re-run with the full-history spatial head.
- Per-region normalization and a train-era smoothed-seismicity background map
  are still required; the model is not target-catalog-free.
- The 2020-2026 window is an out-of-time retrospective replication, not a
  registered prospective forecast.
- The pooled global checkpoint is not unseen-region zero-shot transfer when the
  target region's training window was included in pooled pre-training.
- Bibliography and "first" claims still need a final live sweep before
  submission.

## Replacement Ladder

1. Research preview: pass tests, reproduce the California temporal suite, CSEP
   standalone consistency, cross-regime tables, and `scripts/audit_readiness.py`.
2. Incumbent head-to-head: validate ETAS and FlowQuake through the same CSEP
   harness with matched days and simulation counts.
3. Full-head CSEP: validate the full-history spatial head in a matched
   simulation harness.
4. Prospective deployment: freeze a checkpoint and run rolling forecasts on a
   future catalog window that was not used for model selection.
5. Operational artifact: package one checkpoint, preprocessing, calibration,
   forecast export, audit logs, and failure-mode monitoring.

Until rung 2 is complete, the honest public sentence is:

> FlowQuake is a transferable neural point-process candidate that beats ETAS
> temporally on dense catalogs and, with a full-history neural-ETAS spatial head
> initialized from each region's ETAS inversion, beats ETAS on total likelihood
> across the six tested regions; it is not yet an operational replacement for
> ETAS systems.
