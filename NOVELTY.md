# Novelty assessment (deep-research, 2026-06-26)

102-agent adversarial literature sweep (19 sources, 25 claims verified 3-vote).
Verdict per sub-claim and the framing guardrails that keep it defensible.

## Verdict
| sub-claim | verdict | must-cite / threat |
|---|---|---|
| 1. Cross-regime transfer beating region-fitted ETAS | **NOVEL for the point-process / likelihood class** | SafeNet (broad version); RECAST (only proposed) |
| 2. Foundation model aimed at point-process *forecasting* | **NOVEL (open niche)** | MultiFoundationQuake, SeisLM (both off-target) |
| 3. Beat ETAS temporally, multi-seed significance, all 5 CA catalogs | **NOVEL as a clean significance-tested multi-catalog win** | Zhan NMRP 2026; EarthquakeNPP nuance |
| 4. Data-efficiency via transfer in data-poor regions | **NOVEL (cross-region-transfer mechanism)** | FERN (ties only), NMRP (in-region), SafeNet (broad) |
| 5. Global-embedding → memorization | **Unclaimed but UNCONFIRMED** — no source either way | frame as diagnostic novelty, not "first" |

## THE framing guardrail (load-bearing)
Claim **"first neural point-process / point-process-likelihood forecaster to transfer
across tectonic regimes and beat region-fitted ETAS on temporal log-likelihood."**
Do **NOT** claim "first transfer learning for earthquake forecasting" unqualified —
**SafeNet** preempts the broad version.

## Prior art that MUST be cited and explicitly differentiated
- **SafeNet** — Zhang et al. (2025), *Scientific Reports*, 10.1038/s41598-025-93877-7.
  Cross-region few-shot transfer forecasting that beats ETAS — BUT a 4°×4° gridded
  **annual-max-magnitude classifier** scored on F1/recall; no likelihood, no intensity,
  no point process. → different model class. *(The closest prior claim on the transfer
  axis; the point-process boundary against it must be stated precisely.)*
- **Zhan et al. (2026) NMRP** — *Earth's Future*, 10.1029/2025EF007342 (Apr 2026).
  Neural modulated renewal process; "matches and in some cases surpasses ETAS" temporally
  on EarthquakeNPP — BUT no spatial head, no transfer, no multi-seed significance, CA only.
  → closest temporal-axis competitor; differentiate on spatial head + transfer + significance.
- **RECAST** — Dascher-Cousineau et al. (2023), *GRL*, 10.1029/2023GL103909.
  Neural TPP; multi-region transfer only *proposed* in press, never demonstrated; worse than
  ETAS below ~10⁴ events. Berkeley team flagged transfer as active work → re-sweep before submit.
- **FERN/FERN+** — Zlydenko et al. (2023), *Sci Reports*, 10.1038/s41598-023-38033-9.
  Neural rate model only *ties* ETAS; gains come from sub-Mc magnitudes, per-region, no transfer.
- **EarthquakeNPP** — Stockman, Lawson, Werner, TMLR 03/2026, arXiv:2410.08226.
  Canonical baseline: none of 5 NPPs beat ETAS; ETAS wins spatial LL vs all. Concedes NPPs
  "marginally better in isolated cases" (SCEDC high-Mc) → our novelty = significance +
  all-catalog consistency, state precisely.
- **EPBench** (2025, arXiv:2505.15588), **SeisLM** (NeurIPS 2024), **MultiFoundationQuake**
  (2024) — corroborating context (multi-region≠transfer; detection≠forecasting; generic
  TSFM energy regression ≠ point process).

## Venue
Candidate venues, none decided: *GRL* and *JGR Solid Earth* are the natural fits for
the seismological audience; *Seismica* is the open-access alternative. The
transfer/foundation-model framing would also be in scope for a broader journal, but
that is a judgement about breadth of interest, not something the novelty assessment
above establishes.

## Before claiming "first" at submission
1. Fresh **May–June 2026 EarthArXiv/arXiv sweep** (fast field; RECAST team active on transfer).
2. Targeted search for sub-claim 5 (global-conditioning memorization) in NPP + general ML lit.
3. Ensure multi-seed significance survives scrutiny (benchmark concedes marginal NPP edges).
4. Crisp point-process-vs-classifier boundary vs SafeNet in related work.
