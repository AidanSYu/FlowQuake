"""Published reference numbers (per-event test TLL/SLL) from the EarthquakeNPP benchmark.

Source: reference/Experiments/out_metrics.csv and the ETAS ll_scores.json files.
ETAS is the incumbent to beat; the headline gap is SPATIAL (SLL): on every California
dataset ETAS's SLL exceeds that of every published neural point process.  Beating ETAS
means SLL > ETAS_SLL (ideally with TLL >= ETAS_TLL too).
"""
from __future__ import annotations

# (tll, sll) per-event on the test split.
ETAS = {
    "ComCat_25":   (1.4343, -8.6898),
    "SCEDC_20":    (2.5410, -7.5342),
    "SCEDC_25":    (2.1394, -7.5380),
    "SCEDC_30":    (1.8403, -7.6206),
    "SaltonSea_10": (2.3320, -2.3151),
    "SanJac_10":   (1.1325, -5.3981),
    "WHITE_06":    (2.0211, -4.2611),
}

POISSON = {
    "ComCat_25":   (0.5126, -13.7745),
    "SCEDC_20":    (0.6431, -12.9165),
    "SCEDC_25":    (-0.1609, -12.9165),
    "SCEDC_30":    (-1.0644, -12.9165),
    "SaltonSea_10": (0.1244, -9.0241),
    "SanJac_10":   (0.7939, -9.2405),
    "WHITE_06":    (1.7146, -8.2941),
}

# Best published NEURAL spatial LL (max SLL across deep-stpp / autoint / neural_stpp seeds).
# Every value is worse (more negative) than the corresponding ETAS SLL.
BEST_NEURAL_SLL = {
    "ComCat_25":   -9.907,   # autoint
    "SCEDC_20":    -8.592,   # autoint
    "SCEDC_25":    -8.387,   # autoint
    "SCEDC_30":    -8.201,   # autoint
    "SaltonSea_10": -3.038,  # autoint
    "SanJac_10":   -6.389,   # autoint
    "WHITE_06":    -5.310,   # autoint
}


def summarize(dataset: str, tll: float, sll: float) -> str:
    et, es = ETAS.get(dataset, (float("nan"), float("nan")))
    bn = BEST_NEURAL_SLL.get(dataset, float("nan"))
    lines = [
        f"=== {dataset} : test per-event log-likelihood ===",
        f"  model      TLL = {tll:+.4f}   SLL = {sll:+.4f}",
        f"  ETAS       TLL = {et:+.4f}   SLL = {es:+.4f}   (40-yr incumbent)",
        f"  best NPP   ...           SLL = {bn:+.4f}   (published neural)",
    ]
    dsll = sll - es
    verdict_sll = "BEATS ETAS spatially" if dsll > 0 else "loses to ETAS spatially"
    lines.append(f"  -> SLL - ETAS = {dsll:+.4f}  ({verdict_sll})")
    lines.append(f"  -> SLL - bestNPP = {sll - bn:+.4f}")
    dtll = tll - et
    lines.append(f"  -> TLL - ETAS = {dtll:+.4f}  ({'>=' if dtll >= 0 else '<'} ETAS temporally)")
    return "\n".join(lines)
