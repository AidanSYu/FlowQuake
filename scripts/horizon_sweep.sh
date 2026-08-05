#!/usr/bin/env bash
# Where does the probe actually have forecast skill?
#
# RV5 on WHITE at a 30-day horizon returned corr(n_expected, n_observed) of
# -0.06 to -0.10 across mc, every 95% interval spanning zero (p = 0.47-0.67,
# n = 55 windows). That is not anti-correlation, it is NO detectable skill --
# and invariant 1f says a run with no skill on the informative arm certifies
# nothing, whatever its verdict line says.
#
# Two candidate explanations, and they call for different fixes:
#
#   HORIZON   ETAS skill is concentrated at short lags because Omori decay is
#             steep. A 30-day window integrates over most of the decay, so the
#             timing signal is averaged away. If so, shorter horizons show
#             skill and the design should use one.
#   POWER     55 windows and 132 targets is simply too little to resolve a
#             correlation; the standard error alone is ~1/sqrt(55) = 0.135.
#             Shorter horizons also help here, by giving more windows.
#
# The two are not exclusive and this sweep separates them: if corr rises
# sharply as the horizon shortens, it is the first; if it stays near zero but
# the intervals tighten, it is the second.
#
# The ETAS fit is horizon-INDEPENDENT, so most of this cost is re-scoring.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MC=${MC:-1.5}
BIN=${BIN:-2.0}

for H in 1 3 7 14 30; do
  OUT="runs/horizon_sweep/h${H}"
  echo
  echo "=================================================================="
  echo "[horizon] ${H} days"
  echo "=================================================================="
  $PY scripts/prep_real_validation.py \
      --out "$OUT" --horizon "$H" --bin-km "$BIN" 2>&1 | \
      grep -E "clustering|frame|ABORT|target set"
  $PY scripts/validate_with_etas.py \
      --root "$OUT" --arms informative --mc $MC \
      --out "$OUT/etas" --n-sims 100 --n-iter 4 --concurrency 1 2>&1 | \
      grep -E "corr|slope|PROBE CHECK" || true
done

echo
echo "=================================================================="
$PY - <<'PYEOF'
import glob, json, numpy as np
from scipy import stats
print(f"{'horizon':>9}{'windows':>9}{'targets':>9}{'corr':>9}{'p':>8}{'95% CI':>22}")
for h in (1, 3, 7, 14, 30):
    fs = sorted(glob.glob(f"runs/horizon_sweep/h{h}/etas/informative/*/target_process.json"))
    if not fs:
        continue
    d = json.load(open(fs[0])); w = d["windows"]
    ne = np.array([x["n_expected"] for x in w])
    no = np.array([float(x["n_target_obs"]) for x in w])
    n = len(w)
    if ne.std() == 0 or n < 5:
        continue
    r, p = stats.pearsonr(ne, no)
    z, se = np.arctanh(r), 1 / np.sqrt(max(n - 3, 1))
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"{h:>9}{n:>9}{int(no.sum()):>9}{r:>9.3f}{p:>8.3f}"
          f"   [{lo:+.3f}, {hi:+.3f}]")
print()
print("invariant 1f: the informative arm needs corr > 0.20 before any")
print("verdict on the null is readable. Pick the horizon from THIS table,")
print("not from convention.")
PYEOF
