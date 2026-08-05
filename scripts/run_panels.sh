#!/usr/bin/env bash
# Gate G3 across all four panels, then the pooled figure.
#
# Ordered lead-panel-first ON PURPOSE. WHITE is the only panel with both a
# 1.5-decade grid and a well-distributed target set (132 targets, 45 of 56
# windows non-empty, most-populated window 11%), so it is the one that decides
# whether the multi-region design works at all. If WHITE comes back flat, the
# other three are unlikely to rescue it and gate G3 says publish the null
# rather than keep spending. Read WHITE before launching the rest.
#
# BIN SIZE VARIES BY PANEL and is not cosmetic. Each value is chosen to give
# ~6 active grid cells per target event, matching the ComCat frame, so the
# panels' shape scores are on comparable footing before they are pooled. The
# per-panel value is fixed for every point on that panel's curve and for its
# ETAS control; see the header of each config.
#
# HORIZON is 1 DAY, not the 30 the design originally specified. Measured
# lag-1 correlation of target counts between consecutive windows -- the
# between-window signal any forecaster must capture:
#
#   panel            1d       3d       7d      14d      30d
#   WHITE          +.116*   +.107*   +.189*   +.001    -.133
#   QTM SanJac     +.066*   +.090    -.059    -.081    -.062
#   QTM SaltonSea  +.252*   +.244*   +.074    +.002    +.196
#   ComCat M>=4    +.097*   +.021    +.131*   +.162*   +.004     (* p<0.05)
#
# 1 day is significant in ALL FOUR panels; 30 days in none. Sequences here
# start and finish inside a 30-day window, so their burstiness never crosses a
# window boundary -- WHITE is 2.90x overdispersed at 30 days yet its
# between-window correlation is -0.133. Confirmed on real forecasts: at 30 days
# a fitted ETAS scored corr = -0.084 and so did persistence (-0.150). Set
# HORIZON=3 for the robustness check. See MOONSHOT.md invariant 1h.
#
# MEMORY. Concurrency 2 and a 5 GB/worker floor, not because the tooling needs
# it but because this machine has been driven into swap once already this
# session (0 GB free, 34/41 GB swap, killed at 12.79 GB RSS). The dense panels
# are small enough that 2 workers is not the bottleneck.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CONC=${CONC:-2}
MEM=${MEM:-5.0}
SEEDS=${SEEDS:-0}
STEPS=${STEPS:-12000}
NSIMS=${NSIMS:-200}

run_panel () {
  local name=$1 cfg=$2 out=$3 bin=$4 mtgt=$5 mlarge=$6; shift 6
  echo
  echo "======================================================================"
  echo "[panel] $name   grid: $*   bin=${bin}km  M_TARGET=${mtgt}"
  echo "======================================================================"
  $PY scripts/scaling_curve.py \
    --base "$cfg" --out "$out" --mc "$@" \
    --arms matched_window matched_n --seeds $SEEDS \
    --m-target "$mtgt" --m-large "$mlarge" \
    --horizon "${HORIZON:-1}" --bin-km "$bin" --tail-mode fixed \
    --n-sims $NSIMS --steps $STEPS \
    --concurrency $CONC --mem-per-worker $MEM --device cpu
}

# lead panel first — read this before committing to the rest
run_panel "WHITE (San Jacinto)"  configs/panel_white.yaml \
          runs/panel_white          2.0 3.0 4.0  2.5 2.0 1.5 1.0

if [[ "${LEAD_ONLY:-0}" == "1" ]]; then
  echo; echo "LEAD_ONLY=1 — stopping after WHITE. Inspect runs/panel_white/curve.json,"
  echo "then rerun without LEAD_ONLY to do the remaining three panels."
  exit 0
fi

run_panel "QTM San Jacinto"      configs/panel_qtm_sanjac.yaml \
          runs/panel_qtm_sanjac     5.0 3.0 4.0  2.3 1.8 1.3
run_panel "QTM Salton Sea"       configs/panel_qtm_saltonsea.yaml \
          runs/panel_qtm_saltonsea  3.0 3.0 4.0  2.5 2.1 1.7
run_panel "ComCat California"    configs/moonshot_lowmc.yaml \
          runs/panel_comcat        15.0 4.0 5.0  3.5 3.0 2.8

echo
echo "======================================================================"
echo "[pooled figure]"
echo "======================================================================"
for arm in matched_window matched_n; do
  $PY scripts/make_pooled_figure.py \
    --panel "San Jacinto (WHITE)=runs/panel_white" \
    --panel "San Jacinto (QTM)=runs/panel_qtm_sanjac" \
    --panel "Salton Sea (QTM)=runs/panel_qtm_saltonsea" \
    --panel "California (ComCat)=runs/panel_comcat" \
    --arm "$arm" --metric shape \
    --out "figures/moonshot_pooled_${arm}.png"
done

echo
echo "NEXT: the curve alone is not gate G3. Run the ETAS control at every mc"
echo "on the SAME frames (scripts/etas_by_mc.py) — a curve without its"
echo "re-inverted baseline cannot separate 'the model learns more' from"
echo "'the catalog contains more' (MOONSHOT.md invariant 3)."
