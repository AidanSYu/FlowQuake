#!/usr/bin/env bash
# Bootstrap a fresh cloud box and run the MOONSHOT.md gates end to end.
#
# Designed for a bare Ubuntu instance (Vultr / any provider). It is CPU-first by
# deliberate choice: the measured profile says this workload is dispatch-bound,
# not FLOP-bound (MPS was 1.6x SLOWER than CPU at production model size; 12
# threads on one run buys 2.24x while six 1-thread processes buy 3.75x). A
# high-core CPU instance is therefore better value than a GPU for the sweep, and
# far cheaper. Set FQ_DEVICE=cuda only if you have profiled otherwise.
#
#   COST GUARD: the box self-destructs after FQ_MAX_HOURS (default 12) whatever
#   happens, so a hung job cannot silently drain the credit. Results are synced
#   out first. Set FQ_MAX_HOURS=0 to disable (not recommended).
#
# Usage:
#   scp scripts/bootstrap_remote.sh root@<ip>:/root/
#   ssh root@<ip> 'FQ_STAGE=pilot bash bootstrap_remote.sh'
#
# Stages:
#   catalog  fetch the low-mc ComCat and report per-era completeness
#   pilot    gate G3 — 3 mc points, 1 seed. The kill switch. ~1-2 h.
#   gates    G1 (ETAS smoothed background) + G2 (encoder controls)
#   curve    gate G4 — the full sweep. Only after G3 passes.
set -euo pipefail

FQ_REPO="${FQ_REPO:-https://github.com/AidanSYu/FlowQuake.git}"
FQ_BRANCH="${FQ_BRANCH:-main}"
FQ_STAGE="${FQ_STAGE:-catalog}"
FQ_DEVICE="${FQ_DEVICE:-cpu}"
FQ_MAX_HOURS="${FQ_MAX_HOURS:-12}"
FQ_MIN_MAG="${FQ_MIN_MAG:-1.0}"
FQ_WORK="${FQ_WORK:-/root/flowquake}"
FQ_CONC="${FQ_CONC:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- cost guard
if [ "${FQ_MAX_HOURS}" != "0" ]; then
  log "cost guard: powering off in ${FQ_MAX_HOURS}h regardless of progress"
  nohup bash -c "sleep $(( FQ_MAX_HOURS * 3600 )); \
    echo 'COST GUARD FIRED' >> ${FQ_WORK}/GUARD.log; \
    poweroff" >/dev/null 2>&1 &
fi

# ---------------------------------------------------------------- provisioning
if [ ! -d "${FQ_WORK}/.git" ]; then
  log "installing system packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git curl build-essential python3 python3-venv >/dev/null
  log "cloning ${FQ_REPO} (${FQ_BRANCH})"
  git clone --depth 1 -b "${FQ_BRANCH}" "${FQ_REPO}" "${FQ_WORK}"
fi
cd "${FQ_WORK}"

if [ ! -d .venv ]; then
  log "creating venv (uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.11 .venv
  uv pip install -e ".[dev]" --python .venv/bin/python
fi
PY=.venv/bin/python
log "python: $(${PY} --version) | cores: $(nproc) | concurrency: ${FQ_CONC}"

log "running the test suite before anything expensive"
${PY} -m pytest tests/ -q -m "not slow" || { log "TESTS FAILED — stopping"; exit 1; }

# Pin BLAS threads: measured 0.3x throughput at 16-way when left unset, because
# every process tries to use every core. See scaling_curve.train_many.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.

CAT_DIR=reference/Datasets/ComCat_lowmc
CAT=${CAT_DIR}/ComCat_lowmc_catalog.csv

stage_catalog() {
  log "STAGE catalog: USGS ComCat M>=${FQ_MIN_MAG} over the RELM region"
  if [ ! -d reference/Datasets/ComCat ]; then
    log "fetching the benchmark clone (for california_shape.npy)"
    git clone --depth 1 https://github.com/ss15859/EarthquakeNPP.git reference || \
      log "WARNING: benchmark clone failed; falling back to the bbox (NOT the benchmark region)"
  fi
  ${PY} scripts/build_comcat_lowmc.py --min-mag "${FQ_MIN_MAG}" \
      --out "${CAT_DIR}" --completeness-report \
      --verify-against reference/Datasets/ComCat/ComCat_catalog.csv 2>&1 | tee ${CAT_DIR}.log
  log "catalog written; READ THE COMPLETENESS TABLE before choosing mc points."
  log "A point at mc X is only meaningful in eras whose Mc <= X."
}

# The pilot IS gate G3, the honest kill switch: if the slope is flat and tight
# here, MOONSHOT.md says stop and publish the null rather than scale up.
stage_pilot() {
  log "STAGE pilot: gate G3"
  ${PY} scripts/scaling_curve.py --base configs/moonshot_lowmc.yaml \
      --out runs/scaling_pilot --mc 2.5 1.5 1.0 --arms matched_window matched_n \
      --seeds 0 --m-target 4.0 --m-large 6.0 --horizon 30 --bin-km 10 \
      --n-sims 200 --tail-mode fixed --device "${FQ_DEVICE}" \
      --concurrency "${FQ_CONC}" 2>&1 | tee runs/scaling_pilot.log
  log "then score the ETAS control on the SAME frame (invariant 3):"
  ${PY} scripts/etas_by_mc.py --base configs/moonshot_lowmc.yaml \
      --frame runs/scaling_pilot/frame.json --out runs/etas_by_mc_pilot \
      --mc 2.5 1.5 1.0 --background uniform smoothed --n-sims 200 \
      --concurrency "${FQ_CONC}" 2>&1 | tee runs/etas_by_mc_pilot.log
}

stage_gates() {
  log "STAGE gates: G1 (ETAS smoothed background) + G2 (encoder controls)"
  ${PY} scripts/ablation_h_controls.py --base configs/n1_density.yaml \
      --out runs/ablation_h_controls --h 0 4 16 --arms full safe augmented \
      --device "${FQ_DEVICE}" --concurrency "${FQ_CONC}" 2>&1 | tee runs/g2.log
}

stage_curve() {
  if [ ! -f runs/scaling_pilot/curve.json ]; then
    log "REFUSING: gate G3 has not run. MOONSHOT.md requires the pilot first."
    exit 1
  fi
  log "STAGE curve: gate G4 — the full sweep"
  ${PY} scripts/scaling_curve.py --base configs/moonshot_lowmc.yaml \
      --out runs/scaling_curve/california --mc 4.0 3.0 2.5 2.0 1.5 1.0 \
      --arms matched_window matched_n --seeds 0 1 2 \
      --m-target 4.0 --m-large 6.0 --horizon 30 --bin-km 10 --n-sims 500 \
      --tail-mode fixed --device "${FQ_DEVICE}" --concurrency "${FQ_CONC}" \
      2>&1 | tee runs/scaling_curve.log
}

case "${FQ_STAGE}" in
  catalog) stage_catalog ;;
  pilot)   [ -f "${CAT}" ] || stage_catalog; stage_pilot ;;
  gates)   stage_gates ;;
  curve)   stage_curve ;;
  all)     stage_catalog; stage_pilot; stage_gates ;;
  *)       log "unknown FQ_STAGE=${FQ_STAGE}"; exit 1 ;;
esac

log "packaging results (checkpoints and per-event CSVs excluded by .gitignore)"
tar czf /root/flowquake_results.tgz runs/**/*.json runs/*.log 2>/dev/null || true
log "DONE. Pull with:  scp root@<ip>:/root/flowquake_results.tgz ."
log "Destroy the instance when you have the tarball — it bills while it exists."
