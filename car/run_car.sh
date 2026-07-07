#!/usr/bin/env bash
# =====================================================================
# SrCAR / 3-HP production launcher — ISOLATED deploy at /mnt/data/jglee/EvoLiEZ_car
# Runs MY (local) code via PYTHONPATH override; the main server repo + its WIP are
# left untouched. All real backends (Boltz, gnina, DiffDock, LigandMPNN, ThermoMPNN,
# AmberTools) wired to their server envs. GPU-first, CPU-bounded (shared box).
#
#   bash car/run_car.sh doctor
#   bash car/run_car.sh run --to s04_complex
#   bash car/run_car.sh run --from s05_docking --to s09_nonmd_validation --resume
# =====================================================================
set -euo pipefail

# s10 fans MD across GPUs with many subprocess pipes + OpenMM open files; the default
# 1024 fd limit overflows ("OSError: Too many open files"). Raise it (best-effort).
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

CAR_ROOT=/mnt/data/jglee/EvoLiEZ_car
CONFIG="${CAR_CONFIG:-configs/car_srcar_3hp.yaml}"
EVO_PY=/mnt/data/jglee/envs/evoliez/bin/python

# --- my code wins over the env's editable install of the main repo ---
export PYTHONPATH="$CAR_ROOT/src"

# --- storage / weights ---
export EVOLIEZ_ROOT=/mnt/data/jglee
export EVOLIEZ_DATA_DIR=/mnt/data/jglee/evoliez_assets
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache

# --- external tool envs (adapters resolve these) ---
export EVOLIEZ_LIGANDMPNN=/mnt/data/jglee/LigandMPNN
export EVOLIEZ_LIGANDMPNN_PYTHON=/mnt/data/jglee/envs/ligandmpnn/bin/python
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export EVOLIEZ_DIFFDOCK_PYTHON=/mnt/data/jglee/envs/diffdock/bin/python
export EVOLIEZ_DIFFDOCK_BATCH=40
export EVOLIEZ_THERMOMPNN=/mnt/data/jglee/ThermoMPNN
export EVOLIEZ_THERMOMPNN_PYTHON=/mnt/data/jglee/miniforge3/envs/thermompnn/bin/python

# --- s10 MD: AmberTools (AM1-BCC via antechamber/sqm) + pmemd.cuda (Amber tier) ---
AMBERTOOLS_ENV=/mnt/data/jglee/anaconda3/envs/AmberTools25
if [ -x "$AMBERTOOLS_ENV/bin/antechamber" ]; then
  export AMBERHOME="$AMBERTOOLS_ENV"
  export EVOLIEZ_AMBERTOOLS_BIN="$AMBERTOOLS_ENV/bin"
  export EVOLIEZ_PMEMD_CUDA="${EVOLIEZ_PMEMD_CUDA:-/mnt/data/jglee/pmemd24_src/build/src/pmemd/src/pmemd.cuda_SPFP}"
  # antechamber/sqm need AmberTools libs on LD_LIBRARY_PATH (else s10 AM1-BCC fails silently)
  export LD_LIBRARY_PATH="$AMBERTOOLS_ENV/lib:${LD_LIBRARY_PATH:-}"
  AMBER_PATH="$AMBERTOOLS_ENV/bin:"
else
  AMBER_PATH=""
fi

# --- PATH: evoliez env (mmseqs/mafft/foldseek/obabel/blastp), gnina
#     (/mnt/data/jglee/bin), boltz env, ambertools. evoliez bin FIRST so bare
#     `python` + the MSA tools resolve to the orchestrator env. ---
export PATH="/mnt/data/jglee/envs/evoliez/bin:/mnt/data/jglee/bin:/mnt/data/jglee/envs/boltz/bin:${AMBER_PATH}$PATH"

# --- CPU thread caps (shared 48-core box; watchdog throttles >~24 cores/10min) ---
export EVOLIEZ_NUM_THREADS="${EVOLIEZ_NUM_THREADS:-8}"
export OMP_NUM_THREADS="$EVOLIEZ_NUM_THREADS"
export MKL_NUM_THREADS="$EVOLIEZ_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$EVOLIEZ_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$EVOLIEZ_NUM_THREADS"
export NUMEXPR_MAX_THREADS="$EVOLIEZ_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$EVOLIEZ_NUM_THREADS"

# --- GPUs: all 4 A6000 free -> expose the pool; config.compute.gpu_pool fans out ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# --- Boltz speed-only knobs (do not change output) ---
export EVOLIEZ_BOLTZ_MAX_PARALLEL_SAMPLES="${EVOLIEZ_BOLTZ_MAX_PARALLEL_SAMPLES:-8}"
export EVOLIEZ_BOLTZ_NUM_WORKERS="${EVOLIEZ_BOLTZ_NUM_WORKERS:-2}"
export EVOLIEZ_BOLTZ_PREPROCESSING_THREADS="${EVOLIEZ_BOLTZ_PREPROCESSING_THREADS:-4}"

cd "$CAR_ROOT"
echo ">> code: $($EVO_PY -c 'import evoliez,os;print(os.path.dirname(evoliez.__file__))')"
echo ">> config=$CONFIG  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  threads=$EVOLIEZ_NUM_THREADS"
SUB="${1:?usage: run_car.sh <doctor|run> [args...]}"; shift || true
exec "$EVO_PY" -m evoliez.cli "$SUB" -c "$CONFIG" "$@"
