#!/usr/bin/env bash
# =====================================================================
# CAR V5 server runbook launcher — ONE self-contained, idempotent script.
#   bash scripts/run_car_v5.sh preflight     # Day 1: lightweight V5 tests + config load + OpenMM
#   bash scripts/run_car_v5.sh smoke         # Day 2: Mg/OpenMM smoke (level1/0.05ns/top3) + verdict
#   bash scripts/run_car_v5.sh focused       # Day 5: 16-24 focused rerun (production tier) + verdict
#
# Server-only (needs the /mnt/data/jglee envs + GPUs). Mirrors car/run_car.sh env. Re-runs use
# --resume so a re-invocation does not repeat completed stages. Respects the shared-box watchdog
# (EVOLIEZ_NUM_THREADS<=8, GPU pool from CUDA_VISIBLE_DEVICES). NEVER runs before PR #1 is merged.
# =====================================================================
set -euo pipefail
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

MODE="${1:?usage: run_car_v5.sh <preflight|smoke|focused> [extra cli args...]}"; shift || true

CAR_ROOT="${CAR_ROOT:-/mnt/data/jglee/EvoLiEZ_car}"
EVO_PY="${EVO_PY:-/mnt/data/jglee/envs/evoliez/bin/python}"
export PYTHONPATH="$CAR_ROOT/src"

# --- storage / weights / external tool envs (adapters resolve these) ---
export EVOLIEZ_ROOT=/mnt/data/jglee
export EVOLIEZ_DATA_DIR=/mnt/data/jglee/evoliez_assets
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export EVOLIEZ_LIGANDMPNN=/mnt/data/jglee/LigandMPNN
export EVOLIEZ_LIGANDMPNN_PYTHON=/mnt/data/jglee/envs/ligandmpnn/bin/python
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export EVOLIEZ_DIFFDOCK_PYTHON=/mnt/data/jglee/envs/diffdock/bin/python
export EVOLIEZ_THERMOMPNN=/mnt/data/jglee/ThermoMPNN
export EVOLIEZ_THERMOMPNN_PYTHON=/mnt/data/jglee/miniforge3/envs/thermompnn/bin/python

# --- AmberTools (AM1-BCC) — MG ion parameters live in the Amber ion FF ---
AMBERTOOLS_ENV=/mnt/data/jglee/anaconda3/envs/AmberTools25
if [ -x "$AMBERTOOLS_ENV/bin/antechamber" ]; then
  export AMBERHOME="$AMBERTOOLS_ENV"
  export EVOLIEZ_AMBERTOOLS_BIN="$AMBERTOOLS_ENV/bin"
  export LD_LIBRARY_PATH="$AMBERTOOLS_ENV/lib:${LD_LIBRARY_PATH:-}"
  AMBER_PATH="$AMBERTOOLS_ENV/bin:"
else
  AMBER_PATH=""
fi
export PATH="/mnt/data/jglee/envs/evoliez/bin:/mnt/data/jglee/bin:/mnt/data/jglee/envs/boltz/bin:${AMBER_PATH}$PATH"

# --- shared-box safety: cap CPU threads, GPU pool from the environment ---
export EVOLIEZ_NUM_THREADS="${EVOLIEZ_NUM_THREADS:-8}"
export OMP_NUM_THREADS="$EVOLIEZ_NUM_THREADS"; export MKL_NUM_THREADS="$EVOLIEZ_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$EVOLIEZ_NUM_THREADS"; export NUMEXPR_NUM_THREADS="$EVOLIEZ_NUM_THREADS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

cd "$CAR_ROOT"
echo ">> mode=$MODE  code=$($EVO_PY -c 'import evoliez,os;print(os.path.dirname(evoliez.__file__))')"
echo ">> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  threads=$EVOLIEZ_NUM_THREADS"

_provenance_verdict () {  # $1 = run output_dir
  echo ">> acceptance verdict (evidence-validity, NOT lead-hunting):"
  "$EVO_PY" "$CAR_ROOT/scripts/check_car_v5_provenance.py" "$1" || {
    echo "!! provenance verdict = FAIL (see reasons above); do NOT proceed to the next tier"; return 1; }
}

case "$MODE" in
  preflight)
    echo ">> Day 1 preflight: lightweight V5 tests + config load + OpenMM platform"
    "$EVO_PY" -m pytest \
      tests/test_v5_car_config.py tests/test_v5_metal_placement.py tests/test_v5_nac_angle.py \
      tests/test_v5_claim_integrity.py tests/test_v5_s08_lanes.py tests/test_claim_guard.py -q
    "$EVO_PY" - <<'PY'
from evoliez.config import load_config
from evoliez.mechanism.mode import mechanism_mode
for name in ("car_srcar_3hp_v5.yaml", "car_srcar_3hp_v5_smoke.yaml", "prod_fdh_nadp.yaml"):
    c = load_config(f"configs/{name}")
    print(f"  {name}: mechanism_mode={mechanism_mode(c)}")
try:
    import openmm  # noqa: F401
    from openmm import Platform
    print("  OpenMM platforms:", [Platform.getPlatform(i).getName()
                                   for i in range(Platform.getNumPlatforms())])
except Exception as e:  # noqa: BLE001
    print("  !! OpenMM import/platform failed:", e); raise
PY
    echo ">> preflight OK";;

  smoke)
    CONFIG="${CAR_CONFIG:-configs/car_srcar_3hp_v5_smoke.yaml}"
    OUT="/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_v5_smoke"
    echo ">> Day 2 Mg/OpenMM smoke  config=$CONFIG  out=$OUT"
    "$EVO_PY" -m evoliez.cli run -c "$CONFIG" --resume "$@"
    _provenance_verdict "$OUT";;

  focused)
    CONFIG="${CAR_CONFIG:-configs/car_srcar_3hp_v5.yaml}"
    OUT="/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_v5"
    echo ">> Day 5 focused rerun  config=$CONFIG  out=$OUT"
    "$EVO_PY" -m evoliez.cli run -c "$CONFIG" --resume "$@"
    _provenance_verdict "$OUT";;

  *) echo "unknown mode: $MODE (use preflight|smoke|focused)"; exit 2;;
esac
