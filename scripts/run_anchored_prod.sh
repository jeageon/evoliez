#!/usr/bin/env bash
# Anchored CLEAN production run (s01 -> s11) with the functional-state-anchored
# validation methodology: s10 builds the mutant from the WT REFERENCE complex
# (reference NADP/formate pose + ONLY the point mutation, limited relaxation) and
# gates the design pose against the WT (reference_like / alternative_pose), so the
# validation measures the mutation effect, not per-mutant Boltz pose-search noise.
# The Boltz mutant pose is recorded as an alternative-pose hypothesis. A gate-stack
# verdict (structural -> reference-pose -> functional-geometry -> energetic) is
# written per candidate. anchored_validation + pose_gate are config defaults (on).
#
# Mirrors run_s10_paper.sh's full tool env but runs the WHOLE pipeline (no --to) on
# a FRESH output dir, so the original fdh_5track run is preserved. Fingerprint guard
# only fires for an EXISTING run dir (a fresh dir has no purge risk).
#
# Usage:  bash scripts/run_anchored_prod.sh [CONFIG] [RUN_DIR]
#   CONFIG   default configs/target.local.anchored.yaml
#   RUN_DIR  default /mnt/data/jglee/runs/fdh_anchored   (must match config output_dir)
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
GUARD_PY="$CONDA_PREFIX/bin/python"                  # before PATH prepend shadows `python`

# --- Amber tier-3 + softcore-TI RBFE (AmberTools25 + pmemd.cuda) -----------------
AMBERTOOLS_ENV=/mnt/data/jglee/anaconda3/envs/AmberTools25
PMEMD_BUILD=/mnt/data/jglee/pmemd24_src/build
[ -f "$PMEMD_BUILD/pathscripts/amber.sh" ] && source "$PMEMD_BUILD/pathscripts/amber.sh"
export AMBERHOME=$AMBERTOOLS_ENV
export EVOLIEZ_AMBERTOOLS_BIN=$AMBERTOOLS_ENV/bin
export EVOLIEZ_PMEMD_CUDA=$PMEMD_BUILD/src/pmemd/src/pmemd.cuda_SPFP   # RBFE softcore TI

# --- per-tool isolated interpreters (s07/s08b/s09) + GPU -------------------------
DD=/mnt/data/jglee/envs/diffdock
export PATH=/mnt/data/jglee/bin:$DD/bin:/mnt/data/jglee/envs/boltz/bin:$AMBERTOOLS_ENV/bin:$PATH
export LD_LIBRARY_PATH=$DD/lib:$AMBERTOOLS_ENV/lib:${LD_LIBRARY_PATH:-}
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export EVOLIEZ_DIFFDOCK_PYTHON=$DD/bin/python
export EVOLIEZ_LIGANDMPNN=/mnt/data/jglee/LigandMPNN
export EVOLIEZ_LIGANDMPNN_PYTHON=/mnt/data/jglee/envs/ligandmpnn/bin/python
export EVOLIEZ_THERMOMPNN=/mnt/data/jglee/ThermoMPNN
export EVOLIEZ_THERMOMPNN_PYTHON=/mnt/data/jglee/miniforge3/envs/thermompnn/bin/python
export WANDB_MODE=disabled
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export EVOLIEZ_ROOT=/mnt/data/jglee
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2,3}            # leave GPU1 for others
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

CFG=${1:-configs/target.local.anchored.yaml}
RD=${2:-/mnt/data/jglee/runs/fdh_anchored}
cd /mnt/data/jglee/EvoLiEZ

echo "=== sanity: confirmatory tools ==="
for t in tleap antechamber parmchk2; do
  command -v "$t" >/dev/null 2>&1 && echo "  $t: OK" || echo "  !! $t MISSING"
done
[ -x "$EVOLIEZ_PMEMD_CUDA" ] && echo "  pmemd.cuda: OK ($EVOLIEZ_PMEMD_CUDA)" \
  || echo "  !! pmemd.cuda MISSING -> RBFE will skip (binding MD + NAC still run)"

echo "=== pre-flight: purge-safe fingerprint guard (existing dir only) ==="
if [ -f "$RD/_state.json" ] && ! "$GUARD_PY" scripts/check_run_fingerprint.py "$CFG" "$RD"; then
  echo "   !!! fingerprint MISMATCH on existing $RD -- aborting to avoid a purge."
  echo "   (config-only change: $GUARD_PY scripts/check_run_fingerprint.py $CFG $RD --patch)"
  exit 1
fi
[ -f "$RD/_state.json" ] && echo "   resuming existing run (fingerprint OK)" \
  || echo "   fresh run dir ($RD) -- no purge risk"

echo "=== launching anchored full pipeline (s01 -> s11), no purge ==="
TS=$(date +%Y%m%d-%H%M%S)
LOG="${RD}_anchored_prod_${TS}.log"; echo "$LOG" > /tmp/fdh_anchored_logpath
nohup evoliez run -c "$CFG" --resume > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 25
grep -qaE "purged stale|invalidating resume" "$LOG" && echo "   !!! PURGE -- ABORT" || echo "   no-purge OK"
grep -aE "\[run \]|\[done\]|MD shortlist|MD real-execution|MD NAC|gate-stack|RBFE|Traceback|Error" "$LOG" | tail -25
echo "   (follow live: tail -f $LOG)"
