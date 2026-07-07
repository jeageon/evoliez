#!/usr/bin/env bash
# Production paper-grade s10 MD launcher: resume --to s10_md with the full confirmatory
# stack — real-Boltz-structure MD shortlist (s09 require_real_structure), OpenMM implicit
# production at protocol_level 3 + the catalytic-power NAC screen (ΔNAC vs WT) + the
# alchemical RBFE ΔΔG_bind (softcore TI on Amber pmemd.cuda) on the top candidates.
# Purge-safe (fingerprint-guarded).
#
# s10 resume re-runs s07 (LigandMPNN), s08/s08b (Boltz), s09 (gnina+diffdock redock +
# ThermoMPNN), then s10 (OpenMM MD + RBFE), so the env carries EVERY tool's isolated
# interpreter PLUS the Amber/pmemd.cuda tier for RBFE.
#
# Usage:  bash scripts/run_s10_paper.sh [CONFIG] [RUN_DIR]
#   CONFIG   default configs/target.local.5track.yaml  (the fdh_5track run's config)
#   RUN_DIR  default /mnt/data/jglee/runs/fdh_5track
# For the explicit-solvent / multi-replica / MM-GBSA confirmatory pass on the very top
# picks, set md.engine: amber + md.solvent: explicit in the config and re-run.
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

# --- per-tool isolated interpreters (s07/s08b/s09 resume) + GPU -----------------
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

CFG=${1:-configs/target.local.5track.yaml}
RD=${2:-/mnt/data/jglee/runs/fdh_5track}
cd /mnt/data/jglee/EvoLiEZ

echo "=== sanity: paper-grade s10 confirmatory tools ==="
for t in tleap antechamber parmchk2; do
  command -v "$t" >/dev/null 2>&1 && echo "  $t: OK" || echo "  !! $t MISSING"
done
[ -x "$EVOLIEZ_PMEMD_CUDA" ] && echo "  pmemd.cuda: OK ($EVOLIEZ_PMEMD_CUDA)" \
  || echo "  !! pmemd.cuda MISSING -> RBFE will skip (binding MD + NAC still run)"

echo "=== pre-flight: purge-safe fingerprint guard ==="
if ! "$GUARD_PY" scripts/check_run_fingerprint.py "$CFG" "$RD"; then
  echo "   !!! fingerprint MISMATCH -- aborting to avoid purging $RD."
  echo "   (If you changed ONLY the config, run: $GUARD_PY scripts/check_run_fingerprint.py $CFG $RD --patch)"
  exit 1
fi
echo "=== fingerprint OK -- launching paper-grade s10 (no purge) ==="
TS=$(date +%Y%m%d-%H%M%S)
LOG="${RD}_s10_paper_${TS}.log"; echo "$LOG" > /tmp/fdh_s10_logpath
nohup evoliez run -c "$CFG" --resume --to s10_md > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 25
grep -qaE "purged stale|invalidating resume" "$LOG" && echo "   !!! PURGE -- ABORT" || echo "   no-purge OK"
grep -aE "\[run \]|\[done\]|MD shortlist|MD real-execution|MD NAC|RBFE|ΔΔG_bind|Traceback|Error" "$LOG" | tail -20
echo "   (follow live: tail -f $LOG)"
