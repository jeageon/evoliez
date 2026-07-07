#!/usr/bin/env bash
# Resume the fdh_5track run --to s09_nonmd: non-MD validation — redock each candidate
# vs the Boltz reference with gnina + diffdock KEEPING the formate context (P1.1),
# per-GPU DiffDock batch + GNINA queue (P1.2), None-safe scores (P1.3), real-Boltz-Δ
# MD selection (P2.4), real-mutant mechanism (P2.6), and ThermoMPNN ΔΔG stability
# (open-source FoldX replacement). Purge-safe (guarded).
#
# Multi-env resume: s09 redock uses DiffDock + gnina; --resume also re-runs s07
# (LigandMPNN) and s08b (Boltz, per-stem skip), and s09 stability shells out to
# ThermoMPNN — four different conda envs. Each tool uses its OWN explicit interpreter
# (EVOLIEZ_*_PYTHON) so there is no PATH `python` collision.
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
GUARD_PY="$CONDA_PREFIX/bin/python"                 # before PATH prepend shadows `python`
DD=/mnt/data/jglee/envs/diffdock
LM=/mnt/data/jglee/envs/ligandmpnn
export PATH=/mnt/data/jglee/bin:$DD/bin:/mnt/data/jglee/envs/boltz/bin:$PATH
export LD_LIBRARY_PATH=$DD/lib                      # diffdock subprocess libstdc++ (rerun_s05 proven)
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export EVOLIEZ_DIFFDOCK_PYTHON=$DD/bin/python       # s09 redock interpreter
export EVOLIEZ_LIGANDMPNN=/mnt/data/jglee/LigandMPNN
export EVOLIEZ_LIGANDMPNN_PYTHON=$LM/bin/python     # s07 re-run interpreter
export EVOLIEZ_THERMOMPNN=/mnt/data/jglee/ThermoMPNN
export EVOLIEZ_THERMOMPNN_PYTHON=/mnt/data/jglee/miniforge3/envs/thermompnn/bin/python  # s09 stability ΔΔG
export WANDB_MODE=disabled                          # ThermoMPNN import pulls wandb; never phone home
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export CUDA_VISIBLE_DEVICES=0,2,3                   # leave GPU1 for other users
export EVOLIEZ_DOCK_CPU=4
export EVOLIEZ_ROOT=/mnt/data/jglee
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
RD=/mnt/data/jglee/runs/fdh_5track
CFG=configs/target.local.5track.yaml
cd /mnt/data/jglee/EvoLiEZ
echo "=== pre-flight: purge-safe fingerprint guard ==="
if ! "$GUARD_PY" scripts/check_run_fingerprint.py "$CFG" "$RD"; then
  echo "   !!! fingerprint MISMATCH -- aborting to avoid purging $RD."
  echo "   (If you changed only validation.stability.method, run check_run_fingerprint.py --patch first.)"
  exit 1
fi
echo "=== fingerprint OK -- launching s09 (no purge) ==="
TS=$(date +%Y%m%d-%H%M%S)
LOG="/mnt/data/jglee/runs/fdh_5track_s09_$TS.log"; echo "$LOG" > /tmp/fdh_s09_logpath
nohup evoliez run -c "$CFG" --resume --to s09_nonmd > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 25
grep -qaE "purged stale|invalidating resume" "$LOG" && echo "   !!! PURGE -- ABORT" || echo "   no-purge OK"
grep -aE "\[run \]|\[done\]|\[skip\]|redock|gnina|diffdock|thermompnn|stability|context|batch|Traceback|Error|parse filetype" "$LOG" | tail -20