#!/usr/bin/env bash
# Resume the fdh_5track run --to s09_nonmd: non-MD validation of the candidates —
# redock each vs the Boltz reference with gnina + diffdock KEEPING the formate
# context (P1.1), per-GPU DiffDock batch + GNINA queue (P1.2), None-safe scores
# (P1.3), real-Boltz-Δ MD selection (P2.4), real-mutant mechanism (P2.6). Purge-safe.
#
# Multi-env resume: s09 redock uses DiffDock + gnina, but --resume ALSO re-runs s07
# (LigandMPNN) — a DIFFERENT bare-`python` conda env. Both adapters now honour
# explicit EVOLIEZ_{DIFFDOCK,LIGANDMPNN}_PYTHON so there is no PATH `python`
# collision; gnina is a binary; boltz (s08b per-stem skip) uses its own CLI. The
# docking env mirrors the proven scripts/rerun_s05.sh.
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
export EVOLIEZ_LIGANDMPNN_PYTHON=$LM/bin/python     # s07 re-run interpreter (no PATH clash)
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
  echo "   !!! fingerprint MISMATCH -- aborting to avoid purging $RD."; exit 1
fi
echo "=== fingerprint OK -- launching s09 (no purge) ==="
TS=$(date +%Y%m%d-%H%M%S)
LOG="/mnt/data/jglee/runs/fdh_5track_s09_$TS.log"; echo "$LOG" > /tmp/fdh_s09_logpath
nohup evoliez run -c "$CFG" --resume --to s09_nonmd > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 25
grep -qaE "purged stale|invalidating resume" "$LOG" && echo "   !!! PURGE -- ABORT" || echo "   no-purge OK"
grep -aE "\[run \]|\[done\]|\[skip\]|redock|gnina|diffdock|context|batch|queue|ligandmpnn|Traceback|Error|parse filetype" "$LOG" | tail -20