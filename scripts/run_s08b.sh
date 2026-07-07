#!/usr/bin/env bash
# Resume the fdh_5track run --to s08b_mutant_boltz: REAL mutant Boltz Δ for the
# top-40 candidates (by the s08 proxy), using the HARDENED per-GPU Boltz batch
# (P2.5 — one model-load per GPU). Purge-safe (guarded). s07/s08 re-run first
# (cheap, deterministic) since they have no load(); everything else loads/skips.
# Boltz GPU env (isolated /mnt/data/jglee/envs/boltz on PATH).
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
GUARD_PY="$CONDA_PREFIX/bin/python"        # capture before the PATH prepend shadows `python`
export PATH=/mnt/data/jglee/envs/ligandmpnn/bin:/mnt/data/jglee/bin:/mnt/data/jglee/envs/boltz/bin:$PATH
export EVOLIEZ_LIGANDMPNN=/mnt/data/jglee/LigandMPNN
export CUDA_VISIBLE_DEVICES=0,2,3          # leave GPU1 for other users
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
echo "=== fingerprint OK -- launching s08b (no purge) ==="
TS=$(date +%Y%m%d-%H%M%S)
LOG="/mnt/data/jglee/runs/fdh_5track_s08b_$TS.log"; echo "$LOG" > /tmp/fdh_s08b_logpath
nohup evoliez run -c "$CFG" --resume --to s08b_mutant_boltz > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 22
grep -qaE "purged stale|invalidating resume" "$LOG" && echo "   !!! PURGE -- ABORT" || echo "   no-purge OK"
grep -aE "\[run \]|\[done\]|\[skip\]|mutant|Boltz|boltz|batch|GPU|gpu|Traceback|Error|delta|Δ" "$LOG" | tail -18