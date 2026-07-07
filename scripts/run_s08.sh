#!/usr/bin/env bash
# Run s08 (reranker) on the fdh_5track run for paper data: score the 467 s07
# candidates with the RETRAINED (honest, CV 0.9826) s06b interaction model and
# select the redocking + MD shortlist. CODE-ONLY changes since s07 -> run
# fingerprint UNCHANGED -> no patch, no purge. s07 has no load() so it re-runs
# (full run() -> all ctx.put, resume-safe; ~15s, deterministic) which needs the
# ligandmpnn env; everything else loads/boltz-skips. Stops AT s08 so s08b/s09/s10
# stay separate (their GPU cost is assessed after seeing s08's shortlist sizes).
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
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
# Use the evoliez-cu124 python explicitly: line 12 prepends the ligandmpnn/boltz
# env bins to PATH (needed by the stage subprocess tools), which would otherwise
# shadow `python` with an env that lacks the evoliez module. The evoliez CLI is
# unaffected (its shebang pins its own interpreter).
if ! "$CONDA_PREFIX/bin/python" scripts/check_run_fingerprint.py "$CFG" "$RD"; then
  echo "   !!! fingerprint MISMATCH -- a bare --resume WOULD PURGE $RD."
  echo "   ABORTING. Investigate the CHG component; if it is only a version bump"
  echo "   (config/input unchanged), patch _state.json.fingerprint to current first."
  exit 1
fi
echo "=== fingerprint OK -- launching s08 (no purge) ==="
TS=$(date +%Y%m%d-%H%M%S)
LOG="/mnt/data/jglee/runs/fdh_5track_s08_$TS.log"; echo "$LOG" > /tmp/fdh_s08_logpath
nohup evoliez run -c "$CFG" --resume --to s08_reranker > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 22
grep -qaE "purged stale|invalidating resume" "$LOG" && echo "   !!! PURGE -- ABORT" || echo "   no-purge OK"
grep -aE "\[run \]|\[done\]|\[skip\]|reranker|ranked|redock|shortlist|for_md|for_redock|Traceback|Error|AssertionError" "$LOG" | tail -22