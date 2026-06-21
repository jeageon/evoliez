#!/usr/bin/env bash
# Re-run ONLY s07 after the _ligandmpnn per-position-consensus fix. This is a
# CODE-ONLY change (no config edit) so the run fingerprint is UNCHANGED -> no
# fingerprint patch, no purge. We just drop s07_mutation_gen from completed_stages
# so --resume re-runs it; s01 load()s the new fixed, s06_graph recomputes the 40
# designable, s04 boltz-skips, s03/s05/s06b load() -> s07 re-runs with the fix.
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
export PATH=/mnt/data/jglee/envs/ligandmpnn/bin:/mnt/data/jglee/bin:/mnt/data/jglee/envs/boltz/bin:$PATH
export EVOLIEZ_LIGANDMPNN=/mnt/data/jglee/LigandMPNN
export CUDA_VISIBLE_DEVICES=0,2,3
export EVOLIEZ_ROOT=/mnt/data/jglee
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
RD=/mnt/data/jglee/runs/fdh_5track
CFG=configs/target.local.5track.yaml
PYBIN=/mnt/data/jglee/miniforge3/envs/evoliez-cu124/bin/python
cd /mnt/data/jglee/EvoLiEZ
TS=$(date +%Y%m%d-%H%M%S)
cp "$RD/_state.json" "$RD/_state.json.pre-s07rerun.$TS.bak"
"$PYBIN" - <<PY
import json, os
from pathlib import Path
RD=Path("$RD"); st=json.load(open(RD/"_state.json"))
before=list(st["completed_stages"])
st["completed_stages"]=[s for s in st["completed_stages"] if s != "s07_mutation_gen"]
tmp=RD/"_state.json.tmp"; tmp.write_text(json.dumps(st,indent=2)); os.replace(tmp,RD/"_state.json")
print("   dropped s07; completed:", before)
print("                       ->", st["completed_stages"])
PY
LOG="/mnt/data/jglee/runs/fdh_5track_s07rerun_$TS.log"; echo "$LOG" > /tmp/fdh_s07_logpath
nohup evoliez run -c "$CFG" --resume --to s07_mutation_gen > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 22
grep -qaE "purged stale|invalidating resume" "$LOG" && echo "   !!! PURGE -- ABORT" || echo "   no-purge OK"
grep -aE "ligandmpnn|generated [0-9]+ candidates|by generator|\[done\] s07|protected|Traceback|Error|AssertionError" "$LOG" | tail -18