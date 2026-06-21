#!/usr/bin/env bash
# Re-run ONLY s05 after the gnina.redock min-RMSD-to-reference selection fix.
# gnina's CNN/affinity ranking put an end-for-end-FLIPPED NADP pose at rank-1
# (nicotinamide reactive end away from the catalytic site / formate); the fix
# selects the reference-consistent mode (gnina DID sample it, just ranked it low).
# CODE-ONLY change -> run fingerprint UNCHANGED -> no patch, no purge. Drop
# s05_docking so --resume re-docks the WT with the fix; STOP AT s05 (--to
# s05_docking) so s06_graph/s06b/s07 are untouched. s06b consumes redock_all (every
# mode, classified vs the family consensus) and is unaffected by this selection.
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
export PATH=/mnt/data/jglee/bin:/mnt/data/jglee/envs/diffdock/bin:/mnt/data/jglee/envs/boltz/bin:$PATH
export LD_LIBRARY_PATH=/mnt/data/jglee/envs/diffdock/lib
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export CUDA_VISIBLE_DEVICES=0,2,3          # leave GPU1 for other users
export EVOLIEZ_DOCK_CPU=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
RD=/mnt/data/jglee/runs/fdh_5track
CFG=configs/target.local.5track.yaml
PYBIN=/mnt/data/jglee/miniforge3/envs/evoliez-cu124/bin/python
cd /mnt/data/jglee/EvoLiEZ
TS=$(date +%Y%m%d-%H%M%S)
cp "$RD/_state.json" "$RD/_state.json.pre-s05rerun.$TS.bak"
"$PYBIN" - <<PY
import json, os
from pathlib import Path
RD=Path("$RD"); st=json.load(open(RD/"_state.json"))
before=list(st["completed_stages"])
st["completed_stages"]=[s for s in st["completed_stages"] if s != "s05_docking"]
tmp=RD/"_state.json.tmp"; tmp.write_text(json.dumps(st,indent=2)); os.replace(tmp,RD/"_state.json")
print("   dropped s05; completed:", before)
print("                       ->", st["completed_stages"])
PY
LOG="/mnt/data/jglee/runs/fdh_5track_s05rerun_$TS.log"; echo "$LOG" > /tmp/fdh_s05_logpath
nohup evoliez run -c "$CFG" --resume --to s05_docking > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 25
grep -qaE "purged stale|invalidating resume" "$LOG" && echo "   !!! PURGE -- ABORT" || echo "   no-purge OK"
grep -aE "reusing.*Boltz|\[run \]|\[done\]|\[skip\]|gnina|diffdock|WT reference docking|Traceback|Error|AssertionError" "$LOG" | tail -18