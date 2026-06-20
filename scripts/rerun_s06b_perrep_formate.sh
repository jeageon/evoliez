#!/usr/bin/env bash
# "완전체" s06b re-run: formate-co-folded reps + PER-REP multi-engine docking.
#
#   1. re-folds all ~150 representative ensembles WITH formate co-folded (so the
#      FAMILY consensus is multi-ligand-correct, not just the WT), reusing the
#      cached per-rep MSAs (structures/representatives/local_msa is kept).
#   2. then docks EACH rep (gnina + diffdock) and classifies every pose against
#      the family Boltz consensus -> per-rep weighted augmentation rows.
#
# Multi-GPU (CVD=0,2,3 -> dir-batch FRESH fold; old single-ligand reps are moved
# aside first so nothing is skip-reused). ~5-6 h. NOTE: run only AFTER the
# per-rep code (_run_per_rep_docking) is deployed + smoke-verified on a few reps.
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
# gnina + diffdock (docking) + boltz (rep fold) on PATH; script's own python via $PYBIN
export PATH=/mnt/data/jglee/bin:/mnt/data/jglee/envs/diffdock/bin:/mnt/data/jglee/envs/boltz/bin:$PATH
export LD_LIBRARY_PATH=/mnt/data/jglee/envs/diffdock/lib
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export CUDA_VISIBLE_DEVICES=0,2,3          # 3 GPUs (leave GPU1); >1 -> dir-batch fresh fold
export EVOLIEZ_NUM_THREADS=6               # keep CPU modest on the shared box
export EVOLIEZ_DOCK_CPU=4                  # per-gnina search threads (3 GPUs x 4 = 12 cores)
export OMP_NUM_THREADS=4                   # cap diffdock/torch CPU threads
export MKL_NUM_THREADS=4
REPO=/mnt/data/jglee/EvoLiEZ
RD=/mnt/data/jglee/runs/fdh_5track
CFG=configs/target.local.5track.yaml
PYBIN=/mnt/data/jglee/miniforge3/envs/evoliez-cu124/bin/python
REPS=$RD/structures/representatives
cd "$REPO"

echo ">> [1/5] preflight: multi_engine on + per-rep code deployed + tools"
"$PYBIN" - <<PY
from evoliez.config import load_config
c = load_config("$CFG")
assert c.interaction_model.multi_engine is True, "multi_engine not enabled in config!"
import evoliez.stages.s06b_interaction_model as s
assert hasattr(s, "_run_per_rep_docking"), "per-rep docking code NOT deployed!"
print("   multi_engine on + _run_per_rep_docking present")
PY
command -v gnina >/dev/null && command -v boltz >/dev/null \
  && [ -x /mnt/data/jglee/envs/diffdock/bin/python ] \
  && echo "   gnina + boltz + diffdock ok" || { echo "   TOOL MISSING"; exit 1; }

echo ">> [2/5] move old single-ligand rep structures aside (keep local_msa cache)"
TS=$(date +%s); BAK="$RD/structures/representatives_singlelig_bak_$TS"
mkdir -p "$BAK"; shopt -s nullglob
moved=0; for d in "$REPS"/hom_* "$REPS"/boltz_results_*; do [ -e "$d" ] && { mv "$d" "$BAK/"; moved=$((moved+1)); }; done
echo "   moved $moved old rep dir(s) -> $BAK ; local_msa kept: $([ -d "$REPS/local_msa" ] && echo yes || echo no)"

echo ">> [3/5] patch _state.json (fingerprint -> config; drop s06b so it re-runs)"
cp "$RD/_state.json" "$RD/_state.json.bak.perrep.$TS"
"$PYBIN" - <<PY
import json
from evoliez.config import load_config
from evoliez.context import RunContext
sf = "$RD/_state.json"
fp = RunContext(load_config("$CFG")).run_fingerprint()
d = json.load(open(sf)); before = list(d["completed_stages"])
d["fingerprint"] = fp
d["completed_stages"] = [s for s in before if s != "s06b_interaction"]
json.dump(d, open(sf, "w"))
print("   completed:", before, "->", d["completed_stages"])
PY

echo ">> [4/5] launch (--resume --to s06b_interaction, 3 GPUs, fresh formate fold + per-rep dock)"
LOG="$RD/s06b_perrep_$TS.log"; echo "$LOG" > /tmp/s06b_perrep_logpath
nohup evoliez run -c "$CFG" --resume --to s06b_interaction > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 10

echo ">> [5/5] early sanity"
grep -aE "\[run \]|\[done\]|\[skip\]|representative|extra ligand|co-fold|multi_engine|per.rep|invalidat|purge|Traceback" "$LOG" | tail -14 || true
