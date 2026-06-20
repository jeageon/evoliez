#!/usr/bin/env bash
# Re-run fdh_5track s06b with the MULTI-ENGINE augmentation enabled.
#
# The Boltz consensus stays the positive teacher; gnina + DiffDock dock the WT
# ligand and each pose is classified against that consensus (weak-positive /
# hard-negative / excluded) and folded into training as a weighted, source-
# tagged row. This validates the real weak-pos/hard-neg split on REAL docking.
#
# FAST because it REUSES the cached representative ensembles
# (structures/representatives/hom_NNN, 150x15 poses already folded): we run on a
# SINGLE GPU so s06b takes the serial per-rep path (skip-if-complete reuse), not
# the multi-GPU dir-batch path (which writes fresh dirs and would re-fold ~3 h).
#
# Idempotent: re-running re-enables nothing already set and relaunches.
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
# gnina + diffdock (multi-engine docking) + boltz (rep skip-recompute require) on
# PATH. diffdock's python is first so `python -m inference` resolves to it; the
# script's OWN python calls use $PYBIN (the pipeline env) explicitly so they are
# not shadowed.
export PATH=/mnt/data/jglee/bin:/mnt/data/jglee/envs/diffdock/bin:/mnt/data/jglee/envs/boltz/bin:$PATH
export LD_LIBRARY_PATH=/mnt/data/jglee/envs/diffdock/lib
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export CUDA_VISIBLE_DEVICES=0                 # 1 GPU -> serial per-rep -> REUSE
REPO=/mnt/data/jglee/EvoLiEZ
RD=/mnt/data/jglee/runs/fdh_5track
CFG=configs/target.local.5track.yaml
PYBIN=/mnt/data/jglee/miniforge3/envs/evoliez-cu124/bin/python
cd "$REPO"

echo ">> [1/4] enable multi_engine in $CFG (idempotent)"
"$PYBIN" - <<PY
f = "$CFG"; txt = open(f).read()
if "multi_engine:" not in txt:
    out = []
    for ln in txt.splitlines():
        out.append(ln)
        if ln.strip().startswith("rep_msa_max_seqs:"):
            out += ["  multi_engine: true",
                    "  multi_engine_methods: [gnina, diffdock]"]
    open(f, "w").write("\n".join(out) + "\n"); print("   inserted multi_engine")
else:
    print("   multi_engine already present")
PY
"$PYBIN" - <<PY
from evoliez.config import load_config
c = load_config("$CFG")
assert c.interaction_model.multi_engine is True, "multi_engine not enabled!"
print("   multi_engine =", c.interaction_model.multi_engine,
      "| methods =", c.interaction_model.multi_engine_methods)
PY

echo ">> [2/4] patch _state.json (fingerprint -> new config; drop s06b so it re-runs)"
cp "$RD/_state.json" "$RD/_state.json.bak.multiengine.$(date +%s)"
"$PYBIN" - <<PY
import json
from evoliez.config import load_config
from evoliez.context import RunContext
sf = "$RD/_state.json"
fp = RunContext(load_config("$CFG")).run_fingerprint()   # DICT, no .setup() -> no purge
d = json.load(open(sf)); before = list(d["completed_stages"])
d["fingerprint"] = fp
d["completed_stages"] = [s for s in before if s != "s06b_interaction"]
json.dump(d, open(sf, "w"))
print("   completed:", before, "->", d["completed_stages"])
PY

echo ">> [3/4] preflight docking tools"
command -v gnina >/dev/null && echo "   gnina: $(command -v gnina)" || { echo "   gnina MISSING"; exit 1; }
[ -x /mnt/data/jglee/envs/diffdock/bin/python ] && echo "   diffdock python ok" || { echo "   diffdock MISSING"; exit 1; }

echo ">> [4/4] launch (--resume --to s06b_interaction, reuse cached reps)"
LOG="$RD/s06b_multiengine_$(date +%s).log"; echo "$LOG" > /tmp/s06b_me_logpath
nohup evoliez run -c "$CFG" --resume --to s06b_interaction > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 8
grep -aE "\[run \]|\[done\]|\[skip\]|representative|multi_engine|invalidat|purge|Traceback" "$LOG" | tail -12 || true
