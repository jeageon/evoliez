#!/usr/bin/env bash
# Paper-grade FRESH re-run: NADP net -3 (was +1) + s01 annotation + diffdock
# provenance + provenance stamp + report wording. Runs s01->s06b in a FRESH dir
# -> a NEW DB carrying the nullable docking_pose schema (no live-DB migration).
# The old +1 run must already be moved to _archive. s02 homolog + s03 MSA hit the
# content-keyed caches (sequence unchanged) so only the ligand-dependent stages
# s01/s04/s05/s06/s06b recompute with NADP(-3). ~5-6 h. Deploy ALL code first.
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
export PATH=/mnt/data/jglee/bin:/mnt/data/jglee/envs/diffdock/bin:/mnt/data/jglee/envs/boltz/bin:$PATH
export LD_LIBRARY_PATH=/mnt/data/jglee/envs/diffdock/lib
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export CUDA_VISIBLE_DEVICES=0,2,3          # 3 GPUs, leave GPU1 for others
export EVOLIEZ_NUM_THREADS=6
export EVOLIEZ_BOLTZ_MAX_PARALLEL_SAMPLES=8
export EVOLIEZ_BOLTZ_NUM_WORKERS=2
export EVOLIEZ_BOLTZ_PREPROCESSING_THREADS=4
export EVOLIEZ_DIFFDOCK_BATCH=40
export EVOLIEZ_DOCK_CPU=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
REPO=/mnt/data/jglee/EvoLiEZ
RD=/mnt/data/jglee/runs/fdh_5track
CFG=configs/target.local.5track.yaml
PYBIN=/mnt/data/jglee/miniforge3/envs/evoliez-cu124/bin/python
cd "$REPO"

echo ">> [1/3] preflight: NADP net -3 + annotation + multi_engine + per-rep + tools + FRESH dir"
[ -d "$RD" ] && { echo "   ABORT: $RD still exists -- archive it first"; exit 1; }
"$PYBIN" - <<PY
from evoliez.config import load_config
from rdkit import Chem
c = load_config("$CFG")
q = Chem.GetFormalCharge(Chem.MolFromSmiles(c.input.ligand.value))
assert q == -3, f"NADP net charge {q} != -3 -- ABORT"
assert c.interaction_model.multi_engine is True, "multi_engine off"
assert c.input.ec_number and c.input.catalytic_residues, "annotation missing"
import evoliez.stages.s06b_interaction_model as s
assert hasattr(s, "_run_per_rep_docking"), "per-rep code not deployed"
import evoliez.adapters.diffdock
print("   NADP net -3 | multi_engine on | EC", c.input.ec_number,
      "| catalytic", c.input.catalytic_residues, "| per-rep ok")
PY
command -v gnina >/dev/null && command -v boltz >/dev/null \
  && [ -x /mnt/data/jglee/envs/diffdock/bin/python ] \
  && echo "   gnina + boltz + diffdock ok" || { echo "   TOOL MISSING"; exit 1; }

TS=$(date +%Y%m%d-%H%M%S)
echo ">> [2/3] launch FRESH -3 run (new dir -> new nullable DB; s02/s03 cache-hit)"
LOG="/mnt/data/jglee/runs/fdh_5track_nadp3_$TS.log"; echo "$LOG" > /tmp/fdh_nadp3_logpath
nohup evoliez run -c "$CFG" --to s06b_interaction > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 12

echo ">> [3/3] early sanity (cache-hit / NADP / errors)"
grep -aE "\[run \]|\[done\]|\[skip\]|cache|hit|reuse|NADP|net charge|representative|homolog|MSA|multi_engine|Traceback|Error|AssertionError" "$LOG" | tail -18 || true
