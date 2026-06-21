#!/usr/bin/env bash
# s07 1st-pass mutation generation on the existing fdh_5track run (s01-s06b done).
#
# The config now protects P103,G127,G129,G341 (-> fixed_residues, 6 total) and adds
# accession/pdb_id/numbering, so config_sha1 changed (09105fe8 -> f3ffb81a) while
# EVERY other run-fingerprint field is byte-identical (verified read-only). A bare
# --resume would see the changed fingerprint and _purge_stale_outputs() = rmtree
# complexes/ docking/ interaction_graphs/ ml_datasets/  (the ~4 h paper data!). So:
#   * PATCH _state.json's stored fingerprint to the new config's  -> stored==current
#     -> NO purge.
#   * drop ONLY s01_input from completed_stages so it re-runs with the new
#     fixed_residues (s06_graph has no load() and re-runs regardless; dropped too
#     for clarity).
# On --resume:  s01 re-runs (new fixed)   s02 re-runs (homolog content-cache ~1 s)
#   s03 load()   s04 re-runs but boltz REUSES the existing complex (skip-recompute,
#   NO re-fold)  s05 load()   s06_graph re-runs (new designable)   s06b load()
#   -> s07 runs.  No expensive recompute.
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
# ligandmpnn/bin FIRST so the s07 LigandMPNN adapter's `python run.py` subprocess
# resolves to the ligandmpnn env; `evoliez` itself still resolves to evoliez-cu124.
export PATH=/mnt/data/jglee/envs/ligandmpnn/bin:/mnt/data/jglee/bin:/mnt/data/jglee/envs/boltz/bin:$PATH
export EVOLIEZ_LIGANDMPNN=/mnt/data/jglee/LigandMPNN
export CUDA_VISIBLE_DEVICES=0,2,3          # leave GPU1 for other users
export EVOLIEZ_ROOT=/mnt/data/jglee
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
REPO=/mnt/data/jglee/EvoLiEZ
RD=/mnt/data/jglee/runs/fdh_5track
CFG=configs/target.local.5track.yaml
PYBIN=/mnt/data/jglee/miniforge3/envs/evoliez-cu124/bin/python
cd "$REPO"

echo ">> [1/4] preflight: config sanity + fingerprint diff == [config_sha1] + ligandmpnn"
"$PYBIN" - <<PY
import json, hashlib
from pathlib import Path
from evoliez.config import load_config
from evoliez import __version__
from evoliez.io.provenance import RANKING_FORMULA_VERSION
from rdkit import Chem
def _s(t): return hashlib.sha1(t.encode()).hexdigest()[:16]
RD=Path("$RD"); c=load_config("$CFG"); ci=c.input
assert Chem.GetFormalCharge(Chem.MolFromSmiles(ci.ligand.value))==-3, "NADP not -3"
for r in ("P103","G127","G129","G341","Q227","H229"):
    assert r in ci.fixed_residues, "%s not protected"%r
assert ci.accession=="Q9S7E4", "accession missing"
seq=ci.target_sequence.strip().upper()
ib="%s|%s|%s"%(seq,ci.ligand.type,ci.ligand.value)
cb=json.dumps(c.model_dump(mode="json"),sort_keys=True,default=str)
g="none"
if c.gnn.enabled:
    from evoliez.io.provenance import _sha256_file
    g=_sha256_file(str(RD/c.gnn.checkpoint)) or "missing"
fp={"evoliez_version":__version__,"ranking_formula_version":RANKING_FORMULA_VERSION,
    "backend":c.backend.value,"dry_run":"0","input_sha1":_s(ib),"config_sha1":_s(cb),
    "gnn_checkpoint_sha256":g}
st=json.load(open(RD/"_state.json"))
diff=sorted(k for k in set(st["fingerprint"])|set(fp) if st["fingerprint"].get(k)!=fp.get(k))
assert diff==["config_sha1"], "UNEXPECTED fingerprint diff %s -- ABORT (purge risk)"%diff
json.dump(fp,open("/tmp/s07_fp.json","w"))
print("   NADP -3 | fixed",ci.fixed_residues)
print("   fingerprint diff == [config_sha1] -> patch is purge-safe")
PY
command -v python >/dev/null && [ -x /mnt/data/jglee/envs/ligandmpnn/bin/python ] \
  && echo "   ligandmpnn on PATH ok" || { echo "   LIGANDMPNN MISSING"; exit 1; }

echo ">> [2/4] backup + patch _state.json (fingerprint -> new; drop s01_input,s06_graph)"
TS=$(date +%Y%m%d-%H%M%S)
cp "$RD/_state.json" "$RD/_state.json.pre-s07.$TS.bak"
"$PYBIN" - <<PY
import json, os
from pathlib import Path
RD=Path("$RD"); st=json.load(open(RD/"_state.json")); fp=json.load(open("/tmp/s07_fp.json"))
before=list(st["completed_stages"])
st["fingerprint"]=fp
st["completed_stages"]=[s for s in st["completed_stages"] if s not in ("s01_input","s06_graph")]
tmp=RD/"_state.json.tmp"; tmp.write_text(json.dumps(st,indent=2)); os.replace(tmp,RD/"_state.json")
print("   completed:",before)
print("           ->",st["completed_stages"])
PY

echo ">> [3/4] verify patch (fp==new; s02-s06b kept; s01/s06_graph dropped)"
"$PYBIN" - <<PY
import json
from pathlib import Path
RD=Path("$RD"); st=json.load(open(RD/"_state.json")); fp=json.load(open("/tmp/s07_fp.json"))
assert st["fingerprint"]==fp, "fingerprint patch mismatch -- ABORT"
cs=st["completed_stages"]
assert "s01_input" not in cs and "s06_graph" not in cs, "drop failed"
for s in ("s02_homolog","s03_msa","s04_complex","s05_docking","s06b_interaction"):
    assert s in cs, "%s missing -> would recompute!"%s
print("   verified, kept:",cs)
PY

echo ">> [4/4] launch --resume --to s07_mutation_gen"
LOG="/mnt/data/jglee/runs/fdh_5track_s07_$TS.log"; echo "$LOG" > /tmp/fdh_s07_logpath
nohup evoliez run -c "$CFG" --resume --to s07_mutation_gen > "$LOG" 2>&1 < /dev/null &
echo "   PID $! -> $LOG"; sleep 22

echo ">> early sanity (NO purge; boltz reuse not re-fold; s07 starts)"
if grep -qaE "purged stale|invalidating resume" "$LOG"; then
  echo "   !!! PURGE DETECTED -- data-loss risk, investigate $LOG"
else echo "   no-purge OK"; fi
grep -aE "reusing.*Boltz|skip recompute|\[run \]|\[done\]|\[load\]|designable|s07_mutation|ligandmpnn|LigandMPNN|generated|Traceback|Error|AssertionError" "$LOG" | tail -22 || true
