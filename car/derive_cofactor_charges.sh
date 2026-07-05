#!/usr/bin/env bash
# Generic phosphate-rich cofactor MD charge template: single-point AM1-BCC at the
# Boltz-bound pose (heavy atoms from the s04 model + H added onto that geometry via a
# SMILES template). Avoids the gas-phase optimisation that crashes tri/tetra-anionic
# phosphates. Used for ATP(-4) and NADPH(-4).
#
#   bash car/derive_cofactor_charges.sh <NAME> <SMILES> <NET_CHARGE> <CHAIN> <MODEL.pdb>
# e.g. bash car/derive_cofactor_charges.sh NADPH "NC(=O)C1=CN..." -4 D /path/model_0.pdb
# -> params/<name_lower>_<|nc|>minus/<NAME>.fixed.mol2 (+ .ref.sdf)
set -euo pipefail
NAME="${1:?NAME}"; SMILES="${2:?SMILES}"; NC="${3:?NET_CHARGE}"; CHAIN="${4:?CHAIN}"; PDB="${5:?MODEL.pdb}"
OUT="/mnt/data/jglee/EvoLiEZ_car/params/$(echo "$NAME" | tr 'A-Z' 'a-z')_${NC#-}minus"
mkdir -p "$OUT"

# 1) extract this chain's heavy atoms, add H via SMILES template at the bound geometry
/mnt/data/jglee/envs/evoliez/bin/python - "$PDB" "$CHAIN" "$SMILES" "$OUT/$NAME.ref.sdf" <<'PY'
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
pdb, chain, smi, out = sys.argv[1:5]
lines = [l for l in open(pdb) if l.startswith("HETATM") and l[21] == chain]
open("/tmp/_lig.pdb", "w").writelines(lines)
tmpl = Chem.MolFromSmiles(smi)
pose = Chem.MolFromPDBFile("/tmp/_lig.pdb", removeHs=True, sanitize=False)
pose = AllChem.AssignBondOrdersFromTemplate(tmpl, pose)
poseH = Chem.AddHs(pose, addCoords=True)
Chem.MolToMolFile(poseH, out)
print("ref: %d atoms, charge %d" % (poseH.GetNumAtoms(), Chem.GetFormalCharge(poseH)))
PY

# 2) single-point AM1-BCC (maxcyc=0) at that geometry
AMBERTOOLS_ENV=/mnt/data/jglee/anaconda3/envs/AmberTools25
export AMBERHOME=$AMBERTOOLS_ENV PATH=$AMBERTOOLS_ENV/bin:$PATH LD_LIBRARY_PATH=$AMBERTOOLS_ENV/lib:${LD_LIBRARY_PATH:-}
RN=$(echo "$NAME" | cut -c1-3)
cd "$OUT"; rm -f "$NAME.fixed.mol2" sqm.in sqm.out ac.log
antechamber -i "$NAME.ref.sdf" -fi sdf -o "$NAME.fixed.mol2" -fo mol2 -c bcc -nc "$NC" -at gaff2 -rn "$RN" \
  -ek "qm_theory='AM1', maxcyc=0, scfconv=1.d-10, ndiis_attempts=700, printcharges=1" >ac.log 2>&1 || true
grep -iE "Total Mulliken" sqm.out 2>/dev/null | tail -1
/mnt/data/jglee/envs/evoliez/bin/python - "$OUT/$NAME.fixed.mol2" "$NC" <<'PY'
import sys
try:
    L = open(sys.argv[1]).read().splitlines()
    i = L.index("@<TRIPOS>ATOM"); j = L.index("@<TRIPOS>BOND")
    q = [float(l.split()[-1]) for l in L[i+1:j]]
    print("*** %s OK: %d atoms, charge sum=%.4f (target %s) ***" % (sys.argv[1].split('/')[-1], len(q), sum(q), sys.argv[2]))
except Exception as e:
    print("FAIL:", e)
PY
