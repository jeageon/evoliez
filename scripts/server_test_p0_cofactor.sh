#!/usr/bin/env bash
# scripts/server_test_p0_cofactor.sh
#
# Single-file server test for the P0 cofactor-guard branch. Runs Steps 0-5
# of docs/SERVER_TEST_PLAN_p0_cofactor.md against the active conda env,
# streams a structured log to runs/server_test/p0_cofactor_<ts>.log, and
# prints a final summary block with exactly the lines to send back.
#
# Usage (on the server, with the MD env active):
#   bash scripts/server_test_p0_cofactor.sh
#   bash scripts/server_test_p0_cofactor.sh path/to/wt_boltz_model.pdb
#
# Going-forward convention: each test branch ships a sibling
# scripts/server_test_<topic>.sh so the server only has to git pull +
# bash that one file. Per-branch script ages out with the branch.

set -u -o pipefail

cd "$(dirname "$0")/.."                # repo root, regardless of cwd

WT_PDB=${1:-tests/fixtures/tool_outputs/captured/boltz_real.pdb}
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/p0_cofactor_${TS}.log
mkdir -p "$LOG_DIR"

say()    { printf '%s\n' "$*" | tee -a "$LOG" ; }
banner() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG" ; }

say "[server_test] branch:  $(git rev-parse --abbrev-ref HEAD)"
say "[server_test] HEAD:    $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
say "[server_test] env:     ${CONDA_DEFAULT_ENV:-?} (python $(python --version 2>&1 | awk '{print $2}'))"
say "[server_test] cwd:     $(pwd)"
say "[server_test] wt_pdb:  $WT_PDB"
say "[server_test] log:     $LOG"

# ----- Step 0: env sanity --------------------------------------------------
banner "Step 0: env sanity (MD stack)"
python - <<'PY' 2>&1 | tee -a "$LOG"
import importlib.util as u, shutil, sys
miss = []
for m in ["openmm","openff.toolkit","openmmforcefields","pdbfixer","rdkit"]:
    ok = u.find_spec(m) is not None
    print(f"  py:{m:<22}", "OK" if ok else "MISSING")
    if not ok: miss.append("py:"+m)
for b in ["antechamber","parmchk2","tleap"]:
    ok = shutil.which(b) is not None
    print(f"  bin:{b:<22}", "OK" if ok else "MISSING")
    if not ok: miss.append("bin:"+b)
# espaloma is OPTIONAL; report status but don't fail
try:
    import espaloma   # noqa: F401
    print("  py:espaloma             OK (optional FF for cofactor probe)")
except Exception:
    print("  py:espaloma             MISSING (optional; Step 4 will gate on GAFF only)")
sys.exit(0 if not miss else 2)
PY
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    say "[server_test] FAIL Step 0: MD stack incomplete; aborting"
    exit 2
fi

# ----- Step 1: P0 guard sanity --------------------------------------------
banner "Step 1: P0 guard sanity (every shipped config -> config:cofactor ok)"
GUARD_FAIL=0
for c in configs/smoke.yaml configs/server_fdh_nadp.yaml configs/example_fdh_nadp.yaml; do
    say "--- $c ---"
    # `evoliez doctor` is informational (never raises); we grep the cofactor
    # row. BLOCK or absent => Step 1 fail.
    out=$(evoliez doctor -c "$c" 2>&1 | grep -E "config:cofactor" || true)
    if [ -z "$out" ]; then
        say "  (no config:cofactor row found - guard not wired?)"
        GUARD_FAIL=1
    else
        printf '%s\n' "$out" | tee -a "$LOG"
        if printf '%s\n' "$out" | grep -qE "block"; then GUARD_FAIL=1; fi
    fi
done
if [ $GUARD_FAIL -ne 0 ]; then
    say "[server_test] FAIL Step 1: a shipped config still trips the guard"
    exit 3
fi

# ----- Step 2: deliberate NAD+ under cofactor:NADP must BLOCK/WARN --------
banner "Step 2: deliberate NAD+ under cofactor:NADP -> BLOCK/WARN"
cp configs/smoke.yaml /tmp/smoke_bad.yaml
python - <<'PY' 2>&1 | tee -a "$LOG"
import re, pathlib
p = pathlib.Path("/tmp/smoke_bad.yaml"); s = p.read_text()
s = re.sub(r"type: cofactor\n\s+value: NADP",
           'type: smiles\n    value: "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP'
           '([O-])(=O)OP([O-])(=O)OC[C@H]2O[C@@H](n3cnc4c3ncnc4N)[C@H](O)'
           '[C@@H]2O)[C@@H](O)[C@H]1O"', s, count=1)
p.write_text(s)
print("rewrote /tmp/smoke_bad.yaml back to the NAD+ SMILES under cofactor: NADP")
PY
out=$(evoliez doctor -c /tmp/smoke_bad.yaml 2>&1 | grep -E "config:cofactor" || true)
printf '%s\n' "$out" | tee -a "$LOG"
if ! printf '%s\n' "$out" | grep -qiE "looks like NAD\+|mismatch"; then
    say "[server_test] FAIL Step 2: guard did not fire on a deliberate mismatch"
    exit 3
fi

# ----- Step 3: resolver formula sanity ------------------------------------
banner "Step 3: resolver -> real NADP+ (P3 / O17 / 48 heavy)"
python - <<'PY' 2>&1 | tee -a "$LOG"
import sys
from evoliez.config import load_config
from evoliez.features.cofactors import resolve_ligand_spec, formula_of
ok = True
for path in ("configs/smoke.yaml", "configs/server_fdh_nadp.yaml"):
    cfg = load_config(path); eff = resolve_ligand_spec(cfg.input)
    f = formula_of(eff.value); n = sum(f.values())
    print(f"{path}\n  type={eff.type}  n_heavy={n}  formula={f}")
    if f.get("P") != 3 or f.get("O") != 17 or n != 48:
        print("  !! does NOT match real NADP+ (expected P=3, O=17, 48 heavy)")
        ok = False
sys.exit(0 if ok else 4)
PY
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    say "[server_test] FAIL Step 3: resolver did not produce real NADP+"
    exit 4
fi

# ----- Step 4: HEADLINE - GAFF/espaloma probe on REAL NADP+ ---------------
banner "Step 4: GAFF/espaloma probe on REAL NADP+ (VERDICT)"
python - <<'PY' 2>&1 | tee -a "$LOG"
import tempfile, traceback
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from evoliez.features.cofactors import resolve_cofactor
from evoliez.adapters.openmm_engine import (
    _ligand_offmol_at_pose, _ligand_system_generator, _LigandParamUnsupported,
)

smi = resolve_cofactor("NADP", redox_state="oxidized").smiles
m = Chem.AddHs(Chem.MolFromSmiles(smi))
assert AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE) == 0
AllChem.MMFFOptimizeMolecule(m)
pdb = Chem.MolToPDBBlock(m, flavor=4)             # HETATM + CONECT (both dirs)
work = Path(tempfile.mkdtemp(prefix="probe_")); pdb_path = work / "nadp.pdb"
out = []
for ln in pdb.splitlines():
    if ln.startswith(("ATOM  ", "HETATM")):
        ln = "HETATM" + ln[6:17] + "LIG" + ln[20:]
    out.append(ln)
pdb_path.write_text("\n".join(out) + "\n")
print(">> NADP+ pose PDB:", pdb_path)

off = _ligand_offmol_at_pose(pdb_path, smi)
print(">> OpenFF mol  :", off.n_atoms, "atoms (incl. H), charge=", off.total_charge)

try:
    sg = _ligand_system_generator(off, work)
    print(">> VERDICT     : PARAMETERIZED OK ({})".format(type(sg).__name__))
    print("   'NADP unparameterizable' was a wrong-SMILES artifact (NAD+).")
except _LigandParamUnsupported as exc:
    print(">> VERDICT     : STILL UNPARAMETERIZABLE on real NADP+")
    print(">>   reason   :", exc)
    print("   genuine GAFF/espaloma limit -> curated parameter / tleap is next.")
except Exception:
    print(">> VERDICT     : PROBE CRASHED")
    traceback.print_exc()
PY

# ----- Step 5: end-to-end check_real_md.py --------------------------------
banner "Step 5: check_real_md.py on captured WT Boltz PDB"
if [ ! -f "$WT_PDB" ]; then
    say "[server_test] SKIP Step 5: WT PDB not found at $WT_PDB"
    say "[server_test]   (pass an absolute path as the first arg to override)"
else
    # Don't propagate non-zero from check_real_md.py (status=skipped* returns
    # non-zero by design); we only care about the printed status/failure
    # lines for the summary.
    python scripts/check_real_md.py "$WT_PDB" configs/smoke.yaml 2>&1 | tee -a "$LOG" || true
fi

# ----- Summary: one paste, send these lines back --------------------------
banner "Summary (paste this block back)"
{
    echo "branch: $(git rev-parse --abbrev-ref HEAD)  HEAD: $(git rev-parse --short HEAD)"
    echo "env:    ${CONDA_DEFAULT_ENV:-?}"
    echo "--- Step 1 (config:cofactor on shipped configs)"
    grep -E "config:cofactor" "$LOG" | head -6
    echo "--- Step 2 (deliberate mismatch)"
    grep -E "config:cofactor.*mismatch|looks like NAD" "$LOG" | head -2
    echo "--- Step 3 (resolver formula)"
    grep -E "n_heavy=|formula=" "$LOG"
    echo "--- Step 4 (VERDICT)"
    grep -E ">> VERDICT|>>   reason" "$LOG"
    echo "--- Step 5 (end-to-end)"
    grep -E ">> status|>> failure|>> REAL OpenMM MD path:" "$LOG"
} | tee -a "$LOG"
say "[server_test] full log: $LOG"
