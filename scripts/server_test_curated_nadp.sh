#!/usr/bin/env bash
# scripts/server_test_curated_nadp.sh
#
# End-to-end check that the curated AMBER cofactor path actually builds a
# working OpenMM System for real NADP+ via tleap + Bryce Lab params,
# bypassing GAFF/AM1-BCC (which HARD-FAILS on real NADP+, confirmed by
# server_test_p0_cofactor.sh Step 4).
#
# Usage:
#   bash scripts/server_test_curated_nadp.sh
#   bash scripts/server_test_curated_nadp.sh path/to/real_wt_boltz.pdb
#
# Steps:
#   0. Env sanity (MD stack + tleap + parmchk2 + Bryce Lab files).
#   1. Fetch hint: if amber/cofactors/ is empty, point at fetch script.
#   2. Curated probe: build (system, topology, positions) for NADP+ via
#      tleap end-to-end; report particle count + ligand atoms found.
#   3. Dispatch check: lookup_by_smiles(NADP+ SMILES) returns the spec
#      and its files resolve.
#   4. Optional end-to-end: scripts/check_real_md.py on a full-atom WT
#      PDB; the curated dispatch is now what runs inside _run_real.

set -u -o pipefail
cd "$(dirname "$0")/.."

WT_PDB=${1:-tests/fixtures/tool_outputs/captured/boltz_real.pdb}
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/curated_nadp_${TS}.log
mkdir -p "$LOG_DIR"

say()    { printf '%s\n' "$*" | tee -a "$LOG" ; }
banner() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG" ; }

say "[server_test] branch:  $(git rev-parse --abbrev-ref HEAD)"
say "[server_test] HEAD:    $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
say "[server_test] env:     ${CONDA_DEFAULT_ENV:-?} (python $(python --version 2>&1 | awk '{print $2}'))"
say "[server_test] cwd:     $(pwd)"
say "[server_test] log:     $LOG"

# ----- Step 0: env sanity -------------------------------------------------
banner "Step 0: env sanity (MD stack + Bryce Lab cofactor files)"
python - <<'PY' 2>&1 | tee -a "$LOG"
import importlib.util as u, shutil, sys
from pathlib import Path
from evoliez.features.cofactors import amber_params_root, resolve_cofactor

miss = []
for m in ["openmm","openff.toolkit","openmmforcefields","pdbfixer","rdkit"]:
    ok = u.find_spec(m) is not None
    print(f"  py:{m:<22}", "OK" if ok else "MISSING")
    if not ok: miss.append(m)
for b in ["antechamber","parmchk2","tleap"]:
    ok = shutil.which(b) is not None
    print(f"  bin:{b:<22}", "OK" if ok else "MISSING")
    if not ok: miss.append(b)

root = amber_params_root()
print(f"  amber_params_root: {root}")
have_files = False
for redox in ("oxidized", "reduced"):
    sp = resolve_cofactor("NADP", redox_state=redox)
    files = sp.resolved_amber_files()
    mark = "OK" if files else "MISSING"
    print(f"  bryce:{sp.name:<10}    {mark}  "
          f"({sp.amber_lib}+{sp.amber_frcmod} under {root})")
    if files:
        have_files = True
if not have_files:
    print("  -> no Bryce Lab files present; curated dispatch will fall "
          "back to the probe (which fails for NADP+).")
sys.exit(0 if not miss else 2)
PY
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    say "[server_test] FAIL Step 0: MD stack incomplete"
    exit 2
fi

# ----- Step 1: fetch hint -------------------------------------------------
banner "Step 1: Bryce Lab files (fetch if missing)"
if ! ls amber/cofactors/*.frcmod >/dev/null 2>&1; then
    say "[server_test]   amber/cofactors/ has no .frcmod files."
    say "[server_test]   run: bash scripts/fetch_amber_cofactors.sh"
    say "[server_test]   or:  drop NAD/NDH/NAP/NDP {.lib,.frcmod,.mol2} manually"
    say "[server_test]   then re-run this script. Aborting Step 2+."
    exit 3
fi
ls -la amber/cofactors/ | tee -a "$LOG"

# ----- Step 2: curated probe (the headline) -------------------------------
banner "Step 2: curated tleap probe on REAL NADP+ (VERDICT)"
python - <<'PY' 2>&1 | tee -a "$LOG"
import tempfile, traceback
from pathlib import Path
from openff.toolkit import Molecule
from evoliez.features.cofactors import resolve_cofactor, lookup_by_smiles
from evoliez.adapters.openmm_engine import (
    _ligand_offmol_at_pose, _curated_param_system_generator,
    _CuratedParamUnavailable, _write_ligand_pdb,
)

spec = resolve_cofactor("NADP", redox_state="oxidized")
print(">> spec      :", spec.name, "  res:", spec.amber_residue_name)
files = spec.resolved_amber_files()
print(">> files     :", files)
if files is None:
    print(">> VERDICT   : NO BRYCE LAB FILES -> curated dispatch can't run")
    raise SystemExit(0)

# Build OFF mol with a conformer (no Boltz PDB needed for the probe).
off = Molecule.from_smiles(spec.smiles, allow_undefined_stereo=True)
off.name = "LIG"
off.generate_conformers(n_conformers=1)
print(">> OFF mol   :", off.n_atoms, "atoms incl. H")

# A protein-only PDB is required - use a captured Boltz output (CA-only is
# fine here because the curated probe runs PDBFixer; we only need SOMETHING
# protein-like). If user has a real full-atom PDB, pass it as $1 to this
# script and we'll use that path; otherwise fall back to the captured one.
import sys, os
protein_pdb = Path(os.environ.get("WT_PDB", "tests/fixtures/tool_outputs/captured/boltz_real.pdb"))
print(">> protein   :", protein_pdb, "exists =", protein_pdb.exists())

work = Path(tempfile.mkdtemp(prefix="curated_probe_"))
try:
    system, topology, positions = _curated_param_system_generator(
        off, protein_pdb, spec, work,
    )
    n_lig = sum(1 for a in topology.atoms()
                if a.residue.name == spec.amber_residue_name)
    print(f">> VERDICT   : CURATED PARAMETERIZATION OK")
    print(f">>   system  : {system.getNumParticles()} particles")
    print(f">>   ligand  : {n_lig} atoms (resname {spec.amber_residue_name})")
    print(">>   tleap files: prmtop +", str(work / "complex.inpcrd"))
    print("   The curated AMBER path produces a working OpenMM System on")
    print("   real NADP+ - the production bypass of GAFF/AM1-BCC works.")
except _CuratedParamUnavailable as exc:
    print(">> VERDICT   : FILES MISSING (re-check amber/cofactors/)")
    print(">>   reason :", exc)
except Exception:
    print(">> VERDICT   : CURATED PROBE FAILED (real MD failure, not param)")
    traceback.print_exc()
PY

# ----- Step 3: dispatch lookup sanity -------------------------------------
banner "Step 3: lookup_by_smiles dispatch sanity"
python - <<'PY' 2>&1 | tee -a "$LOG"
from evoliez.features.cofactors import resolve_cofactor, lookup_by_smiles
for fam, redox, expect in [
    ("NAD",  "oxidized", "NAD+"),
    ("NAD",  "reduced",  "NADH"),
    ("NADP", "oxidized", "NADP+"),
    ("NADP", "reduced",  "NADPH"),
]:
    sp = resolve_cofactor(fam, redox_state=redox)
    got = lookup_by_smiles(sp.smiles)
    ok = got is not None and got.name == expect
    print(f"  {expect:<6}  lookup -> {got.name if got else None:<8}  files={sp.resolved_amber_files() is not None}  {'OK' if ok else 'WRONG'}")
PY

# ----- Step 4: end-to-end (optional) --------------------------------------
banner "Step 4: end-to-end check_real_md.py (uses curated dispatch inside)"
if [ ! -f "$WT_PDB" ]; then
    say "[server_test]   SKIP - no WT PDB at $WT_PDB"
    say "[server_test]   (pass an absolute path as the first arg)"
else
    python scripts/check_real_md.py "$WT_PDB" configs/smoke.yaml 2>&1 | tee -a "$LOG" || true
fi

# ----- Summary ------------------------------------------------------------
banner "Summary (paste back)"
{
    echo "branch: $(git rev-parse --abbrev-ref HEAD)  HEAD: $(git rev-parse --short HEAD)"
    echo "env:    ${CONDA_DEFAULT_ENV:-?}"
    echo "--- Step 0 (bryce files)"
    grep -E "bryce:" "$LOG" | head -8
    echo "--- Step 2 (VERDICT)"
    grep -E ">> VERDICT|>>   system|>>   ligand|>>   reason" "$LOG"
    echo "--- Step 3 (dispatch lookup)"
    grep -E "OK|WRONG" "$LOG" | grep -v "config:" | tail -4
    echo "--- Step 4 (end-to-end)"
    grep -E ">> status|>> failure|>> REAL OpenMM MD path:" "$LOG"
} | tee -a "$LOG"
say "[server_test] full log: $LOG"
