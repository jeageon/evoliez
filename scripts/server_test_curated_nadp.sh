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

# ----- Step 1: tier-1 files (Bryce Lab) inventory --------------------------
banner "Step 1: Bryce Lab files for Tier 1 (optional)"
if ls amber/cofactors/*.frcmod >/dev/null 2>&1; then
    say "[server_test]   amber/cofactors/ has files -> Tier 1 (curated) will be tried"
    ls -la amber/cofactors/ | tee -a "$LOG"
    HAVE_CURATED=1
else
    say "[server_test]   amber/cofactors/ empty -> Tier 1 will be skipped"
    say "[server_test]   Tier 2 (Gasteiger-charge GAFF) is the fallback for known"
    say "[server_test]   cofactors and gets exercised below. To enable Tier 1:"
    say "[server_test]     bash scripts/fetch_amber_cofactors.sh   # may need URL"
    say "[server_test]     OR drop Bryce Lab .lib/.frcmod manually into amber/cofactors/"
    HAVE_CURATED=0
fi

# ----- Step 2: VERDICT - whichever tier actually parameterizes real NADP+ -
banner "Step 2: parameterization VERDICT on REAL NADP+"
python - <<'PY' 2>&1 | tee -a "$LOG"
import tempfile, traceback
from pathlib import Path
from openff.toolkit import Molecule
from evoliez.features.cofactors import resolve_cofactor
from evoliez.adapters.openmm_engine import (
    _curated_param_system_generator, _CuratedParamUnavailable,
    _gasteiger_charge_system_generator, _LigandParamUnsupported,
)

spec = resolve_cofactor("NADP", redox_state="oxidized")
print(">> spec      :", spec.name, "  res:", spec.amber_residue_name)
files = spec.resolved_amber_files()
print(">> files     :", files)

# Real OFF mol w/ conformer (no Boltz PDB needed for ligand-only probe).
off = Molecule.from_smiles(spec.smiles, allow_undefined_stereo=True)
off.name = "LIG"
off.generate_conformers(n_conformers=1)
print(">> OFF mol   :", off.n_atoms, "atoms incl. H")

import os
protein_pdb = Path(os.environ.get(
    "WT_PDB", "tests/fixtures/tool_outputs/captured/boltz_real.pdb",
))
work = Path(tempfile.mkdtemp(prefix="probe_"))
verdict = None

# Tier 1: curated AMBER (tleap) ---------------------------------------------
if files is not None:
    try:
        system, topology, positions = _curated_param_system_generator(
            off, protein_pdb, spec, work,
        )
        n_lig = sum(1 for a in topology.atoms()
                    if a.residue.name == spec.amber_residue_name)
        print(">> Tier 1    : CURATED (tleap)  OK")
        print(f">>   system  : {system.getNumParticles()} particles")
        print(f">>   ligand  : {n_lig} atoms (resname {spec.amber_residue_name})")
        verdict = "TIER1_CURATED_OK"
    except _CuratedParamUnavailable as exc:
        print(">> Tier 1    : FILES MISSING ->", exc)
    except Exception:
        print(">> Tier 1    : tleap FAILED (real MD failure, not param)")
        traceback.print_exc()
        verdict = "TIER1_CURATED_FAILED"
else:
    print(">> Tier 1    : SKIPPED (no curated files)")

# Tier 2: Gasteiger-charge GAFF ---------------------------------------------
# Always exercise this so the fallback is verified independently, even if
# Tier 1 succeeded. (Production dispatch picks Tier 1 first.)
try:
    sg = _gasteiger_charge_system_generator(off, work)
    print(">> Tier 2    : GASTEIGER GAFF      OK  (sg={})".format(
        type(sg).__name__))
    if verdict is None:
        verdict = "TIER2_GASTEIGER_OK"
except _LigandParamUnsupported as exc:
    print(">> Tier 2    : Gasteiger FALLBACK FAILED ->", str(exc)[:200])
    if verdict is None:
        verdict = "TIER2_GASTEIGER_FAILED"
except Exception:
    print(">> Tier 2    : Gasteiger PROBE CRASHED")
    traceback.print_exc()
    if verdict is None:
        verdict = "TIER2_GASTEIGER_CRASHED"

print()
print(">> VERDICT   :", verdict or "UNKNOWN")
if verdict == "TIER2_GASTEIGER_OK":
    print("   With Bryce Lab files missing, the Tier-2 self-contained")
    print("   Gasteiger-charge GAFF path parameterizes real NADP+ in")
    print("   seconds - pipeline is unblocked. When Bryce Lab files are")
    print("   added later, Tier 1 will take over automatically.")
elif verdict == "TIER1_CURATED_OK":
    print("   Curated AMBER path is operational - best-in-class params.")
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
    echo "--- Step 2 (tier verdicts)"
    grep -E "^>> Tier [12]|^>> VERDICT|^>>   system|^>>   ligand" "$LOG"
    echo "--- Step 3 (dispatch lookup)"
    grep -E "OK|WRONG" "$LOG" | grep -v "config:" | tail -4
    echo "--- Step 4 (end-to-end)"
    grep -E ">> status|>> failure|>> REAL OpenMM MD path:" "$LOG"
} | tee -a "$LOG"
say "[server_test] full log: $LOG"
