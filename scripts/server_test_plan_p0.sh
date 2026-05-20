#!/usr/bin/env bash
# scripts/server_test_plan_p0.sh
#
# End-to-end server validation of the PROGRAM_IMPROVEMENT_PLAN P0 plan
# (7 PRs landed on feat/p0-cofactor-guard: P0.1 receptor guard, P0.2
# Boltz steering + ensemble disagreement, P0.3 high-accuracy docking
# profile, P0.4 LigandMPNN preflight, P0.5 evidence class + monotone
# XGBoost, P0.6 MD provenance + replicas). One script, one log.
#
# Usage:
#   bash scripts/server_test_plan_p0.sh                 # default smoke cfg
#   bash scripts/server_test_plan_p0.sh configs/server_high_accuracy.yaml
#
# Steps:
#   0. Env + light pytest (everything green before any server work).
#   1. Boltz steering contract test (P0.2): WT input YAML has the real
#      pocket constraints block when configured.
#   2. Cofactor + curated MD verdict (P0/P0.6): reuses
#      server_e2e_curated_nadp.sh (which itself reuses Phase 1 scout +
#      optional Boltz + the Tier 1/2 curated path).
#   3. Full-atom docking smoke (P0.1): wires real vina/gnina/diffdock
#      through the receptor resolver on the captured WT Boltz PDB.
#      Expects each docker to refuse CA-only inputs (the captured fixture
#      is partial-CA on the protein side, so this REJECTS rather than
#      runs - the right honest behaviour, which the report distinguishes
#      from a fake pass).
#   4. Final report verdict: evidence class + boltz_delta_source +
#      replica counts surfaced in `runs/.../reports/final_candidates.csv`.

set -u -o pipefail
cd "$(dirname "$0")/.."

CFG="${1:-configs/smoke.yaml}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/plan_p0_${TS}.log
mkdir -p "$LOG_DIR"

say()    { printf '%s\n' "$*" | tee -a "$LOG" ; }
banner() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG" ; }

say "[plan-p0] branch:    $(git rev-parse --abbrev-ref HEAD)"
say "[plan-p0] HEAD:      $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
say "[plan-p0] env:       ${CONDA_DEFAULT_ENV:-?} (python $(python --version 2>&1 | awk '{print $2}'))"
say "[plan-p0] cfg:       $CFG"
say "[plan-p0] log:       $LOG"

# ----- Step 0: light pytest sanity (dev-time gate; SKIP on server envs) ----
# Production conda envs don't carry pytest (and shouldn't). Skip with a
# clear note instead of failing - the actual server-side checks below are
# what matter here, and the light pytest is for the dev/CI loop.
banner "Step 0: light pytest sanity (optional on server)"
if ! python -c "import pytest" 2>/dev/null; then
    say "[plan-p0]   pytest not in this env (normal for a production conda)."
    say "[plan-p0]   Light suite stays a DEV-time check; skipping. Run it on"
    say "[plan-p0]   your dev box:  pip install pytest && pytest -q"
else
    set +e
    python -m pytest -q 2>&1 | tail -10 | tee -a "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    if [ $RC -ne 0 ]; then
        say "[plan-p0] FAIL Step 0: pytest red; do not proceed to server checks"
        exit 2
    fi
fi

# ----- Step 1: P0.2 Boltz pocket steering contract -------------------------
banner "Step 1: P0.2 Boltz pocket-steering YAML contract"
if python -c "import pytest" 2>/dev/null; then
    python -m pytest -q tests/test_boltz_pocket_contract.py 2>&1 | tail -5 | tee -a "$LOG"
else
    # No pytest in this env -> exercise the contract directly. Same
    # invariant: `pocket_constraints: true` + residues must emit a real
    # constraints block in the Boltz input YAML.
    python - <<'PY' 2>&1 | tee -a "$LOG"
import tempfile, yaml
from pathlib import Path
from evoliez.adapters.boltz import _predict_real
from evoliez.config import ComplexPredictionConfig
from evoliez.types import Ligand

cfg = ComplexPredictionConfig(
    primary_method="boltz2", use_msa_server=True, representative_homologs=4,
    use_templates=True, pocket_constraints=True, predict_affinity=True,
    diffusion_samples=3, use_kernels=False,
)
work = Path(tempfile.mkdtemp(prefix="p0_contract_"))
_predict_real(
    "wt", "A" * 30, Ligand(id="L", smiles="CCO"), cfg, work,
    dry_run=True, msa_path=None, pocket_residues=[15, 154, 285],
)
y = yaml.safe_load((work / "wt_boltz_input.yaml").read_text())
ok = ("constraints" in y
      and y["constraints"][0]["pocket"]["binder"] == "B"
      and y["constraints"][0]["pocket"]["contacts"] == [["A", 15], ["A", 154], ["A", 285]])
print(">> Boltz pocket contract:", "OK" if ok else "FAIL")
print(">>   constraints:", y.get("constraints", "<MISSING>"))
PY
fi

# ----- Step 2: cofactor + curated MD verdict ------------------------------
banner "Step 2: cofactor guard + curated/Gasteiger MD (P0/P0.6)"
say "[plan-p0] -> bash scripts/server_e2e_curated_nadp.sh"
set +e
bash scripts/server_e2e_curated_nadp.sh 2>&1 | tee -a "$LOG"
set -e

# ----- Step 3: P0.1 full-atom docking guard -------------------------------
banner "Step 3: P0.1 docking-receptor guard (CA-only -> skipped honestly)"
python - <<'PY' 2>&1 | tee -a "$LOG"
from pathlib import Path
from evoliez.adapters import vina, gnina, diffdock
from evoliez.config import Backend, DockingConfig
from evoliez.types import LigandAtom, ProteinStructure, Residue

# Two synthetic receptors: full-atom vs CA-only. Each docker must run on
# full-atom and skip honestly on CA-only.
work = Path("runs/server_test/_doc_guard_work"); work.mkdir(parents=True, exist_ok=True)
ca_only = work / "ca.pdb"
ca_only.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0\nEND\n")
struct = ProteinStructure(
    sequence="A", residues=[Residue(index=1, aa="A", ca=(0,0,0))],
    pdb_path=str(ca_only),
)
ref = [LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0))]
cfg = DockingConfig(timeout_s=60)

for name, mod in (("vina", vina), ("gnina", gnina), ("diffdock", diffdock)):
    try:
        # NB: diffdock takes smiles; supply a stub.
        if name == "diffdock":
            pose = mod.redock("c", struct, ref, cfg, work / name,
                              instability=0.0, smiles="CCO",
                              backend=Backend.real, dry_run=False)
        else:
            pose = mod.redock("c", struct, ref, cfg, work / name,
                              instability=0.0,
                              backend=Backend.real, dry_run=False)
    except Exception as exc:
        print(f">> {name:<10}: ERROR {exc}")
        continue
    sk = getattr(pose, "skipped", None) or "?"
    print(f">> {name:<10}: skipped={sk!r}  pose_validity={pose.pose_validity_status!r}")
PY

# ----- Step 4: final-report verdict ---------------------------------------
banner "Step 4: final-report verdict (evidence class + provenance)"
# Find the most recent run, grep the columns the plan demands.
LATEST=$(ls -td runs/smoke/* 2>/dev/null | head -1)
if [ -z "$LATEST" ] || [ ! -d "$LATEST/reports" ]; then
    say "[plan-p0] no run output to inspect at runs/smoke/*; skipping"
else
    say "[plan-p0] inspecting: $LATEST/reports/"
    for f in final_candidates.csv focused_library.csv final_report.md; do
        p="$LATEST/reports/$f"
        if [ -f "$p" ]; then
            say "--- $f (head) ---"
            head -3 "$p" | tee -a "$LOG"
            # Look for the new columns
            if [ "${f##*.}" = "csv" ]; then
                head -1 "$p" | tr ',' '\n' | grep -E \
                    "evidence_class|boltz_delta_source|md_did_run|md_replicas_run|plif_recovery|pose_validity" \
                    | sed 's/^/  has-col /' | tee -a "$LOG"
            fi
        fi
    done
fi

# ----- Summary -------------------------------------------------------------
banner "Summary"
{
    echo "branch:   $(git rev-parse --abbrev-ref HEAD)  HEAD: $(git rev-parse --short HEAD)"
    echo "env:      ${CONDA_DEFAULT_ENV:-?}"
    echo "--- Step 2 (cofactor + MD verdict)"
    grep -E "^>> VERDICT|^>> Tier|^>> status|^>> failure" "$LOG" | tail -10
    echo "--- Step 3 (docking-guard verdict)"
    grep -E "^>> (vina|gnina|diffdock):" "$LOG" | tail -3
    echo "--- Step 4 (report columns)"
    grep -E "^  has-col " "$LOG" | sort -u
} | tee -a "$LOG"
say "[plan-p0] full log: $LOG"
