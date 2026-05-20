#!/usr/bin/env bash
# scripts/server_test_per_mutant_md.sh
#
# Step 1 of "what's next" after `server_test_plan_p0.sh` reached VALIDATED.
# This is the REAL production case: per-mutant Boltz + per-mutant real MD
# on the SMILES-authoritative + Gasteiger + cofactor-guard stack. The WT-
# only end-to-end (server_test_curated_nadp.sh) is now VALIDATED; this
# checks that the same machinery works candidate-by-candidate.
#
# Composes the existing `server_smoke.sh md` step (which already does the
# real work: s04 real WT Boltz + s08b real per-mutant Boltz + s10 real MD)
# and adds a P0-aware pre-flight + a post-flight summary that greps the
# verdict columns landed by the P0.5/P0.6 PRs (evidence_class,
# md_did_run, md_replicas_run, plif_recovery, ...).
#
# Wall time budget (smoke cfg):
#   - WT Boltz       : ~16 min (CACHED from previous run if $RUN/md exists)
#   - per-mutant Boltz: ~5-15 min × mutant_boltz_top_n (=3 for smoke)
#   - per-mutant MD  : ~2-5 min × top_for_md (=4 for smoke)
# Net: ~40-60 min cold; ~15-30 min if $RUN/md/complexes/boltz is reused.
#
# Usage:
#   bash scripts/server_test_per_mutant_md.sh                 # smoke cfg
#   bash scripts/server_test_per_mutant_md.sh configs/server_fdh_nadp.yaml

set -u -o pipefail
cd "$(dirname "$0")/.."

CFG="${1:-configs/smoke.yaml}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/per_mutant_md_${TS}.log
mkdir -p "$LOG_DIR"

say()    { printf '%s\n' "$*" | tee -a "$LOG" ; }
banner() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG" ; }

say "[per-mut] branch:   $(git rev-parse --abbrev-ref HEAD)"
say "[per-mut] HEAD:     $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
say "[per-mut] env:      ${CONDA_DEFAULT_ENV:-?} (python $(python --version 2>&1 | awk '{print $2}'))"
say "[per-mut] cfg:      $CFG"
say "[per-mut] log:      $LOG"

# ----- Phase 1: P0 invariants pre-flight ---------------------------------
banner "Phase 1: P0 invariants (config + resolver)"
CFG_PATH="$CFG" python - <<'PY' 2>&1 | tee -a "$LOG"
import os
from evoliez.config import load_config
from evoliez.features.cofactors import (
    resolve_ligand_spec, formula_of, is_known_cofactor,
)

cfg = load_config(os.environ["CFG_PATH"])
# P0 guard: ligand must be resolver-typed, cofactor known, formula matches.
assert cfg.input.ligand.type == "cofactor", (
    f"input.ligand.type={cfg.input.ligand.type!r}; "
    "P0 guard requires 'cofactor' so resolver can fill canonical SMILES"
)
assert is_known_cofactor(cfg.input.cofactor), (
    f"unknown cofactor {cfg.input.cofactor!r}"
)
eff = resolve_ligand_spec(cfg.input)
f = formula_of(eff.value)
print(f">> input.ligand.type    : {cfg.input.ligand.type}")
print(f">> input.cofactor       : {cfg.input.cofactor}")
print(f">> cofactor_redox       : {cfg.input.cofactor_redox}")
print(f">> resolved species     : {sum(f.values())} heavy, formula={f}")
print(f">> pocket_constraints   : {cfg.complex_prediction.pocket_constraints}")
print(f">> diffusion_samples WT : {cfg.complex_prediction.diffusion_samples}")
print(f">> mutant_boltz top_n   : {cfg.reranking.mutant_boltz_top_n}")
print(f">> mutant samples       : {cfg.reranking.mutant_boltz_diffusion_samples}")
print(f">> validation.md top    : {cfg.validation.md.top_candidates}")
print(f">> md.replicas          : {cfg.validation.md.replicas}")
print(f">> md.final_tier_replicas: {cfg.validation.md.final_tier_replicas}")
print(f">> md.ligand_forcefield : {cfg.validation.md.ligand_forcefield}")
PY
PRE_RC=${PIPESTATUS[0]}
if [ $PRE_RC -ne 0 ]; then
    say "[per-mut] FAIL Phase 1: config invariants broken; do not run per-mutant Boltz"
    exit 2
fi

# ----- Phase 2: run server_smoke.sh md (the real work) -------------------
banner "Phase 2: server_smoke.sh md (WT Boltz + per-mutant Boltz + real MD)"
say "[per-mut] this is the slow rung. Per-mutant Boltz is the dominant"
say "[per-mut] cost; expect ~15-30 min total on smoke cfg with WT cached,"
say "[per-mut] ~40-60 min cold. The script will tail logs as it goes."
say "[per-mut] -> bash scripts/server_smoke.sh md $CFG"
set +e
bash scripts/server_smoke.sh md "$CFG" 2>&1 | tee -a "$LOG"
SMOKE_RC=${PIPESTATUS[0]}
set -e

# ----- Phase 3: post-flight summary --------------------------------------
banner "Phase 3: post-flight verdict (md_did_run + evidence class)"
EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data/${USER}}"
RUN="$EVOLIEZ_ROOT/runs/smoke/md"
RUN_LOG="$RUN/run.log"

if [ -f "$RUN_LOG" ]; then
    say "[per-mut] grepping verdicts from $RUN_LOG"
    grep -E "MD \(L[0-9]+|MD real-execution|MD failed|generated [0-9]+ candidates|n_mutant_boltz|final ranking" \
        "$RUN_LOG" | head -20 | tee -a "$LOG" || true
fi

REPORT_DIR="$RUN/reports"
if [ -d "$REPORT_DIR" ] && [ -f "$REPORT_DIR/final_candidates.csv" ]; then
    say "[per-mut] inspecting $REPORT_DIR/final_candidates.csv"
    # Columns the P0.5/P0.6 PRs added (proves the report carries them).
    head -1 "$REPORT_DIR/final_candidates.csv" | tr ',' '\n' | \
        grep -E "evidence_class|md_did_run|md_replicas_run|md_ligand_forcefield|pose_validity|plif_recovery|boltz_delta_source" | \
        sed 's/^/  has-col /' | tee -a "$LOG" || true
    # Per-candidate counts.
    CSV="$REPORT_DIR/final_candidates.csv" python - <<'PY' 2>&1 | tee -a "$LOG"
import csv, os
from collections import Counter
path = os.environ["CSV"]
with open(path) as fh:
    rows = list(csv.DictReader(fh))
print(f"  total candidates    : {len(rows)}")
ec = Counter(r.get("evidence_class", "") for r in rows)
for k in ("Strong", "Promising", "Uncertain", "Reject"):
    if ec.get(k):
        print(f"  evidence={k:<10} {ec[k]}")
md_ran  = sum(1 for r in rows if r.get("md_did_run") in ("1", "True"))
md_pass = sum(1 for r in rows if r.get("md_passed") in ("1", "True"))
print(f"  md_did_run=true     : {md_ran}/{len(rows)}")
print(f"  md_passed=true      : {md_pass}/{len(rows)}")
bs = Counter(r.get("boltz_delta_source", "?") for r in rows)
for k, v in bs.most_common():
    print(f"  boltz_delta_source={k:<8} {v}")
PY
else
    say "[per-mut] (no final_candidates.csv yet - smoke step may have aborted)"
fi

# ----- Summary -----------------------------------------------------------
banner "Summary (paste back)"
{
    echo "branch:    $(git rev-parse --abbrev-ref HEAD)  HEAD: $(git rev-parse --short HEAD)"
    echo "env:       ${CONDA_DEFAULT_ENV:-?}"
    echo "smoke rc:  $SMOKE_RC"
    echo "--- pre-flight"
    grep -E "^>> resolved species|^>> pocket_constraints|^>> mutant_boltz" "$LOG" | head -5
    echo "--- per-mutant MD verdicts"
    grep -E "MD real-execution|MD \(L|n_mutant_boltz_evaluated|generated [0-9]+ candidates" "$LOG" | tail -8
    echo "--- evidence class + did-run counts"
    grep -E "^  evidence=|^  md_did_run|^  md_passed|^  total candidates|^  boltz_delta_source" "$LOG" || true
    echo "--- report columns present"
    grep -E "^  has-col " "$LOG" | sort -u || true
} | tee -a "$LOG"
say "[per-mut] full log: $LOG"
exit $SMOKE_RC
