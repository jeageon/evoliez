#!/usr/bin/env bash
# scripts/server_e2e_curated_nadp.sh
#
# End-to-end driver for the P0 cofactor MD validation. Three phases, one
# command, single log:
#   Phase 1: locate an existing full-atom WT Boltz PDB (cheap; reuses
#            scripts/find_full_atom_wt_pdb.sh).
#   Phase 2: if Phase 1 found nothing usable, RUN Boltz on the smoke
#            config (10-20 min GPU; reuses scripts/server_smoke.sh boltz)
#            and re-scout. --skip-boltz disables this.
#   Phase 3: run scripts/server_test_curated_nadp.sh <wt_pdb> -> the
#            full Tier 1/2 verdict + end-to-end check_real_md.py read.
#
# Usage:
#   bash scripts/server_e2e_curated_nadp.sh                     # smoke cfg
#   bash scripts/server_e2e_curated_nadp.sh configs/server_fdh_nadp.yaml
#   bash scripts/server_e2e_curated_nadp.sh --skip-boltz        # local-only
#
# Convention: every server-side action ships as a single script. This one
# composes three existing scripts; no fresh logic of its own.

set -u -o pipefail
cd "$(dirname "$0")/.."

CFG="configs/smoke.yaml"
SKIP_BOLTZ=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-boltz) SKIP_BOLTZ=1; shift ;;
        -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
        *)
            if [ -f "$1" ]; then CFG="$1"; shift
            else echo "unknown arg: $1" >&2; exit 2; fi ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/e2e_curated_nadp_${TS}.log
mkdir -p "$LOG_DIR"

say()    { printf '%s\n' "$*" | tee -a "$LOG" ; }
banner() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG" ; }

say "[e2e] branch:    $(git rev-parse --abbrev-ref HEAD)"
say "[e2e] HEAD:      $(git rev-parse --short HEAD)"
say "[e2e] env:       ${CONDA_DEFAULT_ENV:-?} (python $(python --version 2>&1 | awk '{print $2}'))"
say "[e2e] cfg:       $CFG"
say "[e2e] skip-boltz: $SKIP_BOLTZ"
say "[e2e] log:       $LOG"

# Extract the recommended PDB path from the find script's output. The
# scout only prints the "bash scripts/server_test_curated_nadp.sh <path>"
# line when WT identity is >= 95%, so this naturally gates out homologs.
recommended_pdb() {
    grep -E '^\s+bash scripts/server_test_curated_nadp\.sh ' "$LOG" \
        | tail -1 | awk '{print $NF}'
}

# ----- Phase 1: scout -----------------------------------------------------
banner "Phase 1: scout for existing full-atom WT PDB"
set +e
bash scripts/find_full_atom_wt_pdb.sh "$CFG" 2>&1 | tee -a "$LOG"
SCOUT_RC=${PIPESTATUS[0]}
set -e
WT_PDB="$(recommended_pdb)"
if [ -n "$WT_PDB" ]; then
    say "[e2e] Phase 1 OK: using existing full-atom WT $WT_PDB"
fi

# ----- Phase 2: run Boltz if needed --------------------------------------
if [ -z "$WT_PDB" ]; then
    if [ $SKIP_BOLTZ -eq 1 ]; then
        say "[e2e] FAIL Phase 1: no full-atom WT located and --skip-boltz set"
        exit 3
    fi
    banner "Phase 2: run Boltz to produce a full-atom WT (10-20 min GPU)"
    say "[e2e] no usable PDB on disk -> bash scripts/server_smoke.sh boltz $CFG"
    set +e
    bash scripts/server_smoke.sh boltz "$CFG" 2>&1 | tee -a "$LOG"
    BOLTZ_RC=${PIPESTATUS[0]}
    set -e
    if [ $BOLTZ_RC -ne 0 ]; then
        say "[e2e] FAIL Phase 2: Boltz step exited $BOLTZ_RC; check the log."
        exit 4
    fi
    # Re-scout: Boltz wrote into $EVOLIEZ_ROOT/runs/smoke/boltz which the
    # find script's wide net already covers (/mnt/data2, /mnt/data/jglee,
    # ~/EvoLiEZ/runs).
    say "[e2e] Boltz finished - re-scouting"
    set +e
    bash scripts/find_full_atom_wt_pdb.sh "$CFG" 2>&1 | tee -a "$LOG"
    SCOUT_RC2=${PIPESTATUS[0]}
    set -e
    WT_PDB="$(recommended_pdb)"
    if [ -z "$WT_PDB" ]; then
        say "[e2e] FAIL Phase 2: Boltz ran but no full-atom WT found in the"
        say "[e2e]   standard output dirs. Check Boltz logs above; the most"
        say "[e2e]   likely cause is an output path the scout doesn't yet"
        say "[e2e]   cover - rerun with: bash scripts/find_full_atom_wt_pdb.sh"
        say "[e2e]     --extra /path/to/your/boltz_output_dir"
        exit 5
    fi
    say "[e2e] Phase 2 OK: produced $WT_PDB"
fi

# ----- Phase 3: curated NADP test end-to-end -----------------------------
banner "Phase 3: curated NADP test on $WT_PDB"
set +e
bash scripts/server_test_curated_nadp.sh "$WT_PDB" 2>&1 | tee -a "$LOG"
TEST_RC=${PIPESTATUS[0]}
set -e

# ----- Summary -----------------------------------------------------------
banner "End-to-end summary (paste back)"
{
    echo "branch:   $(git rev-parse --abbrev-ref HEAD)  HEAD: $(git rev-parse --short HEAD)"
    echo "env:      ${CONDA_DEFAULT_ENV:-?}"
    echo "wt_pdb:   $WT_PDB"
    echo "--- Phase 3 (tier verdicts)"
    grep -E "^>> Tier [12]|^>> VERDICT" "$LOG" | tail -6
    echo "--- Phase 3 (end-to-end MD)"
    grep -E "^>> status|^>> failure|^>> REAL OpenMM MD path:" "$LOG" | tail -6
    echo "--- exit codes: scout=$SCOUT_RC test=$TEST_RC"
} | tee -a "$LOG"

say "[e2e] full log: $LOG"
exit $TEST_RC
