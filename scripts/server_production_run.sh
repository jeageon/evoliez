#!/usr/bin/env bash
# scripts/server_production_run.sh
#
# First REAL production run on configs/server_fdh_nadp.yaml (PseFDH
# P33160 NAD->NADP cofactor switching). Three phases, single command:
#
#   Phase 1: evoliez doctor - BLOCK 0 required; aborts if not.
#   Phase 2: evoliez run with the stage-backend override pattern this
#            project already uses (server_smoke.sh md), at PRODUCTION
#            scale (configs/server_fdh_nadp.yaml: 30 diffusion samples
#            WT, mutant_boltz_top_n=50, top_for_md=30, ...):
#              backend=mock (default), stage overrides to real for
#              s04_complex / s08b_mutant_boltz / s10_md only.
#            (Homolog/MSA stay mock so the UniRef30 DB isn't required;
#            switch to backend=real later when the DB is in place.)
#   Phase 3: open the new output_dir, list reports, print top-K
#            candidates.
#
# Wall time: ~20-30 hours on 1 A6000 (1 WT + ~50 mutant Boltz +
# ~30 real MD with 3 replicas for final-tier). Run inside tmux/screen.
#
# Usage:
#   bash scripts/server_production_run.sh                # real run
#   bash scripts/server_production_run.sh --dry-run      # command preview only
#   bash scripts/server_production_run.sh --cfg path     # alt config

set -u -o pipefail
cd "$(dirname "$0")/.."

CFG="configs/server_fdh_nadp.yaml"
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --cfg)     CFG="$2"; shift 2 ;;
        -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/production_run_${TS}.log
mkdir -p "$LOG_DIR"

say()    { printf '%s\n' "$*" | tee -a "$LOG" ; }
banner() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG" ; }

say "[prod] branch:    $(git rev-parse --abbrev-ref HEAD)"
say "[prod] HEAD:      $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
say "[prod] env:       ${CONDA_DEFAULT_ENV:-?} (python $(python --version 2>&1 | awk '{print $2}'))"
say "[prod] cfg:       $CFG"
say "[prod] dry-run:   $DRY_RUN"
say "[prod] log:       $LOG"

# ----- Environment setup (Boltz isolated env + GPU pin) -------------------
# Boltz lives in its own conda env (numpy<2 etc. - must NOT pollute the
# evoliez env). Prepend its bin so the `boltz` CLI resolves there while
# evoliez runs from its own env. Same logic as scripts/server_smoke.sh.
EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data/${USER}}"
BOLTZ_ENV="${BOLTZ_ENV:-$EVOLIEZ_ROOT/envs/boltz}"
if [ -x "$BOLTZ_ENV/bin/boltz" ]; then
    export PATH="$BOLTZ_ENV/bin:$PATH"
    export BOLTZ_CACHE="${BOLTZ_CACHE:-$EVOLIEZ_ROOT/evoliez_assets/boltz_cache}"
    mkdir -p "$BOLTZ_CACHE"
    say "[prod] using isolated Boltz: $BOLTZ_ENV/bin/boltz"
    say "[prod] BOLTZ_CACHE=$BOLTZ_CACHE"
else
    say "[prod] WARNING: $BOLTZ_ENV/bin/boltz not found - real Boltz will fail"
fi

# Pin a GPU with enough free memory (matches server_smoke.sh `pin_gpu`).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    minf="${EVOLIEZ_GPU_MIN_FREE_MIB:-20000}"
    q="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
         --format=csv,noheader,nounits 2>/dev/null)"
    idx="$(echo "$q" | awk -F', *' -v m="$minf" \
         '($2+0)>=m {print ($3+0), -($2+0), $1}' \
         | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')"
    if [ -n "$idx" ]; then
        export CUDA_VISIBLE_DEVICES="$idx"
        say "[prod] pinned GPU $idx (>= ${minf} MiB free)"
    else
        say "[prod] WARNING: no GPU with >= ${minf} MiB free; run may queue"
    fi
fi

# ----- Phase 1: doctor pre-flight ----------------------------------------
banner "Phase 1: evoliez doctor (BLOCK 0 required)"
set +e
evoliez doctor -c "$CFG" 2>&1 | tee -a "$LOG"
set -e
N_BLOCK=$(grep -c '^\[BLOCK\]' "$LOG" || true)
N_WARN=$(grep -c '^\[WARN\]' "$LOG" || true)
say "[prod] doctor verdict: $N_BLOCK BLOCK, $N_WARN WARN"
if [ "${N_BLOCK:-0}" -gt 0 ]; then
    say "[prod] FAIL Phase 1: $N_BLOCK BLOCK lines; aborting production run"
    grep '^\[BLOCK\]' "$LOG" | head -5 | sed 's/^/  /' | tee -a "$LOG"
    exit 3
fi

# ----- Phase 2: production run -------------------------------------------
banner "Phase 2: evoliez run (production scale, real Boltz + MD)"
say "[prod] this is the LONG rung. Production-scale Boltz/MD with the"
say "[prod] proven stage-override pattern (server_smoke.sh md). Homolog"
say "[prod] / MSA stay mock to avoid the UniRef30 DB dependency. ~20-30"
say "[prod] hours on 1 A6000. STRONGLY RECOMMEND tmux/screen so the run"
say "[prod] survives a disconnect:"
say "[prod]     tmux new -s prod"
say "[prod]     bash scripts/server_production_run.sh"
say "[prod]     # Ctrl+B then D to detach; tmux attach -t prod to reattach"

CMD=(
    evoliez run -c "$CFG" --backend mock
    --stage-backend s04_complex=real
    --stage-backend s08b_mutant_boltz=real
    --stage-backend s10_md=real
)
if [ $DRY_RUN -eq 1 ]; then
    CMD+=(--dry-run)
fi
say "[prod] command: ${CMD[*]}"

set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG"
RUN_RC=${PIPESTATUS[0]}
set -e
say "[prod] evoliez run exit: $RUN_RC"

# ----- Phase 3: post-run summary -----------------------------------------
banner "Phase 3: post-run verdict"
OUT_DIR="$(python - <<PY
from evoliez.config import load_config
print(load_config("$CFG").project.output_dir)
PY
)"
say "[prod] output_dir: $OUT_DIR"

if [ -d "$OUT_DIR/reports" ]; then
    say "[prod] reports under $OUT_DIR/reports/:"
    ls -la "$OUT_DIR/reports/" | sed 's/^/  /' | tee -a "$LOG"
    if [ -f "$OUT_DIR/reports/final_candidates.csv" ]; then
        say "[prod] header + top 10 candidates:"
        head -11 "$OUT_DIR/reports/final_candidates.csv" | tee -a "$LOG"
        # New P0.5/P0.6 columns sanity
        say "[prod] new evidence/provenance columns present:"
        head -1 "$OUT_DIR/reports/final_candidates.csv" | tr ',' '\n' \
            | grep -E "evidence_class|md_did_run|md_replicas_run|md_ligand_forcefield|pose_validity|plif_recovery|boltz_delta_source" \
            | sed 's/^/    /' | tee -a "$LOG"
        # Per-candidate counts
        CSV="$OUT_DIR/reports/final_candidates.csv" python - <<'PY' 2>&1 | tee -a "$LOG"
import csv, os
from collections import Counter
with open(os.environ["CSV"]) as fh:
    rows = list(csv.DictReader(fh))
print(f"  total candidates       : {len(rows)}")
ec = Counter(r.get("evidence_class", "") for r in rows)
for k in ("Strong", "Promising", "Uncertain", "Reject"):
    if ec.get(k):
        print(f"  evidence={k:<10}     {ec[k]}")
md_ran = sum(1 for r in rows if r.get("md_did_run") in ("1", "True"))
md_pass = sum(1 for r in rows if r.get("md_passed") in ("1", "True"))
print(f"  md_did_run=true        : {md_ran}/{len(rows)}")
print(f"  md_passed=true         : {md_pass}/{len(rows)}")
bs = Counter(r.get("boltz_delta_source", "?") for r in rows)
for k, v in bs.most_common():
    print(f"  boltz_delta_source={k:<8} {v}")
# Look for the canonical Tishkov D222 mutations - if any appear at top
# rank, that's instant retrospective-benchmark signal.
d222 = [r for r in rows if "D222" in r.get("mutations", "")]
if d222:
    print(f"  D222 candidates        : {len(d222)} (Tishkov NAD->NADP switch position)")
    for r in d222[:5]:
        print(f"    rank={r.get('rank')} mut={r.get('mutations')} "
              f"score={r.get('final_score')} ev={r.get('evidence_class')}")
PY
    fi
else
    say "[prod] no reports/ directory found - run may have aborted; check log"
fi

banner "Summary"
say "[prod] full log: $LOG"
say "[prod] output dir: $OUT_DIR"
say "[prod] exit code: $RUN_RC"
exit $RUN_RC
