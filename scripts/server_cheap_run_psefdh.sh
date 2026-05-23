#!/usr/bin/env bash
# scripts/server_cheap_run_psefdh.sh
#
# CHEAP-RUN validator for the in-process MD cleanup + bench-summary
# harness, before any multi-enzyme full production. Driven by the
# expert validation plan: "5개 enzyme benchmark card 작성 → cheap run
# → metric 확인 → full run".
#
# What this does (single command):
#
#   1. evoliez doctor          - preflight; aborts on BLOCK 0.
#   2. evoliez run -c configs/server_fdh_nadp_cheap.yaml
#                              - full pipeline on PseFDH at cheap
#                                scale (Boltz=8 samples / homologs=24 /
#                                top_for_md=5 / production_ns=0.5).
#                                Wall time ~30-60 min on 1 A6000.
#   3. evoliez bench-summary   - score the output against the curated
#                                PseFDH benchmark.csv. Emits markdown
#                                report and JSON pass/fail.
#                                --strict so a failure exits non-zero.
#
# Pass criteria (mirrors evoliez.ml.bench_summary.DEFAULT_THRESHOLDS):
#
#   reject_top_leakage_count == 0    (P0a contract)
#   md_failure_rate            < 0.30
#   md_timeout_rate            < 0.15
#   beneficial_recall@10       >= 0.30
#
# If those hold:
#   - Copy the cheap-run knobs back into server_fdh_nadp.yaml's
#     production set with confidence the MD path doesn't leak.
#   - Run the same script for the other four enzyme cards (XR,
#     TEM-1, Bgl3, P450 BM3) by changing CFG / BENCH / NAME below.
#
# If MD fails / times out / leaks RSS:
#   - DO NOT proceed to full prod. Land MD subprocess isolation first.
#
# Usage:
#   bash scripts/server_cheap_run_psefdh.sh                 # real cheap-run
#   bash scripts/server_cheap_run_psefdh.sh --dry-run       # preview commands
#   CFG=configs/server_xr_cheap.yaml \
#     BENCH=examples/xylose_reductase/benchmark.csv \
#     NAME=XR_cheap \
#     bash scripts/server_cheap_run_psefdh.sh               # alt enzyme

set -u -o pipefail
cd "$(dirname "$0")/.."

CFG="${CFG:-configs/server_fdh_nadp_cheap.yaml}"
BENCH="${BENCH:-examples/pseudomonas_fdh/benchmark.csv}"
NAME="${NAME:-PseFDH_cheap}"
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; fi

echo "=========================================================="
echo "EvoLiEZ cheap-run validator"
echo "  config : $CFG"
echo "  bench  : $BENCH"
echo "  name   : $NAME"
echo "  dry?   : $DRY_RUN"
echo "=========================================================="

# Resolve the run output_dir from the config so we know where to look
# for final_candidates.csv afterwards. Prefer yaml-via-python (handles
# quotes / multi-line values), fall back to grep so the script also
# works on a barebones laptop that doesn't have pyyaml installed.
OUTPUT_DIR=""
if command -v python >/dev/null 2>&1; then
    OUTPUT_DIR="$(python -c "
import sys, yaml
try:
    cfg = yaml.safe_load(open('$CFG'))
    print(cfg['project']['output_dir'])
except Exception:
    sys.exit(0)
" 2>/dev/null || true)"
fi
if [ -z "$OUTPUT_DIR" ]; then
    # grep fallback: 'output_dir: <path>' in the project: block.
    # POSIX character classes ([[:space:]]) so this also works on the
    # macOS BSD sed where \s isn't whitespace.
    OUTPUT_DIR="$(grep -E '^[[:space:]]*output_dir[[:space:]]*:' "$CFG" \
        | head -1 \
        | sed -E 's/^[[:space:]]*output_dir[[:space:]]*:[[:space:]]*//; s/^["'\'']//; s/["'\'']$//')"
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERR: could not parse project.output_dir from $CFG"
    exit 2
fi
REPORTS_DIR="$OUTPUT_DIR/reports"
CAND_CSV="$REPORTS_DIR/final_candidates.csv"
MD_DIR="$OUTPUT_DIR/md"
SUMMARY_MD="$REPORTS_DIR/bench_summary.md"

run() {
    echo "+ $*"
    if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

# -------- Phase 1: preflight --------
echo
echo "--- Phase 1: doctor (preflight) ---"
run evoliez doctor -c "$CFG"

# -------- Phase 2: pipeline cheap-run --------
echo
echo "--- Phase 2: cheap-run pipeline ---"
echo "(expected wall time ~30-60 min on 1 A6000)"
START_TS="$(date +%s)"
run evoliez run -c "$CFG"
END_TS="$(date +%s)"
ELAPSED=$(( END_TS - START_TS ))
echo "pipeline wall time: ${ELAPSED}s"

# -------- Phase 3: bench-summary --------
echo
echo "--- Phase 3: bench-summary (post-hoc metric scorer) ---"
if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$CAND_CSV" ]; then
    echo "ERR: $CAND_CSV not found — pipeline did not produce a ranking."
    exit 3
fi
MD_ARG=()
if [ -d "$MD_DIR" ]; then
    MD_ARG=( "--md-root" "$MD_DIR" )
    echo "(scanning $MD_DIR for per-candidate analysis.json)"
else
    echo "(no md/ directory — MD failure/timeout rate will use the"
    echo " CSV's md_status column only)"
fi
# ${MD_ARG[@]+"${MD_ARG[@]}"} is the set-u-safe way to expand a
# possibly-empty array. Plain "${MD_ARG[@]}" tripping `unbound variable`
# was the original v1 bug — kept this comment so future maintainers
# know why the expansion is written this way.
run evoliez bench-summary \
    --candidates "$CAND_CSV" \
    --benchmark "$BENCH" \
    --name "$NAME" \
    ${MD_ARG[@]+"${MD_ARG[@]}"} \
    --out "$SUMMARY_MD" \
    --strict

EXIT=$?

# -------- Summary --------
echo
echo "=========================================================="
if [ "$EXIT" -eq 0 ]; then
    echo "PASS — cheap-run met every threshold."
    echo "  See $SUMMARY_MD for the full metric card."
    echo "Next: copy the cheap-run knobs back to the production"
    echo "      config and run scripts/server_production_run.sh,"
    echo "      OR repeat this script with the next enzyme card."
else
    echo "FAIL — cheap-run did NOT meet one or more thresholds."
    echo "  See $SUMMARY_MD for the failed-checks list."
    echo "DO NOT proceed to full production until the failures are"
    echo "  fixed. Most likely candidate: MD subprocess isolation"
    echo "  (the remaining open P0 item)."
fi
echo "=========================================================="
exit "$EXIT"
