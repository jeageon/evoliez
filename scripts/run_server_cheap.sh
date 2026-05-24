#!/usr/bin/env bash
# scripts/run_server_cheap.sh
#
# ONE-SHOT cheap-run validator for the multi-enzyme campaign.
#
# Single bash entry-point that handles EVERYTHING the operator would
# otherwise paste line-by-line (and forget). Self-contained on purpose:
# the user gets corrected dozens of times when their conda env drops
# back to base inside a screen subshell or they forget to fetch the
# FASTA; the runbook version of this kept hitting those edge cases.
#
# The script:
#
#   * auto-detects + activates the evoliez conda env (looks under
#     /mnt/data*/$USER/envs first, falls back to plain `conda activate
#     evoliez`),
#   * git pulls the latest feat/html-report-package head IF behind,
#   * re-installs the editable package when a known new module is
#     missing (so `pip install -e .` doesn't get skipped after a fast-
#     forward that added files),
#   * fetches the per-enzyme FASTA via scripts/fetch_target_fasta.sh
#     when target.fasta is missing,
#   * runs `evoliez doctor` (preflight; aborts on BLOCK 0),
#   * runs `evoliez run` at cheap scale (smaller Boltz + 5-candidate
#     real MD with subprocess isolation enabled),
#   * runs `evoliez bench-summary --strict` on the result,
#   * prints a clear PASS / FAIL line at the end + path to the
#     markdown card.
#
# Re-runnable: every step is idempotent. Safe to re-launch after a
# crash; it picks up at the first phase that needs work.
#
# Default target: PseFDH (the validated cofactor-switching positive
# control). Override per-enzyme via env vars:
#
#   ENZYME=xr ./scripts/run_server_cheap.sh
#   ENZYME=tem1 ./scripts/run_server_cheap.sh
#   ENZYME=bgl3 ./scripts/run_server_cheap.sh
#   ENZYME=p450 ./scripts/run_server_cheap.sh
#   ENZYME=fdh ./scripts/run_server_cheap.sh        # default
#
# Or fully custom:
#   CFG=configs/my_cfg.yaml BENCH=examples/my_card/benchmark.csv \
#     NAME=my_card UNIPROT=Q12345 SLUG=my_card \
#     ./scripts/run_server_cheap.sh
#
# Background-friendly: pipe to log + nohup yourself, or wrap in
# screen/tmux. Exit code 0 = pass, 2 = fail, anything else = error
# before the metric check.

set -euo pipefail

# ---- self-locate the repo root (script may be invoked from anywhere) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- per-enzyme dispatch table (defaults to PseFDH) ----
ENZYME="${ENZYME:-fdh}"
case "$ENZYME" in
    fdh|psefdh)
        : "${CFG:=configs/server_fdh_nadp_cheap.yaml}"
        : "${BENCH:=examples/pseudomonas_fdh/benchmark.csv}"
        : "${NAME:=PseFDH_cheap}"
        : "${UNIPROT:=P33160}"
        : "${SLUG:=pseudomonas_fdh}"
        : "${FASTA_RESIDUES:=R285 H333 D222}"
        ;;
    xr)
        : "${CFG:=configs/server_xr_cheap.yaml}"
        : "${BENCH:=examples/xylose_reductase/benchmark.csv}"
        : "${NAME:=XR_cheap}"
        : "${UNIPROT:=O74237}"
        : "${SLUG:=xylose_reductase}"
        : "${FASTA_RESIDUES:=Y51 K80 D46 H110}"
        ;;
    tem1)
        : "${CFG:=configs/server_tem1_cheap.yaml}"
        : "${BENCH:=examples/tem1_betalactamase/benchmark.csv}"
        : "${NAME:=TEM1_cheap}"
        : "${UNIPROT:=P62593}"
        : "${SLUG:=tem1_betalactamase}"
        : "${FASTA_RESIDUES:=S70 K73 E166}"
        ;;
    bgl3)
        : "${CFG:=configs/server_bgl3_cheap.yaml}"
        : "${BENCH:=examples/beta_glucosidase_bgl3/benchmark.csv}"
        : "${NAME:=Bgl3_cheap}"
        : "${UNIPROT:=P22073}"
        : "${SLUG:=beta_glucosidase_bgl3}"
        : "${FASTA_RESIDUES:=E166 E352}"
        ;;
    p450|bm3)
        : "${CFG:=configs/server_p450_bm3_cheap.yaml}"
        : "${BENCH:=examples/p450_bm3/benchmark.csv}"
        : "${NAME:=P450BM3_cheap}"
        : "${UNIPROT:=P14779}"
        : "${SLUG:=p450_bm3}"
        : "${FASTA_RESIDUES:=F87 A82 C400 D251}"
        ;;
    *)
        # user provided custom env vars only — accept and proceed
        : "${CFG:?ENZYME=$ENZYME not recognised; set CFG explicitly}"
        : "${BENCH:?set BENCH= when ENZYME is unknown}"
        : "${NAME:?set NAME= when ENZYME is unknown}"
        ;;
esac

# ---- nice-to-have output helpers ----
_LOG_FILE="${RUN_LOG:-$REPO_ROOT/cheap_run_${NAME}_$(date +%F_%H%M).log}"
exec > >(tee -a "$_LOG_FILE") 2>&1
say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()  { printf '   \033[1;32mOK\033[0m  %s\n' "$*"; }
warn(){ printf '   \033[1;33mWARN\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mFAIL\033[0m %s\n' "$*"; exit 1; }

say "EvoLiEZ cheap-run validator (ENZYME=$ENZYME)"
echo "   repo       : $REPO_ROOT"
echo "   config     : $CFG"
echo "   benchmark  : $BENCH"
echo "   name       : $NAME"
echo "   uniprot    : ${UNIPROT:-(custom)}"
echo "   log file   : $_LOG_FILE"

# ===========================================================
# Phase 0  --  Self-bootstrap: conda env + git + editable install
# ===========================================================

say "Phase 0a — conda env activation"
# Source-of-truth: "is the evoliez CLI on PATH?". Don't try to guess
# which conda env we're in from CONDA_PREFIX / CONDA_DEFAULT_ENV —
# miniforge3's base env reports CONDA_PREFIX=.../miniforge3 (basename
# != "base"), which fooled an earlier version into thinking it was
# already in the right env. The only check that always tells the truth
# is `command -v evoliez`.
if command -v evoliez >/dev/null 2>&1; then
    ok "evoliez CLI already on PATH ($(command -v evoliez))"
    ok "current env: ${CONDA_PREFIX:-(none)}"
else
    echo "   evoliez CLI not on PATH; searching for an evoliez conda env..."

    # Search common locations. /mnt/data{,2}/$USER first because that
    # mirrors how this server installs envs; HOME paths next; finally
    # ask conda's env list.
    _candidate=""
    for prefix in \
        "/mnt/data/${USER}/envs/evoliez" \
        "/mnt/data2/${USER}/envs/evoliez" \
        "${HOME}/envs/evoliez" \
        "${HOME}/miniforge3/envs/evoliez" \
        "${HOME}/miniconda3/envs/evoliez" \
        "${HOME}/anaconda3/envs/evoliez"
    do
        if [ -x "$prefix/bin/evoliez" ]; then
            _candidate="$prefix"; break
        fi
        if [ -d "$prefix/conda-meta" ]; then
            _candidate="$prefix"
            # keep looking - prefer one that actually has the CLI installed
        fi
    done

    if [ -z "$_candidate" ] && command -v conda >/dev/null 2>&1; then
        # Ask conda for any env named 'evoliez'.
        _candidate="$(conda env list 2>/dev/null \
            | awk '/^evoliez[[:space:]]/ {print $NF; exit}')"
    fi

    [ -n "$_candidate" ] || die "could not auto-locate an evoliez conda env. \
Activate manually (e.g. conda activate /mnt/data/\$USER/envs/evoliez) then re-run."

    echo "   candidate env: $_candidate"

    # Source the candidate's conda hook first (works even when no
    # `conda` is on PATH yet), then fall back to the currently-active
    # conda's shell hook.
    if [ -f "$_candidate/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$_candidate/etc/profile.d/conda.sh"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]; then
        # base env's conda.sh — works because activate is just a function
        # shellcheck disable=SC1091
        source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        die "no conda hook available to drive `conda activate $_candidate`"
    fi

    conda activate "$_candidate" \
        || die "`conda activate $_candidate` returned non-zero"

    # Re-verify the CLI is now visible.
    command -v evoliez >/dev/null \
        || die "evoliez CLI STILL not on PATH after `conda activate $_candidate` \
(CONDA_PREFIX=$CONDA_PREFIX). Is the package editable-installed into that env?"
    ok "activated $_candidate"
    ok "evoliez CLI -> $(command -v evoliez)"
fi

say "Phase 0b — git pull (only if behind origin/feat/html-report-package)"
git fetch --quiet origin feat/html-report-package
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse origin/feat/html-report-package)"
if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
    ok "already at origin head ($LOCAL_SHA)"
else
    if ! git diff --quiet || ! git diff --cached --quiet; then
        warn "uncommitted local changes detected; stashing before pull"
        git stash push -u -m "pre-cheap-run-$(date +%F_%H%M)" >/dev/null
    fi
    git checkout feat/html-report-package --quiet
    if ! git pull --ff-only origin feat/html-report-package; then
        warn "fast-forward pull failed; falling back to hard reset to origin"
        git reset --hard origin/feat/html-report-package
    fi
    ok "now at $(git rev-parse HEAD)"
fi

say "Phase 0c — editable install (only if a new module fails to import)"
# Probe a module that only exists at HEAD (so a stale install gets re-bound).
if ! python -c "from evoliez.adapters.openmm_subprocess import run_md_in_subprocess" 2>/dev/null; then
    pip install -e . --no-deps --quiet \
        && ok "pip install -e . refreshed"
fi
python -c "from evoliez.adapters.openmm_subprocess import run_md_in_subprocess" \
    || die "openmm_subprocess module STILL missing after pip install -e ."
python -c "from evoliez.ml.bench_summary import compute_summary" \
    || die "bench_summary module missing"
evoliez --help | grep -q bench-summary \
    || die "bench-summary subcommand not registered"
ok "all expected new modules + CLI commands resolve"

say "Phase 0d — target FASTA + cheap config sanity"
[ -f "$CFG" ]   || die "cheap config not found: $CFG  (only ENZYME=fdh has a config so far; create configs/server_${ENZYME}_cheap.yaml first)"
[ -f "$BENCH" ] || die "benchmark CSV not found: $BENCH"

FASTA_PATH="$REPO_ROOT/examples/${SLUG}/target.fasta"
if [ ! -f "$FASTA_PATH" ] && [ -n "${UNIPROT:-}" ]; then
    warn "target.fasta missing; fetching via scripts/fetch_target_fasta.sh"
    bash scripts/fetch_target_fasta.sh "$UNIPROT" "$SLUG" ${FASTA_RESIDUES:-} \
        || die "fetch_target_fasta.sh failed (offline server? UniProt down?)"
fi
[ -f "$FASTA_PATH" ] \
    && ok "FASTA at $FASTA_PATH ($(wc -l < "$FASTA_PATH") lines)" \
    || warn "no FASTA at $FASTA_PATH; config may rely on a different path"

# Auto-rewrite output_dir for the current user (jglee → $USER), idempotent.
if grep -q "/mnt/data/jglee/" "$CFG" && [ "$USER" != "jglee" ]; then
    sed -i.bak "s|/mnt/data/jglee/|/mnt/data/${USER}/|g" "$CFG"
    ok "rewrote output_dir for user=$USER (backup at $CFG.bak)"
fi

# Resolve output_dir from the config so we know where to look for outputs.
OUTPUT_DIR="$(python -c "
import yaml, sys
try:
    print(yaml.safe_load(open('$CFG'))['project']['output_dir'])
except Exception:
    sys.exit(0)
")"
[ -n "$OUTPUT_DIR" ] || die "could not parse project.output_dir from $CFG"
mkdir -p "$(dirname "$OUTPUT_DIR")"
ok "output_dir: $OUTPUT_DIR"

# Check disk room (need ~5 GB headroom per cheap run).
AVAIL_KB="$(df -Pk "$(dirname "$OUTPUT_DIR")" | awk 'NR==2 {print $4}')"
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
[ "$AVAIL_GB" -lt 5 ] && warn "only ${AVAIL_GB} GB free on $(dirname "$OUTPUT_DIR") — cheap-run typically uses 3-5 GB" \
                     || ok "${AVAIL_GB} GB free on $(dirname "$OUTPUT_DIR")"

# ===========================================================
# Phase 0e  --  Boltz isolated env + GPU pin
# ===========================================================
# Boltz lives in its OWN conda env (numpy<2 etc. — must NOT pollute the
# evoliez env). Prepend its bin so the `boltz` CLI resolves there while
# evoliez itself stays in its own env. Same pattern as
# scripts/server_production_run.sh / scripts/server_smoke.sh.
say "Phase 0e — Boltz env + GPU pin"
EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data/${USER}}"
BOLTZ_ENV="${BOLTZ_ENV:-$EVOLIEZ_ROOT/envs/boltz}"
if [ -x "$BOLTZ_ENV/bin/boltz" ]; then
    export PATH="$BOLTZ_ENV/bin:$PATH"
    export BOLTZ_CACHE="${BOLTZ_CACHE:-$EVOLIEZ_ROOT/evoliez_assets/boltz_cache}"
    mkdir -p "$BOLTZ_CACHE"
    ok "isolated Boltz: $(command -v boltz)"
    ok "BOLTZ_CACHE=$BOLTZ_CACHE"
else
    warn "$BOLTZ_ENV/bin/boltz not found — Phase 2 will die at stage s04_complex."
    warn "Fix: install Boltz into $BOLTZ_ENV, OR override BOLTZ_ENV=/path/to/your/env."
    # Don't die yet — let the doctor + dry-run logic surface this with
    # better context if the user does have it elsewhere.
fi

# Pin a GPU with enough free memory (mirrors server_production_run.sh).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    minf="${EVOLIEZ_GPU_MIN_FREE_MIB:-20000}"
    q="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
         --format=csv,noheader,nounits 2>/dev/null)"
    idx="$(echo "$q" | awk -F', *' -v m="$minf" \
         '($2+0)>=m {print ($3+0), -($2+0), $1}' \
         | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')"
    if [ -n "$idx" ]; then
        export CUDA_VISIBLE_DEVICES="$idx"
        ok "pinned GPU $idx (>= ${minf} MiB free)"
    else
        warn "no GPU with >= ${minf} MiB free; cheap-run may queue"
    fi
else
    ok "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(default)}"
fi

# ===========================================================
# Phase 1  --  doctor preflight
# ===========================================================
say "Phase 1 — evoliez doctor (preflight)"
if ! evoliez doctor -c "$CFG"; then
    die "evoliez doctor reported a blocking failure. Fix that first, then re-run."
fi
ok "doctor passed"

# ===========================================================
# Phase 2  --  cheap pipeline run
# ===========================================================
say "Phase 2 — evoliez run (cheap config; subprocess MD isolation ON)"
# Stage-backend pattern (mirrors scripts/server_production_run.sh):
# default backend=mock so the homolog/MSA stages don't need the
# UniRef30 mmseqs DB on /mnt/data2, and only the GPU-heavy stages run
# REAL. Boltz uses its own MSA server (use_msa_server: true in the
# config) so it never reads the mock s02/s03 output.
EVOLIEZ_STAGE_OVERRIDES=(
    --backend mock
    --stage-backend s04_complex=real
    --stage-backend s08b_mutant_boltz=real
    --stage-backend s10_md=real
)
# Allow an operator override for the rare case they want full-real
# (e.g. when /mnt/data2 mmseqs DB IS available + populated).
if [ "${ALL_REAL:-0}" = "1" ]; then
    EVOLIEZ_STAGE_OVERRIDES=( --backend real )
    warn "ALL_REAL=1 set — running every stage at backend=real (needs UniRef30 DB)"
fi

# RERANK_ONLY=1 → re-run JUST the final-ranking stage on top of the
# existing per-stage output (Boltz / MD outputs on disk are unchanged).
# Use this when a bug fix touched ONLY s11_final or io/report, so we
# don't re-pay the 2-hour Boltz+MD cost just to regenerate the CSV.
# Wall time: ~2 seconds.
if [ "${RERANK_ONLY:-0}" = "1" ]; then
    EVOLIEZ_STAGE_OVERRIDES+=( --from s11_final --resume )
    warn "RERANK_ONLY=1 — re-running only s11_final on existing pipeline output"
    echo "   expected wall time: <5 s (no Boltz / MD re-execution)"
else
    echo "   expected wall time: 30-60 min on 1 A6000 (PseFDH cheap)"
fi

START_TS="$(date +%s)"
if ! evoliez run -c "$CFG" "${EVOLIEZ_STAGE_OVERRIDES[@]}"; then
    die "evoliez run exited non-zero. Inspect the log at $_LOG_FILE."
fi
END_TS="$(date +%s)"
ELAPSED=$(( END_TS - START_TS ))
ok "pipeline finished in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"

CAND_CSV="$OUTPUT_DIR/reports/final_candidates.csv"
[ -f "$CAND_CSV" ] || die "no $CAND_CSV — pipeline produced no ranking"

# ===========================================================
# Phase 3  --  bench-summary (post-hoc metrics)
# ===========================================================
say "Phase 3 — evoliez bench-summary --strict"
MD_DIR="$OUTPUT_DIR/md"
SUMMARY_MD="$OUTPUT_DIR/reports/bench_summary.md"
MD_ARG=()
[ -d "$MD_DIR" ] && MD_ARG=( "--md-root" "$MD_DIR" )

set +e
# --profile cheap: cheap-run scale (top_for_md=12 / 8 Boltz samples /
# 0.5 ns MD) cannot mathematically guarantee recall@10 for the D222
# family on a 17-binding-site enzyme like PseFDH. The cheap profile
# relaxes the recall@K threshold (still surfaced in the markdown) and
# adds a benchmark_in_pool_rate gate instead — validates that the
# generator + reranker floors actually surface literature mutations
# into the pool, even when cheap scale can't push them to top-K.
# Override with PROFILE=prod for the strict full-prod thresholds.
evoliez bench-summary \
    --candidates "$CAND_CSV" \
    --benchmark  "$BENCH" \
    --name       "$NAME" \
    --config     "$CFG" \
    --profile    "${PROFILE:-cheap}" \
    ${MD_ARG[@]+"${MD_ARG[@]}"} \
    --out        "$SUMMARY_MD" \
    --strict
RC=$?
set -e

# ===========================================================
# Summary
# ===========================================================
say "Result"
if [ "$RC" -eq 0 ]; then
    printf '\033[1;42m PASS \033[0m  cheap-run met every threshold (%s)\n' "$NAME"
    echo "   markdown report: $SUMMARY_MD"
    echo
    echo "   Next:"
    echo "     - cat \"$SUMMARY_MD\"     # full metric card"
    echo "     - then either repeat with ENZYME=xr/tem1/bgl3/p450 ./$(basename "$0"),"
    echo "       or run scripts/server_production_run.sh for full prod."
    exit 0
else
    printf '\033[1;41m FAIL \033[0m  bench-summary reported threshold violations\n'
    echo "   markdown report: $SUMMARY_MD"
    echo
    echo "   Quick triage:"
    echo "     grep -E 'Reject|MD failure|MD timeout|recall' \"$SUMMARY_MD\""
    echo "     ls \"$MD_DIR\"/  # which candidate workdirs exist"
    echo "     for d in \"$MD_DIR\"/*/; do echo \"=== \$d\"; tail -30 \"\$d/_md_subprocess.log\" 2>/dev/null; done"
    echo
    echo "   DO NOT proceed to full prod until the failure is understood."
    exit 2
fi
