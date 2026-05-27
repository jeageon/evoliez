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
# More robust than `evoliez --help | grep bench-summary` — typer wraps
# help output to terminal width, so the literal "bench-summary" string
# can be hyphenated when the column is narrow. Invoking the subcommand
# directly with --help returns 0 if it exists, non-zero otherwise,
# independent of terminal formatting.
evoliez bench-summary --help >/dev/null 2>&1 \
    || die "bench-summary subcommand not registered (try: pip install -e . --no-deps)"
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

# CHEAP-RUN-6 LESSON: a config can pass YAML validation but silently
# carry wrong residue numbering (e.g. literature paper numbering vs.
# UniProt numbering). When this happens, evoliez warns at s01 but
# proceeds — burning 1-3 hours of Boltz+MD on a wrong-position run.
# Fail HARD here instead. Same check evoliez doctor does post-Phase-1,
# pulled forward so we never waste pipeline time.
if [ -f "$FASTA_PATH" ]; then
    say "Phase 0d-2 — residue-token validation against target.fasta"
    PYRES=$(python - "$CFG" "$FASTA_PATH" <<'PYEOF'
import sys, yaml
from pathlib import Path

cfg_path, fasta_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(cfg_path))
seq = "".join(
    l.strip() for l in open(fasta_path)
    if l and not l.startswith(">")
)
n = len(seq)
mismatches = []
for kind in ("catalytic_residues", "fixed_residues", "known_binding_site"):
    for tok in cfg.get("input", {}).get(kind, []) or []:
        try:
            wt, pos = tok[0], int(tok[1:])
        except ValueError:
            continue
        if pos < 1 or pos > n:
            mismatches.append(f"  {kind:>20} {tok}  position out of range (seq len {n})")
            continue
        actual = seq[pos - 1]
        if actual != wt:
            mismatches.append(f"  {kind:>20} {tok}  asserts {wt} but seq has {actual} at position {pos}")
if mismatches:
    print("MISMATCH")
    for m in mismatches[:30]:
        print(m)
    if len(mismatches) > 30:
        print(f"  ... ({len(mismatches) - 30} more)")
else:
    print("OK")
PYEOF
)
    if echo "$PYRES" | head -1 | grep -q "^MISMATCH$"; then
        echo "$PYRES" | tail -n +2
        die "$(cat <<EOF
residue tokens in $CFG don't match $FASTA_PATH.

This is the cheap-run-6 lesson: literature paper numbering often
DOESN'T match UniProt numbering. Run

    bash scripts/fetch_target_fasta.sh $UNIPROT $SLUG <residue tokens>

and look at the "UniProt feature annotations (authoritative)"
section it prints — copy those positions into $CFG and re-run.
Do NOT proceed with a wrong-numbering run; it burns 1-3 hours.
EOF
)"
    fi
    ok "residue tokens all match $FASTA_PATH"
fi

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

# BENCH_ONLY=1 → skip Phase 2 entirely; re-score the EXISTING
# final_candidates.csv via Phase 3 only. Use when a bug fix touched
# ONLY the bench-summary scorer / thresholds / cheap profile, so we
# don't need to re-run the pipeline at all. Wall time: ~1 s.
#
# (RERANK_ONLY=1 was the old name for this — but `evoliez run
# --from s11_final --resume` doesn't work in practice because s11
# requires upstream in-memory `candidates` that aren't reloaded from
# disk. BENCH_ONLY skips the pipeline entirely instead.)
if [ "${BENCH_ONLY:-${RERANK_ONLY:-0}}" = "1" ]; then
    warn "BENCH_ONLY=1 — skipping Phase 2 entirely; rescoring existing CSV"
    echo "   expected wall time: <5 s (Phase 3 only)"
    SKIP_PIPELINE_RUN=1
else
    echo "   expected wall time: 30-90 min on 1 A6000 ($NAME)"
    SKIP_PIPELINE_RUN=0
fi

if [ "$SKIP_PIPELINE_RUN" = "0" ]; then
    START_TS="$(date +%s)"
    if ! evoliez run -c "$CFG" "${EVOLIEZ_STAGE_OVERRIDES[@]}"; then
        die "evoliez run exited non-zero. Inspect the log at $_LOG_FILE."
    fi
    END_TS="$(date +%s)"
    ELAPSED=$(( END_TS - START_TS ))
    ok "pipeline finished in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
else
    ok "Phase 2 skipped (BENCH_ONLY)"
fi

CAND_CSV="$OUTPUT_DIR/reports/final_candidates.csv"
[ -f "$CAND_CSV" ] || die "no $CAND_CSV — pipeline produced no ranking"

# ===========================================================
# Phase 2.5 -- MD preflight status (operator-facing)
# ===========================================================
# C / H (post-expert-audit): s01 writes md_preflight_status to
# project meta. Surfacing it here lets the operator see UP-FRONT
# whether this enzyme's ligand was MD-parameterisable, instead of
# discovering it via "MD failed on every candidate" 25 hours later.
# Read-only; the actual gate is enforced by s08b when
# mdcfg.strict_preflight=True.
say "Phase 2.5 — MD preflight verdict"
META_FILE="$OUTPUT_DIR/_state.json"
if [ -f "$META_FILE" ]; then
    PRE_STATUS="$(python3 -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    meta = d.get("meta") or {}
    print(meta.get("md_preflight_status", "?"))
except Exception as e:
    print(f"<read-error: {e}>", file=sys.stderr)
    print("?")
' "$META_FILE" 2>/dev/null || echo "?")"
    PRE_REASON="$(python3 -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    meta = d.get("meta") or {}
    print(meta.get("md_preflight_reason", ""))
except Exception:
    print("")
' "$META_FILE" 2>/dev/null || echo "")"
    case "$PRE_STATUS" in
        ok|curated_available)
            ok "MD preflight: $PRE_STATUS — MD will run"
            ;;
        skipped_md_disabled|skipped_no_smiles|skipped_no_md_libs)
            warn "MD preflight: $PRE_STATUS — MD will NOT run (config / env)"
            ;;
        unsupported|timeout_preflight|failed_preflight)
            warn "MD preflight: $PRE_STATUS — MD will SKIP (no FF supports this ligand)"
            [ -n "$PRE_REASON" ] && echo "   reason: $PRE_REASON"
            warn "Set MDConfig.strict_preflight=true to short-circuit s08b on this status."
            ;;
        *)
            warn "MD preflight: status=$PRE_STATUS (unrecognised — possibly older run)"
            ;;
    esac
else
    warn "no $META_FILE — preflight status unavailable (pipeline may have crashed early)"
fi

# ===========================================================
# Phase 2b -- output-package inventory check
# ===========================================================
# Catch silent partial failures (Boltz wrote 0 of 12 PDBs, MD
# analysis.json empty, ml_datasets/ missing, etc.) BEFORE the
# bench-summary makes pass/fail noise about metrics that may be
# downstream consequences of a missing artefact.
say "Phase 2b — output-package inventory"
_missing=0
_check() {
    local label="$1"; local path="$2"; local min_bytes="${3:-1}"
    if [ ! -e "$path" ]; then
        warn "MISSING  $label  ($path)"
        _missing=$((_missing + 1))
    elif [ -f "$path" ]; then
        local sz="$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0)"
        if [ "$sz" -lt "$min_bytes" ]; then
            warn "TOO SMALL $label  ($sz bytes < $min_bytes; $path)"
            _missing=$((_missing + 1))
        else
            ok "$label  ($sz bytes)"
        fi
    elif [ -d "$path" ]; then
        local nf="$(find "$path" -maxdepth 2 -type f 2>/dev/null | wc -l)"
        ok "$label  ($nf files in $path)"
    fi
}

_check "final_candidates.csv" "$CAND_CSV" 500
_check "focused_library.csv" "$OUTPUT_DIR/reports/focused_library.csv" 100
_check "final_report.md" "$OUTPUT_DIR/reports/final_report.md" 200
_check "ml_datasets/" "$OUTPUT_DIR/ml_datasets"
_check "complexes/boltz (WT)" "$OUTPUT_DIR/complexes/boltz"

# Per-candidate MD outputs — count PRIMARY candidate dirs with a
# non-empty analysis.json. Skip replica dirs (mut_XXXXX_r1 / _r2):
# by design they hold replica trajectories but only the primary dir
# gets the to_json() summary. Counting them as "missing" would
# falsely flag 6 of 18 dirs on a 3-final-tier-replicas run.
#
# `set +e` around this block: a bad bash glob OR an `ls | wc -l`
# pipeline can pipefail-kill the script silently. Disable strict
# mode for the inventory check; re-enable after.
set +e
if [ -d "$OUTPUT_DIR/md" ]; then
    _ok_md=0; _bad_md=0; _replica=0
    for d in "$OUTPUT_DIR/md/"*/; do
        bname="$(basename "$d")"
        # Skip replicas (anything ending in _r<digit>); they don't
        # carry their own analysis.json by design.
        case "$bname" in *_r[0-9]*) _replica=$((_replica + 1)); continue ;; esac
        if [ ! -f "$d/analysis.json" ] || [ ! -s "$d/analysis.json" ]; then
            _bad_md=$((_bad_md + 1)); continue
        fi
        _ok_md=$((_ok_md + 1))
    done
    if [ "$_bad_md" -gt 0 ]; then
        warn "md/  $_ok_md primary ok / $_bad_md primary missing analysis.json (+ $_replica replica dirs)"
        _missing=$((_missing + 1))
    else
        ok "md/  $_ok_md primary candidates ok (+ $_replica replica dirs)"
    fi
else
    warn "MISSING  md/ directory ($OUTPUT_DIR/md)"
    _missing=$((_missing + 1))
fi

# Per-candidate mutant Boltz outputs. Boltz writes either flat files
# (mut_NNNNN_*.pdb) or per-input subdirs (mut_NNNNN/) depending on
# config — count both. `find` returns 0 on no-match (unlike ls), so
# it's pipefail-safe.
if [ -d "$OUTPUT_DIR/complexes/mutant_boltz" ]; then
    _n_mb=$(find "$OUTPUT_DIR/complexes/mutant_boltz" -maxdepth 1 \
                 -name "mut_*" 2>/dev/null | wc -l)
    if [ "$_n_mb" -gt 0 ]; then
        ok "complexes/mutant_boltz  $_n_mb entries (mix of files / dirs)"
    else
        warn "complexes/mutant_boltz empty"
        _missing=$((_missing + 1))
    fi
else
    warn "MISSING  complexes/mutant_boltz/"
    _missing=$((_missing + 1))
fi
set -e

if [ "$_missing" -gt 0 ]; then
    warn "$_missing artefact(s) missing or too small — bench-summary may"
    warn "  reflect partial-pipeline output. Inspect $OUTPUT_DIR before"
    warn "  trusting the cheap-run pass/fail."
fi

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
# Phase 4 -- HTML report package (only on PASS; opt-out via SKIP_HTML=1)
# ===========================================================
HTML_ZIP="$OUTPUT_DIR/report_package.zip"
HTML_DIR="$OUTPUT_DIR/report_package"      # builder also leaves an unzipped tree
# Always-on (was: PASS-only). Debugging a FAIL is exactly when the HTML
# package is most useful — `Bgl3 FAIL (deleterious-bottom)` had no
# dashboard to inspect H121A/G ranking. SKIP_HTML=1 still works as opt-out.
if [ "${SKIP_HTML:-0}" != "1" ]; then
    if [ "$RC" -ne 0 ]; then
        say "Phase 4 — HTML report package (FAIL debug mode)"
    else
        say "Phase 4 — HTML report package"
    fi
    set +e
    evoliez figures \
        -c "$CFG" \
        -o "$HTML_ZIP" \
        --benchmark "$BENCH" 2>&1 | tail -20
    _html_rc=$?
    set -e
    if [ "$_html_rc" -eq 0 ] && [ -f "$HTML_ZIP" ]; then
        _html_size="$(stat -c%s "$HTML_ZIP" 2>/dev/null \
                     || stat -f%z "$HTML_ZIP" 2>/dev/null || echo 0)"
        ok "HTML zip:      $HTML_ZIP ($((_html_size / 1024)) KB)"
        # The builder also leaves an unzipped tree alongside (typically).
        _html_idx="$(find "$OUTPUT_DIR" -name index.html -path '*/report_package*' 2>/dev/null | head -1)"
        [ -n "$_html_idx" ] && ok "HTML index:    $_html_idx"
    else
        warn "evoliez figures exited rc=$_html_rc — HTML package may be incomplete"
        warn "  inspect $OUTPUT_DIR for partial output"
    fi
fi

# ===========================================================
# Summary
# ===========================================================
say "Result"
if [ "$RC" -eq 0 ]; then
    printf '\033[1;42m PASS \033[0m  cheap-run met every threshold (%s)\n' "$NAME"
    echo "   markdown report: $SUMMARY_MD"
    [ -f "$HTML_ZIP" ] && \
        echo "   HTML zip:        $HTML_ZIP"
    echo
    echo "   Next:"
    echo "     - cat \"$SUMMARY_MD\"     # full metric card"
    [ -f "$HTML_ZIP" ] && echo "     - unzip \"$HTML_ZIP\" -d /tmp/${NAME} && open /tmp/${NAME}/index.html"
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
