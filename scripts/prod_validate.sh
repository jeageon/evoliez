#!/usr/bin/env bash
# =====================================================================
# One-shot, PER-STAGE production validation on the server.
#
#   activate env  ->  merge YOUR target (configs/target.local.yaml) over
#   configs/prod_fdh_nadp.yaml  ->  doctor gate  ->  run ONE stage with the
#   real backend  ->  AUTO acceptance checks (grounded in
#   docs/PRODUCTION_PLAYBOOK.md).  Deterministic + re-enterable (see RESUME
#   NOTE below); no sudo, nothing on root "/".
#
# ---------------------------------------------------------------------
#   # one-time: copy the input form, paste YOUR sequence + ligand + numbering
#   cp configs/target.local.yaml.example configs/target.local.yaml
#   $EDITOR configs/target.local.yaml
#
#   export EVOLIEZ_ROOT=/mnt/data2/$USER
#
#   # CHEAP stages (s01-s03, CPU) — validate INCREMENTALLY, gate each:
#   bash scripts/prod_validate.sh s01
#   bash scripts/prod_validate.sh s02
#   bash scripts/prod_validate.sh s03
#
#   # EXPENSIVE tail (s04-s11, GPU) — run the tail ONCE, then gate each stage
#   # from its persisted output (no recompute). Cross-process --resume only
#   # reloads s01/s06b; every other stage RE-RUNS on resume, so do NOT walk
#   # s04..s11 one process at a time (that re-runs Boltz/MD each gate). See the
#   # RESUME NOTE in docs/PRODUCTION_PLAYBOOK.md.
#   bash scripts/prod_validate.sh s11                  # one pass: s04..s10..s11
#   bash scripts/prod_validate.sh s04 --accept-only    # gate each, reads disk
#   bash scripts/prod_validate.sh s06b --accept-only   # ... s08b s09 s10 s11
#
#   bash scripts/prod_validate.sh s04 --dry-run        # preview real cmds only
#
# Advanced (local plumbing test on the Mac dev box, no GPU/server):
#   EVOLIEZ_SKIP_CONDA=1 EVOLIEZ_BACKEND=mock \
#     .venv-md/bin/... ; bash scripts/prod_validate.sh s01   # mock, advisory
# =====================================================================
set -euo pipefail

# --- shared 48-core server: bound thread pools (SERVER_RUNBOOK #2) ------------
export EVOLIEZ_NUM_THREADS="${EVOLIEZ_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$EVOLIEZ_NUM_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$EVOLIEZ_NUM_THREADS}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

BASE_CFG="${BASE_CFG:-configs/prod_fdh_nadp.yaml}"
USER_CFG="${USER_CFG:-configs/target.local.yaml}"
EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/${USER}}"
export EVOLIEZ_ROOT
export EVOLIEZ_DATA_DIR="${EVOLIEZ_DATA_DIR:-$EVOLIEZ_ROOT/evoliez_assets}"
ENV_PREFIX="${EVOLIEZ_ENV_PREFIX:-$EVOLIEZ_ROOT/envs/evoliez}"
BACKEND="${EVOLIEZ_BACKEND:-real}"

# --- arg parse: <stage> [--accept-only] [--no-doctor] [--dry-run] ------------
STAGE=""; ACCEPT_ONLY=0; NO_DOCTOR=0; DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --accept-only) ACCEPT_ONLY=1 ;;
    --no-doctor)   NO_DOCTOR=1 ;;
    --dry-run)     DRY_RUN=1 ;;
    -*)            echo "unknown flag '$a'"; exit 2 ;;
    *)             STAGE="$a" ;;
  esac
done
[ -n "$STAGE" ] || { echo "usage: prod_validate.sh <s01|s02|...|s11> [--accept-only] [--dry-run]"; exit 2; }

# short name -> (full Stage.name, ordinal). `case`, not `declare -A`, so this
# runs on bash 3.2 (macOS) as well as the server's bash 4+. Ordinal gates GPU /
# Boltz setup: stages s04+ touch the GPU.
case "$STAGE" in
  s01)  FULL_STAGE=s01_input;        STAGE_ORD=1 ;;
  s02)  FULL_STAGE=s02_homolog;      STAGE_ORD=2 ;;
  s03)  FULL_STAGE=s03_msa;          STAGE_ORD=3 ;;
  s04)  FULL_STAGE=s04_complex;      STAGE_ORD=4 ;;
  s05)  FULL_STAGE=s05_docking;      STAGE_ORD=5 ;;
  s06)  FULL_STAGE=s06_graph;        STAGE_ORD=6 ;;
  s06b) FULL_STAGE=s06b_interaction; STAGE_ORD=6 ;;
  s07)  FULL_STAGE=s07_mutation_gen; STAGE_ORD=7 ;;
  s08)  FULL_STAGE=s08_reranker;     STAGE_ORD=8 ;;
  s08b) FULL_STAGE=s08b_mutant_boltz; STAGE_ORD=8 ;;
  s09)  FULL_STAGE=s09_nonmd;        STAGE_ORD=9 ;;
  s10)  FULL_STAGE=s10_md;           STAGE_ORD=10 ;;
  s11)  FULL_STAGE=s11_final;        STAGE_ORD=11 ;;
  *) echo "unknown stage '$STAGE'. valid: s01 s02 s03 s04 s05 s06 s06b s07 s08 s08b s09 s10 s11"; exit 2 ;;
esac

# --- activate conda env (env lives on /mnt/data2; conda may be on root) -------
if [ "${EVOLIEZ_SKIP_CONDA:-0}" != "1" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
      [ -f "$c/etc/profile.d/conda.sh" ] && { source "$c/etc/profile.d/conda.sh"; break; }
    done
  fi
  command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found. Run scripts/setup_server_env.sh first."; exit 1; }
  source "$(conda info --base)/etc/profile.d/conda.sh"
  [ -d "$ENV_PREFIX" ] || {
    echo "ERROR: env $ENV_PREFIX missing. EVOLIEZ_ROOT=$EVOLIEZ_ROOT bash scripts/setup_server_env.sh"; exit 1; }
  conda activate "$ENV_PREFIX"
fi
command -v evoliez >/dev/null 2>&1 || { echo "ERROR: 'evoliez' not on PATH (activate the env / pip install -e .)"; exit 1; }
echo ">> python : $(command -v python) ($(python -V 2>&1))"
echo ">> stage  : $STAGE -> $FULL_STAGE   backend=$BACKEND"

# --- merge YOUR input over the base prod config -> $RUN/run_config.yaml -------
[ -f "$USER_CFG" ] || {
  echo "ERROR: $USER_CFG missing. Create it:"
  echo "   cp configs/target.local.yaml.example $USER_CFG && \$EDITOR $USER_CFG"; exit 1; }

# heredoc -> temp file (NOT nested in $()), so this is robust on bash 3.2 too.
MERGE_PY="$(mktemp "${TMPDIR:-/tmp}/evoliez_merge.XXXXXX")"
cat > "$MERGE_PY" <<'PY'
import os, sys, yaml
base = yaml.safe_load(open(sys.argv[1])) or {}
user = yaml.safe_load(open(sys.argv[2])) or {}

def deep(b, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(b.get(k), dict):
            deep(b[k], v)
        else:
            b[k] = v

# input: REPLACE wholesale (avoid base placeholder target_fasta leaking through)
if "input" in user:
    base["input"] = user["input"]
for k, v in user.items():
    if k == "input":
        continue
    if isinstance(v, dict) and isinstance(base.get(k), dict):
        deep(base[k], v)
    else:
        base[k] = v

ic = base.setdefault("input", {})
# normalise sequence: strip ALL whitespace + uppercase. doctor (diagnostics.py
# :192) only .strip()s, so an internal space from a wrapped YAML scalar would
# shift residue-token positions. Do it once, here, for both doctor and s01.
seq = ic.get("target_sequence")
if seq:
    ic["target_sequence"] = "".join(str(seq).split()).upper()
    seq = ic["target_sequence"]

# guard the placeholders so a half-filled form can't burn a real run
if not seq or "PASTE" in seq or len(seq) < 20:
    sys.exit("ERROR: input.target_sequence is missing/placeholder/too short "
             "(<20 aa). Paste your real sequence into %s." % sys.argv[2])
od = (base.get("project") or {}).get("output_dir")
if not od:
    sys.exit("ERROR: project.output_dir not set in %s." % sys.argv[2])
od = os.path.expanduser(os.path.expandvars(od))
if "<" in od or od in ("/", os.path.expanduser("~")):
    sys.exit("ERROR: project.output_dir=%r is a placeholder / unsafe root." % od)
base["project"]["output_dir"] = od

os.makedirs(os.path.join(od, "logs"), exist_ok=True)
with open(os.path.join(od, "run_config.yaml"), "w") as fh:
    yaml.safe_dump(base, fh, sort_keys=False)
print(od)
PY
if ! RUN="$(python "$MERGE_PY" "$BASE_CFG" "$USER_CFG")"; then
  rm -f "$MERGE_PY"; echo "ERROR: config merge/guard failed (see above)"; exit 1
fi
rm -f "$MERGE_PY"
RUN_CFG="$RUN/run_config.yaml"
echo ">> run dir: $RUN"
echo ">> config : $RUN_CFG (merged)"

# --- GPU + isolated Boltz env only for stages that touch them (s04+) ----------
pin_gpu() {
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && { echo ">> honoring preset CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; return 0; }
  command -v nvidia-smi >/dev/null 2>&1 || { echo ">> no nvidia-smi; not pinning"; return 0; }
  local minf="${EVOLIEZ_GPU_MIN_FREE_MIB:-20000}" q idx
  q="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"
  idx="$(echo "$q" | awk -F', *' -v m="$minf" '($2+0)>=m {print ($3+0), -($2+0), $1}' | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')"
  if [ -n "$idx" ]; then export CUDA_VISIBLE_DEVICES="$idx"; echo ">> pinned GPU $idx (>= ${minf} MiB free)"; return 0; fi
  idx="$(echo "$q" | awk -F', *' '{print ($2+0), $1}' | sort -k1,1nr | head -1 | awk '{print $2}')"
  [ -n "$idx" ] && { export CUDA_VISIBLE_DEVICES="$idx"; echo ">> WARNING: no GPU >= ${minf} MiB free; using most-free GPU $idx"; }
}
if [ "$STAGE_ORD" -ge 4 ] && [ "$BACKEND" = real ] && [ "$ACCEPT_ONLY" -eq 0 ]; then
  BOLTZ_ENV="${BOLTZ_ENV:-$EVOLIEZ_ROOT/envs/boltz}"
  if [ -x "$BOLTZ_ENV/bin/boltz" ]; then
    export PATH="$BOLTZ_ENV/bin:$PATH"
    export BOLTZ_CACHE="${BOLTZ_CACHE:-$EVOLIEZ_DATA_DIR/boltz_cache}"
    mkdir -p "$BOLTZ_CACHE"
    echo ">> isolated Boltz: $BOLTZ_ENV/bin/boltz  cache=$BOLTZ_CACHE"
  fi
  pin_gpu
fi

# --- doctor gate (strict under real; advisory under mock) ---------------------
if [ "$ACCEPT_ONLY" -eq 0 ] && [ "$NO_DOCTOR" -eq 0 ]; then
  echo "==== doctor (gate) ===="
  if [ "$BACKEND" = real ]; then
    evoliez doctor -c "$RUN_CFG"      # exits 1 on any BLOCK -> set -e aborts
  else
    evoliez doctor -c "$RUN_CFG" || echo ">> doctor advisory under mock backend"
  fi
fi

# --- run the ONE stage (resume skips already-complete predecessors) -----------
LOG="$RUN/logs/validate_${STAGE}.log"
if [ "$ACCEPT_ONLY" -eq 0 ]; then
  # `--to <stage> --resume` (NOT --from): --resume reloads every already-done
  # upstream stage's artifacts and runs only what is missing up to <stage>.
  # Run gates IN ORDER (s01, s02, ...) so each call advances exactly one stage;
  # the tee'd log's [skip]/[run] lines show what actually executed.
  echo "==== run up to $FULL_STAGE [$BACKEND] ===="
  if [ "$DRY_RUN" -eq 1 ]; then
    evoliez run -c "$RUN_CFG" --backend "$BACKEND" \
        --to "$FULL_STAGE" --resume --dry-run 2>&1 | tee "$LOG"
    echo ">> --dry-run: previewed commands only, no acceptance."; exit 0
  fi
  evoliez run -c "$RUN_CFG" --backend "$BACKEND" \
      --to "$FULL_STAGE" --resume 2>&1 | tee "$LOG"
fi

# --- acceptance ---------------------------------------------------------------
echo "==== acceptance: $STAGE ===="
case "$STAGE" in
s01)
  ACC_PY="$(mktemp "${TMPDIR:-/tmp}/evoliez_accept.XXXXXX")"
  cat > "$ACC_PY" <<'PY'
import json, os, sys, sqlite3, re
run, cfg_path, log_path = sys.argv[1], sys.argv[2], sys.argv[3]
from evoliez.config import load_config
from evoliez.features.ligand import parse_ligand

fails = warns = 0
def line(tag, name, detail):
    global fails, warns
    if tag == "FAIL": fails += 1
    if tag == "WARN": warns += 1
    print(f"  [{tag:<4}] {name:<26} {detail}")

state = json.load(open(os.path.join(run, "_state.json"))) if os.path.exists(os.path.join(run, "_state.json")) else {}
meta = state.get("meta", {})
done = state.get("completed_stages", [])
cfg = load_config(cfg_path)

# 1) stage actually completed
line("PASS" if "s01_input" in done else "FAIL", "stage_complete",
     f"completed_stages={done}")

# 2) ligand is REAL chemistry (rdkit), not the silent synthetic fallback
lig = parse_ligand(cfg.input.ligand)
src = getattr(lig, "source", "?")
line("PASS" if src == "rdkit" else "FAIL", "ligand_source",
     f"source={src!r}, n_heavy={lig.n_heavy}, n_atoms={len(lig.atoms)}"
     + ("" if src == "rdkit" else "  <-- RDKit fallback: downstream chemistry is fabricated"))

# 3) persisted ligand meta matches the freshly-parsed ligand
mh, mids = meta.get("ligand_n_heavy"), meta.get("ligand_atom_ids") or []
line("PASS" if mh == lig.n_heavy else "WARN", "ligand_meta_heavy",
     f"_state.ligand_n_heavy={mh} vs parsed={lig.n_heavy}")
line("PASS" if len(mids) == len(lig.atoms) else "WARN", "ligand_atom_ids",
     f"{len(mids)} canonical atom ids persisted (full lock verified at s04)")

# 4) sequence length sane (>=10) and consistent with the FASTA s01 wrote
slen = meta.get("sequence_length", 0)
fa = os.path.join(run, "inputs", "target.fasta")
faseq = ""
if os.path.exists(fa):
    faseq = "".join(l.strip() for l in open(fa) if not l.startswith(">"))
line("PASS" if slen and slen >= 10 else "FAIL", "sequence_length",
     f"{slen} aa (fasta={len(faseq)} aa)")
line("PASS" if (faseq and len(faseq) == slen) else "WARN", "fasta_written",
     f"inputs/target.fasta {'present' if faseq else 'MISSING'}")

# 5) ligand.smi written
smi = os.path.join(run, "inputs", "ligand.smi")
ok_smi = os.path.exists(smi) and os.path.getsize(smi) > 0
line("PASS" if ok_smi else "FAIL", "ligand_smi_written",
     f"inputs/ligand.smi {'present' if ok_smi else 'MISSING'}")

# 6) residue tokens clean in the run log (doctor hard-blocks token<->seq
#    mismatch; the log also surfaces OUT-OF-RANGE + non-standard-AA which
#    doctor does not block on)
log = open(log_path).read() if os.path.exists(log_path) else ""
oor = len(re.findall(r"position \d+ is out of range", log))
mis = len(re.findall(r"asserts .* at position", log))
nsr = len(re.findall(r"non-standard residues replaced", log))
line("FAIL" if oor else "PASS", "tokens_in_range",
     f"{oor} out-of-range catalytic/fixed token(s)")
line("WARN" if mis else "PASS", "tokens_match_seq",
     f"{mis} WT-letter/seq mismatch warning(s) (doctor already gates these)")
line("WARN" if nsr else "PASS", "standard_aa",
     f"{nsr} non-standard-residue replacement(s)")

# 7) DB target row: source='target', identity_to_target==1.0
db = os.path.join(run, "evoliez.sqlite")
if os.path.exists(db):
    con = sqlite3.connect(db)
    row = con.execute("SELECT identity_to_target, length(fasta) FROM sequence "
                      "WHERE source='target' LIMIT 1").fetchone()
    nseq = con.execute("SELECT count(*) FROM sequence").fetchone()[0]
    con.close()
    if row and abs((row[0] or 0) - 1.0) < 1e-9 and (row[1] or 0) > 0:
        line("PASS", "db_target_row", f"identity_to_target=1.0, {nseq} sequence row(s)")
    else:
        line("FAIL", "db_target_row", f"row={row}")
else:
    line("FAIL", "db_target_row", "evoliez.sqlite missing")

print()
verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
print(f"  ==> s01 {verdict}  ({fails} fail, {warns} warn)")
sys.exit(1 if fails else 0)
PY
  set +e; python "$ACC_PY" "$RUN" "$RUN_CFG" "$LOG"; rc=$?; set -e
  rm -f "$ACC_PY"
  exit $rc
  ;;
*)
  # Auto-acceptance is wired stage-by-stage as we validate each gate together;
  # s01 is live. Until $STAGE is wired, drive it by the code-grounded checklist
  # in the playbook (every check is a `file:line` you can verify against output):
  echo ">> $FULL_STAGE done. Gate it with the code-grounded checklist:"
  echo "     docs/PRODUCTION_PLAYBOOK.md   section '$STAGE'"
  echo "     run dir : $RUN"
  echo "     inspect : reports/  _state.json  evoliez.sqlite  $LOG"
  echo "     re-check later without recompute:  bash $0 $STAGE --accept-only"
  ;;
esac
