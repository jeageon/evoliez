#!/usr/bin/env bash
# scripts/find_full_atom_wt_pdb.sh
#
# Scout the server for a FULL-ATOM WT Boltz PDB so the next
# server_test_curated_nadp.sh round can actually run end-to-end MD
# (the captured fixture under tests/fixtures/.../boltz_real.pdb is
# protein-CA-only + real ligand HETATM, so it honestly trips the
# skipped_no_full_atom_structure guard).
#
# Usage:
#   bash scripts/find_full_atom_wt_pdb.sh
#   bash scripts/find_full_atom_wt_pdb.sh configs/server_fdh_nadp.yaml
#   bash scripts/find_full_atom_wt_pdb.sh --extra /some/extra/dir
#
# Output:
#   1. Inventory of candidate PDB/CIF files labelled FULL / CA-only / empty
#   2. For FULL candidates: 1-letter sequence + identity vs the config's
#      target.fasta WT (so we don't accidentally point MD at a homolog)
#   3. The exact command to feed the best candidate into the next test:
#         bash scripts/server_test_curated_nadp.sh <abs/path>

set -u -o pipefail
cd "$(dirname "$0")/.."

CFG="configs/smoke.yaml"
EXTRA_ROOTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --extra) EXTRA_ROOTS+=("$2"); shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *)
            if [ -f "$1" ]; then CFG="$1"; shift
            else echo "unknown arg: $1" >&2; exit 2; fi ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/find_wt_pdb_${TS}.log
mkdir -p "$LOG_DIR"

say()    { printf '%s\n' "$*" | tee -a "$LOG" ; }
banner() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG" ; }

say "[scout] cwd:   $(pwd)"
say "[scout] cfg:   $CFG"
say "[scout] extra: ${EXTRA_ROOTS[*]:-(none)}"
say "[scout] log:   $LOG"

banner "Scout: full-atom WT Boltz PDB"
EXTRA_JSON=$(printf '%s\n' "${EXTRA_ROOTS[@]:-}" | python -c "
import json, sys
print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))
")
EXTRA_JSON="$EXTRA_JSON" CFG_PATH="$CFG" python - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
from pathlib import Path

# --- which sequence are we matching against? -------------------------------
cfg_path = Path(os.environ["CFG_PATH"])
target_fasta = None
try:
    from evoliez.config import load_config
    cfg = load_config(cfg_path)
    if cfg.input.target_sequence:
        wt_seq = cfg.input.target_sequence.strip().upper()
        wt_src = f"{cfg_path} (target_sequence)"
    elif cfg.input.target_fasta and Path(cfg.input.target_fasta).exists():
        ls = Path(cfg.input.target_fasta).read_text().splitlines()
        wt_seq = "".join(x.strip() for x in ls if x and not x.startswith(">")).upper()
        wt_src = cfg.input.target_fasta
    else:
        wt_seq, wt_src = "", "(no target sequence resolvable)"
except Exception as exc:
    wt_seq, wt_src = "", f"(failed to load {cfg_path}: {exc})"
print(f"[scout] WT seq source : {wt_src}")
print(f"[scout] WT length     : {len(wt_seq)} aa")

# Resolve the expected ligand heavy-atom count from the config's cofactor
# (P0 resolver). A candidate WT PDB whose HETATM ligand count differs is
# almost certainly STALE - generated before the SMILES fix - and routing
# real MD onto it raises "AssignBondOrdersFromTemplate: No matching found"
# downstream (verified on the server). Reject at scout time so the e2e
# driver naturally falls through to re-running Boltz.
expected_lig_heavy = None
try:
    from evoliez.features.cofactors import resolve_ligand_spec, formula_of
    eff = resolve_ligand_spec(cfg.input)
    if eff.type == "smiles":
        f = formula_of(eff.value)
        expected_lig_heavy = sum(f.values())
        print(f"[scout] expected lig  : {expected_lig_heavy} heavy ({f})")
except Exception as exc:
    print(f"[scout] expected lig  : (could not resolve cofactor: {exc})")

# --- candidate locations ---------------------------------------------------
extra = json.loads(os.environ.get("EXTRA_JSON", "[]"))
roots = [
    Path.home() / "EvoLiEZ" / "runs",
    Path("/mnt/data2"),
    Path("/mnt/data") / "jglee",
    Path.home() / ".cache" / "boltz",
    Path.home() / "EvoLiEZ" / "tests" / "fixtures",
    *(Path(p) for p in extra),
]
patterns = ("*model*.pdb", "*boltz*.pdb", "wt*.pdb", "*model*.cif",
            "*predictions*/*.pdb")
cands = []
for r in roots:
    if not r.exists(): continue
    for pat in patterns:
        cands.extend(r.rglob(pat))

# --- classify --------------------------------------------------------------
AA3TO1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q",
          "GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K",
          "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
          "TYR":"Y","VAL":"V"}

def classify(p: Path):
    try:
        text = p.read_text(errors="replace")
    except Exception as exc:
        return ("err", 0, 0, "", str(exc)[:80])
    n_atom = n_ca = n_het = 0
    ca_only = True
    seq = []
    for ln in text.splitlines():
        if ln.startswith("ATOM"):
            n_atom += 1
            name = ln[12:16].strip()
            if name == "CA":
                n_ca += 1
                seq.append(AA3TO1.get(ln[17:20].strip(), "X"))
            elif name:
                ca_only = False
        elif ln.startswith("HETATM"):
            n_het += 1                          # ligand heavy atoms
    if n_atom == 0: return ("empty", 0, 0, "", "no ATOM records")
    if ca_only:    return ("CA-only", n_atom, n_het, "".join(seq), f"{n_ca} residues, CA trace only")
    return ("FULL", n_atom, n_het, "".join(seq), f"{n_atom} atoms / {n_ca} residues")

def identity(a: str, b: str) -> float:
    if not a or not b: return 0.0
    n = min(len(a), len(b))
    if n == 0: return 0.0
    m = sum(1 for i in range(n) if a[i] == b[i] or a[i] == "X" or b[i] == "X")
    return 100.0 * m / max(len(a), len(b))

rows = []
seen = set()
for p in cands:
    rp = p.resolve()
    if rp in seen or not p.is_file(): continue
    seen.add(rp)
    label, n_atom, n_het, seq, why = classify(p)
    ident = identity(wt_seq, seq) if (label == "FULL" and wt_seq) else 0.0
    # Ligand-side staleness: HETATM count must match the resolved cofactor
    # heavy count (e.g. NADP+ -> 48). A 44-heavy HETATM block under a
    # NADP+ config is the classic "Boltz output predates the SMILES fix"
    # situation - reject so the e2e driver re-runs Boltz instead of
    # silently MD'ing the wrong species.
    lig_ok = (expected_lig_heavy is None or n_het == 0
              or n_het == expected_lig_heavy)
    rows.append((label, ident, lig_ok, n_atom, n_het, p, seq, why))

# Pass-everything (FULL + WT match + ligand-OK) FIRST, then FULL+WT but
# stale ligand (so user sees them flagged), then everything else.
rows.sort(key=lambda r: (
    0 if (r[0] == "FULL" and r[1] >= 95 and r[2]) else
    1 if (r[0] == "FULL" and r[1] >= 95)           else
    2 if r[0] == "FULL"                            else
    3 if r[0] == "CA-only"                         else 4,
    -r[1], -r[3],
))

print()
print(f"  {'label':<10}{'wt%':>6}  {'lig':>10}  {'atoms':>7}  path")
print("  " + "-" * 96)
for label, ident, lig_ok, n_atom, n_het, p, _seq, _why in rows[:30]:
    lig_tag = (f"{n_het}/{expected_lig_heavy}" if expected_lig_heavy
               else f"{n_het}")
    if expected_lig_heavy and n_het and not lig_ok:
        lig_tag += " STALE"
    print(f"  {label:<10}{ident:6.1f}  {lig_tag:>10}  {n_atom:>7}  {p}")
print(f"\n  total candidates: {len(rows)}")

# --- recommend the best FULL+WT+ligand-OK candidate ------------------------
ok = [r for r in rows if r[0] == "FULL" and r[1] >= 95.0 and r[2]]
stale = [r for r in rows if r[0] == "FULL" and r[1] >= 95.0 and not r[2]]
print()
if ok:
    label, ident, lig_ok, n_atom, n_het, p, seq, why = ok[0]
    print(f"[scout] BEST full-atom candidate (WT + ligand OK):")
    print(f"        {p}")
    print(f"        {why}; wt identity = {ident:.1f}%; "
          f"lig heavy = {n_het} (expected {expected_lig_heavy or '?'})")
    print(f"        seq[:60]  = {seq[:60]}")
    print(f"        seq[-60:] = {seq[-60:]}")
    print()
    print("[scout] WT match + ligand match -> next:")
    print(f"        bash scripts/server_test_curated_nadp.sh {p}")
    sys.exit(0)
elif stale:
    label, ident, lig_ok, n_atom, n_het, p, seq, why = stale[0]
    print(f"[scout] STALE Boltz output detected (WT match but wrong ligand):")
    print(f"        {p}")
    print(f"        ligand heavy = {n_het}, expected = {expected_lig_heavy}")
    print(f"        (44 = NAD+, 48 = NADP+; this output predates the P0 fix)")
    print("[scout] Re-run Boltz so the new prediction uses the corrected")
    print("        NADP+ SMILES. The e2e driver does this automatically:")
    print("        bash scripts/server_e2e_curated_nadp.sh --regen-boltz")
    sys.exit(2)                                  # distinct exit code
else:
    full = [r for r in rows if r[0] == "FULL"]
    if full:
        label, ident, lig_ok, n_atom, n_het, p, seq, why = full[0]
        print(f"[scout] BEST full-atom candidate (NOT WT):")
        print(f"        {p}")
        print(f"        {why}; wt identity = {ident:.1f}%")
        print(f"        seq[:60]  = {seq[:60]}")
        print(f"        seq[-60:] = {seq[-60:]}")
        print()
        if ident >= 30.0:
            print(f"[scout] Sequence identity {ident:.1f}% - looks like a")
            print( "        homolog, NOT the WT. Re-run Boltz on the WT")
            print( "        config or supply the actual WT prediction.")
        else:
            print(f"[scout] Sequence identity {ident:.1f}% - not WT. Re-run")
            print( "        Boltz on the WT config to produce a fresh PDB.")
        sys.exit(1)

print("[scout] no full-atom PDB found anywhere in the scanned roots.")
print("[scout] Run Boltz to produce one (config picks the protein + ligand):")
print("        bash scripts/server_smoke.sh boltz")
print("        # or, for the full server config:")
print("        evoliez run -c configs/server_fdh_nadp.yaml --to s04_complex")
print("[scout] Then re-run this scout to confirm the new PDB is full-atom.")
sys.exit(1)
PY
RC=${PIPESTATUS[0]}

banner "Summary"
say "[scout] exit code: $RC  (0 = full-atom WT found; 1 = nothing usable)"
say "[scout] full log:  $LOG"
exit $RC
