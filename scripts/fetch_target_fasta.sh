#!/usr/bin/env bash
# scripts/fetch_target_fasta.sh
#
# Fetch a target protein FASTA from UniProt by accession, write it under
# examples/<species>/target.fasta, and print the residue at each
# expected catalytic position so the user can sanity-check before the
# config update.
#
# Usage:
#   bash scripts/fetch_target_fasta.sh P33160 pseudomonas_fdh \
#        H332 N146 D196
#
# The trailing positional args are residue tokens (e.g., H332 = His at
# position 332). The script DOESN'T mutate any config - it only fetches
# the FASTA and reports what's at each named position so you can confirm
# the numbering before editing configs/server_fdh_nadp.yaml.

set -u -o pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
    sed -n '2,17p' "$0"
    exit 2
fi

ACC="$1"; shift
NAME="$1"; shift
RESIDUES=("$@")

OUT_DIR="examples/${NAME}"
OUT_FASTA="$OUT_DIR/target.fasta"
mkdir -p "$OUT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/fetch_target_${NAME}_${TS}.log
mkdir -p "$LOG_DIR"

say() { printf '%s\n' "$*" | tee -a "$LOG" ; }

say "[fetch-target] accession: $ACC"
say "[fetch-target] name:      $NAME"
say "[fetch-target] residues:  ${RESIDUES[*]:-(none)}"
say "[fetch-target] out:       $OUT_FASTA"
say "[fetch-target] log:       $LOG"

URL="https://rest.uniprot.org/uniprotkb/${ACC}.fasta"
if ! curl -fsSL --max-time 60 -o "$OUT_FASTA.partial" "$URL"; then
    say "[fetch-target] FAIL: curl could not fetch $URL"
    say "[fetch-target]   check the accession + network; if behind a proxy,"
    say "[fetch-target]   download the FASTA manually and drop it at $OUT_FASTA"
    rm -f "$OUT_FASTA.partial"
    exit 3
fi
mv "$OUT_FASTA.partial" "$OUT_FASTA"

# Also fetch the Swiss-Prot flat (TXT) file with the curated feature
# annotations - ACT_SITE / BINDING / SITE entries are the authoritative
# source for catalytic & NAD-binding residues, not literature numbering
# which often disagrees due to signal-peptide / mature-protein offsets.
TXT_URL="https://rest.uniprot.org/uniprotkb/${ACC}.txt"
OUT_TXT="$OUT_DIR/uniprot_${ACC}.txt"
if curl -fsSL --max-time 60 -o "$OUT_TXT.partial" "$TXT_URL"; then
    mv "$OUT_TXT.partial" "$OUT_TXT"
else
    rm -f "$OUT_TXT.partial"
    say "[fetch-target] note: could not fetch annotations TXT (continuing)"
    OUT_TXT=""
fi

# Strip header + concatenate
SEQ=$(grep -v '^>' "$OUT_FASTA" | tr -d '\n')
LEN=${#SEQ}
HDR=$(head -1 "$OUT_FASTA")
say "[fetch-target] header:    $HDR"
say "[fetch-target] length:    $LEN aa"

# Validate each residue position
say
say "[fetch-target] residue validation:"
say "  token   expect actual  ok?"
say "  ----------------------------------------"
ALL_OK=1
for tok in "${RESIDUES[@]}"; do
    # H332 -> letter=H, pos=332
    if [[ "$tok" =~ ^([A-Za-z])([0-9]+)$ ]]; then
        letter="${BASH_REMATCH[1]}"
        pos="${BASH_REMATCH[2]}"
        # 1-based, so index = pos - 1
        if (( pos < 1 || pos > LEN )); then
            say "  $tok  $letter      OUT-OF-RANGE  ($LEN aa)"
            ALL_OK=0
            continue
        fi
        actual="${SEQ:$((pos-1)):1}"
        if [[ "${actual^}" == "${letter^}" ]]; then
            say "  $tok  $letter      $actual         OK"
        else
            say "  $tok  $letter      $actual         MISMATCH"
            ALL_OK=0
        fi
    else
        say "  $tok  ?      ?           BAD-TOKEN"
        ALL_OK=0
    fi
done

say
if [ $ALL_OK -eq 1 ] && [ ${#RESIDUES[@]} -gt 0 ]; then
    say "[fetch-target] OK: all residue tokens consistent with the fasta"
elif [ ${#RESIDUES[@]} -gt 0 ]; then
    say "[fetch-target] some residue tokens don't match the fasta - that's"
    say "[fetch-target]   usually literature-numbering drift. Check the"
    say "[fetch-target]   UniProt annotations below for the authoritative"
    say "[fetch-target]   positions and re-run with the corrected tokens."
fi

# UniProt curated feature annotations - the authoritative source for
# catalytic + NAD-binding residues. Print everything relevant.
if [ -n "$OUT_TXT" ] && [ -f "$OUT_TXT" ]; then
    say
    say "[fetch-target] UniProt feature annotations (authoritative):"
    say "  source: $OUT_TXT"
    say

    OUT_TXT="$OUT_TXT" SEQ="$SEQ" python - <<'PY' 2>&1 | tee -a "$LOG"
import os, re, sys

path = os.environ["OUT_TXT"]
seq = os.environ["SEQ"]
text = open(path).read().splitlines()

# Parse FT (feature) blocks. Format (UniProt SwissProt flat):
#   FT   ACT_SITE        332
#   FT                   /note="Proton donor"
#   FT   BINDING         204..209
#   FT                   /ligand="NAD(+)"
#   FT                   /ligand_id="ChEBI:CHEBI:57540"
#   FT                   /ligand_part="..."
# A new feature starts on a line with a feature key in the
# first sub-field; subsequent lines with leading spaces extend it.
def parse_features():
    feats = []
    cur = None
    for ln in text:
        if not ln.startswith("FT"):
            if cur:
                feats.append(cur); cur = None
            continue
        # `FT   KEY            range` (new feature)
        m = re.match(r"^FT\s+([A-Z_]+)\s+(\S+)\s*$", ln)
        if m:
            if cur:
                feats.append(cur)
            cur = {"key": m.group(1), "range": m.group(2), "notes": {}}
            continue
        # continuation: `FT                   /name="value"`
        m2 = re.match(r"^FT\s+/([A-Za-z_]+)=(.+)$", ln)
        if m2 and cur is not None:
            v = m2.group(2).strip().strip('"')
            cur["notes"].setdefault(m2.group(1), []).append(v)
    if cur:
        feats.append(cur)
    return feats

def aa_at(pos: int) -> str:
    if 1 <= pos <= len(seq):
        return seq[pos - 1]
    return "?"

def positions(rng: str):
    m = re.match(r"^(\d+)(?:\.\.(\d+))?$", rng)
    if not m: return []
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return list(range(a, b + 1))

feats = parse_features()

# Print ACT_SITE, BINDING (esp. for NAD / formate / substrate), SITE.
shown_any = False
print(f"  {'kind':<10}{'pos':<10}{'aa':<4}{'note':<40}{'ligand':<25}")
print("  " + "-" * 95)

want_keys = ("ACT_SITE", "BINDING", "SITE")
ligand_priority = ("NAD", "NADP", "NADPH", "FORMATE", "SUBSTRATE")
collected = {"catalytic": [], "nad_binding": [], "other_binding": [], "site": []}

for f in feats:
    if f["key"] not in want_keys:
        continue
    notes = f["notes"]
    note = "; ".join(notes.get("note", []))[:38]
    lig  = "; ".join(notes.get("ligand", []))[:23]
    for pos in positions(f["range"]):
        aa = aa_at(pos)
        print(f"  {f['key']:<10}{str(pos):<10}{aa:<4}{note:<40}{lig:<25}")
        shown_any = True
        tok = f"{aa}{pos}"
        if f["key"] == "ACT_SITE":
            collected["catalytic"].append(tok)
        elif f["key"] == "BINDING":
            if any(x in lig.upper() for x in ("NAD", "NADP")):
                collected["nad_binding"].append(tok)
            else:
                collected["other_binding"].append((tok, lig))
        elif f["key"] == "SITE":
            collected["site"].append((tok, note))

if not shown_any:
    print("  (no ACT_SITE / BINDING / SITE features in this record)")

print()
print("  ----- suggested config tokens (verify against the comparison table) -----")
if collected["catalytic"]:
    print(f"  catalytic_residues : {collected['catalytic']}")
if collected["nad_binding"]:
    print(f"  known_binding_site : {collected['nad_binding']}    # NAD/NADP-binding")
if collected["site"]:
    print("  notable SITEs:")
    for tok, note in collected["site"]:
        print(f"    {tok}  - {note}")
PY
fi

say
say "[fetch-target] FASTA at: $(realpath "$OUT_FASTA")"
if [ -n "$OUT_TXT" ]; then
    say "[fetch-target] TXT  at: $(realpath "$OUT_TXT")"
fi
exit 0
