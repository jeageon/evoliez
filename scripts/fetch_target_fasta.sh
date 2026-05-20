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
    say "[fetch-target] next steps:"
    say "  1. update configs/server_fdh_nadp.yaml:"
    say "       target_fasta: $OUT_FASTA"
    say "       catalytic_residues: [\"<verified tokens here>\"]"
    say "  2. run: evoliez doctor -c configs/server_fdh_nadp.yaml"
    say "     (the 'illustrative placeholder' BLOCK should clear)"
elif [ ${#RESIDUES[@]} -gt 0 ]; then
    say "[fetch-target] WARN: some residue tokens don't match the fasta -"
    say "[fetch-target]   check the literature numbering (signal peptide vs"
    say "[fetch-target]   mature protein numbering, often differs by ~24 aa)"
    say "[fetch-target]   before updating the config."
fi

say
say "[fetch-target] FASTA at: $(realpath "$OUT_FASTA")"
exit 0
