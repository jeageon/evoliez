#!/usr/bin/env bash
# scripts/fetch_amber_cofactors.sh
#
# Download Bryce Lab / Manchester curated AMBER parameters for NAD+ / NADH
# / NADP+ / NADPH into amber/cofactors/. Read amber/cofactors/MANIFEST.yaml
# (default) or a user-supplied manifest. The dispatcher
# evoliez.adapters.openmm_engine._curated_param_system_generator only runs
# when these files are present, so on a fresh server clone this is the
# one-time setup step.
#
# Usage:
#   bash scripts/fetch_amber_cofactors.sh
#   bash scripts/fetch_amber_cofactors.sh --manifest /path/custom.yaml
#   bash scripts/fetch_amber_cofactors.sh --dest /mnt/data2/amber_cofactors
#
# This script does NOT vendor the parameter files into the repo - they are
# licensed by the Manchester group for academic use. Drop downloaded files
# in amber/cofactors/ manually if you already have them and the curated
# path will pick them up.

set -u -o pipefail
cd "$(dirname "$0")/.."

MANIFEST="amber/cofactors/MANIFEST.yaml"
DEST="amber/cofactors"

while [ $# -gt 0 ]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --dest)     DEST="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f "$MANIFEST" ]; then
    echo "manifest not found: $MANIFEST" >&2
    exit 2
fi
mkdir -p "$DEST"

# tiny YAML reader: lines like '  base_url: "..."' + species blocks.
# Robust enough for our minimal MANIFEST.yaml schema.
BASE_URL=$(awk -F'"' '/^base_url:/ {print $2; exit}' "$MANIFEST")
if [ -z "$BASE_URL" ]; then
    BASE_URL=$(awk '/^base_url:/ {print $2; exit}' "$MANIFEST" | tr -d '"')
fi
if [ -z "$BASE_URL" ]; then
    echo "no base_url in $MANIFEST" >&2
    exit 2
fi
echo "[fetch] manifest : $MANIFEST"
echo "[fetch] dest     : $DEST"
echo "[fetch] base_url : $BASE_URL"

# Extract per-species filename lists from the manifest. Format:
#   - residue: NAP
#     files:
#       lib:    NAP.lib
#       frcmod: NAP.frcmod
#       mol2:   NAP.mol2
python - <<PY > /tmp/amber_cofactor_dl.list
import re, sys, pathlib
m = pathlib.Path("$MANIFEST").read_text()
cur = None
for raw in m.splitlines():
    ln = raw.rstrip()
    if re.match(r"\s*-\s*residue:\s*(\S+)", ln):
        cur = re.match(r"\s*-\s*residue:\s*(\S+)", ln).group(1)
    m2 = re.match(r"\s+(lib|frcmod|mol2):\s+(\S+)", ln)
    if cur and m2:
        print(f"{cur}\t{m2.group(2)}")
PY

GOT=0
SKIPPED=0
FAILED=0
while IFS=$'\t' read -r residue fname; do
    [ -z "$residue" ] && continue
    dst="$DEST/$fname"
    if [ -s "$dst" ]; then
        printf '[fetch] %-12s already present (%d bytes), skipping\n' \
            "$fname" "$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst")"
        SKIPPED=$((SKIPPED+1))
        continue
    fi
    url="$BASE_URL/$fname"
    printf '[fetch] %-12s <- %s ... ' "$fname" "$url"
    if curl -fsSL --max-time 60 -o "$dst.partial" "$url" 2>/dev/null; then
        mv "$dst.partial" "$dst"
        echo "OK ($(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst") bytes)"
        GOT=$((GOT+1))
    else
        rm -f "$dst.partial"
        echo "FAILED"
        FAILED=$((FAILED+1))
    fi
done < /tmp/amber_cofactor_dl.list
rm -f /tmp/amber_cofactor_dl.list

echo
echo "[fetch] downloaded: $GOT, already present: $SKIPPED, failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    cat <<MSG

[fetch] Some files failed - the Bryce Lab URL layout may have changed.
[fetch] Options:
[fetch]   1) Browse http://amber.manchester.ac.uk/ and update base_url
[fetch]      in $MANIFEST.
[fetch]   2) Download the missing files manually and drop them into $DEST -
[fetch]      _curated_param_system_generator only checks file existence.
[fetch]   3) Run with --manifest /path/to/your.yaml pointing at a mirror.
MSG
    exit 1
fi

# Quick verification: each .lib should at least contain "!entry."
for f in "$DEST"/*.lib; do
    [ -e "$f" ] || continue
    if ! grep -q "!entry\." "$f"; then
        echo "[fetch] WARNING: $f does not look like a tleap library "
        echo "        (missing '!entry.' marker). Re-check the source URL."
    fi
done

echo "[fetch] done. _run_real will now use the curated path for any"
echo "        cofactor whose CofactorSpec points at a file present here."
