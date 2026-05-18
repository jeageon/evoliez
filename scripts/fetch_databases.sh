#!/usr/bin/env bash
# OPTIONAL: download sequence DBs for local homolog search onto /mnt/data2.
# Skip this and set msa.remote_server=true to use a hosted MSA service instead.
#
# Usage:  bash scripts/fetch_databases.sh [uniref30|pdb70]
set -euo pipefail

DEST="${EVOLIEZ_DB_DIR:-/mnt/data2/$USER/evoliez_db}"
case "$DEST" in
  /mnt/data2/*|/mnt/data/*) : ;;
  *) echo "REFUSING: \$EVOLIEZ_DB_DIR ('$DEST') must be under /mnt/data2."; exit 1 ;;
esac
mkdir -p "$DEST"; cd "$DEST"
WHAT="${1:-uniref30}"

avail_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
echo ">> $avail_gb G free at $DEST (UniRef30 needs ~120 G unpacked)"
[ "${avail_gb:-0}" -lt 200 ] && echo "WARNING: may not fit; consider remote MSA."

case "$WHAT" in
  uniref30)
    echo ">> downloading UniRef30 (ColabFold mirror)"
    wget -c https://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2302.tar.gz
    tar xzf uniref30_2302.tar.gz
    echo "Set homologs.database: $DEST/uniref30_2302/uniref30_2302_db"
    ;;
  pdb70)
    wget -c https://wwwuser.gwdg.de/~compbiol/colabfold/pdb100_230517.tar.gz
    tar xzf pdb100_230517.tar.gz
    ;;
  *) echo "unknown DB: $WHAT (use: uniref30 | pdb70)"; exit 1 ;;
esac
