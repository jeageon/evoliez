#!/usr/bin/env bash
# scripts/server_test_boltz_isolation.sh
#
# SURGICAL test for the boltz output-discovery cross-contamination bug.
# Does NOT re-run Boltz or MD - just exercises the discovery + parsing
# logic against the per-mutant Boltz outputs ALREADY on disk from a
# previous smoke run. Settles in ~1-2 seconds instead of 25-30 minutes
# of full per-mutant smoke.
#
# Convention: when a fix is localized to a downstream stage, write a
# focused script that exercises ONLY that stage's surface using the
# upstream artifacts already on disk. Saves GPU + wall time + iterates
# fast on residual bugs.
#
# Pre-requisite: at least one per-mutant smoke run has completed, so
# `<EVOLIEZ_ROOT>/runs/smoke/md/complexes/mutant_boltz/boltz_results_*`
# exists. (Today's session has this.)
#
# Usage:
#   bash scripts/server_test_boltz_isolation.sh
#   bash scripts/server_test_boltz_isolation.sh /alt/path/complexes/mutant_boltz

set -u -o pipefail
cd "$(dirname "$0")/.."

DEFAULT_DIR="${EVOLIEZ_ROOT:-/mnt/data/${USER}}/runs/smoke/md/complexes/mutant_boltz"
MUT_DIR="${1:-$DEFAULT_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=runs/server_test
LOG=$LOG_DIR/boltz_isolation_${TS}.log
mkdir -p "$LOG_DIR"

say()    { printf '%s\n' "$*" | tee -a "$LOG" ; }
banner() { printf '\n========== %s ==========\n' "$*" | tee -a "$LOG" ; }

say "[isolation] branch:  $(git rev-parse --abbrev-ref HEAD)"
say "[isolation] HEAD:    $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
say "[isolation] env:     ${CONDA_DEFAULT_ENV:-?}"
say "[isolation] mut_dir: $MUT_DIR"
say "[isolation] log:     $LOG"

if [ ! -d "$MUT_DIR" ]; then
    say "[isolation] FAIL: $MUT_DIR does not exist"
    say "[isolation] run scripts/server_test_per_mutant_md.sh once first to"
    say "[isolation] produce the per-mutant boltz_results_* outputs we're"
    say "[isolation] going to inspect here."
    exit 2
fi

banner "Per-mutant boltz_results_* directories on disk"
ls -1 "$MUT_DIR" | grep -E "^boltz_results_mut" | sed 's/^/  /' | tee -a "$LOG"

# Surgical test: exercise predict_complex's output discovery + sequence
# parsing per label, WITHOUT invoking the boltz binary. The patched
# `boltz.run` is a no-op so we skip the GPU/MSA work entirely; the rest
# of predict_complex (glob, _parse_real_structure, _parse_real_samples)
# runs against the outputs already on disk.
banner "Per-label discovery verdict (no Boltz invocation, no MD)"
MUT_DIR="$MUT_DIR" python - <<'PY' 2>&1 | tee -a "$LOG"
import os, sys
from pathlib import Path
from unittest.mock import patch

mut_dir = Path(os.environ["MUT_DIR"])
labels = sorted({
    p.name.replace("boltz_results_", "").replace("_boltz_input", "")
    for p in mut_dir.glob("boltz_results_mut_*")
})
if not labels:
    print("no boltz_results_mut_* directories found")
    sys.exit(1)
print(f"found {len(labels)} per-mutant outputs: {labels}")

# evoliez imports
from evoliez.adapters import boltz
from evoliez.config import Backend, load_config
from evoliez.features.cofactors import resolve_ligand_spec
from evoliez.features.ligand import parse_ligand
from evoliez.stages.s08b_mutant_boltz import _mutant_sequence

cfg = load_config("configs/smoke.yaml")
ic = cfg.input
wt_seq = (ic.target_sequence or "").strip().upper()
if not wt_seq and ic.target_fasta and Path(ic.target_fasta).exists():
    wt_seq = "".join(
        l.strip() for l in Path(ic.target_fasta).read_text().splitlines()
        if l and not l.startswith(">")
    ).upper()
ligand = parse_ligand(resolve_ligand_spec(ic))

# Build minimal candidates so we can compute their intended mutant sequence
from evoliez.types import Mutation, Candidate

def _toy_cand(label, position, mut):
    return Candidate(candidate_id=label, mutations=[Mutation("A", position, mut)],
                     generator="probe")

# Use the most common mutation position for an arbitrary mut_id - we just
# need ONE residue different from WT so _pdb_one_letter_seq vs intended
# mutant sequence comparison is meaningful. For real validation we'd
# read each label's mut from the previous smoke's mutation_level.csv,
# but for this surgical test we only need the discovery + path test.
def _intended_seq_for(label: str) -> str:
    # Heuristic: each label corresponds to a unique mutation; here we
    # don't actually need the right mutant sequence - we just need to
    # verify that predict_complex returns the LABEL's OWN PDB. The
    # sequence guard in _run_real catches cross-contamination by
    # comparing two sequences read from DIFFERENT PDBs. Our surgical
    # check: discovered structure file MUST be under boltz_results_<label>_*.
    return wt_seq  # any seq works for the discovery check

# Patch the binary run so predict_complex skips the boltz GPU invocation
# but still walks the discovery + parsing path. dry_run=False keeps the
# real discovery branch active.
calls = []
def noop_run(cmd, *args, **kwargs):
    calls.append(list(cmd) if isinstance(cmd, (list, tuple)) else [cmd])
    return None

results = []
with patch.object(boltz, "run", side_effect=noop_run):
    for label in labels:
        try:
            cx = boltz.predict_complex(
                label, _intended_seq_for(label), ligand,
                cfg.complex_prediction, mut_dir,
                backend=Backend.real, dry_run=False,
            )
            path = str(cx.path or "")
            # Discovery invariant: returned PDB must be under THIS label's
            # results directory, not someone else's (the bug).
            expected_sub = f"boltz_results_{label}_boltz_input"
            isolated = expected_sub in path
            results.append((label, path, isolated))
        except Exception as exc:
            results.append((label, f"ERROR {exc!r}", False))

print()
print(f"  {'label':<14}{'isolated':<11}path")
print("  " + "-" * 110)
n_ok = 0
for label, path, isolated in results:
    tag = "OK" if isolated else "CONTAMIN"
    if isolated: n_ok += 1
    print(f"  {label:<14}{tag:<11}{path}")

print()
print(f"[isolation] {n_ok}/{len(results)} per-mutant outputs correctly isolated")
if n_ok < len(results):
    print("[isolation] FAIL: at least one mutant returned another mutant's PDB")
    print("[isolation]   This is the cross-contamination bug - the discovery")
    print("[isolation]   glob is not properly scoped to boltz_results_<label>_*")
    sys.exit(3)
print("[isolation] OK: every mutant's discovery returned its OWN PDB path.")
print("[isolation]   The next full per-mutant smoke should produce")
print("[isolation]   `MD real-execution: <N>/<N>` with no sequence-guard skips.")
PY
RC=${PIPESTATUS[0]}

banner "Summary"
say "[isolation] exit code: $RC"
say "[isolation]   0 = every mutant's output is properly isolated (fix works)"
say "[isolation]   2 = no per-mutant outputs on disk yet (run smoke first)"
say "[isolation]   3 = cross-contamination detected (fix didn't take)"
say "[isolation] full log: $LOG"
exit $RC
