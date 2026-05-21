"""PseFDH (UniProt P33160) literature-derived benchmark scaffold.

The scaffold in `examples/pseudomonas_fdh/benchmark.csv` is keyed to
UniProt P33160 numbering (NOT the older PDB-based numbering most papers
use). These tests pin two contracts:

  1. The CSV is loadable via :func:`evoliez.ml.benchmark.load_benchmark`
     and contains only `beneficial` / `deleterious` / `neutral` labels.
  2. If a `target.fasta` is present alongside the CSV, every row's WT
     letter matches the residue at that position in the FASTA. If the
     FASTA isn't checked in yet (it's fetched on the server via
     ``scripts/fetch_target_fasta.sh P33160``), the WT-match check is
     skipped rather than fabricated.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from evoliez.ml.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "examples" / "pseudomonas_fdh" / "benchmark.csv"
FASTA = ROOT / "examples" / "pseudomonas_fdh" / "target.fasta"

_VALID_LABELS = {"beneficial", "deleterious", "neutral"}


def _load_rows() -> List[dict]:
    assert BENCH.exists(), f"missing scaffold benchmark at {BENCH}"
    return load_benchmark(BENCH)


def test_psefdh_benchmark_loads_via_load_benchmark():
    rows = _load_rows()
    # Scaffold contract: small but non-empty (5-10 rows per the task spec).
    assert 1 <= len(rows) <= 10, f"unexpected scaffold size: {len(rows)}"
    for row in rows:
        assert row["mutation"], f"empty mutation in row {row}"
        assert row["label"] in _VALID_LABELS, (
            f"row {row['mutation']} has unsupported label {row['label']!r}; "
            f"must be one of {sorted(_VALID_LABELS)}"
        )
        # activity must parse (load_benchmark already coerces to float|None).
        assert row["activity"] is None or isinstance(row["activity"], float)


def test_psefdh_benchmark_has_documented_switch_target():
    # The whole point of this scaffold is the D222 NAD->NADP switch
    # (UniProt P33160 equivalent of literature D196). If a refactor drops
    # the D222 rows the scaffold has lost its anchor mutation.
    rows = _load_rows()
    d222 = [r for r in rows if r["mutation"].startswith("D222")]
    assert d222, (
        "scaffold must retain at least one D222 mutation - it's the "
        "documented NAD->NADP specificity switch target in "
        "configs/server_fdh_nadp.yaml"
    )
    # And the catalytic-deleterious controls.
    cat = [r for r in rows if r["mutation"][0] in "RH" and r["label"] == "deleterious"]
    assert cat, (
        "scaffold must retain catalytic-residue deleterious controls "
        "(R285*, H333*) so the deleterious-avoidance metric has signal"
    )


def test_psefdh_benchmark_wt_letters_match_fasta_when_present():
    if not FASTA.exists():
        pytest.skip(
            "target.fasta not checked in; fetched on the server via "
            "scripts/fetch_target_fasta.sh P33160. Sequence-consistency "
            "is verified by `evoliez bench` at runtime instead."
        )
    seq = "".join(
        line.strip()
        for line in FASTA.read_text().splitlines()
        if line and not line.startswith(">")
    )
    rows = _load_rows()
    mismatches = []
    for row in rows:
        m = row["mutation"]
        # Single mutation form: <WT><pos><MUT> (e.g. "D222S"). Multi-mutants
        # use ';' as per load_benchmark; this scaffold has none currently.
        for token in m.split(";"):
            wt = token[0]
            try:
                pos = int(token[1:-1])
            except ValueError:
                mismatches.append(f"unparseable mutation: {token!r}")
                continue
            if pos < 1 or pos > len(seq):
                mismatches.append(
                    f"{token}: position {pos} out of range (sequence length {len(seq)})"
                )
                continue
            actual = seq[pos - 1]
            if actual != wt:
                mismatches.append(
                    f"{token}: WT letter {wt!r} does not match "
                    f"target.fasta position {pos} (actual {actual!r})"
                )
    assert not mismatches, "PseFDH P33160 numbering mismatches:\n" + "\n".join(mismatches)
