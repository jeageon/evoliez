"""Server smoke proved that P0.5/P0.6 evidence + provenance fields were
computed on `cand.scores`/`cand.details` but NEVER reached the user-
facing CSV report - the writer carried a fixed column list. These tests
lock the new columns into `final_candidates.csv` so a future refactor
can't silently drop them.
"""

from __future__ import annotations

import csv
import inspect

from evoliez.io import report
from evoliez.io.report import _write_candidates_csv
from evoliez.types import Candidate, Mutation


def test_csv_writer_lists_evidence_and_provenance_columns():
    src = inspect.getsource(_write_candidates_csv)
    # P0.5 — evidence class + Boltz delta source + pool tag + uncertainty.
    for col in ("evidence_class", "pool", "boltz_delta_source",
                "uncertainty"):
        assert f'"{col}"' in src, f"final_candidates.csv missing {col}"
    # P0.1 / P0.3 — pose-quality surface.
    for col in ("pose_validity_status", "plif_recovery",
                "plif_recovery_min", "docking_method_disagreement"):
        assert f'"{col}"' in src, f"final_candidates.csv missing {col}"
    # P0.6 — MD provenance.
    for col in ("md_did_run", "md_passed", "md_status",
                "md_ligand_forcefield", "md_hmr_enabled",
                "md_timestep_fs", "md_replicas_run"):
        assert f'"{col}"' in src, f"final_candidates.csv missing {col}"


def test_csv_writer_emits_new_columns_per_candidate(tmp_path):
    # Build a minimal candidate carrying the new P0.5/P0.6 fields and
    # verify the writer renders them in the CSV row.
    class _Paths:
        def __init__(self, root):
            self.reports = root
            self.reports.mkdir(parents=True, exist_ok=True)
    c = Candidate(
        candidate_id="c1", mutations=[Mutation("A", 10, "K")],
        generator="test",
    )
    c.scores.update({
        "final_score": 3.14, "ml_score": 1.0, "ddg_fold": 0.5,
        "docking_score": -8.1, "md_lite_score": 0.5,
        "evidence_class": "Strong", "pool": "exploit",
        "boltz_delta_source": "real", "uncertainty": 0.12,
        "pose_validity_status": "valid", "plif_recovery": 0.9,
        "plif_recovery_min": 0.7, "docking_method_disagreement": 0.4,
        "md_did_run": 1, "md_status": "ok",
        "md_ligand_forcefield": "openff-2.2.0", "md_hmr_enabled": 0,
        "md_timestep_fs": 2.0, "md_replicas_run": 3,
    })
    c.details["md_passed"] = True
    out = _write_candidates_csv(_Paths(tmp_path), [c])
    with out.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    # P0.5
    assert row["evidence_class"] == "Strong"
    assert row["pool"] == "exploit"
    assert row["boltz_delta_source"] == "real"
    # P0.1 / P0.3
    assert row["pose_validity_status"] == "valid"
    assert row["plif_recovery"] == "0.9"
    # P0.6
    assert row["md_did_run"] == "1"
    assert row["md_passed"] == "1"
    assert row["md_ligand_forcefield"] == "openff-2.2.0"
    assert row["md_replicas_run"] == "3"


def test_csv_writer_handles_missing_fields_with_blanks(tmp_path):
    # Mock-pipeline candidates may not carry the new fields. Writer must
    # emit blank cells (not crash, not write "None").
    class _Paths:
        def __init__(self, root):
            self.reports = root
            self.reports.mkdir(parents=True, exist_ok=True)
    c = Candidate(
        candidate_id="c2", mutations=[Mutation("A", 10, "K")],
        generator="mock",
    )
    c.scores["final_score"] = 1.0
    out = _write_candidates_csv(_Paths(tmp_path), [c])
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    # Empty cells where nothing is set, so the column is present but blank.
    for col in ("evidence_class", "pool", "boltz_delta_source",
                "pose_validity_status", "md_did_run", "md_status"):
        assert col in row
        assert row[col] in ("", "0", "0.0"), f"{col} bad blank: {row[col]!r}"
