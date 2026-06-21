"""Store forward-compat migration: a run DB created by an OLDER schema (missing a
newer nullable column) must keep working after the lightweight ADD COLUMN pass in
``Store``. Regression for the paper-grade QC fix that persists
``docking_pose.score_type`` (so a stored GNINA Vina energy is self-describing and
can never be read back as a CNNscore)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from evoliez.db.schema import DockingPose
from evoliez.db.store import Store


def _legacy_docking_pose_db(db: Path) -> None:
    """A docking_pose table WITHOUT the score_type column (the pre-fix schema)."""
    con = sqlite3.connect(db)
    con.execute(
        "create table docking_pose (pose_id integer primary key, "
        "project_id integer, candidate_id text, method text, score real, "
        "confidence real, pose_cluster integer, ligand_rmsd_to_reference real)")
    con.execute("insert into docking_pose (project_id,candidate_id,method,score) "
                "values (1,'wt','gnina',-12.18)")
    con.commit()
    con.close()


def _cols(db: Path):
    return {r[1] for r in sqlite3.connect(db).execute(
        "PRAGMA table_info(docking_pose)")}


def test_store_adds_missing_score_type_column(tmp_path):
    db = tmp_path / "evoliez.sqlite"
    _legacy_docking_pose_db(db)
    assert "score_type" not in _cols(db)            # pre-condition: legacy schema

    Store(db)                                        # opening migrates in place
    assert "score_type" in _cols(db)                 # column added

    # the pre-existing row is untouched; the new column reads as NULL (never a
    # fabricated value that could be misread as a score type)
    row = list(sqlite3.connect(db).execute(
        "select candidate_id,score,score_type from docking_pose"))
    assert row == [("wt", -12.18, None)]


def test_store_migration_is_idempotent_and_allows_typed_insert(tmp_path):
    db = tmp_path / "evoliez.sqlite"
    _legacy_docking_pose_db(db)
    st = Store(db)
    Store(db)                                        # second open: no error, no dup
    assert sum(1 for c in _cols(db) if c == "score_type") == 1

    # an insert that SETS score_type now succeeds against the migrated DB
    with st.session() as s:
        s.add(DockingPose(project_id=1, candidate_id="wt__cof", method="gnina",
                          score=0.897, score_type="minimizedAffinity",
                          confidence=1.0, pose_cluster=0,
                          ligand_rmsd_to_reference=0.0))
    got = list(sqlite3.connect(db).execute(
        "select score,score_type from docking_pose where candidate_id='wt__cof'"))
    assert got == [(0.897, "minimizedAffinity")]


def test_fresh_store_has_score_type_natively(tmp_path):
    # a brand-new run DB created by the current schema already has the column
    db = tmp_path / "fresh.sqlite"
    Store(db)
    assert "score_type" in _cols(db)
