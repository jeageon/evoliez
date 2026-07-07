"""Audit P0 #4 — reusing an output_dir with a CHANGED fingerprint must not
leave stale SQLite rows / artifacts.

Fingerprint invalidation used to clear `_state.json` only; the DB kept the old
target's rows and the stages' idempotent 'skip if a row exists' inserts then
preserved them, contaminating DB-backed evidence. RunContext now purges the DB
+ stale artifact dirs on a fingerprint change (keeping checkpoints/ + logs/).
"""

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.db.schema import Sequence
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]

_SEQ_A = "MKVLAYCVRPDEEQHGNWFKST"
_SEQ_B = "GGSSAATTCCDDEEFFHHIIKKLLNN"     # different -> different fingerprint


def _cfg(out, seq):
    return load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {"project.output_dir": out,
         "input.target_sequence": seq,
         "input.catalytic_residues": [],
         "input.fixed_residues": [],
         "backend": "mock"},
    )


def test_changed_fingerprint_purges_db_and_artifacts(tmp_path, monkeypatch):
    # Purging is now OPT-IN behind a data-loss guard (a config change once deleted a
    # finished run). This test exercises the purge PATH, so it opts in explicitly.
    monkeypatch.setenv("EVOLIEZ_ALLOW_PURGE", "1")
    out = str(tmp_path / "run")

    # --- run A (target A) ---
    seed_everything(1234)
    ctxA = RunContext(_cfg(out, _SEQ_A), allow_small_disk=True).setup()
    Pipeline().run(ctxA, to_stage="s01_input")
    with ctxA.store.session() as s:
        row = s.query(Sequence).filter_by(source="target").one()
        assert _SEQ_A in row.fasta
    sentinel = ctxA.paths.structures / "stale_from_run_A.txt"
    sentinel.write_text("old")
    keep = ctxA.paths.checkpoints / "evoligand_gnn.pt"   # expensive, must survive
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("weights")

    # --- reuse the SAME dir with target B -> fingerprint changes on setup ---
    ctxB = RunContext(_cfg(out, _SEQ_B), allow_small_disk=True).setup()
    assert ctxB.invalidated
    assert not sentinel.exists()          # stale artifact purged
    assert keep.exists()                  # checkpoint preserved
    # DB was reset: no target row survived the purge
    with ctxB.store.session() as s:
        assert s.query(Sequence).filter_by(source="target").count() == 0

    # --- run B writes its OWN target, no contamination from A ---
    Pipeline().run(ctxB, to_stage="s01_input")
    with ctxB.store.session() as s:
        rows = s.query(Sequence).filter_by(source="target").all()
        assert len(rows) == 1
        assert _SEQ_B in rows[0].fasta and _SEQ_A not in rows[0].fasta


def test_same_fingerprint_keeps_db(tmp_path):
    # the control: re-entering with the SAME config must NOT purge (resume).
    out = str(tmp_path / "run")
    seed_everything(1234)
    ctx1 = RunContext(_cfg(out, _SEQ_A), allow_small_disk=True).setup()
    Pipeline().run(ctx1, to_stage="s01_input")
    ctx2 = RunContext(_cfg(out, _SEQ_A), allow_small_disk=True).setup()
    assert not ctx2.invalidated
    with ctx2.store.session() as s:                       # row still present
        assert s.query(Sequence).filter_by(source="target").count() == 1
