"""Thin session/engine wrapper around the run's SQLite DB."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from evoliez.db.schema import Base


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)

        # On the shared server the Snakefile chains stages that reopen the SAME
        # sqlite file, and per-stage iteration can overlap runs. WAL allows a
        # reader alongside a writer, busy_timeout makes a contended write WAIT
        # instead of failing instantly with "database is locked", and
        # foreign_keys enforces the cross-table links SQLite ignores by default.
        @event.listens_for(self.engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=10000")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        Base.metadata.create_all(self.engine)
        self._add_missing_columns()
        self._Session = sessionmaker(bind=self.engine, future=True)

    def _add_missing_columns(self) -> None:
        """Lightweight forward-compat migration for run DBs created by an OLDER
        schema. ``create_all`` only creates missing TABLES, never missing COLUMNS,
        so a new nullable column (e.g. ``docking_pose.score_type``) added to the
        ORM would be absent in a pre-existing run's SQLite file and every INSERT
        would fail with "no column named ...". For each ORM table that already
        exists on disk, ADD COLUMN for any nullable column the file lacks. Only
        nullable/defaulted columns are added (SQLite cannot ADD a NOT-NULL column
        without a default), so this never corrupts existing rows — they read the
        new column as NULL. Idempotent."""
        from sqlalchemy import inspect, text

        insp = inspect(self.engine)
        existing_tables = set(insp.get_table_names())
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if table.name not in existing_tables:
                    continue  # create_all just made it with the full column set
                have = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name in have or not col.nullable:
                        continue
                    coltype = col.type.compile(dialect=self.engine.dialect)
                    conn.execute(text(
                        f'ALTER TABLE "{table.name}" '
                        f'ADD COLUMN "{col.name}" {coltype}'))

    def reset(self) -> None:
        """Drop every table and recreate it empty. Used when the run
        fingerprint changes (new target/config in a reused output_dir): the
        stages' idempotent 'skip if a row already exists' inserts would
        otherwise preserve the PREVIOUS run's rows and contaminate the
        DB-backed evidence (audit P0 #4)."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()
