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
        self._Session = sessionmaker(bind=self.engine, future=True)

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
