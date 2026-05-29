"""Thin session/engine wrapper around the run's SQLite DB."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from evoliez.db.schema import Base


def _enable_sqlite_concurrency(dbapi_conn, _conn_record):
    """Per-connection PRAGMAs so concurrent stages / a parallel figures run
    don't trip 'database is locked'. WAL lets readers and a writer coexist;
    busy_timeout makes a momentarily-locked write wait instead of failing
    immediately. Applied on every new DBAPI connection via the engine's
    ``connect`` event."""
    cur = dbapi_conn.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    finally:
        cur.close()


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # timeout=30 -> the sqlite3 driver itself waits up to 30s on a locked
        # DB before raising; the busy_timeout PRAGMA below reinforces this at
        # the engine level for connections that bypass the driver default.
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            future=True,
            connect_args={"timeout": 30},
        )
        event.listen(self.engine, "connect", _enable_sqlite_concurrency)
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
