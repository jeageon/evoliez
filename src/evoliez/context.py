"""Run context: the single object threaded through every stage.

Owns config, the directory layout, the DB, the logger, the resume state, and
the inter-stage artifact bus.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from evoliez.config import Config
from evoliez.db.schema import Project
from evoliez.db.store import Store
from evoliez.io.paths import ProjectPaths
from evoliez.logging_utils import get_logger, setup_logging

log = get_logger("evoliez.context")

# Server constraint: root "/" is ~99% full. Refuse to scatter big run outputs
# onto a filesystem with little headroom unless explicitly allowed.
_MIN_FREE_GB = 5.0


class RunContext:
    def __init__(self, config: Config, *, allow_small_disk: bool = False):
        self.config = config
        self.root = Path(config.project.output_dir).resolve()
        self.paths = ProjectPaths(self.root)
        self._allow_small_disk = allow_small_disk
        self.artifacts: Dict[str, Any] = {}  # rich, in-memory only
        self._state: Dict[str, Any] = {"completed_stages": [], "meta": {}}
        self.project_id: Optional[int] = None
        self.dry_run: bool = False
        self.store: Optional[Store] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def setup(self) -> "RunContext":
        self._check_disk()
        self.paths.create_all()
        setup_logging(logfile=self.paths.logs / "pipeline.log")
        self.store = Store(self.paths.db_path)
        self._load_state()
        self._ensure_project_row()
        log.info("project root: %s", self.root)
        return self

    def _check_disk(self) -> None:
        target = self.root
        probe = target
        while not probe.exists():
            probe = probe.parent
        free_gb = shutil.disk_usage(probe).free / 1e9
        if free_gb < _MIN_FREE_GB and not self._allow_small_disk:
            raise RuntimeError(
                f"only {free_gb:.1f} GB free on the filesystem holding "
                f"{target}. On the server, point project.output_dir at "
                f"/mnt/data2 (NOT '/'). Override with allow_small_disk=True "
                f"for tiny mock runs."
            )
        if free_gb < _MIN_FREE_GB:
            log.warning("low disk: %.1f GB free at %s", free_gb, probe)

    def _ensure_project_row(self) -> None:
        assert self.store is not None
        with self.store.session() as s:
            existing = s.query(Project).first()
            if existing is None:
                proj = Project(
                    target_name=self.config.input.target_id,
                    enzyme_family=self.config.input.ec_number,
                    objective=self.config.project.objective,
                    ligand_id=self.config.input.ligand.id,
                )
                s.add(proj)
                s.flush()
                self.project_id = proj.project_id
            else:
                self.project_id = existing.project_id

    # ------------------------------------------------------------------ #
    # resume state
    # ------------------------------------------------------------------ #
    def _load_state(self) -> None:
        sp = self.paths.state_path
        if sp.exists():
            self._state = json.loads(sp.read_text())
            self._state.setdefault("completed_stages", [])
            self._state.setdefault("meta", {})

    def _save_state(self) -> None:
        self.paths.state_path.write_text(json.dumps(self._state, indent=2, default=str))

    def is_stage_done(self, name: str) -> bool:
        return name in self._state.get("completed_stages", [])

    def mark_stage_done(self, name: str) -> None:
        if name not in self._state["completed_stages"]:
            self._state["completed_stages"].append(name)
        self._save_state()

    # ------------------------------------------------------------------ #
    # artifact bus
    # ------------------------------------------------------------------ #
    def put(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def persist_meta(self, key: str, value: Any) -> None:
        """Store a small JSON-safe value that survives process restarts."""
        self._state["meta"][key] = value
        self._save_state()

    def meta(self, key: str, default: Any = None) -> Any:
        return self._state.get("meta", {}).get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.artifacts:
            raise KeyError(
                f"missing artifact '{key}' - an upstream stage did not run"
            )
        return self.artifacts[key]
