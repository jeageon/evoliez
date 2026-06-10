"""Run context: the single object threaded through every stage.

Owns config, the directory layout, the DB, the logger, the resume state, and
the inter-stage artifact bus.
"""

from __future__ import annotations

import hashlib
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
        self._state: Dict[str, Any] = {
            "completed_stages": [], "meta": {}, "fingerprint": {}
        }
        self.project_id: Optional[int] = None
        self.dry_run: bool = False
        self.store: Optional[Store] = None
        self.invalidated: bool = False  # resume state wiped (inputs changed)

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
    @staticmethod
    def _sha1(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:16]

    def run_fingerprint(self) -> Dict[str, str]:
        """Identity of this run. Any change here invalidates the resume
        checkpoint so stale artifacts are never silently reused (expert
        review: config / input / backend / software-version invalidation)."""
        from evoliez import __version__
        from evoliez.io.provenance import RANKING_FORMULA_VERSION

        ci = self.config.input
        if ci.target_sequence:
            seq = ci.target_sequence.strip().upper()
        elif ci.target_fasta and Path(ci.target_fasta).exists():
            seq = Path(ci.target_fasta).read_text()
        else:
            seq = ci.target_fasta or ""
        input_blob = f"{seq}|{ci.ligand.type}|{ci.ligand.value}"
        cfg_blob = json.dumps(
            self.config.model_dump(mode="json"), sort_keys=True, default=str
        )
        # Pin the GNN weights by CONTENT: an in-place `train-gnn` at the same
        # path changes ml_score but not the config, so without this a retrain
        # would not invalidate the resume checkpoint. "none" when GNN disabled,
        # "missing" when enabled but the file is absent (both stable).
        gnn_ckpt_sha = "none"
        g = self.config.gnn
        if g.enabled:
            from evoliez.io.provenance import _sha256_file
            ckpt = str(self.paths.root / g.checkpoint)
            gnn_ckpt_sha = _sha256_file(ckpt) or "missing"
        return {
            "evoliez_version": __version__,
            "ranking_formula_version": RANKING_FORMULA_VERSION,
            "backend": self.config.backend.value,
            # dry-run artifacts must never be reused by a later real
            # --resume into the same output dir (and vice-versa).
            "dry_run": "1" if self.dry_run else "0",
            "input_sha1": self._sha1(input_blob),
            "config_sha1": self._sha1(cfg_blob),
            "gnn_checkpoint_sha256": gnn_ckpt_sha,
        }

    def _load_state(self) -> None:
        sp = self.paths.state_path
        current = self.run_fingerprint()
        if sp.exists():
            try:
                self._state = json.loads(sp.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                # a corrupt _state.json (power-loss / partial NFS write / a
                # pre-fix crash) must NOT brick --resume: discard it and re-run
                # from scratch rather than aborting setup().
                log.warning(
                    "resume checkpoint %s is unreadable/corrupt (%s); "
                    "discarding it and re-running all stages", sp, exc,
                )
                self._state = {}
                self.invalidated = True
            self._state.setdefault("completed_stages", [])
            self._state.setdefault("meta", {})
            stored = self._state.get("fingerprint", {})
            if stored and stored != current:
                changed = [
                    k for k in current
                    if stored.get(k) != current.get(k)
                ]
                log.warning(
                    "run fingerprint changed (%s) - invalidating resume "
                    "checkpoint; all stages will re-run",
                    ", ".join(changed),
                )
                self._state["completed_stages"] = []
                self._state["meta"] = {}
                self.invalidated = True
        self._state["fingerprint"] = current
        self._save_state()

    def _save_state(self) -> None:
        # Atomic + DURABLE write: a crash / full-disk / power-loss mid-write
        # would otherwise leave a truncated _state.json that the next --resume
        # fails to parse, bricking the checkpoint. fsync the data before the
        # rename and the directory after, so recovery sees either the old or the
        # new complete file - never a torn write.
        target = self.paths.state_path
        tmp = target.with_name(target.name + ".tmp")
        data = json.dumps(self._state, indent=2, default=str)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # directory fsync unsupported on some platforms; rename is atomic

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
