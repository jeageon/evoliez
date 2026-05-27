"""MD parameterisability preflight (P0 C).

Runs at **s01 time** (right after ligand parsing) instead of inside s10
per-candidate. Three wins:

  1. Honesty up-front. The report can label "MD-skipped because cofactor"
     before stages 02-09 even start, instead of after a 25-hour pipeline
     where every candidate independently discovers the same fact.
  2. Cache priming. Result is written to the same probe-cache sidecar
     (see openmm_engine._ligand_probe_cache_path) that s10's per-
     candidate `_ligand_system_generator` consults. Every s10 candidate
     becomes a cache hit — zero per-candidate probe overhead.
  3. Subprocess-isolation friendliness. Each fresh MD worker inherits
     the disk sidecar at startup, instead of re-paying the probe N
     times in N separate Python processes.

Status values:

  - ``curated_available`` — curated AMBER cofactor files are on disk;
    MD takes the tleap path (no OpenFF probe needed).
  - ``ok`` — OpenFF + openmmforcefields probe succeeded on this SMILES.
  - ``unsupported`` — no small-molecule FF can parameterise this ligand
    (e.g. AM1-BCC bouncing on real NADP+ without curated params). s10
    treats this as NEUTRAL ``skipped_parameterization``.
  - ``skipped_no_md_libs`` — OpenFF / openmmforcefields not importable
    (light mac venv). Defer the decision to s10.
  - ``skipped_md_disabled`` — ``mdcfg.enabled = False``. No probe ran.
  - ``skipped_no_smiles`` — defensive: ligand has no canonical SMILES.

Importing this module is cheap (no OpenMM at import time); the heavy
imports happen inside `run_md_preflight` and degrade gracefully when
absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evoliez.adapters.openmm_engine import (
    _LIGAND_PROBE_CACHE_MEM,
    _canonical_smiles_for_cache,
    _hydrate_probe_cache,
    _ligand_probe_cache_path,
    _persist_probe_result,
    _probe_cache_key,
)
from evoliez.features.cofactors import lookup_by_smiles
from evoliez.logging_utils import get_logger

log = get_logger("evoliez.md_preflight")


@dataclass
class MDPreflightResult:
    """Outcome of the up-front MD parameterisation check."""
    status: str
    ff_used: Optional[str] = None
    reason: Optional[str] = None


def _build_offmol(smiles: str):
    """OpenFF Molecule from SMILES. Returns None when OpenFF isn't
    installed (light mac venv) so the preflight degrades to a deferred
    decision instead of crashing s01."""
    try:
        from openff.toolkit.topology import Molecule
    except Exception as exc:
        log.info("MD preflight: openff.toolkit not importable (%s)", exc)
        return None
    try:
        return Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    except Exception as exc:
        log.warning("MD preflight: OpenFF couldn't parse SMILES (%s)", exc)
        return None


def _ff_priority(prefer_ff: Optional[str]) -> list[str]:
    """Match `_ligand_system_generator`'s priority order so the cache
    keys agree (s01 primes the same `(smiles, ff)` keys that s10 will
    look up)."""
    pref = prefer_ff or "openff-2.2.0"
    ffs: list[str] = [pref]
    if "gaff-2.11" not in ffs:
        ffs.append("gaff-2.11")
    try:
        import espaloma  # noqa: F401
        ffs.append("espaloma-0.3.2")
    except Exception:
        pass
    return ffs


def run_md_preflight(
    smiles: str,
    md_dir: Path,
    prefer_ff: Optional[str] = None,
    *,
    md_enabled: bool = True,
    subprocess_isolation: bool = False,
    timeout_seconds: int = 300,
) -> MDPreflightResult:
    """Run the parameterisation probe up-front and prime the disk sidecar.

    Parameters
    ----------
    smiles
        Canonical ligand SMILES from s01.
    md_dir
        ``ctx.paths.md`` — the MD-run directory. The probe-cache sidecar
        lives at ``md_dir/.ligand_probe_cache.json`` (same convention
        `_ligand_probe_cache_path` uses, so the sidecar is shared with
        per-candidate workdirs).
    prefer_ff
        Forwarded to the probe loop; matches `MDConfig.ligand_forcefield`.
    md_enabled
        Pass ``False`` to short-circuit when MD is disabled in config.
    subprocess_isolation
        G (post-expert-audit): when True, the probe runs in its own
        Python subprocess so a sqm hang inside `SystemGenerator.
        create_system` is bounded by ``timeout_seconds`` (default 300 s).
        Without this, s01 can hang indefinitely on a real cofactor
        without curated params — worse than today's s10 hang because
        the rest of the pipeline never starts. Off by default so the
        light mac venv (no openmmforcefields) keeps using the in-
        process path; server configs should turn it on.
    timeout_seconds
        Hard wall clock for the subprocess preflight. Status =
        ``"timeout_preflight"`` on hit.
    """
    if not md_enabled:
        return MDPreflightResult(status="skipped_md_disabled")
    if not smiles:
        return MDPreflightResult(status="skipped_no_smiles")

    if subprocess_isolation:
        return _run_md_preflight_in_subprocess(
            smiles, md_dir, prefer_ff, timeout_seconds=timeout_seconds,
        )
    return _run_md_preflight_inproc(smiles, md_dir, prefer_ff)


def _run_md_preflight_inproc(
    smiles: str,
    md_dir: Path,
    prefer_ff: Optional[str],
) -> MDPreflightResult:
    """The original synchronous implementation. Kept as the worker body
    AND as the fallback when subprocess isolation is disabled."""

    md_dir.mkdir(parents=True, exist_ok=True)

    # Curated cofactor short-circuit: if Bryce Lab files are on disk, MD
    # takes the tleap path entirely and never runs the OpenFF probe.
    # Skipping the probe here saves seconds AND avoids spuriously
    # marking the ligand "unsupported" for the OpenFF stack (which it
    # really IS, but the curated path doesn't care).
    spec = lookup_by_smiles(smiles)
    if spec is not None and spec.resolved_amber_files() is not None:
        log.info(
            "MD preflight: curated AMBER files available for %s — MD will "
            "take the tleap path (probe skipped)", spec.name,
        )
        return MDPreflightResult(
            status="curated_available",
            ff_used=f"amber:{spec.amber_residue_name}",
        )

    off_lig = _build_offmol(smiles)
    if off_lig is None:
        return MDPreflightResult(
            status="skipped_no_md_libs",
            reason="OpenFF / openff.toolkit not importable",
        )

    # `_ligand_probe_cache_path(workdir)` returns `workdir.parent /
    # .ligand_probe_cache.json`, so a synthetic per-candidate workdir
    # under md_dir lands the sidecar in md_dir — exactly where s10's
    # per-candidate code will read it.
    probe_disk = _ligand_probe_cache_path(md_dir / "_preflight")
    _hydrate_probe_cache(probe_disk)

    smiles_key = _canonical_smiles_for_cache(off_lig)
    if not smiles_key:
        return MDPreflightResult(
            status="skipped_no_smiles",
            reason="OpenFF molecule has no canonical SMILES",
        )

    try:
        import openmm.app as app
        from openmmforcefields.generators import SystemGenerator
    except Exception as exc:
        log.info(
            "MD preflight: openmmforcefields not importable (%s) — "
            "deferring to s10", exc,
        )
        return MDPreflightResult(
            status="skipped_no_md_libs",
            reason=f"openmmforcefields not importable: {exc}",
        )

    last_exc: Optional[Exception] = None
    for ff in _ff_priority(prefer_ff):
        key = _probe_cache_key(smiles_key, ff)
        cached = _LIGAND_PROBE_CACHE_MEM.get(key)
        if cached == "ok":
            log.info(
                "MD preflight: cache already records OK for %s — done", ff,
            )
            return MDPreflightResult(status="ok", ff_used=ff)
        if cached == "unsupported":
            log.info(
                "MD preflight: cache records UNSUPPORTED for %s — next FF",
                ff,
            )
            last_exc = last_exc or RuntimeError(f"{ff}: cached unsupported")
            continue
        try:
            sg = SystemGenerator(
                forcefields=["amber14-all.xml", "implicit/obc2.xml"],
                small_molecule_forcefield=ff,
                molecules=[off_lig],
                forcefield_kwargs={"constraints": app.HBonds},
                nonperiodic_forcefield_kwargs={
                    "nonbondedMethod": app.CutoffNonPeriodic,
                },
            )
            sg.create_system(
                off_lig.to_topology().to_openmm(), molecules=[off_lig],
            )
            _persist_probe_result(probe_disk, key, "ok")
            log.info(
                "MD preflight OK: ligand parameterised with %s "
                "(disk sidecar primed for s10)", ff,
            )
            return MDPreflightResult(status="ok", ff_used=ff)
        except Exception as exc:
            _persist_probe_result(probe_disk, key, "unsupported")
            last_exc = exc
            log.warning(
                "MD preflight: %s cannot parameterise the ligand (%s)",
                ff, str(exc)[:200],
            )

    log.warning(
        "MD preflight: NO small-molecule FF can parameterise this ligand. "
        "s10 will record neutral skipped_parameterization. Last error: %s",
        str(last_exc)[:200] if last_exc else "?",
    )
    return MDPreflightResult(
        status="unsupported",
        reason=(
            f"no small-molecule FF supports this ligand: "
            f"{str(last_exc)[:200] if last_exc else 'unknown'}"
        ),
    )


# --- G (post-expert-audit) — subprocess wrapper ------------------------
#
# `_run_md_preflight_inproc` calls `SystemGenerator(...).create_system(...)`
# synchronously. On a real cofactor without curated AMBER files, sqm
# inside antechamber can hang indefinitely (server-observed). A hang at
# s01 is worse than at s10 — no per-candidate retry, no other candidates
# in flight to mask it, the whole pipeline just stalls.
#
# The wrapper below pickles `(smiles, md_dir, prefer_ff)` into a tmp
# file, runs `evoliez.adapters.md_preflight_worker` as a fresh Python
# process with `start_new_session=True`, and enforces a hard wall-clock
# timeout. On TimeoutExpired we SIGKILL the entire process group
# (worker + any antechamber/sqm subprocesses it spawned) and return
# `status="timeout_preflight"` so the report can label the failure
# honestly.
#
# When the worker module isn't importable / pickling fails / unpickling
# fails, we fall back to in-process so the pipeline still runs (just
# without timeout protection). This matches the openmm_subprocess
# fallback policy.

import pickle as _pickle  # noqa: E402 — keep heavy imports out of the
import subprocess as _subprocess  # noqa: E402   module-import critical path
import sys as _sys  # noqa: E402
import time as _time  # noqa: E402

_WORKER_MODULE = "evoliez.adapters.md_preflight_worker"
_INPUTS_PKL = "_preflight_inputs.pkl"
_RESULT_PKL = "_preflight_result.pkl"
_WORKER_LOG = "_preflight_worker.log"


def _run_md_preflight_in_subprocess(
    smiles: str,
    md_dir: Path,
    prefer_ff: Optional[str],
    *,
    timeout_seconds: int,
) -> MDPreflightResult:
    """Spawn the preflight worker; enforce a wall-clock timeout."""
    md_dir.mkdir(parents=True, exist_ok=True)
    workdir = md_dir / "_preflight"
    workdir.mkdir(parents=True, exist_ok=True)

    inputs_path = workdir / _INPUTS_PKL
    result_path = workdir / _RESULT_PKL
    log_path = workdir / _WORKER_LOG

    # Stale pickle cleanup so a previous run's result can't leak into
    # this one (the worker writes result_path on success; absence is
    # how we detect a crash).
    for p in (inputs_path, result_path):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    try:
        with inputs_path.open("wb") as fh:
            _pickle.dump(
                {
                    "smiles": smiles,
                    "md_dir": str(md_dir),
                    "prefer_ff": prefer_ff,
                },
                fh,
            )
    except Exception as exc:  # noqa: BLE001
        # M (post-expert-audit): when the caller explicitly requested
        # subprocess isolation, a silent in-process fallback would
        # undermine the wall-clock guarantee the caller asked for.
        # Fail-closed instead so the report can surface "preflight
        # infrastructure broken" honestly.
        log.error(
            "MD preflight: failed to pickle inputs (%s) — fail-closed "
            "(caller requested subprocess_isolation=True)", exc,
        )
        return MDPreflightResult(
            status="failed_preflight",
            reason=f"preflight inputs pickle failed: {exc}",
        )

    cmd = [
        _sys.executable, "-u",
        "-m", _WORKER_MODULE,
        str(inputs_path), str(result_path),
    ]
    log.info(
        "MD preflight: launching subprocess (timeout=%ds, workdir=%s)",
        timeout_seconds, workdir,
    )
    start = _time.monotonic()
    proc = None
    try:
        with log_path.open("wb") as logf:
            proc = _subprocess.Popen(
                cmd,
                stdout=logf, stderr=_subprocess.STDOUT,
                start_new_session=True,    # SIGKILL hits sqm too
            )
        try:
            proc.wait(timeout=timeout_seconds)
        except _subprocess.TimeoutExpired:
            log.warning(
                "MD preflight: exceeded timeout=%ds; killing process group",
                timeout_seconds,
            )
            _kill_process_group(proc)
            return MDPreflightResult(
                status="timeout_preflight",
                reason=(
                    f"preflight subprocess timeout after {timeout_seconds}s "
                    f"(probe likely hung in sqm / antechamber)"
                ),
            )
    except Exception as exc:  # noqa: BLE001
        # M (post-expert-audit): the caller asked for subprocess
        # isolation — Popen failing is an env-level problem (fork
        # limits, missing python, FS issue). Silently running in-process
        # would erase the timeout guarantee. Fail-closed.
        log.error(
            "MD preflight: subprocess launch failed (%s) — fail-closed "
            "(caller requested subprocess_isolation=True)", exc,
        )
        if proc is not None:
            _kill_process_group(proc)
        return MDPreflightResult(
            status="failed_preflight",
            reason=f"preflight subprocess launch failed: {exc}",
        )

    elapsed = _time.monotonic() - start
    rc = proc.returncode if proc is not None else -1
    if rc != 0 or not result_path.exists():
        tail = _read_log_tail(log_path)
        log.warning(
            "MD preflight: subprocess crashed (rc=%d, %.1fs); tail=%s",
            rc, elapsed, tail[:200],
        )
        return MDPreflightResult(
            status="failed_preflight",
            reason=(
                f"preflight subprocess crashed (rc={rc}) "
                f"after {elapsed:.1f}s: {tail[-400:]}"
            ),
        )

    try:
        with result_path.open("rb") as fh:
            result: MDPreflightResult = _pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "MD preflight: failed to unpickle result (%s) — treating "
            "as failed_preflight", exc,
        )
        return MDPreflightResult(
            status="failed_preflight",
            reason=f"preflight unpickle error: {exc}",
        )

    log.info(
        "MD preflight: subprocess finished status=%s in %.1fs",
        result.status, elapsed,
    )
    return result


def _kill_process_group(proc) -> None:
    """SIGTERM then SIGKILL the process group started with
    ``start_new_session=True``. Mirrors openmm_subprocess._kill_process_group
    so we don't leak antechamber/sqm subprocesses."""
    import os
    import signal
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except _subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except _subprocess.TimeoutExpired:
            pass


def _read_log_tail(log_path: Path, max_bytes: int = 4096) -> str:
    """Best-effort tail read for a debuggable failure_reason."""
    try:
        data = log_path.read_bytes()
        return data[-max_bytes:].decode("utf-8", errors="replace")
    except Exception:
        return ""
