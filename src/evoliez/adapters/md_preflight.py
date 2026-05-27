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
    """
    if not md_enabled:
        return MDPreflightResult(status="skipped_md_disabled")
    if not smiles:
        return MDPreflightResult(status="skipped_no_smiles")

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
