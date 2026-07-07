"""Consistent PROVENANCE STAMP for every EvoLiEZ HTML report + run output.

Date alone is insufficient — two same-day re-runs with similar inputs would show
an identical ``generated <date>`` subtitle and be indistinguishable. This module
makes every report self-identifying: run name, generation timestamp, the config
fingerprint (the same sha1 the resume checkpoint keys on), the source git
commit, the EvoLiEZ version, target/length, and — the key SAFETY field — each
ligand's id with its RDKit-computed NET FORMAL CHARGE. A wrong protonation (e.g.
NADP entered as net +1) then surfaces on the header of EVERY report rather than
silently flowing through the whole pipeline.

Everything is best-effort and GENERIC: no ligand/target identity is hardcoded,
net charge is computed from whatever SMILES is present, and a missing/unparseable
field degrades to a blank (``''`` / ``None``) instead of raising — so the stamp
still renders when reconstructed standalone from a finished run dir (the
``scripts/gen_s0*_report.py`` helpers) where parts of the in-memory context are
unavailable.
"""

from __future__ import annotations

import html as _html
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# field collection (every accessor is best-effort: never raise)
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    """Best-effort short HEAD sha of the EvoLiEZ source tree ('' on any
    failure: outside a repo, git missing, timeout). A ``-dirty`` suffix flags
    uncommitted changes so a stamp can be tied to exactly the code that ran."""
    try:
        repo = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return ""
        sha = out.stdout.strip()
        try:
            dirty = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            dirty = ""
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return ""


def _net_formal_charge(smiles: Optional[str]) -> Optional[int]:
    """RDKit net formal charge of a ligand SMILES (``None`` when RDKit is
    unavailable or the SMILES does not parse). This is the safety feature: a
    mis-protonated ligand shows a non-zero net charge on every report."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except Exception:
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return int(Chem.GetFormalCharge(m))
    except Exception:
        return None


def _evoliez_version() -> str:
    try:
        from evoliez import __version__
        return __version__
    except Exception:
        return ""


def _config_fingerprint(ctx: Any) -> str:
    """First 12 chars of the run's config sha1 — the SAME hash the resume
    checkpoint keys on (``RunContext.run_fingerprint()['config_sha1']``), so the
    stamp on a report matches ``_state.json`` for that run. Recomputed directly
    from the config when the ctx cannot produce a full fingerprint (e.g. no DB /
    standalone reconstruction)."""
    # Preferred: the real run fingerprint (matches _state.json exactly).
    try:
        fp = ctx.run_fingerprint().get("config_sha1")
        if fp:
            return str(fp)[:12]
    except Exception:
        pass
    # Fallback: recompute config_sha1 the same way RunContext does, so a
    # standalone caller still gets a stable, matching fingerprint.
    try:
        import hashlib
        cfg = getattr(ctx, "config", None)
        if cfg is None:
            return ""
        blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True,
                          default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]
    except Exception:
        return ""


def _ligand_entry(lig: Any) -> Dict[str, Any]:
    """{id, net_formal_charge} for a config LigandInput (type/value). Net charge
    only when type == 'smiles' (a path/CCD code has no SMILES to read here)."""
    lid = getattr(lig, "id", None) or "ligand"
    ltype = getattr(lig, "type", None)
    lval = getattr(lig, "value", None)
    smiles = lval if ltype == "smiles" else None
    return {"id": str(lid), "net_formal_charge": _net_formal_charge(smiles)}


def _seq_len(ctx: Any) -> Optional[int]:
    """Target sequence length, best-effort: the in-context parsed sequence
    first, else the inline config sequence, else a target.fasta on disk."""
    try:
        seq = ctx.get("target_sequence") if hasattr(ctx, "get") else None
        if seq:
            return len("".join(str(seq).split()))
    except Exception:
        pass
    cfg = getattr(ctx, "config", None)
    ci = getattr(cfg, "input", None) if cfg is not None else None
    if ci is not None:
        if getattr(ci, "target_sequence", None):
            return len(ci.target_sequence.strip())
        fa = getattr(ci, "target_fasta", None)
        if fa:
            try:
                p = Path(fa)
                if p.exists():
                    return sum(len(ln.strip()) for ln in p.read_text().splitlines()
                               if ln and not ln.startswith(">"))
            except Exception:
                pass
    return None


def _docking_backends(cfg: Any) -> List[str]:
    """The docking engines this run is configured to use, de-duplicated in a
    stable order: s06b multi_engine_methods (per-rep augmentation) + the s05
    redocking methods. Generic — whatever the config lists."""
    out: List[str] = []
    try:
        adv = getattr(cfg, "advanced", None)
        for m in (getattr(adv, "multi_engine_methods", None) or []):
            if m and m not in out:
                out.append(str(m))
    except Exception:
        pass
    try:
        red = getattr(getattr(cfg, "validation", None), "redocking", None)
        for m in (getattr(red, "methods", None) or []):
            if m and m not in out:
                out.append(str(m))
    except Exception:
        pass
    return out


def provenance_fields(ctx: Any) -> Dict[str, Any]:
    """The full self-identifying record for a run, assembled best-effort from a
    RunContext (or any object exposing ``.config`` and, ideally, ``.get`` /
    ``.run_fingerprint``). Every field degrades to ''/None/[] rather than
    raising, so this is safe to call from the standalone report scripts that
    reconstruct a partial context from a finished run dir."""
    cfg = getattr(ctx, "config", None)
    proj = getattr(cfg, "project", None) if cfg is not None else None
    ci = getattr(cfg, "input", None) if cfg is not None else None

    ligands: List[Dict[str, Any]] = []
    primary_lig = getattr(ci, "ligand", None) if ci is not None else None
    if primary_lig is not None:
        ligands.append(_ligand_entry(primary_lig))
    for el in (getattr(ci, "extra_ligands", None) or []) if ci is not None else []:
        ligands.append(_ligand_entry(el))

    primary_method = ""
    try:
        primary_method = str(getattr(cfg.complex_prediction, "primary_method", "")
                             or "")
    except Exception:
        primary_method = ""

    return {
        "run_name": (getattr(proj, "name", None) or "") if proj is not None else "",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_fingerprint": _config_fingerprint(ctx),
        "git_commit": _git_commit(),
        "evoliez_version": _evoliez_version(),
        "target_name": (getattr(ci, "target_id", None) or "") if ci is not None else "",
        "seq_len": _seq_len(ctx),
        "ligands": ligands,
        "docking_backends": _docking_backends(cfg),
        "primary_method": primary_method,
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt_charge(n: Optional[int]) -> str:
    """'net +1' / 'net -2' / 'net 0' for a known charge; 'net ?' when RDKit
    could not compute it (so the unknown is visible rather than implied 0)."""
    if n is None:
        return "net ?"
    return f"net {n:+d}" if n != 0 else "net 0"


def provenance_html(fields: Dict[str, Any]) -> str:
    """Compact muted one-line stamp reusing the existing ``.sub`` / ``var(--mut)``
    idiom (the caller drops it into a ``<p class="sub">`` directly under the
    report ``<h1>``). Each segment is omitted when its field is empty, so a
    standalone render with partial data still produces a clean line.

    Sample::

        run fdh_demo · generated 2026-06-21T00:09:11 · cfg 3f9a1c2b7e04 ·
        git b3ed305 · 384 aa · NADP (net +1) · formate (net -1) · docking [gnina, diffdock]
    """
    g = _html.escape
    parts: List[str] = []
    if fields.get("run_name"):
        parts.append(f"run {g(str(fields['run_name']))}")
    if fields.get("generated_at"):
        parts.append(f"generated {g(str(fields['generated_at']))}")
    if fields.get("config_fingerprint"):
        parts.append(f"cfg {g(str(fields['config_fingerprint']))}")
    if fields.get("git_commit"):
        parts.append(f"git {g(str(fields['git_commit']))}")
    if fields.get("evoliez_version"):
        parts.append(f"v{g(str(fields['evoliez_version']))}")
    if fields.get("seq_len"):
        parts.append(f"{g(str(fields['seq_len']))} aa")
    for lig in (fields.get("ligands") or []):
        lid = g(str(lig.get("id", "ligand")))
        parts.append(f"{lid} ({_fmt_charge(lig.get('net_formal_charge'))})")
    backends = fields.get("docking_backends") or []
    if backends:
        parts.append(f"docking [{g(', '.join(str(b) for b in backends))}]")
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# RUN_INFO.json — greppable per-run manifest
# --------------------------------------------------------------------------- #
def write_run_info(run_dir, fields: Dict[str, Any]):
    """Persist the full fields dict to ``<RUN_DIR>/RUN_INFO.json`` (a greppable
    manifest of who/what/when produced this run dir). Best-effort: returns the
    path on success, ``None`` if the write fails (never raises — a report or
    stage must not die because a manifest could not be written)."""
    try:
        p = Path(run_dir) / "RUN_INFO.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fields, indent=2, default=str), encoding="utf-8")
        return p
    except Exception:
        return None


def run_fingerprint_str(ctx: Any) -> str:
    """The full config-identity fingerprint as a single short string for tagging
    data JSONs: ``<config_fp12>`` (matches the report stamp's ``cfg`` field).
    Best-effort: '' when it cannot be derived."""
    return _config_fingerprint(ctx)


def tag_json(path, ctx: Any) -> None:
    """NON-BREAKING in-place enrichment of an already-written JSON object file:
    add top-level ``generated_at`` (ISO seconds) + ``run_fingerprint`` (config
    fp) so a data artifact is tied to the run that produced it. No-op on any
    failure or when the file isn't a JSON object (never raises — purely
    additive)."""
    try:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            return
        data.setdefault("generated_at",
                        datetime.now().isoformat(timespec="seconds"))
        data["run_fingerprint"] = run_fingerprint_str(ctx)
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        return


def stamp(ctx: Any, run_dir=None) -> str:
    """Convenience: compute the fields, write ``RUN_INFO.json`` (under
    ``run_dir`` when given, else ``ctx.paths.root``/``ctx.root``), and return the
    rendered ``provenance_html``. Fully best-effort — any failure still returns
    whatever stamp HTML could be built."""
    try:
        fields = provenance_fields(ctx)
    except Exception:
        fields = {"generated_at": datetime.now().isoformat(timespec="seconds")}
    target = run_dir
    if target is None:
        target = (getattr(getattr(ctx, "paths", None), "root", None)
                  or getattr(ctx, "root", None))
    if target is not None:
        write_run_info(target, fields)
    try:
        return provenance_html(fields)
    except Exception:
        return ""
