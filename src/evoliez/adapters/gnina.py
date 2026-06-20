"""GNINA CNN docking / rescoring (spec section 10). GPU tool (server only).

real: gnina with autobox around the reference ligand.
mock: deterministic perturbed pose (shared helper).

The s06b multi-engine redesign parses EVERY gnina mode (not just rank-1) with
full score provenance — ``minimizedAffinity`` (-> ``Pose.score``), ``CNNscore``,
``CNNaffinity`` — so the downstream hard-negative logic can find a pose that
scores well yet disagrees with the family consensus.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

from evoliez.adapters.base import (
    fail_unless_mock_allowed,
    full_atom_receptor_pdb,
    lock_pose_to_reference,
    mock_redock,
    mock_redock_modes,
    parse_sdf_all_poses,
    tool_version,
    write_min_pdb,
)
from evoliez.config import Backend, DockingConfig
from evoliez.logging_utils import get_logger
from evoliez.types import LigandAtom, Pose, ProteinStructure
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.gnina")
METHOD = "gnina"
SCORE_TYPE = "minimizedAffinity"


def redock(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    instability: float,
    backend: Backend,
    dry_run: bool = False,
    context_chains=None,
    receptor_pdb: Optional[Path] = None,
) -> Pose:
    """Back-compat single-pose entry point: the rank-1 (best) gnina mode.

    Kept verbatim for existing callers (s05, s06b, docking.py). Internally this
    now just takes the first of ``redock_all`` so the parse/provenance path is
    shared and rank-1 here is identical to ``redock_all(...)[0]``."""
    poses = redock_all(
        candidate_id, structure, reference_atoms, cfg, workdir,
        instability=instability, backend=backend, dry_run=dry_run,
        context_chains=context_chains, receptor_pdb=receptor_pdb,
    )
    return poses[0]


def redock_all(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    instability: float,
    backend: Backend,
    dry_run: bool = False,
    context_chains=None,
    receptor_pdb: Optional[Path] = None,
) -> List[Pose]:
    """ALL gnina modes (best first), each a fully-provenanced :class:`Pose`.

    Real backend: one ``gnina`` run writing N modes into a single SDF, then
    :func:`parse_all_modes` over every mode. Mock backend: a deterministic
    N-mode ranked ensemble (descending plausibility) so tests need no real tool.
    Always returns a NON-EMPTY list (rank-1 at index 0).

    ``receptor_pdb``: an optional pre-rendered receptor PDB. When given AND the
    file exists it is COPIED to the per-candidate ``rec`` path verbatim instead
    of re-rendering from ``structure.pdb_path`` — a perf win when the caller
    (s06b) renders one context-retained receptor per rep and reuses it across
    targets/engines. It MUST already carry the right cofactor/substrate context
    chains (gnina keeps ``keep_het_chains=context_chains``); a protein-only
    receptor here would silently drop that context. Default ``None`` = render
    exactly as before (byte-identical)."""
    if backend is Backend.real:
        return _redock_real_all(
            candidate_id, structure, reference_atoms, cfg, workdir,
            dry_run=dry_run, context_chains=context_chains,
            receptor_pdb=receptor_pdb,
        )
    # CNN scoring tends to be a touch more optimistic than Vina; small offset.
    return mock_redock_modes(
        candidate_id, METHOD, reference_atoms,
        n_modes=max(1, int(cfg.poses_per_candidate)),
        score_type=SCORE_TYPE, higher_is_better=False, with_cnn=True,
        base_instability=max(0.02, instability), score_offset=-0.5,
    )


def _gnina_cmd(rec: Path, lig: Path, out: Path, cfg: DockingConfig,
               *, seed: int, cpu: int) -> List[str]:
    """The gnina command line. Centralised so the exact arg list recorded into
    ``Pose.command_args`` is the one actually run (audit fidelity)."""
    return [
        "gnina", "-r", str(rec), "-l", str(lig), "--autobox_ligand", str(lig),
        "--num_modes", str(cfg.poses_per_candidate), "--cpu", str(cpu),
        "--seed", str(seed), "-o", str(out),
    ]


def _redock_real_all(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    dry_run: bool,
    context_chains=None,
    receptor_pdb: Optional[Path] = None,
) -> List[Pose]:
    require("gnina")
    apply_gpu_selection()
    workdir.mkdir(parents=True, exist_ok=True)
    rec = workdir / f"{candidate_id}_rec.pdb"
    lig = workdir / f"{candidate_id}_ref_lig.pdb"
    out = workdir / f"{candidate_id}_gnina_out.sdf"
    # Full-atom Boltz receptor; keep the OTHER co-modelled ligands' chains as
    # fixed context (physics-based gnina docks this ligand around them). Honest
    # mock fallback if the upstream structure is a CA-only trace.
    #
    # RENDER-ONCE: if the caller pre-rendered a context-retained receptor, copy
    # it to the expected ``rec`` path verbatim and skip the (re-)render. The
    # supplied PDB is assumed to already carry the right context chains; we do
    # not re-filter it (gnina would otherwise re-render once per target).
    if receptor_pdb is not None and Path(receptor_pdb).exists():
        shutil.copyfile(Path(receptor_pdb), rec)
    elif not full_atom_receptor_pdb(structure, rec, keep_het_chains=context_chains):
        if dry_run:
            write_min_pdb(rec, structure)
        else:
            fail_unless_mock_allowed(
                f"gnina: no full-atom receptor for {candidate_id} "
                "(upstream s04/s08b emitted a CA-only/mock structure)")
            log.warning(
                "gnina: no full-atom receptor for %s; mock fallback "
                "(NOT a real dock)", candidate_id,
            )
            return _mock_all(candidate_id, reference_atoms, cfg)
    write_min_pdb(lig, structure.__class__(sequence="", residues=[]), reference_atoms)
    # Cap gnina's search threads. gnina defaults --cpu to ALL cores; the s06b
    # per-rep augmentation runs one gnina per GPU concurrently, so the default
    # would 3x-saturate the shared 48-core box and trip the core-usage watchdog.
    # Modest default; override with EVOLIEZ_DOCK_CPU.
    cpu = max(1, int(os.environ.get("EVOLIEZ_DOCK_CPU", "4") or "4"))
    # Deterministic per-candidate seed so a rerun reproduces the same modes.
    seed = derive_seed(0x6E10A, candidate_id) % 2147483647
    cmd = _gnina_cmd(rec, lig, out, cfg, seed=seed, cpu=cpu)
    run(cmd, dry_run=dry_run)
    if dry_run:
        return _mock_all(candidate_id, reference_atoms, cfg)
    if not out.exists():
        # Real run produced no output (tool crash / wrong path). A fabricated
        # mock pose must not pass silently as a real dock score into ranking:
        # hard-fail unless mock fallback is explicitly allowed (audit P0 #3).
        fail_unless_mock_allowed(
            f"gnina produced no output for {candidate_id} ({out})")
        log.warning(
            "gnina produced no output for %s (%s); using mock fallback "
            "(NOT a real dock)", candidate_id, out,
        )
        return _mock_all(candidate_id, reference_atoms, cfg)
    version = tool_version("gnina")
    poses = parse_all_modes(
        out, reference_atoms, candidate_id=candidate_id,
        command_args=" ".join(cmd), engine_version=version,
    )
    if not poses:
        # SDF present but unparseable (no minimizedAffinity / no atoms): same
        # honesty contract as "no output".
        fail_unless_mock_allowed(
            f"gnina output for {candidate_id} parsed to zero modes ({out})")
        log.warning(
            "gnina output for %s parsed to zero modes (%s); mock fallback "
            "(NOT a real dock)", candidate_id, out,
        )
        return _mock_all(candidate_id, reference_atoms, cfg)
    return poses


def _mock_all(candidate_id, reference_atoms, cfg) -> List[Pose]:
    return mock_redock_modes(
        candidate_id, METHOD, reference_atoms,
        n_modes=max(1, int(cfg.poses_per_candidate)),
        score_type=SCORE_TYPE, higher_is_better=False, with_cnn=True,
        base_instability=0.2, score_offset=-0.5,
    )


def parse_all_modes(
    sdf_path,
    reference_atoms: Sequence[LigandAtom],
    *,
    candidate_id: str = "",
    command_args: str = "",
    engine_version: str = "",
) -> List[Pose]:
    """Parse EVERY mode in a gnina output SDF into a provenanced :class:`Pose`.

    gnina writes its N modes as N ``$$$$``-separated molecules in ONE SDF, each
    carrying score tags AFTER the molecule block::

        > <minimizedAffinity>
        -9.42

        > <CNNscore>
        0.873

        > <CNNaffinity>
        6.21

    For mode i (0-based file order) we emit a Pose with ``rank=i+1`` (1-based),
    the docked ``ligand_atoms`` (locked onto the canonical reference, so RMSD-to-
    reference is computable), ``score`` = minimizedAffinity (``score_type``=
    "minimizedAffinity", LOWER is better), ``cnn_score`` = CNNscore,
    ``cnn_affinity`` = CNNaffinity, plus the run's ``command_args`` /
    ``engine_version``. A mode missing minimizedAffinity is SKIPPED (an
    unscorable mode must not enter ranking as 0.0)."""
    coords = parse_sdf_all_poses(sdf_path)
    tags = _parse_mode_tags(sdf_path)
    poses: List[Pose] = []
    for i, aff in enumerate(tags["minimizedAffinity"]):
        if aff is None:                       # unscored mode -> not a real pose
            continue
        atoms = coords[i] if i < len(coords) else []
        locked, rmsd = lock_pose_to_reference(
            atoms, reference_atoms,
            candidate_id=candidate_id, method=METHOD, logger=log,
        )
        cnn = tags["CNNscore"]
        cnn_aff = tags["CNNaffinity"]
        poses.append(Pose(
            candidate_id=candidate_id, method=METHOD,
            score=round(aff, 4), ligand_atoms=locked, rmsd_to_reference=rmsd,
            cluster=i, rank=i + 1, score_type=SCORE_TYPE,
            cnn_score=(round(cnn[i], 4) if i < len(cnn) and cnn[i] is not None
                       else None),
            cnn_affinity=(round(cnn_aff[i], 4)
                          if i < len(cnn_aff) and cnn_aff[i] is not None
                          else None),
            engine_version=engine_version, command_args=command_args,
        ))
    return poses


def _parse_mode_tags(out) -> dict:
    """Per-mode gnina score tags, ALIGNED to molecule order.

    Returns ``{"minimizedAffinity": [...], "CNNscore": [...], "CNNaffinity":
    [...]}`` where list index i is mode i (file order). gnina puts each tag value
    on the line AFTER ``> <tag>`` (the old in-line ``float()`` parse missed it);
    a rare single-line ``minimizedAffinity X`` form is also handled. Per record
    a tag may be absent -> ``None`` at that index, so the lists stay index-aligned
    with the ``$$$$``-delimited molecules."""
    lines = Path(out).read_text().splitlines()
    tags = {"minimizedAffinity": [], "CNNscore": [], "CNNaffinity": []}
    # current[tag] = value seen since the last "$$$$" (record boundary).
    current = {k: None for k in tags}

    def _flush():
        for k in tags:
            tags[k].append(current[k])
            current[k] = None

    def _value_after(idx):
        for j in range(idx + 1, min(idx + 3, len(lines))):
            tok = lines[j].strip().split()
            if tok:
                try:
                    return float(tok[0])
                except ValueError:
                    return None
        return None

    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "$$$$":
            _flush()
            continue
        for tag in tags:
            if f"<{tag}>" in ln:
                v = _value_after(i)
                if v is None:                 # rare single-line "tag X" form
                    parts = s.split()
                    try:
                        v = float(parts[-1])
                    except (ValueError, IndexError):
                        v = None
                current[tag] = v
                break
    # gnina SDFs end with a trailing "$$$$" so every record is flushed there;
    # flush any unterminated final record (defensive — malformed/truncated SDF).
    if any(current[k] is not None for k in current):
        _flush()
    return tags


def _parse_gnina(out) -> float:
    """Best (lowest) minimizedAffinity across all modes of a gnina SDF. Back-
    compat shim (tests/the pre-redesign single-score path); now delegates to the
    per-mode tag parser so there is one source of truth. 0.0 if no scored mode."""
    scores = [a for a in _parse_mode_tags(out)["minimizedAffinity"] if a is not None]
    return round(min(scores), 4) if scores else 0.0
