"""DiffDock diffusion docking (spec section 10). GPU tool (server only).

real: DiffDock inference CLI.
mock: deterministic perturbed pose (shared helper).

The s06b multi-engine redesign adds GPU-BATCH inference (:func:`redock_batch`:
ONE model load per GPU for many targets) and parses ALL ranks (every
``rankN_confidence*.sdf``) with confidence provenance, not just rank-1.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evoliez.adapters.base import (
    fail_unless_mock_allowed,
    full_atom_receptor_pdb,
    lock_pose_to_reference,
    mock_redock_modes,
    parse_sdf_first_pose,
    write_min_pdb,
)
from evoliez.config import Backend, DockingConfig
from evoliez.logging_utils import get_logger
from evoliez.types import LigandAtom, Pose, ProteinStructure
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.diffdock")
METHOD = "diffdock"
SCORE_TYPE = "diffdock_confidence"

# One batch task: (complex_id, receptor_structure, reference_atoms, smiles).
BatchTask = Tuple[str, ProteinStructure, Sequence[LigandAtom], str]

# Memoized DiffDock version string. Resolving it can spawn a subprocess
# (``git rev-parse``) and stat the inference script; the result is constant for
# the life of the process, so compute it AT MOST ONCE. Sentinel ``None`` = not
# yet computed (the resolved string itself is always a non-empty clean line).
_DIFFDOCK_VERSION: Optional[str] = None


def _diffdock_repo_dir() -> Optional[Path]:
    """The DiffDock clone directory (``$EVOLIEZ_DIFFDOCK``), or None if unset.

    This is the same dir used as ``cwd`` for ``python -m inference``; the
    inference entry point is ``<repo>/inference.py``. Provenance is keyed off
    THIS path so the recorded version matches the code that actually ran."""
    d = os.environ.get("EVOLIEZ_DIFFDOCK")
    if d and str(d).strip():
        return Path(str(d).strip())
    return None


def _diffdock_version() -> str:
    """A MEANINGFUL one-line DiffDock provenance string, computed at most once.

    DiffDock has no clean ``--version`` and is run from a repo clone (not a pip
    package), so the old ``importlib.metadata.version('diffdock')`` probe always
    raised and the captured stderr stored the literal "Traceback (most recent
    call last):" as the engine version. This resolves a real identifier instead,
    with robust fallbacks (NEVER a traceback):

      1. ``git -C <repo> rev-parse --short HEAD`` -> ``"DiffDock @ <sha>"`` (the
         exact commit of the clone that produced the poses);
      2. else the resolved ``inference.py`` path + its mtime ->
         ``"DiffDock inference.py <path> (mtime <iso>)"``;
      3. else a pinned ``"DiffDock"`` constant.

    On ANY exception -> ``"unknown"``. The result is stripped to a single clean
    line. Best-effort: provenance is recorded, never a run-blocker."""
    global _DIFFDOCK_VERSION
    if _DIFFDOCK_VERSION is None:
        _DIFFDOCK_VERSION = _resolve_diffdock_version()
    return _DIFFDOCK_VERSION


def _one_line(s: str, limit: int = 200) -> str:
    """First non-empty line of ``s``, stripped and length-capped (no traceback
    multi-line spill ever reaches ``Pose.engine_version``)."""
    for ln in str(s).splitlines():
        ln = ln.strip()
        if ln:
            return ln[:limit]
    return ""


def _resolve_diffdock_version() -> str:
    from datetime import datetime, timezone

    from evoliez.utils.subprocess_utils import run, which
    try:
        repo = _diffdock_repo_dir()
        # 1) git short SHA of the clone (the most precise provenance).
        if repo is not None and repo.exists() and which("git") is not None:
            try:
                res = run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          check=False)
                sha = _one_line((res.stdout or "").strip(), limit=40)
                # rev-parse prints the error to STDERR on failure, so a clean
                # stdout line here is a real SHA, never a traceback/usage blurb.
                if sha and res.returncode == 0:
                    return f"DiffDock @ {sha}"
            except Exception:
                pass  # fall through to the path+mtime fallback
        # 2) resolved inference.py path + mtime (works without git/.git).
        if repo is not None:
            inf = repo / "inference.py"
            if inf.exists():
                mtime = datetime.fromtimestamp(
                    inf.stat().st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds")
                return _one_line(f"DiffDock inference.py {inf} (mtime {mtime})")
        # 3) pinned constant (DiffDock is installed but unidentifiable).
        return "DiffDock"
    except Exception:
        return "unknown"


def redock(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    instability: float,
    smiles: str,
    backend: Backend,
    dry_run: bool = False,
    context_chains=None,  # accepted for a uniform call signature but IGNORED:
                          # DiffDock's learned model takes a protein receptor +
                          # one ligand, with no cofactor-context input.
    receptor_pdb: Optional[Path] = None,
) -> Pose:
    """Back-compat single-pose entry point: the rank-1 (best) DiffDock pose.

    Kept verbatim for existing callers; internally returns ``redock_all(...)[0]``
    so the rank-parsing/provenance path is shared with the batch entry point."""
    return redock_all(
        candidate_id, structure, reference_atoms, cfg, workdir,
        instability=instability, smiles=smiles, backend=backend,
        dry_run=dry_run, receptor_pdb=receptor_pdb,
    )[0]


def redock_all(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    instability: float,
    smiles: str,
    backend: Backend,
    dry_run: bool = False,
    receptor_pdb: Optional[Path] = None,
) -> List[Pose]:
    """ALL DiffDock ranks (best first) for a SINGLE target, fully provenanced.
    Always non-empty (rank-1 at index 0). The batch path (:func:`redock_batch`)
    is preferred on the server (one model load for many targets); this stays for
    one-off / per-target callers.

    ``receptor_pdb``: an optional pre-rendered PROTEIN-ONLY receptor PDB; when
    given AND it exists it is copied to the expected ``rec`` path instead of
    re-rendering from ``structure.pdb_path``. DiffDock is protein-only, so this
    must NOT be a context-retained (cofactor-bearing) receptor. Default
    ``None`` = render exactly as before (byte-identical)."""
    if backend is Backend.real:
        return _redock_real_all(
            candidate_id, structure, smiles, cfg, workdir,
            reference_atoms, dry_run=dry_run, receptor_pdb=receptor_pdb,
        )
    return mock_redock_modes(
        candidate_id, METHOD, reference_atoms,
        n_modes=max(1, int(cfg.poses_per_candidate)),
        score_type=SCORE_TYPE, higher_is_better=True, with_cnn=False,
        base_instability=max(0.02, instability), score_offset=-0.2,
    )


def redock_batch(
    tasks: Sequence[BatchTask],
    out_root: Path,
    cfg: DockingConfig,
    *,
    backend: Backend,
    dry_run: bool = False,
    gpu_device: Optional[int] = None,
    receptor_pdb: Optional[Dict[str, Path]] = None,
) -> Dict[str, List[Pose]]:
    """GPU-BATCH DiffDock: ONE model load for MANY targets.

    ``tasks`` is a sequence of ``(complex_id, receptor_structure,
    reference_atoms, smiles)``. The real path writes ONE ``protein_ligand_csv``
    with a row per task (``complex_name``=complex_id, ``protein_path``=the
    per-task receptor PDB written under ``out_root``, ``ligand_description``=
    smiles, ``protein_sequence``=""), runs a SINGLE ``python -m inference
    --protein_ligand_csv <csv> --out_dir <out_root> --samples_per_complex <cfg>``
    (the same flags as the single-target path; the GPU pin is the CALLER's job
    via ``CUDA_VISIBLE_DEVICES`` — ``gpu_device`` is accepted only for logging),
    then parses ALL ranks for EACH complex.

    Returns ``{complex_id: [Pose, ...]}`` (best rank first per complex). A task
    whose receptor is unavailable degrades per the usual honesty contract (mock
    list if allowed, else raises).

    ``receptor_pdb``: an optional ``{complex_id: Path}`` map of pre-rendered
    PROTEIN-ONLY receptors (DiffDock is protein-only — these must NOT carry
    cofactor context). When a task's cid is present AND the file exists, that
    PDB is copied to the per-task ``rec`` path instead of re-rendering from the
    structure (RENDER-ONCE: the caller renders one receptor per rep and reuses
    it across engines). cids absent from the map render as before. Default
    ``None`` = render every task exactly as before (byte-identical).

    Mock backend: a deterministic N-rank ensemble per complex, no real tool."""
    if backend is not Backend.real:
        return {
            cid: mock_redock_modes(
                cid, METHOD, ref,
                n_modes=max(1, int(cfg.poses_per_candidate)),
                score_type=SCORE_TYPE, higher_is_better=True, with_cnn=False,
                base_instability=0.08, score_offset=-0.2,
            )
            for (cid, _struct, ref, _smiles) in tasks
        }
    return _redock_batch_real(
        tasks, Path(out_root), cfg, dry_run=dry_run, gpu_device=gpu_device,
        receptor_pdb=receptor_pdb,
    )


def _redock_batch_real(
    tasks: Sequence[BatchTask],
    out_root: Path,
    cfg: DockingConfig,
    *,
    dry_run: bool,
    gpu_device: Optional[int],
    receptor_pdb: Optional[Dict[str, Path]] = None,
) -> Dict[str, List[Pose]]:
    require("python")
    apply_gpu_selection()
    out_root.mkdir(parents=True, exist_ok=True)
    if gpu_device is not None:
        log.info("diffdock batch: %d task(s), pinned to GPU %s (by caller)",
                 len(tasks), gpu_device)

    pre = receptor_pdb or {}
    rows: List[str] = ["complex_name,protein_path,ligand_description,protein_sequence"]
    refs: Dict[str, Sequence[LigandAtom]] = {}
    smis: Dict[str, str] = {}
    result: Dict[str, List[Pose]] = {}
    want = max(1, int(cfg.poses_per_candidate))
    reuse_cids: set = set()
    for cid, structure, ref_atoms, smiles in tasks:
        refs[cid] = ref_atoms
        smis[cid] = smiles
        # SKIP-IF-EXISTS (re-run reuse): a COMPLETE prior DiffDock output for this
        # complex is re-parsed instead of re-docked, so an s06b classification
        # re-run reuses the existing rank SDFs (no GPU inference). "Complete" =
        # the per-complex dir already holds >= samples_per_complex rank*.sdf
        # files (a partial dir from an interrupted run is NOT reused — it is
        # re-docked). DiffDock's per-complex output is deterministic given the
        # same receptor+ligand, so reuse and a fresh run agree.
        if not dry_run:
            cdir = _complex_out_dir(out_root, cid)
            if cdir is not None and len(_rank_files(cdir)) >= want:
                reuse_cids.add(cid)
                continue
        rec = out_root / f"{cid}_rec.pdb"
        # RENDER-ONCE: reuse a caller-supplied protein-only receptor if present.
        pre_rec = pre.get(cid)
        if pre_rec is not None and Path(pre_rec).exists():
            shutil.copyfile(Path(pre_rec), rec)
        elif not full_atom_receptor_pdb(structure, rec):
            if dry_run:
                write_min_pdb(rec, structure)
            else:
                fail_unless_mock_allowed(
                    f"diffdock: no full-atom receptor for {cid} "
                    "(upstream s04/s08b emitted a CA-only/mock structure)")
                log.warning(
                    "diffdock: no full-atom receptor for %s; mock fallback "
                    "(NOT a real dock)", cid,
                )
                result[cid] = _mock_all(cid, ref_atoms, cfg)
                continue
        # CSV fields must not contain commas; receptor paths/SMILES here don't,
        # but resolve() keeps the path absolute for DiffDock's cwd-relative run.
        rows.append(f"{cid},{rec.resolve()},{smiles},")

    if reuse_cids:
        log.info("diffdock batch: reusing %d existing complex output(s) "
                 "(skip re-dock)", len(reuse_cids))

    csv_targets = [r for r in rows[1:]]
    if not csv_targets:               # every task reused/degraded -> nothing to run
        version = _diffdock_version()
        for cid in reuse_cids:
            cdir = _complex_out_dir(out_root, cid)
            poses = parse_all_ranks(
                cdir, refs[cid], candidate_id=cid,
                command_args="(reused existing DiffDock output)",
                engine_version=version, smiles=smis[cid],
            ) if cdir is not None else []
            result[cid] = poses or _mock_all(cid, refs[cid], cfg)
        return result

    csv = out_root / "batch_input.csv"
    csv.write_text("\n".join(rows) + "\n")
    # DiffDock's `inference` module lives in its repo, not on the pipeline's
    # import path. Run it FROM that repo (cwd) so `-m inference` resolves, rather
    # than putting it on PYTHONPATH globally. EVOLIEZ_DIFFDOCK points at the
    # clone; its model weights live under <repo>/workdir.
    cmd = [
        "python", "-m", "inference",
        "--protein_ligand_csv", str(csv.resolve()),
        "--out_dir", str(out_root.resolve()),
        "--samples_per_complex", str(cfg.poses_per_candidate),
    ]
    _bs = os.environ.get("EVOLIEZ_DIFFDOCK_BATCH")
    if _bs and str(_bs).strip():        # GPU packing (DiffDock default 10); quality-neutral
        cmd += ["--batch_size", str(int(_bs))]
    run(cmd, cwd=os.environ.get("EVOLIEZ_DIFFDOCK"), dry_run=dry_run)
    version = _diffdock_version()

    for cid, structure, ref_atoms, smiles in tasks:
        if cid in result:             # already degraded to mock above
            continue
        if dry_run:
            result[cid] = _mock_all(cid, ref_atoms, cfg)
            continue
        complex_dir = _complex_out_dir(out_root, cid)
        if complex_dir is None:
            fail_unless_mock_allowed(
                f"diffdock produced no output for {cid} (under {out_root})")
            log.warning(
                "diffdock produced no output for %s (%s); mock fallback "
                "(NOT a real dock)", cid, out_root,
            )
            result[cid] = _mock_all(cid, ref_atoms, cfg)
            continue
        poses = parse_all_ranks(
            complex_dir, ref_atoms, candidate_id=cid,
            command_args=("(reused existing DiffDock output)"
                          if cid in reuse_cids else " ".join(cmd)),
            engine_version=version, smiles=smiles,
        )
        result[cid] = poses or _mock_all(cid, ref_atoms, cfg)
    return result


def _redock_real_all(
    candidate_id: str,
    structure: ProteinStructure,
    smiles: str,
    cfg: DockingConfig,
    workdir: Path,
    reference_atoms: Sequence[LigandAtom],
    *,
    dry_run: bool,
    receptor_pdb: Optional[Path] = None,
) -> List[Pose]:
    require("python")
    apply_gpu_selection()
    workdir.mkdir(parents=True, exist_ok=True)
    rec = workdir / f"{candidate_id}_rec.pdb"
    # Full-atom Boltz receptor for real docking; honest mock fallback if the
    # upstream structure is a CA-only trace (dry-run keeps a placeholder).
    # RENDER-ONCE: copy a caller-supplied protein-only receptor if present.
    if receptor_pdb is not None and Path(receptor_pdb).exists():
        shutil.copyfile(Path(receptor_pdb), rec)
    elif not full_atom_receptor_pdb(structure, rec):
        if dry_run:
            write_min_pdb(rec, structure)
        else:
            fail_unless_mock_allowed(
                f"diffdock: no full-atom receptor for {candidate_id} "
                "(upstream s04/s08b emitted a CA-only/mock structure)")
            log.warning(
                "diffdock: no full-atom receptor for %s; mock fallback "
                "(NOT a real dock)", candidate_id,
            )
            return _mock_all(candidate_id, reference_atoms, cfg)
    csv = workdir / f"{candidate_id}_input.csv"
    csv.write_text(
        "complex_name,protein_path,ligand_description,protein_sequence\n"
        f"{candidate_id},{rec},{smiles},\n"
    )
    out = workdir / f"{candidate_id}_dd_out"
    cmd = [
        "python", "-m", "inference",
        "--protein_ligand_csv", str(csv.resolve()),
        "--out_dir", str(out.resolve()),
        "--samples_per_complex", str(cfg.poses_per_candidate),
    ]
    _bs = os.environ.get("EVOLIEZ_DIFFDOCK_BATCH")
    if _bs and str(_bs).strip():        # GPU packing (DiffDock default 10); quality-neutral
        cmd += ["--batch_size", str(int(_bs))]
    run(cmd, cwd=os.environ.get("EVOLIEZ_DIFFDOCK"), dry_run=dry_run)
    if dry_run:
        return _mock_all(candidate_id, reference_atoms, cfg)
    if not out.exists():
        fail_unless_mock_allowed(
            f"diffdock produced no output for {candidate_id} ({out})")
        log.warning(
            "diffdock produced no output for %s (%s); using mock fallback "
            "(NOT a real dock)", candidate_id, out,
        )
        return _mock_all(candidate_id, reference_atoms, cfg)
    version = _diffdock_version()
    poses = parse_all_ranks(
        out, reference_atoms, candidate_id=candidate_id,
        command_args=" ".join(cmd), engine_version=version, smiles=smiles,
    )
    return poses or _mock_all(candidate_id, reference_atoms, cfg)


def _mock_all(candidate_id, reference_atoms, cfg) -> List[Pose]:
    return mock_redock_modes(
        candidate_id, METHOD, reference_atoms,
        n_modes=max(1, int(cfg.poses_per_candidate)),
        score_type=SCORE_TYPE, higher_is_better=True, with_cnn=False,
        base_instability=0.2, score_offset=-0.2,
    )


def _complex_out_dir(out_root: Path, complex_id: str) -> Optional[Path]:
    """DiffDock writes each complex's ranks under ``<out_dir>/<complex_name>/``.
    Return that per-complex dir if it holds any ``rank*.sdf``, else the deepest
    dir under ``out_root`` that does (some versions nest differently), else None.

    Scoping per-complex matters in the BATCH case: ``out_root`` holds one
    sub-dir per task, so we must not let one complex's ranks leak into another's.

    PERF: the fallbacks are SCOPED to ``out_root.glob('*<cid>*')`` (and descend
    at most one further level), NOT ``rglob('*')`` over the whole chunk tree —
    walking every file per complex was O(all files) and risked O(N^2) per chunk.
    """
    cand = out_root / complex_id
    if cand.is_dir() and any(_is_rank(p) for p in cand.glob("rank*.sdf")):
        return cand
    # Fallback: a directory (matching the complex_id) directly holding this
    # complex's rank files, or one level below it (some versions nest deeper).
    # Globbing on the cid keeps batch dirs separated AND bounds the walk.
    for top in sorted(p for p in out_root.glob(f"*{complex_id}*") if p.is_dir()):
        if any(_is_rank(p) for p in top.glob("rank*.sdf")):
            return top
        for sub in sorted(p for p in top.glob("*") if p.is_dir()):
            if any(_is_rank(p) for p in sub.glob("rank*.sdf")):
                return sub
    # Last resort (single-target layout): out_root itself or a cid-matching dir
    # (one level down) with ranks. Still scoped to the cid — never a full walk.
    if any(_is_rank(p) for p in out_root.glob("rank*.sdf")):
        return out_root
    for top in sorted(p for p in out_root.glob(f"*{complex_id}*") if p.is_dir()):
        if any(_is_rank(p) for p in top.glob("rank*.sdf")):
            return top
    return None


def parse_all_ranks(
    complex_out_dir,
    reference_atoms: Sequence[LigandAtom],
    *,
    candidate_id: str = "",
    command_args: str = "",
    engine_version: str = "",
    smiles: Optional[str] = None,
) -> List[Pose]:
    """Parse EVERY DiffDock rank for ONE complex into provenanced :class:`Pose`s.

    DiffDock writes per-complex ``rank{N}_confidence{score}.sdf`` (N=1 is best)
    and sometimes a bare ``rank1.sdf``; the confidence is encoded IN THE
    FILENAME (``confidence-0.53`` etc.). For each rank file we emit a Pose with
    ``rank`` = the parsed integer, docked ``ligand_atoms`` (locked onto the
    canonical reference), ``score`` = confidence (``score_type``=
    "diffdock_confidence", HIGHER is better), and ``command_args`` /
    ``engine_version``. When the confidence is genuinely ABSENT (a bare
    ``rankN.sdf``, an unparseable token, or DiffDock's own ``-1000`` failure
    sentinel) the pose ``score`` is set to ``None`` (with a short ``note`` and a
    warning) — NEVER a fabricated ``-1000`` / ``0.0`` that would pollute the
    score range and min/range stats downstream. Returned best rank first (rank
    ascending)."""
    files = _rank_files(complex_out_dir)
    poses: List[Pose] = []
    from evoliez.features.ligand import rmsd_template
    tmpl = rmsd_template(smiles) if smiles else None

    for rank_n, path in files:
        atoms = parse_sdf_first_pose(path)
        # The raw SDF text feeds the SMILES-template lock: it recovers the real
        # docked frame + a correct atom-id correspondence for a flexible cofactor
        # (NADP) whose coordinate-inferred chemistry graph the legacy lock could
        # not verify (it substituted the reference -> a false RMSD of 0 for ~27%
        # of DiffDock NADP ranks). best-effort read; missing text -> coord lock.
        try:
            pose_text = Path(path).read_text()
        except OSError:
            pose_text = None
        locked, rmsd = lock_pose_to_reference(
            atoms, reference_atoms,
            candidate_id=candidate_id, method=METHOD, logger=log,
            pose_text=pose_text, pose_fmt="sdf", template=tmpl,
        )
        score, note = _parse_confidence(path.name)
        if score is None:
            log.warning(
                "diffdock rank-%d pose %s: %s; recorded score=None (unscored)",
                rank_n, path.name, note or "no confidence",
            )
        poses.append(Pose(
            candidate_id=candidate_id, method=METHOD, score=score,
            ligand_atoms=locked, rmsd_to_reference=rmsd,
            cluster=rank_n - 1, rank=rank_n, score_type=SCORE_TYPE,
            engine_version=engine_version, command_args=command_args,
            note=note,
        ))
    return poses


# DiffDock's confidence model emits ``-1000`` as a FAILURE sentinel (no usable
# confidence). We must not treat it as a real (extremely negative) score: it
# would dominate any min/range statistic. Anything at or below this is unscored.
_DIFFDOCK_CONF_SENTINEL = -1000.0


def _parse_confidence(filename: str):
    """Parse the DiffDock ``confidence<float>`` token from a rank filename.

    Returns ``(score, note)`` where ``score`` is the rounded float, or ``None``
    when the confidence is genuinely absent / unusable (with a short ``note``
    explaining why). Robust to a leading sign (``-``/``+``), a leading-dot
    mantissa (``confidence-.53``), and scientific notation (``confidence1e-1``);
    a missing token, a non-finite value, or DiffDock's ``-1000`` failure
    sentinel all map to ``(None, <reason>)`` rather than a fabricated number."""
    import math
    import re

    m = re.search(r"confidence([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
                  str(filename))
    if not m:
        return None, "no confidence token"
    try:
        val = float(m.group(1))
    except (TypeError, ValueError):
        return None, "unparseable confidence token"
    if not math.isfinite(val):
        return None, "non-finite confidence"
    if val <= _DIFFDOCK_CONF_SENTINEL:
        return None, "diffdock -1000 confidence sentinel (no usable score)"
    return round(val, 4), ""


def _is_rank(p: Path) -> bool:
    import re
    return re.match(r"rank\d+", p.name) is not None


def _rank_files(out_dir) -> List[Tuple[int, Path]]:
    """All ``rankN[...].sdf`` for a complex as ``[(N, path), ...]`` sorted by N.

    Parses the rank INTEGER (a glob can't: ``rank1*`` also matches rank10..19,
    and ``sorted()`` puts 'rank10' BEFORE 'rank1_' since '0' < '_'). When BOTH a
    bare ``rankN.sdf`` and a ``rankN_confidence*.sdf`` exist for the same N, the
    confidence-bearing file is kept (so the score and the adopted coordinates
    come from the SAME real pose)."""
    import re

    best: Dict[int, Path] = {}
    for p in Path(out_dir).glob("rank*.sdf"):
        m = re.match(r"rank(\d+)", p.name)
        if not m:
            continue
        n = int(m.group(1))
        # Prefer the confidence-bearing file for a given rank.
        if n not in best or ("confidence" in p.name and "confidence" not in best[n].name):
            best[n] = p
    return [(n, best[n]) for n in sorted(best)]


# --------------------------------------------------------------------------- #
# Back-compat single-pose helpers (referenced by tests/test_diffdock_rank.py and
# the pre-redesign single-pose path). Kept working verbatim.
# --------------------------------------------------------------------------- #
def _rank1_pose_files(out_dir):
    """The DiffDock rank-1 (best) pose SDF(s), confidence-bearing file first.

    DiffDock writes ``rank{N}_confidence{score}.sdf`` (N=1 is best) and sometimes
    a bare ``rank1.sdf``. We parse the rank INTEGER and keep only N==1: a glob
    like ``rank1*`` also matches rank10..19, and ``sorted()`` puts 'rank10'
    BEFORE 'rank1_' ('0' < '_'), so the old code silently selected the WORST
    pose. The confidence-bearing file is returned first so the score and the
    adopted coordinates come from the SAME real pose."""
    import re
    from pathlib import Path

    def _is_rank1(p):
        m = re.match(r"rank(\d+)", p.name)
        return m is not None and int(m.group(1)) == 1

    r1 = [p for p in Path(out_dir).rglob("rank*.sdf") if _is_rank1(p)]
    return sorted(p for p in r1 if "confidence" in p.name) or sorted(r1)


def _parse_diffdock(out_dir):
    """DiffDock writes `rank1_confidence-X.XX.sdf`; the confidence is encoded in
    the filename. Return it (higher = better; 0.0 if no pose). A pose with no
    usable confidence token is treated as unscored (0.0) WITH a warning - never
    the old fabricated -7.0, which looked like a real affinity paired with real
    coords. (Back-compat scalar helper kept for the single-pose path/tests; the
    production rank parser :func:`parse_all_ranks` records ``None``, not 0.0, for
    the unscored case so it never pollutes score statistics.)"""
    confs = _rank1_pose_files(out_dir)
    if not confs:
        return 0.0
    score, note = _parse_confidence(confs[0].name)
    if score is None:
        log.warning(
            "diffdock rank-1 pose %s: %s; treating as unscored",
            confs[0].name, note or "no confidence",
        )
        return 0.0
    return score
