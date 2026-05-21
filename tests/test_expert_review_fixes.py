"""Expert review (production stability sweep) - 3 issues landed:

Issue 1: LigandMPNN / DiffDock adapters were cwd-relative
    `python run.py` / `python -m inference` only worked if the caller
    happened to be chdir'd into the install dir. We now require
    EVOLIEZ_LIGANDMPNN / EVOLIEZ_DIFFDOCK and fail loudly otherwise.

Issue 3: GNINA / DiffDock SDF parsers dropped atom coordinates
    The adapters returned `ligand_atoms=list(reference_atoms)`, so
    `pose.rmsd_to_reference` was None -> `None or 0.0` -> s09's
    ligand_escape gate (>4.5) was vacuously passing for real backends.

Issue 6: s11 silently re-ranked ALL candidates when validated=[]
    `ctx.get(...) or ctx.require(...)` treated the empty-list case
    as "didn't run", so an honest "every candidate failed non-MD
    validation" run produced a Strong-looking final report on the
    UNFILTERED set. Now we distinguish None (didn't run) from [] (ran,
    all rejected).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from evoliez.adapters import diffdock as diffdock_adapter
from evoliez.adapters import gnina as gnina_adapter
from evoliez.adapters import ligandmpnn as lmpnn_adapter
from evoliez.types import LigandAtom


# --------------------------------------------------------------------------- #
# Issue 1: env var requirements
# --------------------------------------------------------------------------- #
def test_ligandmpnn_resolve_requires_env_var(monkeypatch):
    monkeypatch.delenv("EVOLIEZ_LIGANDMPNN", raising=False)
    with pytest.raises(RuntimeError, match="EVOLIEZ_LIGANDMPNN"):
        lmpnn_adapter._resolve_lmpnn_install()


def test_ligandmpnn_resolve_validates_run_py_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLIEZ_LIGANDMPNN", str(tmp_path))
    with pytest.raises(RuntimeError, match="run.py"):
        lmpnn_adapter._resolve_lmpnn_install()


def test_ligandmpnn_resolve_accepts_valid_install(monkeypatch, tmp_path):
    (tmp_path / "run.py").write_text("# stub")
    monkeypatch.setenv("EVOLIEZ_LIGANDMPNN", str(tmp_path))
    assert lmpnn_adapter._resolve_lmpnn_install() == tmp_path


def test_diffdock_resolve_requires_env_var(monkeypatch):
    monkeypatch.delenv("EVOLIEZ_DIFFDOCK", raising=False)
    with pytest.raises(RuntimeError, match="EVOLIEZ_DIFFDOCK"):
        diffdock_adapter._resolve_diffdock_install()


def test_diffdock_resolve_validates_inference_module(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLIEZ_DIFFDOCK", str(tmp_path))
    with pytest.raises(RuntimeError, match="inference"):
        diffdock_adapter._resolve_diffdock_install()


def test_diffdock_resolve_accepts_inference_py(monkeypatch, tmp_path):
    (tmp_path / "inference.py").write_text("# stub")
    monkeypatch.setenv("EVOLIEZ_DIFFDOCK", str(tmp_path))
    assert diffdock_adapter._resolve_diffdock_install() == tmp_path


def test_diffdock_resolve_accepts_inference_package(monkeypatch, tmp_path):
    pkg = tmp_path / "inference"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.setenv("EVOLIEZ_DIFFDOCK", str(tmp_path))
    assert diffdock_adapter._resolve_diffdock_install() == tmp_path


# --------------------------------------------------------------------------- #
# Issue 3: GNINA / DiffDock atom coord parsing
# --------------------------------------------------------------------------- #
def _write_gnina_sdf(path: Path, atoms_block: str, score: float):
    """Minimal V2000 SDF with one pose, the given atoms_block string
    (already newline-terminated), and a minimizedAffinity tag."""
    n_atoms = len([ln for ln in atoms_block.splitlines() if ln.strip()])
    body = (
        "pose1\n"
        " test  \n"
        "\n"
        f"  {n_atoms}  0  0  0  0  0  0  0  0999 V2000\n"
        + atoms_block
        + "M  END\n"
        f"> <minimizedAffinity>\n{score}\n\n"
        "$$$$\n"
    )
    path.write_text(body)


def test_gnina_parser_returns_atoms_for_best_pose(tmp_path):
    """Two poses in one SDF; the one with lower minimizedAffinity wins
    and its atom coordinates are returned."""
    pose_a = (
        "    1.000    2.000    3.000 C   0  0  0  0  0  0\n"
        "    4.000    5.000    6.000 N   0  0  0  0  0  0\n"
    )
    pose_b = (
        "    9.000    8.000    7.000 C   0  0  0  0  0  0\n"
        "    6.000    5.000    4.000 N   0  0  0  0  0  0\n"
    )
    sdf = tmp_path / "out.sdf"
    # Lower score = better. pose_b is "better" (-9.0 < -7.0).
    body = ""
    for ablock, score in ((pose_a, -7.0), (pose_b, -9.0)):
        n = 2
        body += (
            "pose\n test\n\n"
            f"  {n}  0  0  0  0  0  0  0  0999 V2000\n"
            f"{ablock}M  END\n"
            f"> <minimizedAffinity>\n{score}\n\n"
            "$$$$\n"
        )
    sdf.write_text(body)
    score, cnn, atoms = gnina_adapter._parse_gnina(sdf)
    assert score == -9.0
    assert len(atoms) == 2
    # Best pose was pose_b; its first atom is at (9, 8, 7).
    assert atoms[0].coord == (9.0, 8.0, 7.0)


def test_gnina_lock_to_reference_computes_rmsd():
    ref = [
        LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="N1", element="N", coord=(0.0, 0.0, 0.0)),
    ]
    parsed = [
        LigandAtom(id="C0", element="C", coord=(1.0, 0.0, 0.0)),
        LigandAtom(id="N1", element="N", coord=(1.0, 0.0, 0.0)),
    ]
    locked, rmsd = gnina_adapter._lock_to_reference(parsed, ref, "cand_X")
    # Each atom moved 1.0 in x; mean square = 1.0; RMSD = 1.0.
    assert rmsd is not None
    assert abs(rmsd - 1.0) < 1e-6


def test_gnina_lock_to_reference_returns_none_rmsd_on_empty_parse():
    """No parsed atoms -> rmsd is None, NOT 0.0. The whole point of the
    fix is that None propagates to s09 where it's then handled
    consciously, instead of `None or 0.0` silently passing the gate."""
    ref = [LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0))]
    locked, rmsd = gnina_adapter._lock_to_reference([], ref, "cand_X")
    assert rmsd is None
    assert locked == ref


def test_diffdock_parser_returns_atoms(tmp_path):
    sdf = tmp_path / "rank1_confidence-0.42.sdf"
    sdf.write_text(
        "pose\n test\n\n"
        "  2  0  0  0  0  0  0  0  0999 V2000\n"
        "    1.000    2.000    3.000 C   0  0  0  0  0  0\n"
        "    4.000    5.000    6.000 O   0  0  0  0  0  0\n"
        "M  END\n"
        "$$$$\n"
    )
    score, atoms = diffdock_adapter._parse_diffdock(tmp_path)
    assert score == -0.42
    assert len(atoms) == 2
    assert atoms[0].coord == (1.0, 2.0, 3.0)


# --------------------------------------------------------------------------- #
# Issue 6: s11 fallback distinguishes None (didn't run) from [] (rejected all)
# --------------------------------------------------------------------------- #
def test_s11_distinguishes_empty_validated_from_unset():
    """Source guard: s11's run() must distinguish `validated_candidates=[]`
    (s09 ran, every candidate was rejected) from `validated_candidates`
    missing (s09 didn't run -> fall back to full candidate set).

    The old `ctx.get(...) or ctx.require(...)` was falsy on `[]` and
    silently re-ranked the full candidate list as if validation never
    happened - a Strong-looking final report on a run where nothing
    actually passed. This guard pins the discriminator: a literal
    `is None` check must be in the source.
    """
    import inspect

    from evoliez.stages import s11_final_ranking

    src = inspect.getsource(s11_final_ranking.FinalRankingStage.run)
    assert "validated_candidates" in src
    # The discriminator must be `is None`, not the falsy `or` chain.
    assert "is None" in src, (
        "s11 must use `if validated is None` to distinguish missing key "
        "from empty list; the old `or` chain treated [] as truthy-false"
    )
    # The warning must mention the empty-validation case so anyone
    # reading server logs can spot a silent zero-yield run.
    assert "EMPTY" in src or "empty" in src
