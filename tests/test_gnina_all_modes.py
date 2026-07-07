"""GNINA all-modes parsing + provenance (s06b multi-engine redesign).

gnina writes its N docked modes as N ``$$$$``-separated molecules in ONE output
SDF, each carrying ``minimizedAffinity`` / ``CNNscore`` / ``CNNaffinity`` tags.
``parse_all_modes`` must recover EVERY mode with the right rank + scores so the
downstream hard-negative logic ("good score yet discordant") has the data. The
mock backend must synthesize an equivalent N-mode ranked ensemble (tests can run
without real gnina).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evoliez.adapters import gnina
from evoliez.config import Backend, DockingConfig
from evoliez.types import LigandAtom, ProteinStructure


def _ref():
    # 3 heavy atoms; coords kept simple so RMSD-to-reference is computable.
    return [
        LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="O1", element="O", coord=(1.2, 0.0, 0.0)),
        LigandAtom(id="N2", element="N", coord=(0.0, 1.3, 0.0)),
    ]


def _structure():
    return ProteinStructure(sequence="AG", residues=[])


def _mode_block(aff: float, cnn: float, cnn_aff: float, dz: float) -> str:
    """One V2000 molecule (3 atoms) + the three gnina score tags, value on the
    line AFTER the ``> <tag>`` header (the real gnina layout)."""
    atoms = "\n".join(
        f"{x:10.4f}{y:10.4f}{z:10.4f} {el}   0  0  0  0  0  0  0  0  0  0  0  0"
        for (x, y, z, el) in [
            (0.0, 0.0, 0.0 + dz, "C"),
            (1.2, 0.0, 0.0 + dz, "O"),
            (0.0, 1.3, 0.0 + dz, "N"),
        ]
    )
    return (
        "MODE\n"
        "     RDKit          3D\n"
        "\n"
        "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
        f"{atoms}\n"
        "  1  2  1  0\n"
        "  1  3  1  0\n"
        "M  END\n"
        f"> <minimizedAffinity>\n{aff}\n\n"
        f"> <CNNscore>\n{cnn}\n\n"
        f"> <CNNaffinity>\n{cnn_aff}\n\n"
        "$$$$\n"
    )


def _multi_mode_sdf(path: Path) -> None:
    # 3 modes, monotonically worse affinity, drifting coords.
    path.write_text(
        _mode_block(-9.42, 0.873, 6.21, 0.0)
        + _mode_block(-8.10, 0.640, 5.40, 1.5)
        + _mode_block(-6.77, 0.512, 4.90, 3.0)
    )


# --------------------------- parse_all_modes -------------------------------- #
def test_parse_all_modes_recovers_every_mode(tmp_path):
    sdf = tmp_path / "gnina_out.sdf"
    _multi_mode_sdf(sdf)
    poses = gnina.parse_all_modes(
        sdf, _ref(), candidate_id="wt",
        command_args="gnina -r r -l l --seed 7", engine_version="gnina 1.0",
    )
    assert len(poses) == 3
    # ranks are 1-based file order
    assert [p.rank for p in poses] == [1, 2, 3]
    # minimizedAffinity -> .score (lower = better; rank-1 is the best)
    assert [p.score for p in poses] == [-9.42, -8.1, -6.77]
    assert poses[0].score == min(p.score for p in poses)
    # full score provenance captured
    assert all(p.score_type == "minimizedAffinity" for p in poses)
    assert poses[0].cnn_score == 0.873 and poses[0].cnn_affinity == 6.21
    assert poses[1].cnn_score == 0.64 and poses[2].cnn_affinity == 4.9
    assert all(p.method == "gnina" for p in poses)
    assert all(p.command_args.startswith("gnina") for p in poses)
    assert all(p.engine_version == "gnina 1.0" for p in poses)
    # docked coords were adopted (locked onto the reference) -> RMSD computable
    assert poses[0].ligand_atoms and poses[0].rmsd_to_reference is not None
    # rank-1 sits on the reference (dz=0) -> ~0 RMSD; later ranks drift further
    assert poses[0].rmsd_to_reference < poses[2].rmsd_to_reference


def test_parse_all_modes_skips_unscored_mode(tmp_path):
    # a trailing molecule with NO minimizedAffinity tag is not a scorable pose.
    sdf = tmp_path / "g.sdf"
    sdf.write_text(
        _mode_block(-7.0, 0.5, 5.0, 0.0)
        + "JUNK\n     RDKit          3D\n\n"
          "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
          "    0.0000    0.0000    0.0000 C   0  0\nM  END\n$$$$\n"
    )
    poses = gnina.parse_all_modes(sdf, _ref())
    assert len(poses) == 1 and poses[0].score == -7.0


def test_parse_all_modes_empty(tmp_path):
    empty = tmp_path / "empty.sdf"
    empty.write_text("")
    assert gnina.parse_all_modes(empty, _ref()) == []


# --------------------------- mock redock_all -------------------------------- #
def test_mock_redock_all_synthesizes_ranked_modes():
    cfg = DockingConfig()  # poses_per_candidate default
    n = cfg.poses_per_candidate
    poses = gnina.redock_all(
        "cand", _structure(), _ref(), cfg, Path("/unused"),
        instability=0.1, backend=Backend.mock,
    )
    assert len(poses) == n >= 1
    assert [p.rank for p in poses] == list(range(1, n + 1))
    # descending plausibility: scores monotonically WORSEN (affinity ascends).
    scores = [p.score for p in poses]
    assert scores == sorted(scores)
    assert poses[0].score == min(scores)        # rank-1 best
    assert all(p.score_type == "minimizedAffinity" for p in poses)
    assert all(p.cnn_score is not None for p in poses)   # mock gnina has CNN tags
    assert all(p.method == "gnina" for p in poses)
    # rank-1 closest to reference (smallest RMSD)
    rmsds = [p.rmsd_to_reference for p in poses]
    assert rmsds[0] == min(rmsds)


def test_mock_redock_back_compat_returns_rank1():
    # the single-pose redock() must equal redock_all()[0]. The mock backend writes
    # no SDF, so the reference-RMSD selection has nothing to score and falls back
    # to rank-1 (unchanged behaviour).
    cfg = DockingConfig()
    one = gnina.redock("c", _structure(), _ref(), cfg, Path("/unused"),
                       instability=0.1, backend=Backend.mock, smiles="CCO")
    first = gnina.redock_all("c", _structure(), _ref(), cfg, Path("/unused"),
                             instability=0.1, backend=Backend.mock)[0]
    assert one.rank == 1
    assert one.score == first.score and one.method == "gnina"


# ---------------- reference-consistent single-pose selection ---------------- #
# The core fix: gnina's CNN/affinity rank-1 can be an end-for-end-FLIPPED pose for
# a large/flexible ligand. The representative must be the mode whose RAW docked
# geometry best matches the reference (min symmetry-corrected RMSD), keeping its
# OWN gnina rank + minimizedAffinity (honest provenance). Built with a real RDKit
# ligand so the symmetry-corrected RMSD is meaningful.
rdkit = pytest.importorskip("rdkit")
from rdkit import Chem            # noqa: E402
from rdkit.Chem import AllChem    # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from evoliez.types import LigandAtom as _LA  # noqa: E402

_SMI = "CC(=O)Oc1ccccc1C(=O)O"   # aspirin: asymmetric, so a flip really differs


def _real_mol(seed=7, dx=0.0):
    m = Chem.AddHs(Chem.MolFromSmiles(_SMI))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    m = Chem.RemoveHs(m)
    if dx:
        c = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            p = c.GetAtomPosition(i)
            c.SetAtomPosition(i, Point3D(p.x + dx, p.y, p.z))
    return m


def _ref_atoms_from(mol):
    conf = mol.GetConformer()
    return [_LA(id=f"{a.GetSymbol()}{i}", element=a.GetSymbol(),
               coord=(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                      conf.GetAtomPosition(i).z))
            for i, a in enumerate(mol.GetAtoms())]


def _gnina_record(mol, aff, cnn, cnn_aff):
    """A gnina-style SDF record: a real molblock + the three score tags."""
    return (Chem.MolToMolBlock(mol).rstrip("$\n")
            + f"\n>  <minimizedAffinity>\n{aff}\n"
              f"\n>  <CNNscore>\n{cnn}\n"
              f"\n>  <CNNaffinity>\n{cnn_aff}\n\n$$$$\n")


def test_select_reference_consistent_picks_geometry_not_cnn_rank1(tmp_path):
    ref = _real_mol(seed=7, dx=0.0)
    ref_atoms = _ref_atoms_from(ref)
    # mode 0 = gnina CNN-rank-1 but FLIPPED far (6 A translation) -> best affinity;
    # mode 1 = moderate (2 A); mode 2 = matches the reference (~0 A) but WORST aff.
    flipped = _real_mol(seed=7, dx=6.0)
    middle = _real_mol(seed=7, dx=2.0)
    onref = _real_mol(seed=7, dx=0.0)
    sdf = tmp_path / "c_gnina_out.sdf"
    sdf.write_text(_gnina_record(flipped, -12.182, 0.95, 6.6)    # rank 1
                   + _gnina_record(middle, -9.0, 0.70, 5.5)       # rank 2
                   + _gnina_record(onref, -7.5, 0.40, 4.8))       # rank 3
    poses = gnina.parse_all_modes(sdf, ref_atoms, candidate_id="c")
    assert [p.rank for p in poses] == [1, 2, 3]
    chosen = gnina.select_reference_consistent(poses, sdf, ref_atoms, _SMI)
    # the rank-3 mode (on the reference) is chosen, NOT the rank-1 flipped pose
    assert chosen.rank == 3 and chosen.cluster == 2
    # it KEEPS its own gnina minimizedAffinity (the worse value, honest provenance)
    assert chosen.score == -7.5
    # and is stamped with the recomputed (small) symmetry-corrected RMSD-to-ref
    assert chosen.rmsd_to_reference is not None and chosen.rmsd_to_reference < 1.0
    # sanity: the rank-1 flipped pose really is far from the reference
    from evoliez.features.ligand import symmetry_corrected_rmsd
    blocks = gnina._mode_molblocks(sdf)
    r0 = symmetry_corrected_rmsd(blocks[0], ref_atoms, _SMI)
    assert r0 is not None and r0 > 4.0          # rank-1 is the far/flipped one


def test_select_reference_consistent_falls_back_to_rank1_without_smiles(tmp_path):
    ref = _real_mol(seed=7, dx=0.0)
    ref_atoms = _ref_atoms_from(ref)
    sdf = tmp_path / "c_gnina_out.sdf"
    sdf.write_text(_gnina_record(_real_mol(7, 6.0), -12.182, 0.95, 6.6)
                   + _gnina_record(_real_mol(7, 0.0), -7.5, 0.40, 4.8))
    poses = gnina.parse_all_modes(sdf, ref_atoms, candidate_id="c")
    # no SMILES template -> nothing scorable -> honest fallback to rank-1
    chosen = gnina.select_reference_consistent(poses, sdf, ref_atoms, None)
    assert chosen.rank == 1 and chosen.score == -12.182


def test_mode_molblocks_aligns_with_parse_all_modes(tmp_path):
    # the molblock splitter must be index-aligned with parse_all_modes' mode order
    sdf = tmp_path / "c_gnina_out.sdf"
    sdf.write_text(_gnina_record(_real_mol(7, 0.0), -9.0, 0.9, 6.0)
                   + _gnina_record(_real_mol(7, 3.0), -8.0, 0.7, 5.0))
    blocks = gnina._mode_molblocks(sdf)
    poses = gnina.parse_all_modes(sdf, _ref_atoms_from(_real_mol(7, 0.0)))
    assert len(blocks) == len(poses) == 2
    # each block parses to the right heavy-atom count via the shared normalizer
    from evoliez.features.ligand import normalize_pose_mol, rmsd_template
    tmpl = rmsd_template(_SMI)
    for b in blocks:
        _m, qc = normalize_pose_mol(b, "sdf", tmpl)
        assert qc["heavy"] == 13          # aspirin heavy-atom count
