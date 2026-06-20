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
    # the single-pose redock() must equal redock_all()[0].
    cfg = DockingConfig()
    one = gnina.redock("c", _structure(), _ref(), cfg, Path("/unused"),
                       instability=0.1, backend=Backend.mock)
    first = gnina.redock_all("c", _structure(), _ref(), cfg, Path("/unused"),
                             instability=0.1, backend=Backend.mock)[0]
    assert one.rank == 1
    assert one.score == first.score and one.method == "gnina"
