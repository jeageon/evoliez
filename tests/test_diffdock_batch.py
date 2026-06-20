"""DiffDock GPU-batch + all-ranks parsing (s06b multi-engine redesign).

``redock_batch`` runs ONE model load for MANY targets and returns
``{complex_id: [Pose, ...]}`` (all ranks per complex). ``parse_all_ranks``
recovers EVERY ``rankN_confidence*.sdf`` for a complex with the confidence
encoded in the filename. Both must work on the mock backend / a synthetic rank
layout (no real DiffDock).
"""

from __future__ import annotations

from pathlib import Path

from evoliez.adapters import diffdock
from evoliez.config import Backend, DockingConfig
from evoliez.types import LigandAtom, ProteinStructure


def _ref():
    return [
        LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="O1", element="O", coord=(1.2, 0.0, 0.0)),
        LigandAtom(id="N2", element="N", coord=(0.0, 1.3, 0.0)),
    ]


def _structure():
    return ProteinStructure(sequence="AG", residues=[])


def _sdf_3atoms(dz: float = 0.0) -> str:
    atoms = "\n".join(
        f"{x:10.4f}{y:10.4f}{z:10.4f} {el}   0  0  0  0  0  0  0  0  0  0  0  0"
        for (x, y, z, el) in [
            (0.0, 0.0, 0.0 + dz, "C"),
            (1.2, 0.0, 0.0 + dz, "O"),
            (0.0, 1.3, 0.0 + dz, "N"),
        ]
    )
    return (
        "lig\n     RDKit          3D\n\n"
        "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
        f"{atoms}\n  1  2  1  0\n  1  3  1  0\nM  END\n$$$$\n"
    )


def _rank_layout(complex_dir: Path) -> None:
    """A DiffDock-style per-complex output: rankN_confidence-X.XX.sdf (+ a bare
    rank1.sdf that must be superseded by the confidence-bearing one)."""
    complex_dir.mkdir(parents=True, exist_ok=True)
    (complex_dir / "rank1.sdf").write_text(_sdf_3atoms(0.0))            # bare
    (complex_dir / "rank1_confidence-0.53.sdf").write_text(_sdf_3atoms(0.0))
    (complex_dir / "rank2_confidence-1.20.sdf").write_text(_sdf_3atoms(1.5))
    (complex_dir / "rank3_confidence-2.80.sdf").write_text(_sdf_3atoms(3.0))
    # decoys that the integer-rank parse must NOT confuse for rank 1
    (complex_dir / "rank10_confidence-4.00.sdf").write_text(_sdf_3atoms(4.0))


# --------------------------- parse_all_ranks -------------------------------- #
def test_parse_all_ranks_recovers_every_rank(tmp_path):
    d = tmp_path / "cplx"
    _rank_layout(d)
    poses = diffdock.parse_all_ranks(
        d, _ref(), candidate_id="cplx",
        command_args="python -m inference ...", engine_version="diffdock 1.1",
    )
    # 4 distinct ranks (1,2,3,10); the bare rank1.sdf is folded into rank-1.
    assert [p.rank for p in poses] == [1, 2, 3, 10]
    # confidence from the FILENAME -> .score (higher = better; rank-1 best)
    assert poses[0].score == -0.53 and poses[1].score == -1.2
    assert poses[3].score == -4.0
    assert all(p.score_type == "diffdock_confidence" for p in poses)
    assert all(p.method == "diffdock" for p in poses)
    assert all(p.engine_version == "diffdock 1.1" for p in poses)
    assert all("inference" in p.command_args for p in poses)
    # rank-1 used the confidence-bearing file's coords (dz=0 -> ~0 RMSD)
    assert poses[0].ligand_atoms and poses[0].rmsd_to_reference is not None
    assert poses[0].rmsd_to_reference < poses[2].rmsd_to_reference


def test_parse_all_ranks_bare_rank_unscored(tmp_path):
    d = tmp_path / "bare"
    d.mkdir(parents=True)
    (d / "rank1.sdf").write_text(_sdf_3atoms(0.0))   # no confidence token
    poses = diffdock.parse_all_ranks(d, _ref())
    assert len(poses) == 1 and poses[0].rank == 1 and poses[0].score == 0.0


def test_parse_all_ranks_empty(tmp_path):
    assert diffdock.parse_all_ranks(tmp_path / "nope", _ref()) == []


# ------------------------------ redock_batch -------------------------------- #
def test_redock_batch_mock_three_tasks():
    cfg = DockingConfig()
    n = cfg.poses_per_candidate
    tasks = [
        (f"cx{i}", _structure(), _ref(), "OP(=O)(O)O")
        for i in range(3)
    ]
    out = diffdock.redock_batch(
        tasks, Path("/unused_out"), cfg, backend=Backend.mock,
    )
    # one entry per complex id
    assert set(out) == {"cx0", "cx1", "cx2"}
    for cid, poses in out.items():
        assert len(poses) == n >= 1
        assert [p.rank for p in poses] == list(range(1, n + 1))
        # diffdock confidence: rank-1 is the HIGHEST score (descending ladder)
        scores = [p.score for p in poses]
        assert scores == sorted(scores, reverse=True)
        assert poses[0].score == max(scores)
        assert all(p.method == "diffdock" for p in poses)
        assert all(p.score_type == "diffdock_confidence" for p in poses)
        assert all(p.candidate_id == cid for p in poses)


def test_redock_single_back_compat_returns_rank1():
    cfg = DockingConfig()
    one = diffdock.redock("c", _structure(), _ref(), cfg, Path("/unused"),
                          instability=0.1, smiles="CCO", backend=Backend.mock)
    first = diffdock.redock_all("c", _structure(), _ref(), cfg, Path("/unused"),
                                instability=0.1, smiles="CCO",
                                backend=Backend.mock)[0]
    assert one.rank == 1 and one.method == "diffdock"
    assert one.score == first.score
    # diffdock confidence: higher is better -> rank-1 is the max over all ranks
    allp = diffdock.redock_all("c", _structure(), _ref(), cfg, Path("/unused"),
                               instability=0.1, smiles="CCO",
                               backend=Backend.mock)
    assert one.score == max(p.score for p in allp)
