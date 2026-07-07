"""Unit tests for the reference-like pose gate (pure-numpy, synthetic coords)."""
import numpy as np

from evoliez.md.pose_gate import (
    ALTERNATIVE_POSE, DISPLACED, REFERENCE_LIKE, SKIPPED,
    evaluate_pose, gate_from_pdb, kabsch_apply, read_pdb_atoms, thresholds_for,
)


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    pocket = rng.normal(0, 8, (20, 3))   # 20 pocket Cα
    lig = rng.normal(0, 4, (12, 3))      # 12 ligand heavy atoms
    return pocket, lig


def test_reference_like():
    pocket, lig = _setup()
    test_lig = lig + np.random.default_rng(1).normal(0, 0.1, lig.shape)
    r = evaluate_pose(pocket, pocket.copy(), lig, test_lig, role="cofactor")
    assert r.status == REFERENCE_LIKE, r.to_json()
    assert r.pocket_pose_rmsd < 0.5 and r.internal_rmsd < 0.5


def test_alternative_pose_translation():
    pocket, lig = _setup()
    test_lig = lig + np.array([6.0, 0.0, 0.0])      # 6 Å rigid shift
    r = evaluate_pose(pocket, pocket.copy(), lig, test_lig, role="cofactor")
    # pocket aligns to identity -> pose RMSD ~6 (>3 max, <8 displaced); internal ~0
    assert r.status == ALTERNATIVE_POSE, r.to_json()
    assert 5.5 < r.pocket_pose_rmsd < 6.5
    assert r.internal_rmsd < 1e-6


def test_displaced():
    pocket, lig = _setup()
    test_lig = lig + np.array([9.0, 0.0, 0.0])      # 9 Å > cofactor displaced 8
    r = evaluate_pose(pocket, pocket.copy(), lig, test_lig, role="cofactor")
    assert r.status == DISPLACED, r.to_json()


def test_internal_distortion():
    pocket, lig = _setup()
    test_lig = lig + np.random.default_rng(2).normal(0, 2.5, lig.shape)  # shape change
    r = evaluate_pose(pocket, pocket.copy(), lig, test_lig, role="cofactor")
    # internal-shape RMSD exceeds the 2.0 cofactor cutoff
    assert r.status == ALTERNATIVE_POSE, r.to_json()
    assert r.internal_rmsd > 2.0


def test_role_thresholds_differ():
    assert thresholds_for("metal").pocket_rmsd_max < thresholds_for("substrate").pocket_rmsd_max
    assert thresholds_for(None).pocket_rmsd_max == thresholds_for("other").pocket_rmsd_max


def test_correspondence_mismatch_skips():
    pocket, lig = _setup()
    r = evaluate_pose(pocket, pocket.copy(), lig, lig[:-2], role="cofactor")
    assert r.status == SKIPPED


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 5, (15, 3))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Y = X @ R.T + np.array([3.0, -2.0, 1.0])
    back = kabsch_apply(Y, X, Y)
    assert np.sqrt(((back - X) ** 2).sum(1).mean()) < 1e-6


def _pdb(atoms):
    """atoms: list of (record, serial, name, resname, chain, resseq, x,y,z, elem)."""
    out = []
    for rec, ser, name, resn, ch, rs, x, y, z, el in atoms:
        out.append(f"{rec:<6}{ser:>5} {name:<4} {resn:>3} {ch}{rs:>4}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}")
    return "\n".join(out) + "\n"


def test_gate_from_pdb_roundtrip(tmp_path):
    # 4 pocket residues (CA) around origin + a 3-atom ligand
    base = []
    coords = [(0, 0, 0), (3, 0, 0), (0, 3, 0), (0, 0, 3)]
    for i, (x, y, z) in enumerate(coords, 1):
        base.append(("ATOM", i, "CA", "ALA", "A", i, x, y, z, "C"))
    lig = [("HETATM", 10, "C1", "LIG", "A", 900, 1.0, 1.0, 1.0, "C"),
           ("HETATM", 11, "O1", "LIG", "A", 900, 2.0, 1.0, 1.0, "O"),
           ("HETATM", 12, "N1", "LIG", "A", 900, 1.0, 2.0, 1.0, "N")]
    ref = tmp_path / "ref.pdb"; ref.write_text(_pdb(base + lig))
    # test: same protein, ligand shifted 0.2 Å -> reference_like
    lig2 = [(r, s, n, rn, c, rs, x + 0.2, y, z, e)
            for (r, s, n, rn, c, rs, x, y, z, e) in lig]
    test = tmp_path / "test.pdb"; test.write_text(_pdb(base + lig2))
    res = gate_from_pdb(ref, test, ligand_resname="LIG", role="cofactor",
                        pocket_cutoff=12.0)
    assert res.status == REFERENCE_LIKE, res.to_json()
    assert res.n_ligand_atoms == 3 and res.n_pocket_ca == 4
    assert len(read_pdb_atoms(ref)) == 7
