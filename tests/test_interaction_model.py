import numpy as np

from evoliez.adapters.base import place_ligand_in_pocket, synthetic_structure
from evoliez.config import LigandInput
from evoliez.features.interaction_descriptor import (
    complex_fingerprint,
    fingerprint_dim,
)
from evoliez.features.ligand import parse_ligand
from evoliez.ml.interaction_model import InteractionModel
from evoliez.ml.pose_selection import PoseRecord, select_poses


def _complex(seq, seed):
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="CC(=O)OP(=O)(O)O"))
    s = synthetic_structure(seq, seed=seed)
    placed = place_ligand_in_pocket(s, lig, seed=seed)
    return s, placed


def test_fingerprint_is_fixed_length_regardless_of_size():
    dim = fingerprint_dim(6, 8)
    s1, l1 = _complex("ACDEFGHIKLMNPQRSTVWY", 1)
    s2, l2 = _complex("ACDEFGHIKLMNPQRSTVWY" * 4, 2)
    fp1 = complex_fingerprint(s1, l1, cutoff=6.0, k_nearest=6, n_bins=8)
    fp2 = complex_fingerprint(s2, l2, cutoff=6.0, k_nearest=6, n_bins=8)
    assert fp1.shape == (dim,)
    assert fp2.shape == (dim,)  # different protein length -> same descriptor size


def test_pose_selection_splits_consensus_and_outliers():
    rng = np.random.RandomState(0)
    base = np.linspace(0.2, 0.9, 20)
    records = []
    for i in range(24):  # consensus cluster
        records.append(
            PoseRecord(f"hom_{i % 4}", base + rng.normal(0, 0.01, 20),
                       1.0, 0.7, 5.0)
        )
    for i in range(6):  # clear outliers
        records.append(
            PoseRecord(f"hom_{i % 4}", base + 5.0 + rng.normal(0, 0.2, 20),
                       1.0, 0.3, 1.0)
        )
    sel = select_poses(records, select_z=2.5, outlier_z=4.0,
                       min_decoys_per_group=4, seed=7)
    assert sel.n_positive > 10
    assert sel.n_outlier >= 1
    assert sel.n_decoy == 4 * 4  # 4 groups x 4 easy decoys
    assert sel.n_hard_decoy == 4 * 2  # 4 groups x 2 hard wrong-pose decoys
    assert set(sel.y.tolist()) == {0, 1}
    assert sel.X.shape[1] == 20 + 3  # fingerprint + [membership, identity, pred]
    # 4-class soft labels (expert review #3)
    assert set(sel.pose_class.tolist()).issubset({0, 1, 2, 3})
    assert sel.soft_y.min() >= 0.0 and sel.soft_y.max() <= 1.0
    # consensus has highest soft target, decoys lowest
    assert sel.soft_y[sel.pose_class == 3].mean() > sel.soft_y[
        sel.pose_class == 0
    ].mean()


def test_alternative_band_not_auto_negative():
    rng = np.random.RandomState(1)
    base = np.linspace(0.2, 0.9, 16)
    records = [PoseRecord("h0", base + rng.normal(0, 0.01, 16), 1.0, 0.6, 4.0)
               for _ in range(20)]
    # mid-band poses (between select_z and outlier_z)
    records += [PoseRecord("h0", base + 0.6, 1.0, 0.4, 2.0) for _ in range(8)]
    # outlier_z huge so the far group is "alternative", not "outlier"
    sel = select_poses(records, select_z=0.1, outlier_z=1e9,
                       keep_alternative_band=True)
    assert sel.n_alternative >= 1
    # alternative poses are positive-ish (soft 0.5), NOT hard negatives
    alt = sel.soft_y[sel.pose_class == 2]
    assert alt.size and abs(float(alt.mean()) - 0.5) < 1e-6


def test_model_trains_and_ranks_consensus_above_outlier():
    rng = np.random.RandomState(1)
    base = np.linspace(0.3, 0.8, 16)
    records = [
        PoseRecord(f"h{i % 3}", base + rng.normal(0, 0.01, 16), 1.0, 0.6, 4.0)
        for i in range(30)
    ]
    records += [
        PoseRecord(f"h{i % 3}", base + 6.0, 1.0, 0.2, 0.5) for i in range(6)
    ]
    sel = select_poses(records, seed=3)
    model = InteractionModel(cutoff=6.0, k_nearest=6).fit(sel)

    near = np.concatenate([base, [0.0, 1.0, 4.0]])
    far = np.concatenate([base + 6.0, [0.0, 1.0, 0.5]])
    p_near = model.score_vector(near)
    p_far = model.score_vector(far)
    assert 0.0 <= p_far <= p_near <= 1.0
    assert p_near > p_far  # family-consistent geometry scores higher


def test_model_save_load_roundtrip(tmp_path):
    base = np.linspace(0.3, 0.8, 12)
    records = [PoseRecord("h0", base, 1.0, 0.5, 3.0) for _ in range(10)]
    records += [PoseRecord("h0", base + 5, 1.0, 0.1, 0.0) for _ in range(3)]
    model = InteractionModel(cutoff=6.0, k_nearest=6).fit(select_poses(records))
    p = tmp_path / "im.json"
    model.save(p)
    reloaded = InteractionModel.load(p)
    v = np.concatenate([base, [0.0, 1.0, 3.0]])
    assert abs(reloaded.score_vector(v) - model.score_vector(v)) < 1e-9
