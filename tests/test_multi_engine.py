"""Multi-engine pose consensus (s06b augmentation): classify docking poses by
their relationship to the Boltz consensus into weighted training rows.

Fully synthetic + mocked — NO real gnina/diffdock. We build a Boltz consensus
pose, then three docking poses (one agreeing, one discordant-high-score, one
clashing) and assert each lands in the right role / source / weight, and that
``select_poses`` still runs once those mixed-source rows are merged in.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from evoliez.adapters.base import place_ligand_in_pocket, synthetic_structure
from evoliez.config import InteractionModelConfig, LigandInput
from evoliez.features.interaction_descriptor import complex_fingerprint
from evoliez.features.ligand import parse_ligand
from evoliez.features.geometry import rmsd
from evoliez.ml.multi_engine import (
    DockingPoseInput,
    _assign_role,
    _fp_overlap,
    _has_clash,
    classify_docking_poses,
)
from evoliez.ml.pose_selection import PoseRecord, select_poses

SEQ = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


def _structure_and_consensus():
    """A synthetic receptor + a consensus ligand pose placed in its pocket, plus
    the Boltz consensus fingerprint that pose induces."""
    struct = synthetic_structure(SEQ, seed=7)
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="CC(=O)OP(=O)(O)O"))
    consensus_atoms = place_ligand_in_pocket(struct, lig, seed=7)
    consensus_fp = complex_fingerprint(struct, consensus_atoms, cutoff=6.0, k_nearest=6)
    return struct, consensus_atoms, consensus_fp


def _translate(atoms, dx, dy, dz):
    out = []
    for a in atoms:
        na = copy.deepcopy(a)
        na.coord = (a.coord[0] + dx, a.coord[1] + dy, a.coord[2] + dz)
        out.append(na)
    return out


def _cfg():
    # real config -> carries the multi-engine threshold defaults
    return InteractionModelConfig(
        multi_engine=True, multi_engine_methods=["gnina", "diffdock"],
        consensus_overlap_rmsd=2.0, consensus_overlap_fp=0.6,
        hard_negative_weight=1.0, weak_positive_weight=0.3,
    )


# --------------------------------------------------------------------------- #
# The decision table, exercised directly (pure, no fingerprints needed).
# --------------------------------------------------------------------------- #
def test_assign_role_table():
    kw = dict(rmsd_thr=2.0, fp_thr=0.6, w_weak=0.3, w_hard=1.0,
              primary_only=False, score_gate_pass=True)
    # AGREE: close RMSD + high overlap + no clash + key ok -> weak positive
    role, w, _ = _assign_role(rmsd_to_consensus=0.5, fp_overlap=0.95,
                              clash=False, key_ok=True, **kw)
    assert role == "weak_positive" and w == 0.3
    # DISCORDANT (low overlap) but no clash -> hard negative
    role, w, _ = _assign_role(rmsd_to_consensus=0.5, fp_overlap=0.2,
                              clash=False, key_ok=True, **kw)
    assert role == "hard_negative" and w == 1.0
    # high overlap but RMSD too far -> still discordant (hard negative)
    role, _, _ = _assign_role(rmsd_to_consensus=9.0, fp_overlap=0.95,
                              clash=False, key_ok=True, **kw)
    assert role == "hard_negative"
    # clash -> excluded (weight irrelevant)
    role, _, _ = _assign_role(rmsd_to_consensus=0.1, fp_overlap=0.99,
                              clash=True, key_ok=True, **kw)
    assert role == "excluded"
    # catalytic contact broken -> strong negative (kept as a negative)
    role, w, _ = _assign_role(rmsd_to_consensus=0.1, fp_overlap=0.99,
                              clash=False, key_ok=False, **kw)
    assert role == "strong_negative" and w == 1.0


def test_assign_role_primary_only_stricter_bar():
    # an overlap that AGREES for a full pose is NOT enough for a primary-only pose
    kw = dict(rmsd_to_consensus=0.5, fp_overlap=0.65, clash=False, key_ok=True,
              score_gate_pass=True, rmsd_thr=2.0, fp_thr=0.6, w_weak=0.3,
              w_hard=1.0)
    full = _assign_role(primary_only=False, **kw)[0]
    prim = _assign_role(primary_only=True, **kw)[0]
    assert full == "weak_positive"        # clears 0.6
    assert prim == "hard_negative"        # must clear the stricter 0.75


def test_assign_role_discordant_requires_score_gate():
    kw = dict(rmsd_to_consensus=0.5, fp_overlap=0.2, clash=False, key_ok=True,
              primary_only=False, rmsd_thr=2.0, fp_thr=0.6, w_weak=0.3,
              w_hard=1.0)
    assert _assign_role(score_gate_pass=True, **kw)[0] == "hard_negative"
    assert _assign_role(score_gate_pass=False, **kw)[0] == "excluded"


# --------------------------------------------------------------------------- #
# End-to-end classification of three synthetic docking poses.
# --------------------------------------------------------------------------- #
def test_classify_agree_discordant_clash():
    struct, consensus_atoms, consensus_fp = _structure_and_consensus()
    cfg = _cfg()

    # 1) AGREE: the docked pose IS the consensus pose (gnina) -> weak positive.
    agree = DockingPoseInput(
        source="gnina", structure=struct, ligand_atoms=consensus_atoms,
        score=-9.0, score_is_better_low=True,
    )
    # 2) DISCORDANT but high-scoring (diffdock): shifted to a different pocket
    #    region -> different contact fingerprint, but no clash -> hard negative.
    discordant_atoms = _translate(consensus_atoms, 0.0, 0.0, 18.0)
    discordant = DockingPoseInput(
        source="diffdock", structure=struct, ligand_atoms=discordant_atoms,
        score=0.95, score_is_better_low=False,
    )
    # 3) CLASH: bury two ligand atoms essentially ON two residue sidechain points
    #    (interpenetration) -> excluded.
    clash_atoms = copy.deepcopy(consensus_atoms)
    for li, ri in ((0, 10), (1, 12)):
        sc = struct.residues[ri].sidechain_centroid
        clash_atoms[li].coord = (sc[0] + 0.1, sc[1], sc[2])
    clashing = DockingPoseInput(
        source="gnina", structure=struct, ligand_atoms=clash_atoms, score=-12.0,
    )

    kept = classify_docking_poses(
        consensus_fp, consensus_atoms, [agree, discordant, clashing],
        cfg=cfg, key_positions=[],
    )

    # the clashing pose is dropped; the other two are kept
    assert len(kept) == 2
    by_role = {r.role: r for r in kept}
    assert set(by_role) == {"weak_positive", "hard_negative"}

    wp = by_role["weak_positive"]
    assert wp.source == "gnina"
    assert wp.sample_weight == pytest.approx(0.3)
    assert wp.pred_score == pytest.approx(-9.0)

    hn = by_role["hard_negative"]
    assert hn.source == "diffdock"
    assert hn.sample_weight == pytest.approx(1.0)

    # diagnostics cover ALL three poses, including the excluded clash
    diags = kept.diagnostics
    assert len(diags) == 3
    assert any(d.role == "excluded" and d.clash for d in diags)
    # every kept record carries a recognised role + a non-boltz source tag
    assert all(r.source in {"gnina", "diffdock"} for r in kept)
    assert all(r.role in {"weak_positive", "hard_negative", "strong_negative"}
               for r in kept)


def _classify_reference(consensus_fp, consensus_atoms, poses, cfg, key_positions):
    """Recompute the per-pose diagnostics the OLD (per-POSE) way: every shared
    quantity (consensus contact pattern + norm, consensus coords, clash points)
    rebuilt INSIDE the loop via the un-cached helpers. This is the pre-hoist
    reference the hoisted ``classify_docking_poses`` must reproduce BIT-FOR-BIT.
    """
    from evoliez.features.geometry import catalytic_distances
    from evoliez.features.interaction_descriptor import complex_fingerprint
    from evoliez.ml.multi_engine import (
        PoseClassification, _assign_role, _key_contacts_preserved,
    )

    cutoff = getattr(cfg, "contact_cutoff", 6.0)
    k_nearest = getattr(cfg, "k_nearest_residues", 6)
    n_bins = 8
    rmsd_thr = getattr(cfg, "consensus_overlap_rmsd", 2.0)
    fp_thr = getattr(cfg, "consensus_overlap_fp", 0.6)
    w_hard = getattr(cfg, "hard_negative_weight", 1.0)
    w_weak = getattr(cfg, "weak_positive_weight", 0.3)
    cfp = np.asarray(consensus_fp, dtype=float).ravel()

    consensus_cat = {}
    key_positions = list(key_positions or [])
    if key_positions and consensus_atoms and poses:
        consensus_cat = catalytic_distances(
            poses[0].structure, consensus_atoms, key_positions)

    out = []
    for dp in poses:
        fp = complex_fingerprint(
            dp.structure, dp.ligand_atoms, cutoff=cutoff, k_nearest=k_nearest)
        # un-cached helpers, consensus side rebuilt every pose:
        overlap = _fp_overlap(fp, cfp, n_bins=n_bins)
        if consensus_atoms and dp.ligand_atoms:
            r = rmsd(
                np.array([a.coord for a in consensus_atoms], dtype=float),
                np.array([a.coord for a in dp.ligand_atoms], dtype=float))
        else:
            r = float("nan")
        clash = _has_clash(dp.structure, dp.ligand_atoms)
        key_ok = _key_contacts_preserved(
            dp.structure, dp.ligand_atoms, consensus_cat, key_positions)
        role, weight, note = _assign_role(
            rmsd_to_consensus=r, fp_overlap=overlap, clash=clash, key_ok=key_ok,
            primary_only=dp.primary_only, score_gate_pass=dp.score_gate_pass,
            rmsd_thr=rmsd_thr, fp_thr=fp_thr, w_weak=w_weak, w_hard=w_hard)
        out.append((role, round(weight, 3),
                    (round(r, 3) if r == r else float("nan")),
                    round(overlap, 3)))
    return out


def test_hoist_is_bit_identical_to_per_pose_reference():
    """The per-TARGET hoist (consensus pattern/norm, consensus coords, clash pts
    lifted out of the per-pose loop) must leave EVERY per-pose diagnostic
    byte-for-byte unchanged vs the un-cached, recomputed-per-pose reference.

    Poses are built to exercise all four hoisted quantities AND every role:
    an agreeing pose (clears overlap+rmsd), a discordant high-score pose, a
    clashing pose, a primary-only pose (stricter overlap bar), and a pose that
    breaks a declared catalytic contact (key-contacts path)."""
    struct, consensus_atoms, consensus_fp = _structure_and_consensus()
    cfg = _cfg()

    # a declared catalytic residue genuinely in contact with the consensus pose
    from evoliez.features.geometry import catalytic_distances
    key = None
    for rsd in struct.residues:
        if list(catalytic_distances(struct, consensus_atoms, [rsd.index]).values())[0] < 4.0:
            key = rsd.index
            break
    assert key is not None

    clash_atoms = copy.deepcopy(consensus_atoms)
    for li, ri in ((0, 10), (1, 12)):
        sc = struct.residues[ri].sidechain_centroid
        clash_atoms[li].coord = (sc[0] + 0.1, sc[1], sc[2])

    poses = [
        DockingPoseInput(source="gnina", structure=struct,
                         ligand_atoms=consensus_atoms, score=-9.0),
        DockingPoseInput(source="diffdock", structure=struct,
                         ligand_atoms=_translate(consensus_atoms, 0.0, 0.0, 18.0),
                         score=0.95, score_is_better_low=False),
        DockingPoseInput(source="gnina", structure=struct,
                         ligand_atoms=clash_atoms, score=-12.0),
        DockingPoseInput(source="gnina", structure=struct,
                         ligand_atoms=consensus_atoms, score=-8.0,
                         primary_only=True),
        DockingPoseInput(source="gnina", structure=struct,
                         ligand_atoms=_translate(consensus_atoms, 0.0, 0.0, 12.0),
                         score=-8.0),
    ]

    for key_positions in ([], [key]):
        reference = _classify_reference(
            consensus_fp, consensus_atoms, poses, cfg, key_positions)
        kept = classify_docking_poses(
            consensus_fp, consensus_atoms, poses, cfg=cfg,
            key_positions=key_positions)
        diags = kept.diagnostics
        assert len(diags) == len(reference) == len(poses)
        for d, (role, weight, r_ref, ov_ref) in zip(diags, reference):
            assert d.role == role
            assert d.sample_weight == weight            # exact (both round(.,3))
            assert d.fp_overlap == ov_ref               # exact, bit-identical
            if r_ref == r_ref:                          # finite rmsd
                assert d.rmsd_to_consensus == r_ref     # exact
            else:
                assert d.rmsd_to_consensus != d.rmsd_to_consensus  # NaN both


def test_classify_empty_consensus_atoms_skips_rmsd_gate():
    """With NO consensus pose atoms, the hoisted ``cons_coords`` guard must leave
    the rmsd branch skipped exactly as the per-pose code did (rmsd -> NaN), and
    classification still proceeds on overlap + clash."""
    struct, consensus_atoms, consensus_fp = _structure_and_consensus()
    cfg = _cfg()
    pose = DockingPoseInput(source="gnina", structure=struct,
                            ligand_atoms=consensus_atoms, score=-9.0)
    kept = classify_docking_poses(
        consensus_fp, [], [pose], cfg=cfg, key_positions=[])  # empty consensus
    assert len(kept.diagnostics) == 1
    d = kept.diagnostics[0]
    assert d.rmsd_to_consensus != d.rmsd_to_consensus       # NaN: gate skipped
    # reference (per-pose) yields the same NaN rmsd + same overlap/role
    ref = _classify_reference(consensus_fp, [], [pose], cfg, [])[0]
    assert d.role == ref[0]
    assert d.fp_overlap == ref[3]


def test_catalytic_contact_break_is_strong_negative():
    """A pose that moves the ligand off a declared catalytic residue (its contact
    distance changes by more than the tolerance) is a strong negative, not a
    silently dropped row."""
    struct, consensus_atoms, consensus_fp = _structure_and_consensus()
    cfg = _cfg()
    # catalytic residue = the pocket residue the consensus ligand sits on.
    from evoliez.features.geometry import catalytic_distances
    # find a residue genuinely in contact with the consensus ligand
    near = None
    for r in struct.residues:
        cd = catalytic_distances(struct, consensus_atoms, [r.index])
        if list(cd.values())[0] < 4.0:
            near = r.index
            break
    assert near is not None
    # move the ligand far from that catalytic residue (breaks the contact) but
    # keep it near OTHER residues so it does not clash and still has contacts.
    moved = _translate(consensus_atoms, 0.0, 0.0, 12.0)
    pose = DockingPoseInput(source="gnina", structure=struct,
                            ligand_atoms=moved, score=-8.0)
    kept = classify_docking_poses(
        consensus_fp, consensus_atoms, [pose], cfg=cfg, key_positions=[near],
    )
    assert len(kept) == 1
    assert kept[0].role == "strong_negative"
    assert kept[0].source == "gnina"


# --------------------------------------------------------------------------- #
# select_poses must still run once Boltz + docking rows are merged.
# --------------------------------------------------------------------------- #
def test_select_poses_runs_with_mixed_source_records():
    struct, consensus_atoms, consensus_fp = _structure_and_consensus()
    cfg = _cfg()
    fp_dim = consensus_fp.shape[0]
    rng = np.random.RandomState(0)

    # a Boltz ensemble (the teacher) across 4 homolog groups
    boltz = [
        PoseRecord(f"hom_{i % 4:03d}", consensus_fp + rng.normal(0, 0.01, fp_dim),
                   1.0, 0.7, 5.0)
        for i in range(24)
    ]
    boltz += [  # a couple of clear Boltz outliers
        PoseRecord(f"hom_{i % 4:03d}", consensus_fp + 5.0, 1.0, 0.3, 1.0)
        for i in range(4)
    ]

    # docking augmentation: 1 agree + 1 discordant
    agree = DockingPoseInput(source="gnina", structure=struct,
                             ligand_atoms=consensus_atoms, score=-9.0)
    discordant = DockingPoseInput(
        source="diffdock", structure=struct,
        ligand_atoms=_translate(consensus_atoms, 0.0, 0.0, 18.0), score=0.9,
        score_is_better_low=False)
    dock = classify_docking_poses(consensus_fp, consensus_atoms,
                                  [agree, discordant], cfg=cfg, key_positions=[])

    records = boltz + list(dock)
    sources = {r.source for r in records}
    assert sources == {"boltz", "gnina", "diffdock"}

    sel = select_poses(records, select_z=2.5, outlier_z=4.0,
                       min_decoys_per_group=4, seed=7)
    # mixed-source selection produced a usable training matrix
    assert sel.X.shape[0] > 0
    assert sel.X.shape[1] == fp_dim          # fingerprint-only feature space
    assert sel.weights.shape[0] == sel.X.shape[0]
    assert set(sel.y.tolist()) <= {0, 1}
    # the docking rows were counted as multi-engine roles
    assert sel.n_weak_positive == 1
    assert sel.n_hard_negative == 1
    # the Boltz consensus is unchanged by appending docking rows: it must equal
    # the median of the Boltz-only fingerprints (docking poses never shift it).
    boltz_only_consensus = np.median(
        np.vstack([r.fingerprint for r in boltz]), axis=0)
    assert np.allclose(sel.consensus, boltz_only_consensus)


def test_multi_engine_off_is_byte_identical():
    """With no docking rows (multi_engine off), select_poses output is identical
    to before the extension — same consensus, same counts, zero augmentation."""
    _, _, consensus_fp = _structure_and_consensus()
    fp_dim = consensus_fp.shape[0]
    rng = np.random.RandomState(1)
    records = [PoseRecord(f"h{i % 3}", consensus_fp + rng.normal(0, 0.01, fp_dim),
                          1.0, 0.6, 4.0) for i in range(18)]
    sel = select_poses(records, seed=3)
    assert sel.n_weak_positive == 0 and sel.n_hard_negative == 0
    assert np.allclose(
        sel.consensus, np.median(np.vstack([r.fingerprint for r in records]), 0))
