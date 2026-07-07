"""Per-representative multi-engine docking augmentation (s06b).

The WT-only docking augmentation is extended to dock the design ligand into the
WT complex AND every representative homolog's Boltz structure, classifying each
pose against the SHARED family Boltz consensus. The crux the design pins:

  * overlap reference   = the FAMILY consensus fingerprint (frame-independent) —
                          the SAME numpy array for every target.
  * RMSD / catalytic ref = the TARGET's OWN Boltz ligand atoms — DIFFERENT per
                          target (each rep is its own coordinate frame, so a
                          cross-frame RMSD against the family's atoms is junk).

Fully synthetic + mock backend (no real gnina/diffdock): each rep gets a minimal
Boltz-style ``predictions/<stem>/`` dir (model pdb + confidence + plddt, exactly
the layout ``parse_prediction_dir`` scopes to), plus a WT stem dir. We run the
SERIAL per-rep path (``gpu_list=[]``) and assert (a) each target yields docking
PoseRecords tagged with its own group_id/source, (b) ``classify_docking_poses``
was called with the family consensus as the overlap reference and each target's
own atoms as the RMSD reference, and (c) ``select_poses`` runs on the combined
Boltz + docking records.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evoliez.config import Backend, DockingConfig, InteractionModelConfig
from evoliez.features.interaction_descriptor import complex_fingerprint
from evoliez.logging_utils import get_logger
from evoliez.ml import multi_engine as me
from evoliez.ml.pose_selection import PoseRecord, select_poses
from evoliez.stages.s06b_interaction_model import _augment_with_docking
from evoliez.types import Ligand, LigandAtom

SEQ = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues -> a real (non-degenerate) pocket
LOG = get_logger("test.s06b.perrep")


# --------------------------------------------------------------------------- #
# Minimal Boltz-style prediction dir (mirrors tests/test_s06b_batch.py).
# --------------------------------------------------------------------------- #
def _write_model_pdb(path: Path, n_res: int, lig_xyz):
    """A model pdb with n_res CA atoms + a ligand (HETATM chain B). Coordinates
    are spread out so the receptor is a genuine 3-D pocket (not collinear)."""
    lines = []
    for i in range(n_res):
        x, y, z = 2.0 + 1.7 * i, 3.0 + (i % 5), 4.0 + (i % 3)
        lines.append(
            f"ATOM  {i + 1:>5d}  CA  ALA A{i + 1:>4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C"
        )
    for serial, (el, (x, y, z)) in zip(range(900, 999), zip("OPC", lig_xyz)):
        rec = (f"HETATM{serial:>5d}  {el}1  LIG B   1    "
               f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
        lines.append(f"{rec:<76}{el:>2s}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _make_stem(stem_dir: Path, n_res: int, lig_xyz, n_samples: int = 2):
    """One target's Boltz prediction in the predictions/<stem>/ layout that
    parse_prediction_dir scopes to (model pdb + confidence + plddt per sample)."""
    stem_dir.mkdir(parents=True, exist_ok=True)
    name = stem_dir.name
    for k in range(n_samples):
        # small per-sample drift so the ensemble is non-degenerate
        drift = [(x + 0.2 * k, y, z) for (x, y, z) in lig_xyz]
        _write_model_pdb(stem_dir / f"{name}_model_{k}.pdb", n_res, drift)
        (stem_dir / f"confidence_{name}_model_{k}.json").write_text(json.dumps(
            {"confidence_score": 0.9 - 0.05 * k, "complex_plddt": 0.8,
             "iptm": 0.7}))
        np.savez(stem_dir / f"plddt_{name}_model_{k}.npz",
                 plddt=np.full(n_res, 0.8))


def _ligand():
    # 3-atom ligand matching the HETATM block (O/P/C), with a SMILES diffdock
    # can consume in the mock path.
    return Ligand(id="L", smiles="OP(=O)(O)O",
                  atoms=[LigandAtom(id="O0", element="O", coord=(0.0, 0.0, 0.0)),
                         LigandAtom(id="P1", element="P", coord=(0.5, 0.0, 0.0)),
                         LigandAtom(id="C2", element="C", coord=(1.0, 0.0, 0.0))])


# --------------------------------------------------------------------------- #
# A fake RunContext exposing exactly what _augment_with_docking reads.
# --------------------------------------------------------------------------- #
def _fake_ctx(root: Path, ligand):
    (root / "interaction_graphs").mkdir(parents=True, exist_ok=True)
    paths = SimpleNamespace(
        docking=root / "docking",
        complexes=root / "complexes",
        interaction_graphs=root / "interaction_graphs",
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(redocking=DockingConfig(poses_per_candidate=2)),
        complex_prediction=SimpleNamespace(primary_method="boltz2"),
        backend_for=lambda name: Backend.mock,
    )
    store = {"ligand": ligand, "catalytic_positions": [], "extra_ligands": []}

    def get(key, default=None):
        return store.get(key, default)

    def require(key):
        return store[key]

    return SimpleNamespace(config=config, paths=paths, get=get, require=require)


def _family_records(struct, lig_atoms, n_groups=4):
    """A Boltz family ensemble whose median fingerprint IS the family consensus
    (the overlap teacher). Slightly noised around the placed ligand pose."""
    base = complex_fingerprint(struct, lig_atoms, cutoff=6.0, k_nearest=6)
    rng = np.random.RandomState(0)
    recs = []
    for i in range(24):
        fp = base + rng.normal(0, 0.01, base.shape[0])
        recs.append(PoseRecord(group_id=f"hom_{i % n_groups:03d}", fingerprint=fp,
                               msa_membership=1.0, identity_to_target=0.6,
                               pred_score=5.0))
    return recs


# --------------------------------------------------------------------------- #
# The per-rep augmentation end-to-end (serial, gpu_list=[]).
# --------------------------------------------------------------------------- #
def test_per_rep_augmentation_serial(tmp_path, monkeypatch):
    n_res = len(SEQ)
    # Each target (WT + 3 reps) gets its OWN stem dir with a DIFFERENT ligand
    # placement -> different coordinate frame, so the per-target RMSD reference
    # genuinely differs (the design's crux).
    # Ligand placements sit ~3 Å off the nearest residue point (a real contact,
    # NOT within the 1.0 Å clash shell even after the mock dock's ~0.5 Å jitter)
    # with 7 contacts each -> a non-degenerate fingerprint that classifies as a
    # weak positive rather than an excluded clash. Each target uses a DIFFERENT
    # placement so its coordinate frame (the per-target RMSD reference) differs.
    def _lig(cx, cy, cz):
        return [(cx, cy, cz), (cx + 0.4, cy + 0.2, cz + 0.1),
                (cx + 0.8, cy + 0.4, cz + 0.2)]

    wt_stem = tmp_path / "complexes" / "boltz" / "boltz_results_wt" / \
        "predictions" / "wt_boltz_input"
    _make_stem(wt_stem, n_res, _lig(6.5, 3.0, 4.0))

    rep_centers = [(7.0, 3.0, 4.0), (6.5, 2.5, 5.0), (7.5, 3.0, 3.0)]
    rep_dirs = []
    for i, c in enumerate(rep_centers):
        d = tmp_path / "structures" / "representatives" / f"hom_{i:03d}" / \
            f"boltz_results_hom_{i:03d}_boltz_input" / "predictions" / \
            f"hom_{i:03d}_boltz_input"
        _make_stem(d, n_res, _lig(*c))
        rep_dirs.append(d)

    ligand = _ligand()
    ctx = _fake_ctx(tmp_path, ligand)

    # WT Complex: parse its own stem to get a real structure + ligand pose, and
    # point structure.pdb_path at the model file inside the stem dir (so
    # _augment_with_docking derives the WT stem from pdb_path.parent).
    from evoliez.adapters.boltz import parse_prediction_dir
    wt_cx = parse_prediction_dir(wt_stem, SEQ, ligand, "boltz2")
    assert wt_cx is not None and wt_cx.ligand.atoms
    wt_cx.structure.pdb_path = str(wt_stem / "wt_boltz_input_model_0.pdb")

    reps = [SimpleNamespace(sequence=SEQ) for _ in range(3)]
    rep_stem_dirs = list(rep_dirs)

    cfg = InteractionModelConfig(
        multi_engine=True, multi_engine_methods=["gnina", "diffdock"],
        consensus_overlap_rmsd=2.0, consensus_overlap_fp=0.6,
        hard_negative_weight=1.0, weak_positive_weight=0.3,
        contact_cutoff=6.0, k_nearest_residues=6,
    )

    # Family Boltz consensus records (the overlap teacher).
    records = _family_records(wt_cx.structure, wt_cx.ligand.atoms)
    family_consensus = np.median(
        np.vstack([r.fingerprint for r in records]), axis=0)
    n_boltz = len(records)

    # Spy on classify_docking_poses to capture the (consensus_fp, target_atoms)
    # each target passes -> proves overlap uses the FAMILY consensus while the
    # RMSD reference is the TARGET's own atoms.
    seen = []
    real_classify = me.classify_docking_poses

    def spy(consensus_fp, consensus_pose_atoms, docking_poses, **kw):
        seen.append((np.asarray(consensus_fp, float),
                     list(consensus_pose_atoms),
                     [dp.candidate_id for dp in docking_poses]))
        return real_classify(consensus_fp, consensus_pose_atoms,
                             docking_poses, **kw)

    # patch where the worker imports it from (module-level symbol in multi_engine)
    monkeypatch.setattr(me, "classify_docking_poses", spy)

    diags = _augment_with_docking(
        ctx, cfg, wt_cx, records, rep_stem_dirs, reps, [], LOG)

    # --- (a) every target yielded docking PoseRecords with its own group_id ---
    added = records[n_boltz:]
    assert added, "per-rep augmentation produced no docking rows"
    sources = {r.source for r in added}
    assert sources <= {"gnina", "diffdock"} and sources, sources
    # one classify call per target that produced poses (WT + 3 reps = 4)
    cand_ids = {cid for (_fp, _atoms, cids) in seen for cid in cids}
    assert "wt" in cand_ids
    assert {"rep_000", "rep_001", "rep_002"} <= cand_ids
    # the kept rows are tagged with their target via group_id
    # (classify builds group_id=f"dock_{source}_{candidate_id}_{k}")
    groups = {r.group_id for r in added}
    assert any("wt" in g for g in groups)
    assert any("rep_000" in g for g in groups)
    assert any("rep_002" in g for g in groups)

    # --- (b) overlap reference == FAMILY consensus for EVERY target; the RMSD
    #         reference (target atoms) DIFFERS across targets ---------------- #
    assert len(seen) >= 4
    for (fp_arg, _atoms, _cids) in seen:
        assert np.allclose(fp_arg, family_consensus), \
            "overlap reference is not the family consensus"
    # target atom sets are per-frame: WT atoms != rep_002 atoms
    atom_sigs = [tuple(round(c, 3) for a in atoms for c in a.coord)
                 for (_fp, atoms, _cids) in seen]
    assert len(set(atom_sigs)) >= 2, \
        "RMSD references did not differ across targets (cross-frame bug)"

    # diagnostics returned cover the classified poses
    assert len(diags) == len(added) or len(diags) >= len(added)

    audit = tmp_path / "interaction_graphs" / "multi_engine_docking.json"
    status_file = tmp_path / "interaction_graphs" / "multi_engine_status.json"
    assert audit.exists() and status_file.exists()
    audit_data = json.loads(audit.read_text())
    assert audit_data["status"]
    assert audit_data["rows"]
    row = audit_data["rows"][0]
    assert {"rank", "score_type", "score_gate_pass", "context_mode",
            "ligand_id", "engine_version", "command_args"} <= set(row)

    # --- (c) select_poses runs on the COMBINED Boltz + docking records ------- #
    sel = select_poses(records, select_z=2.5, outlier_z=4.0,
                       min_decoys_per_group=4, seed=7)
    assert sel.X.shape[0] > 0
    assert sel.X.shape[1] == family_consensus.shape[0]
    # the appended docking rows were counted as multi-engine roles
    assert (sel.n_weak_positive + sel.n_hard_negative) == len(added)
    # the family consensus is unmoved by the docking rows (Boltz-only median)
    boltz_only = np.median(
        np.vstack([r.fingerprint for r in records if not r.role]), axis=0)
    assert np.allclose(sel.consensus, boltz_only)


def test_per_rep_skips_missing_stub_dirs(tmp_path):
    """A rep whose structure dir is missing (Boltz skipped/failed it, stem=None)
    contributes no rows and never crashes the augmentation."""
    n_res = len(SEQ)
    wt_stem = tmp_path / "complexes" / "boltz" / "boltz_results_wt" / \
        "predictions" / "wt_boltz_input"
    _make_stem(wt_stem, n_res,
               [(6.5, 3.0, 4.0), (6.9, 3.2, 4.1), (7.3, 3.4, 4.2)])
    ligand = _ligand()
    ctx = _fake_ctx(tmp_path, ligand)
    from evoliez.adapters.boltz import parse_prediction_dir
    wt_cx = parse_prediction_dir(wt_stem, SEQ, ligand, "boltz2")
    wt_cx.structure.pdb_path = str(wt_stem / "wt_boltz_input_model_0.pdb")

    cfg = InteractionModelConfig(
        multi_engine=True, multi_engine_methods=["gnina"],
        contact_cutoff=6.0, k_nearest_residues=6,
    )
    records = _family_records(wt_cx.structure, wt_cx.ligand.atoms)
    n_boltz = len(records)
    # two reps, BOTH with stem=None -> only the WT target docks
    reps = [SimpleNamespace(sequence=SEQ), SimpleNamespace(sequence=SEQ)]
    diags = _augment_with_docking(
        ctx, cfg, wt_cx, records, [None, None], reps, [], LOG)
    added = records[n_boltz:]
    assert added, "WT target should still dock when reps are missing"
    assert all("wt" in r.group_id for r in added)
    assert diags  # WT poses were classified
