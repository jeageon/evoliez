"""Multi-engine pose consensus: classify GNINA / DiffDock docking poses by
their relationship to the Boltz family consensus, turning them into *weighted*
augmenting training rows rather than equal-weight examples.

Why not a naive merge? Stacking docking poses as ordinary positives trains an
**averaged docking classifier**, not a family-geometry model. The Boltz
consensus is the positive teacher (the family interaction-geometry prior); a
docking pose only earns a label by how it relates to that teacher:

  * AGREE   — low RMSD-to-consensus-pose AND high contact-fingerprint overlap
              AND no steric clash AND key/catalytic contacts preserved
              -> **weak positive** (low weight: mild corroboration).
  * DISCORDANT but high docking score — scores well yet the contact pattern
              differs from the consensus
              -> **hard negative** (the valuable signal: a plausible-but-wrong
                 pose the model must learn to reject).
  * clash / ligand-inverted / catalytic-contact broken
              -> **excluded** (dropped) or **strong negative**.

Every emitted row is tagged with its ``source`` (gnina|diffdock) and carries a
per-role ``sample_weight`` so :func:`evoliez.ml.pose_selection.select_poses`
honours the role directly instead of re-running its robust-z statistics.

Pure + unit-testable: takes plain data containers, shells out to nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from evoliez.features.geometry import catalytic_distances, dist, rmsd
from evoliez.features.interaction_descriptor import _ITYPES, complex_fingerprint
from evoliez.ml.pose_selection import PoseRecord
from evoliez.types import LigandAtom, ProteinStructure


@dataclass
class DockingPoseInput:
    """One docking pose handed to :func:`classify_docking_poses`.

    Carries everything needed to fingerprint and judge the pose without any
    pipeline/context coupling: the receptor ``structure`` it was docked into,
    the docked ``ligand_atoms``, the engine ``source`` (gnina|diffdock), and the
    raw docking ``score`` (engine-native; gnina affinity is lower=better,
    diffdock confidence is higher=better — see ``score_is_better_low``).

    ``primary_only`` flags a pose that docked the DESIGN ligand alone, with no
    cofactor/substrate context (multi-ligand systems). Such a pose must never be
    allowed to override the cofactor+substrate geometry; it is at minimum tagged
    and is held to a stricter agreement bar.
    """

    source: str
    structure: ProteinStructure
    ligand_atoms: Sequence[LigandAtom]
    score: float = 0.0
    score_is_better_low: bool = True   # gnina affinity: lower is better
    score_gate_pass: bool = True       # required for discordant hard negatives
    primary_only: bool = False         # docked design ligand only (no cofactor)
    candidate_id: str = "wt"
    rank: int = 1
    score_type: str = ""
    cnn_score: Optional[float] = None
    cnn_affinity: Optional[float] = None
    engine_version: str = ""
    command_args: str = ""
    context_mode: str = ""
    ligand_id: str = ""
    ligand_role: str = ""


@dataclass
class PoseClassification:
    """Diagnostic record for one classified docking pose (audit/report)."""

    source: str
    role: str            # weak_positive | hard_negative | strong_negative | excluded
    rmsd_to_consensus: float
    fp_overlap: float
    clash: bool
    key_contacts_ok: bool
    score: float
    sample_weight: float
    score_gate_pass: bool = True
    primary_only: bool = False
    candidate_id: str = ""   # the docking target (WT / "rep_NNN") this pose is for
    rank: int = 1
    score_type: str = ""
    cnn_score: Optional[float] = None
    cnn_affinity: Optional[float] = None
    engine_version: str = ""
    command_args: str = ""
    context_mode: str = ""
    ligand_id: str = ""
    ligand_role: str = ""
    note: str = ""


def _contact_pattern(fp: np.ndarray, *, n_bins: int) -> np.ndarray:
    """The CONTACT-DISCRIMINATIVE slice of a complex_fingerprint: the per-type
    contact composition (interaction-type fractions, MINUS the 'none' channel) +
    the contact fraction + the per-atom min-distance histogram.

    Raw cosine over the WHOLE descriptor is a poor contact-overlap measure: the
    'none' type channel and the large k-shell distance block dominate the norm,
    so even a ligand pulled entirely off the protein scores ~0.94 against a
    well-bound pose. Restricting to the channels that encode WHICH contacts are
    made (and their distances) makes overlap actually track contact agreement
    (self ≈ 1, contactless-vs-bound ≈ 0). Layout (interaction_descriptor):
    ``hist[n_bins] | type_vec[len(_ITYPES)] | summary[4] | kshell[k]``; 'none'
    is the LAST type and ``summary[3]`` is the contact fraction."""
    fp = np.asarray(fp, dtype=float).ravel()
    n_types = len(_ITYPES)
    hist = fp[:n_bins]
    type_vec = fp[n_bins:n_bins + n_types]
    contact_types = type_vec[:-1]                # drop the 'none' channel
    contact_fraction = fp[n_bins + n_types + 3:n_bins + n_types + 4]
    return np.concatenate([contact_types, contact_fraction, hist])


def _fp_overlap_cached(
    a: np.ndarray, cons_pattern: np.ndarray, cons_norm: float, *, n_bins: int = 8
) -> float:
    """:func:`_fp_overlap` with the CONSENSUS side pre-computed. When classifying
    many poses against ONE fixed consensus fingerprint, ``_contact_pattern(b)``
    and its norm are loop-invariant; this variant takes them in so only the
    per-pose ``a`` side is recomputed. Bit-identical to ``_fp_overlap(a, b)``
    when ``cons_pattern == _contact_pattern(b, n_bins=n_bins)`` and
    ``cons_norm == norm(cons_pattern)`` (same operations, same order)."""
    pa = _contact_pattern(a, n_bins=n_bins)
    na = float(np.linalg.norm(pa))
    if na == 0.0 or cons_norm == 0.0:
        return 0.0
    return float(np.clip(np.dot(pa, cons_pattern) / (na * cons_norm), 0.0, 1.0))


def _fp_overlap(a: np.ndarray, b: np.ndarray, *, n_bins: int = 8) -> float:
    """Contact-fingerprint overlap in [0, 1]: cosine over the contact-
    discriminative subspace (see :func:`_contact_pattern`). Non-negative inputs
    -> bounded to [0, 1]; a contactless pose vs a bound one -> ~0."""
    pb = _contact_pattern(b, n_bins=n_bins)
    nb = float(np.linalg.norm(pb))
    return _fp_overlap_cached(a, pb, nb, n_bins=n_bins)


def _has_clash(
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom],
    *,
    clash_dist: float = 1.0,
    max_clashes: int = 1,
) -> bool:
    """Steric clash = a ligand atom interpenetrating the receptor. We have only
    coarse residue reference points (sidechain centroid / CA), NOT full atoms, so
    the hard-shell radius must be TIGHT: a real H-bond/contact already sits ~1.5–3
    Å from a sidechain centroid, so a 2 Å shell would flag legitimate bound poses
    (incl. the Boltz consensus itself). ``clash_dist`` (Å) is the
    interpenetration radius against a residue point; ``max_clashes`` tolerates a
    single borderline atom so a snug-but-valid pose is not falsely excluded."""
    if not ligand_atoms or not structure.residues:
        return False
    pts = [(r.sidechain_centroid or r.ca) for r in structure.residues]
    return _has_clash_pts(
        pts, ligand_atoms, clash_dist=clash_dist, max_clashes=max_clashes
    )


def _has_clash_pts(
    pts: Sequence,
    ligand_atoms: Sequence[LigandAtom],
    *,
    clash_dist: float = 1.0,
    max_clashes: int = 1,
) -> bool:
    """:func:`_has_clash` with the receptor reference points pre-built. When
    classifying many poses that share ONE receptor ``structure``, the
    ``pts = [sidechain_centroid or ca for r in residues]`` list is loop-invariant;
    this variant takes it in so it is built once per target, not once per pose.
    Bit-identical to ``_has_clash(structure, ligand_atoms)`` when ``pts`` is that
    same list (same distance test, same early-exit, same ``max_clashes``)."""
    if not ligand_atoms or not pts:
        return False
    n = 0
    for atom in ligand_atoms:
        for pt in pts:
            if dist(pt, atom.coord) < clash_dist:
                n += 1
                if n > max_clashes:
                    return True
                break  # one clash per ligand atom is enough
    return False


def _key_contacts_preserved(
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom],
    consensus_cat: Dict[str, float],
    key_positions: Sequence[int],
    *,
    tol: float = 2.0,
) -> bool:
    """Catalytic/key contacts preserved iff every key residue's minimum distance
    to the ligand stays within ``tol`` Å of its consensus-pose distance. A key
    residue that was IN contact in the consensus (<= ~5 Å) but is now far (its
    distance grew well beyond tol) means a broken catalytic contact -> reject.
    With no key positions declared this is vacuously True (nothing to preserve)."""
    if not key_positions:
        return True
    pose_cat = catalytic_distances(structure, ligand_atoms, key_positions)
    for k, ref in consensus_cat.items():
        cur = pose_cat.get(k)
        if cur is None:
            continue
        # Only penalise a catalytic contact that was present and is now lost /
        # substantially displaced; a residue far in BOTH poses is not "broken".
        if abs(cur - ref) > tol:
            return False
    return True


def classify_docking_poses(
    consensus_fp: np.ndarray,
    consensus_pose_atoms: Sequence[LigandAtom],
    docking_poses: Sequence[DockingPoseInput],
    *,
    cfg,
    key_positions: Optional[Sequence[int]] = None,
) -> List[PoseRecord]:
    """Classify each docking pose against the Boltz consensus and return the
    KEPT, weighted :class:`PoseRecord`s (``role`` == 'excluded' rows dropped).

    Parameters
    ----------
    consensus_fp:
        The Boltz family consensus interaction fingerprint (the positive-teacher
        contact pattern). Docking-pose fingerprints are compared to THIS.
    consensus_pose_atoms:
        Ligand atoms of the Boltz consensus pose, for RMSD-to-consensus. May be
        empty -> the RMSD gate is skipped (overlap + clash + key contacts still
        apply).
    docking_poses:
        :class:`DockingPoseInput`s (each carries its receptor + docked ligand +
        engine + score).
    cfg:
        ``InteractionModelConfig`` supplying the thresholds
        ``consensus_overlap_rmsd``, ``consensus_overlap_fp``,
        ``hard_negative_weight``, ``weak_positive_weight`` plus
        ``contact_cutoff`` / ``k_nearest_residues`` for the fingerprint.
    key_positions:
        Catalytic / key residue indices whose ligand contact must be preserved
        for a pose to qualify as a weak positive.

    Returns
    -------
    list[PoseRecord]
        One record per KEPT pose, each tagged ``source`` and carrying a per-role
        ``sample_weight`` and ``role`` for :func:`select_poses`. The returned
        list also exposes a ``.diagnostics`` attribute (one
        :class:`PoseClassification` per INPUT pose, including excluded ones) for
        callers that want to log/report the full breakdown.
    """
    key_positions = list(key_positions or [])
    cutoff = getattr(cfg, "contact_cutoff", 6.0)
    k_nearest = getattr(cfg, "k_nearest_residues", 6)
    # complex_fingerprint's distance-histogram bin count (its default; the stage
    # never overrides it). Needed to slice the contact-discriminative subspace.
    n_bins = 8
    rmsd_thr = getattr(cfg, "consensus_overlap_rmsd", 2.0)
    fp_thr = getattr(cfg, "consensus_overlap_fp", 0.6)
    w_hard = getattr(cfg, "hard_negative_weight", 1.0)
    w_weak = getattr(cfg, "weak_positive_weight", 0.3)

    consensus_fp = np.asarray(consensus_fp, dtype=float).ravel()

    # Consensus catalytic-contact baseline (computed once, from any pose's
    # receptor since they share the WT structure + the consensus ligand atoms).
    consensus_cat: Dict[str, float] = {}
    if key_positions and consensus_pose_atoms and docking_poses:
        consensus_cat = catalytic_distances(
            docking_poses[0].structure, consensus_pose_atoms, key_positions
        )

    # --- per-TARGET invariants, hoisted out of the per-POSE loop ------------- #
    # Every pose here shares the SAME consensus (the positive teacher) and the
    # SAME receptor structure, so these three quantities are loop-invariant.
    # Recomputing them per pose was pure redundant work; the per-pose outputs
    # (overlap / rmsd / clash) are byte-for-byte unchanged.
    #   1. consensus-side contact pattern + its norm for the fp-overlap cosine.
    cons_pattern = _contact_pattern(consensus_fp, n_bins=n_bins)
    cons_norm = float(np.linalg.norm(cons_pattern))
    #   2. consensus ligand coordinates for the RMSD gate (guard the empty case
    #      so the rmsd branch is still skipped exactly as before).
    cons_coords = (
        np.array([a.coord for a in consensus_pose_atoms], dtype=float)
        if consensus_pose_atoms
        else None
    )
    #   3. receptor reference points for the clash test, built once from the
    #      shared structure (first non-None pose). When no pose carries a
    #      structure, clash_pts is None and the clash test is skipped — matching
    #      the per-pose ``_has_clash`` guard for that (degenerate) case.
    _clash_struct = next(
        (dp.structure for dp in docking_poses if dp.structure is not None), None
    )
    clash_pts = (
        [(r.sidechain_centroid or r.ca) for r in _clash_struct.residues]
        if _clash_struct is not None
        else None
    )

    kept: List[PoseRecord] = []
    diagnostics: List[PoseClassification] = []
    for k, dp in enumerate(docking_poses):
        fp = complex_fingerprint(
            dp.structure, dp.ligand_atoms, cutoff=cutoff, k_nearest=k_nearest
        )
        overlap = _fp_overlap_cached(fp, cons_pattern, cons_norm, n_bins=n_bins)
        if cons_coords is not None and dp.ligand_atoms:
            r = rmsd(
                cons_coords,
                np.array([a.coord for a in dp.ligand_atoms], dtype=float),
            )
        else:
            r = float("nan")
        clash = _has_clash_pts(clash_pts or [], dp.ligand_atoms)
        key_ok = _key_contacts_preserved(
            dp.structure, dp.ligand_atoms, consensus_cat, key_positions
        )

        role, weight, note = _assign_role(
            rmsd_to_consensus=r, fp_overlap=overlap, clash=clash,
            key_ok=key_ok, primary_only=dp.primary_only,
            score_gate_pass=dp.score_gate_pass,
            rmsd_thr=rmsd_thr, fp_thr=fp_thr, w_weak=w_weak, w_hard=w_hard,
        )
        diag = PoseClassification(
            source=dp.source, role=role,
            rmsd_to_consensus=(round(r, 3) if r == r else float("nan")),
            fp_overlap=round(overlap, 3), clash=clash, key_contacts_ok=key_ok,
            score=round(float(dp.score), 4), sample_weight=round(weight, 3),
            score_gate_pass=bool(dp.score_gate_pass),
            primary_only=dp.primary_only, candidate_id=dp.candidate_id, note=note,
            rank=int(dp.rank), score_type=dp.score_type,
            cnn_score=dp.cnn_score, cnn_affinity=dp.cnn_affinity,
            engine_version=dp.engine_version, command_args=dp.command_args,
            context_mode=dp.context_mode, ligand_id=dp.ligand_id,
            ligand_role=dp.ligand_role,
        )
        diagnostics.append(diag)
        if role == "excluded":
            continue
        rec = PoseRecord(
            group_id=f"dock_{dp.source}_{dp.candidate_id}_{k}",
            fingerprint=fp,
            msa_membership=0.0,             # not a homolog/MSA pose
            identity_to_target=0.0,
            pred_score=round(float(dp.score), 4),
            source=dp.source,
            role=role,
            sample_weight=round(weight, 4),
        )
        kept.append(rec)

    # Drop-in list[PoseRecord] that also carries the per-pose diagnostics (every
    # pose, including excluded ones) so the stage can log/report the breakdown.
    out = _RecordList(kept)
    out.diagnostics = diagnostics
    return out


class _RecordList(list):
    """A list of PoseRecords that also carries the per-pose ``diagnostics``
    (including excluded poses) for logging/reporting. Subclassing list keeps it
    a drop-in ``list[PoseRecord]`` for every caller."""

    diagnostics: List[PoseClassification]


def _assign_role(
    *,
    rmsd_to_consensus: float,
    fp_overlap: float,
    clash: bool,
    key_ok: bool,
    primary_only: bool,
    score_gate_pass: bool,
    rmsd_thr: float,
    fp_thr: float,
    w_weak: float,
    w_hard: float,
):
    """Map the geometric verdict to (role, sample_weight, note). Pure, so the
    decision table is exercised directly in unit tests.

    Decision order (most disqualifying first):
      1. clash OR catalytic contact broken      -> excluded.
      2. AGREE (RMSD ok + overlap >= fp_thr)     -> weak_positive.
         (a primary-only pose must clear a STRICTER overlap bar so it can never
          stand in for the full cofactor+substrate geometry.)
      3. DISCORDANT (overlap < fp_thr) AND score-gated -> hard_negative.
         Low-score discordant docking poses are excluded; they are easy
         engine failures, not useful hard negatives.
    """
    rmsd_ok = (rmsd_to_consensus != rmsd_to_consensus) or (
        rmsd_to_consensus <= rmsd_thr  # NaN (no consensus atoms) -> gate off
    )
    if clash:
        return "excluded", 0.0, "steric clash"
    if not key_ok:
        # a broken catalytic contact is the strong-negative case: keep it as a
        # negative the model must reject rather than silently dropping the signal.
        return "strong_negative", w_hard, "catalytic contact broken"

    # primary-ligand-only poses are held to a tighter agreement bar (they lack
    # the cofactor/substrate context, so a "high" overlap is less trustworthy).
    eff_fp_thr = fp_thr if not primary_only else min(0.95, fp_thr + 0.15)

    if rmsd_ok and fp_overlap >= eff_fp_thr:
        note = "agrees with consensus" + (" (primary-only, tagged)"
                                          if primary_only else "")
        return "weak_positive", w_weak, note
    if not score_gate_pass:
        return "excluded", 0.0, "discordant but low-score / low-rank"
    # scores well in the engine but the contact pattern disagrees -> the
    # valuable hard negative.
    return "hard_negative", w_hard, "discordant contact pattern"
