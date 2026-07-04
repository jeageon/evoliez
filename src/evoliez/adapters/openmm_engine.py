"""Molecular-dynamics validation engine (spec section 15).

real: OpenMM. Protein FF (Amber ff14SB), ligand via OpenFF/GAFF
(openmmforcefields), implicit (GBSA) or explicit (TIP3P) solvent, the spec's
restraint schedule, and protocol levels 0-3.
mock: a deterministic synthetic trajectory whose stability scales with how
disruptive the mutation is - lets MD analysis/scoring run with no GPU.

Returns a backend-agnostic :class:`MDResult` consumed by ``md.analysis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from evoliez.adapters.base import is_full_atom_pdb, write_min_pdb
from evoliez.config import Backend, MDConfig
from evoliez.logging_utils import get_logger
from evoliez.types import Complex, LigandAtom
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.seeds import derive_seed

log = get_logger("evoliez.openmm")

# L1/L2 are short screening rungs run across many candidates; cap their step
# count so wall-time stays bounded on the shared box. L3 honours the full
# configured production length. 25_000 steps = 50 ps at a 2 fs timestep.
_LITE_MAX_STEPS = 25_000

# Tiered positional-restraint shells, in nm from the ligand centroid: the
# POCKET backbone is left FREE so pocket_rmsd measures the binding-site
# response, the active-site SHELL is held weakly, distant backbone strongly.
_POCKET_NM = 0.8   # 8 Å
_CONTACT_NM = 0.45  # 4.5 Å heavy-atom ligand contact (occupancy)
_HBOND_NM = 0.35    # 3.5 Å donor-acceptor H-bond proxy (distance-only, no angle)
_SHELL_NM = 1.2    # 12 Å


def _production_nsteps(cfg: "MDConfig") -> tuple:
    """(nsteps, actual_ns) for the production block.

    steps = total simulated time / timestep (1 ns = 1e6 fs). The previous
    formula divided by 1000 twice, running ~1 ps for a "1 ns" request. L1/L2
    are capped at _LITE_MAX_STEPS; floor at 50 steps. actual_ns is derived FROM
    nsteps so the report reflects what ACTUALLY ran, never echoes production_ns.
    """
    nsteps = int(cfg.production_ns * 1e6 / cfg.timestep_fs)
    if cfg.protocol_level < 3:          # L1/L2 are short screening rungs
        nsteps = min(nsteps, _LITE_MAX_STEPS)
    nsteps = max(50, nsteps)
    actual_ns = nsteps * cfg.timestep_fs / 1e6
    return nsteps, actual_ns


def _ca_restraint_tier(distance_nm: "float | None") -> str:
    """Positional-restraint tier for one CA at ``distance_nm`` from the ligand
    centroid (None = apo / no ligand atoms). 'free' = pocket backbone, added
    with NO restraint; 'weak' = active-site shell; 'strong' = distant backbone.
    The pocket being free is the unfreeze-pocket fix - the previous build
    strong-restrained EVERY CA, pinning the pocket so pocket_rmsd was ~0."""
    if distance_nm is None:             # apo / no ligand: hold backbone
        return "strong"
    if distance_nm <= _POCKET_NM:
        return "free"
    return "strong" if distance_nm > _SHELL_NM else "weak"


@dataclass
class MDResult:
    candidate_id: str
    status: str  # "ok" | "unstable" | "failed"
    protocol_level: int
    solvent_mode: str
    simulation_time_ns: float
    minimized_pdb: Optional[str] = None
    trajectory_path: Optional[str] = None
    ligand_rmsd_series: List[float] = field(default_factory=list)
    pocket_rmsd_series: List[float] = field(default_factory=list)
    key_distances: Dict[str, List[float]] = field(default_factory=dict)
    contact_occupancy: Dict[str, float] = field(default_factory=dict)
    hbond_occupancy: float = 0.0
    energy_drift: float = 0.0
    # endpoint binding free energy (kcal/mol) per method, e.g. {"gbsa": -28.4,
    # "pbsa": -24.1}; populated by the Amber tier-3 MM-PB/GBSA, empty otherwise.
    binding_dg: Dict[str, float] = field(default_factory=dict)
    # Catalytic-power (near-attack-conformation) screen, when md.reactive_geometry
    # is enabled and the donor+acceptor were both resolved in the trajectory.
    # ``nac_occupancy`` is the reaction-competent frame fraction; ``nac`` is the
    # full NACResult.to_json() (distances/angles/atoms/note). None / empty when
    # NAC was off, not applicable (mock / no production frames), or the reaction
    # partners were absent. See evoliez.md.nac.
    nac_occupancy: Optional[float] = None
    nac: Dict[str, object] = field(default_factory=dict)
    integration_failed: bool = False
    failure_reason: Optional[str] = None


_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _pdb_one_letter_seq(path: Path) -> str:
    """1-letter sequence from a PDB's CA records (unknown -> 'X')."""
    seq: List[str] = []
    try:
        for line in path.read_text().splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                seq.append(_AA3TO1.get(line[17:20].strip().upper(), "X"))
    except OSError:
        return ""
    return "".join(seq)


_STD_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
    "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ASH", "GLH",
    "ACE", "NME", "HOH", "WAT",
}


def _rdkit_from_het_lines(het_lines: "Sequence[str]", smiles: str):
    """RDKit ligand mol at the pose from ONE set of HETATM(+CONECT) records,
    bond orders from the SMILES template, explicit H with coords. Returns
    ``None`` when the template does not match this atom set (a DIFFERENT
    molecule - the basis for telling NADP apart from a formate co-substrate).
    Raises only on genuinely corrupt input. Pure RDKit, so unit-testable
    WITHOUT the conda-only openff stack (this is the bug-prone part)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if not any(ln.startswith("HETATM") for ln in het_lines):
        raise ValueError("no ligand HETATM records in the predicted PDB")
    rd = Chem.MolFromPDBBlock(
        "\n".join(het_lines) + "\nEND\n",
        removeHs=False, sanitize=False, proximityBonding=False,
    )
    if rd is None:
        raise ValueError("RDKit could not read the ligand PDB block")
    tmpl = Chem.MolFromSmiles(smiles)
    if tmpl is None:
        raise ValueError(f"unparsable ligand SMILES: {smiles!r}")
    # Heavy-atom count gate FIRST: AssignBondOrdersFromTemplate substructure-
    # matches the template, so a SMALLER template (formate) could partial-match
    # a LARGER group (NADP) and silently mis-assign. Requiring equal heavy-atom
    # counts forces a whole-molecule match, so each group binds to its true
    # ligand. (Protonation differences don't change the heavy count.)
    rd_heavy = sum(1 for a in rd.GetAtoms() if a.GetAtomicNum() > 1)
    if rd_heavy != tmpl.GetNumAtoms():       # SMILES heavy-atom count
        return None
    try:
        rd = AllChem.AssignBondOrdersFromTemplate(tmpl, rd)  # orders from SMILES
    except Exception:
        return None                          # template != this molecule
    # Molecule.from_rdkit REQUIRES a SANITIZED mol (valence/aromaticity/ring
    # perception). MolFromPDBBlock used sanitize=False and AddHs leaves the
    # property cache stale, so an unsanitized mol here yields an OpenFF
    # graph that won't round-trip through openmmforcefields' residue matcher
    # -> "No template for residue LIG / did you forget .add_molecules()".
    Chem.SanitizeMol(rd)
    rd = Chem.AddHs(rd, addCoords=True)                   # explicit H + coords
    Chem.SanitizeMol(rd)                                  # AddHs left it stale
    rd.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistryFrom3D(rd)                  # stereo from the pose
    return rd


def _ligand_rdkit_at_pose(pdb_path: Path, smiles: str):
    """RDKit ligand mol AT THE BOLTZ POSE for the SINGLE-ligand path: all
    HETATM records as one molecule. Raises on failure - a mis-built/mis-placed
    ligand must surface as a REAL MD failure, never a silent pass."""
    het = [
        ln for ln in Path(pdb_path).read_text().splitlines()
        if ln.startswith(("HETATM", "CONECT"))
    ]
    rd = _rdkit_from_het_lines(het, smiles)
    if rd is None:
        raise ValueError(
            "ligand SMILES template did not match the PDB HETATM block "
            "(atom count / connectivity mismatch)"
        )
    return rd


def _hetatm_groups(pdb_text: str) -> "List[tuple]":
    """Split a PDB's HETATM records into per-molecule groups keyed by
    (chain, resSeq, iCode, resName), each carrying its own HETATM lines plus the
    CONECT lines whose base atom belongs to the group. This is what lets a
    co-modelled active site (NADP + a formate co-substrate, ions, ...) be read
    as SEPARATE molecules. Pure text. Returns [(key, lines), ...] in
    first-appearance order."""
    order: List[tuple] = []
    groups: Dict[tuple, List[str]] = {}
    serial_key: Dict[int, tuple] = {}
    conects: List[str] = []
    for ln in pdb_text.splitlines():
        if ln.startswith("HETATM"):
            try:
                serial = int(ln[6:11])
            except ValueError:
                continue
            key = (ln[21:22], ln[22:26].strip(), ln[26:27], ln[17:20].strip())
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(ln)
            serial_key[serial] = key
        elif ln.startswith("CONECT"):
            conects.append(ln)
    for c in conects:
        try:
            base = int(c[6:11])
        except ValueError:
            continue
        key = serial_key.get(base)
        if key is not None:
            groups[key].append(c)
    return [(k, groups[k]) for k in order]


def _ligands_at_pose(pdb_path: Path, specs: "Sequence[tuple]") -> "List[tuple]":
    """Place the design ligand + any co-substrates/cofactors at the Boltz pose,
    each as its OWN molecule. ``specs`` = ``[(id, smiles), ...]`` with the design
    ligand FIRST. Groups the PDB HETATM by residue and matches each group to the
    spec whose template fits (heavy-atom count + bond-order assignment - no
    residue-name reliance). Returns ``[(id, rdkit_mol), ...]`` in spec order for
    the ones found; logs every spec not matched and every hetero group left
    over (never silent - a missing reaction partner must be visible)."""
    groups = _hetatm_groups(Path(pdb_path).read_text())
    out: List[tuple] = []
    used = set()
    for sid, smi in specs:
        found = None
        for gi, (_key, lines) in enumerate(groups):
            if gi in used:
                continue
            rd = _rdkit_from_het_lines(lines, smi)
            if rd is not None:
                found, _u = rd, used.add(gi)
                break
        if found is not None:
            out.append((sid, found))
        else:
            log.warning("MD: ligand %s (%s...) not matched to any hetero group "
                        "in %s", sid, smi[:32], Path(pdb_path).name)
    leftover = len(groups) - len(used)
    if leftover:
        log.info("MD: %d hetero group(s) in %s not matched to a configured "
                 "ligand (ions/water/unmodelled)", leftover, Path(pdb_path).name)
    return out


def _offmol_from_rdkit(rd):
    """OpenFF Molecule from a pose RDKit mol, with PDB hierarchy metadata
    cleared. The metadata (residue name "LIG", atom/chain names) makes
    openmmforcefields' GAFFTemplateGenerator FAIL to residue-match the molecule
    against its OWN to_openmm() topology - server-confirmed: a chemically-
    isomorphic SMILES-built molecule parameterizes, the metadata-carrying PDB
    one does NOT ("No template found for residue 0 (LIG) ... missing N H
    atoms"). This was a real, silent s10 blocker (EVERY pose-built ligand fell
    to skipped_parameterization). Clearing per-atom metadata restores the match
    while keeping the pose conformer. OpenFF is conda-only -> server-only line."""
    from openff.toolkit import Molecule

    mol = Molecule.from_rdkit(rd, allow_undefined_stereo=True)
    for atom in mol.atoms:          # drop PDB hierarchy metadata (see above)
        atom.metadata.clear()
    return mol


def _ligand_offmol_at_pose(pdb_path: Path, smiles: str):
    """OpenFF Molecule for the SINGLE design ligand at the Boltz pose. Extracts the
    design ligand from its OWN hetero group (not the merged HETATM block), so a
    co-modelled complex (e.g. NADP + a formate co-substrate) still builds the design
    ligand alone instead of failing the whole-molecule atom-count gate. Falls back to
    the all-HETATM read (single-group complexes) so the verified path is unchanged."""
    matched = _ligands_at_pose(pdb_path, [("design", smiles)])
    if matched:
        return _offmol_from_rdkit(matched[0][1])
    return _offmol_from_rdkit(_ligand_rdkit_at_pose(pdb_path, smiles))


class _FixedChargeError(Exception):
    """A fixed-charge cofactor (e.g. the -3 NADP) could not be charged: a missing/
    unmatched template, or a high-risk cofactor with no template + AM1-BCC forbidden.
    Surfaced as a candidate `failed` with a precise reason - never a silent skip and
    never an on-the-fly sqm that won't converge."""


def _apply_fixed_charges(off_lig, lig) -> bool:
    """Inject fixed partial charges onto the design-ligand OpenFF Molecule so the
    SystemGenerator uses them instead of on-the-fly AM1-BCC. Returns True if charges
    were injected. Policy:
      * ``lig.charges_mol2`` set  -> load the template, graph-match its charges onto
        this pose's ligand, set ``partial_charges`` (skips AM1-BCC). Raises if the
        template can't be found or doesn't match (no silent guess).
      * else high-risk cofactor (``requires_fixed_charge_template``) -> RAISE: a
        config error (needs charges_mol2 + allow_am1bcc:false), not a silent sqm hang.
      * else (small/neutral ligand) -> no-op; the SystemGenerator computes AM1-BCC.
    """
    from evoliez.md import charges as _charges

    mol2 = getattr(lig, "charges_mol2", None)
    if mol2:
        import numpy as np
        from openff.units import unit

        mol2_path = Path(mol2)
        if not mol2_path.is_absolute():
            mol2_path = (Path.cwd() / mol2_path).resolve()
        sdf_path = Path(str(mol2_path)[:-5] + ".ref.sdf") if str(mol2_path).endswith(".mol2") \
            else mol2_path.with_suffix(".ref.sdf")
        if not mol2_path.exists() or not sdf_path.exists():
            raise _FixedChargeError(
                f"{lig.id}: charges_mol2 template not found "
                f"(mol2={mol2_path}, graph-sdf={sdf_path})")
        ref, ref_q = _charges.load_reference(sdf_path, mol2_path)
        q = _charges.transfer_charges(off_lig.to_rdkit(), ref, ref_q,
                                      int(getattr(lig, "formal_charge", 0) or 0))
        off_lig.partial_charges = np.asarray(q) * unit.elementary_charge
        log.info("ligand %s: injected %d fixed charges (sum %+.3f) from %s - "
                 "skipping on-the-fly AM1-BCC", lig.id, len(q), sum(q), mol2_path.name)
        return True
    if _charges.requires_fixed_charge_template(lig):
        raise _FixedChargeError(
            f"{lig.id}: high-risk cofactor (net charge "
            f"{getattr(lig, 'formal_charge', '?')}, {getattr(lig, 'n_heavy', '?')} "
            "heavy atoms) needs a fixed-charge template - set charges_mol2 and "
            "allow_am1bcc:false. Refusing on-the-fly AM1-BCC (sqm will not converge).")
    return False


def _add_cosubstrate_retention_restraint(system, mol_blocks, nac_map,
                                         radius_A: float, k_kcal: float) -> bool:
    """Flat-bottom restraint keeping the co-substrate (the DONOR molecule, e.g. formate)
    centre of mass within ``radius_A`` of the acceptor atom. RETENTION ONLY: it stops a
    free co-substrate diffusing out of the implicit-solvent pocket, but the reactive
    ANGLE is never restrained -- restraining the angle would MANUFACTURE NAC, so the
    energy is a function of the COM<->acceptor DISTANCE alone (flat below r0, harmonic
    above). Returns True if the restraint was added (donor molecule located)."""
    import openmm

    donor = nac_map.get("donor_heavy")
    cosub = next((gidx for _off, gidx in mol_blocks if donor in gidx), None)
    if not cosub or "acceptor" not in nac_map:
        return False
    force = openmm.CustomCentroidBondForce(
        2, "0.5*k*step(d-r0)*(d-r0)^2; d=distance(g1,g2)")
    force.addGlobalParameter("k", float(k_kcal) * 418.4)     # kcal/mol/Å² -> kJ/mol/nm²
    force.addGlobalParameter("r0", float(radius_A) * 0.1)    # Å -> nm
    force.addGroup([int(i) for i in cosub])                  # co-substrate COM
    force.addGroup([int(nac_map["acceptor"])])               # acceptor atom
    force.addBond([0, 1], [])
    system.addForce(force)
    return True


def _apply_nac4_placement(matched, nac_cfg) -> bool:
    """NAC-4: reposition the co-substrate (the reaction donor, e.g. formate) into the
    near-attack geometry off the design ligand's acceptor face, in place on the matched
    RDKit mols. Best-effort -- if the reacting atoms can't be resolved the predictor pose
    is left untouched (the NAC gate then handles it honestly). Returns True if a mol was
    repositioned."""
    from rdkit.Geometry import Point3D

    from evoliez.md.cosubstrate_placement import (place_donor_at_acceptor,
                                                  place_formate_for_nac)
    from evoliez.md.nac import ReactiveSpec, identify_acceptor, identify_donor

    if len(matched) < 2:
        return False
    spec = ReactiveSpec(
        donor_smarts=nac_cfg.donor_smarts, acceptor_smarts=nac_cfg.acceptor_smarts,
        donor_idx=nac_cfg.donor_idx, acceptor_idx=nac_cfg.acceptor_idx,
        transfer_is_h=nac_cfg.transfer_is_h)
    # Which molecule carries the acceptor, which the donor — DON'T assume the design
    # ligand is the acceptor (that's only the FDH layout). For CAR the design ligand
    # 3-HP is the DONOR and the acceptor (ATP alpha-P) is on a cofactor.
    acc_hit = next(((i, m) for i, (_id, m) in enumerate(matched)
                    if identify_acceptor(m, spec) is not None), None)
    don_hit = next(((i, m) for i, (_id, m) in enumerate(matched)
                    if identify_donor(m, spec) is not None), None)
    if acc_hit is None or don_hit is None or acc_hit[0] == don_hit[0]:
        return False
    _, amol = acc_hit
    di, dmol = don_hit
    ring_acceptor = amol.GetAtomWithIdx(identify_acceptor(amol, spec)).IsInRingSize(6)
    try:
        if ring_acceptor:                        # FDH: donor co-substrate off the ring face
            new = place_formate_for_nac(amol, dmol, spec, spec)
        else:                                    # CAR: design-ligand donor in-line to the phosphate
            new = place_donor_at_acceptor(
                amol, dmol, spec, spec,
                nac_distance=min(3.2, getattr(nac_cfg, "distance_max", 3.6) - 0.2))
    except Exception as exc:                     # placement must never break MD
        log.warning("MD NAC-4 placement failed (%s); keeping pose", exc)
        return False
    if new is None:
        return False
    conf = dmol.GetConformer()
    for i, (x, y, z) in enumerate(new):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    log.info("MD NAC-4: placed donor '%s' near-attack (%s acceptor)",
             matched[di][0], "ring" if ring_acceptor else "phosphate")
    return True


_WORKING_PLATFORM = None
_WORKING_PLATFORM_PROBED = False


def _working_openmm_platform():
    """Fastest OpenMM platform whose Context ACTUALLY initialises on this box
    (cached). The conda CUDA build is PTX-broken vs the 12.4 driver, so a raw
    ``mm.Context(system, integrator)`` with no explicit platform picks CUDA by
    speed and throws CUDA_ERROR_UNSUPPORTED_PTX_VERSION with NO fallback (unlike
    ``app.Simulation``, which try-catches each platform in turn). Probe
    fastest->slowest with a 1-particle system and return the first that loads
    (OpenCL on this box); None if even that failed (leave callers on the default)."""
    global _WORKING_PLATFORM, _WORKING_PLATFORM_PROBED
    if _WORKING_PLATFORM_PROBED:
        return _WORKING_PLATFORM
    _WORKING_PLATFORM_PROBED = True
    # If torch is IMPORTED in this process, OpenMM's separately-built CUDA fails to
    # load its force-kernel PTX against torch's CUDA runtime libs (torch-poisoned
    # main process). Verified: merely `import torch` (no .cuda(), is_initialized()
    # still False) is enough to poison it -- torch's libcudart/libnvrtc get loaded
    # and shadow OpenMM's. So `"torch" in sys.modules` is the reliable signal, NOT
    # torch.cuda.is_initialized(). Do NOT trust a CUDA probe here either: a small
    # NonbondedForce probe can FALSELY pass on the poisoned CUDA (simple kernels
    # load) while the real protein system's kernels (GBSA/constraints) then fail.
    # Skip CUDA outright. Spawn fan-out workers don't import torch -> CUDA stays
    # available (faster) there; if a worker ever does, it safely uses OpenCL.
    import sys
    skip_cuda = "torch" in sys.modules
    try:
        import openmm as mm
        from openmm import unit
        # Still probe the remaining platforms with a real force + energy eval so a
        # genuinely-broken platform is skipped, not just assumed to work.
        probe = mm.System()
        for _ in range(4):
            probe.addParticle(12.0)
        nbf = mm.NonbondedForce()
        nbf.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
        for _ in range(4):
            nbf.addParticle(0.0, 0.2, 0.1)
        probe.addForce(nbf)
        pos = [(i * 0.25, 0.0, 0.0) for i in range(4)] * unit.nanometer
        plats = sorted((mm.Platform.getPlatform(i)
                        for i in range(mm.Platform.getNumPlatforms())),
                       key=lambda p: -p.getSpeed())
        for plat in plats:
            if skip_cuda and plat.getName() == "CUDA":
                continue
            try:
                ctx = mm.Context(probe, mm.VerletIntegrator(0.001), plat)
                ctx.setPositions(pos)
                ctx.getState(getEnergy=True).getPotentialEnergy()   # forces kernel load
                _WORKING_PLATFORM = plat
                log.info("OpenMM working platform: %s%s", plat.getName(),
                         " (CUDA skipped: torch owns the CUDA runtime)" if skip_cuda else "")
                break
            except Exception:
                continue
    except Exception:
        _WORKING_PLATFORM = None
    return _WORKING_PLATFORM


def _protein_only_pdbfixed(pdb_path: Path):
    """PDBFixer-repaired PROTEIN-ONLY (topology, positions).

    Uses PDBFixer's own removeHeterogens() to drop the ligand/ions/water
    RELIABLY - the previous residue-name Modeller.delete() let the Boltz
    H-less LIG survive into create_system ("No template for residue LIG").
    Then adds missing terminal/heavy atoms (OXT). The real ligand is added
    back separately from the OpenFF molecule. Returns (None, None) if
    pdbfixer is unavailable (caller degrades honestly). Needs only pdbfixer
    (no openff) so it is unit-testable in .venv-md."""
    try:
        from pdbfixer import PDBFixer
    except ImportError:
        return None, None
    fx = PDBFixer(filename=str(pdb_path))
    fx.findMissingResidues()
    fx.missingResidues = {}                   # don't model unseen loops
    fx.removeHeterogens(False)                # ligand/ions/water OUT
    fx.findMissingAtoms()                     # incl. terminal OXT
    # addMissingAtoms() builds a raw mm.Context to minimise the added atoms with
    # NO explicit platform -> OpenMM picks the PTX-broken CUDA build (fastest) and
    # throws CUDA_ERROR_UNSUPPORTED_PTX_VERSION with no fallback. Force PDBFixer's
    # Context onto the first platform that actually initialises (OpenCL here). The
    # monkeypatch is process-local and undone in finally; the s10 fan-out uses
    # spawn so workers never share it.
    import openmm as _mm
    _plat = _working_openmm_platform()
    if _plat is not None:
        _orig_ctx = _mm.Context

        def _forced_ctx(*a, **k):
            if len(a) == 2 and "platform" not in k:      # PDBFixer's (system, integrator)
                return _orig_ctx(a[0], a[1], _plat)
            return _orig_ctx(*a, **k)

        _mm.Context = _forced_ctx
        try:
            fx.addMissingAtoms()
        finally:
            _mm.Context = _orig_ctx
    else:
        fx.addMissingAtoms()
    return fx.topology, fx.positions


class _LigandParamUnsupported(Exception):
    """No available small-molecule FF can parameterize this ligand (e.g.
    GAFF/AM1-BCC on a large multiply-phosphorylated cofactor like NADP -
    confirmed on the server: ligand-only create_system fails). This is NOT
    a candidate failure: MD-lite is NEUTRALLY skipped and the candidate is
    judged on the other layers (expert: metals/cofactors legitimately
    skip)."""


def _ligand_cache_key(off_lig) -> str:
    """Short, stable key identifying THIS ligand, for the shared FF cache
    filename. AM1-BCC/antechamber charge derivation is keyed on the ligand,
    not the candidate, so the cache file is named by the ligand's isomeric
    SMILES (covers connectivity AND stereo). Changing the ligand changes the
    key, so a shared cache is never silently reused for a different molecule."""
    import hashlib

    try:
        smi = off_lig.to_smiles(isomeric=True, explicit_hydrogens=False)
    except Exception:
        # Fall back to a repr-based key; still ligand-specific, just opaque.
        smi = repr(off_lig)
    return hashlib.sha1(smi.encode("utf-8")).hexdigest()[:16]


def _ligand_system_generator(off_lig, workdir: Path, cache_dir: "Path | None" = None):
    """A SystemGenerator whose small-molecule FF can ACTUALLY parameterize
    this ligand. Probe ligand-only create_system per FF: gaff-2.11, then
    espaloma-0.3.2 if the `espaloma` package is installed (a graph-net FF,
    no antechamber, that handles cofactors GAFF/AM1-BCC cannot). Raise
    _LigandParamUnsupported if none can - decisive, since the probe IS the
    exact ligand-only parameterization that failed for NADP on the server.

    The openmmforcefields ``cache`` (where AM1-BCC charges land) is the slow,
    ligand-identical part - re-deriving it per candidate re-runs antechamber
    ~md.top_candidates times for the SAME cofactor. When ``cache_dir`` is
    given (a SHARED, stable per-run dir), the cache is written there keyed on
    the ligand, so every candidate after the first HITS the cache and skips
    re-parameterization. ``cache_dir`` defaults to ``workdir`` (the legacy
    per-candidate behaviour) for callers that don't supply a shared dir."""
    import openmm.app as app
    from openmmforcefields.generators import SystemGenerator

    cache_root = cache_dir if cache_dir is not None else workdir
    cache_root.mkdir(parents=True, exist_ok=True)
    # Ligand-keyed cache file: shared across candidates (same ligand -> same
    # file -> antechamber runs ONCE per run), invalidated if the ligand changes.
    lig_key = _ligand_cache_key(off_lig)

    ffs = ["gaff-2.11"]
    try:
        import espaloma  # noqa: F401

        ffs.append("espaloma-0.3.2")
    except Exception:
        pass
    last = None
    for ff in ffs:
        try:
            sg = SystemGenerator(
                forcefields=["amber14-all.xml", "implicit/obc2.xml"],
                small_molecule_forcefield=ff,
                molecules=[off_lig],
                cache=str(cache_root / f"ligff_{lig_key}_{ff}.json"),
                forcefield_kwargs={"constraints": app.HBonds},
                nonperiodic_forcefield_kwargs={
                    "nonbondedMethod": app.CutoffNonPeriodic,
                },
            )
            sg.create_system(                       # PROBE: ligand ALONE
                off_lig.to_topology().to_openmm(), molecules=[off_lig]
            )
            log.info("ligand parameterized with %s", ff)
            return sg, ff
        except Exception as exc:
            last = exc
            log.warning(
                "small-molecule FF %s cannot parameterize the ligand: %s",
                ff, str(exc)[:200],
            )
    raise _LigandParamUnsupported(str(last))


def _ligand_system_generator_multi(off_ligs, workdir: Path,
                                   cache_dir: "Path | None" = None):
    """A SystemGenerator for the catalytic-screen path with MULTIPLE small
    molecules (design ligand + co-substrate/cofactor, e.g. NADP + formate).
    Registers every molecule and probes ligand-only create_system for EACH
    (the binding constraint is the hardest, usually the cofactor): the FF must
    handle ALL of them. Same FF ladder + shared ligand-keyed cache as the
    single-ligand path. Separate from ``_ligand_system_generator`` so the
    server-verified single-ligand path is left byte-for-byte unchanged."""
    import hashlib

    import openmm.app as app
    from openmmforcefields.generators import SystemGenerator

    off_ligs = list(off_ligs)
    cache_root = cache_dir if cache_dir is not None else workdir
    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        "|".join(_ligand_cache_key(o) for o in off_ligs).encode("utf-8")
    ).hexdigest()[:16]

    ffs = ["gaff-2.11"]
    try:
        import espaloma  # noqa: F401

        ffs.append("espaloma-0.3.2")
    except Exception:
        pass
    last = None
    for ff in ffs:
        try:
            sg = SystemGenerator(
                forcefields=["amber14-all.xml", "implicit/obc2.xml"],
                small_molecule_forcefield=ff,
                molecules=list(off_ligs),
                cache=str(cache_root / f"ligff_multi_{key}_{ff}.json"),
                forcefield_kwargs={"constraints": app.HBonds},
                nonperiodic_forcefield_kwargs={
                    "nonbondedMethod": app.CutoffNonPeriodic,
                },
            )
            for o in off_ligs:                  # PROBE each ligand ALONE
                sg.create_system(o.to_topology().to_openmm(), molecules=[o])
            log.info("%d ligand(s) parameterized with %s", len(off_ligs), ff)
            return sg, ff
        except Exception as exc:
            last = exc
            log.warning(
                "small-molecule FF %s cannot parameterize the ligand set: %s",
                ff, str(exc)[:200],
            )
    raise _LigandParamUnsupported(str(last))


def run_md(
    cx: Complex,
    candidate_id: str,
    cfg: MDConfig,
    workdir: Path,
    *,
    instability: float,
    catalytic_positions: Sequence[int],
    backend: Backend,
    dry_run: bool = False,
    ligand_cache_dir: "Path | None" = None,
    extra_ligands: "Sequence[tuple] | None" = None,
    fail_loud_on_cpu: bool = False,
) -> MDResult:
    """``ligand_cache_dir``: shared, stable per-run directory for the ligand
    force-field cache. The ligand is identical across all candidates in a run,
    so AM1-BCC charge derivation (the slow part) is cached here ONCE instead of
    per-candidate under ``workdir``. Defaults to ``workdir`` (legacy behaviour)
    when not supplied.

    ``extra_ligands``: ``[(id, smiles), ...]`` co-substrates/cofactors to ALSO
    place in the system (e.g. formate alongside NADP). Used ONLY by the OpenMM
    catalytic-screen path when ``cfg.reactive_geometry.enabled`` - the reaction
    donor (formate hydride) and acceptor (NADP-C4) are then both present so the
    near-attack-conformation occupancy is measurable. The single-ligand binding
    path ignores it, so the verified default MD is unchanged."""
    workdir.mkdir(parents=True, exist_ok=True)
    if backend is Backend.real:
        # tier-3 confirmatory engine: dispatch to the Amber pmemd.cuda backend
        # (md.engine: amber). Lazy import — amber_engine imports from here.
        if getattr(cfg, "engine", "openmm") == "amber":
            from evoliez.adapters.amber_engine import run_md_amber
            try:
                return run_md_amber(
                    cx, candidate_id, cfg, workdir,
                    catalytic_positions=catalytic_positions, dry_run=dry_run,
                    ligand_cache_dir=ligand_cache_dir,
                    extra_ligands=extra_ligands,
                )
            except Exception as exc:
                log.warning("Amber MD failed for %s (%s); recording failure",
                            candidate_id, exc)
                return MDResult(
                    candidate_id=candidate_id, status="failed",
                    protocol_level=cfg.protocol_level, solvent_mode="implicit",
                    simulation_time_ns=0.0, integration_failed=True,
                    failure_reason=str(exc),
                )
        try:
            return _run_real(
                cx, candidate_id, cfg, workdir,
                catalytic_positions=catalytic_positions, dry_run=dry_run,
                ligand_cache_dir=ligand_cache_dir, extra_ligands=extra_ligands,
                fail_loud_on_cpu=fail_loud_on_cpu,
            )
        except Exception as exc:  # spec 23 Risk 4: fail gracefully per candidate
            log.warning("MD failed for %s (%s); recording failure", candidate_id, exc)
            return MDResult(
                candidate_id=candidate_id, status="failed",
                protocol_level=cfg.protocol_level, solvent_mode="implicit",
                simulation_time_ns=0.0, integration_failed=True,
                failure_reason=str(exc),
            )
    return _run_mock(
        cx, candidate_id, cfg, workdir, instability=instability,
        catalytic_positions=catalytic_positions,
    )


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
def _run_mock(
    cx: Complex,
    candidate_id: str,
    cfg: MDConfig,
    workdir: Path,
    *,
    instability: float,
    catalytic_positions: Sequence[int],
) -> MDResult:
    seed = derive_seed(0x0DDB, candidate_id, str(cfg.protocol_level))
    nframes = {0: 1, 1: 20, 2: 50, 3: 100}.get(cfg.protocol_level, 20)
    minpdb = workdir / f"{candidate_id}_minimized.pdb"
    write_min_pdb(minpdb, cx.structure, cx.ligand.atoms)

    base_lig = 0.3 + 3.5 * instability
    base_pkt = 0.2 + 1.8 * instability
    lig_series, pkt_series = [], []
    for i in range(nframes):
        h = derive_seed(seed, str(i))
        wob = (h % 100) / 100.0 * 0.6
        lig_series.append(round(base_lig * (1 + i / max(1, nframes)) + wob, 3))
        pkt_series.append(round(base_pkt + wob * 0.5, 3))

    contacts = {}
    for r in cx.structure.residues[:: max(1, len(cx.structure.residues) // 12 or 1)]:
        occ = max(0.0, 1.0 - instability - (derive_seed(seed, str(r.index)) % 30) / 100.0)
        contacts[f"R{r.index}"] = round(occ, 3)
    hbond = round(max(0.0, 0.8 - instability), 3)
    key_d = {
        f"cat_{p}": [round(3.0 + instability * 3 + (derive_seed(seed, str(p), str(i)) % 50) / 100.0, 3)
                     for i in range(nframes)]
        for p in catalytic_positions
    }
    drift = round(instability * 0.5 + (seed % 20) / 100.0, 3)
    status = "unstable" if (instability > 0.7 or lig_series[-1] > 5.0) else "ok"
    sim_ns = {0: 0.0, 1: cfg.production_ns, 2: cfg.production_ns,
              3: max(cfg.production_ns, 10.0)}.get(cfg.protocol_level, cfg.production_ns)
    return MDResult(
        candidate_id=candidate_id,
        status=status,
        protocol_level=cfg.protocol_level,
        solvent_mode=cfg.solvent,
        simulation_time_ns=sim_ns,
        minimized_pdb=str(minpdb),
        trajectory_path=None,
        ligand_rmsd_series=lig_series,
        pocket_rmsd_series=pkt_series,
        key_distances=key_d,
        contact_occupancy=contacts,
        hbond_occupancy=hbond,
        energy_drift=drift,
    )


# --------------------------------------------------------------------------- #
# Real (OpenMM)
# --------------------------------------------------------------------------- #
def _run_real(
    cx: Complex,
    candidate_id: str,
    cfg: MDConfig,
    workdir: Path,
    *,
    catalytic_positions: Sequence[int],
    dry_run: bool,
    ligand_cache_dir: "Path | None" = None,
    extra_ligands: "Sequence[tuple] | None" = None,
    fail_loud_on_cpu: bool = False,
) -> MDResult:
    if dry_run:
        log.info("[dry-run] would run OpenMM L%d (%s) for %s",
                 cfg.protocol_level, cfg.solvent, candidate_id)
        return _run_mock(cx, candidate_id, cfg, workdir, instability=0.2,
                         catalytic_positions=catalytic_positions)

    import numpy as np
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    # Solvent honesty (paper integrity): this path builds an IMPLICIT GBSA system
    # (implicit/obc2.xml in the SystemGenerator below) — there is NO addSolvent /
    # PME / periodic box yet. So we run AND report implicit regardless of the
    # requested mode, warning if explicit was asked, rather than mislabelling a
    # GBSA run as explicit-solvent. Real explicit-solvent replicas are a
    # final-tier TODO (top candidates), not the per-candidate screening path.
    actual_solvent = "implicit"
    if cfg.solvent == "explicit":
        log.warning(
            "MD solvent='explicit' requested for %s, but the OpenMM path runs "
            "implicit GBSA only (no PME/periodic box); running + reporting "
            "IMPLICIT, not explicit", candidate_id,
        )

    apply_gpu_selection()
    # OpenFF AM1-BCC charging needs antechamber/sqm on PATH; inject
    # EVOLIEZ_AMBERTOOLS_BIN if the launcher didn't (else EVERY ligand silently
    # falls to skipped_parameterization — the real cause of the "s10 skips
    # everything" bug, not the cofactor).
    from evoliez.adapters.amber_engine import ensure_amber_on_path
    ensure_amber_on_path()
    src = getattr(cx.structure, "pdb_path", None)
    if src and Path(src).exists() and is_full_atom_pdb(Path(src)):
        pdb_path = Path(src)                       # real full-atom structure
    else:
        # Our internal ProteinStructure is a CA-only trace and write_min_pdb
        # emits CA-only records; OpenMM cannot build residue templates from
        # that, so real MD genuinely CANNOT run on a mock upstream. Record
        # an HONEST skip DISTINCT from skipped_parameterization (= optional
        # ligand FF missing, legitimately neutral). This must NOT pass - it
        # is the difference between "MD validated" and "MD never ran".
        log.warning(
            "MD needs a full-atom structure but %s is a CA-only trace "
            "(upstream mock); recording skipped_no_full_atom_structure",
            candidate_id,
        )
        return MDResult(
            candidate_id=candidate_id,
            status="skipped_no_full_atom_structure",
            protocol_level=cfg.protocol_level, solvent_mode=actual_solvent,
            simulation_time_ns=0.0,
            failure_reason="MD requires a full-atom protein (got CA-only); "
                           "run s04_complex real or supply a full-atom PDB",
        )
    pdb = app.PDBFile(str(pdb_path))

    # (#2) The full-atom PDB must actually be THIS candidate's structure.
    # _mutant_complex only swaps dataclass residue letters; pdb_path stays
    # the shared WT Boltz file, so MD-ing it for a mutant would silently
    # "validate" WT, not the mutant. If the PDB sequence != the intended
    # (mutant) sequence, honestly skip (analysis.py -> NOT a pass). WT
    # reference (no mutations) matches its own PDB and proceeds.
    want = (cx.structure.sequence or "").upper()
    have = _pdb_one_letter_seq(pdb_path)
    if want and have and want != have:
        nmut = sum(1 for a, b in zip(want, have) if a != b)
        log.warning(
            "MD input for %s is the shared WT structure, not the mutant "
            "(%d residue(s) differ); recording skipped_no_mutant_structure",
            candidate_id, nmut,
        )
        return MDResult(
            candidate_id=candidate_id,
            status="skipped_no_mutant_structure",
            protocol_level=cfg.protocol_level, solvent_mode=actual_solvent,
            simulation_time_ns=0.0,
            failure_reason=(f"{nmut} residue(s) differ: the full-atom PDB is "
                            "WT, not this candidate's mutant - real per-mutant "
                            "MD needs a per-mutant predicted structure"),
        )

    # (#4a) Ligand FF STACK availability only. Absence of the openff /
    # openmmforcefields stack is the legitimately NEUTRAL skip - the
    # candidate is judged on the other layers, not penalised.
    try:
        from openmmforcefields.generators import SystemGenerator
    except Exception as exc:
        log.warning(
            "ligand FF stack unavailable for %s (%s); recording "
            "skipped_parameterization (NOT a candidate failure)",
            candidate_id, exc,
        )
        return MDResult(
            candidate_id=candidate_id, status="skipped_parameterization",
            protocol_level=cfg.protocol_level, solvent_mode=actual_solvent,
            simulation_time_ns=0.0, failure_reason=str(exc),
        )

    # (#4b) Protein + ligand topology assembly + system creation. Standard
    # openmmforcefields recipe: each ligand enters the system from the OpenFF
    # molecule placed at the BOLTZ POSE (PDB+CONECT, bond orders from the
    # SMILES template) - NOT the bond-sparse/H-less PDB HETATM, which can't
    # graph-match GAFF ("No template for residue LIG"). The protein is
    # PDBFixer-repaired (terminal OXT / missing heavy atoms) with the
    # ligand stripped, then recombined. A failure HERE is a REAL MD failure
    # (structure / chemistry / template), surfaced as failed - never a pass.
    #
    # CATALYTIC-SCREEN path (cfg.reactive_geometry.enabled AND extra_ligands):
    # ALSO place the co-substrate(s)/cofactor(s) (e.g. formate beside NADP) as
    # SEPARATE molecules so the reaction donor and acceptor are both present and
    # the near-attack-conformation occupancy is measurable. The DEFAULT binding
    # path (NAC off / no extras) is the verified single-ligand build, untouched.
    nac_cfg = getattr(cfg, "reactive_geometry", None)
    nac_on = bool(nac_cfg and getattr(nac_cfg, "enabled", False))
    _extras = list(extra_ligands or [])
    multi = nac_on and bool(_extras)
    try:
        if multi:
            specs = [(cx.ligand.id or "design", cx.ligand.smiles)] + [
                (str(t[0]), str(t[1])) for t in _extras
            ]
            matched = _ligands_at_pose(pdb_path, specs)
            if not matched or matched[0][0] != specs[0][0]:
                raise ValueError(
                    "design ligand not found among the complex HETATM groups "
                    "(catalytic-screen multi-ligand build)"
                )
            # NAC-4: replace the predictor's (often random) co-substrate pose with the
            # constructed near-attack geometry off the acceptor face, so a mis-placed
            # formate is not wrongly skipped. The MD + retention restraint then test
            # whether the active site MAINTAINS the reactive arrangement.
            if getattr(nac_cfg, "template_cosubstrate_placement", False):
                _apply_nac4_placement(matched, nac_cfg)
            off_ligs = [_offmol_from_rdkit(rd) for _id, rd in matched]
            # Fixed-charge templates BY LIGAND ID: design ligand from cx.ligand, each
            # cofactor from its (id, smiles, charges_mol2, formal_charge, n_heavy) spec.
            # EVERY ligand must get its pre-derived charges injected here, not just the
            # design one -- else the SystemGenerator runs on-the-fly AM1-BCC/sqm on a
            # high-charge cofactor (ATP/NADPH, -4), which never converges, so the
            # cofactor silently drops from the FF set (len 1 not 3) and the WHOLE MD
            # fails. (Bug: the multi-ligand NAC path only charged off_ligs[0].)
            from types import SimpleNamespace
            tpl_by_id = {(cx.ligand.id or "design"): cx.ligand}
            for t in _extras:
                tpl_by_id[str(t[0])] = SimpleNamespace(
                    id=str(t[0]),
                    charges_mol2=(t[2] if len(t) > 2 else None),
                    formal_charge=(t[3] if len(t) > 3 else 0),
                    n_heavy=(t[4] if len(t) > 4 else None),
                )
            for (_mid, _rd), off in zip(matched, off_ligs):
                tpl = tpl_by_id.get(_mid)
                if tpl is not None:
                    _apply_fixed_charges(off, tpl)
        else:
            off_ligs = [_ligand_offmol_at_pose(pdb_path, cx.ligand.smiles)]
            # Fixed-charge cofactor template (design ligand is off_ligs[0]): inject
            # pre-derived charges a ligand AM1-BCC/sqm can't converge (the -3 NADP), or
            # RAISE loudly for a high-risk cofactor with no template.
            _apply_fixed_charges(off_ligs[0], cx.ligand)
    except Exception as exc:
        # Ligand chemistry could not be built from PDB+CONECT+SMILES
        # (rare; verified-correct for NADP locally) - a real structure
        # problem, surfaced as failed.
        log.warning("MD ligand build failed for %s (%s)", candidate_id, exc)
        return MDResult(
            candidate_id=candidate_id, status="failed",
            protocol_level=cfg.protocol_level, solvent_mode=actual_solvent,
            simulation_time_ns=0.0, integration_failed=True,
            failure_reason=f"ligand build: {exc}",
        )

    # Ligand parameterization PROBE -> classify precisely: a ligand the
    # small-molecule FFs can't handle (NADP-class cofactor) is the NEUTRAL
    # skipped_parameterization (candidate judged on the other layers), NOT
    # a hard failure. Only protein/assembly failures below are `failed`.
    try:
        # Ligand FF cache lives in the SHARED per-run dir (ligand-keyed), so
        # AM1-BCC/antechamber charge derivation runs ONCE for the run's
        # (identical) ligand instead of re-running under each candidate's
        # workdir. Falls back to workdir when no shared dir was threaded in.
        if multi:
            system_generator, _ff = _ligand_system_generator_multi(
                off_ligs, workdir, cache_dir=ligand_cache_dir,
            )
        else:
            system_generator, _ff = _ligand_system_generator(
                off_ligs[0], workdir, cache_dir=ligand_cache_dir,
            )
    except _LigandParamUnsupported as exc:
        log.warning(
            "no small-molecule FF can parameterize the ligand(s) for %s "
            "(large/charged cofactor e.g. NADP); recording "
            "skipped_parameterization (NEUTRAL - judged on other layers): "
            "%s", candidate_id, str(exc)[:200],
        )
        return MDResult(
            candidate_id=candidate_id, status="skipped_parameterization",
            protocol_level=cfg.protocol_level, solvent_mode=actual_solvent,
            simulation_time_ns=0.0,
            failure_reason=f"ligand FF unsupported (cofactor): {exc}",
        )

    # mol_blocks: per added ligand, (OpenFF mol, [global topology indices in
    # atom order]) - the RDKit<->trajectory index map the NAC layer needs. The
    # OpenFF round-trip preserves atom order, so the block is exactly the
    # contiguous topology range the molecule occupies.
    mol_blocks: List[tuple] = []
    try:
        topo, posns = _protein_only_pdbfixed(pdb_path)
        if topo is None:                          # pdbfixer absent
            log.warning(
                "pdbfixer not installed - Amber may reject uncapped "
                "termini / leftover heterogens for %s. "
                "`conda install -c conda-forge pdbfixer`", candidate_id,
            )
            topo, posns = pdb.topology, pdb.positions
        modeller = app.Modeller(topo, posns)
        # Invariant: prep MUST be protein-only. A H-less Boltz LIG that
        # survives here is exactly what produced the confusing "No template
        # for residue LIG / missing 57 H" - fail honestly and specifically
        # instead of letting create_system emit that.
        stray = sorted({r.name for r in modeller.topology.residues()
                        if r.name not in _STD_RES})
        if stray:
            raise RuntimeError(
                f"non-protein residues survived structure prep {stray[:5]} "
                "(removeHeterogens unavailable / ineffective); the OpenFF "
                "ligand is added separately so these must not be present"
            )
        # addHydrogens() also builds a raw Context (to optimise H positions) that
        # otherwise picks the PTX-broken CUDA in the torch-poisoned main process;
        # pass the probed working platform (OpenCL here) explicitly.
        modeller.addHydrogens(system_generator.forcefield,
                              platform=_working_openmm_platform())
        # Add each ligand SOLELY from its OpenFF molecule at its Boltz-pose
        # conformer; the only ligands in the system are these, which
        # create_system matches via GAFF. Record each one's topology block.
        for off in off_ligs:
            n0 = modeller.topology.getNumAtoms()
            modeller.add(
                off.to_topology().to_openmm(),
                off.conformers[0].to_openmm(),
            )
            mol_blocks.append(
                (off, list(range(n0, modeller.topology.getNumAtoms())))
            )
        system = system_generator.create_system(
            modeller.topology, molecules=off_ligs
        )
    except Exception as exc:
        import traceback as _tb
        log.warning(
            "MD protein+ligand assembly / system creation failed for "
            "%s (%s: %s)\nTRACEBACK:\n%s", candidate_id,
            type(exc).__name__, exc, _tb.format_exc(),
        )
        return MDResult(
            candidate_id=candidate_id, status="failed",
            protocol_level=cfg.protocol_level, solvent_mode=actual_solvent,
            simulation_time_ns=0.0, integration_failed=True,
            failure_reason=f"protein+ligand assembly / system "
                           f"creation: {exc}",
        )

    # Catalytic-power (NAC) setup is resolved BEFORE minimization so the
    # optional co-substrate retention restraint is active during the first
    # relaxation step. NAC-4 deliberately places formate in a near-attack pose;
    # if the restraint is only added after minimization, that pose can drift
    # before the placement gate ever sees frame 0.
    nac_idx = None
    nac_map = None
    nac_spec = None
    nac_subframes: List = []
    nac_initial_subframe = None
    nac_restrained = False
    if nac_on and mol_blocks:
        try:
            from evoliez.md.nac import (ReactiveSpec,
                                        resolve_reactive_indices)
            if not (nac_cfg.donor_smarts and nac_cfg.acceptor_smarts):
                log.warning("MD NAC enabled for %s but donor/acceptor SMARTS "
                            "are empty; NAC skipped", candidate_id)
            else:
                nac_spec = ReactiveSpec(
                    donor_smarts=nac_cfg.donor_smarts,
                    acceptor_smarts=nac_cfg.acceptor_smarts,
                    donor_idx=nac_cfg.donor_idx,
                    acceptor_idx=nac_cfg.acceptor_idx,
                    transfer_is_h=nac_cfg.transfer_is_h,
                    distance_max=nac_cfg.distance_max,
                    angle_min=nac_cfg.angle_min,
                    label=nac_cfg.label or "reaction",
                    placement_distance_max=getattr(nac_cfg, "placement_distance_max", 4.0),
                    retention_distance_max=getattr(nac_cfg, "retention_distance_max", 6.0),
                    retention_min_fraction=getattr(nac_cfg, "retention_min_fraction", 0.8),
                )
                rd_blocks = [(off.to_rdkit(), blk) for off, blk in mol_blocks]
                nac_map = resolve_reactive_indices(rd_blocks, nac_spec)
                if nac_map:
                    nac_idx = (nac_map["donor_heavy"], nac_map["transfer"],
                               nac_map["acceptor"])
                    # V5-1: for an O->P attack resolve_reactive_indices also returns a
                    # "leaving" atom; append it so the collected subframe is 4 rows and
                    # nac_from_subframes computes the real O_nuc-Palpha-O_leaving angle
                    # (the 3-atom hydride case is unchanged).
                    if "leaving" in nac_map:
                        nac_idx = nac_idx + (nac_map["leaving"],)
                    log.info("MD NAC for %s: donor_heavy=%d transfer=%d "
                             "acceptor=%d%s", candidate_id, nac_map["donor_heavy"],
                             nac_map["transfer"], nac_map["acceptor"],
                             f" leaving={nac_map['leaving']}" if "leaving" in nac_map else "")
                else:
                    log.warning("MD NAC for %s: donor+acceptor not both "
                                "resolved in the system (reaction partners "
                                "absent); NAC skipped", candidate_id)
        except Exception as exc:                  # NAC must never break MD
            log.warning("MD NAC setup failed for %s (%s); NAC skipped",
                        candidate_id, exc)
            nac_idx = None

    # Co-substrate retention restraint (NAC-3): active for minimization AND
    # production. Distance-only flat-bottom; the reactive angle is never
    # restrained, so it cannot manufacture NAC occupancy.
    if nac_idx is not None and getattr(nac_cfg, "restrain_cosubstrate", False):
        try:
            if _add_cosubstrate_retention_restraint(
                    system, mol_blocks, nac_map,
                    nac_cfg.restraint_radius_A, nac_cfg.restraint_k):
                nac_restrained = True
                log.info("MD NAC for %s: co-substrate retention restraint ON "
                         "(flat-bottom r0=%.1f Å, k=%.1f, distance-only; "
                         "active during minimization)",
                         candidate_id, nac_cfg.restraint_radius_A, nac_cfg.restraint_k)
        except Exception as exc:                  # restraint must never break MD
            log.warning("MD NAC restraint failed for %s (%s); unrestrained",
                        candidate_id, exc)

    # The DESIGN ligand (off_ligs[0] / mol_blocks[0]) defines the BINDING
    # metrics; any co-substrate added for the catalytic screen (formate) is NOT
    # counted as "the ligand", so its free diffusion can't inflate ligand_rmsd.
    # For the single-ligand path this is exactly the old residue-name set.
    design_lig_idx = mol_blocks[0][1] if mol_blocks else [
        a.index for a in modeller.topology.atoms()
        if a.residue.name in ("LIG", "UNL", "UNK")
    ]

    # Restraint schedule (spec 15.5): tiered POSITIONAL restraints keyed on
    # distance from the ligand, computed in OpenMM (nm) space so they stay
    # consistent with the pocket used for pocket_rmsd below. Distant backbone
    # is held firmly (a short implicit-solvent run must not drift/unfold), the
    # active-site shell is held weakly, and the POCKET backbone + ligand are
    # left FREE so ligand_rmsd / pocket_rmsd actually measure how the binding
    # site responds to the mutation. (The previous build added a strong
    # restraint to EVERY CA, pinning the pocket and making the core
    # MD-stability signal meaningless.)
    # modeller.positions is a Quantity that, in this OpenMM build, wraps a
    # numpy array (no per-element .x/.y/.z) once an OpenFF conformer has been
    # added - value_in_unit gives a clean (N,3) array either way.
    pos_nm = np.array(modeller.positions.value_in_unit(unit.nanometer))
    if nac_idx is not None:
        nac_initial_subframe = pos_nm[list(nac_idx)] * 10.0
    _lig_idx0 = design_lig_idx          # restrain pocket around the DESIGN ligand
    _ca_atoms = [a for a in modeller.topology.atoms() if a.name == "CA"]
    K_STRONG = (5.0 * unit.kilocalories_per_mole / unit.angstrom**2
                ).value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
    K_WEAK = (0.5 * unit.kilocalories_per_mole / unit.angstrom**2
              ).value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
    restraint = mm.CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addPerParticleParameter("k")
    for p in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)
    lc0 = pos_nm[_lig_idx0].mean(axis=0) if _lig_idx0 else None
    n_strong = n_weak = 0
    for ca in _ca_atoms:
        d = (None if lc0 is None
             else float(np.linalg.norm(pos_nm[ca.index] - lc0)))
        tier = _ca_restraint_tier(d)
        if tier == "free":                  # pocket backbone: FREE to move
            continue
        is_strong = tier == "strong"
        kval = K_STRONG if is_strong else K_WEAK
        x0, y0, z0 = pos_nm[ca.index]
        restraint.addParticle(int(ca.index), [kval, x0, y0, z0])
        n_strong += int(is_strong)
        n_weak += int(not is_strong)
    if restraint.getNumParticles():
        system.addForce(restraint)
    log.info(
        "MD restraints for %s: %d strong + %d weak CA, %d pocket CA free",
        candidate_id, n_strong, n_weak,
        len(_ca_atoms) - n_strong - n_weak,
    )

    integrator = mm.LangevinMiddleIntegrator(
        cfg.temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        cfg.timestep_fs * unit.femtoseconds,
    )
    # Pass the platform that actually initialises IN THIS PROCESS. The main
    # process pre-loads torch, which initialises a mismatched CUDA runtime, so
    # OpenMM's CUDA build throws CUDA_ERROR_UNSUPPORTED_PTX_VERSION and -- contrary
    # to the old assumption -- app.Simulation with no platform does NOT fall
    # through to OpenCL (a raw Context picks the fastest and throws). Probe for the
    # first platform whose FORCE kernels load (OpenCL here) and pass it explicitly.
    # Spawn fan-out workers carry no torch, so the probe returns CUDA there (faster).
    _mplat = _working_openmm_platform()
    sim = (app.Simulation(modeller.topology, system, integrator, _mplat)
           if _mplat is not None
           else app.Simulation(modeller.topology, system, integrator))
    _plat_name = sim.context.getPlatform().getName()
    log.info("MD %s running on the %s platform", candidate_id, _plat_name)
    if fail_loud_on_cpu and _plat_name == "CPU":
        # Safety net (ROADMAP_V2 compute.fail_loud_on_cpu_md): a CPU fallback means no usable
        # GPU CUDA/OpenCL — CPU MD is ~200x slower (weeks for a real shortlist) and would
        # oversubscribe the shared box. Fail THIS candidate fast (the caller's try/except marks
        # it failed) instead of silently grinding on CPU. The s10 fan-out uses spawn so workers
        # re-init OpenCL on their pinned GPU; this catches the case where that still fails.
        raise RuntimeError(
            f"MD {candidate_id}: OpenMM selected the CPU platform (no usable GPU "
            f"CUDA/OpenCL); refusing ~200x-slow CPU MD (compute.fail_loud_on_cpu_md). "
            f"Fix the GPU env, else set the flag False to allow CPU.")
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=cfg.minimize_steps)

    minpdb = workdir / f"{candidate_id}_minimized.pdb"
    with minpdb.open("w") as fh:
        app.PDBFile.writeFile(
            sim.topology, sim.context.getState(getPositions=True).getPositions(), fh
        )

    # ligand-specific + pocket-specific atom indices (was whole-system RMSD).
    # DESIGN ligand only: binding metrics must not include a catalytic-screen
    # co-substrate (formate), whose free diffusion would dominate ligand_rmsd.
    lig_idx = design_lig_idx
    init = np.array([[p.x, p.y, p.z] for p in
                     sim.context.getState(getPositions=True).getPositions()])
    if lig_idx:
        lc = init[lig_idx].mean(axis=0)
        ca_idx = [a.index for a in modeller.topology.atoms()
                  if a.name == "CA"]
        pkt_idx = [i for i in ca_idx
                   if float(((init[i] - lc) ** 2).sum()) ** 0.5 <= _POCKET_NM]
    else:
        pkt_idx = [a.index for a in modeller.topology.atoms()
                   if a.name == "CA"]

    def _rmsd(cur, idx, ref):
        if not idx:
            return 0.0
        d = cur[idx] - ref[idx]
        return float(np.sqrt((d * d).sum(axis=1).mean())) * 10.0  # nm->Å

    # ---- (#2) real-trajectory geometry: catalytic<->ligand distances, pocket
    # contact occupancy, and a distance-only H-bond proxy (protein N/O within
    # _HBOND_NM of a ligand N/O; no angle term -- a screening proxy, not a full
    # H-bond definition). All are ligand-relative (populated only when a ligand
    # is present) and use HEAVY atoms only. catalytic_positions = 1-based PDB
    # residue numbers (match by residue id, ordinal fallback).
    _res_of_atom, _elem = {}, {}
    for _a in modeller.topology.atoms():
        _res_of_atom[_a.index] = _a.residue
        _elem[_a.index] = _a.element.symbol if _a.element is not None else ""
    _heavy = {i for i, s in _elem.items() if s and s != "H"}
    _lig_set = set(lig_idx)
    lig_heavy = np.array([i for i in lig_idx if i in _heavy], dtype=int)
    lig_polar = np.array([i for i in lig_idx if _elem.get(i) in ("N", "O")],
                         dtype=int)
    prot_polar = np.array(
        [i for i, s in _elem.items()
         if s in ("N", "O") and i not in _lig_set
         and _res_of_atom[i].name in _STD_RES
         and _res_of_atom[i].name not in ("HOH", "WAT")], dtype=int)

    def _prot_res_name(name):
        return name in _STD_RES and name not in ("HOH", "WAT")

    def _res_heavy_idx(res):
        return np.array([a.index for a in res.atoms() if a.index in _heavy],
                        dtype=int)

    def _min_dist(cur, a_idx, b_idx):
        d = cur[a_idx][:, None, :] - cur[b_idx][None, :, :]
        return float(np.sqrt((d * d).sum(-1)).min())

    geom_on = lig_heavy.size > 0
    cat_idx, contact_res = {}, {}
    if geom_on:
        _prot_res = [r for r in modeller.topology.residues()
                     if _prot_res_name(r.name)]
        _res_by_num = {}
        for _r in _prot_res:
            try:
                _res_by_num.setdefault(int(_r.id), _r)
            except (TypeError, ValueError):
                pass
        for _p in catalytic_positions:
            _r = _res_by_num.get(int(_p))
            if _r is None and 1 <= int(_p) <= len(_prot_res):
                _r = _prot_res[int(_p) - 1]        # ordinal fallback
            if _r is not None:
                _ai = _res_heavy_idx(_r)
                if _ai.size:
                    cat_idx[int(_p)] = _ai
        for _ca in pkt_idx:
            _r = _res_of_atom.get(_ca)
            if _r is not None and _prot_res_name(_r.name):
                _ai = _res_heavy_idx(_r)
                if _ai.size:
                    contact_res[_r] = _ai
    cat_series = {p: [] for p in cat_idx}
    contact_hits = {r: 0 for r in contact_res}
    hbond_hits = 0

    lig_series: List[float] = []
    pkt_series: List[float] = []
    e_start = e_last = None
    actual_ns = 0.0
    if cfg.protocol_level >= 1:
        sim.context.setVelocitiesToTemperature(cfg.temperature_K * unit.kelvin)
        nsteps, actual_ns = _production_nsteps(cfg)   # report what ACTUALLY ran
        traj = workdir / f"{candidate_id}.dcd"
        sim.reporters.append(app.DCDReporter(str(traj), max(1, nsteps // 50)))
        for blk in range(50):
            sim.step(max(1, nsteps // 50))
            st = sim.context.getState(getPositions=True, getEnergy=True)
            cur = np.array([[p.x, p.y, p.z] for p in st.getPositions()])
            lig_series.append(round(_rmsd(cur, lig_idx, init), 3))
            pkt_series.append(round(_rmsd(cur, pkt_idx, init), 3))
            if nac_idx is not None:               # 3 reacting atoms, nm -> Å
                nac_subframes.append(cur[list(nac_idx)] * 10.0)
            if geom_on:
                for _p, _ai in cat_idx.items():
                    cat_series[_p].append(
                        round(_min_dist(cur, _ai, lig_heavy) * 10.0, 3))
                for _r, _ai in contact_res.items():
                    if _min_dist(cur, _ai, lig_heavy) <= _CONTACT_NM:
                        contact_hits[_r] += 1
                if (prot_polar.size and lig_polar.size
                        and _min_dist(cur, prot_polar, lig_polar) <= _HBOND_NM):
                    hbond_hits += 1
            e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            e_start = e if e_start is None else e_start
            e_last = e
        trajectory_path = str(traj)
    else:
        trajectory_path = None

    if geom_on and lig_series:
        _nfr = len(lig_series)
        key_distances = {f"cat_{p}": s for p, s in cat_series.items() if s}
        contact_occupancy = {f"R{r.id}": round(h / _nfr, 3)
                             for r, h in contact_hits.items()}
        hbond_occupancy = round(hbond_hits / _nfr, 3)
    else:
        key_distances, contact_occupancy, hbond_occupancy = {}, {}, 0.0

    # Catalytic-power (NAC) occupancy from the per-frame reacting-atom geometry.
    nac_occupancy = None
    nac_json: Dict[str, object] = {}
    if nac_idx is not None and nac_subframes:
        from evoliez.md.nac import nac_from_subframes
        nac_res = nac_from_subframes(nac_subframes, nac_spec, atoms=nac_map,
                                     restrained=nac_restrained,
                                     initial_subframe=nac_initial_subframe)
        # GATED occupancy: None unless the co-substrate was actually retained in a
        # reactive arrangement (a diffused / mis-placed formate is NOT "low reactivity").
        nac_occupancy = nac_res.occupancy_or_none
        nac_json = nac_res.to_json()
        log.info("MD NAC for %s: status=%s occupancy=%s retained=%.2f over %d frames "
                 "(d0=%.2f, d_min=%.2f Å, ang_mean=%.1f°)", candidate_id,
                 nac_res.status, nac_res.occupancy_or_none, nac_res.retention_fraction,
                 nac_res.n_frames, nac_res.distance_initial, nac_res.distance_min,
                 nac_res.angle_mean)

    drift = abs((e_last - e_start) / e_start) if e_start else 0.0
    status = "unstable" if (lig_series and lig_series[-1] > 5.0) else "ok"
    return MDResult(
        candidate_id=candidate_id,
        status=status,
        protocol_level=cfg.protocol_level,
        solvent_mode=actual_solvent,
        simulation_time_ns=round(actual_ns, 6),
        minimized_pdb=str(minpdb),
        trajectory_path=trajectory_path,
        ligand_rmsd_series=lig_series or [0.0],
        pocket_rmsd_series=pkt_series or [0.0],
        key_distances=key_distances,
        contact_occupancy=contact_occupancy,
        hbond_occupancy=hbond_occupancy,
        nac_occupancy=nac_occupancy,
        nac=nac_json,
        energy_drift=round(float(drift), 4),
    )
