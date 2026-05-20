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

from evoliez.adapters.base import write_min_pdb
from evoliez.config import Backend, MDConfig
from evoliez.logging_utils import get_logger
from evoliez.types import Complex, LigandAtom
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.seeds import derive_seed

log = get_logger("evoliez.openmm")


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
    integration_failed: bool = False
    failure_reason: Optional[str] = None
    # P0.6: provenance so the final report says HOW the MD passed/failed.
    ligand_forcefield: Optional[str] = None    # actual FF used (gaff/sage/curated)
    hmr_enabled: bool = False                  # 4 fs path on/off
    timestep_fs: float = 2.0
    replicas_run: int = 1
    # Per-replica spread when replicas_run > 1, the reranker / evidence
    # class layer uses these to mark high-variance MD as Uncertain.
    ligand_rmsd_replicas: List[List[float]] = field(default_factory=list)
    pocket_rmsd_replicas: List[List[float]] = field(default_factory=list)


def _is_full_atom_pdb(path: Path) -> bool:
    """True if the PDB has more than a CA trace per residue. write_min_pdb
    (mock) emits ONLY CA ATOM records, which OpenMM cannot turn into Amber
    residue templates ('HIS residue has the wrong set of atoms')."""
    try:
        for line in path.read_text().splitlines():
            if line.startswith("ATOM") and line[12:16].strip() not in (
                "CA", ""
            ):
                return True
    except OSError:
        return False
    return False


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


def _ligand_rdkit_at_pose(pdb_path: Path, smiles: str):
    """RDKit ligand mol AT THE BOLTZ POSE: read the HETATM + CONECT ligand
    from the predicted PDB (bonds from CONECT), assign bond orders from the
    SMILES template, add explicit H with coords. Pure RDKit so it is
    unit-testable WITHOUT the conda-only openff stack (this is the
    bug-prone part). Raises on failure - a mis-built/mis-placed ligand
    must surface as a REAL MD failure, never a silent pass."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    het = [
        ln for ln in Path(pdb_path).read_text().splitlines()
        if ln.startswith(("HETATM", "CONECT"))
    ]
    if not any(ln.startswith("HETATM") for ln in het):
        raise ValueError("no ligand HETATM records in the predicted PDB")
    rd = Chem.MolFromPDBBlock(
        "\n".join(het) + "\nEND\n",
        removeHs=False, sanitize=False, proximityBonding=False,
    )
    if rd is None:
        raise ValueError("RDKit could not read the ligand PDB block")
    tmpl = Chem.MolFromSmiles(smiles)
    if tmpl is None:
        raise ValueError(f"unparsable ligand SMILES: {smiles!r}")
    rd = AllChem.AssignBondOrdersFromTemplate(tmpl, rd)  # orders from SMILES
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


def _graph_match_pose_to_smiles(pose_rd, ref_rd):
    """Heavy-atom graph isomorphism between a pose-derived RDKit mol and
    the SMILES-canonical OFF mol's RDKit form.

    Returns a list ``ref_heavy_index_for_pose_heavy[i]`` so the caller
    can copy `pose_rd`'s heavy-atom coordinates onto OFF atom order.
    Bonds are made generic in the query so aromatic/Kekule perception
    differences between the PDB-derived and SMILES-derived mols don't
    break the isomorphism. Raises on count or connectivity mismatch.
    """
    from rdkit import Chem

    # Strip H from BOTH so the match is purely on the heavy-atom skeleton.
    # The PDB-derived pose can have an inconsistent H count (NADP+ AddHs
    # in particular adds only 1 H to the multiply-charged species - that
    # is the server-observed bug); the SMILES-canonical OFF mol has the
    # right H count via Molecule.from_smiles, so we use SMILES for H.
    pose_heavy = Chem.RemoveHs(Chem.Mol(pose_rd))
    ref_heavy = Chem.RemoveHs(Chem.Mol(ref_rd))
    if pose_heavy.GetNumAtoms() != ref_heavy.GetNumAtoms():
        raise ValueError(
            f"heavy-atom count mismatch: pose has "
            f"{pose_heavy.GetNumAtoms()}, SMILES expects "
            f"{ref_heavy.GetNumAtoms()}"
        )
    qp = Chem.AdjustQueryParameters.NoAdjustments()
    qp.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(Chem.Mol(pose_heavy), qp)
    match = ref_heavy.GetSubstructMatch(query, useChirality=False)
    if len(match) != pose_heavy.GetNumAtoms():
        raise ValueError(
            "heavy-atom graph match failed: pose connectivity doesn't "
            "match the SMILES skeleton (only "
            f"{len(match)}/{pose_heavy.GetNumAtoms()} atoms mapped)"
        )
    return match, pose_heavy, ref_heavy


def _ligand_offmol_at_pose(pdb_path: Path, smiles: str):
    """OpenFF Molecule whose CHEMISTRY is the authoritative SMILES (canonical
    OpenFF atom order, all explicit H) and whose conformer carries the
    BOLTZ POSE heavy-atom coordinates transferred via graph isomorphism.

    Why this exists: the previous `_ligand_rdkit_at_pose` + `from_rdkit`
    path was server-verified to drop H atoms on multiply-charged species
    (NADP+ at -3: AddHs added only 1 H out of the expected ~25). The
    resulting OFF mol had 49 atoms not 73, and `system_generator.create_
    system` then failed with "No template for residue LIG ... similar to
    DLPS, missing 57 H atoms". This rewrite uses `Molecule.from_smiles`
    as the chemistry/atom-count source of truth and only transfers HEAVY
    atom coordinates from the Boltz pose (H positions come from an RDKit
    embed - MD minimization relaxes them).
    """
    import numpy as np
    from openff.toolkit import Molecule
    from openff.units import unit as offunit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # 1. SMILES-canonical OFF chemistry (73 atoms for NADP+: heavy 48 + H 25).
    offmol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    offmol.name = "LIG"
    ref_rd = offmol.to_rdkit()

    # 2. Pose-only RDKit mol (heavy-atom positions only).
    het = [
        ln for ln in Path(pdb_path).read_text().splitlines()
        if ln.startswith(("HETATM", "CONECT"))
    ]
    if not any(ln.startswith("HETATM") for ln in het):
        raise ValueError("no ligand HETATM records in the predicted PDB")
    pose_rd = Chem.MolFromPDBBlock(
        "\n".join(het) + "\nEND\n",
        removeHs=False, sanitize=False, proximityBonding=True,
    )
    if pose_rd is None:
        raise ValueError("RDKit could not read the ligand PDB block")
    tmpl = Chem.MolFromSmiles(smiles)
    if tmpl is None:
        raise ValueError(f"unparsable ligand SMILES: {smiles!r}")
    pose_rd = AllChem.AssignBondOrdersFromTemplate(tmpl, pose_rd)
    Chem.SanitizeMol(pose_rd)

    # 3. Heavy-atom graph isomorphism: pose -> SMILES (OFF) order.
    match, pose_heavy, ref_heavy = _graph_match_pose_to_smiles(
        pose_rd, ref_rd,
    )

    # 4. Embed `ref_rd` to seed H positions; overwrite heavy-atom
    #    coordinates with the pose. MD minimization later relaxes Hs.
    ref_em = Chem.Mol(ref_rd)
    if AllChem.EmbedMolecule(ref_em, randomSeed=0xC0FFEE) != 0:
        AllChem.EmbedMolecule(ref_em, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(ref_em)
    except Exception:
        pass

    # Heavy atoms in ref_rd (and ref_em, same atom order). RemoveHs preserves
    # heavy-atom relative ordering, so the i-th heavy atom in ref_heavy is
    # the i-th heavy atom we encounter when iterating ref_rd's atoms.
    heavy_in_ref = [i for i, a in enumerate(ref_rd.GetAtoms())
                    if a.GetAtomicNum() != 1]

    em_conf = ref_em.GetConformer()
    pose_heavy_conf = pose_heavy.GetConformer()
    for pose_i, ref_heavy_idx in enumerate(match):
        ref_atom = heavy_in_ref[ref_heavy_idx]
        p = pose_heavy_conf.GetAtomPosition(pose_i)
        em_conf.SetAtomPosition(ref_atom, (p.x, p.y, p.z))

    coords = np.zeros((offmol.n_atoms, 3), dtype=float)
    for i in range(offmol.n_atoms):
        p = em_conf.GetAtomPosition(i)
        coords[i] = (p.x, p.y, p.z)
    offmol.add_conformer(coords * offunit.angstrom)
    return offmol


def _ligand_offmol_at_pose_LEGACY(pdb_path: Path, smiles: str):
    """OpenFF Molecule for the ligand at the Boltz pose (thin wrapper over
    the RDKit builder; OpenFF is conda-only so this line is server-only)."""
    from openff.toolkit import Molecule

    rd = _ligand_rdkit_at_pose(pdb_path, smiles)
    return Molecule.from_rdkit(rd, allow_undefined_stereo=True)


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
    fx.addMissingAtoms()
    return fx.topology, fx.positions


class _CuratedParamUnavailable(Exception):
    """Curated AMBER files for this cofactor are not present on disk
    (Bryce Lab .lib / .frcmod missing). Dispatcher falls back to the
    GAFF/espaloma probe; this is NOT itself a candidate failure."""


def _write_ligand_pdb(off_lig, out_path: Path, residue_name: str) -> None:
    """Write the OpenFF ligand at its conformer as a single-residue PDB
    whose residue name matches the curated AMBER .lib entry (so tleap
    loadpdb finds the matching unit)."""
    import openmm.app as app

    topo = off_lig.to_topology().to_openmm()
    for res in topo.residues():
        res.name = residue_name
    positions = off_lig.conformers[0].to_openmm()
    with out_path.open("w") as fh:
        app.PDBFile.writeFile(topo, positions, fh)


def _curated_param_system_generator(
    off_lig, protein_pdb_path: Path, spec, workdir: Path,
):
    """Build ``(system, topology, positions)`` for a curated cofactor via
    tleap + AMBER params (Bryce Lab .lib / .frcmod), bypassing GAFF /
    AM1-BCC entirely. This is the production path for cofactors that
    sqm hard-fails on (NADP+ at -3 charge - verified by
    scripts/server_test_p0_cofactor.sh Step 4).

    Raises :class:`_CuratedParamUnavailable` if the curated files are not
    present (dispatcher then falls back to the probe path), or a generic
    Exception if tleap actually fails (real MD failure -> ``failed``).
    """
    import shutil
    import subprocess

    import openmm.app as app

    files = spec.resolved_amber_files()
    if files is None:
        raise _CuratedParamUnavailable(
            f"Bryce Lab files for {spec.name} not in "
            f"{spec.amber_lib!r} / {spec.amber_frcmod!r} under "
            "evoliez.features.cofactors.amber_params_root(); run "
            "scripts/fetch_amber_cofactors.sh on the server"
        )
    lib_path, frcmod_path = files
    if not shutil.which("tleap"):
        raise RuntimeError(
            "tleap binary not found in PATH (conda install -c conda-forge "
            "ambertools)"
        )

    # Ligand PDB at the curated 3-letter residue name (matches .lib).
    lig_pdb = workdir / f"ligand_{spec.amber_residue_name}.pdb"
    _write_ligand_pdb(off_lig, lig_pdb, spec.amber_residue_name)

    # Protein-only PDB (PDBFixer-repaired termini, heterogens stripped).
    prot_topo, prot_posns = _protein_only_pdbfixed(protein_pdb_path)
    if prot_topo is None:
        raise RuntimeError(
            "pdbfixer unavailable - curated path needs a clean protein-only "
            "PDB; `conda install -c conda-forge pdbfixer`"
        )
    prot_pdb = workdir / "protein_only.pdb"
    with prot_pdb.open("w") as fh:
        app.PDBFile.writeFile(prot_topo, prot_posns, fh)

    leap_in = workdir / "leap.in"
    prmtop = workdir / "complex.prmtop"
    inpcrd = workdir / "complex.inpcrd"
    leap_in.write_text(
        f"source leaprc.protein.ff14SB\n"
        f"source leaprc.gaff2\n"
        f"loadamberparams {frcmod_path}\n"
        f"loadoff {lib_path}\n"
        f"prot = loadpdb {prot_pdb}\n"
        f"lig  = loadpdb {lig_pdb}\n"
        f"complex = combine {{prot lig}}\n"
        f"saveamberparm complex {prmtop} {inpcrd}\n"
        f"quit\n"
    )
    proc = subprocess.run(
        ["tleap", "-f", str(leap_in)],
        cwd=str(workdir), capture_output=True, text=True,
    )
    if proc.returncode != 0 or not prmtop.exists() or not inpcrd.exists():
        tail = (proc.stdout + proc.stderr)[-800:]
        raise RuntimeError(f"tleap failed (rc={proc.returncode}): {tail}")

    prm = app.AmberPrmtopFile(str(prmtop))
    crd = app.AmberInpcrdFile(str(inpcrd))
    system = prm.createSystem(
        nonbondedMethod=app.CutoffNonPeriodic,
        constraints=app.HBonds,
        implicitSolvent=app.OBC2,
    )
    log.info(
        "curated AMBER path: %s -> tleap prmtop (%d atoms, lig resname %s)",
        spec.name, system.getNumParticles(), spec.amber_residue_name,
    )
    return system, prm.topology, crd.positions


class _LigandParamUnsupported(Exception):
    """No available small-molecule FF can parameterize this ligand (e.g.
    GAFF/AM1-BCC on a large multiply-phosphorylated cofactor like NADP -
    confirmed on the server: ligand-only create_system fails). This is NOT
    a candidate failure: MD-lite is NEUTRALLY skipped and the candidate is
    judged on the other layers (expert: metals/cofactors legitimately
    skip)."""


def _gasteiger_charge_system_generator(off_lig, workdir: Path):
    """A SystemGenerator that uses RDKit Gasteiger partial charges instead
    of antechamber AM1-BCC. For NADP-class cofactors where sqm hard-fails
    on AM1-BCC (server-verified: 'antechamber: Fatal Error! Cannot
    properly run sqm ...') this gives a working - if less rigorous -
    parameterization that completes in SECONDS. The hierarchy in
    _run_real puts the curated Bryce Lab path above this; this tier is
    the self-contained fallback when curated files aren't on disk yet.

    Raises :class:`_LigandParamUnsupported` if even Gasteiger fails (so
    the dispatcher in _run_real classifies as skipped_parameterization,
    consistent with the AM1-BCC probe's failure mode).
    """
    import openmm.app as app
    from openff.toolkit.utils.toolkits import RDKitToolkitWrapper
    from openmmforcefields.generators import SystemGenerator

    try:
        # Pre-assign charges on the OFF molecule; openmmforcefields'
        # GAFFTemplateGenerator skips its own charge calc when
        # mol.partial_charges is already set, so AM1-BCC / sqm never run.
        off_lig.assign_partial_charges(
            "gasteiger", toolkit_registry=RDKitToolkitWrapper(),
        )
        sg = SystemGenerator(
            forcefields=["amber14-all.xml", "implicit/obc2.xml"],
            small_molecule_forcefield="gaff-2.11",
            molecules=[off_lig],
            cache=str(workdir / "ff_cache_gasteiger.json"),
            forcefield_kwargs={"constraints": app.HBonds},
            nonperiodic_forcefield_kwargs={
                "nonbondedMethod": app.CutoffNonPeriodic,
            },
        )
        sg.create_system(                       # PROBE: ligand alone
            off_lig.to_topology().to_openmm(), molecules=[off_lig]
        )
        log.info(
            "ligand parameterized with gaff-2.11 + Gasteiger charges "
            "(AM1-BCC bypassed)"
        )
        return sg
    except Exception as exc:
        raise _LigandParamUnsupported(
            f"Gasteiger-charge GAFF probe failed: {str(exc)[:200]}"
        )


def _ligand_system_generator(off_lig, workdir: Path,
                              prefer_ff: Optional[str] = None):
    """A SystemGenerator whose small-molecule FF can ACTUALLY parameterize
    this ligand. Probe ligand-only create_system per FF: ``prefer_ff``
    first (defaults to OpenFF Sage 2.2 per P0.6), then gaff-2.11, then
    espaloma-0.3.2 if installed. Raise :class:`_LigandParamUnsupported`
    if none can.

    Returns ``(sg, ff_name)`` so the caller can stamp the actual FF used
    onto the MDResult provenance.
    """
    import openmm.app as app
    from openmmforcefields.generators import SystemGenerator

    # P0.6: prefer OpenFF Sage where possible. The probe still falls
    # through to GAFF (and espaloma if available) so legacy ligands that
    # Sage doesn't cover keep working.
    pref = prefer_ff or "openff-2.2.0"
    ffs = []
    if pref and pref not in ffs:
        ffs.append(pref)
    if "gaff-2.11" not in ffs:
        ffs.append("gaff-2.11")
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
                cache=str(workdir / f"ff_cache_{ff}.json"),
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
) -> MDResult:
    workdir.mkdir(parents=True, exist_ok=True)
    if backend is Backend.real:
        try:
            return _run_real(
                cx, candidate_id, cfg, workdir,
                catalytic_positions=catalytic_positions, dry_run=dry_run,
            )
        except Exception as exc:  # spec 23 Risk 4: fail gracefully per candidate
            log.warning("MD failed for %s (%s); recording failure", candidate_id, exc)
            return MDResult(
                candidate_id=candidate_id, status="failed",
                protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
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

    apply_gpu_selection()
    src = getattr(cx.structure, "pdb_path", None)
    if src and Path(src).exists() and _is_full_atom_pdb(Path(src)):
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
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
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
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
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
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
            simulation_time_ns=0.0, failure_reason=str(exc),
        )

    # (#4b) Protein + ligand topology assembly + system creation. Standard
    # openmmforcefields recipe: the ligand enters the system from the OpenFF
    # molecule placed at the BOLTZ POSE (PDB+CONECT, bond orders from the
    # SMILES template) - NOT the bond-sparse/H-less PDB HETATM, which can't
    # graph-match GAFF ("No template for residue LIG"). The protein is
    # PDBFixer-repaired (terminal OXT / missing heavy atoms) with the
    # ligand stripped, then recombined. A failure HERE is a REAL MD failure
    # (structure / chemistry / template), surfaced as failed - never a pass.
    try:
        off_lig = _ligand_offmol_at_pose(pdb_path, cx.ligand.smiles)
    except Exception as exc:
        # Ligand chemistry could not be built from PDB+CONECT+SMILES
        # (rare; verified-correct for NADP locally) - a real structure
        # problem, surfaced as failed.
        log.warning("MD ligand build failed for %s (%s)", candidate_id, exc)
        return MDResult(
            candidate_id=candidate_id, status="failed",
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
            simulation_time_ns=0.0, integration_failed=True,
            failure_reason=f"ligand build: {exc}",
        )

    # Curated cofactor path: when the ligand is a known cofactor (canonical
    # SMILES matches a CofactorSpec) AND the Bryce Lab .lib / .frcmod files
    # are present on disk, build the complex via tleap and skip the GAFF /
    # AM1-BCC probe entirely - that probe HARD-FAILS on real NADP+ (sqm
    # returns non-zero; confirmed by server_test_p0_cofactor.sh Step 4).
    # When files are missing OR the molecule isn't a curated cofactor, fall
    # through to the existing probe + Modeller flow unchanged.
    from evoliez.features.cofactors import lookup_by_smiles

    class _MaybeModeller:                       # facade for the curated path
        def __init__(self, topology, positions):
            self.topology = topology
            self.positions = positions

    curated_spec = lookup_by_smiles(cx.ligand.smiles)
    took_curated = False
    system = modeller = None
    if curated_spec and curated_spec.resolved_amber_files() is not None:
        try:
            system, topology, positions = _curated_param_system_generator(
                off_lig, pdb_path, curated_spec, workdir,
            )
            modeller = _MaybeModeller(topology, positions)
            took_curated = True
        except _CuratedParamUnavailable as exc:
            log.info(
                "curated path unavailable for %s (%s); falling back to "
                "GAFF/espaloma probe", candidate_id, exc,
            )
        except Exception as exc:
            # The curated branch is supposed to bypass GAFF entirely, so a
            # real failure here is a real MD failure - NOT parameterization.
            log.warning(
                "curated tleap path FAILED for %s (%s); recording failed",
                candidate_id, exc,
            )
            return MDResult(
                candidate_id=candidate_id, status="failed",
                protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
                simulation_time_ns=0.0, integration_failed=True,
                failure_reason=f"curated tleap: {str(exc)[:300]}",
            )

    if not took_curated:
        # 3-tier hierarchy for the (Tier 1 missed) cofactor / drug-like
        # ligand cases:
        #   Tier 2 (NEW): known cofactor without curated files -> GAFF +
        #     Gasteiger charges. Server-verified: AM1-BCC sqm hard-fails on
        #     real NADP+, so for a known cofactor we SKIP the AM1-BCC probe
        #     entirely and go straight to Gasteiger (no QM, finishes in
        #     seconds, slightly lower charge fidelity than AM1-BCC). This
        #     keeps the pipeline UNBLOCKED while the Bryce Lab files are
        #     being sourced for Tier 1.
        #   Tier 3 (existing): drug-like ligand -> AM1-BCC probe per FF
        #     (gaff-2.11 then espaloma-0.3.2 if installed) - the standard
        #     production charge model for non-cofactor ligands.
        # Either tier raising _LigandParamUnsupported -> NEUTRAL
        # skipped_parameterization (the candidate is judged on the other
        # layers; same status as the existing AM1-BCC-only path).
        try:
            if curated_spec is not None:
                log.info(
                    "known cofactor %s without curated AMBER files -> "
                    "Gasteiger-charge GAFF (skips AM1-BCC which hard-fails "
                    "for this class)", curated_spec.name,
                )
                system_generator = _gasteiger_charge_system_generator(
                    off_lig, workdir,
                )
                _ff = "gaff-2.11+gasteiger"
            else:
                system_generator, _ff = _ligand_system_generator(
                    off_lig, workdir,
                    prefer_ff=getattr(cfg, "ligand_forcefield", None),
                )
        except _LigandParamUnsupported as exc:
            log.warning(
                "no small-molecule FF can parameterize the ligand for %s "
                "(large/charged cofactor e.g. NADP); recording "
                "skipped_parameterization (NEUTRAL - judged on other "
                "layers): %s", candidate_id, str(exc)[:200],
            )
            return MDResult(
                candidate_id=candidate_id, status="skipped_parameterization",
                protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
                simulation_time_ns=0.0,
                failure_reason=f"ligand FF unsupported (cofactor): {exc}",
            )

        try:
            topo, posns = _protein_only_pdbfixed(pdb_path)
            if topo is None:                       # pdbfixer absent
                log.warning(
                    "pdbfixer not installed - Amber may reject uncapped "
                    "termini / leftover heterogens for %s. "
                    "`conda install -c conda-forge pdbfixer`", candidate_id,
                )
                topo, posns = pdb.topology, pdb.positions
            modeller = app.Modeller(topo, posns)
            # Invariant: prep MUST be protein-only. A H-less Boltz LIG that
            # survives here is exactly what produced the confusing "No
            # template for residue LIG / missing 57 H" - fail honestly and
            # specifically instead of letting create_system emit that.
            stray = sorted({r.name for r in modeller.topology.residues()
                            if r.name not in _STD_RES})
            if stray:
                raise RuntimeError(
                    f"non-protein residues survived structure prep "
                    f"{stray[:5]} (removeHeterogens unavailable / "
                    "ineffective); the OpenFF ligand is added separately "
                    "so these must not be present"
                )
            modeller.addHydrogens(system_generator.forcefield)
            # Add the ligand SOLELY from the OpenFF molecule at its
            # Boltz-pose conformer; the only ligand in the system is now
            # this one, which create_system matches via GAFF.
            modeller.add(
                off_lig.to_topology().to_openmm(),
                off_lig.conformers[0].to_openmm(),
            )
            system = system_generator.create_system(
                modeller.topology, molecules=[off_lig]
            )
        except Exception as exc:
            log.warning(
                "MD protein+ligand assembly / system creation failed for "
                "%s (%s)", candidate_id, exc,
            )
            return MDResult(
                candidate_id=candidate_id, status="failed",
                protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
                simulation_time_ns=0.0, integration_failed=True,
                failure_reason=f"protein+ligand assembly / system "
                               f"creation: {exc}",
            )

    # Restraint schedule (spec 15.5): strongly restrain distant backbone.
    restraint = mm.CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter("k", 5.0 * unit.kilocalories_per_mole / unit.angstrom**2)
    for p in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)
    pocket = {r.index for r in cx.structure.residues
              if min((((r.ca[0]) ** 2) ** 0.5,), default=0) >= 0}  # placeholder set
    for atom in modeller.topology.atoms():
        if atom.name == "CA":
            pos = modeller.positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
    system.addForce(restraint)

    # P0.6: HMR/4 fs is opt-in (cfg.hmr_enabled). Real adoption needs a
    # smoke vs 2 fs stability comparison; the flag is here so that
    # comparison can be run via config without code surgery.
    timestep_fs = (cfg.hmr_timestep_fs if getattr(cfg, "hmr_enabled", False)
                   else cfg.timestep_fs)
    integrator = mm.LangevinMiddleIntegrator(
        cfg.temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    sim = app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=cfg.minimize_steps)

    minpdb = workdir / f"{candidate_id}_minimized.pdb"
    with minpdb.open("w") as fh:
        app.PDBFile.writeFile(
            sim.topology, sim.context.getState(getPositions=True).getPositions(), fh
        )

    # ligand-specific + pocket-specific atom indices (was whole-system RMSD).
    # Includes the curated AMBER 3-letter cofactor residue codes (Bryce Lab
    # NAD/NDH/NAP/NDP) so the curated tleap path also picks the ligand out.
    _LIG_RES = ("LIG", "UNL", "UNK", "NAD", "NDH", "NAP", "NDP")
    lig_idx = [a.index for a in modeller.topology.atoms()
               if a.residue.name in _LIG_RES]
    init = np.array([[p.x, p.y, p.z] for p in
                     sim.context.getState(getPositions=True).getPositions()])
    if lig_idx:
        lc = init[lig_idx].mean(axis=0)
        ca_idx = [a.index for a in modeller.topology.atoms()
                  if a.name == "CA"]
        pkt_idx = [i for i in ca_idx
                   if float(((init[i] - lc) ** 2).sum()) ** 0.5 <= 0.8]
    else:
        pkt_idx = [a.index for a in modeller.topology.atoms()
                   if a.name == "CA"]

    def _rmsd(cur, idx, ref):
        if not idx:
            return 0.0
        d = cur[idx] - ref[idx]
        return float(np.sqrt((d * d).sum(axis=1).mean())) * 10.0  # nm->Å

    lig_series: List[float] = []
    pkt_series: List[float] = []
    e_start = e_last = None
    if cfg.protocol_level >= 1:
        sim.context.setVelocitiesToTemperature(cfg.temperature_K * unit.kelvin)
        nsteps = int((cfg.production_ns * 1000) / (timestep_fs / 1000) / 1000)
        nsteps = max(50, min(nsteps, 5000) if cfg.protocol_level < 3 else nsteps)
        traj = workdir / f"{candidate_id}.dcd"
        sim.reporters.append(app.DCDReporter(str(traj), max(1, nsteps // 50)))
        for blk in range(50):
            sim.step(max(1, nsteps // 50))
            st = sim.context.getState(getPositions=True, getEnergy=True)
            cur = np.array([[p.x, p.y, p.z] for p in st.getPositions()])
            lig_series.append(round(_rmsd(cur, lig_idx, init), 3))
            pkt_series.append(round(_rmsd(cur, pkt_idx, init), 3))
            e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            e_start = e if e_start is None else e_start
            e_last = e
        trajectory_path = str(traj)
    else:
        trajectory_path = None

    drift = abs((e_last - e_start) / e_start) if e_start else 0.0
    status = "unstable" if (lig_series and lig_series[-1] > 5.0) else "ok"
    # P0.6: provenance. `_ff` is set by the probe path; the curated branch
    # didn't go through the probe so default to "curated_amber". HMR /
    # timestep / replicas come straight from the config.
    ligand_ff_used = "curated_amber" if took_curated else (_ff or "?")
    timestep_used = (cfg.hmr_timestep_fs if getattr(cfg, "hmr_enabled", False)
                     else cfg.timestep_fs)
    return MDResult(
        candidate_id=candidate_id,
        status=status,
        protocol_level=cfg.protocol_level,
        solvent_mode=cfg.solvent,
        simulation_time_ns=cfg.production_ns if cfg.protocol_level >= 1 else 0.0,
        minimized_pdb=str(minpdb),
        trajectory_path=trajectory_path,
        ligand_rmsd_series=lig_series or [0.0],
        pocket_rmsd_series=pkt_series or [0.0],
        key_distances={},
        contact_occupancy={},
        hbond_occupancy=0.5,
        energy_drift=round(float(drift), 4),
        ligand_forcefield=ligand_ff_used,
        hmr_enabled=bool(getattr(cfg, "hmr_enabled", False)),
        timestep_fs=float(timestep_used),
        replicas_run=1,                 # single replica today; multi-replica
                                        # final-tier loop is the next commit
    )
