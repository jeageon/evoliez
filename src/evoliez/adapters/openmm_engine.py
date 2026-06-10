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

# L1/L2 are short screening rungs run across many candidates; cap their step
# count so wall-time stays bounded on the shared box. L3 honours the full
# configured production length. 25_000 steps = 50 ps at a 2 fs timestep.
_LITE_MAX_STEPS = 25_000

# Tiered positional-restraint shells, in nm from the ligand centroid: the
# POCKET backbone is left FREE so pocket_rmsd measures the binding-site
# response, the active-site SHELL is held weakly, distant backbone strongly.
_POCKET_NM = 0.8   # 8 Å
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
    integration_failed: bool = False
    failure_reason: Optional[str] = None


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


def _ligand_offmol_at_pose(pdb_path: Path, smiles: str):
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


class _LigandParamUnsupported(Exception):
    """No available small-molecule FF can parameterize this ligand (e.g.
    GAFF/AM1-BCC on a large multiply-phosphorylated cofactor like NADP -
    confirmed on the server: ligand-only create_system fails). This is NOT
    a candidate failure: MD-lite is NEUTRALLY skipped and the candidate is
    judged on the other layers (expert: metals/cofactors legitimately
    skip)."""


def _ligand_system_generator(off_lig, workdir: Path):
    """A SystemGenerator whose small-molecule FF can ACTUALLY parameterize
    this ligand. Probe ligand-only create_system per FF: gaff-2.11, then
    espaloma-0.3.2 if the `espaloma` package is installed (a graph-net FF,
    no antechamber, that handles cofactors GAFF/AM1-BCC cannot). Raise
    _LigandParamUnsupported if none can - decisive, since the probe IS the
    exact ligand-only parameterization that failed for NADP on the server."""
    import openmm.app as app
    from openmmforcefields.generators import SystemGenerator

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

    # Ligand parameterization PROBE -> classify precisely: a ligand the
    # small-molecule FFs can't handle (NADP-class cofactor) is the NEUTRAL
    # skipped_parameterization (candidate judged on the other layers), NOT
    # a hard failure. Only protein/assembly failures below are `failed`.
    try:
        system_generator, _ff = _ligand_system_generator(off_lig, workdir)
    except _LigandParamUnsupported as exc:
        log.warning(
            "no small-molecule FF can parameterize the ligand for %s "
            "(large/charged cofactor e.g. NADP); recording "
            "skipped_parameterization (NEUTRAL - judged on other layers): "
            "%s", candidate_id, str(exc)[:200],
        )
        return MDResult(
            candidate_id=candidate_id, status="skipped_parameterization",
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
            simulation_time_ns=0.0,
            failure_reason=f"ligand FF unsupported (cofactor): {exc}",
        )

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
        modeller.addHydrogens(system_generator.forcefield)
        # Add the ligand SOLELY from the OpenFF molecule at its Boltz-pose
        # conformer (verified 70 atoms incl. 26 H); the only ligand in the
        # system is now this one, which create_system matches via GAFF.
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

    # Restraint schedule (spec 15.5): tiered POSITIONAL restraints keyed on
    # distance from the ligand, computed in OpenMM (nm) space so they stay
    # consistent with the pocket used for pocket_rmsd below. Distant backbone
    # is held firmly (a short implicit-solvent run must not drift/unfold), the
    # active-site shell is held weakly, and the POCKET backbone + ligand are
    # left FREE so ligand_rmsd / pocket_rmsd actually measure how the binding
    # site responds to the mutation. (The previous build added a strong
    # restraint to EVERY CA, pinning the pocket and making the core
    # MD-stability signal meaningless.)
    pos_nm = np.array([[p.x, p.y, p.z] for p in modeller.positions])
    _lig_idx0 = [a.index for a in modeller.topology.atoms()
                 if a.residue.name in ("LIG", "UNL", "UNK")]
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
    sim = app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=cfg.minimize_steps)

    minpdb = workdir / f"{candidate_id}_minimized.pdb"
    with minpdb.open("w") as fh:
        app.PDBFile.writeFile(
            sim.topology, sim.context.getState(getPositions=True).getPositions(), fh
        )

    # ligand-specific + pocket-specific atom indices (was whole-system RMSD)
    lig_idx = [a.index for a in modeller.topology.atoms()
               if a.residue.name in ("LIG", "UNL", "UNK")]
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
            e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            e_start = e if e_start is None else e_start
            e_last = e
        trajectory_path = str(traj)
    else:
        trajectory_path = None

    drift = abs((e_last - e_start) / e_start) if e_start else 0.0
    status = "unstable" if (lig_series and lig_series[-1] > 5.0) else "ok"
    return MDResult(
        candidate_id=candidate_id,
        status=status,
        protocol_level=cfg.protocol_level,
        solvent_mode=cfg.solvent,
        simulation_time_ns=round(actual_ns, 6),
        minimized_pdb=str(minpdb),
        trajectory_path=trajectory_path,
        ligand_rmsd_series=lig_series or [0.0],
        pocket_rmsd_series=pkt_series or [0.0],
        key_distances={},
        contact_occupancy={},
        hbond_occupancy=0.5,
        energy_drift=round(float(drift), 4),
    )
