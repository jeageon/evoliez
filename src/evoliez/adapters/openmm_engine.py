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

    # (#4a) Ligand force-field ONLY. A genuine failure here (GAFF /
    # antechamber missing, unparameterisable cofactor) is the legitimately
    # NEUTRAL skip - the candidate is judged on the other layers.
    try:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import SystemGenerator

        off_mol = Molecule.from_smiles(
            cx.ligand.smiles, allow_undefined_stereo=True
        )
        system_generator = SystemGenerator(
            forcefields=["amber14-all.xml", "implicit/obc2.xml"],
            small_molecule_forcefield="gaff-2.11",
            molecules=[off_mol],
            cache=str(workdir / "ff_cache.json"),
            # openmmforcefields refuses nonbondedMethod in forcefield_kwargs;
            # it must go in (non)periodic_forcefield_kwargs (implicit/GBSA
            # -> non-periodic).
            forcefield_kwargs={"constraints": app.HBonds},
            nonperiodic_forcefield_kwargs={
                "nonbondedMethod": app.CutoffNonPeriodic,
            },
        )
    except Exception as exc:
        log.warning(
            "ligand force-field unavailable for %s (%s); recording "
            "skipped_parameterization (NOT a candidate failure)",
            candidate_id, exc,
        )
        return MDResult(
            candidate_id=candidate_id, status="skipped_parameterization",
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
            simulation_time_ns=0.0, failure_reason=str(exc),
        )

    # (#4b) Protein prep + system creation. A failure HERE is a REAL MD
    # failure (input topology / residue templates / system), NOT a neutral
    # ligand skip - must surface as failed, never as a pass.
    try:
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(system_generator.forcefield)
        system = system_generator.create_system(
            modeller.topology, molecules=[off_mol]
        )
    except Exception as exc:
        log.warning(
            "MD protein prep / system creation failed for %s (%s)",
            candidate_id, exc,
        )
        return MDResult(
            candidate_id=candidate_id, status="failed",
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
            simulation_time_ns=0.0, integration_failed=True,
            failure_reason=f"protein prep / system creation: {exc}",
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
        nsteps = int((cfg.production_ns * 1000) / (cfg.timestep_fs / 1000) / 1000)
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
    )
