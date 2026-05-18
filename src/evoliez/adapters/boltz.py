"""Protein-ligand complex prediction (spec section 9).

real: Boltz / Boltz-2 (https://github.com/jwohlwend/boltz).
mock: deterministic synthetic complex (structure + pocketed ligand + scores).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from evoliez.adapters.base import (
    place_ligand_in_pocket,
    synthetic_structure,
    write_min_pdb,
)
from evoliez.config import Backend, ComplexPredictionConfig
from evoliez.logging_utils import get_logger
from evoliez.types import Complex, Ligand
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.boltz")


def predict_complex(
    label: str,
    sequence: str,
    ligand: Ligand,
    cfg: ComplexPredictionConfig,
    outdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
    msa_path: Optional[Path] = None,
) -> Complex:
    outdir.mkdir(parents=True, exist_ok=True)
    if backend is Backend.real:
        return _predict_real(
            label, sequence, ligand, cfg, outdir, dry_run=dry_run, msa_path=msa_path
        )
    return _predict_mock(label, sequence, ligand, outdir)


def _predict_real(
    label: str,
    sequence: str,
    ligand: Ligand,
    cfg: ComplexPredictionConfig,
    outdir: Path,
    *,
    dry_run: bool,
    msa_path: Optional[Path],
) -> Complex:
    require("boltz")
    apply_gpu_selection()
    spec = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},
            {"ligand": {"id": "B", "smiles": ligand.smiles}},
        ],
    }
    if cfg.predict_affinity:
        spec["properties"] = [{"affinity": {"binder": "B"}}]
    if msa_path is not None:
        spec["sequences"][0]["protein"]["msa"] = str(msa_path)

    yml = outdir / f"{label}_boltz_input.yaml"
    yml.write_text(yaml.safe_dump(spec, sort_keys=False))

    cmd = ["boltz", "predict", str(yml), "--out_dir", str(outdir)]
    if cfg.use_msa_server and msa_path is None:
        cmd.append("--use_msa_server")
    run(cmd, dry_run=dry_run, timeout=None)

    if dry_run:
        return _predict_mock(label, sequence, ligand, outdir)

    pdbs = sorted(outdir.rglob("*.pdb")) + sorted(outdir.rglob("*.cif"))
    if not pdbs:
        log.warning("Boltz produced no structure for %s; using mock fallback", label)
        return _predict_mock(label, sequence, ligand, outdir)
    cx = _parse_real_structure(pdbs[0], sequence, ligand)
    cx.method = cfg.primary_method
    cx.path = str(pdbs[0])
    conf, aff = _parse_real_scores(outdir)
    cx.confidence = conf
    cx.affinity_score = aff
    return cx


def _parse_real_structure(pdb: Path, sequence: str, ligand: Ligand) -> Complex:
    """Parse a Boltz structure. Uses Biopython when available; otherwise a
    light CA/HETATM parser sufficient for the geometry layer."""
    from evoliez.types import LigandAtom, ProteinStructure, Residue

    residues: list[Residue] = []
    lig_atoms: list[LigandAtom] = []
    text = pdb.read_text() if pdb.suffix == ".pdb" else ""
    if text:
        for line in text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                idx = int(line[22:26])
                x, y, z = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
                residues.append(
                    Residue(index=idx, aa="X", ca=(x, y, z),
                            sidechain_centroid=(x, y, z))
                )
            elif line.startswith("HETATM"):
                x, y, z = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
                el = line[76:78].strip() or line[12:14].strip()
                lig_atoms.append(
                    LigandAtom(id=f"{el}{len(lig_atoms)}", element=el or "C",
                               coord=(x, y, z))
                )
    for i, r in enumerate(residues):
        if i < len(sequence):
            r.aa = sequence[i]
    if not lig_atoms:
        lig_atoms = ligand.atoms
    struct = ProteinStructure(
        sequence=sequence, residues=residues, method="boltz", pdb_path=str(pdb)
    )
    return Complex(structure=struct, ligand=Ligand(
        id=ligand.id, smiles=ligand.smiles, atoms=lig_atoms,
        formal_charge=ligand.formal_charge, source=ligand.source,
    ))


def _parse_real_scores(outdir: Path) -> tuple[float, Optional[float]]:
    import json

    conf, aff = 0.0, None
    for jf in outdir.rglob("confidence*.json"):
        try:
            data = json.loads(jf.read_text())
            conf = float(data.get("confidence_score", data.get("ptm", 0.0)))
        except Exception:
            pass
    for jf in outdir.rglob("affinity*.json"):
        try:
            data = json.loads(jf.read_text())
            aff = float(
                data.get("affinity_pred_value", data.get("affinity", 0.0))
            )
        except Exception:
            pass
    return conf, aff


def _predict_mock(
    label: str, sequence: str, ligand: Ligand, outdir: Path
) -> Complex:
    seed = derive_seed(0xB01D, label, sequence[:32])
    struct = synthetic_structure(sequence, seed=seed, method="boltz-mock")
    placed = place_ligand_in_pocket(struct, ligand, seed=seed)
    pdb = outdir / f"{label}_complex.pdb"
    write_min_pdb(pdb, struct, placed)
    struct.pdb_path = str(pdb)
    cx = Complex(
        structure=struct,
        ligand=Ligand(
            id=ligand.id, smiles=ligand.smiles, atoms=placed,
            formal_charge=ligand.formal_charge,
            n_rotatable_bonds=ligand.n_rotatable_bonds, source=ligand.source,
        ),
        method="boltz-mock",
        confidence=round(0.55 + (seed % 40) / 100.0, 3),
        affinity_score=round(-4.0 - (seed % 600) / 100.0, 3),
        path=str(pdb),
    )
    return cx
