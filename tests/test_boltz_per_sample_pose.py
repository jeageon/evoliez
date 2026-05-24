"""A1 fix: per-sample HETATM parsing.

Lock that `_parse_real_samples` reads HETATM coordinates from each
`*_model_<k>.{pdb,cif}` so the diffusion ensemble carries distinct
ligand poses (and `pose_consensus` / `ligand_rmsd_std` are non-zero).

Cheap-run diagnosis: the production output on /mnt/data2 had 8 distinct
pose SHAs under `boltz_results_*/predictions/<name>/<name>_model_*.pdb`,
but in-memory `BoltzSample.ligand_atoms` was the same object across
every sample because `_parse_real_samples` ignored the per-model files
and reused the ensemble-level `lig_atoms`. That collapsed the consensus
to zero variance and fed the reranker a "perfectly consistent" pose
even when Boltz disagreed across diffusion samples.

These fixtures are pure text (PDB + JSON), so the test runs in any
local venv - no rdkit / openff / boltz install needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoliez.adapters import boltz as boltz_adapter
from evoliez.types import LigandAtom


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _pdb_hetatm(serial: int, name: str, resn: str, resi: int,
                xyz: tuple[float, float, float], el: str) -> str:
    """Produce a column-correct PDB HETATM record so the strict
    `line[30:38]` / `line[76:78]` slices in `_parse_ligand_atoms_from_struct`
    actually parse the fields we put in.

    PDB column map: 1-6 record, 7-11 serial, 13-16 name, 17 altLoc,
    18-20 resName, 22 chainID, 23-26 resSeq, 27 iCode, 31-38 x,
    39-46 y, 47-54 z, 55-60 occ, 61-66 temp, 77-78 element."""
    x, y, z = xyz
    return (
        f"HETATM{serial:>5d} {name:<4s} {resn:>3s} A{resi:>4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}"
    )


def _write_sample(pred_dir: Path, name: str, k: int,
                  hetatm_coords: list[tuple[float, float, float]],
                  plddt: float, iptm: float) -> None:
    """Write the trio Boltz emits per diffusion sample:
    `confidence_<name>_model_<k>.json`, `<name>_model_<k>.pdb`, and
    `plddt_<name>_model_<k>.npz` (npz optional; we skip it - the parser
    is robust to missing plddt)."""
    (pred_dir / f"confidence_{name}_model_{k}.json").write_text(
        json.dumps({
            "complex_plddt": plddt,
            "iptm": iptm,
            "ligand_iptm": 0.6,
            "complex_iplddt": plddt,
            "complex_ipde": 1.0,
            "complex_pde": 1.0,
            "ptm": 0.75,
        })
    )
    lines = ["REMARK synthetic per-sample fixture"]
    # one CA so the parser sees a protein too (not required for HETATM
    # parsing, but keeps the file shaped like a real Boltz output).
    lines.append(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C"
    )
    for i, c in enumerate(hetatm_coords, start=2):
        # 3 distinct elements so element-labelled graph matching has
        # something to chew on if relabel_to_canonical kicks in.
        el = ("C", "N", "O")[(i - 2) % 3]
        lines.append(_pdb_hetatm(i, f"{el}{i-1}", "LIG", 999,
                                 c, el))
    lines.append("END")
    (pred_dir / f"{name}_model_{k}.pdb").write_text("\n".join(lines) + "\n")


@pytest.fixture
def per_sample_outdir(tmp_path: Path) -> Path:
    """Two samples (model_0, model_1) at the same predictions path,
    with deliberately DIFFERENT HETATM coords. Mimics what Boltz
    actually writes for a 2-sample diffusion ensemble."""
    name = "label_boltz_input"
    pred = tmp_path / "predictions" / name
    pred.mkdir(parents=True, exist_ok=True)

    # sample 0: a triangle near the origin
    _write_sample(pred, name, 0,
                  [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                  plddt=85.0, iptm=0.72)
    # sample 1: same chemistry, SHIFTED by +5Å in x  (distinct pose)
    _write_sample(pred, name, 1,
                  [(5.0, 0.0, 0.0), (6.0, 0.0, 0.0), (5.0, 1.0, 0.0)],
                  plddt=80.0, iptm=0.70)
    (pred / f"affinity_{name}.json").write_text(
        json.dumps({"affinity_pred_value": 5.2,
                    "affinity_probability_binary": 0.8})
    )
    return tmp_path


# Canonical ligand: same element sequence (C, N, O) so
# `relabel_to_canonical` can lock by graph (3 atoms, all heavy, no H).
_CANONICAL = [
    LigandAtom(id="C0_canonical", element="C", coord=(0.0, 0.0, 0.0)),
    LigandAtom(id="N1_canonical", element="N", coord=(0.0, 0.0, 0.0)),
    LigandAtom(id="O2_canonical", element="O", coord=(0.0, 0.0, 0.0)),
]


# --------------------------------------------------------------------------- #
# Helpers under test
# --------------------------------------------------------------------------- #
def test_model_index_parses_trailing_integer(tmp_path: Path):
    """`_model_index` powers the confidence↔structure pairing - if it
    silently mis-parses, samples cross-pollinate."""
    p0 = tmp_path / "abc_model_0.pdb"
    p7 = tmp_path / "long-name_with_dashes_model_7.cif"
    p_no = tmp_path / "no_model_marker.pdb"
    p_bad = tmp_path / "abc_model_X.pdb"
    for p in (p0, p7, p_no, p_bad):
        p.touch()
    assert boltz_adapter._model_index(p0) == 0
    assert boltz_adapter._model_index(p7) == 7
    assert boltz_adapter._model_index(p_no) is None
    assert boltz_adapter._model_index(p_bad) is None


def test_parse_ligand_atoms_from_pdb_reads_hetatm(tmp_path: Path):
    """Pure-text PDB HETATM parser - the building block for per-sample
    pose ingestion."""
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "REMARK demo\n"
        + _pdb_hetatm(1, "C1", "LIG", 1, (1.1, 2.2, 3.3), "C") + "\n"
        + _pdb_hetatm(2, "N1", "LIG", 1, (4.4, 5.5, 6.6), "N") + "\n"
        + "END\n"
    )
    atoms = boltz_adapter._parse_ligand_atoms_from_struct(pdb)
    assert len(atoms) == 2
    assert atoms[0].element == "C"
    assert atoms[0].coord == pytest.approx((1.1, 2.2, 3.3))
    assert atoms[1].element == "N"
    assert atoms[1].coord == pytest.approx((4.4, 5.5, 6.6))


def test_scan_output_dir_collects_model_struct(per_sample_outdir: Path):
    """`_scan_output_dir` must surface per-model structure files so
    `_parse_real_samples` can pair them with confidence JSONs without
    re-walking the tree."""
    files = boltz_adapter._scan_output_dir(per_sample_outdir)
    assert len(files["model_struct"]) == 2
    for sp in files["model_struct"]:
        assert sp.name.endswith(".pdb")
        assert "_model_" in sp.name


# --------------------------------------------------------------------------- #
# The real fix: distinct per-sample ligand_atoms in the in-memory ensemble
# --------------------------------------------------------------------------- #
def test_parse_real_samples_attaches_distinct_per_sample_poses(
    per_sample_outdir: Path,
):
    """Each BoltzSample.ligand_atoms must hold the HETATM coords from
    its OWN model file, not a shared reference to the ensemble atoms.
    Before A1, every sample shared one object, so the diffusion-ensemble
    pose variance collapsed to zero and pose_consensus was useless."""
    samples = boltz_adapter._parse_real_samples(per_sample_outdir,
                                                _CANONICAL)
    assert len(samples) == 2

    # Coords differ across samples (the +5Å shift in x we wrote).
    coords_per_sample = [
        [a.coord for a in s.ligand_atoms] for s in samples
    ]
    assert coords_per_sample[0] != coords_per_sample[1], (
        "per-sample HETATM coords collapsed to a single shared object"
    )

    # Variance across samples is real (the entire point of the fix).
    xs = [c[0] for sample in coords_per_sample for c in sample]
    assert max(xs) - min(xs) >= 4.9, (
        "sample-to-sample ligand displacement not captured; "
        "pose_consensus would still report 0.0 variance"
    )

    # And id-lock to canonical ids stayed intact (downstream id-keyed
    # features rely on this). Identity check on the id strings.
    for s in samples:
        ids = [a.id for a in s.ligand_atoms]
        assert ids == [a.id for a in _CANONICAL], (
            "relabel_to_canonical did not lock per-sample atoms to "
            "canonical ids; id-keyed features would break"
        )


def test_parse_real_samples_falls_back_to_ensemble_when_no_model_file(
    tmp_path: Path,
):
    """If a Boltz output only has confidence JSONs (no model_k structure
    files), the parser must NOT crash and must NOT produce empty ligand
    atoms - it falls back to the ensemble `lig_atoms` so downstream
    features still see a ligand (slightly degraded: pose_consensus
    variance will be 0, but that's correct given no per-sample data)."""
    name = "label_boltz_input"
    pred = tmp_path / "predictions" / name
    pred.mkdir(parents=True, exist_ok=True)
    for k in range(2):
        (pred / f"confidence_{name}_model_{k}.json").write_text(
            json.dumps({
                "complex_plddt": 80.0,
                "iptm": 0.7,
                "ligand_iptm": 0.6,
                "complex_iplddt": 80.0,
                "complex_ipde": 1.0,
                "complex_pde": 1.0,
                "ptm": 0.75,
            })
        )
    samples = boltz_adapter._parse_real_samples(tmp_path, _CANONICAL)
    assert len(samples) == 2
    for s in samples:
        assert s.ligand_atoms is _CANONICAL or [
            a.id for a in s.ligand_atoms
        ] == [a.id for a in _CANONICAL], (
            "fallback path should preserve ensemble ligand atoms"
        )


def test_parse_real_samples_pairs_within_same_predictions_dir(
    tmp_path: Path,
):
    """Cross-directory pairing would silently mix samples from
    different mutants. Lock that the (parent, k) key respects directory
    boundaries: two mutants each with their own model_0 must NOT share
    coordinates."""
    for mutant in ("WT", "Y223F"):
        pred = tmp_path / "predictions" / f"{mutant}_boltz_input"
        pred.mkdir(parents=True, exist_ok=True)
        shift = 0.0 if mutant == "WT" else 100.0  # huge displacement
        _write_sample(
            pred, f"{mutant}_boltz_input", 0,
            [(shift + 0.0, 0.0, 0.0),
             (shift + 1.0, 0.0, 0.0),
             (shift + 0.0, 1.0, 0.0)],
            plddt=80.0, iptm=0.7,
        )

    # Parse the WT label only (scoped, as production does).
    samples = boltz_adapter._parse_real_samples(
        tmp_path / "predictions" / "WT_boltz_input", _CANONICAL
    )
    assert len(samples) == 1
    xs = [a.coord[0] for a in samples[0].ligand_atoms]
    assert max(xs) < 50.0, (
        "WT sample picked up Y223F's HETATMs; (parent, k) keying broke"
    )
