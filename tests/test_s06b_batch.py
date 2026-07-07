"""GPU-batched s06b ensemble: the Phase-2 per-stem scoped parse must NOT
cross-contaminate representatives.

A batched `boltz predict <dir>` writes EVERY rep under ONE shared
`boltz_results_<chunk>/predictions/` tree, one `<stem>/` subdir per rep.
`parse_prediction_dir` scopes each rep to its own stem subdir — this pins that a
rep's parse sees ONLY its own confidence/plddt/pdb files, never a sibling's (the
exact unscoped-glob contamination defect we fixed once before).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evoliez.adapters.boltz import parse_prediction_dir
from evoliez.types import Ligand, LigandAtom


def _write_model_pdb(path: Path, lig_xyz):
    lines = [
        "ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00 85.00           C",
        "ATOM      4  CA  GLY A   2       4.000   5.000   6.000  1.00 78.00           C",
        "ATOM      5  CA  SER A   3       7.000   8.000   9.000  1.00 60.00           C",
    ]
    for serial, (el, (x, y, z)) in zip((6, 7, 8), zip("OPC", lig_xyz)):
        rec = (f"HETATM{serial:>5d}  {el}1  LIG B   1    "
               f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
        lines.append(f"{rec:<76}{el:>2s}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _fake_batch_rep(predictions_dir: Path, stem: str, n_samples: int):
    """One rep's prediction in the batched layout: predictions/<stem>/<stem>_model_k.*"""
    d = predictions_dir / stem
    d.mkdir(parents=True)
    for k in range(n_samples):
        _write_model_pdb(d / f"{stem}_model_{k}.pdb",
                         [(7.0 + k, 8.0, 9.0), (7.4 + k, 8.4, 9.3), (7.8 + k, 8.8, 9.6)])
        (d / f"confidence_{stem}_model_{k}.json").write_text(json.dumps(
            {"confidence_score": 0.9 - 0.05 * k, "complex_plddt": 0.8, "iptm": 0.7}))
        np.savez(d / f"plddt_{stem}_model_{k}.npz", plddt=np.array([0.85, 0.78, 0.60]))


def _ligand():
    return Ligand(id="L", smiles="OPC",
                  atoms=[LigandAtom(id="O0", element="O", coord=(0.0, 0.0, 0.0)),
                         LigandAtom(id="P1", element="P", coord=(0.5, 0.0, 0.0)),
                         LigandAtom(id="C2", element="C", coord=(1.0, 0.0, 0.0))])


def test_batched_parse_is_per_stem_scoped(tmp_path):
    # ONE shared chunk output tree, TWO reps with DIFFERENT sample counts.
    preds = tmp_path / "boltz_results_chunk" / "predictions"
    _fake_batch_rep(preds, "hom_000_boltz_input", 2)
    _fake_batch_rep(preds, "hom_001_boltz_input", 3)

    cx0 = parse_prediction_dir(preds / "hom_000_boltz_input", "AGS", _ligand(), "boltz2")
    cx1 = parse_prediction_dir(preds / "hom_001_boltz_input", "AGS", _ligand(), "boltz2")
    assert cx0 is not None and cx1 is not None
    assert len(cx0.samples) == 2      # rep_000 sees ONLY its own 2 samples
    assert len(cx1.samples) == 3      # rep_001 sees ONLY its own 3 — no leak either way


def test_missing_stem_returns_none(tmp_path):
    preds = tmp_path / "boltz_results_chunk" / "predictions"
    preds.mkdir(parents=True)
    assert parse_prediction_dir(
        preds / "hom_404_boltz_input", "AGS", _ligand(), "boltz2") is None


def test_empty_rep_chunk_guard_skips_boltz(tmp_path):
    """REGRESSION (E1 subset crash): fewer reps than GPUs (3 homologs / 4 GPUs) -> _lpt_partition
    yields empty buckets. An empty rep chunk must return cleanly WITHOUT `boltz predict` on a
    never-created _batch_in_gpuN dir ('Path does not exist' killed s06b). Same class as PR #9 (s08b)."""
    from evoliez.config import ComplexPredictionConfig
    from evoliez.stages.s06b_interaction_model import _run_batch_chunk
    cp = ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False)
    payload = ("3", [], _ligand(), cp, str(tmp_path), 42, None, None)
    gpu, results_dir, reps = _run_batch_chunk(payload)
    assert gpu == "3" and results_dir == "" and reps == []
    assert not (tmp_path / "_batch_in_gpu3").exists()
