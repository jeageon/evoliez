"""Boltz output scoping: the s06b ensemble cross-contamination bug + resume-skip.

Boltz writes boltz_results_* and predict_complex DISCOVERS them by globbing the
out_dir. The s06b ensemble pointed EVERY representative at one shared out_dir, so
each rep parsed found[0]=hom_000 and `_parse_real_samples` mixed every rep's
confidence files into one prediction (silent corruption + an O(N^2) re-parse of
the whole tree per rep). Per-rep out_dirs fix it. We also pin the resume-skip:
a complete existing output is reused, not recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import evoliez.adapters.boltz as boltz_mod
from evoliez.adapters.boltz import predict_complex
from evoliez.config import Backend, ComplexPredictionConfig
from evoliez.types import Ligand, LigandAtom


@pytest.fixture(autouse=True)
def _no_boltz_binary(monkeypatch):
    # boltz isn't on PATH locally (server-only); these tests exercise the
    # discovery/skip/parse logic, not the binary.
    monkeypatch.setattr(boltz_mod, "require", lambda *a, **k: "boltz")


def _write_model_pdb(path: Path, lig_xyz):
    lines = [
        "ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00 85.00           C",
        "ATOM      4  CA  GLY A   2       4.000   5.000   6.000  1.00 78.00           C",
        "ATOM      5  CA  SER A   3       7.000   8.000   9.000  1.00 60.00           C",
    ]
    for serial, (el, (x, y, z)) in zip((6, 7, 8), zip("OPC", lig_xyz)):
        rec = (f"HETATM{serial:>5d}  {el}1  LIG L   1    "
               f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
        lines.append(f"{rec:<76}{el:>2s}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _fake_rep_output(outdir: Path, label: str, n_samples: int):
    """A finished Boltz run for ONE rep: n_samples models + confidence + plddt."""
    stem = f"{label}_boltz_input"
    preds = outdir / f"boltz_results_{stem}" / "predictions" / stem
    preds.mkdir(parents=True)
    for k in range(n_samples):
        _write_model_pdb(preds / f"{stem}_model_{k}.pdb",
                         [(7.0 + k, 8.0, 9.0), (7.4 + k, 8.4, 9.3), (7.8 + k, 8.8, 9.6)])
        (preds / f"confidence_{stem}_model_{k}.json").write_text(json.dumps(
            {"confidence_score": 0.9 - 0.1 * k, "complex_plddt": 0.8, "iptm": 0.7}))
        np.savez(preds / f"plddt_{stem}_model_{k}.npz", plddt=np.array([0.85, 0.78, 0.60]))


def _ligand():
    return Ligand(id="L", smiles="OPC",
                  atoms=[LigandAtom(id="O0", element="O", coord=(0.0, 0.0, 0.0)),
                         LigandAtom(id="P1", element="P", coord=(0.5, 0.0, 0.0)),
                         LigandAtom(id="C2", element="C", coord=(1.0, 0.0, 0.0))])


def test_per_rep_outdir_is_not_cross_contaminated(tmp_path, monkeypatch):
    """Two reps with DIFFERENT sample counts, each in its OWN subdir: parsing one
    sees only its own samples, and a complete output is reused (no recompute)."""
    parent = tmp_path / "representatives"
    _fake_rep_output(parent / "hom_000", "hom_000", 2)
    _fake_rep_output(parent / "hom_001", "hom_001", 3)
    cfg = ComplexPredictionConfig(diffusion_samples=2)
    called = {"run": 0}
    monkeypatch.setattr(boltz_mod, "run",
                        lambda *a, **k: called.__setitem__("run", called["run"] + 1))

    cx = predict_complex("hom_000", "AGS", _ligand(), cfg, parent / "hom_000",
                         backend=Backend.real, dry_run=False, seed=7)
    assert len(cx.samples) == 2          # rep_001's 3 samples never leak in
    assert called["run"] == 0            # complete output present -> skip recompute


def test_shared_outdir_mixes_reps(tmp_path, monkeypatch):
    """Characterises the bug the per-rep fix avoids: BOTH reps in one shared
    out_dir -> a parse there sees 2+3 = 5 samples (cross-contamination)."""
    shared = tmp_path / "shared"
    _fake_rep_output(shared, "hom_000", 2)
    _fake_rep_output(shared, "hom_001", 3)
    cfg = ComplexPredictionConfig(diffusion_samples=2)
    monkeypatch.setattr(boltz_mod, "run", lambda *a, **k: None)

    cx = predict_complex("hom_000", "AGS", _ligand(), cfg, shared,
                         backend=Backend.real, dry_run=False, seed=7)
    assert len(cx.samples) == 5          # the contamination per-rep scoping prevents


def test_incomplete_output_triggers_recompute(tmp_path, monkeypatch):
    """Fewer models than diffusion_samples -> NOT a complete output -> recompute
    (the skip must not reuse a half-finished/crashed run)."""
    rep = tmp_path / "hom_000"
    _fake_rep_output(rep, "hom_000", 1)          # only 1 of 4 requested
    cfg = ComplexPredictionConfig(diffusion_samples=4)
    called = {"run": 0}
    monkeypatch.setattr(boltz_mod, "run",
                        lambda *a, **k: called.__setitem__("run", called["run"] + 1))

    predict_complex("hom_000", "AGS", _ligand(), cfg, rep,
                    backend=Backend.real, dry_run=False, seed=7)
    assert called["run"] == 1                    # re-ran Boltz (output incomplete)


def _het(serial, el, chain, resn, resseq, xyz):
    x, y, z = xyz
    rec = (f"HETATM{serial:>5d}  {el}1  {resn} {chain}{resseq:>4d}    "
           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
    return f"{rec:<76}{el:>2s}"


def test_pdb_parser_keeps_only_primary_ligand_chain(tmp_path):
    """Multi-ligand fix: when extra ligands are co-modelled (chains C, D, …) the
    parser must keep ONLY the primary ligand (chain B), not merge formate/etc.
    into the primary's atom list (the NADP-48 vs 51-atom contamination)."""
    from evoliez.adapters.boltz import _parse_pdb_atoms

    lines = [
        "ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00 85.00           C",
        _het(6, "P", "B", "NAP", 1, (7.0, 8.0, 9.0)),    # primary ligand, chain B
        _het(7, "O", "B", "NAP", 1, (7.5, 8.5, 9.5)),
        _het(8, "N", "B", "NAP", 1, (8.0, 9.0, 10.0)),
        _het(9, "C", "C", "FMT", 2, (2.0, 3.0, 4.0)),    # co-modelled formate, chain C
        _het(10, "O", "C", "FMT", 2, (2.5, 3.5, 4.5)),
        "END",
    ]
    p = tmp_path / "multilig.pdb"
    p.write_text("\n".join(lines) + "\n")
    residues, lig = _parse_pdb_atoms(p)
    assert len(residues) == 1
    assert len(lig) == 3                              # chain B only, formate dropped
    assert [a.element for a in lig] == ["P", "O", "N"]
