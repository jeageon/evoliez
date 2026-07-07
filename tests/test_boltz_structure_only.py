"""Two QUALITY-NEUTRAL Boltz perf fixes (output-identical):

(A) ``parse_prediction_dir(..., structure_only=True)`` parses ONLY the
    representative model (receptor + reference ligand) and SKIPS the full
    diffusion-sample ensemble — the s06b docking phase never reads ``cx.samples``.
    Pinned: structure_only returns a populated ``.structure`` + ``.ligand.atoms``
    with ``.samples == []`` and the SAME ``.structure`` as the full parse, so the
    default-False path stays byte-identical.

(B) ``predict_batch`` resume is PER-STEM: a chunk where only some stems are
    incomplete re-folds ONLY the missing stems (not the whole chunk). Pinned by
    mocking the Boltz ``run`` to record exactly which stems it was asked to fold.

Fixtures/patterns reuse tests/test_s06b_batch.py and tests/test_boltz_cif.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evoliez.adapters import boltz
from evoliez.adapters.boltz import parse_prediction_dir, predict_batch
from evoliez.config import ComplexPredictionConfig
from evoliez.types import Ligand, LigandAtom


# --------------------------------------------------------------------------- #
# Shared fixtures (mirrors tests/test_s06b_batch.py)
# --------------------------------------------------------------------------- #
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
    """One rep's prediction in the batched layout:
    predictions/<stem>/<stem>_model_k.*"""
    d = predictions_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n_samples):
        _write_model_pdb(
            d / f"{stem}_model_{k}.pdb",
            [(7.0 + k, 8.0, 9.0), (7.4 + k, 8.4, 9.3), (7.8 + k, 8.8, 9.6)],
        )
        (d / f"confidence_{stem}_model_{k}.json").write_text(json.dumps(
            {"confidence_score": 0.9 - 0.05 * k, "complex_plddt": 0.8, "iptm": 0.7}))
        np.savez(d / f"plddt_{stem}_model_{k}.npz", plddt=np.array([0.85, 0.78, 0.60]))


def _ligand():
    return Ligand(id="L", smiles="OPC",
                  atoms=[LigandAtom(id="O0", element="O", coord=(0.0, 0.0, 0.0)),
                         LigandAtom(id="P1", element="P", coord=(0.5, 0.0, 0.0)),
                         LigandAtom(id="C2", element="C", coord=(1.0, 0.0, 0.0))])


# --------------------------------------------------------------------------- #
# (A) STRUCTURE-ONLY parse: receptor + reference ligand, NO sample ensemble
# --------------------------------------------------------------------------- #
def test_structure_only_skips_samples_keeps_structure_and_ligand(tmp_path):
    preds = tmp_path / "boltz_results_chunk" / "predictions"
    _fake_batch_rep(preds, "hom_000_boltz_input", 3)
    stem_dir = preds / "hom_000_boltz_input"

    cx = parse_prediction_dir(stem_dir, "AGS", _ligand(), "boltz2",
                              structure_only=True)
    assert cx is not None
    # Receptor + reference ligand are populated...
    assert cx.structure.residues, "structure_only dropped the receptor backbone"
    assert cx.ligand.atoms, "structure_only dropped the reference ligand atoms"
    assert cx.structure.pdb_path, "structure_only dropped structure.pdb_path"
    # ...but the (expensive) diffusion-sample ensemble is skipped entirely.
    assert cx.samples == [], "structure_only must leave cx.samples == []"


def test_structure_only_structure_matches_full_parse(tmp_path):
    """The receptor + reference ligand from a structure_only parse are IDENTICAL
    to a full parse — only the sample ensemble differs."""
    preds = tmp_path / "boltz_results_chunk" / "predictions"
    _fake_batch_rep(preds, "hom_000_boltz_input", 3)
    stem_dir = preds / "hom_000_boltz_input"

    full = parse_prediction_dir(stem_dir, "AGS", _ligand(), "boltz2")
    only = parse_prediction_dir(stem_dir, "AGS", _ligand(), "boltz2",
                                structure_only=True)
    assert full is not None and only is not None

    # full parse builds the whole ensemble; structure_only skips it
    assert len(full.samples) == 3
    assert only.samples == []

    # SAME structure: same backbone (residue indices + CA coords) ...
    assert ([r.index for r in only.structure.residues]
            == [r.index for r in full.structure.residues])
    assert ([r.ca for r in only.structure.residues]
            == [r.ca for r in full.structure.residues])
    assert only.structure.pdb_path == full.structure.pdb_path
    assert only.structure.method == full.structure.method
    # ... and the SAME reference-ligand atom ids + coords (the docking reference)
    assert ([(a.id, a.element, a.coord) for a in only.ligand.atoms]
            == [(a.id, a.element, a.coord) for a in full.ligand.atoms])


def test_default_is_full_parse_unchanged(tmp_path):
    """Default (no structure_only kwarg) is byte-identical to before: full
    ensemble + populated per-residue pLDDT."""
    preds = tmp_path / "boltz_results_chunk" / "predictions"
    _fake_batch_rep(preds, "hom_000_boltz_input", 2)
    cx = parse_prediction_dir(preds / "hom_000_boltz_input", "AGS", _ligand(),
                              "boltz2")
    assert cx is not None
    assert len(cx.samples) == 2
    assert any(r.plddt > 0 for r in cx.structure.residues)  # full path wires pLDDT


# --------------------------------------------------------------------------- #
# (B) PER-STEM CHUNK-SKIP: re-fold ONLY the incomplete stem on resume
# --------------------------------------------------------------------------- #
def _write_batch_yaml(in_dir: Path, stem: str):
    """A minimal Boltz input YAML for one stem (single-sequence, no MSA)."""
    in_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / f"{stem}.yaml").write_text(
        "version: 1\n"
        "sequences:\n"
        "- protein:\n"
        "    id: A\n"
        "    sequence: AGS\n"
        "    msa: empty\n"
        "- ligand:\n"
        "    id: B\n"
        "    smiles: OPC\n"
    )


def _install_fake_boltz_run(monkeypatch, folded_log: list):
    """Patch boltz.run to SIMULATE a Boltz subprocess: record which stems it was
    asked to fold (the YAML stems in the input dir = cmd[2]) and emit the
    expected predictions/<stem>/<stem>_model_k.pdb under --out_dir so the caller's
    merge step succeeds. Also no-op boltz.require (binary not on PATH in tests)."""
    monkeypatch.setattr(boltz, "require", lambda _exe: _exe)

    def _fake_run(cmd, *, dry_run=False, timeout=None, env=None, **kw):
        cmd = [str(c) for c in cmd]
        in_path = Path(cmd[2])
        out_dir = Path(cmd[cmd.index("--out_dir") + 1])
        # number of diffusion samples Boltz was told to draw
        try:
            n = int(cmd[cmd.index("--diffusion_samples") + 1])
        except ValueError:
            n = 1
        stems = sorted(p.stem for p in in_path.glob("*.yaml"))
        folded_log.append(list(stems))           # WHICH stems were folded
        preds = out_dir / f"boltz_results_{in_path.name}" / "predictions"
        for stem in stems:
            _fake_batch_rep(preds, stem, n)
        from evoliez.utils.subprocess_utils import RunResult
        return RunResult(cmd=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(boltz, "run", _fake_run)


def test_partial_resume_refolds_only_incomplete_stem(tmp_path, monkeypatch):
    """3 stems, 2 already complete on disk, 1 missing -> Boltz is asked to fold
    ONLY the missing stem, and every stem is present in the returned results."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    stems = ["hom_000_boltz_input", "hom_001_boltz_input", "hom_002_boltz_input"]
    for s in stems:
        _write_batch_yaml(in_dir, s)

    cfg = ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False)

    # Pre-seed the canonical results tree with 2 of 3 stems already complete.
    results_dir = out_dir / f"boltz_results_{in_dir.name}"
    preds = results_dir / "predictions"
    _fake_batch_rep(preds, "hom_000_boltz_input", 2)
    _fake_batch_rep(preds, "hom_001_boltz_input", 2)
    # hom_002 is MISSING (no predictions/<stem>/ subdir).

    got = predict_batch(in_dir, out_dir, cfg, seed=42, gpu_device=None)
    assert got == results_dir

    # Boltz ran exactly once, over exactly the one incomplete stem.
    assert folded_log == [["hom_002_boltz_input"]], (
        f"expected only the missing stem re-folded, got {folded_log}")

    # All three stems are parseable from the (unchanged-name) results dir.
    for s in stems:
        cx = parse_prediction_dir(preds / s, "AGS", _ligand(), "boltz2")
        assert cx is not None and len(cx.samples) == 2, f"{s} missing/incomplete"

    # The two pre-existing stems were NOT touched (no scratch dirs left behind).
    assert not (out_dir / f"{in_dir.name}_resume").exists()
    assert not (out_dir / f"_resume_out_{in_dir.name}").exists()


def test_partial_resume_underfilled_stem_is_refolded(tmp_path, monkeypatch):
    """A stem with FEWER than diffusion_samples models counts as incomplete and
    is re-folded (a half-finished stem is not mistaken for done)."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    for s in ("hom_000_boltz_input", "hom_001_boltz_input"):
        _write_batch_yaml(in_dir, s)

    cfg = ComplexPredictionConfig(diffusion_samples=3, use_msa_server=False)
    preds = (out_dir / f"boltz_results_{in_dir.name}" / "predictions")
    _fake_batch_rep(preds, "hom_000_boltz_input", 3)   # complete (3/3)
    _fake_batch_rep(preds, "hom_001_boltz_input", 1)   # UNDERFILLED (1/3)

    predict_batch(in_dir, out_dir, cfg, seed=7, gpu_device=None)
    assert folded_log == [["hom_001_boltz_input"]], (
        f"only the underfilled stem should be re-folded, got {folded_log}")
    cx = parse_prediction_dir(preds / "hom_001_boltz_input", "AGS", _ligand(),
                              "boltz2")
    assert cx is not None and len(cx.samples) == 3  # now topped up to 3


def test_all_complete_skips_subprocess(tmp_path, monkeypatch):
    """Every stem already complete -> the Boltz subprocess is skipped entirely
    (mirrors the per-rep resume-skip)."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    for s in ("hom_000_boltz_input", "hom_001_boltz_input"):
        _write_batch_yaml(in_dir, s)

    cfg = ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False)
    preds = (out_dir / f"boltz_results_{in_dir.name}" / "predictions")
    _fake_batch_rep(preds, "hom_000_boltz_input", 2)
    _fake_batch_rep(preds, "hom_001_boltz_input", 2)

    predict_batch(in_dir, out_dir, cfg, seed=1, gpu_device=None)
    assert folded_log == [], "no stem should be folded when all are complete"


def test_full_launch_when_no_prior_output(tmp_path, monkeypatch):
    """Fresh chunk (no predictions tree yet) -> ONE launch over the whole in_dir
    (every stem at once), landing directly in boltz_results_<in_dir.name>/."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    stems = ["hom_000_boltz_input", "hom_001_boltz_input"]
    for s in stems:
        _write_batch_yaml(in_dir, s)

    cfg = ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False)
    results_dir = predict_batch(in_dir, out_dir, cfg, seed=3, gpu_device=None)

    # one launch, over the full in_dir (both stems together)
    assert folded_log == [stems]
    assert results_dir == out_dir / f"boltz_results_{in_dir.name}"
    for s in stems:
        cx = parse_prediction_dir(results_dir / "predictions" / s, "AGS",
                                  _ligand(), "boltz2")
        assert cx is not None and len(cx.samples) == 2
