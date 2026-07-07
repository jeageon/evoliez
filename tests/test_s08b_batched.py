"""s08b GPU-batched mutant Boltz (one model-load PER GPU, not per mutant).

s08b reuses s06b's batched-Boltz-per-GPU pattern for the top-N mutant set: ONE
``boltz predict <chunk_dir>`` per GPU (= one model load per GPU) then a parallel
per-mutant SCOPED parse, instead of one ``predict_complex`` (one model load) per
mutant. Real Boltz is server-only, so these tests fake the Boltz subprocess
(``boltz.run`` + ``boltz.require``) so the REAL batch primitives
(``write_batch_input`` / ``predict_batch`` / ``parse_prediction_dir``) run
end-to-end, and drive the two Phase workers DIRECTLY (no ProcessPool — spawn on
macOS would drop the monkeypatch in children; mirrors tests/test_s06b_batch.py).
Asserts:

  * Phase 1 writes every mutant YAML into the GPU's shared chunk IN_DIR and
    folds them in ONE batched launch per chunk (= one model load per GPU);
  * Phase 2 scoped-parses each mutant into a finalized Complex (samples +
    metrics) keyed by candidate_id, exactly as predict_complex would build it;
  * a mutant whose Boltz stem is MISSING parses to None -> the orchestration
    OMITS it (so s08b keeps the proxy Δ honestly — never fabricates a real Δ).

Fixtures reuse tests/test_boltz_structure_only.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evoliez.adapters import boltz
from evoliez.config import ComplexPredictionConfig
from evoliez.stages.s08b_mutant_boltz import (
    _parse_mutant_worker,
    _run_mutant_batch_chunk,
    _shared_msa_a3m,
)
from evoliez.stages.s06b_interaction_model import _lpt_partition
from evoliez.types import Ligand, LigandAtom


# --------------------------------------------------------------------------- #
# Shared fixtures (mirror tests/test_boltz_structure_only.py)
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
    d = predictions_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n_samples):
        _write_model_pdb(
            d / f"{stem}_model_{k}.pdb",
            [(7.0 + k, 8.0, 9.0), (7.4 + k, 8.4, 9.3), (7.8 + k, 8.8, 9.6)],
        )
        (d / f"confidence_{stem}_model_{k}.json").write_text(json.dumps(
            {"confidence_score": 0.9 - 0.05 * k, "complex_plddt": 0.8,
             "iptm": 0.7}))
        np.savez(d / f"plddt_{stem}_model_{k}.npz",
                 plddt=np.array([0.85, 0.78, 0.60]))


def _ligand():
    return Ligand(id="L", smiles="OPC",
                  atoms=[LigandAtom(id="O0", element="O", coord=(0.0, 0.0, 0.0)),
                         LigandAtom(id="P1", element="P", coord=(0.5, 0.0, 0.0)),
                         LigandAtom(id="C2", element="C", coord=(1.0, 0.0, 0.0))])


def _install_fake_boltz_run(monkeypatch, folded_log: list, *, skip_stems=()):
    """Patch boltz.run to SIMULATE a Boltz subprocess: record which stems each
    launch was asked to fold, and emit predictions/<stem>/<stem>_model_k.pdb under
    --out_dir so the per-stem parse succeeds. Any stem in ``skip_stems`` is NOT
    emitted (simulates a failed/skipped mutant prediction). require is a no-op."""
    monkeypatch.setattr(boltz, "require", lambda _exe: _exe)

    def _fake_run(cmd, *, dry_run=False, timeout=None, env=None, **kw):
        cmd = [str(c) for c in cmd]
        in_path = Path(cmd[2])
        out_dir = Path(cmd[cmd.index("--out_dir") + 1])
        try:
            n = int(cmd[cmd.index("--diffusion_samples") + 1])
        except ValueError:
            n = 1
        stems = sorted(p.stem for p in in_path.glob("*.yaml"))
        folded_log.append(list(stems))
        preds = out_dir / f"boltz_results_{in_path.name}" / "predictions"
        for stem in stems:
            if stem in skip_stems:
                continue
            _fake_batch_rep(preds, stem, n)
        from evoliez.utils.subprocess_utils import RunResult
        return RunResult(cmd=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(boltz, "run", _fake_run)


def _drive_two_phase(tmp_path, cand_ids, gpu_list, lig, cp, msa_path=None,
                     extra_ligands=None, seed=1234):
    """Run s08b's batched orchestration DIRECTLY (no ProcessPool): LPT-partition
    the mutants across GPUs exactly as _run_batched_mutants does, run Phase 1
    (_run_mutant_batch_chunk) then Phase 2 (_parse_mutant_worker) per chunk, and
    assemble {cand_id: Complex} dropping any stem that parsed to None. Returns
    (predicted, n_launches_per_chunk_list)."""
    muts_meta_all = [(cid, "AGS") for cid in cand_ids]
    msa_path = _shared_msa_a3m(msa_path, tmp_path)  # mirror _run_batched_mutants
    buckets = _lpt_partition(muts_meta_all, len(gpu_list),
                             weight=lambda m: len(m[1]) ** 2)
    chunks = [
        (g, buckets[gi], lig, cp, str(tmp_path), seed, extra_ligands, msa_path, None)
        for gi, g in enumerate(gpu_list)
        if buckets[gi]                       # mirror the prod skip of empty GPU buckets
    ]
    parse_jobs = []
    for ch in chunks:
        _gpu, results_dir, muts_in_chunk = _run_mutant_batch_chunk(ch)
        for (cid, seq) in muts_in_chunk:
            parse_jobs.append((cid, seq, lig, cp, results_dir))
    predicted = {}
    for pj in parse_jobs:
        cid, cx = _parse_mutant_worker(pj)
        if cx is not None:
            predicted[cid] = cx
    return predicted


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_empty_chunk_guard_skips_boltz(tmp_path):
    """REGRESSION (2-mutant/4-GPU smoke crash): with fewer mutants than GPUs, _lpt_partition yields
    empty buckets. An empty chunk must return cleanly WITHOUT launching `boltz predict` on a
    never-created _batch_in_gpuN dir (which errored 'Path does not exist' and killed the stage)."""
    from evoliez.stages.s08b_mutant_boltz import _run_mutant_batch_chunk
    cp = ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False)
    payload = ("3", [], _ligand(), cp, str(tmp_path), 1234, None, None, None)
    gpu, results_dir, muts = _run_mutant_batch_chunk(payload)
    assert gpu == "3" and results_dir == "" and muts == []
    assert not (tmp_path / "_batch_in_gpu3").exists()   # never created a dir to fold


def test_fewer_mutants_than_gpus_folds_all_no_crash(tmp_path, monkeypatch):
    """2 mutants over 4 GPUs: the 2 empty buckets are skipped, the 2 real mutants still fold."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)
    predicted = _drive_two_phase(
        tmp_path, ["mut_00001", "mut_00002"], ["0", "1", "2", "3"],
        _ligand(), ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False))
    assert set(predicted) == {"mut_00001", "mut_00002"}


def test_batched_mutants_one_launch_per_gpu_parses_all(tmp_path, monkeypatch):
    """4 mutants over 2 GPUs -> 2 Boltz launches (one per GPU, chunked via the
    LPT partition), every mutant parsed into a finalized Complex keyed by id."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)
    cand_ids = ["mut_00001", "mut_00002", "mut_00003", "mut_00004"]
    cp = ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False)

    predicted = _drive_two_phase(tmp_path, cand_ids, ["0", "1"], _ligand(), cp)

    # One launch PER GPU (= per chunk), NOT one per mutant.
    assert len(folded_log) == 2, f"expected 2 per-GPU launches, got {folded_log}"
    # Every mutant parsed; full Complex (samples + metrics) like predict_complex.
    assert set(predicted) == set(cand_ids)
    for cid in cand_ids:
        cx = predicted[cid]
        assert cx is not None
        assert len(cx.samples) == 2, f"{cid} missing diffusion ensemble"
        assert cx.metrics.get("confidence_score"), f"{cid} missing metrics"
        assert cx.structure.residues and cx.ligand.atoms


def test_batched_mutants_single_gpu_one_chunk(tmp_path, monkeypatch):
    """All mutants on ONE GPU -> a single batched launch over the whole set."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)
    cand_ids = ["mut_00001", "mut_00002", "mut_00003"]
    cp = ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False)

    predicted = _drive_two_phase(tmp_path, cand_ids, ["2"], _ligand(), cp)

    assert len(folded_log) == 1, "single GPU should be one batched launch"
    assert sorted(folded_log[0]) == sorted(f"{c}_boltz_input" for c in cand_ids)
    assert set(predicted) == set(cand_ids)


def test_batched_mutant_missing_prediction_is_omitted(tmp_path, monkeypatch):
    """A mutant whose Boltz stem fails/skips parses to None and is OMITTED from
    the result dict, so s08b keeps that candidate's proxy Δ (never fabricated)."""
    folded_log: list = []
    # mut_00002's stem is never emitted -> a failed prediction.
    _install_fake_boltz_run(monkeypatch, folded_log,
                            skip_stems=("mut_00002_boltz_input",))
    cand_ids = ["mut_00001", "mut_00002", "mut_00003"]
    cp = ComplexPredictionConfig(diffusion_samples=2, use_msa_server=False)

    predicted = _drive_two_phase(tmp_path, cand_ids, ["0"], _ligand(), cp)

    assert "mut_00002" not in predicted, "failed mutant must not be fabricated"
    assert {"mut_00001", "mut_00003"} <= set(predicted)


def test_phase1_writes_mutant_yamls_into_chunk_in_dir(tmp_path, monkeypatch):
    """Phase 1 writes one <cand_id>_boltz_input.yaml per mutant into the GPU's
    shared chunk IN_DIR (the per-stem layout predict_batch/parse rely on)."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)
    cp = ComplexPredictionConfig(diffusion_samples=1, use_msa_server=False)
    chunk = ("0", [("mut_00001", "AGS"), ("mut_00002", "AGS")],
             _ligand(), cp, str(tmp_path), 1234, None, None, None)

    gpu, results_dir, muts = _run_mutant_batch_chunk(chunk)

    in_dir = tmp_path / "_batch_in_gpu0"
    assert (in_dir / "mut_00001_boltz_input.yaml").exists()
    assert (in_dir / "mut_00002_boltz_input.yaml").exists()
    # IN_DIR and OUT_DIR are distinct (Boltz errors if OUT nests under IN).
    assert Path(results_dir).is_relative_to(tmp_path / "_batch_out_gpu0")


def test_batched_fasta_msa_not_left_in_chunk_dir(tmp_path, monkeypatch):
    """A FASTA WT MSA (what s03 writes) must be pre-converted to an a3m that lives
    OUTSIDE the chunk IN_DIR. Left as FASTA, write_batch_input rewrites a stray
    <label>_msa.a3m INTO the chunk dir, and ``boltz predict <chunk_dir>`` globs it
    as an input and aborts the chunk ("Unable to parse filetype .a3m"). Regression:
    the chunk IN_DIR must hold ONLY <cand>_boltz_input.yaml — no stray .a3m."""
    folded_log: list = []
    _install_fake_boltz_run(monkeypatch, folded_log)
    cp = ComplexPredictionConfig(diffusion_samples=1, use_msa_server=False)
    # s03-style aligned FASTA MSA (equal-length, '-' gaps -> a3m-rewritable).
    fasta = tmp_path / "wt_msa.fasta"
    fasta.write_text(">wt\nAGS\n>hom1\nAG-\n>hom2\nA-S\n")

    shared = _shared_msa_a3m(fasta, tmp_path)
    # Converted to an a3m that lives NEXT TO (not inside) the chunk dirs.
    assert shared is not None and Path(shared).suffix == ".a3m"
    assert Path(shared).parent == tmp_path

    chunk = ("0", [("mut_00001", "AGS"), ("mut_00002", "AGS")],
             _ligand(), cp, str(tmp_path), 1234, None, shared, None)
    _gpu, results_dir, _muts = _run_mutant_batch_chunk(chunk)

    in_dir = tmp_path / "_batch_in_gpu0"
    yamls = sorted(p.name for p in in_dir.glob("*.yaml"))
    strays = sorted(p.name for p in in_dir.iterdir() if p.suffix != ".yaml")
    assert yamls == ["mut_00001_boltz_input.yaml", "mut_00002_boltz_input.yaml"]
    assert strays == [], f"chunk IN_DIR must hold only YAMLs; stray inputs: {strays}"
    # The mutant YAML references the shared a3m BY PATH (outside the chunk dir).
    assert str(shared) in (in_dir / "mut_00001_boltz_input.yaml").read_text()
