"""Server-observed: per-mutant Boltz takes ~20 min each, dominated by
`--use_msa_server` HTTP round-trips (~18 min / mutant); pure inference
is ~2 min. Same enzyme's point-mutant variants share the same MSA
(homologs were retrieved using WT as query - the mutation doesn't
change which sequences are returned). Two surgical fixes:

1. `predict_complex(..., reuse_existing=True)` - new default: if a
   prior `boltz_results_<label>_boltz_input/` already has output, parse
   it directly instead of re-invoking `boltz predict`. Lets a partial
   production run resume without re-doing the ~20-min/mutant work.

2. s08b passes the WT MSA path to predict_complex. The boltz adapter
   then embeds it in the YAML and skips `--use_msa_server`, eliminating
   the per-mutant HTTP fetch.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from evoliez.adapters import boltz as boltz_adapter
from evoliez.adapters.boltz import _load_existing_real, predict_complex
from evoliez.config import Backend, ComplexPredictionConfig
from evoliez.stages import s08b_mutant_boltz
from evoliez.types import Ligand


def _fake_real_pdb(outdir: Path, label: str, sequence: str = "ACDE"):
    """Write a minimal boltz_results_<label>_boltz_input/ on disk."""
    root = outdir / f"boltz_results_{label}_boltz_input"
    pred = root / "predictions" / f"{label}_boltz_input"
    pred.mkdir(parents=True, exist_ok=True)
    # CA-only PDB; just enough that _parse_real_structure picks up residues.
    lines = []
    for i, aa in enumerate(sequence, start=1):
        aa3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU"}.get(aa, "ALA")
        lines.append(
            f"ATOM  {i:>5}  CA  {aa3} A{i:>4}    "
            f"{i*1.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00 50.00           C"
        )
    lines.append("END")
    pdb = pred / f"{label}_boltz_input_model_1.pdb"
    pdb.write_text("\n".join(lines) + "\n")
    return root, pdb


# --------------------------------------------------------------------------- #
# Fix 1: predict_complex reuses existing on-disk output
# --------------------------------------------------------------------------- #
def test_predict_complex_reuses_existing_real_output(tmp_path):
    cfg = ComplexPredictionConfig(use_kernels=False)
    lig = Ligand(id="L", smiles="CCO")
    _fake_real_pdb(tmp_path, "wt", "AAACDEAAAC")

    # Ensure _predict_real is NEVER called when cache is present.
    with patch.object(boltz_adapter, "_predict_real",
                      side_effect=AssertionError("must not be called")):
        cx = predict_complex(
            "wt", "AAACDEAAAC", lig, cfg, tmp_path,
            backend=Backend.real, dry_run=False, reuse_existing=True,
        )
    assert cx.method == cfg.primary_method
    # Path points into the label-scoped results dir, not someone else's.
    assert "boltz_results_wt_boltz_input" in str(cx.path)


def test_predict_complex_skips_cache_when_disabled(tmp_path):
    cfg = ComplexPredictionConfig(use_kernels=False)
    lig = Ligand(id="L", smiles="CCO")
    _fake_real_pdb(tmp_path, "wt")

    # With reuse_existing=False, _predict_real MUST be called.
    called = {"n": 0}
    def fake_real(*a, **kw):
        called["n"] += 1
        from evoliez.types import Complex, ProteinStructure
        return Complex(structure=ProteinStructure(sequence=""), ligand=lig,
                       method=cfg.primary_method)
    with patch.object(boltz_adapter, "_predict_real", side_effect=fake_real):
        predict_complex(
            "wt", "ACDE", lig, cfg, tmp_path,
            backend=Backend.real, dry_run=False, reuse_existing=False,
        )
    assert called["n"] == 1


def test_load_existing_real_returns_none_when_no_output(tmp_path):
    cfg = ComplexPredictionConfig(use_kernels=False)
    lig = Ligand(id="L", smiles="CCO")
    # No boltz_results_*_boltz_input dir on disk -> None.
    assert _load_existing_real("wt", "ACDE", lig, cfg, tmp_path) is None


def test_load_existing_real_label_scoped_isolation(tmp_path):
    # Multiple per-mutant results sharing the same outdir: the cache load
    # for "mut_A" must NOT cross-contaminate from "mut_B"'s PDB.
    cfg = ComplexPredictionConfig(use_kernels=False)
    lig = Ligand(id="L", smiles="CCO")
    _fake_real_pdb(tmp_path, "mut_A", "AAACDEAAAA")
    _fake_real_pdb(tmp_path, "mut_B", "CCCCDECCCC")
    cx_a = _load_existing_real("mut_A", "AAACDEAAAA", lig, cfg, tmp_path)
    cx_b = _load_existing_real("mut_B", "CCCCDECCCC", lig, cfg, tmp_path)
    assert cx_a is not None and cx_b is not None
    assert "mut_A" in str(cx_a.path)
    assert "mut_B" in str(cx_b.path)
    assert cx_a.path != cx_b.path


# --------------------------------------------------------------------------- #
# Fix 2: s08b passes WT MSA to per-mutant predict_complex
# --------------------------------------------------------------------------- #
def test_s08b_passes_wt_msa_to_predict_complex():
    src = inspect.getsource(s08b_mutant_boltz.MutantBoltzStage.run)
    # WT MSA path is resolved and threaded into the per-mutant call.
    assert 'ctx.paths.msa / "alignment.fasta"' in src
    assert "wt_msa" in src
    assert "msa_path=wt_msa" in src
    # Comment names the wall-time motivation so this can't quietly regress.
    assert "use_msa_server" in src
