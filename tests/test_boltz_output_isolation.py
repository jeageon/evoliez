"""Server-confirmed bug: when multiple `predict_complex(label=mut_NNNNN)`
calls share the same `--out_dir` (s08b runs per-mutant Boltz for several
candidates under `complexes/mutant_boltz/`), the unscoped
`outdir.glob("boltz_results_*")` returned EVERY mutant's directory and
the function picked the alphabetically-first one for every candidate.

Result: `mut_complexes[mut_00015].structure.pdb_path` actually pointed at
mut_00014's PDB. The sequence guard in `_run_real` then fired for the
others ("1 residue(s) differ"), silently leaving 3/4 candidates with
proxy WT structure even after s08b ran on all of them.

Fix: scope the glob to `boltz_results_{label}*`. These tests lock that
invariant in source so a refactor can't silently regress.
"""

from __future__ import annotations

import inspect

from evoliez.adapters import boltz


def test_predict_real_scopes_output_glob_to_label():
    src = inspect.getsource(boltz._predict_real)
    # The label-scoped glob pattern must appear in the discovery section.
    # The candidate's results live under boltz_results_<label>_boltz_input/
    # or, for older Boltz CLI flags, boltz_results_<label>*.
    assert "f\"boltz_results_{label}_boltz_input\"" in src or \
           'f"boltz_results_{label}_boltz_input"' in src
    assert 'f"boltz_results_{label}*"' in src
    # Unscoped glob ONLY appears as last-resort fallback with a loud
    # warning, not as the primary discovery path.
    fallback_section = src.split("Last-resort")[-1] if "Last-resort" in src \
        else src.split("falling back to wide glob")[-1]
    assert 'outdir.glob("boltz_results_*")' in fallback_section


def test_parse_real_samples_uses_label_scope_not_outdir():
    src = inspect.getsource(boltz._predict_real)
    # _parse_real_samples receives a SCOPED root, not the unscoped outdir,
    # so confidence*.json / affinity*.json discovery can't cross-contaminate
    # between mutants sharing complexes/mutant_boltz/.
    assert "_parse_real_samples(sample_scope" in src
    assert "sample_scope = roots[0]" in src
