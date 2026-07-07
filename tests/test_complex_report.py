"""s04 complex-prediction HTML report: render produces a self-contained page
with the 3D viewer + all confidence panels and a correct affinity conversion."""

from __future__ import annotations


def _stats():
    return {
        "name": "x", "best_model": 0, "n_models": 3,
        "metrics": {"confidence_score": 0.9, "ptm": 0.9, "iptm": 0.95,
                    "ligand_iptm": 0.95, "complex_plddt": 0.92,
                    "complex_iplddt": 0.93, "complex_pde": 0.5,
                    "complex_ipde": 0.6},
        "chains": [{"id": "A", "kind": "protein", "start": 0, "n": 3,
                    "plddt_mean": 90.0, "ptm": 0.9},
                   {"id": "B", "kind": "ligand", "start": 3, "n": 2,
                    "plddt_mean": 85.0, "ptm": 0.8}],
        "pair_iptm": {"0,0": 0.9, "0,1": 0.8, "1,0": 0.8, "1,1": 0.8},
        "chain_keys": ["0", "1"],
        "plddt": [90.0, 88.0, 92.0, 85.0, 80.0], "protein_len": 3,
        "plddt_bands": [{"label": "very high", "color": "#0053D6", "count": 3},
                        {"label": "confident", "color": "#65CBF3", "count": 0},
                        {"label": "low", "color": "#FFDB13", "count": 0},
                        {"label": "very low", "color": "#FF7D45", "count": 0}],
        "pae": [[0.0, 5.0], [5.0, 0.0]], "pae_factor": 1, "pae_max": 5.0,
        "chain_bounds": [0, 3, 5],
        "affinity": {"affinity_pred_value": 0.5,
                     "affinity_probability_binary": 0.6},
        "ensemble": [0.9, 0.89, 0.88],
        "pdb": "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 90.00      C\n",
    }


def test_complex_report_builds_valid_html():
    from evoliez.io.complex_report import build_complex_report_html

    html = build_complex_report_html(
        target_id="x", stats=_stats(), ligand_names=["CofactorZ"],
        conditions=[("backend", "mock")], generated="2026-01-01 00:00")
    assert "%%" not in html                              # every token filled
    for marker in ('3Dmol-min.js', 'id="viewer"', 'id="pdbdata"', 'plddtChart',
                   '"pae"', 'iptmChart', 'ensChart', 'Methods', 'Boltz-2'):
        assert marker in html, marker
    # affinity: IC50 = 10^0.5 = 3.16 uM, dG = (0.5-6)*1.364 = -7.5 kcal/mol
    assert "3.16" in html and "-7.5" in html
    assert "ALA A   1" in html                           # embedded structure
    # GENERIC: the caller's ligand name is used; no target-specific chemistry baked in
    assert "CofactorZ" in html
    for hard in ("NADP", "formate", "fdh"):
        assert hard not in html, hard


def test_complex_report_generic_without_ligand_names():
    """No ligand_names → falls back to chain labels; works for any target."""
    from evoliez.io.complex_report import build_complex_report_html

    html = build_complex_report_html(
        target_id="t", stats=_stats(), conditions=[], generated="x")
    assert "%%" not in html and "chain B" in html and "NADP" not in html
