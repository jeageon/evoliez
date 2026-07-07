"""s10 MD validation report builder (evoliez.io.md_report).

Reads only reports/provenance/md_candidates.json (+ _state.json meta), so the test
synthesises that on-disk record and checks the paper-honesty surfaces: the
"what actually ran" panel (short/implicit, not 2 ns explicit), the skipped-is-NOT-
validated framing, the NAC/ΔNAC section, MM-GBSA ΔG, and NAC-off omission.
"""
import json

from evoliez.io.md_report import write_md_report


def _cand(m, st, mdl, passed, occ, dnac, dg):
    return {
        "mutation_string": m, "status": st, "solvent_mode": "implicit",
        "simulation_time_ns": 0.05, "protocol_level": 2,
        "md_lite_score": mdl, "passed": passed, "ligand_rmsd_mean": 1.8,
        "pocket_rmsd_mean": 1.1, "hbond_occupancy": 0.6,
        "nac_occupancy": occ, "nac_delta_vs_wt": dnac,
        "nac": {"distance_min": 2.9, "angle_mean": 162.0} if occ is not None else {},
        "binding_dg": dg, "failure_reasons": [],
    }


def _payload(nac_enabled=True):
    return {
        "nac_enabled": nac_enabled, "wt_nac_occupancy": 0.42,
        "wt_reference": {"candidate_id": "_wt_reference", "status": "ok",
                         "solvent_mode": "implicit", "simulation_time_ns": 0.05,
                         "nac_occupancy": 0.42},
        "requested": {"protocol_level": 2, "solvent": "explicit",
                      "production_ns": 2.0, "engine": "openmm"},
        "counts": {"n_real_ran": 3, "n_passed": 2, "n_skipped": 1,
                   "n_failed": 0, "n_total": 4},
        "candidates": [
            _cand("D227Q", "ok", 0.71, True, 0.55, 0.13, {"gbsa": -28.4}),
            _cand("L229H", "unstable", 0.33, False, 0.30, -0.12, {}),
            _cand("A198G", "ok", 0.50, True, 0.48, 0.06, {}),
            {"mutation_string": "C145S", "status": "skipped_parameterization",
             "solvent_mode": "implicit", "md_lite_score": 0.0, "passed": True,
             "nac_occupancy": None, "nac_delta_vs_wt": None, "nac": {},
             "binding_dg": {}, "failure_reasons": ["cofactor FF unsupported"]},
        ],
    }


def _write_run(tmp_path, payload):
    prov = tmp_path / "reports" / "provenance"
    prov.mkdir(parents=True)
    (tmp_path / "_state.json").write_text(json.dumps(
        {"meta": {"catalytic_positions": [290, 338],
                  "designable_positions": [103, 127]}}))
    (prov / "md_candidates.json").write_text(json.dumps(payload))


def test_md_report_paper_honesty_surfaces(tmp_path):
    _write_run(tmp_path, _payload(nac_enabled=True))
    html = write_md_report(tmp_path).read_text()
    # honesty: short implicit screen, not the requested 2 ns explicit
    assert "What actually ran" in html
    assert "SHORT implicit-solvent SCREEN" in html        # protocol_level<3 cap
    assert "explicit was requested" in html               # solvent mismatch flagged
    # skip is NOT validated
    assert "skipped (NOT validated)" in html
    # NAC / catalytic power
    assert "near-attack-conformation" in html
    assert "WT NAC occupancy" in html
    assert "ΔNAC" in html
    # MM-GBSA binding ΔG carried through
    assert "gbsa:-28.4" in html
    # per-candidate rows present
    assert "D227Q" in html and "C145S" in html


def test_md_report_omits_nac_when_disabled(tmp_path):
    _write_run(tmp_path, _payload(nac_enabled=False))
    html = write_md_report(tmp_path).read_text()
    assert "near-attack-conformation" not in html
    assert "WT NAC occupancy" not in html
    assert "Per-candidate MD metrics" in html             # binding report still there
