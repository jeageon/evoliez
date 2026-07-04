"""ROADMAP_V5 — the CAR V5 acceptance judge (scripts/check_car_v5_provenance.py) classifies
Mg/OpenMM + O->P provenance as PASS / CONDITIONAL / FAIL on evidence validity."""
import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_car_v5_provenance",
    Path(__file__).resolve().parents[1] / "scripts" / "check_car_v5_provenance.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
NAN = float("nan")


def _md(metal=None, geom="legacy_reactive_geometry_from_mechanism_spec",
        wt_angle=178.0, cands=None):
    return {
        "metal_setup": metal if metal is not None else {
            "requested": True, "inserted_in_pdb": True, "openff_parameterized": False,
            "amber_standard_ion": "server_verified_pending", "status": "valid_metal_setup"},
        "geometry_source": geom,
        "wt_reference": {"nac": {"angle_mean": wt_angle}},
        "candidates": cands if cands is not None else [],
    }


def _c(cid, angle=175.0, delta=0.0, occ=0.5):
    return {"candidate_id": cid,
            "nac": {"angle_mean": angle, "nac_delta_vs_wt": delta, "nac_occupancy": occ}}


def test_pass_when_valid_and_discriminating():
    md = _md(cands=[_c("m1", delta=0.12), _c("m2", delta=0.0), _c("m3", delta=-0.05)])
    r = mod.evaluate(md)
    assert r["verdict"] == mod.PASS


def test_conditional_when_valid_but_no_discrimination():
    md = _md(cands=[_c("m1", delta=0.0), _c("m2", delta=0.0), _c("m3", delta=0.0)])
    r = mod.evaluate(md)
    assert r["verdict"] == mod.CONDITIONAL
    assert r["escalation"]                       # an escalation path is offered


def test_fail_on_nan_angle():
    md = _md(cands=[_c("m1", angle=NAN), _c("m2")])
    r = mod.evaluate(md)
    assert r["verdict"] == mod.FAIL
    assert any("NaN" in x for x in r["reasons"])


def test_fail_when_wt_baseline_nan():
    md = _md(wt_angle=NAN, cands=[_c("m1")])
    assert mod.evaluate(md)["verdict"] == mod.FAIL


def test_fail_when_mg_openff_parameterized():
    md = _md(metal={"requested": True, "inserted_in_pdb": True,
                    "openff_parameterized": True, "status": "valid_metal_setup"},
             cands=[_c("m1")])
    r = mod.evaluate(md)
    assert r["verdict"] == mod.FAIL
    assert any("OpenFF" in x for x in r["reasons"])


def test_fail_when_mg_missing_but_nac_zero():
    md = _md(metal={"requested": True, "inserted_in_pdb": False,
                    "openff_parameterized": False, "status": "skipped_invalid_metal_setup"},
             cands=[_c("m1", occ=0)])
    r = mod.evaluate(md)
    assert r["verdict"] == mod.FAIL


def test_no_metal_requested_is_not_a_metal_failure():
    # FDH-style run with no Mg: metal not requested -> metal checks don't fail it
    md = _md(metal={"requested": False, "inserted_in_pdb": False, "openff_parameterized": False},
             geom="legacy_reactive_geometry", cands=[_c("m1", delta=0.1)])
    assert mod.evaluate(md)["verdict"] in (mod.PASS, mod.CONDITIONAL)
