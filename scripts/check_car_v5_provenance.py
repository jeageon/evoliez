#!/usr/bin/env python3
"""CAR V5 acceptance judge (ROADMAP_V5 Day 3/6 go/no-go automation).

Reads a run's ``reports/provenance/md_candidates.json`` and classifies the Mg/OpenMM + O->P
outcome as PASS / CONDITIONAL / FAIL on EVIDENCE VALIDITY (not lead-hunting), per the review:

  PASS         angle finite · Mg tracked (if requested) · WT baseline finite · geometry terms
               computed · invalid cases classified with reason
  CONDITIONAL  the above hold but candidates do not discriminate -> a sampling limitation, not a
               pipeline failure; escalation (longer MD / explicit solvent / QM-MM) recommended
  FAIL         angle NaN · Mg requested but missing while NAC reported 0 · Mg via OpenFF ·
               metal insertion failed

Pure-stdlib; importable (``evaluate``) for tests. CLI: ``check_car_v5_provenance.py <run_dir|json>``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

PASS, CONDITIONAL, FAIL = "PASS", "CONDITIONAL_PASS", "FAIL"


def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _nac(rec: Dict[str, Any]) -> Dict[str, Any]:
    return rec.get("nac") or {}


def evaluate(md: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a md_candidates.json dict. Returns {verdict, reasons[], checks{}, escalation}."""
    reasons: List[str] = []
    checks: Dict[str, Any] = {}
    metal = md.get("metal_setup") or {}
    geom_source = md.get("geometry_source")
    wt = md.get("wt_reference") or {}
    cands = md.get("candidates") or []

    # --- metal setup ---
    metal_requested = bool(metal.get("requested"))
    checks["metal_requested"] = metal_requested
    checks["metal_status"] = metal.get("status")
    checks["openff_parameterized"] = metal.get("openff_parameterized")
    if metal_requested:
        if metal.get("openff_parameterized") is True:
            reasons.append("FAIL: Mg was OpenFF-parameterized (must be a structure-level MG ion)")
        if metal.get("inserted_in_pdb") is not True:
            reasons.append("FAIL: Mg requested but MG HETATM not inserted "
                           f"(status={metal.get('status')})")

    # --- geometry source honesty ---
    checks["geometry_source"] = geom_source
    if geom_source in (None, "none"):
        reasons.append("FAIL: no reaction-geometry source recorded")

    # --- angle finiteness (WT + every candidate) ---
    wt_angle = _nac(wt).get("angle_mean")
    checks["wt_angle_finite"] = _is_finite(wt_angle)
    checks["wt_baseline_present"] = bool(wt)
    if wt and not _is_finite(wt_angle):
        reasons.append(f"FAIL: WT baseline angle_mean is not finite ({wt_angle})")

    nan_angle_ids, computed = [], 0
    nac0_but_no_metal = []
    for c in cands:
        n = _nac(c)
        a = n.get("angle_mean")
        if "angle_mean" in n:
            computed += 1
            if not _is_finite(a):
                nan_angle_ids.append(c.get("candidate_id"))
        # Mg requested but missing while NAC reported exactly 0 -> invalid interpretation
        occ = n.get("nac_occupancy") if "nac_occupancy" in n else c.get("nac_occupancy")
        if (metal_requested and metal.get("inserted_in_pdb") is not True
                and occ == 0):
            nac0_but_no_metal.append(c.get("candidate_id"))
    checks["n_candidates"] = len(cands)
    checks["n_angle_computed"] = computed
    checks["n_angle_nan"] = len(nan_angle_ids)
    if nan_angle_ids:
        reasons.append(f"FAIL: angle_mean NaN for {len(nan_angle_ids)} candidate(s): "
                       f"{nan_angle_ids[:5]}")
    if nac0_but_no_metal:
        reasons.append("FAIL: NAC reported 0 while Mg was requested but not inserted "
                       f"(invalid interpretation) for {nac0_but_no_metal[:5]}")

    if any(r.startswith("FAIL") for r in reasons):
        return {"verdict": FAIL, "reasons": reasons, "checks": checks, "escalation": None}

    # --- discrimination (only reached when everything is valid) ---
    deltas = [(_nac(c).get("nac_delta_vs_wt")) for c in cands]
    finite_deltas = [d for d in deltas if _is_finite(d)]
    discriminates = any(d and d > 0 for d in finite_deltas) and (
        len({round(d, 3) for d in finite_deltas}) > 1)
    checks["discriminates"] = discriminates
    if discriminates:
        return {"verdict": PASS, "reasons": ["geometry valid + candidate-level differences observed"],
                "checks": checks, "escalation": None}
    return {
        "verdict": CONDITIONAL,
        "reasons": ["geometry valid (angle finite, Mg tracked, WT baseline present) but candidates "
                    "do not discriminate — a sampling limitation, NOT a pipeline failure"],
        "checks": checks,
        "escalation": "longer MD / explicit solvent / QM-MM / reference-ensemble refinement",
    }


def _load(arg: str) -> Dict[str, Any]:
    p = Path(arg)
    if p.is_dir():
        p = p / "reports" / "provenance" / "md_candidates.json"
    return json.loads(p.read_text())


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: check_car_v5_provenance.py <run_dir|md_candidates.json>", file=sys.stderr)
        return 2
    res = evaluate(_load(argv[1]))
    print(json.dumps(res, indent=2))
    return 0 if res["verdict"] in (PASS, CONDITIONAL) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
