"""Gate-stack verdict — the honest final claim for one candidate.

A protein-improvement claim is only as strong as the WEAKEST gate it passes. Rather
than collapse everything into one score, evaluate an ordered stack and report the
verdict + which gate decided it:

  1. structural stability       — MD ran + stayed stable (md_lite valid, not unstable)
  2. reference-state preservation — the design ligand kept a WT-like pose
     (pose_gate reference_like); a non-reference pose = an ALTERNATIVE-pose
     hypothesis, never a WT-improvement claim
  3. cofactor/co-substrate pose  — the same for the other functional ligands
  4. functional-geometry gain    — role-specific reactivity improved over WT
     (e.g. ΔNAC > 0); WT-like-or-worse = a valid binder with NO catalytic gain
  5. energetic confirmatory      — RBFE/ΔΔG_bind converged + favourable
  6. experimental                — always pending until wet-lab

Verdicts (worst decisive gate wins):
  rejected            — structural stability failed
  alternative_pose    — not on a WT-like pose (gate 2/3) -> alt-pose hypothesis
  candidate_no_gain   — WT-like + stable but no functional-geometry gain
  candidate_improved  — WT-like + stable + functional gain (needs confirm + experiment)
  confirmed_computational — + energetic gate confirms (still needs experiment)

This encodes the lesson from S340G: on the WT-anchored pose it is candidate_no_gain
(ΔNAC 0), NOT a catalytic lead; on the fresh-Boltz pose it is alternative_pose
(NADP 8 Å off WT), NOT a validated improvement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

REJECTED = "rejected"
ALTERNATIVE_POSE = "alternative_pose"
CANDIDATE_NO_GAIN = "candidate_no_gain"
CANDIDATE_IMPROVED = "candidate_improved"
CONFIRMED_COMPUTATIONAL = "confirmed_computational"


@dataclass
class GateStackResult:
    verdict: str
    deciding_gate: str
    gates: Dict[str, str] = field(default_factory=dict)   # gate -> pass/fail/na
    notes: List[str] = field(default_factory=list)
    experimental: str = "pending"

    def to_json(self) -> dict:
        return {"verdict": self.verdict, "deciding_gate": self.deciding_gate,
                "gates": self.gates, "experimental": self.experimental,
                "notes": self.notes}


def _num(x):
    return x if isinstance(x, (int, float)) else None


def evaluate_gate_stack(
    m: Dict, *, ddg_max: float = 2.5, require_pose_for_claim: bool = True,
) -> GateStackResult:
    """Evaluate the gate stack for one candidate's metrics dict.

    Expected keys (all optional; missing -> that gate is 'na'):
      md_lite_status ('valid'/...), md_instability (0/1), passed (bool),
      pose_gate {design_ligand:{status}, cosubstrate:{status}}, nac_status,
      nac_delta_vs_wt, rbfe_ddg_bind, rbfe_mode, ddg_fold.
    """
    gates: Dict[str, str] = {}
    notes: List[str] = []

    # 1) structural stability
    unstable = _num(m.get("md_instability"))
    mdl_status = m.get("md_lite_status")
    stable = (m.get("passed") is not False) and (unstable in (None, 0, 0.0)) \
        and (mdl_status in (None, "valid"))
    ddg = _num(m.get("ddg_fold"))
    if ddg is not None and ddg > ddg_max:
        stable = False
        notes.append(f"ddG_fold {ddg:.2f} > {ddg_max}")
    gates["structural_stability"] = "pass" if stable else "fail"
    if not stable:
        return GateStackResult(REJECTED, "structural_stability", gates, notes)

    # 2/3) reference-state pose preservation (design ligand + co-substrate)
    pg = m.get("pose_gate") or {}
    dl = (pg.get("design_ligand") or {}).get("status")
    co = (pg.get("cosubstrate") or {}).get("status")
    gates["design_pose"] = {None: "na"}.get(dl, "pass" if dl == "reference_like"
                                            else "fail")
    gates["cosubstrate_pose"] = ("na" if co is None
                                 else ("pass" if co == "reference_like" else "fail"))
    pose_ok = (dl == "reference_like") or (dl is None and not require_pose_for_claim)
    if dl is not None and dl != "reference_like":
        notes.append(f"design-ligand pose {dl} (not WT-like)")
        return GateStackResult(ALTERNATIVE_POSE, "design_pose", gates, notes)
    if dl is None and require_pose_for_claim:
        notes.append("no pose gate -> cannot assert WT-like binding")

    # 4) functional-geometry gain (role-specific reactivity over WT)
    dnac = _num(m.get("nac_delta_vs_wt"))
    nac_status = m.get("nac_status")
    nac_valid = (nac_status is None) or str(nac_status).startswith("valid")
    if dnac is None or not nac_valid:
        gates["functional_geometry"] = "na"
        notes.append("functional-geometry gain not measurable")
        return GateStackResult(CANDIDATE_NO_GAIN, "functional_geometry", gates, notes)
    if dnac > 0:
        gates["functional_geometry"] = "pass"
    else:
        gates["functional_geometry"] = "fail"
        notes.append(f"ΔNAC {dnac:+.3f} (no gain over WT)")
        return GateStackResult(CANDIDATE_NO_GAIN, "functional_geometry", gates, notes)

    # 5) energetic confirmatory (RBFE)
    rb = _num(m.get("rbfe_ddg_bind"))
    rb_mode = m.get("rbfe_mode")
    if rb is None or rb_mode == "failed_softcore_ti_nan":
        gates["energetic"] = "na"
        notes.append("RBFE not converged")
        return GateStackResult(CANDIDATE_IMPROVED, "functional_geometry", gates, notes)
    gates["energetic"] = "pass" if rb < 0 else "fail"
    verdict = CONFIRMED_COMPUTATIONAL if rb < 0 else CANDIDATE_IMPROVED
    if rb >= 0:
        notes.append(f"ΔΔG_bind {rb:+.2f} (not tighter than WT)")
    return GateStackResult(verdict, "energetic", gates, notes)
