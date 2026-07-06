#!/usr/bin/env python
"""E2 (reviewer Day 2-3) — build a CRYSTAL-GROUNDED active-state reference ensemble for the CAR
adenylation reaction from the SrCAR structures (Gahloth et al.): 5MST = AMP + fumarate (carboxylate
substrate analog) + divalent metal = the PRODUCTIVE Michaelis-like state; 5MSW = AMP-only.

E1 showed explicit solvent fixes gross diffusion but the pose relaxes to a non-productive ~5 A O->P.
The diagnosis moved to "the reference/initial pose is not in the true active-state basin". This script
measures the REAL productive geometry from the crystal active sites — the substrate-carboxylate-O to
AMP-Palpha distance and the in-line O_nuc-Palpha-O_leaving angle — and builds a ReferenceEnsemble from
it (NOT from the relaxed MD, NOT from hardcoded fixtures). That ensemble is the productive anchor E2
starts candidates from.

Read-only, CPU (gemmi). Run on the server where car/refs/*.cif live:
  PYTHONPATH=src python scripts/e2_build_reference_ensemble.py
Writes runs/car_v5_e2/reference_ensemble.json + prints the extracted geometry.
"""
import json
import math
import os
import sys

REFS = os.environ.get("CAR_REFS", "car/refs")
# carboxylate O atom names on fumarate; phosphate atoms on AMP
_FUM_CARBOXYL_O = {"O7", "O8", "OXT", "O"}
_AMP_P = "P"
_AMP_PHOS_O = {"O1P", "O2P", "O3P"}
_METALS = {"CA", "MG", "MN", "ZN"}


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _angle(a, b, c):
    """angle a-b-c in degrees (b is the vertex)."""
    v1 = [a[i] - b[i] for i in range(3)]
    v2 = [c[i] - b[i] for i in range(3)]
    n1 = math.sqrt(sum(x * x for x in v1)) or 1e-9
    n2 = math.sqrt(sum(x * x for x in v2)) or 1e-9
    cos = sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _collect(structure):
    """Return lists of (resname, atomname, chain, xyz) for FUM-O, AMP-P, AMP-O, metals."""
    fum_o, amp_p, amp_o, metals = [], [], [], []
    for model in structure:
        for chain in model:
            for res in chain:
                rn = res.name
                for at in res:
                    xyz = (at.pos.x, at.pos.y, at.pos.z)
                    nm = at.name
                    if rn == "FUM" and nm in _FUM_CARBOXYL_O:
                        fum_o.append((chain.name, nm, xyz))
                    elif rn == "AMP" and nm == _AMP_P:
                        amp_p.append((chain.name, nm, xyz))
                    elif rn == "AMP" and nm in _AMP_PHOS_O:
                        amp_o.append((chain.name, nm, xyz))
                    elif rn in _METALS and rn == nm:
                        metals.append((chain.name, rn, xyz))
    return fum_o, amp_p, amp_o, metals


def extract_productive_geometry(cif_path, max_reactive_A=4.5):
    """For each AMP Palpha, find the nearest fumarate carboxylate O (the nucleophile) within
    max_reactive_A; that is a productive/near-productive reactive pair. Return per-pair geometry."""
    import gemmi
    st = gemmi.read_structure(cif_path)
    fum_o, amp_p, amp_o, metals = _collect(st)
    obs = []
    for (pch, _pn, ppos) in amp_p:
        # nucleophile = nearest fumarate carboxylate O to this Palpha
        near = sorted(((_dist(ppos, o[2]), o) for o in fum_o), key=lambda t: t[0])
        if not near or near[0][0] > max_reactive_A:
            continue
        d_nuc, o_nuc = near[0]
        # leaving O = the AMP phosphate O most in-line (largest O_nuc-P-O angle), same P
        p_os = [o for o in amp_o if o[0] == pch]
        best_ang, best_leav = None, None
        for o_leav in p_os:
            ang = _angle(o_nuc[2], ppos, o_leav[2])
            if best_ang is None or ang > best_ang:
                best_ang, best_leav = ang, o_leav
        # nearest metal to the nucleophile O (bridging cation), if any
        m_near = min((_dist(o_nuc[2], m[2]) for m in metals), default=None)
        obs.append({
            "chain": pch, "distance_A": round(d_nuc, 3),
            "angle_deg": round(best_ang, 2) if best_ang is not None else None,
            "nuc_atom": o_nuc[1], "leaving_atom": best_leav[1] if best_leav else None,
            "metal_to_nuc_A": round(m_near, 2) if m_near is not None else None,
        })
    return obs


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/car_v5_e2"
    os.makedirs(out_dir, exist_ok=True)
    all_obs = []
    for pdb in ("5MST", "5MSW"):
        p = os.path.join(REFS, f"{pdb}.cif")
        if not os.path.exists(p):
            print(f"  (skip {pdb}: not found at {p})")
            continue
        obs = extract_productive_geometry(p)
        print(f"{pdb}: {len(obs)} reactive pair(s) (carboxylate-O <=4.5A of AMP Palpha)")
        for o in obs:
            print("   chain %s: d(O_nuc->Palpha)=%s A, angle(O_nuc-P-O_leaving)=%s, metal->nuc=%s A" % (
                o["chain"], o["distance_A"], o["angle_deg"], o["metal_to_nuc_A"]))
            o["source_pdb"] = pdb
        all_obs.extend(obs)
    if not all_obs:
        print("!! no productive reactive pairs found — check atom naming / structures")
        sys.exit(2)
    ds = [o["distance_A"] for o in all_obs]
    angs = [o["angle_deg"] for o in all_obs if o["angle_deg"] is not None]
    print("\nPRODUCTIVE reference geometry (crystal-grounded):")
    print("  O_nuc->Palpha distance:  min %.2f / med %.2f / max %.2f A" % (
        min(ds), sorted(ds)[len(ds) // 2], max(ds)))
    if angs:
        print("  in-line angle:           min %.0f / med %.0f / max %.0f deg" % (
            min(angs), sorted(angs)[len(angs) // 2], max(angs)))

    # build the ReferenceEnsemble via the merged builder (real observations, not fixtures)
    try:
        from evoliez.guided.ensemble_builder import build_reference_ensemble_from_observations
        tuples = [(o["distance_A"], o["angle_deg"] if o["angle_deg"] is not None else 175.0)
                  for o in all_obs]
        ens = build_reference_ensemble_from_observations(
            "srcar", tuples, source_label="crystal_5MST_5MSW_active_state")
        ens_path = os.path.join(out_dir, "reference_ensemble.json")
        with open(ens_path, "w") as f:
            json.dump({"ensemble": ens.model_dump(), "observations": all_obs}, f, indent=2, default=str)
        print("\nwrote %s (%d conformers, evidence=%s, insufficiency=%s)" % (
            ens_path, len(ens.conformers), ens.uncertainty.evidence_density, ens.insufficiency_reason))
    except Exception as exc:  # noqa: BLE001
        print("ensemble build note:", exc)
        json.dump(all_obs, open(os.path.join(out_dir, "reference_observations.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
