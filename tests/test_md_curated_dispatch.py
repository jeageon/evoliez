"""Curated AMBER cofactor dispatch in _run_real.

These tests run in the LIGHT env (no OpenMM stack); they assert the dispatch
LOGIC (when does curated win vs probe? what happens on missing files? on
tleap failure?) at the source level + via small unit-tested helpers, mirror-
ing the style of test_md_real_fixture.py. The end-to-end "is it actually a
working System?" question is answered by scripts/server_test_curated_nadp.sh
on the server (Bryce Lab files present + tleap binary), not in this suite.
"""

from __future__ import annotations

import inspect

import pytest

from evoliez.adapters import openmm_engine
from evoliez.features.cofactors import lookup_by_smiles, resolve_cofactor


# --------------------------------------------------------------------------- #
# Source-level guards: the dispatcher exists and routes correctly.
# --------------------------------------------------------------------------- #
def test_run_real_dispatches_to_curated_before_gaff_probe():
    src = inspect.getsource(openmm_engine._run_real)
    # Curated path is mentioned and checked BEFORE the GAFF probe.
    assert "_curated_param_system_generator" in src
    assert "lookup_by_smiles" in src
    assert "took_curated" in src
    # Curated success bypasses _ligand_system_generator entirely.
    pre = src.split("_ligand_system_generator")[0]
    assert "_curated_param_system_generator" in pre
    # A curated failure surfaces as `failed`, not as a parameterization skip.
    assert 'failure_reason=f"curated tleap' in src


def test_gasteiger_tier_is_wired_between_curated_and_am1bcc():
    src = inspect.getsource(openmm_engine._run_real)
    gas = inspect.getsource(openmm_engine._gasteiger_charge_system_generator)

    # Tier 2 exists and gates on a known cofactor without curated files.
    assert "_gasteiger_charge_system_generator" in src
    assert 'gaff-2.11+gasteiger' in src
    # Order is curated -> Gasteiger -> AM1-BCC: curated branch appears in
    # the source BEFORE Gasteiger, which appears BEFORE _ligand_system_generator.
    pos_curated  = src.index("_curated_param_system_generator(")
    pos_gas      = src.index("_gasteiger_charge_system_generator(")
    pos_am1bcc   = src.index("_ligand_system_generator(")
    assert pos_curated < pos_gas < pos_am1bcc, (
        "tier order regressed: must be curated -> Gasteiger -> AM1-BCC probe"
    )

    # Gasteiger generator uses RDKit Gasteiger charges (not am1bcc) and
    # converts a probe failure into _LigandParamUnsupported so the
    # dispatcher uniformly maps it to skipped_parameterization.
    assert "gasteiger" in gas.lower()
    assert "RDKitToolkitWrapper" in gas
    assert "assign_partial_charges" in gas
    assert "_LigandParamUnsupported" in gas
    # ligand-alone probe so a failure surfaces here, not later
    assert "create_system" in gas


def test_drug_like_ligand_still_uses_am1bcc_probe():
    # A non-cofactor (curated_spec is None) must NOT short-circuit into
    # the Gasteiger tier - drug-like ligands keep the standard AM1-BCC
    # production charge model.
    src = inspect.getsource(openmm_engine._run_real)
    assert "if curated_spec is not None:" in src
    # The else-branch calls the original AM1-BCC probe.
    after_if = src.split("if curated_spec is not None:")[1]
    assert "else:" in after_if
    assert "_ligand_system_generator(off_lig, workdir)" in after_if


def test_curated_generator_signals_missing_files_distinctly():
    # _CuratedParamUnavailable is the missing-files-on-disk signal; only the
    # dispatcher catches it (-> fall back to probe). A REAL tleap failure
    # raises a generic Exception which the dispatcher classifies as `failed`.
    assert issubclass(openmm_engine._CuratedParamUnavailable, Exception)
    src = inspect.getsource(openmm_engine._curated_param_system_generator)
    assert "_CuratedParamUnavailable" in src
    assert "tleap" in src
    assert "AmberPrmtopFile" in src and "AmberInpcrdFile" in src


def test_lig_idx_detection_includes_curated_amber_resnames():
    src = inspect.getsource(openmm_engine._run_real)
    # AMBER 3-letter resnames the Bryce Lab .lib entries use, so the curated
    # tleap path's ligand atoms get picked up by the RMSD/contact logic.
    for code in ("NAD", "NDH", "NAP", "NDP"):
        assert code in src, f"curated ligand resname {code!r} not in _LIG_RES"


# --------------------------------------------------------------------------- #
# Dispatch decision (no MD stack needed): given a SMILES + files state,
# does the dispatcher pick the curated path?
# --------------------------------------------------------------------------- #
def test_curated_path_chosen_when_files_present(tmp_path, monkeypatch):
    # arrange: drop fake Bryce Lab files in a tmp dir + redirect the
    # resolver to look there
    spec = resolve_cofactor("NADP", redox_state="oxidized")
    (tmp_path / spec.amber_lib).write_text("!entry.NAP.unit.atoms\n")
    (tmp_path / spec.amber_frcmod).write_text("# stub\n")
    monkeypatch.setenv("EVOLIEZ_AMBER_PARAMS", str(tmp_path))

    # act: look up the spec, then ask if its curated files resolve
    sp = lookup_by_smiles(spec.smiles)
    assert sp is not None and sp.name == "NADP+"
    files = sp.resolved_amber_files()
    assert files is not None
    lib, fr = files
    assert lib.parent == tmp_path and fr.parent == tmp_path


def test_curated_path_not_chosen_when_files_absent(tmp_path, monkeypatch):
    # Empty params dir => dispatcher must NOT take the curated branch
    # (resolved_amber_files() returns None -> falls through to probe).
    monkeypatch.setenv("EVOLIEZ_AMBER_PARAMS", str(tmp_path))
    spec = lookup_by_smiles(resolve_cofactor("NADP",
                                              redox_state="oxidized").smiles)
    assert spec is not None
    assert spec.resolved_amber_files() is None


def test_curated_path_skipped_for_unknown_drug_like_ligand():
    # A non-cofactor SMILES never hits the curated branch.
    assert lookup_by_smiles("CC(=O)Oc1ccccc1C(=O)O") is None  # aspirin
    assert lookup_by_smiles("CCO") is None
    assert lookup_by_smiles("") is None


def test_lookup_by_smiles_distinguishes_oxidized_vs_reduced():
    # NAD+ and NADH share heavy-atom formula but differ in the SMILES
    # ([n+] pyridinium vs 1,4-dihydropyridine); the canonical SMILES match
    # must keep them apart - else the curated dispatcher would route NADH
    # to NAD+ parameters (wrong species).
    nad_plus = resolve_cofactor("NAD", redox_state="oxidized")
    nadh = resolve_cofactor("NAD", redox_state="reduced")
    assert lookup_by_smiles(nad_plus.smiles).name == "NAD+"
    assert lookup_by_smiles(nadh.smiles).name == "NADH"
    nadp_plus = resolve_cofactor("NADP", redox_state="oxidized")
    nadph = resolve_cofactor("NADP", redox_state="reduced")
    assert lookup_by_smiles(nadp_plus.smiles).name == "NADP+"
    assert lookup_by_smiles(nadph.smiles).name == "NADPH"


# --------------------------------------------------------------------------- #
# Helper source-level guard: the writer renames residues to the curated code
# (full functional check lives in scripts/server_test_curated_nadp.sh because
# it needs real OpenMM + OpenFF, which the light env doesn't carry).
# --------------------------------------------------------------------------- #
def test_write_ligand_pdb_renames_residue_to_curated_code():
    src = inspect.getsource(openmm_engine._write_ligand_pdb)
    # Loops over topology.residues() and assigns the curated residue_name -
    # tleap's `loadpdb` matches on this 3-letter code against the .lib
    # entry, else it silently builds a UNK residue with no params.
    assert "topo.residues" in src or "topology.residues" in src
    assert "res.name = residue_name" in src
    assert "writeFile" in src
