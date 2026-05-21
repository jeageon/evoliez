"""pH/redox cofactor resolver + formula guard (P0 data-integrity gate).

The bug this prevents: ``cofactor: NADP`` + an NAD+ SMILES sneaks all the
way to MD and re-surfaces there as 'GAFF can't parameterise NADP' - a wrong
diagnosis. These tests lock in the resolver math and both fail-fast paths
(doctor preflight + s01_input). The whole suite stays light - no RDKit /
OpenMM / openff required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evoliez.config import InputConfig, LigandInput, load_config, Backend
from evoliez.diagnostics import collect, BLOCK, WARN, OK
from evoliez.features.cofactors import (
    amber_params_root,
    check_cofactor_matches,
    formula_of,
    is_known_cofactor,
    resolve_cofactor,
    resolve_ligand_spec,
)

# Canonical formulas (verified by hand from the SMILES; the resolver carries
# the same numbers in its curated table).
NADP_PLUS = {"C": 21, "N": 7, "O": 17, "P": 3}
NADPH = {"C": 21, "N": 7, "O": 17, "P": 3}
NAD_PLUS = {"C": 21, "N": 7, "O": 14, "P": 2}
NADH = {"C": 21, "N": 7, "O": 14, "P": 2}


# --------------------------------------------------------------------------- #
# Curated species + formula counting
# --------------------------------------------------------------------------- #
def test_curated_species_have_expected_heavy_atom_counts():
    nadp = resolve_cofactor("NADP", redox_state="oxidized")
    assert nadp.name == "NADP+"
    assert nadp.formula == NADP_PLUS
    assert nadp.n_heavy == 48                       # 21+7+17+3

    nadph = resolve_cofactor("NADP", redox_state="reduced")
    assert nadph.name == "NADPH"
    assert nadph.formula == NADPH

    nad = resolve_cofactor("NAD", redox_state="oxidized")
    assert nad.name == "NAD+"
    assert nad.formula == NAD_PLUS
    assert nad.n_heavy == 44                        # 21+7+14+2 - the bug

    nadh = resolve_cofactor("NAD", redox_state="reduced")
    assert nadh.name == "NADH"
    assert nadh.formula == NADH


def test_resolver_picks_up_redox_hint_in_name():
    # 'NADPH' encodes 'reduced' in the name itself; an explicit oxidized
    # request via cofactor_redox is still honoured (last word wins).
    assert resolve_cofactor("NADPH").name == "NADPH"
    assert resolve_cofactor("NAD+").name == "NAD+"
    assert resolve_cofactor("nadp+").name == "NADP+"


def test_resolver_rejects_unknown_or_bad_redox():
    with pytest.raises(KeyError):
        resolve_cofactor("NOT_A_REAL_COFACTOR")
    with pytest.raises(ValueError):
        resolve_cofactor("NADP", redox_state="banana")


def test_formula_of_smiles_matches_curated_table():
    # Regex element-counter (no RDKit needed in light env): must reproduce
    # the curated counts exactly. RDKit path (if installed) does the same.
    for spec in (resolve_cofactor("NADP", redox_state="oxidized"),
                 resolve_cofactor("NADP", redox_state="reduced"),
                 resolve_cofactor("NAD", redox_state="oxidized"),
                 resolve_cofactor("NAD", redox_state="reduced")):
        assert formula_of(spec.smiles) == spec.formula


def test_formula_of_handles_bracketed_and_aromatic_atoms(monkeypatch):
    # Bracketed [N+] / [O-] / aromatic lowercase / two-letter Cl all count.
    # The intent is to exercise the REGEX fallback (the path used when RDKit
    # isn't installed in the light env). Newer RDKit (>=2024) refuses
    # syntactically valid but valence-invalid SMILES like the prior fixture,
    # so we use a well-formed SMILES AND force the import-fallback path so
    # the regex counter is actually the code under test.
    import builtins

    real_import = builtins.__import__

    def _no_rdkit(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("rdkit disabled for this test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_rdkit)

    # 1-methylpyridinium fragment + ethoxide + chloride salt - well-formed
    # under modern RDKit AND exercises every construct the regex must handle:
    # aromatic lowercase ring atoms, bracketed [n+] / [O-], and two-letter Cl.
    s = "Cc1ccc[n+](c1)CC[O-].Cl"
    f = formula_of(s)
    assert f.get("C", 0) == 7 + 1                  # methyl + 5 ring + 2 ethyl
    assert f.get("N", 0) == 1
    assert f.get("O", 0) == 1
    assert f.get("Cl", 0) == 1


def test_formula_of_regex_fallback_when_rdkit_missing(monkeypatch):
    # Direct check that when RDKit can't be imported the regex path still
    # reproduces the curated NADP+ formula - locks in the light-env contract.
    import builtins

    real_import = builtins.__import__

    def _no_rdkit(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("rdkit disabled for this test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_rdkit)
    spec = resolve_cofactor("NADP", redox_state="oxidized")
    assert formula_of(spec.smiles) == spec.formula


# --------------------------------------------------------------------------- #
# Guard: declared cofactor vs actual ligand formula
# --------------------------------------------------------------------------- #
def test_guard_catches_nad_smiles_labelled_as_nadp():
    # The exact bug from the field: id says NADP but the SMILES is NAD+.
    g = check_cofactor_matches("NADP", NAD_PLUS, redox_state="oxidized")
    assert not g.ok
    msg = g.message
    assert "NADP+" in msg                          # what we expected
    assert "looks like NAD+" in msg                # what we ACTUALLY got
    assert "44" in msg and "48" in msg             # heavy-atom diff is named


def test_guard_passes_when_formula_matches():
    g = check_cofactor_matches("NADP", NADP_PLUS, redox_state="oxidized")
    assert g.ok
    assert "NADP+" in g.message


def test_guard_is_noop_for_unknown_cofactor():
    # Drug-like ligand (no curated entry): the guard must NOT fail
    # arbitrarily - it just reports 'no curated cofactor declared'.
    g = check_cofactor_matches("ibuprofen_like", {"C": 13, "O": 2})
    assert g.ok
    g2 = check_cofactor_matches(None, {"C": 13, "O": 2})
    assert g2.ok


def test_is_known_cofactor_aliases():
    assert is_known_cofactor("NADP") and is_known_cofactor("nadph")
    assert is_known_cofactor("NAD+") and is_known_cofactor("nad")
    assert not is_known_cofactor(None)
    assert not is_known_cofactor("aspirin")


# --------------------------------------------------------------------------- #
# resolve_ligand_spec: type='cofactor' -> resolved SMILES
# --------------------------------------------------------------------------- #
def test_resolve_ligand_spec_rewrites_cofactor_to_smiles():
    ic = InputConfig(
        target_sequence="ACDEFGHIKL",
        ligand=LigandInput(id="x", type="cofactor", value="NADP"),
        cofactor="NADP", cofactor_redox="oxidized", target_ph=7.4,
    )
    out = resolve_ligand_spec(ic)
    assert out.type == "smiles"
    assert out.id == "x"
    assert formula_of(out.value) == NADP_PLUS


def test_resolve_ligand_spec_is_noop_for_smiles():
    ic = InputConfig(
        target_sequence="ACDEFGHIKL",
        ligand=LigandInput(id="x", type="smiles", value="CCO"),
    )
    out = resolve_ligand_spec(ic)
    assert out is ic.ligand                        # untouched object


# --------------------------------------------------------------------------- #
# s01_input wiring: hard-fail on mismatch, pass-through when consistent
# --------------------------------------------------------------------------- #
def _mini_cfg_yaml(tmp_path: Path, *, ligand: dict) -> Path:
    fa = tmp_path / "target.fasta"
    fa.write_text(">t\n" + "A" * 30 + "\n")
    cfg_path = tmp_path / f"cfg_{ligand['id']}.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "project": {"name": "x", "output_dir": str(tmp_path / "out")},
        "input": {
            "target_id": "x", "target_fasta": str(fa),
            "ligand": ligand,
            "cofactor": "NADP", "cofactor_redox": "oxidized",
        },
        "backend": "mock",
    }))
    return cfg_path


def test_s01_input_raises_on_cofactor_ligand_mismatch(tmp_path):
    # cofactor=NADP but explicit ligand SMILES is NAD+ -> ValueError BEFORE
    # any downstream stage sees the wrong molecule.
    from evoliez.context import RunContext
    from evoliez.stages.s01_input_preprocess import InputPreprocessStage

    nad_plus_smiles = resolve_cofactor("NAD", redox_state="oxidized").smiles
    cfg_path = _mini_cfg_yaml(tmp_path, ligand={
        "id": "L", "type": "smiles", "value": nad_plus_smiles,
    })
    cfg = load_config(str(cfg_path))
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    with pytest.raises(ValueError, match="cofactor.*mismatch"):
        InputPreprocessStage().run(ctx)


def test_s01_input_accepts_cofactor_type_and_resolves(tmp_path):
    from evoliez.context import RunContext
    from evoliez.stages.s01_input_preprocess import InputPreprocessStage

    cfg_path = _mini_cfg_yaml(tmp_path, ligand={
        "id": "L", "type": "cofactor", "value": "NADP",
    })
    cfg = load_config(str(cfg_path))
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    InputPreprocessStage().run(ctx)
    lig = ctx.get("ligand")
    # the resolver's SMILES is the source of truth (works with or without
    # RDKit; the synthetic parser samples elements stochastically).
    assert formula_of(lig.smiles) == NADP_PLUS


# --------------------------------------------------------------------------- #
# doctor wiring: WARN on mock, BLOCK on real for a real mismatch
# --------------------------------------------------------------------------- #
def _write_cfg(tmp_path: Path, *, ligand_smiles: str, backend: str) -> Path:
    fa = tmp_path / "target.fasta"
    fa.write_text(">t\n" + "A" * 30 + "\n")
    cfg_path = tmp_path / f"cfg_{backend}.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "project": {"name": "x", "output_dir": str(tmp_path / "out")},
        "input": {
            "target_id": "x", "target_fasta": str(fa),
            "ligand": {"id": "L", "type": "smiles", "value": ligand_smiles},
            "cofactor": "NADP", "cofactor_redox": "oxidized",
            "target_sequence": "A" * 30,             # avoid placeholder block
        },
        "msa": {"remote_server": True},              # avoid homolog_db block
        "backend": backend,
    }))
    return cfg_path


def test_doctor_blocks_real_run_on_cofactor_mismatch(tmp_path):
    nad_plus = resolve_cofactor("NAD", redox_state="oxidized").smiles
    cfg = _write_cfg(tmp_path, ligand_smiles=nad_plus, backend="real")
    rep = collect(str(cfg))
    cof = [c for c in rep.checks if c.name == "config:cofactor"]
    assert cof, "doctor missed the cofactor check"
    assert cof[0].status == BLOCK
    assert "NAD+" in cof[0].detail


def test_doctor_passes_when_cofactor_matches(tmp_path):
    nadp_plus = resolve_cofactor("NADP", redox_state="oxidized").smiles
    cfg = _write_cfg(tmp_path, ligand_smiles=nadp_plus, backend="real")
    rep = collect(str(cfg))
    cof = [c for c in rep.checks if c.name == "config:cofactor"]
    assert cof and cof[0].status == OK


def test_doctor_warns_on_mock_for_cofactor_mismatch(tmp_path):
    # Mock is allowed to iterate on placeholder ligands; surface as WARN so
    # the user sees it but pipeline can still spin.
    nad_plus = resolve_cofactor("NAD", redox_state="oxidized").smiles
    cfg = _write_cfg(tmp_path, ligand_smiles=nad_plus, backend="mock")
    rep = collect(str(cfg))
    cof = [c for c in rep.checks if c.name == "config:cofactor"]
    assert cof and cof[0].status == WARN


# --------------------------------------------------------------------------- #
# Curated AMBER parameter pointers (Bryce Lab / Manchester DB).
# --------------------------------------------------------------------------- #
def test_curated_spec_carries_amber_filenames():
    # Every curated species advertises its Bryce Lab residue + filenames.
    # _run_real uses these to decide whether to take the tleap path.
    cases = {
        ("NAD",  "oxidized"): ("NAD", "NAD.lib", "NAD.frcmod"),
        ("NAD",  "reduced"):  ("NDH", "NDH.lib", "NDH.frcmod"),
        ("NADP", "oxidized"): ("NAP", "NAP.lib", "NAP.frcmod"),
        ("NADP", "reduced"):  ("NDP", "NDP.lib", "NDP.frcmod"),
    }
    for (family, redox), (res, lib, fr) in cases.items():
        spec = resolve_cofactor(family, redox_state=redox)
        assert spec.amber_residue_name == res
        assert spec.amber_lib == lib
        assert spec.amber_frcmod == fr


def test_resolved_amber_files_returns_none_when_missing(tmp_path):
    # With no Bryce Lab files dropped, the resolver honestly returns None
    # -> _run_real falls back to GAFF/espaloma (or skipped_parameterization).
    spec = resolve_cofactor("NADP", redox_state="oxidized")
    assert spec.resolved_amber_files(base=tmp_path) is None


def test_resolved_amber_files_returns_paths_when_present(tmp_path):
    # When both files exist, the resolver hands back absolute paths in the
    # (lib, frcmod) order the tleap dispatcher expects.
    spec = resolve_cofactor("NADP", redox_state="oxidized")
    (tmp_path / spec.amber_lib).write_text("!entry.NAP.unit.atoms\n")
    (tmp_path / spec.amber_frcmod).write_text("# stub frcmod\n")
    got = spec.resolved_amber_files(base=tmp_path)
    assert got is not None
    lib, fr = got
    assert lib.name == "NAP.lib" and fr.name == "NAP.frcmod"
    assert lib.exists() and fr.exists()


def test_resolved_amber_files_requires_BOTH_files(tmp_path):
    # Half-present is treated as missing - we don't want a stale lib without
    # its frcmod silently feeding tleap with mismatched atom types.
    spec = resolve_cofactor("NADP", redox_state="oxidized")
    (tmp_path / spec.amber_lib).write_text("!entry.NAP.unit.atoms\n")
    assert spec.resolved_amber_files(base=tmp_path) is None


def test_amber_params_root_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLIEZ_AMBER_PARAMS", str(tmp_path / "elsewhere"))
    assert amber_params_root() == (tmp_path / "elsewhere").resolve()


def test_amber_params_root_defaults_to_repo_subdir(monkeypatch):
    # No env override -> <repo>/amber/cofactors (this dir is created by
    # scripts/fetch_amber_cofactors.sh; existence is irrelevant to the path).
    monkeypatch.delenv("EVOLIEZ_AMBER_PARAMS", raising=False)
    p = amber_params_root()
    assert p.name == "cofactors" and p.parent.name == "amber"


# --------------------------------------------------------------------------- #
# Existing configs must still load AND must declare NADP+ (the original bug
# is locked out: the old NAD+ SMILES is no longer accepted under NADP).
# --------------------------------------------------------------------------- #
def test_shipped_configs_use_resolved_nadp():
    for path in ("configs/smoke.yaml", "configs/server_fdh_nadp.yaml"):
        cfg = load_config(path)
        assert cfg.input.cofactor and cfg.input.cofactor.upper() == "NADP"
        assert cfg.input.ligand.type == "cofactor", (
            f"{path} still pins an explicit SMILES - regressing to the "
            "'NAD+ labelled NADP' bug is what this guard prevents"
        )
        resolved = resolve_ligand_spec(cfg.input)
        assert formula_of(resolved.value) == NADP_PLUS
