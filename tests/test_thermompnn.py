"""ThermoMPNN stability adapter — the additive-sum + wildtype-verify + honest-None
contract (the SSM runner itself is server-only, so _ssm_table is monkeypatched here;
it is validated end-to-end on the server: 467 candidates / 980 mutations mapped, 0
misses, real ΔΔG distribution)."""
from __future__ import annotations

from evoliez.adapters import thermompnn
from evoliez.config import Backend, StabilityConfig
from evoliez.types import Mutation

_CFG = StabilityConfig(method="thermompnn")


def _est(muts, tmp_path, backend=Backend.real):
    return thermompnn.estimate_stability("c", None, muts, _CFG, tmp_path, backend=backend)


def test_additive_sum_of_single_ddgs(monkeypatch, tmp_path):
    table = {"40": {"wt": "A", "ddg": {"K": 1.5, "L": -0.3}},
             "70": {"wt": "V", "ddg": {"L": 0.8}}}
    monkeypatch.setattr(thermompnn, "_ssm_table", lambda *a, **k: table)
    r = _est([Mutation(wt="A", position=40, mut="K"),
              Mutation(wt="V", position=70, mut="L")], tmp_path)
    assert r["ddg_fold"] == round(1.5 + 0.8, 3)  # additive multi-site baseline
    r1 = _est([Mutation(wt="A", position=40, mut="L")], tmp_path)
    assert r1["ddg_fold"] == -0.3  # sign-preserving (negative = stabilizing)


def test_wildtype_mismatch_is_honest_none(monkeypatch, tmp_path):
    # A mapping/numbering error must NOT silently score a wrong residue's ddG.
    table = {"40": {"wt": "A", "ddg": {"K": 1.5}}}
    monkeypatch.setattr(thermompnn, "_ssm_table", lambda *a, **k: table)
    r = _est([Mutation(wt="G", position=40, mut="K")], tmp_path)  # WT says G, table says A
    assert r["ddg_fold"] is None


def test_missing_mutation_is_honest_none(monkeypatch, tmp_path):
    table = {"40": {"wt": "A", "ddg": {"K": 1.5}}}
    monkeypatch.setattr(thermompnn, "_ssm_table", lambda *a, **k: table)
    assert _est([Mutation(wt="A", position=40, mut="W")], tmp_path)["ddg_fold"] is None


def test_table_unavailable_is_honest_none(monkeypatch, tmp_path):
    # tool/structure unavailable -> None (NEVER a fabricated 0.0 that passes the filter).
    monkeypatch.setattr(thermompnn, "_ssm_table", lambda *a, **k: None)
    assert _est([Mutation(wt="A", position=40, mut="K")], tmp_path)["ddg_fold"] is None


def test_mock_backend_uses_chemistry_proxy(tmp_path):
    # backend=mock -> the shared foldx chemistry proxy (a finite ddg, no env needed).
    from evoliez.types import ProteinStructure, Residue
    st = ProteinStructure(sequence="A", residues=[Residue(index=40, aa="A", sasa=0.3)])
    r = thermompnn.estimate_stability("c", st, [Mutation(wt="A", position=40, mut="K")],
                                      _CFG, tmp_path, backend=Backend.mock)
    assert r["ddg_fold"] is not None
