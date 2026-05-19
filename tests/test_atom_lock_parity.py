"""Atom-index lock + schema parity + GNN-fallback transparency (#5)."""

import dataclasses
import json
from pathlib import Path

from evoliez.adapters.boltz import predict_complex
from evoliez.config import (
    Backend,
    ComplexPredictionConfig,
    LigandInput,
    load_config,
)
from evoliez.context import RunContext
from evoliez.features.ligand import parse_ligand, relabel_to_canonical
from evoliez.pipeline import Pipeline
from evoliez.types import Complex
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def test_relabel_to_canonical_locks_ids():
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="CC(=O)OP(O)=O"))
    # simulate a tool re-emitting atoms in a different id scheme, same count
    parsed = [dataclasses.replace(a, id=f"X{i}", coord=(i, i, i))
              for i, a in enumerate(lig.atoms)]
    locked, ok = relabel_to_canonical(parsed, lig.atoms)
    assert ok
    assert [a.id for a in locked] == [a.id for a in lig.atoms]   # ids kept
    assert locked[0].coord == (0, 0, 0)                            # coords taken
    # mismatch -> not locked, flagged by caller
    locked2, ok2 = relabel_to_canonical(parsed[:-1], lig.atoms)
    assert not ok2


def test_relabel_locks_on_heavy_atoms_when_tool_drops_H():
    """Real server case: canonical parse adds explicit H (Chem.AddHs) e.g.
    NADP 44 heavy + 26 H = 70, but Boltz/Vina re-emit HEAVY ATOMS ONLY.
    The lock must succeed on heavy atoms (canonical heavy ids/chemistry +
    tool coords, H dropped), not bail on the 70!=44 count. rdkit-independent
    so it runs in CI regardless of the ligand-parse backend."""
    from evoliez.types import LigandAtom

    canonical = (
        [LigandAtom(id=f"C{i}", element="C", coord=(0.0, 0.0, 0.0))
         for i in range(3)]
        + [LigandAtom(id=f"H{i}", element="H", coord=(0.0, 0.0, 0.0))
           for i in range(5)]                       # 3 heavy + 5 H = 8
    )
    tool = [LigandAtom(id=f"X{i}", element="C", coord=(float(i), 0.0, 0.0))
            for i in range(3)]                       # 3 heavy, no H

    locked, ok = relabel_to_canonical(tool, canonical)
    assert ok                                        # heavy 3 == 3 -> locked
    assert [a.id for a in locked] == ["C0", "C1", "C2"]   # canonical heavy ids
    assert not any(a.element == "H" for a in locked)      # H dropped
    assert [a.coord[0] for a in locked] == [0.0, 1.0, 2.0]  # tool coords

    # genuine heavy-count disagreement still an honest no-lock
    _, ok_bad = relabel_to_canonical(tool[:-1], canonical)
    assert not ok_bad


def _tiny_cfg(tmp_path, **ov):
    base = {
        "project.output_dir": str(tmp_path / "run"),
        "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
        "mutation_generation": {"methods": ["chemistry_rules"],
                                "max_candidates": 25,
                                "design_radius_angstrom": 9.0},
        "reranking": {"top_for_redocking": 8, "top_for_md": 3},
        "validation": {"md": {"top_candidates": 3}},
        "gnn": {"build_dataset": False},
    }
    base.update(ov)
    return load_config(ROOT / "configs" / "example_fdh_nadp.yaml", base)


def test_atom_ids_stable_through_pipeline(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)

    canonical = [a.id for a in ctx.get("ligand").atoms]
    wt = ctx.get("wt_complex")
    assert [a.id for a in wt.ligand.atoms] == canonical
    # interaction graph ligand nodes use the same ids
    g = ctx.get("interaction_graph")
    lig_nodes = {n[1:] for n, d in g.nodes(data=True)
                 if d.get("kind") == "ligand_atom"}
    assert lig_nodes.issubset(set(canonical))
    # provenance records the canonical id list
    prov = json.loads(
        (ctx.paths.reports / "provenance.json").read_text()
    )
    assert prov["ligand_atom_ids"] == canonical


def test_mock_vs_real_dryrun_schema_parity(tmp_path):
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="CCO"))
    cp = ComplexPredictionConfig(diffusion_samples=4)
    m = predict_complex("t", "ACDEFGHIKLMN", lig, cp, tmp_path,
                        backend=Backend.mock)
    r = predict_complex("t", "ACDEFGHIKLMN", lig, cp, tmp_path,
                        backend=Backend.real, dry_run=True)
    # identical structured contract
    assert {f.name for f in dataclasses.fields(Complex)} == {
        f.name for f in dataclasses.fields(type(r))
    }
    assert set(m.metrics) == set(r.metrics)
    assert [a.id for a in m.ligand.atoms] == [a.id for a in r.ligand.atoms]
    assert len(m.samples) == len(r.samples)
    assert set(m.samples[0].metrics) == set(r.samples[0].metrics)


def test_gnn_fallback_transparency(tmp_path):
    # gnn.enabled but torch absent -> must be reported, not silent
    cfg = _tiny_cfg(tmp_path, gnn={"enabled": True, "build_dataset": False})
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)

    assert ctx.meta("gnn_status") in ("heuristic_fallback", "trained")
    prov = json.loads((ctx.paths.reports / "provenance.json").read_text())
    assert prov["gnn_status"] == ctx.meta("gnn_status")
    report = (ctx.paths.reports / "final_report.md").read_text()
    assert "Model provenance" in report
    if ctx.meta("gnn_status") == "heuristic_fallback":
        assert "heuristic" in report.lower()
