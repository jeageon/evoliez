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


# A deliberately ASYMMETRIC 5-atom ligand with realistic bond lengths.
# Graph: N3-C0-C1(-O2)(-F4). Element-labelled it has NO automorphism, so the
# canonical<->tool isomorphism is unique -> a single correct permutation.
def _asym_ligand():
    from evoliez.types import LigandAtom

    return [
        LigandAtom(id="C0", element="C", coord=(0.00, 0.00, 0.00)),
        LigandAtom(id="C1", element="C", coord=(1.50, 0.00, 0.00)),
        LigandAtom(id="O2", element="O", coord=(2.20, 1.20, 0.00)),
        LigandAtom(id="N3", element="N", coord=(-1.40, 0.00, 0.00)),
        LigandAtom(id="F4", element="F", coord=(1.50, -1.35, 0.00)),
    ]


def test_relabel_maps_reordered_ligand_by_chemistry_graph():
    """Core of expert risk #6: a tool re-emits the SAME molecule with a
    PERMUTED atom order (same formula). relabel_to_canonical must attach each
    tool coordinate to the *chemically corresponding* canonical id - NOT the
    positionally i-th one - and only then report a verified lock."""
    canonical = _asym_ligand()
    # Tool: same atoms, shuffled order, foreign ids, coords shifted +10 in x
    # (uniform shift keeps the graph identical so the mapping is well-defined).
    order = [2, 4, 1, 3, 0]  # permutation of canonical indices
    tool = [
        dataclasses.replace(
            canonical[c], id=f"Z{k}",
            coord=(canonical[c].coord[0] + 10.0,) + canonical[c].coord[1:],
        )
        for k, c in enumerate(order)
    ]

    locked, ok = relabel_to_canonical(tool, canonical)
    assert ok                                              # graph-verified
    assert [a.id for a in locked] == ["C0", "C1", "O2", "N3", "F4"]
    assert [a.element for a in locked] == ["C", "C", "O", "N", "F"]
    # Each canonical id received ITS OWN atom's (shifted) coord. A positional
    # zip would put tool[0] (=O2's coord) onto C0 and fail this assertion.
    for k, c in enumerate(canonical):
        assert locked[k].coord == (c.coord[0] + 10.0,) + c.coord[1:]


def test_relabel_heavy_lock_maps_by_graph_and_drops_H():
    """Real server case: canonical parse adds explicit H (Chem.AddHs) e.g.
    NADP 44 heavy + 26 H = 70, but Boltz/Vina re-emit HEAVY ATOMS ONLY. The
    lock must (a) work on heavy atoms despite 70!=44, (b) map heavy atoms by
    chemistry graph even when the tool reorders them, (c) drop H. rdkit-free
    so it runs in CI regardless of the ligand-parse backend."""
    from evoliez.types import LigandAtom

    heavy = _asym_ligand()
    canonical = heavy + [
        LigandAtom(id=f"H{i}", element="H", coord=(0.30 * i, 2.0, 0.0))
        for i in range(4)
    ]                                                # 5 heavy + 4 H = 9
    order = [3, 0, 4, 1, 2]                           # reordered heavy, no H
    tool = [
        dataclasses.replace(heavy[c], id=f"T{k}",
                            coord=(heavy[c].coord[0] + 5.0,)
                            + heavy[c].coord[1:])
        for k, c in enumerate(order)
    ]

    locked, ok = relabel_to_canonical(tool, canonical)
    assert ok                                        # heavy graph-verified
    assert [a.id for a in locked] == ["C0", "C1", "O2", "N3", "F4"]
    assert not any(a.element == "H" for a in locked)      # H dropped
    for k, c in enumerate(heavy):                          # coords by graph
        assert locked[k].coord == (c.coord[0] + 5.0,) + c.coord[1:]

    # genuine heavy-count disagreement is still an honest no-lock
    _, ok_bad = relabel_to_canonical(tool[:-1], canonical)
    assert not ok_bad


def test_relabel_unverified_falls_back_and_warns(caplog):
    """Counts match but the chemistry graphs do NOT (e.g. a tool emitted a
    different connectivity / non-physical coords). locked MUST be False -
    'counts equal' is no longer accepted as 'ids verified' - and the warning
    must say so, while downstream still gets a (positionally adopted) ligand."""
    from evoliez.types import LigandAtom

    canonical = _asym_ligand()                       # has real bonds
    # Same elements & count, but atoms scattered far apart -> NO inferable
    # bonds -> graphs cannot be matched.
    tool = [
        LigandAtom(id=f"Q{i}", element=a.element,
                   coord=(50.0 * i, 0.0, 0.0))
        for i, a in enumerate(canonical)
    ]
    with caplog.at_level("WARNING"):
        locked, ok = relabel_to_canonical(tool, canonical)
    assert not ok                                    # NOT verified
    assert "atom-order NOT verified" in caplog.text
    # fallback still returns canonical ids + adopted coords (never silent loss)
    assert [a.id for a in locked] == [a.id for a in canonical]
    assert [a.coord for a in locked] == [a.coord for a in tool]

    # genuine count mismatch -> parsed kept, no false lock
    locked2, ok2 = relabel_to_canonical(tool[:-1], canonical)
    assert not ok2
    assert [a.id for a in locked2] == [a.id for a in tool[:-1]]


def test_relabel_is_deterministic_under_symmetry():
    """Determinism must be preserved even when the molecule has automorphic
    (chemically equivalent) atoms: a CO2-like O=C=O has an O<->O symmetry.
    Repeated calls must give byte-identical output (stable tie-break)."""
    from evoliez.types import LigandAtom

    canonical = [
        LigandAtom(id="O0", element="O", coord=(-1.16, 0.0, 0.0)),
        LigandAtom(id="C1", element="C", coord=(0.00, 0.0, 0.0)),
        LigandAtom(id="O2", element="O", coord=(1.16, 0.0, 0.0)),
    ]
    tool = [
        dataclasses.replace(canonical[c], id=f"S{k}",
                            coord=(canonical[c].coord[0] + 7.0, 0.0, 0.0))
        for k, c in enumerate([2, 0, 1])
    ]
    r1, ok1 = relabel_to_canonical(tool, canonical)
    r2, ok2 = relabel_to_canonical(tool, canonical)
    assert ok1 and ok2
    assert [(a.id, a.coord) for a in r1] == [(a.id, a.coord) for a in r2]
    assert [a.id for a in r1] == ["O0", "C1", "O2"]
    # carbon is unambiguous; each O lands on a real tool O coordinate
    assert r1[1].coord == (7.0, 0.0, 0.0)
    o_coords = {a.coord for a in r1 if a.element == "O"}
    assert o_coords == {(-1.16 + 7.0, 0.0, 0.0), (1.16 + 7.0, 0.0, 0.0)}


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
