"""s07_mutation_gen `binding_site_scan` generator.

Cheap-run discovery (cheap PseFDH on the server, no UniRef30 DB
available):

  - chemistry_rules ranks substitutions by the nearest ligand-atom's
    role pool. For PseFDH D222, that atom is a phosphate oxygen → pool
    "KRH". Tishkov's polar/amide D→S/N/T/Q swaps were therefore
    NEVER proposed.
  - msa_sampler can't see real evolutionary signal when s02 homologs
    are mock (which is how cheap-run runs).
  - Result: recall@K on the 7 Tishkov beneficial mutations was 0 even
    after the P0a ranking gate landed.

The new `binding_site_scan` generator does a flat 19-AA scan at every
config-declared `known_binding_site` residue. Universal across enzymes
(every card declares known_binding_site). Respects `fixed_positions`
so catalytic residues are never proposed.

These tests pin the contract without needing the heavy
RunContext / Boltz / s06_graph stack: drive `MutationGenStage.run`
directly with a stub context.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Dict, List

from evoliez.config import MutationGenConfig
from evoliez.stages import s07_mutation_gen
from evoliez.stages.s07_mutation_gen import MutationGenStage
from evoliez.types import Mutation, ProteinStructure, Residue


# ---------------------------------------------------------------------------
# minimal-fixture helpers
# ---------------------------------------------------------------------------


def _residue(idx: int, aa: str) -> Residue:
    return Residue(
        index=idx, aa=aa, ca=(float(idx), 0.0, 0.0),
        sasa=0.5, plddt=80.0,
    )


def _complex(seq: str):
    """The smallest valid Complex shape s07 reads — residues by index +
    one ligand atom (atoms list is only read for chemistry_rules)."""
    from evoliez.types import Complex, Ligand, LigandAtom
    residues = [_residue(i + 1, aa) for i, aa in enumerate(seq)]
    structure = ProteinStructure(sequence=seq, residues=residues)
    ligand = Ligand(
        id="L", smiles="O",
        atoms=[LigandAtom(id="O1", element="O", coord=(5.0, 0.0, 0.0))],
    )
    return Complex(structure=structure, ligand=ligand)


def _stub_ctx(
    cx, designable, known_binding_site, fixed_positions, methods,
    max_candidates=2000,
):
    """In-memory ctx satisfying the MutationGenStage contract.

    Real RunContext threads through SQLAlchemy + project paths; for the
    generator unit test we only need the four `require`d keys + the
    three `get`s with safe defaults. ``persist_meta`` is a no-op.
    """
    cfg = MutationGenConfig(methods=list(methods), max_candidates=max_candidates)
    config = SimpleNamespace(
        mutation_generation=cfg,
        backend=SimpleNamespace(),
    )
    # backend_for() is only read for ligandmpnn; we don't enable it.
    config.backend_for = lambda _name: None
    state: Dict[str, Any] = {
        "wt_complex": cx,
        "position_features": [],
        "designable_positions": list(designable),
        "contacts": [],
        "known_binding_site": list(known_binding_site),
        "fixed_positions": list(fixed_positions),
    }
    persisted: Dict[str, Any] = {}
    ctx = SimpleNamespace(
        require=lambda k: state[k],
        get=lambda k, default=None: state.get(k, default),
        put=lambda k, v: state.__setitem__(k, v),
        persist_meta=lambda k, v: persisted.__setitem__(k, v),
        config=config,
        paths=SimpleNamespace(mutations=None),
        dry_run=False,
        store=None,
    )
    return ctx, state, persisted


# ---------------------------------------------------------------------------
# binding_site_scan generator
# ---------------------------------------------------------------------------


def test_binding_site_scan_proposes_tishkov_d222_family():
    """The PseFDH cheap-run regression case: D222S/N/T/Q/H/A must all
    appear in the candidate pool. Use a sequence that puts D at position
    222 so the run is realistic; only `binding_site_scan` is enabled so
    we measure ONLY its contribution."""
    seq = "A" * 221 + "D" + "A" * 80   # D at position 222 (1-based)
    cx = _complex(seq)
    ctx, _, _ = _stub_ctx(
        cx,
        designable=[222],
        known_binding_site=[222],
        fixed_positions=[],
        methods=["binding_site_scan"],
    )

    MutationGenStage().run(ctx)
    cands = ctx.require("candidates")
    proposed = sorted({str(c.mutations[0]) for c in cands
                       if len(c.mutations) == 1})

    # All six Tishkov-class D222 substitutions in the benchmark must
    # appear; D222D (no-op) must NOT.
    for aa in ("S", "N", "T", "Q", "H", "A"):
        assert f"D222{aa}" in proposed, f"missing D222{aa}; got {proposed}"
    assert "D222D" not in proposed

    # Every candidate carries the generator tag + the exhaustive_scan flag.
    for c in cands:
        assert c.generator == "binding_site_scan"
        assert c.details.get("exhaustive_scan") is True


def test_binding_site_scan_respects_fixed_positions():
    """Catalytic / fixed residues must NEVER be mutated, even if
    they're listed in known_binding_site (some real configs include
    catalytic residues there for the docking pocket constraint)."""
    seq = "A" * 284 + "R" + "A" * 47 + "H" + "A" * 68
    # R at 285, H at 333 (PseFDH catalytic). known_binding_site
    # intentionally includes both — exactly the configs/server_fdh_nadp.yaml
    # shape (R285/H333 are in known_binding_site AND fixed_residues).
    cx = _complex(seq)
    ctx, _, _ = _stub_ctx(
        cx,
        designable=[285, 333, 200],
        known_binding_site=[200, 285, 333],
        fixed_positions=[285, 333],
        methods=["binding_site_scan"],
    )

    MutationGenStage().run(ctx)
    cands = ctx.require("candidates")
    positions = {c.mutations[0].position for c in cands
                 if len(c.mutations) == 1}
    # Position 200 should produce substitutions; 285 / 333 must NOT.
    assert 200 in positions
    assert 285 not in positions
    assert 333 not in positions


def test_binding_site_scan_skips_self_substitutions():
    seq = "DEFGH"
    cx = _complex(seq)
    ctx, _, _ = _stub_ctx(
        cx, designable=[1, 2, 3, 4, 5],
        known_binding_site=[1, 2, 3, 4, 5],
        fixed_positions=[],
        methods=["binding_site_scan"],
    )
    MutationGenStage().run(ctx)
    cands = ctx.require("candidates")
    for c in cands:
        m = c.mutations[0]
        assert m.wt != m.mut, f"self-substitution slipped through: {m}"


def test_binding_site_scan_respects_max_candidates():
    """Even with 100 binding-site residues * 19 substitutions = 1900,
    the global `max_candidates` cap must bound the output."""
    seq = "A" * 200
    cx = _complex(seq)
    ctx, _, _ = _stub_ctx(
        cx,
        designable=list(range(1, 101)),
        known_binding_site=list(range(1, 101)),
        fixed_positions=[],
        methods=["binding_site_scan"],
        max_candidates=50,
    )
    MutationGenStage().run(ctx)
    cands = ctx.require("candidates")
    assert len(cands) == 50


def test_binding_site_scan_complements_chemistry_rules():
    """When run alongside chemistry_rules, binding_site_scan FILLS IN
    the Tishkov-class mutations chemistry_rules' anion pool can't
    propose at a phosphate-adjacent D residue."""
    seq = "A" * 221 + "D" + "A" * 80
    cx = _complex(seq)
    ctx, _, _ = _stub_ctx(
        cx, designable=[222],
        known_binding_site=[222],
        fixed_positions=[],
        methods=["chemistry_rules", "binding_site_scan"],
    )
    MutationGenStage().run(ctx)
    cands = ctx.require("candidates")
    by_gen: Dict[str, List[str]] = {}
    for c in cands:
        by_gen.setdefault(c.generator, []).append(str(c.mutations[0]))

    # chemistry_rules can't see a contact (we passed contacts=[]) so it
    # contributes 0; binding_site_scan must still cover the Tishkov set.
    scanned = set(by_gen.get("binding_site_scan", []))
    for aa in ("S", "N", "T", "Q", "H", "A"):
        assert f"D222{aa}" in scanned


# ---------------------------------------------------------------------------
# Source-level guard so a future refactor can't silently drop the
# generator branch (mirrors the pattern in test_ligandmpnn_preflight).
# ---------------------------------------------------------------------------


def test_binding_site_scan_branch_present_in_source():
    src = inspect.getsource(s07_mutation_gen.MutationGenStage.run)
    assert '"binding_site_scan"' in src or "'binding_site_scan'" in src
    assert "known_binding_site" in src
    assert "fixed_positions" in src
