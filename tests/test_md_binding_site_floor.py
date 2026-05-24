"""s09 → s10 binding-site reserved-slots floor.

Cheap-run-3 finding (PseFDH @ 25 min cheap run):

  Phase 2 log:
    generated 476 candidates ... binding_site_scan: 244
    s08: binding-site reserved-slots: promoted 11 into top-100
    s09: 99/100 passed; 5 advance to MD
    s10: mut_00088 / 00072 / 00073 / 00087 / 00086 — NONE from D222 family

  bench_summary:
    D222S/H/N/T/A  rank=—   (not in candidate pool that reached real MD)
    D222Q          rank=72  (rescued only by s08 floor; got proxy Boltz)
    recall@10 = 0.000

The +0.6 at_binding_site prior in s08 lifts these into top_for_redocking
but the s09→s10 cut by ml_score still favours chemistry_rules picks
because the prior boost (~0.6) is smaller than the chemistry pool match
gain at non-binding-site positions. Without a reservation here too, the
D222 family never gets real Boltz+MD evaluation, so its final_score
stays in the proxy band and the literature mutations never rank
competitively.

Same floor logic that s08 uses (promoted-into-top-N + displace-lowest-
without-binding-site-coverage) applied at the s09→s10 transition via
`_apply_binding_site_floor`. Test pins the contract without needing
the full RunContext / s09 stack.
"""

from __future__ import annotations

import inspect

from evoliez.stages.s09_nonmd_validation import (
    _apply_binding_site_floor,
)
from evoliez.types import Candidate, Mutation


def _cand(mut_str: str, score: float) -> Candidate:
    """Minimal Candidate with single mutation + a `ml_score`."""
    # mut_str like "D222S" → (D, 222, S)
    wt, pos_aa = mut_str[0], mut_str[1:]
    mut = pos_aa[-1]
    pos = int(pos_aa[:-1])
    c = Candidate(
        candidate_id=f"c_{mut_str}",
        mutations=[Mutation(wt, pos, mut)],
        generator="test",
    )
    c.scores["ml_score"] = score
    return c


def test_floor_promotes_binding_site_candidates_into_top_n():
    """Binding-site mutation at rank 50 (below cut) gets promoted into
    top-5 when no other binding-site candidate covers its position."""
    # 4 non-binding-site picks scoring high, then a D222 family at the tail.
    ranked = [
        _cand("G336S", 3.5),   # non-binding-site (336 not in BS for this test)
        _cand("T144S", 3.2),
        _cand("V198Y", 3.1),
        _cand("M330L", 3.0),
        _cand("G201S", 2.9),
        _cand("Y188W", 2.8),
        _cand("D222S", 2.5),   # binding-site mutation, below the cut
        _cand("D222H", 2.4),
    ]
    binding_site = {222}

    new_top, n_promoted = _apply_binding_site_floor(
        ranked, top_n=5, binding_site=binding_site, n_reserve=2,
    )
    muts = [str(c.mutations[0]) for c in new_top]
    assert n_promoted == 2
    assert "D222S" in muts
    assert "D222H" in muts
    assert len(new_top) == 5


def test_floor_preserves_already_covered_positions():
    """When the top-N already has enough binding-site coverage, nothing
    is promoted."""
    ranked = [
        _cand("D222S", 3.5),
        _cand("D222H", 3.4),
        _cand("G336S", 3.3),
        _cand("T144S", 3.2),
        _cand("V198Y", 3.1),
        _cand("D222N", 2.0),   # also BS, but already 2 BS hits in top-5
    ]
    binding_site = {222}

    new_top, n_promoted = _apply_binding_site_floor(
        ranked, top_n=5, binding_site=binding_site, n_reserve=2,
    )
    assert n_promoted == 0
    assert len(new_top) == 5
    # The tail D222N should NOT be in top-5 (already covered).
    assert "D222N" not in [str(c.mutations[0]) for c in new_top]


def test_floor_handles_multiple_binding_site_positions():
    """Two BS positions with no organic coverage → 1 reservation each."""
    ranked = [
        _cand("G336S", 3.5),
        _cand("T144S", 3.4),
        _cand("V198Y", 3.3),
        _cand("M330L", 3.2),
        _cand("Y188W", 3.1),
        _cand("D222S", 2.5),   # BS pos 222
        _cand("S148T", 2.4),   # BS pos 148
    ]
    binding_site = {222, 148}

    new_top, n_promoted = _apply_binding_site_floor(
        ranked, top_n=5, binding_site=binding_site, n_reserve=1,
    )
    assert n_promoted == 2
    muts = [str(c.mutations[0]) for c in new_top]
    assert "D222S" in muts
    assert "S148T" in muts


def test_floor_disabled_when_n_reserve_zero():
    """n_reserve=0 → top_n cut unchanged."""
    ranked = [
        _cand("G336S", 3.5),
        _cand("T144S", 3.2),
        _cand("D222S", 2.5),   # would be promoted at n_reserve>0
    ]
    new_top, n_promoted = _apply_binding_site_floor(
        ranked, top_n=2, binding_site={222}, n_reserve=0,
    )
    assert n_promoted == 0
    assert [str(c.mutations[0]) for c in new_top] == ["G336S", "T144S"]


def test_floor_disabled_when_no_binding_site():
    """Empty binding_site → no-op."""
    ranked = [_cand("A1B", 1.0), _cand("C2D", 0.5)]
    new_top, n_promoted = _apply_binding_site_floor(
        ranked, top_n=1, binding_site=set(), n_reserve=3,
    )
    assert n_promoted == 0
    assert len(new_top) == 1


def test_top_n_size_is_preserved():
    """Promotion swaps in for swaps out — top_n size stable."""
    ranked = [_cand("G336S", 3.5), _cand("T144S", 3.2)]
    ranked += [_cand("D222S", 2.5), _cand("D222H", 2.4)]
    new_top, _ = _apply_binding_site_floor(
        ranked, top_n=2, binding_site={222}, n_reserve=2,
    )
    assert len(new_top) == 2


def test_s09_uses_floor_when_config_enabled():
    """Source-level guard: the s09→s10 cut should call into the floor
    helper when validation.md.binding_site_reserved_per_position > 0.
    A future refactor that drops the call must be caught here."""
    from evoliez.stages import s09_nonmd_validation
    src = inspect.getsource(s09_nonmd_validation.NonMDValidationStage.run)
    assert "_apply_binding_site_floor" in src
    assert "binding_site_reserved_per_position" in src
