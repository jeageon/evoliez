"""s06b representative selection: MSA-derived count + largest-cluster-first.

`representative_homologs` may be a fixed int, -1 (all clusters), or "auto" — in
which case the count is sized from the subfamily-cluster structure (cover
`representative_coverage` of the homolog pool by descending cluster size, clamped
to [representative_min, representative_max]). Selection always takes the LARGEST
clusters first (best-supported subfamilies), not cluster-id order.
"""

from __future__ import annotations

import pytest

from evoliez.adapters.msa_tools import Homolog
from evoliez.config import InteractionModelConfig
from evoliez.stages.s06b_interaction_model import (
    _pick_representatives,
    _resolve_n_reps,
)


def _pool(sizes: dict) -> list:
    """Build a homolog pool. `sizes` maps cluster_id -> member count; identities
    descend within each cluster so the representative is deterministic."""
    homs = []
    for cid, n in sizes.items():
        for i in range(n):
            homs.append(
                Homolog(
                    id=f"c{cid}_m{i}",
                    sequence="ACDEFGHIKLMNPQRSTVWY",
                    identity=0.9 - 0.0001 * (cid * 100 + i),
                    coverage=1.0,
                    cluster_id=cid,
                )
            )
    return homs


# --- selection order ------------------------------------------------------- #
def test_picks_largest_clusters_first_not_cluster_id_order():
    # cluster ids 0 and 2 are SMALL; 9 and 5 are the big ones. A cluster-id-order
    # pick would wrongly grab {0, 2}; size-order must grab {9, 5}.
    pool = _pool({0: 1, 9: 10, 5: 5, 2: 3})
    cfg = InteractionModelConfig(representative_homologs=2)
    reps, note = _pick_representatives(pool, cfg)
    assert {r.cluster_id for r in reps} == {9, 5}
    assert len(reps) == 2
    assert "fixed" in note


def test_one_rep_per_cluster_is_highest_identity():
    pool = _pool({3: 4})  # 4 members, identities descend with i
    cfg = InteractionModelConfig(representative_homologs=1)
    reps, _ = _pick_representatives(pool, cfg)
    assert len(reps) == 1
    assert reps[0].identity == max(h.identity for h in pool)


# --- auto: count derived from coverage ------------------------------------- #
def test_auto_count_from_coverage():
    # sizes desc = [10,5,3,1,1], total 20. 0.9*20 = 18; 10+5+3 = 18 -> 3 clusters.
    pool = _pool({9: 10, 5: 5, 2: 3, 1: 1, 0: 1})
    cfg = InteractionModelConfig(
        representative_homologs="auto", representative_coverage=0.9,
        representative_min=1, representative_max=150,
    )
    reps, note = _pick_representatives(pool, cfg)
    assert len(reps) == 3
    assert {r.cluster_id for r in reps} == {9, 5, 2}
    assert "auto" in note


def test_auto_scales_up_with_diversity():
    # A flatter (more diverse) family needs MORE clusters to hit 90% coverage
    # than a family dominated by one big cluster — the whole point of "auto".
    cfg = InteractionModelConfig(
        representative_homologs="auto", representative_coverage=0.9,
        representative_min=1, representative_max=150,
    )
    dominated, _ = _pick_representatives(_pool({0: 90, **{i: 1 for i in range(1, 11)}}), cfg)
    diverse, _ = _pick_representatives(_pool({i: 10 for i in range(10)}), cfg)
    assert len(diverse) > len(dominated)


# --- auto: clamps ---------------------------------------------------------- #
def test_auto_clamped_to_max():
    pool = _pool({i: 1 for i in range(20)})  # 20 singletons -> needs 18 for 90%
    cfg = InteractionModelConfig(
        representative_homologs="auto", representative_coverage=0.9,
        representative_min=1, representative_max=8,
    )
    reps, note = _pick_representatives(pool, cfg)
    assert len(reps) == 8
    assert "clamped to 8" in note


def test_auto_floored_to_min_tops_up_beyond_clusters():
    # 1 cluster already covers 90%, but min=3 floors it; only 2 clusters exist,
    # so the 3rd rep is an identity top-up (not a new cluster).
    pool = _pool({9: 18, 0: 2})
    cfg = InteractionModelConfig(
        representative_homologs="auto", representative_coverage=0.9,
        representative_min=3, representative_max=150,
    )
    reps, note = _pick_representatives(pool, cfg)
    assert len(reps) == 3
    assert "clamped to 3" in note


# --- -1 = all -------------------------------------------------------------- #
def test_minus_one_is_all_clusters():
    pool = _pool({9: 10, 5: 5, 2: 3})
    reps, note = _pick_representatives(pool, InteractionModelConfig(representative_homologs=-1))
    assert len(reps) == 3
    assert "all 3 clusters" in note


# --- _resolve_n_reps unit -------------------------------------------------- #
def test_resolve_coverage_exact_boundary():
    cfg = InteractionModelConfig(
        representative_homologs="auto", representative_coverage=0.5,
        representative_min=1, representative_max=999,
    )
    # sizes [10,5,5], total 20 -> 0.5*20 = 10; first cluster alone hits 10.
    n, _ = _resolve_n_reps([10, 5, 5], 20, cfg)
    assert n == 1


# --- config validation ----------------------------------------------------- #
def test_auto_string_parses():
    assert InteractionModelConfig(representative_homologs="auto").representative_homologs == "auto"


def test_bad_string_rejected():
    with pytest.raises(Exception):
        InteractionModelConfig(representative_homologs="sometimes")


def test_min_greater_than_max_rejected():
    with pytest.raises(Exception):
        InteractionModelConfig(representative_min=100, representative_max=10)
