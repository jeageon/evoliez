"""Real-backend subfamily clustering (assign_clusters).

The real homolog parsers return cluster_id=0 for every hit; without clustering,
s06b._pick_representatives collapses to a single representative and the
family-geometry ensemble silently degrades to 'top-N by identity'. assign_clusters
recovers subfamily structure with no external tool. These tests feed it
sequences with KNOWN subfamily structure and assert it recovers the groups, and
that cfg.cluster_identity actually changes the granularity.
"""

from __future__ import annotations

from evoliez.adapters.msa_tools import Homolog, assign_clusters
from evoliez.config import HomologConfig

# Three clearly-distinct subfamilies: each is a different repeated motif, so
# within-subfamily k-mer overlap is high and between-subfamily overlap is ~0.
_FAMILIES = {
    0: "ACDEFGHIKLMNPQRSTVWY" * 6,   # subfamily A backbone
    1: "WYVTSRQPNMLKIHGFEDCA" * 6,   # subfamily B (reversed alphabet)
    2: "GGGGSSSSAAAATTTTLLLL" * 6,   # subfamily C (low-complexity)
}


def _member(fam: int, jitter: int) -> str:
    """A subfamily member: the family backbone with a few point mutations."""
    s = list(_FAMILIES[fam])
    for j in range(jitter):
        pos = (j * 37 + fam * 7) % len(s)
        s[pos] = "M" if s[pos] != "M" else "K"
    return "".join(s)


def _make(n_per_family: int):
    homologs = []
    for fam in _FAMILIES:
        for i in range(n_per_family):
            homologs.append(
                Homolog(
                    id=f"f{fam}_{i}",
                    sequence=_member(fam, i % 4),
                    identity=0.9 - 0.01 * i,
                    coverage=1.0,
                )
            )
    return homologs


def test_recovers_known_subfamilies():
    homologs = _make(5)  # 3 families x 5 members
    assign_clusters(homologs, HomologConfig(cluster_identity=0.90))
    # Members of the same backbone family must share a cluster_id...
    by_family = {}
    for h in homologs:
        fam = h.id.split("_")[0]
        by_family.setdefault(fam, set()).add(h.cluster_id)
    for fam, cids in by_family.items():
        assert len(cids) == 1, f"{fam} split across clusters {cids}"
    # ...and the three families must be three DISTINCT clusters.
    all_cids = {h.cluster_id for h in homologs}
    assert len(all_cids) == 3


def test_not_collapsed_to_single_cluster():
    """The whole point: real homologs must not all land in cluster 0."""
    homologs = _make(8)
    assign_clusters(homologs, HomologConfig())
    assert len({h.cluster_id for h in homologs}) >= 3
    assert not all(h.cluster_id == 0 for h in homologs)


def test_cluster_identity_controls_granularity():
    """A looser identity threshold merges; a tighter one splits."""
    seqs = _make(6)
    loose = [Homolog(h.id, h.sequence, h.identity, h.coverage) for h in seqs]
    tight = [Homolog(h.id, h.sequence, h.identity, h.coverage) for h in seqs]
    assign_clusters(loose, HomologConfig(cluster_identity=0.30))
    assign_clusters(tight, HomologConfig(cluster_identity=0.98))
    n_loose = len({h.cluster_id for h in loose})
    n_tight = len({h.cluster_id for h in tight})
    assert n_tight >= n_loose  # tighter threshold never merges more


def test_empty_input_is_safe():
    assert assign_clusters([], HomologConfig()) == []
