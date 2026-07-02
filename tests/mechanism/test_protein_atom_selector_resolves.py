"""V4-10: protein catalytic-residue atom selectors resolve from topology (ROADMAP_V4 §7.11)."""

from evoliez.md.geometry_spec import make_protein_resolver


def _resolver():
    residues = [
        {"resname": "SER", "seqid": 70, "atoms": {"OG": 123, "CA": 120}},
        {"resname": "LYS", "seqid": 73, "atoms": {"NZ": 200, "CA": 197}},
    ]
    return make_protein_resolver(residues)


def test_resolves_by_residue_token_and_atom():
    resolve = _resolver()
    assert resolve("SER70", "OG") == 123
    assert resolve("LYS73", "NZ") == 200


def test_resolves_by_bare_seqid_and_resname():
    resolve = _resolver()
    assert resolve("70", "OG") == 123
    assert resolve("SER", "CA") == 120


def test_unknown_selector_returns_none():
    resolve = _resolver()
    assert resolve("HIS99", "ND1") is None
    assert resolve("SER70", "ZZ") is None
