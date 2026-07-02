"""V4-10: the generic geometry path carries no FDH-specific hardcoding (ROADMAP_V4 §7.11)."""

from pathlib import Path

from evoliez.mechanism.spec import MechanismSpec, ReactionInfo, ReactionState
from evoliez.mechanism.templates import get_template

ROOT = Path(__file__).resolve().parents[2]

# FDH-specific *identity* tokens (atom/ligand names), not the word "FDH" in prose.
_FDH_IDENTITY_TOKENS = ("nadp", "formate", "c4n", "fmt", "nicotinamide")


def test_hydride_transfer_geometry_is_smarts_generic():
    tmpl = get_template("hydride_transfer")
    assert tmpl is not None
    for term in tmpl["default_geometry_terms"]:
        # chemistry-generic: selected by SMARTS on the cofactor/substrate, no protein
        # residue seqid hardcoded into the template
        assert term.get("a_smarts")
        assert term.get("a_residue") is None


def test_generic_geometry_modules_have_no_fdh_identity_tokens():
    for rel in ("src/evoliez/guided/soft_kernels.py",
                "src/evoliez/guided/geometry_sensitivity.py"):
        text = (ROOT / rel).read_text().lower()
        for token in _FDH_IDENTITY_TOKENS:
            assert token not in text, f"{rel} hardcodes FDH-specific token {token!r}"


def test_non_fdh_mechanism_uses_topology_residue_selectors():
    # a non-FDH class resolves its protein nucleophile by residue+atom name (topology),
    # not a fixed FDH sequence position -> proves the path generalizes
    spec = MechanismSpec(
        reaction=ReactionInfo(**{"class": "nucleophilic_acyl_substitution"}),
        reaction_state=ReactionState(
            protonation_model="std", conformational_state="acyl_competent"
        ),
    )
    residues = {(t.a_residue, t.a_atom) for t in spec.geometry_terms}
    assert ("SER", "OG") in residues
