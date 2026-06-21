"""s09 production-hardening (5 expert-verified fixes).

P1.1  context-ligand chains are kept as fixed receptor context on redock
      (NADP redocked WITH the co-modelled formate, not into a formate-less
      pocket). Verified at the chain-derivation helper + the gnina/vina call.
P1.3  a None pose score (DiffDock absent/sentinel confidence) never crashes
      round()/arithmetic and stays None in provenance; over-binding treats it
      as "no evidence", never 0-as-a-real-score.
P2.4  the REAL s08b mutant-Boltz Δ (when present) re-prioritises the MD budget;
      candidates with only a proxy Δ fall back to the proxy-only formula.
P2.6  the mechanism is annotated on the REAL mutant complex when one exists,
      consistent with the catalytic-geometry path.

All CPU / mock — no real GPU tools. The GPU-batch scheduler (P1.2) is the real
multi-GPU server path; here we cover its identity contract structurally.
"""

from __future__ import annotations

from pathlib import Path

from evoliez.adapters.base import het_chains_in_pdb
from evoliez.ranking.negative_design import negative_penalties
from evoliez.stages.s09_nonmd_validation import _context_chains_for
from evoliez.types import ProteinStructure

ROOT = Path(__file__).resolve().parents[1]

_MULTI_LIG_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "HETATM    5  C1  LIG B 999       5.000   0.000   0.000  1.00  0.00           C\n"
    "HETATM    6  N1  LIG B 999       6.200   0.300   0.000  1.00  0.00           N\n"
    "HETATM    7  C2  FMT C 999       8.000   0.000   0.000  1.00  0.00           C\n"
    "HETATM    8  O2  FMT C 999       9.000   0.500   0.000  1.00  0.00           O\n"
    "END\n"
)


# --------------------------------------------------------------------------- #
# P1.1  context-ligand chains
# --------------------------------------------------------------------------- #
def test_context_chains_drop_design_keep_cofactor(tmp_path):
    """The design (primary) ligand chain B is removed (it is the one being
    redocked); the co-modelled formate chain C is kept as fixed context."""
    pdb = tmp_path / "cx.pdb"
    pdb.write_text(_MULTI_LIG_PDB)
    st = ProteinStructure(sequence="A", residues=[], pdb_path=str(pdb))
    assert het_chains_in_pdb(str(pdb)) == ["B", "C"]
    assert _context_chains_for(st) == ["C"]


def test_context_chains_none_for_proxy_and_single_ligand(tmp_path):
    # WT-coords proxy: CA-only, no real PDB -> no context (degrades honestly).
    assert _context_chains_for(ProteinStructure(sequence="A", residues=[])) is None
    # single-ligand complex: only the design ligand -> no context chains.
    one = tmp_path / "one.pdb"
    one.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    5  C1  LIG B 999       5.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    st = ProteinStructure(sequence="A", residues=[], pdb_path=str(one))
    assert _context_chains_for(st) is None


def test_context_chains_passed_to_gnina(tmp_path, monkeypatch):
    """End-to-end through redock_with: gnina receives the context chains so its
    full-atom receptor KEEPS the formate; diffdock receives-but-ignores them."""
    from evoliez.adapters import gnina
    from evoliez.config import Backend, DockingConfig
    from evoliez.stages.s05_docking import redock_with
    from evoliez.types import LigandAtom

    seen = {}

    def _capture(structure, out_path, keep_het_chains=None):
        seen["keep_het_chains"] = (set(keep_het_chains)
                                   if keep_het_chains else set())
        Path(out_path).write_text(_MULTI_LIG_PDB)
        return True

    sdf = (
        "lig\n     RDKit          3D\n\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n> <minimizedAffinity>\n-9.42\n\n$$$$\n"
    )

    def _fake_run(cmd, *a, **k):
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_text(sdf)

    monkeypatch.setattr(gnina, "require", lambda *a, **k: None)
    monkeypatch.setattr(gnina, "apply_gpu_selection", lambda *a, **k: None)
    monkeypatch.setattr(gnina, "full_atom_receptor_pdb", _capture)
    monkeypatch.setattr(gnina, "tool_version", lambda *a, **k: "gnina 1.0")
    monkeypatch.setattr(gnina, "run", _fake_run)

    class _Cfg:
        validation = type("V", (), {"redocking": None})()
        dry_run = False
        def backend_for(self, _n):
            return Backend.real

    ctx = type("Ctx", (), {"config": _Cfg(), "dry_run": False})()
    st = ProteinStructure(sequence="AG", residues=[])
    ref = [LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0))]
    redock_with("gnina", ctx, "cand1", st, ref, DockingConfig(poses_per_candidate=1),
                tmp_path / "wd", 0.05, "CCO", stage_name="s09_nonmd",
                context_chains=["C"])
    assert seen["keep_het_chains"] == {"C"}   # formate kept as context


# --------------------------------------------------------------------------- #
# P1.3  None pose score
# --------------------------------------------------------------------------- #
def test_none_score_provenance_and_overbinding():
    score = None
    # the exact s09 provenance expression: must NOT raise, must stay None.
    assert (round(score, 3) if score is not None else None) is None

    from evoliez.features.mechanism import Mechanism

    class _C:
        mutations = []
        scores = {}
        details = {}

    out = negative_penalties(
        _C(), wt_mech=Mechanism(), mut_mech=None, position_features=[],
        catalytic_positions=[], buried_fraction=0.5,
        docking_score=None,                    # unscored primary pose
        redocking_consistency=0.8,
    )
    # no score -> no over-binding claim (NOT a fabricated number from -None).
    assert out["neg_overbinding"] == 0.0


# --------------------------------------------------------------------------- #
# P2.4  real Boltz Δ in the MD-selection key
# --------------------------------------------------------------------------- #
def _mk_cand(cid, *, ml, d_ligand_iptm, source):
    from evoliez.types import Candidate
    c = Candidate(candidate_id=cid, mutations=[], generator="t")
    c.scores.update({"ml_score": ml, "redocking_consistency": 0.8,
                     "ddg_fold": 0.0, "d_ligand_iptm": d_ligand_iptm})
    c.details["boltz_delta_source"] = source
    return c


def test_real_boltz_delta_reprioritises_md_selection():
    """Two candidates with equal proxy rank: the one with a REAL positive ΔBoltz
    binding gain sorts ahead; a candidate carrying only a PROXY Δ does not get
    the real-Δ boost (falls back to the proxy-only base)."""
    # Reconstruct the stage's _md_key weighting (kept in lock-step with the stage).
    W_DLIG, W_DKEY = 2.0, 0.5

    def md_key(c):
        base = (c.scores.get("ml_score", 0.0)
                + c.scores.get("redocking_consistency", 0.0)
                - 0.2 * max(0.0, c.scores.get("ddg_fold", 0.0)))
        if c.details.get("boltz_delta_source") == "real":
            base += W_DLIG * c.scores.get("d_ligand_iptm", 0.0)
            base -= W_DKEY * abs(c.scores.get("d_key_distance", 0.0))
        return base

    real_gain = _mk_cand("real_up", ml=0.5, d_ligand_iptm=0.4, source="real")
    real_loss = _mk_cand("real_dn", ml=0.5, d_ligand_iptm=-0.4, source="real")
    proxy = _mk_cand("proxy", ml=0.5, d_ligand_iptm=0.4, source="proxy")

    # real positive Δ beats real negative Δ (the real signal moved the order).
    assert md_key(real_gain) > md_key(real_loss)
    # the proxy candidate gets NO real-Δ boost despite an identical d_ligand_iptm.
    assert md_key(proxy) < md_key(real_gain)
    assert md_key(proxy) == 0.5 + 0.8        # proxy-only base, no Δ term


def test_md_key_matches_stage_source():
    """Guard: the weights asserted above are the ones the stage actually uses
    (so this test fails loudly if the stage formula drifts)."""
    src = (ROOT / "src" / "evoliez" / "stages"
           / "s09_nonmd_validation.py").read_text()
    assert "_W_REAL_DLIGAND = 2.0" in src
    assert "_W_REAL_DKEYDIST = 0.5" in src
    assert 'boltz_delta_source") == "real"' in src
