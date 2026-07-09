"""Real tool-output parser contracts, pinned against committed fixtures
(expert review: tool-validation datasets). No external tools/GPU needed.
"""

from pathlib import Path

from evoliez.adapters.boltz import (
    _load_plddt,
    _parse_cif_atoms,
    _parse_real_samples,
    _parse_real_structure,
)
from evoliez.adapters.diffdock import _parse_diffdock
from evoliez.adapters.foldx import _parse_foldx
from evoliez.adapters.gnina import _parse_gnina
from evoliez.adapters.msa_tools import (
    _parse_blast_m8,
    _parse_mmseqs_m8,
    _parse_stockholm,
)
from evoliez.adapters.rosetta import _parse_rosetta
from evoliez.adapters.vina import _parse_vina
from evoliez.config import HomologConfig, LigandInput
from evoliez.features.ligand import parse_ligand
from evoliez.types import LigandAtom

FX = Path(__file__).parent / "fixtures" / "tool_outputs"


def test_boltz_pdb_and_cif():
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="OP(O)=O"))
    cx = _parse_real_structure(FX / "boltz" / "model_0.pdb", "AGS", lig)
    assert [r.index for r in cx.structure.residues] == [1, 2, 3]
    assert cx.structure.residues[0].aa == "A"
    res, latoms, _extra = _parse_cif_atoms(FX / "boltz" / "pred.cif")   # 3-tuple API
    assert [r.index for r in res] == [1, 2, 3]
    assert len(latoms) == 2 and latoms[0].element == "O"


def test_boltz_confidence_affinity_and_plddt():
    samples = _parse_real_samples(FX / "boltz", [])
    assert samples, "no Boltz samples parsed"
    m = samples[0].metrics
    assert abs(m["ligand_iptm"] - 0.69) < 1e-6
    assert abs(m["complex_ipde"] - 2.05) < 1e-6
    assert abs(m["affinity_pred_value"] - (-6.31)) < 1e-6
    plddt = _load_plddt(FX / "boltz", 0)
    assert plddt == [85.0, 78.0, 60.0, 92.0]


def test_boltz_2_2_1_real_output_contract():
    """Pinned against ACTUAL Boltz-2.2.1 output captured on the GPU server
    (boltz_results_*/predictions/<name>/{confidence,affinity,plddt}). Guards
    the real key layout: nested dicts (chains_ptm, pair_chains_iptm) must be
    ignored without error; affinity_pred_value* + scalar confidence metrics
    must populate; plddt npz key='plddt', len = protein+ligand tokens."""
    samples = _parse_real_samples(FX / "boltz_real", [])
    assert samples, "no Boltz-2.2.1 samples parsed from real fixture"
    m = samples[0].metrics
    assert abs(m["confidence_score"] - 0.223751500248909) < 1e-9
    assert abs(m["ptm"] - 0.14932133257389069) < 1e-9
    assert abs(m["iptm"] - 0.1401701271533966) < 1e-9
    assert abs(m["ligand_iptm"] - 0.1401701271533966) < 1e-9
    assert abs(m["complex_plddt"] - 0.2446468323469162) < 1e-9
    assert abs(m["complex_pde"] - 5.7725701332092285) < 1e-9
    assert abs(m["affinity_pred_value"] - (-0.01101875863969326)) < 1e-9
    assert abs(m["affinity_probability_binary"] - 0.4802740812301636) < 1e-9
    assert abs(m["affinity_pred_value1"] - (-0.009195908904075623)) < 1e-9
    assert abs(m["affinity_pred_value2"] - (-0.012841608375310898)) < 1e-9
    # nested-dict keys must NOT leak into the flat float metrics (no crash)
    assert "chains_ptm" not in m and "pair_chains_iptm" not in m
    plddt = _load_plddt(FX / "boltz_real", 0)
    assert len(plddt) == 445  # 401 protein + 44 ligand tokens


def test_vina_pdbqt_best_mode():
    ref = [LigandAtom(id="C0", element="C", coord=(0, 0, 0))]
    pose = _parse_vina("c", FX / "vina" / "out.pdbqt", ref)
    assert abs(pose.score - (-8.7)) < 1e-9


def test_vina_real_pdbqt_heavy_atom_lock_and_rmsd():
    """Real Vina pdbqt (captured on server: REMARK VINA RESULT + AutoDock
    types). Vina emits heavy + polar H (here 5) while the canonical ligand
    has explicit RDKit H (here 7) - the parser must take the BEST model's
    score, lock on HEAVY atoms despite the count asymmetry, and still
    compute RMSD-to-reference (the old exact-count check left rmsd=None)."""
    ref = [
        LigandAtom(id="O0", element="O", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="C1", element="C", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="N2", element="N", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="P3", element="P", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="H4", element="H", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="H5", element="H", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="H6", element="H", coord=(0.0, 0.0, 0.0)),
    ]
    pose = _parse_vina("wt", FX / "vina_real" / "wt_vina_out.pdbqt", ref)
    assert abs(pose.score - (-2.565)) < 1e-9          # BEST (first) model
    assert len(pose.ligand_atoms) == 4                # heavy-locked, H dropped
    assert [a.id for a in pose.ligand_atoms] == ["O0", "C1", "N2", "P3"]
    assert all(a.element != "H" for a in pose.ligand_atoms)
    # heavy pose coords (1,1,1) vs reference (0,0,0) -> rmsd = sqrt(3)
    assert pose.rmsd_to_reference is not None
    assert abs(pose.rmsd_to_reference - 1.732) < 1e-3


def test_gnina_sdf_tag_on_next_line():
    # value is on the line AFTER `> <minimizedAffinity>`; best (lowest) = -7.85
    assert _parse_gnina(FX / "gnina" / "out.sdf") == -7.85


def test_diffdock_confidence_from_filename():
    # DiffDock: rank1_confidence<value>.sdf  (value carries its own sign)
    assert _parse_diffdock(FX / "diffdock") == -0.42


def test_blast_mmseqs_stockholm_fixture_files():
    cfg = HomologConfig(identity_min=0.2, identity_max=0.95)
    bl = _parse_blast_m8(FX / "msa" / "blast.m8", cfg)
    # pident 78.5 kept; 99.1 dropped (cap); identity is pident not qcovs
    assert [round(h.identity, 3) for h in bl] == [0.785, 0.552]
    mm = _parse_mmseqs_m8(FX / "msa" / "mmseqs.m8", HomologConfig())
    assert abs(mm[0].identity - 0.62) < 1e-6
    sto = _parse_stockholm(FX / "msa" / "jackhmmer.sto",
                           HomologConfig(identity_min=0.0, identity_max=1.0),
                           "ACDEFGHIKLMNPQRSTVWY")
    seqs = {h.sequence for h in sto}
    assert "ACDEFGHIKLMNPQRSTVWY" in seqs           # query/hitA aggregated


def test_foldx_and_rosetta_ddg():
    assert _parse_foldx(FX / "foldx") == 1.83
    # no WT/MUT labels -> documented fallback to mean-of-tokens (with a warning)
    assert _parse_rosetta(FX / "rosetta") == 3.0     # mean(2.5, 3.5)


def test_rosetta_wt_mut_ddg_is_mut_minus_wt():
    # realistic cartesian_ddg: ddG = mean(MUT total) - mean(WT total), NOT the
    # mean of every number in the file.
    ddg = _parse_rosetta(FX / "rosetta_wtmut")
    # mean(-608.12, -607.50) - mean(-612.34, -613.10) = -607.81 - -612.72
    assert abs(ddg - 4.91) < 1e-6
