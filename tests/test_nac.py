"""Unit tests for the NAC / catalytic-power core (evoliez.md.nac).

The geometry/occupancy core is pure NumPy (always tested); the reactive-atom
identification needs RDKit (skipped where unavailable, e.g. the light venv).
"""
import numpy as np
import pytest

from evoliez.md.nac import (FDH_HYDRIDE, ReactiveSpec, identify_acceptor,
                            identify_donor, nac_from_frames,
                            nac_from_subframes, parse_protein_donor,
                            resolve_protein_donor_index,
                            resolve_reactive_indices)

_SPEC = ReactiveSpec(donor_smarts="", acceptor_smarts="",
                     distance_max=3.5, angle_min=150.0)


def test_nac_distance_and_angle_both_required():
    # atoms 0=donor_heavy, 1=transfer(H), 2=acceptor
    reactive = np.array([[0, 0, 0], [1.1, 0, 0], [3.9, 0, 0]], float)   # d2.8 ang180
    far = np.array([[0, 0, 0], [1.1, 0, 0], [11.1, 0, 0]], float)       # d10.0
    bent = np.array([[0, 0, 0], [1.1, 0, 0], [1.1, 2.8, 0]], float)     # d2.8 ang90
    res = nac_from_frames([reactive, far, bent], 0, 1, 2, _SPEC)
    assert res.n_frames == 3
    assert res.n_reactive == 1                  # only the close+linear frame
    assert abs(res.occupancy - 1 / 3) < 1e-3
    assert res.distance_min == 2.8
    assert res.angle_mean == 150.0              # (180+180+90)/3


def test_nac_all_and_none_reactive():
    near = np.array([[0, 0, 0], [1.1, 0, 0], [3.6, 0, 0]], float)       # d2.5 ang180
    assert nac_from_frames([near, near], 0, 1, 2, _SPEC).occupancy == 1.0
    far = np.array([[0, 0, 0], [1.1, 0, 0], [11, 0, 0]], float)
    assert nac_from_frames([far], 0, 1, 2, _SPEC).occupancy == 0.0


def test_nac_empty_frames():
    res = nac_from_frames([], 0, 1, 2, _SPEC)
    assert res.occupancy == 0.0 and res.n_frames == 0


_NADP = ("NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP([O-])(=O)OP([O-])(=O)OC[C@H]2O"
         "[C@@H](n3cnc4c3ncnc4N)[C@H](O)[C@@H]2OP([O-])([O-])=O)[C@@H](O)[C@H]1O")


def test_fdh_reactive_atom_identification():
    Chem = pytest.importorskip("rdkit.Chem")
    nadp = Chem.AddHs(Chem.MolFromSmiles(_NADP))
    acc = identify_acceptor(nadp, FDH_HYDRIDE)
    assert acc is not None
    a = nadp.GetAtomWithIdx(acc)
    assert a.GetSymbol() == "C" and a.GetIsAromatic()
    # the nicotinamide C4: not bonded to the ring n+
    npos = {x.GetIdx() for x in nadp.GetAtoms()
            if x.GetSymbol() == "N" and x.GetFormalCharge() == 1}
    assert not any(n.GetIdx() in npos for n in a.GetNeighbors())

    fmt = Chem.AddHs(Chem.MolFromSmiles("[O-]C=O"))
    don = identify_donor(fmt, FDH_HYDRIDE)
    assert don is not None and "transfer" in don
    assert fmt.GetAtomWithIdx(don["transfer"]).GetSymbol() == "H"


def test_resolve_reactive_indices_separate_residues():
    """The crux: donor (formate) and acceptor (NADP-C4) live in SEPARATE
    molecules / residues, each occupying its own global trajectory block. The
    resolver must find each in its own molecule and return GLOBAL indices."""
    Chem = pytest.importorskip("rdkit.Chem")
    nadp = Chem.AddHs(Chem.MolFromSmiles(_NADP))
    fmt = Chem.AddHs(Chem.MolFromSmiles("[O-]C=O"))
    # Disjoint, non-zero global blocks (NADP added first, then formate), so a
    # correct map can't accidentally pass by returning local indices.
    nadp_blk = list(range(100, 100 + nadp.GetNumAtoms()))
    fmt_blk = list(range(900, 900 + fmt.GetNumAtoms()))

    res = resolve_reactive_indices([(nadp, nadp_blk), (fmt, fmt_blk)], FDH_HYDRIDE)
    assert res is not None
    # acceptor resolved INTO the NADP block, donor INTO the formate block
    assert res["acceptor"] == nadp_blk[identify_acceptor(nadp, FDH_HYDRIDE)]
    don = identify_donor(fmt, FDH_HYDRIDE)
    assert res["donor_heavy"] == fmt_blk[don["heavy"]]
    assert res["transfer"] == fmt_blk[don["transfer"]]
    assert res["acceptor"] in nadp_blk and res["transfer"] in fmt_blk

    # order-independent: same answer if formate is added/searched first
    res2 = resolve_reactive_indices([(fmt, fmt_blk), (nadp, nadp_blk)], FDH_HYDRIDE)
    assert res2 == res


def test_resolve_reactive_indices_missing_partner():
    """Acceptor present but no donor molecule -> None (honest skip, not a fake)."""
    Chem = pytest.importorskip("rdkit.Chem")
    nadp = Chem.AddHs(Chem.MolFromSmiles(_NADP))
    nadp_blk = list(range(nadp.GetNumAtoms()))
    assert resolve_reactive_indices([(nadp, nadp_blk)], FDH_HYDRIDE) is None


def test_nac_from_subframes_rows_are_dha():
    # rows [donor_heavy, transfer, acceptor]; close+linear -> reactive
    sub = np.array([[0, 0, 0], [1.1, 0, 0], [3.6, 0, 0]], float)  # d2.5 ang180
    res = nac_from_subframes([sub, sub], FDH_HYDRIDE,
                             atoms={"donor_heavy": 5, "transfer": 6, "acceptor": 7})
    assert res.occupancy == 1.0 and res.n_frames == 2
    assert res.atoms == {"donor_heavy": 5, "transfer": 6, "acceptor": 7}


def test_nac_engine_dataflow_contract():
    """Mirror the OpenMM engine's EXACT NAC data-flow: resolve donor/acceptor to
    GLOBAL indices from per-molecule (rdkit, block) pairs, then for each MD frame
    slice those three atoms out of the FULL-system coord array (stored in nm) and
    scale to Angstrom -- exactly what `_run_real` does with
    ``cur[list(nac_idx)] * 10.0``. Proves the index arithmetic + unit handling
    the (server-only) OpenMM execution relies on."""
    Chem = pytest.importorskip("rdkit.Chem")
    nadp = Chem.AddHs(Chem.MolFromSmiles(_NADP))
    fmt = Chem.AddHs(Chem.MolFromSmiles("[O-]C=O"))
    # global topology blocks as the engine assigns them (protein first at 0..,
    # then each ligand a contiguous block in add order: NADP then formate)
    base = 1000
    nadp_blk = list(range(base, base + nadp.GetNumAtoms()))
    fmt_blk = list(range(base + nadp.GetNumAtoms(),
                         base + nadp.GetNumAtoms() + fmt.GetNumAtoms()))
    nac = resolve_reactive_indices([(nadp, nadp_blk), (fmt, fmt_blk)], FDH_HYDRIDE)
    assert nac is not None
    dh, tr, ac = nac["donor_heavy"], nac["transfer"], nac["acceptor"]
    assert dh in fmt_blk and tr in fmt_blk and ac in nadp_blk

    # full-system coords in Angstrom; place the 3 reacting atoms productively
    # (H->acceptor 2.8 Å, donor_heavy-H-acceptor 180°), rest at origin
    n_atoms = base + nadp.GetNumAtoms() + fmt.GetNumAtoms()
    ang = np.zeros((n_atoms, 3))
    ang[dh], ang[tr], ang[ac] = [0.0, 0, 0], [1.1, 0, 0], [3.9, 0, 0]
    cur_nm = ang / 10.0                       # engine holds positions in nm
    # engine's per-frame slice: cur[list(nac_idx)] * 10.0  (nm -> Å)
    subframes = [cur_nm[[dh, tr, ac]] * 10.0 for _ in range(5)]
    res = nac_from_subframes(subframes, FDH_HYDRIDE, atoms=nac)
    assert res.n_frames == 5 and res.occupancy == 1.0
    assert res.distance_min == 2.8 and res.atoms["acceptor"] == ac


# --------------------------------------------------------------------------- #
# PROTEIN-nucleophile NAC (serine hydrolase / protease): the nucleophile is a
# PROTEIN catalytic-residue atom (e.g. Ser Ogamma), not a ligand atom. Parser +
# topology resolver are pure-python (local); full resolution needs RDKit (server).
# --------------------------------------------------------------------------- #
class _MockAtom:
    def __init__(self, name, index):
        self.name, self.index = name, index


class _MockResidue:
    def __init__(self, name, rid, atoms):
        self.name, self.id = name, rid
        self._atoms = [_MockAtom(n, i) for n, i in atoms]

    def atoms(self):
        return iter(self._atoms)


class _MockTopology:
    """Minimal duck-type of the OpenMM Topology API used by resolve_protein_donor_index."""
    def __init__(self, residues):
        self._res = [_MockResidue(n, rid, ats) for n, rid, ats in residues]

    def residues(self):
        return iter(self._res)


def test_parse_protein_donor():
    assert parse_protein_donor("SER:OG:68") == ("SER", "OG", 68)
    assert parse_protein_donor("ser:OG") == ("SER", "OG", None)     # resnum optional, resname upper
    for bad in ("SER", "SER:OG:68:x", "SER:OG:"):
        with pytest.raises(ValueError):
            parse_protein_donor(bad)


def test_resolve_protein_donor_index():
    topo = _MockTopology([
        ("ALA", "67", [("N", 10), ("CA", 11), ("CB", 12)]),
        ("SER", "68", [("N", 20), ("CA", 21), ("CB", 22), ("OG", 23)]),   # the catalytic Ser68
        ("SER", "99", [("OG", 40)]),                                       # a decoy Ser
    ])
    assert resolve_protein_donor_index(topo, ReactiveSpec("", "", donor_protein="SER:OG:68")) == 23
    # resnum disambiguates against the decoy Ser
    assert resolve_protein_donor_index(topo, ReactiveSpec("", "", donor_protein="SER:OG:99")) == 40
    # unset donor_protein / missing residue / missing atom -> None (honest skip)
    assert resolve_protein_donor_index(topo, ReactiveSpec("", "")) is None
    assert resolve_protein_donor_index(topo, ReactiveSpec("", "", donor_protein="CYS:SG:68")) is None
    assert resolve_protein_donor_index(topo, ReactiveSpec("", "", donor_protein="SER:OD:68")) is None


def test_reactivespec_protein_donor_forces_non_hydride():
    # a protein O/S nucleophile attacks directly; the spec must coerce transfer_is_h=False so the
    # near-attack angle uses the Nuc--C(=O) reference, not the degenerate donor==transfer angle.
    assert ReactiveSpec("", "", donor_protein="SER:OG:68", transfer_is_h=True).transfer_is_h is False
    assert ReactiveSpec("[O-]C=O", "", transfer_is_h=True).transfer_is_h is True   # ligand path intact


def test_resolve_reactive_indices_protein_donor_requires_topology():
    # protein donor declared but no topology supplied -> None (cannot fake a protein index)
    spec = ReactiveSpec("", "[CX3](=[OX1])[NX3]", donor_protein="SER:OG:68")
    assert resolve_reactive_indices([], spec, topology=None) is None


def test_resolve_reactive_indices_protein_donor():
    """Serine-hydrolase geometry: donor = protein Ser68 Ogamma (from topology), acceptor = the
    ligand scissile amide carbonyl C, leaving/reference = its carbonyl O (Nuc--C=O angle)."""
    Chem = pytest.importorskip("rdkit.Chem")
    # N-methylacetamide as a minimal scissile amide: CH3-C(=O)-NH-CH3
    lig = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC"))
    lig_blk = list(range(500, 500 + lig.GetNumAtoms()))
    topo = _MockTopology([
        ("SER", "68", [("N", 20), ("CA", 21), ("CB", 22), ("OG", 23)]),
    ])
    spec = ReactiveSpec(donor_smarts="", acceptor_smarts="[CX3](=[OX1])[NX3]",
                        donor_protein="SER:OG:68", transfer_is_h=False,
                        distance_max=3.5, angle_min=95.0, label="ser_attack")
    res = resolve_reactive_indices([(lig, lig_blk)], spec, topology=topo)
    assert res is not None
    assert res["donor_heavy"] == 23 and res["transfer"] == 23      # the protein Ogamma, direct attack
    acc_local = identify_acceptor(lig, spec)
    assert res["acceptor"] == lig_blk[acc_local]                   # carbonyl C in the ligand block
    # leaving = the carbonyl O double-bonded to the acceptor C (Nuc--C=O reference)
    assert "leaving" in res
    o_local = res["leaving"] - 500
    assert lig.GetAtomWithIdx(o_local).GetSymbol() == "O"
    bond = lig.GetBondBetweenAtoms(acc_local, o_local)
    assert bond is not None and bond.GetBondTypeAsDouble() == 2.0

    # acceptor absent in the ligand -> None (honest skip, never a fabricated donor-only map)
    noacc = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert resolve_reactive_indices([(noacc, list(range(noacc.GetNumAtoms())))],
                                    spec, topology=topo) is None
