from rdkit import Chem
hp = Chem.MolFromSmiles("OCCC(=O)[O-]")
atp = Chem.MolFromSmiles("Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O")
donor = ["[OX1-]", "[O-]C=O"]
acc = ["[PX4]([OX2][CX4])", "[#15]([OX2][CH2])", "[#15]"]
print("=== 3-HP donor SMARTS (want 1 = carboxylate O) ===")
for s in donor:
    m = hp.GetSubstructMatches(Chem.MolFromSmarts(s))
    print("  %-20s -> %d %s" % (s, len(m), m))
print("=== ATP acceptor SMARTS (want 1 = alpha-P) ===")
for s in acc:
    m = atp.GetSubstructMatches(Chem.MolFromSmarts(s))
    print("  %-22s -> %d %s" % (s, len(m), m))
print("=== which P is alpha (bonded via O to a C) ===")
for a in atp.GetAtoms():
    if a.GetSymbol() == "P":
        nbrO = [n for n in a.GetNeighbors() if n.GetSymbol() == "O"]
        has_OC = any(any(nn.GetSymbol() == "C" for nn in o.GetNeighbors()) for o in nbrO)
        print("  P idx %d: alpha(O-C ester)? %s" % (a.GetIdx(), has_OC))
