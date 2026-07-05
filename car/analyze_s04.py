#!/usr/bin/env python3
"""Deep quality check of the s04 Boltz complex for SrCAR + 3-HP + ATP.

Discovers the top-ranked Boltz model, reports Boltz confidence, then checks the
data question that everything downstream depends on:
  - did 3-HP land in a BURIED pocket (not solvent)?
  - what residues line the 3-HP pocket (the design targets)?
  - is 3-HP's carboxylate near the ATP alpha-phosphate (adenylation-competent)?
  - is the A3 catalytic loop (S268/T269/K273) near ATP (site sanity)?
Pure-stdlib PDB parse + numpy; no gemmi/biotite dependency.
"""
import glob, json, os, sys
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_prod"
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}

def find_top_model(run):
    pats = [f"{run}/complexes/boltz/**/*_model_0.pdb", f"{run}/complexes/boltz/**/*model_0.pdb",
            f"{run}/complexes/boltz/**/*.pdb"]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    return None

def parse_pdb(path):
    atoms = []  # (rec, chain, resseq, resname, atomname, elem, xyz, bfac)
    for ln in open(path):
        if ln[:6] not in ("ATOM  ", "HETATM"):
            continue
        try:
            x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
        except ValueError:
            continue
        bfac = float(ln[60:66]) if ln[60:66].strip() else 0.0
        elem = ln[76:78].strip() or ln[12:16].strip()[0]
        atoms.append((ln[:6].strip(), ln[21], ln[22:26].strip(), ln[17:20].strip(),
                      ln[12:16].strip(), elem.upper(), np.array([x, y, z]), bfac))
    return atoms

def main():
    pdb = find_top_model(RUN)
    if not pdb:
        print("NO Boltz model pdb found under", RUN); sys.exit(2)
    print("== model:", pdb)
    # confidence json alongside
    cj = None
    for c in glob.glob(os.path.dirname(pdb) + "/*model_0*.json") + glob.glob(os.path.dirname(pdb) + "/confidence*.json"):
        try: cj = json.load(open(c)); cjp = c; break
        except Exception: pass
    if cj:
        print("== Boltz confidence (", os.path.basename(cjp), ")")
        for k in ("confidence_score","ptm","iptm","ligand_iptm","protein_iptm",
                  "complex_plddt","complex_iplddt","complex_pde","complex_ipde"):
            if k in cj: print(f"   {k:16s} {cj[k]}")
    atoms = parse_pdb(pdb)
    prot = [a for a in atoms if a[3] in AA3]
    het = [a for a in atoms if a[3] not in AA3]
    # group hetero by (chain,resseq,resname)
    groups = {}
    for a in het:
        groups.setdefault((a[1], a[2], a[3]), []).append(a)
    print(f"== atoms: {len(prot)} protein, {len(het)} hetero in {len(groups)} groups")
    for key, ats in groups.items():
        elems = {}
        for a in ats: elems[a[5]] = elems.get(a[5], 0) + 1
        print(f"   het {key} : {len(ats)} atoms  {elems}")
    # classify 3HP (~6 heavy: C3O3) vs ATP (~31 heavy incl 3P)
    def heavy(ats): return [a for a in ats if a[5] != "H"]
    hp = atp = None
    for key, ats in groups.items():
        h = heavy(ats); nP = sum(1 for a in h if a[5] == "P")
        if nP >= 2: atp = (key, ats)
        elif 4 <= len(h) <= 10 and nP == 0: hp = (key, ats)
    prot_xyz = np.array([a[6] for a in prot])
    def contacts_of(ats, cut=4.5):
        hx = np.array([a[6] for a in heavy(ats)])
        d = np.linalg.norm(prot_xyz[:, None, :] - hx[None, :, :], axis=2)
        near = np.where(d.min(axis=1) < cut)[0]
        res = {}
        for i in near:
            a = prot[i]; res.setdefault((a[1], int(a[2]), a[3]), 1)
        return sorted(res.keys(), key=lambda r: r[1]), int(len(near))//1

    if hp:
        cons, nnear = contacts_of(hp[1])
        print(f"\n== 3-HP  {hp[0]}  buried-contact protein atoms(<4.5A)= {nnear}")
        print("   pocket residues (design targets):",
              ", ".join(f"{r[2]}{r[1]}" for r in cons) or "(none — SOLVENT EXPOSED!)")
    else:
        print("\n!! 3-HP group not identified among heteros")
    # adenylation geometry: 3HP carboxylate C -> ATP P atoms
    if hp and atp:
        hp_h = heavy(hp[1]); atp_h = heavy(atp[1])
        # carboxylate carbon of 3HP = the C bonded to 2 O (approx: C with >=2 O within 1.6A)
        carbox_C = None
        for a in hp_h:
            if a[5] != "C": continue
            nO = sum(1 for b in hp_h if b[5]=="O" and np.linalg.norm(a[6]-b[6])<1.7)
            if nO >= 2: carbox_C = a; break
        Ps = [a for a in atp_h if a[5]=="P"]
        if carbox_C is not None and Ps:
            dP = sorted(np.linalg.norm(carbox_C[6]-p[6]) for p in Ps)
            print(f"\n== adenylation geometry: 3HP carboxylate-C -> ATP P atoms (A): "
                  + ", ".join(f"{d:.2f}" for d in dP))
            print(f"   nearest P = {dP[0]:.2f} A  (in-line attack target ~3.0-3.5; <5 = adenylation-competent pose)")
    # catalytic loop sanity: S268/T269/K273 functional atom -> ATP
    if atp:
        atp_x = np.array([a[6] for a in heavy(atp[1])])
        want = {268:("SER","OG"),269:("THR","OG1"),273:("LYS","NZ")}
        print("\n== A3 catalytic loop -> ATP (nearest heavy-atom dist, A):")
        for rs,(rn,an) in want.items():
            fa = [a for a in prot if int(a[2])==rs and a[4]==an]
            if not fa:
                fa = [a for a in prot if int(a[2])==rs]  # any atom of that residue
            if fa:
                dmin = min(np.linalg.norm(fa0[6]-atp_x, axis=1).min() for fa0 in fa)
                got = prot and [a for a in prot if int(a[2])==rs][0][3]
                print(f"   {rn}{rs} ({an}) : {dmin:.2f}   [pdb resname={got}]")
            else:
                print(f"   {rn}{rs} : residue not found in model")
    # global plddt from protein B-factors
    if prot:
        b = np.array([a[7] for a in prot])
        print(f"\n== per-residue pLDDT (protein B-col): mean={b.mean():.1f} min={b.min():.1f} "
              f"frac>70={100*np.mean(b>70):.0f}% frac>90={100*np.mean(b>90):.0f}%")

if __name__ == "__main__":
    main()
