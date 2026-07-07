#!/usr/bin/env python3
"""Quality check of the FULL-LENGTH SrCAR + 3-HP + ATP + NADPH complex.

Verifies the 3-domain functional state everything downstream depends on:
  - fold confidence per domain (A / PCP / R)
  - 3-HP + ATP in the A-domain (near the A3 loop S268/T269/K273)
  - NADPH in the R-domain (near the SDR dyad Y970/K974 + Rossmann T798)
  - adenylation near-attack gap: 3-HP carboxylate-C -> ATP alpha-P
"""
import glob, json, os, sys
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_full"
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}

def find_top_model(run):
    for p in [f"{run}/complexes/boltz/**/*_model_0.pdb", f"{run}/complexes/boltz/**/*.pdb"]:
        h = sorted(glob.glob(p, recursive=True))
        if h: return h[0]
    return None

def parse_pdb(path):
    at = []
    for ln in open(path):
        if ln[:6] not in ("ATOM  ", "HETATM"): continue
        try: x,y,z = float(ln[30:38]),float(ln[38:46]),float(ln[46:54])
        except ValueError: continue
        b = float(ln[60:66]) if ln[60:66].strip() else 0.0
        el = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
        at.append((ln[:6].strip(), ln[21], ln[22:26].strip(), ln[17:20].strip(),
                   ln[12:16].strip(), el, np.array([x,y,z]), b))
    return at

def main():
    pdb = find_top_model(RUN)
    if not pdb: print("NO model pdb under", RUN); sys.exit(2)
    print("== model:", pdb)
    cj = None
    for c in glob.glob(os.path.dirname(pdb)+"/*model_0*.json")+glob.glob(os.path.dirname(pdb)+"/confidence*.json"):
        try: cj = json.load(open(c)); break
        except Exception: pass
    if cj:
        print("== Boltz confidence:", {k: round(cj[k],3) for k in
              ("confidence_score","ptm","iptm","ligand_iptm","complex_plddt") if k in cj})
    at = parse_pdb(pdb)
    prot = [a for a in at if a[3] in AA3]
    het = [a for a in at if a[3] not in AA3]
    groups = {}
    for a in het: groups.setdefault((a[1],a[2],a[3]), []).append(a)
    heavy = lambda ats: [a for a in ats if a[5] != "H"]
    # classify ligands by (nP, n_heavy)
    lig = {"3HP": None, "ATP": None, "NADPH": None}
    cand = []
    for key, ats in groups.items():
        h = heavy(ats); nP = sum(1 for a in h if a[5]=="P")
        cand.append((key, ats, len(h), nP))
        print(f"   het {key}: {len(h)} heavy, {nP} P")
    for key, ats, nh, nP in cand:
        if nP == 0 and 4 <= nh <= 10: lig["3HP"] = (key, ats)
    p_ligs = sorted([c for c in cand if c[3] >= 2], key=lambda c: c[2])
    if len(p_ligs) >= 1: lig["ATP"] = (p_ligs[0][0], p_ligs[0][1])      # smaller P-ligand
    if len(p_ligs) >= 2: lig["NADPH"] = (p_ligs[-1][0], p_ligs[-1][1])  # larger P-ligand
    prot_xyz = np.array([a[6] for a in prot])

    def pocket(ats, cut=4.5):
        hx = np.array([a[6] for a in heavy(ats)])
        d = np.linalg.norm(prot_xyz[:,None,:]-hx[None,:,:], axis=2)
        near = np.where(d.min(axis=1) < cut)[0]
        res = sorted({(prot[i][3], int(prot[i][2])) for i in near}, key=lambda r: r[1])
        return res, len(near)
    def resatom_xyz(rs, an=None):
        hits=[a for a in prot if int(a[2])==rs and (an is None or a[4]==an)]
        if not hits: hits=[a for a in prot if int(a[2])==rs]
        return np.array([a[6] for a in hits]) if hits else None
    def mindist(ats, rs, an=None):
        rx = resatom_xyz(rs, an);  hx = np.array([a[6] for a in heavy(ats)])
        if rx is None: return None
        return float(np.linalg.norm(rx[:,None,:]-hx[None,:,:], axis=2).min())

    for name in ("3HP","ATP","NADPH"):
        if not lig[name]: print(f"\n!! {name} not identified"); continue
        res, n = pocket(lig[name][1])
        print(f"\n== {name} {lig[name][0]}: buried atoms(<4.5A)={n}, pocket:",
              ", ".join(f"{r[0]}{r[1]}" for r in res[:16]))
    # site sanity
    print("\n== A-domain site: ATP -> A3 loop (A)")
    if lig["ATP"]:
        for rs,an in [(268,"OG"),(269,"OG1"),(273,"NZ")]:
            print(f"   {rs}({an}): {mindist(lig['ATP'][1],rs,an)}")
    print("== R-domain site: NADPH -> SDR dyad + Rossmann (A)")
    if lig["NADPH"]:
        for rs,an in [(970,"OH"),(974,"NZ"),(798,"OG1")]:
            print(f"   {rs}({an}): {mindist(lig['NADPH'][1],rs,an)}")
    # adenylation gap
    if lig["3HP"] and lig["ATP"]:
        hp=heavy(lig["3HP"][1]); atp=heavy(lig["ATP"][1])
        cC=None
        for a in hp:
            if a[5]=="C" and sum(1 for b in hp if b[5]=="O" and np.linalg.norm(a[6]-b[6])<1.7)>=2: cC=a;break
        Ps=[a for a in atp if a[5]=="P"]
        if cC is not None and Ps:
            dP=sorted(float(np.linalg.norm(cC[6]-p[6])) for p in Ps)
            print(f"\n== adenylation: 3HP carboxylate-C -> ATP P (A): "+", ".join(f"{d:.2f}" for d in dP)+
                  f"  (near-attack ~3.5; WT non-native gap expected)")
    # per-domain pLDDT
    ca = {int(a[2]): a[7] for a in prot if a[4]=="CA"}
    xs = sorted(ca); b = np.array([ca[i] for i in xs])
    print(f"\n== pLDDT overall: mean={b.mean():.1f} frac>70={100*np.mean(b>70):.0f}%")
    for lo,hi,nm in [(1,660,"A-domain"),(660,720,"PCP"),(720,1188,"R-domain")]:
        w=[ca[i] for i in xs if lo<=i<hi]
        if w: print(f"   {nm:9s} {lo}-{hi}: mean pLDDT {np.mean(w):.1f}")

if __name__ == "__main__":
    main()
