#!/usr/bin/env python3
"""Kabsch CA-RMSD of the Boltz full-length model vs the experimental crystals.
  A-domain: Boltz res 17-651 vs 5MST ; R-domain: Boltz res 668-1188 vs 5MSV.
Quantifies how well the (a3m-fixed) prediction reproduces the real structures.
"""
import glob, sys
import numpy as np

RUN = "/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_full"
REF = "/mnt/data/jglee/EvoLiEZ_car/car/refs"
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}

def boltz_ca():
    p = sorted(glob.glob(f"{RUN}/complexes/boltz/**/*_model_0.pdb", recursive=True))[0]
    ca = {}
    for ln in open(p):
        if ln[:4]=="ATOM" and ln[12:16].strip()=="CA" and ln[17:20].strip() in AA3:
            ca[int(ln[22:26])] = np.array([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])])
    return ca

def cif_ca(path):
    cols=[]; ca={}; in_loop=False; reading=False
    for ln in open(path):
        s=ln.strip()
        if s=="loop_": cols=[]; in_loop=True; reading=False; continue
        if in_loop and s.startswith("_atom_site."): cols.append(s.split(".",1)[1]); continue
        if in_loop and cols and (s.startswith("ATOM") or s.startswith("HETATM")): reading=True
        if reading:
            if s.startswith("ATOM") or s.startswith("HETATM"):
                r=s.split(); idx={c:i for i,c in enumerate(cols)}
                if r[idx["label_atom_id"]]=="CA" and r[idx["label_comp_id"]] in AA3:
                    try:
                        sid=int(r[idx.get("auth_seq_id", idx["label_seq_id"])])
                        ca[sid]=np.array([float(r[idx["Cartn_x"]]),float(r[idx["Cartn_y"]]),float(r[idx["Cartn_z"]])])
                    except: pass
            else: in_loop=False; reading=False
    return ca

def kabsch_rmsd(P,Q):
    Pc=P-P.mean(0); Qc=Q-Q.mean(0)
    V,S,Wt=np.linalg.svd(Pc.T@Qc)
    d=np.sign(np.linalg.det(V@Wt)); D=np.diag([1,1,d])
    U=V@D@Wt
    Pr=Pc@U
    return np.sqrt(((Pr-Qc)**2).sum()/len(P))

def compare(name, bca, cca, lo, hi):
    common=[i for i in cca if lo<=i<=hi and i in bca]
    if len(common)<10: print(f"  {name}: too few common CA ({len(common)})"); return
    P=np.array([bca[i] for i in common]); Q=np.array([cca[i] for i in common])
    print(f"  {name}: {len(common)} CA (res {lo}-{hi}), Kabsch CA-RMSD = {kabsch_rmsd(P,Q):.2f} A")

def compare_set(name, bca, cca, resids):
    common=[i for i in resids if i in cca and i in bca]
    if len(common)<6: print(f"  {name}: too few ({len(common)})"); return
    P=np.array([bca[i] for i in common]); Q=np.array([cca[i] for i in common])
    print(f"  {name}: {len(common)} pocket CA, LOCAL Kabsch RMSD = {kabsch_rmsd(P,Q):.2f} A")

def main():
    b=boltz_ca()
    print(f"Boltz model: {len(b)} CA")
    print("GLOBAL (whole-domain; high = multidomain conformational-state difference, not fold error):")
    compare("A-domain vs 5MST", b, cif_ca(f"{REF}/5MST.cif"), 17, 651)
    compare("R-domain vs 5MSV", b, cif_ca(f"{REF}/5MSV.cif"), 668, 1188)
    print("LOCAL active-site pocket (what the design depends on):")
    POCKET=[265,268,269,271,273,275,315,316,317,320,406,407,408,409,429,430,431,432,433,434,437,438,507,519,522,629]
    compare_set("A-pocket vs 5MST", b, cif_ca(f"{REF}/5MST.cif"), POCKET)
    compare_set("A-pocket vs 5MSW", b, cif_ca(f"{REF}/5MSW.cif"), POCKET)

if __name__=="__main__": main()
