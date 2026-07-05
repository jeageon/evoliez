#!/usr/bin/env python3
"""CAR-specific near-attack (adenylation) geometry pass — STEP 1: placement + static
accommodation score.  The design ligand 3-HP is the DONOR (carboxylate O), the ATP
alpha-P (a cofactor) is the ACCEPTOR — the inverse of the FDH formate case. We build the
in-line near-attack pose (3-HP carboxylate O ~3.2 A anti to the leaving PPi bridge) and
score whether each mutant pocket ACCOMMODATES it (clash / carboxylate-anchor / wrong-pose).
Δ(mutant - WT) is the catalytic-geometry signal; a restrained-MD retention layer builds on
this placement next.  (Reviewer-aligned: distance-anchored, angle free, 5MST-grounded.)
"""
import glob, sys
import numpy as np

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
ANCHOR = {"ARG","LYS","HIS","SER","THR","ASN","GLN","TYR"}   # carboxylate H-bond/salt donors

def parse(pdb):
    P=[]; het={}
    for ln in open(pdb):
        if ln[:6] not in ("ATOM  ","HETATM"): continue
        try: xyz=np.array([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])])
        except ValueError: continue
        el=(ln[76:78].strip() or ln[12:16].strip()[0]).upper()
        rec=(ln[21],ln[22:26].strip(),ln[17:20].strip(),ln[12:16].strip(),el,xyz)
        (P.append(rec) if ln[:4]=="ATOM" and ln[17:20].strip() in AA3
         else het.setdefault(ln[21],[]).append(rec))
    return P, het

def heavy(a): return [x for x in a if x[4]!="H"]

def find_atp(het):
    for ch,ats in het.items():
        h=heavy(ats)
        if sum(1 for a in h if a[4]=="P")>=2 and len(h)>=25 and len(h)<40: return h
    return None
def find_3hp(het):
    for ch,ats in het.items():
        h=heavy(ats); nP=sum(1 for a in h if a[4]=="P")
        if nP==0 and 4<=len(h)<=8: return h
    return None

def bonded(a, atoms, cut=1.9):
    return [b for b in atoms if b is not a and np.linalg.norm(a[5]-b[5])<cut]

def alpha_P(atp):
    # alpha-P = the P bonded (via an ester O) to a carbon (the ribose C5'); beta/gamma are P-O-P
    for a in atp:
        if a[4]!="P": continue
        for o in bonded(a,atp):
            if o[4]=="O" and any(c[4]=="C" for c in bonded(o,atp) if c is not a):
                return a
    return next((a for a in atp if a[4]=="P"), None)

def leaving_dir(pA, atp):
    # anti to the alpha-P -> bridging-O(beta) bond (the PPi leaving direction)
    for o in bonded(pA,atp):
        if o[4]=="O" and any(p[4]=="P" and p is not pA for p in bonded(o,atp)):
            v=pA[5]-o[5]; return v/np.linalg.norm(v)   # points AWAY from the leaving PPi
    # fallback: away from the ester O (ribose)
    for o in bonded(pA,atp):
        if o[4]=="O" and any(c[4]=="C" for c in bonded(o,atp)):
            v=pA[5]-o[5]; return v/np.linalg.norm(v)
    return np.array([1.,0.,0.])

def carboxylate(hp):
    for c in hp:
        if c[4]!="C": continue
        os=[o for o in hp if o[4]=="O" and np.linalg.norm(c[5]-o[5])<1.7]
        if len(os)>=2: return c, os
    return None, []
def hydroxyl_O(hp, carb_os):
    return next((o for o in hp if o[4]=="O" and o not in carb_os), None)

def _rotate(atoms, origin, axis, deg):
    """Rotate atoms about (origin, unit axis) by deg (Rodrigues)."""
    th=np.radians(deg); k=axis/np.linalg.norm(axis); c,s=np.cos(th),np.sin(th)
    out=[]
    for a in atoms:
        v=a[5]-origin
        vr=v*c + np.cross(k,v)*s + k*np.dot(k,v)*(1-c)
        out.append((a[0],a[1],a[2],a[3],a[4],origin+vr))
    return out

def place_near_attack(hp, pA, ld, protein, atp, d0=3.2):
    """Place a carboxylate O at d0 A in-line to alpha-P, then SAMPLE rotations of 3-HP about
    the attack axis (x2 attacker-O choices) and keep the CLASH-MINIMISED, carboxylate-anchored,
    non-flipped pose. This is the fix for the naive rigid placement that crashed the MD run:
    resolve the pose geometrically first (reviewer: sample rotamers, avoid wrong-pose)."""
    c,os=carboxylate(hp)
    if c is None: return None
    target=pA[5]+ld*d0
    px=np.array([a[5] for a in protein])
    best=None; best_obj=1e9
    for oatt in os:                                   # each carboxylate O may be the nucleophile
        base=[(a[0],a[1],a[2],a[3],a[4],a[5]+(target-oatt[5])) for a in hp]
        axis=pA[5]-target                              # attack axis (through the placed O and alpha-P)
        for deg in range(0,360,20):
            pose=_rotate(base, target, axis, deg)
            s=score(pose, protein, atp, pA)
            # objective: minimise clashes + wrong-pose, reward carboxylate anchoring
            obj = s["clashes"] + 6*s["wrong_pose"] - 1.2*s["carboxylate_anchor"]
            if obj<best_obj: best_obj=obj; best=pose
    return best

def score(hp_placed, protein, atp, pA):
    c,os=carboxylate(hp_placed); hyd=hydroxyl_O(hp_placed,os)
    oatt=min(os,key=lambda o:np.linalg.norm(o[5]-pA[5]))
    d=np.linalg.norm(oatt[5]-pA[5])
    # attack angle O_att - P - O_leaving (want ~180 in-line)
    obr=[o for o in atp if o[4]=="O" and np.linalg.norm(o[5]-pA[5])<1.9]
    ang=np.nan
    if obr:
        oL=min(obr,key=lambda o:np.dot(o[5]-pA[5], oatt[5]-pA[5]))  # MOST ANTI to attack (leaving group)
        v1=oatt[5]-pA[5]; v2=oL[5]-pA[5]
        ang=np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1)))
    px=np.array([a[5] for a in protein])
    hx=np.array([a[5] for a in heavy(hp_placed)])
    dmin=np.linalg.norm(px[:,None,:]-hx[None,:,:],axis=2)
    clash=int((dmin<2.4).sum())                                   # heavy-atom overlaps
    # carboxylate anchor: ANCHOR-residue donor atoms within 3.5 A of a carboxylate O
    anchor=0
    for o in os:
        for i,a in enumerate(protein):
            if a[2] in ANCHOR and a[4] in ("N","O") and np.linalg.norm(a[5]-o[5])<3.5:
                anchor+=1
    # wrong-pose: hydroxyl closer to alpha-P than the carboxylate (3-HP flipped)
    wrong = 1 if (hyd is not None and np.linalg.norm(hyd[5]-pA[5]) < d) else 0
    return dict(d_OattP=round(d,2), attack_angle=round(ang,1) if ang==ang else None,
                clashes=clash, carboxylate_anchor=anchor, wrong_pose=wrong)

def analyze(pdb, label):
    P,het=parse(pdb); atp=find_atp(het); hp=find_3hp(het)
    if atp is None or hp is None: return None
    pA=alpha_P(atp); ld=leaving_dir(pA,atp)
    placed=place_near_attack(hp,pA,ld,P,atp)
    if placed is None: return None
    s=score(placed,P,atp,pA)
    print(f"  {label:22s} d(Oatt-αP)={s['d_OattP']} angle={s['attack_angle']} "
          f"clash={s['clashes']} carbox_anchor={s['carboxylate_anchor']} wrong_pose={s['wrong_pose']}")
    return s

if __name__=="__main__":
    for lbl,pat in sys.argv[1:] and [] or []:
        pass
    # usage: car_nac.py <label>=<pdb> ...
    print("=== CAR near-attack placement + static accommodation ===")
    for arg in sys.argv[1:]:
        lbl,pdb=arg.split("=",1); analyze(pdb,lbl)
