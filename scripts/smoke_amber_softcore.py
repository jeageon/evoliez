#!/usr/bin/env python
"""Phase-B increment 2: hybrid-mutation-topology smoke (the RBFE crux).

Builds a WT protein prmtop + a single-residue->ALA mutant prmtop and runs the
vendored, Py3-patched softcore_setup_py3.py (Amber's stock softcore_setup.py is
broken on AmberTools25/Py3 - see that file's header) to align the mutant onto WT
and emit the 3-stage TI protocol (charge->vdW-softcore->charge) with the exact
crgmask/scmask. Validates the alchemical topology generation end to end. Run
with AmberTools on PATH. Usage: python smoke_amber_softcore.py <protein.pdb> [resid]
"""
import shutil
import subprocess
import sys
from pathlib import Path

PROT = Path(sys.argv[1])
MUT = int(sys.argv[2]) if len(sys.argv) > 2 else 84
WORK = Path("/tmp/smoke_softcore")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
KEEP = {"N", "CA", "C", "O", "CB"}   # ALA backbone + CB; drop the rest


def sh(cmd, **k):
    return subprocess.run(cmd, cwd=WORK, capture_output=True, text=True, **k)


prot = [l for l in PROT.read_text().splitlines() if l.startswith("ATOM")]
# WT
(WORK / "wt.pdb").write_text("\n".join(prot) + "\nEND\n")
# mutant: residue MUT -> ALA (keep backbone+CB renamed ALA, drop other sidechain)
mut = []
for l in prot:
    if int(l[22:26]) == MUT:
        if l[12:16].strip() in KEEP:
            mut.append(l[:17] + "ALA" + l[20:])
    else:
        mut.append(l)
(WORK / "mut.pdb").write_text("\n".join(mut) + "\nEND\n")
orig = next((l[17:20].strip() for l in prot if int(l[22:26]) == MUT), "?")
print(f"[setup] mutation: residue {MUT} {orig} -> ALA")


def build(tag):
    r = sh(["pdb4amber", "-i", f"{tag}.pdb", "-o", f"{tag}_c.pdb",
            "--nohyd", "--dry"])
    (WORK / f"{tag}.in").write_text(
        f"source leaprc.protein.ff19SB\nx = loadpdb {tag}_c.pdb\n"
        f"saveamberparm x {tag}.prmtop {tag}.rst\nquit\n")
    sh(["tleap", "-s", "-f", f"{tag}.in"])
    ok = (WORK / f"{tag}.prmtop").exists()
    print(f"  {tag}.prmtop: {ok}")
    return ok


if not (build("wt") and build("mut")):
    print("SMOKE: FAIL - tleap build")
    sys.exit(1)

SC = Path(__file__).resolve().parent / "softcore_setup_py3.py"
print(f"\n=== {SC.name} wt -> mut ===")
r = sh([sys.executable, str(SC), "wt.prmtop", "wt.rst", "mut.prmtop", "mut.rst"])
out = (r.stdout or "") + (r.stderr or "")
# show the suggested 3-stage TI protocol (masks) it emits
for line in out.splitlines():
    if any(k in line.lower() for k in ("icfe", "scmask", "crgmask",
                                       "stage", "generating mut")):
        print("  ", line.strip())
sc_prm = WORK / "mut.SC.prmtop"
print("  generated:", [p.name for p in sorted(WORK.glob("mut.SC.*"))])

ok = sc_prm.exists() and "scmask" in out.lower()
print("\nSMOKE:", "PASS - softcore_setup_py3 built the hybrid topology + TI masks"
      if ok else "FAIL (check output above)")
sys.exit(0 if ok else 1)
