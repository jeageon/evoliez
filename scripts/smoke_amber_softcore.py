#!/usr/bin/env python
"""Phase-B increment 2: softcore_setup.py hybrid-mutation-topology smoke.

The RBFE crux is building the alchemical hybrid topology for a residue mutation.
Amber's softcore_setup.py does it natively: given WT (A) and mutant (B)
prmtop/rst it aligns B onto A and prints the TI masks (timask/scmask). This
smoke builds a WT protein prmtop + a single-residue->ALA mutant prmtop and runs
softcore_setup.py, validating the topology generation. Run with AmberTools on
PATH. Usage: python smoke_amber_softcore.py <protein.pdb> [resid]
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

print("\n=== softcore_setup.py wt -> mut ===")
r = sh(["softcore_setup.py", "wt.prmtop", "wt.rst", "mut.prmtop", "mut.rst"])
out = (r.stdout or "") + (r.stderr or "")
# show the suggested TI masks / generated files
for line in out.splitlines():
    if any(k in line.lower() for k in ("timask", "scmask", "crgmask",
                                       "generated", "written", "softcore",
                                       "new ", "perturb")):
        print("  ", line.strip())
generated = sorted(p.name for p in WORK.glob("*")
                   if p.suffix in (".new", "") and "mut" in p.name
                   and p.name not in ("mut.pdb", "mut.in", "mut_c.pdb"))
print("  generated files:", [p.name for p in WORK.glob("*.new")] or generated[:6])

ok = ("timask" in out.lower() or "scmask" in out.lower()
      or any(WORK.glob("*.new")))
print("\nSMOKE:", "PASS - softcore_setup.py produced the hybrid TI setup"
      if ok else "FAIL (check output above)")
sys.exit(0 if ok else 1)
