"""Curated redox-cofactor library + pH/redox-aware resolver + formula guard.

Why this exists: a config that declares ``cofactor: NADP`` but supplies an
NAD+ SMILES (44 heavy / 2 P / 14 O instead of NADP+'s 48 / 3 / 17) is a
SILENT data-integrity bug that propagates all the way to MD and re-surfaces
there as "GAFF can't parameterise NADP" - a wrong diagnosis. The fix is to
1) resolve cofactor declarations to a curated, biology-correct species
keyed on (cofactor, pH, redox_state) and 2) reject mismatches between the
declaration and the actual ligand at the EARLIEST point (``doctor`` +
``s01_input``), never at MD.

For drug-like ligands (no curated entry) this module is a no-op; declare an
explicit SMILES like always.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from evoliez.logging_utils import get_logger

log = get_logger("evoliez.cofactors")


# --------------------------------------------------------------------------- #
# Curated library
# --------------------------------------------------------------------------- #
# Canonical biological-pH (~7.4) protonation: all phosphates fully
# deprotonated; nicotinamide oxidized = N1(+) pyridinium, reduced = 1,4-
# dihydropyridine; adenine N1 neutral (pKa ~3.5). These SMILES are the
# species ACTUALLY used by openff/openmmforcefields tutorials and by the
# Boltz CCD references for NAD/NADP/NADH/NADPH at pH 7.4. Counted formulas
# are derived from these SMILES (verified by RDKit if installed, by the
# regex element-counter otherwise; see :func:`formula_of`).
@dataclass(frozen=True)
class CofactorSpec:
    name: str                 # canonical species name (e.g. "NADP+", "NADPH")
    family: str               # cofactor family (e.g. "NADP")
    redox_state: str          # "oxidized" | "reduced"
    smiles: str
    formal_charge: int
    formula: Dict[str, int]   # element counts (heavy atoms only, no H)
    # Curated AMBER parameter pointers (Bryce Lab / Manchester database).
    # Filenames are RELATIVE to amber_params_root() (env EVOLIEZ_AMBER_PARAMS
    # or <repo>/amber/cofactors/). When both files exist the curated tleap
    # path runs instead of GAFF/AM1-BCC - which, on real NADP+, hard-fails
    # sqm (sqm returns non-zero) in addition to being too slow for a
    # production per-mutant pipeline. amber_residue_name is the 3-letter
    # PDB resname inside the .lib (HETATM resname in tleap-built prmtops).
    amber_lib: Optional[str] = None        # e.g. "NAP.lib"
    amber_frcmod: Optional[str] = None     # e.g. "NAP.frcmod"
    amber_residue_name: str = "LIG"

    @property
    def n_heavy(self) -> int:
        return sum(v for k, v in self.formula.items() if k != "H")

    def resolved_amber_files(
        self, base: Optional[Path] = None
    ) -> Optional[Tuple[Path, Path]]:
        """``(lib_path, frcmod_path)`` if both curated files exist on disk
        under ``base`` (defaults to :func:`amber_params_root`), else None.
        """
        if not (self.amber_lib and self.amber_frcmod):
            return None
        b = Path(base) if base else amber_params_root()
        lib, fr = b / self.amber_lib, b / self.amber_frcmod
        if lib.exists() and fr.exists():
            return (lib, fr)
        return None


def lookup_by_smiles(smiles: str) -> Optional[CofactorSpec]:
    """Reverse-lookup a curated species from a ligand SMILES.

    Canonical SMILES match (RDKit if available, exact byte match
    otherwise; with the resolver path the byte match always holds). The
    MD curated dispatcher uses this to decide whether the incoming
    ligand is a known cofactor without needing extra wiring through
    Complex/Ligand. NAD+/NADH and NADP+/NADPH share heavy-atom formulas
    but differ in pyridinium vs 1,4-dihydropyridine SMILES, which the
    canonical form distinguishes.
    """
    if not smiles:
        return None
    try:
        from rdkit import Chem                       # type: ignore

        target = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        ref_canon = {
            Chem.MolToSmiles(Chem.MolFromSmiles(spec.smiles)): spec
            for spec in _LIB.values()
        }
        return ref_canon.get(target)
    except Exception:
        # Light env / unparseable SMILES: fall back to byte match (the
        # cofactor resolver writes the curated SMILES verbatim, so it
        # round-trips identically).
        for spec in _LIB.values():
            if spec.smiles == smiles:
                return spec
        return None


def amber_params_root() -> Path:
    """Where curated AMBER cofactor params live. ``EVOLIEZ_AMBER_PARAMS``
    wins; otherwise ``<repo>/amber/cofactors``. The directory is created
    by ``scripts/fetch_amber_cofactors.sh`` (Bryce Lab DB)."""
    ev = os.environ.get("EVOLIEZ_AMBER_PARAMS")
    if ev:
        return Path(ev).expanduser().resolve()
    # walk up from this file: src/evoliez/features/cofactors.py -> repo root
    here = Path(__file__).resolve()
    repo = here.parents[3]
    return repo / "amber" / "cofactors"


# Per-(family, redox_state) canonical species. pH determines protonation;
# in the biologically relevant 5.5-9 window all phosphates are fully
# deprotonated (pKa2 ~6.5) so a single canonical SMILES covers the range
# (resolve_cofactor() WARNs outside it rather than fabricating variants).
_LIB: Dict[Tuple[str, str], CofactorSpec] = {
    ("NAD", "oxidized"): CofactorSpec(
        name="NAD+", family="NAD", redox_state="oxidized",
        smiles=(
            "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP([O-])(=O)OP([O-])(=O)"
            "OC[C@H]2O[C@@H](n3cnc4c3ncnc4N)[C@H](O)[C@@H]2O)"
            "[C@@H](O)[C@H]1O"
        ),
        formal_charge=-1,
        formula={"C": 21, "N": 7, "O": 14, "P": 2},
        amber_lib="NAD.lib", amber_frcmod="NAD.frcmod",
        amber_residue_name="NAD",          # AMBER convention; Bryce Lab DB
    ),
    ("NAD", "reduced"): CofactorSpec(           # NADH: 1,4-dihydropyridine
        name="NADH", family="NAD", redox_state="reduced",
        smiles=(
            "NC(=O)C1=CN([C@@H]2O[C@H](COP([O-])(=O)OP([O-])(=O)"
            "OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)"
            "[C@@H](O)[C@H]2O)C=CC1"
        ),
        formal_charge=-2,
        formula={"C": 21, "N": 7, "O": 14, "P": 2},
        amber_lib="NDH.lib", amber_frcmod="NDH.frcmod",
        amber_residue_name="NDH",
    ),
    ("NADP", "oxidized"): CofactorSpec(         # NADP+: NAD+ + 2'-phosphate
        name="NADP+", family="NADP", redox_state="oxidized",
        smiles=(
            "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP([O-])(=O)OP([O-])(=O)"
            "OC[C@H]2O[C@@H](n3cnc4c3ncnc4N)[C@H](OP([O-])([O-])=O)"
            "[C@@H]2O)[C@@H](O)[C@H]1O"
        ),
        formal_charge=-3,
        formula={"C": 21, "N": 7, "O": 17, "P": 3},
        amber_lib="NAP.lib", amber_frcmod="NAP.frcmod",
        amber_residue_name="NAP",
    ),
    ("NADP", "reduced"): CofactorSpec(          # NADPH
        name="NADPH", family="NADP", redox_state="reduced",
        smiles=(
            "NC(=O)C1=CN([C@@H]2O[C@H](COP([O-])(=O)OP([O-])(=O)"
            "OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP([O-])([O-])=O)"
            "[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1"
        ),
        formal_charge=-4,
        formula={"C": 21, "N": 7, "O": 17, "P": 3},
        amber_lib="NDP.lib", amber_frcmod="NDP.frcmod",
        amber_residue_name="NDP",
    ),
}

# Aliases users may write in YAML.
_FAMILY_ALIASES = {
    "nad": "NAD", "nad+": "NAD", "nadh": "NAD",
    "nadp": "NADP", "nadp+": "NADP", "nadph": "NADP",
}

# A literal redox-state hint baked into the family name (e.g. "NADPH" -> reduced).
_FAMILY_REDOX_HINT = {
    "nadh": "reduced", "nadph": "reduced",
    "nad+": "oxidized", "nadp+": "oxidized",
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def is_known_cofactor(name: Optional[str]) -> bool:
    return bool(name) and name.strip().lower() in _FAMILY_ALIASES


def resolve_cofactor(
    name: str,
    pH: float = 7.4,
    redox_state: Optional[str] = None,
) -> CofactorSpec:
    """Resolve ``(cofactor name, pH, redox_state)`` to a curated species.

    - ``name`` is matched case-insensitively against the family aliases
      (``NAD``, ``NAD+``, ``NADH``, ``NADP``, ``NADP+``, ``NADPH``); if it
      already encodes a redox state (``NADH``, ``NADP+``, ...) that wins.
    - ``pH`` is honoured for the warning surface only: in the 5.5-9 window
      all phosphates are fully deprotonated and the canonical SMILES is
      correct; outside that window we WARN and still return the canonical
      form (override with an explicit ``ligand.value`` SMILES).
    - ``redox_state`` defaults to ``"oxidized"`` (the binding/transition
      proxy used for NAD(P)+-dependent enzymes) with a WARN suggesting an
      explicit declaration.
    """
    key = (name or "").strip().lower()
    if key not in _FAMILY_ALIASES:
        raise KeyError(
            f"unknown cofactor {name!r}; known: "
            f"{sorted({s.name for s in _LIB.values()})}"
        )
    family = _FAMILY_ALIASES[key]
    hinted = _FAMILY_REDOX_HINT.get(key)
    if redox_state is None and hinted is not None:
        redox_state = hinted                      # e.g. cofactor: NADPH
    if redox_state is None:
        redox_state = "oxidized"
        log.warning(
            "cofactor %s has no explicit redox_state; defaulting to "
            "'oxidized' (NAD(P)+-dependent binding proxy). Set "
            "input.cofactor_redox to silence this.", family,
        )
    redox_state = redox_state.lower()
    if redox_state not in ("oxidized", "reduced"):
        raise ValueError(
            f"cofactor_redox must be 'oxidized' or 'reduced', got "
            f"{redox_state!r}"
        )
    if not 5.5 <= pH <= 9.0:
        log.warning(
            "target_ph=%.2f is outside the 5.5-9 window where the canonical "
            "%s SMILES (fully-deprotonated phosphates) is correct; results "
            "may not reflect the actual protonation state - override with "
            "an explicit ligand.value SMILES if needed.", pH, family,
        )
    return _LIB[(family, redox_state)]


# --------------------------------------------------------------------------- #
# Formula counting (RDKit if present; light regex otherwise)
# --------------------------------------------------------------------------- #
# Bracketed atoms ([Na+], [O-], [nH], ...) and bare element symbols. SMILES
# bare-atom organic subset: B, C, N, O, S, P, F, Cl, Br, I (case-sensitive,
# lower-case = aromatic). Anything else MUST be in brackets, so this covers
# the curated cofactors and arbitrary SMILES alike (H is implicit).
_BRACKET = re.compile(r"\[([^\]]+)\]")
_BARE = re.compile(r"Cl|Br|[BCNOPSFI]|[bcnops]")


def formula_of(smiles: str) -> Dict[str, int]:
    """Heavy-atom element counts from a SMILES.

    Uses RDKit if available (exact); otherwise a regex element-counter that
    handles bracketed and bare-atom forms (sufficient for the cofactor guard
    - this is the count we COMPARE, not chemistry we reason about). Both
    paths exclude hydrogens (heavy atoms only).
    """
    try:
        from rdkit import Chem                   # type: ignore

        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        out: Dict[str, int] = {}
        for atom in m.GetAtoms():
            sym = atom.GetSymbol()
            if sym == "H":
                continue
            out[sym] = out.get(sym, 0) + 1
        return out
    except ImportError:
        pass

    out2: Dict[str, int] = {}

    def _bump(sym: str) -> None:
        s = sym.capitalize()
        if s == "H":
            return
        out2[s] = out2.get(s, 0) + 1

    s = smiles
    pos = 0
    while pos < len(s):
        if s[pos] == "[":
            m = _BRACKET.match(s, pos)
            if not m:
                pos += 1
                continue
            inner = m.group(1)
            # strip isotope digits, charge, H-count, atom-map etc.
            inner = re.sub(r"^[0-9]+", "", inner)
            am = re.match(r"([A-Za-z][a-z]?)", inner)
            if am:
                _bump(am.group(1))
            pos = m.end()
            continue
        m = _BARE.match(s, pos)
        if m:
            _bump(m.group(0))
            pos = m.end()
            continue
        pos += 1
    return out2


def formula_from_atoms(atoms) -> Dict[str, int]:
    """Element counts from a parsed :class:`Ligand`'s atoms (no H)."""
    out: Dict[str, int] = {}
    for a in atoms:
        el = (a.element or "").strip().capitalize()
        if not el or el == "H":
            continue
        out[el] = out.get(el, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Guard: cofactor declaration <-> actual ligand formula
# --------------------------------------------------------------------------- #
@dataclass
class GuardResult:
    ok: bool
    expected: Optional[CofactorSpec] = None
    actual: Dict[str, int] = field(default_factory=dict)
    message: str = ""


def resolve_ligand_spec(input_cfg):
    """Rewrite ``input.ligand`` so that ``type='cofactor'`` becomes a real
    ``type='smiles'`` LigandInput at the resolved species' SMILES.

    No-op for any other ligand type (drug-like SMILES / SDF / PDB / ...).
    Centralising this keeps every caller (s01_input, scripts/check_real_md,
    s01.load) on the same resolved SMILES, so the formula guard downstream
    has a single source of truth.
    """
    from evoliez.config import LigandInput

    lig = input_cfg.ligand
    if (lig.type or "").lower() != "cofactor":
        return lig
    spec = resolve_cofactor(
        lig.value,
        pH=getattr(input_cfg, "target_ph", 7.4),
        redox_state=getattr(input_cfg, "cofactor_redox", None),
    )
    log.info(
        "resolved cofactor %r at pH=%.2f -> %s (%d heavy, charge %+d)",
        lig.value, getattr(input_cfg, "target_ph", 7.4), spec.name,
        spec.n_heavy, spec.formal_charge,
    )
    return LigandInput(id=lig.id, type="smiles", value=spec.smiles)


def check_cofactor_matches(
    cofactor: Optional[str],
    actual: Dict[str, int],
    *,
    pH: float = 7.4,
    redox_state: Optional[str] = None,
) -> GuardResult:
    """Compare an actual ligand formula against the resolved cofactor.

    Returns ``ok=True`` if no cofactor is declared, the declaration is not
    in the curated library (drug-like ligand, no curated entry), or the
    actual heavy-atom formula matches the resolved species. Otherwise
    ``ok=False`` with a precise diff message naming the most likely actual
    species ("looks like NAD+") so the caller can fail fast.
    """
    if not is_known_cofactor(cofactor):
        return GuardResult(ok=True, actual=actual,
                           message="no curated cofactor declared")
    spec = resolve_cofactor(cofactor, pH=pH, redox_state=redox_state)
    exp = spec.formula
    if all(actual.get(k, 0) == v for k, v in exp.items()) and \
            sum(actual.values()) == sum(exp.values()):
        return GuardResult(ok=True, expected=spec, actual=actual,
                           message=f"matches {spec.name}")

    # Identify the most likely actual species among the curated library so
    # the error tells the user what they ACTUALLY supplied, not just a diff.
    looks_like = None
    for cand in _LIB.values():
        if cand.formula == actual:
            looks_like = cand.name
            break

    diff_keys = sorted(set(exp) | set(actual))
    diff = ", ".join(
        f"{k}: expected {exp.get(k, 0)} got {actual.get(k, 0)}"
        for k in diff_keys if exp.get(k, 0) != actual.get(k, 0)
    )
    msg = (
        f"cofactor={cofactor!r} (pH={pH}, redox={spec.redox_state}) expects "
        f"{spec.name} formula {exp} ({spec.n_heavy} heavy) but the parsed "
        f"ligand is {actual} ({sum(actual.values())} heavy)"
        + (f"; looks like {looks_like}" if looks_like else "")
        + f". Diff: {diff}. Either fix input.cofactor or supply the matching "
        "ligand SMILES (or set input.ligand.type='cofactor' to auto-resolve)."
    )
    return GuardResult(ok=False, expected=spec, actual=actual, message=msg)
