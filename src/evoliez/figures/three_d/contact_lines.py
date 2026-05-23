"""Section 5 - 3Dmol viewer with dashed ligand-residue contact lines.

Augments the standard inline 3Dmol viewer (see
:mod:`evoliez.figures.three_d.pdb_inline`) with a ``custom_script`` block
that draws yellow dashed cylinders from each top ligand atom to the
matching protein residue's Cα.  The viewer template renders the standard
``addModel``/``setStyle`` calls first, then evaluates whatever JS the
``custom_script`` field contains, so adding cylinders is a non-invasive
overlay.

Returns ``None`` when the WT complex PDB or ``edge_level.csv`` is
missing - the report builder then skips the viewer rather than emitting
an empty 3D pane.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evoliez.figures.three_d.pdb_inline import make_viewer_context
from evoliez.figures.types import ReportArtifacts

_LOGGER = logging.getLogger(__name__)

_TOP_K = 15

# Same alias lists as ``top_contacts_table`` - keep them in lock-step so a
# schema rename in one place doesn't silently break the other.
_FREQ_COLS = ("contact_frequency", "frequency", "freq")
_DIST_COLS = ("mean_distance", "distance", "mean_dist")
_RES_COLS = ("residue_index", "residue", "res_idx")
_ATOM_COLS = ("ligand_atom_id", "ligand_atom", "atom_id")


def _pick_col(row_keys, candidates):
    for k in candidates:
        if k in row_keys:
            return k
    return None


def _load_top_contacts(
    csv_path: Path, top_k: int
) -> List[Tuple[int, str, float, float]]:
    """Read top-K (residue_index, atom_id, frequency, mean_distance) rows."""
    out: List[Tuple[int, str, float, float]] = []
    try:
        with Path(csv_path).open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            res_key = _pick_col(fieldnames, _RES_COLS)
            atom_key = _pick_col(fieldnames, _ATOM_COLS)
            freq_key = _pick_col(fieldnames, _FREQ_COLS)
            dist_key = _pick_col(fieldnames, _DIST_COLS)
            if res_key is None or atom_key is None or freq_key is None:
                return []
            for row in reader:
                try:
                    residue = int(float(row[res_key]))
                    atom = str(row[atom_key]).strip()
                    freq = float(row[freq_key])
                except (TypeError, ValueError, KeyError):
                    continue
                try:
                    dist = (
                        float(row.get(dist_key))
                        if dist_key and row.get(dist_key) not in (None, "")
                        else float("nan")
                    )
                except ValueError:
                    dist = float("nan")
                out.append((residue, atom, freq, dist))
    except OSError:
        return []
    out.sort(key=lambda t: t[2], reverse=True)
    return out[: max(1, int(top_k))]


def _parse_pdb_coords(pdb_path: Path) -> Tuple[
    Dict[int, Tuple[float, float, float]],
    Dict[str, Tuple[float, float, float]],
]:
    """Return ({residue_index: CA xyz}, {ligand atom name/serial: xyz}).

    Ligand atoms come from ``HETATM`` records whose residue name is NOT
    one of the standard amino-acid 3-letter codes; both ``name`` (PDB
    columns 13-16) and ``serial`` (columns 7-11) are exposed as keys so
    the edge CSV's atom id - which may be either - resolves.
    """
    aa_codes = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL", "SEC", "PYL", "MSE",
    }
    ca: Dict[int, Tuple[float, float, float]] = {}
    lig: Dict[str, Tuple[float, float, float]] = {}
    try:
        text = Path(pdb_path).read_text()
    except OSError:
        return ca, lig

    for line in text.splitlines():
        if len(line) < 54:
            continue
        record = line[:6].strip()
        if record not in ("ATOM", "HETATM"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = None

        if record == "ATOM" and atom_name == "CA":
            ca[resnum] = (x, y, z)
            continue
        if record == "HETATM" and resname not in aa_codes and resname != "HOH":
            lig[atom_name] = (x, y, z)
            if serial is not None:
                lig[str(serial)] = (x, y, z)
    return ca, lig


def _build_custom_script(
    contacts: List[Tuple[int, str, float, float]],
    ca_coords: Dict[int, Tuple[float, float, float]],
    lig_coords: Dict[str, Tuple[float, float, float]],
) -> str:
    """Render the additional 3Dmol JS that draws dashed contact cylinders.

    3Dmol.js doesn't expose a first-class "dashed cylinder" - we
    approximate it by stamping a sequence of short cylinder segments
    along the ligand-to-CA vector.  Each segment is ~0.35 Å so the
    eye reads it as a dotted line.
    """
    lines: List[str] = []
    for residue, atom, freq, _dist in contacts:
        lig_xyz = lig_coords.get(atom)
        if lig_xyz is None:
            continue
        ca_xyz = ca_coords.get(residue)
        if ca_xyz is None:
            continue
        sx, sy, sz = lig_xyz
        ex, ey, ez = ca_xyz
        # Emit ~10 short segments with gaps to mimic a dashed line.
        n_seg = 10
        for i in range(n_seg):
            t0 = i / float(n_seg)
            t1 = t0 + (1.0 / (2 * n_seg))  # half-segment then gap
            x0 = sx + (ex - sx) * t0
            y0 = sy + (ey - sy) * t0
            z0 = sz + (ez - sz) * t0
            x1 = sx + (ex - sx) * t1
            y1 = sy + (ey - sy) * t1
            z1 = sz + (ez - sz) * t1
            lines.append(
                "v.addCylinder({{start:{{x:{:.3f},y:{:.3f},z:{:.3f}}},"
                "end:{{x:{:.3f},y:{:.3f},z:{:.3f}}},"
                "radius:0.05,fromCap:1,toCap:1,color:'yellow'}});".format(
                    x0, y0, z0, x1, y1, z1
                )
            )
    if not lines:
        return ""
    return "\n".join(lines) + "\nv.render();\n"


def render(
    artifacts: ReportArtifacts,
    *,
    viewer_id: str = "contact_lines",
    height: int = 480,
    top_k: int = _TOP_K,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Build the augmented 3Dmol viewer context for section 5.

    Returns a dict matching the ``components/threed_viewer.html`` shape,
    extended with ``custom_script`` (extra JS that the template
    evaluates after the default ``setStyle`` calls).  ``None`` when
    inputs are missing or no contacts could be wired up.
    """
    wt_pdb = getattr(artifacts, "wt_complex_pdb", None)
    if wt_pdb is None or not Path(wt_pdb).exists():
        _LOGGER.info("contact_lines: WT complex PDB missing")
        return None

    ml = getattr(artifacts, "ml_datasets", {}) or {}
    edge_csv = ml.get("edge_level")
    if edge_csv is None or not Path(edge_csv).exists():
        _LOGGER.info("contact_lines: edge_level.csv missing")
        return None

    contacts = _load_top_contacts(Path(edge_csv), top_k=top_k)
    if not contacts:
        return None

    ca_coords, lig_coords = _parse_pdb_coords(Path(wt_pdb))
    custom_script = _build_custom_script(contacts, ca_coords, lig_coords)
    if not custom_script:
        # Nothing matched between the edge CSV and the PDB coordinates -
        # the inline viewer alone would just duplicate section 03, so
        # skip rather than emit a redundant pane.
        return None

    ctx = make_viewer_context(
        Path(wt_pdb),
        viewer_id=viewer_id,
        height=height,
        title="Family-consensus contact lines (top contacts)",
    )
    if not ctx or ctx.get("missing"):
        return None
    ctx["custom_script"] = custom_script
    ctx["n_contacts"] = len(contacts)
    return ctx
