"""Stage 01 - input & ligand preprocessing (spec section 6)."""

from __future__ import annotations

import re

from evoliez.context import RunContext
from evoliez.db.schema import Sequence
from evoliez.features.ligand import parse_ligand
from evoliez.stages.base import Stage

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_residue_tokens(tokens: list[str]) -> list[int]:
    """Accept '155', 'H155', 'H155A' -> 155."""
    out: list[int] = []
    for t in tokens:
        m = re.search(r"(\d+)", str(t))
        if m:
            out.append(int(m.group(1)))
    return out


class InputPreprocessStage(Stage):
    name = "s01_input"

    def run(self, ctx: RunContext) -> None:
        cfg = ctx.config.input

        if cfg.target_sequence:
            seq = cfg.target_sequence.strip().upper()
        else:
            seq = _read_fasta(cfg.target_fasta)
        seq = "".join(c for c in seq if not c.isspace())
        invalid = sorted(set(seq) - _VALID_AA)
        if invalid:
            self.log.warning("non-standard residues replaced with X: %s", invalid)
            seq = "".join(c if c in _VALID_AA else "X" for c in seq)
        if len(seq) < 10:
            raise ValueError(f"target sequence too short ({len(seq)} aa)")

        (ctx.paths.inputs / "target.fasta").write_text(
            f">{cfg.target_id}\n{seq}\n"
        )
        ligand = parse_ligand(cfg.ligand)
        (ctx.paths.inputs / "ligand.smi").write_text(
            f"{ligand.smiles}\t{ligand.id}\n"
        )

        catalytic = parse_residue_tokens(cfg.catalytic_residues)
        fixed = parse_residue_tokens(cfg.fixed_residues) or list(catalytic)
        known_site = parse_residue_tokens(cfg.known_binding_site)

        ctx.put("target_sequence", seq)
        ctx.put("ligand", ligand)
        ctx.put("catalytic_positions", catalytic)
        ctx.put("fixed_positions", sorted(set(fixed) | set(catalytic)))
        ctx.put("known_binding_site", known_site)
        ctx.persist_meta("sequence_length", len(seq))
        ctx.persist_meta("ligand_smiles", ligand.smiles)
        ctx.persist_meta("ligand_n_heavy", ligand.n_heavy)
        # canonical atom-id list (atom-index lock reference, expert review #5)
        ctx.persist_meta("ligand_atom_ids", [a.id for a in ligand.atoms])
        ctx.persist_meta("catalytic_positions", catalytic)

        assert ctx.store is not None
        with ctx.store.session() as s:
            if not s.query(Sequence).filter_by(source="target").first():
                s.add(
                    Sequence(
                        project_id=ctx.project_id,
                        fasta=f">{cfg.target_id}\n{seq}\n",
                        source="target",
                        identity_to_target=1.0,
                        coverage=1.0,
                        annotation=cfg.organism or "",
                    )
                )
        self.log.info(
            "target=%d aa, ligand=%s (%d heavy atoms), catalytic=%s",
            len(seq), ligand.id, ligand.n_heavy, catalytic,
        )

    def load(self, ctx: RunContext) -> bool:
        fa = ctx.paths.inputs / "target.fasta"
        if not fa.exists():
            return False
        seq = _read_fasta(str(fa))
        ctx.put("target_sequence", seq)
        ctx.put("ligand", parse_ligand(ctx.config.input.ligand))
        ctx.put("catalytic_positions", ctx.meta("catalytic_positions", []))
        ctx.put(
            "fixed_positions",
            sorted(
                set(parse_residue_tokens(ctx.config.input.fixed_residues))
                | set(ctx.meta("catalytic_positions", []))
            ),
        )
        ctx.put(
            "known_binding_site",
            parse_residue_tokens(ctx.config.input.known_binding_site),
        )
        return True


def _read_fasta(path: str | None) -> str:
    if not path:
        raise ValueError("no target_fasta provided")
    seq_lines: list[str] = []
    for line in open(path):
        if line.startswith(">"):
            continue
        seq_lines.append(line.strip())
    return "".join(seq_lines).upper()
