"""Stage 01 - input & ligand preprocessing (spec section 6)."""

from __future__ import annotations

import re

from evoliez.adapters.base import mock_fallback_allowed
from evoliez.config import Backend
from evoliez.context import RunContext
from evoliez.db.schema import Sequence
from evoliez.features.ligand import parse_ligand, resolve_ligand_manifest
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


def _residue_token_wt(token: str):
    """('H155A') -> ('H', 155). Leading letter = asserted WT identity."""
    m = re.match(r"\s*([A-Za-z]?)\s*(\d+)", str(token))
    if not m:
        return None, None
    return (m.group(1).upper() or None), int(m.group(2))


def validate_residue_tokens(tokens, seq: str, kind: str, log) -> None:
    """A token like 'H155' asserts His at position 155. Silently ignoring
    the letter (old behaviour) lets a wrong numbering/sequence mis-place the
    catalytic site and corrupt every downstream result. Warn loudly on
    out-of-range positions or WT-letter / sequence mismatch."""
    for t in tokens or []:
        wt, pos = _residue_token_wt(t)
        if pos is None:
            continue
        if pos < 1 or pos > len(seq):
            log.warning(
                "%s residue token %r: position %d is out of range "
                "(target sequence is %d aa)", kind, str(t), pos, len(seq)
            )
            continue
        actual = seq[pos - 1]
        if wt and wt != "X" and actual != "X" and actual != wt:
            log.warning(
                "%s residue token %r asserts %s at position %d but the target "
                "sequence has %s there - check residue numbering / that this "
                "is the right sequence", kind, str(t), wt, pos, actual
            )


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
        # Under backend=real a synthetic ligand (RDKit absent / SMILES
        # unparseable) means EVERY downstream chemistry feature is fabricated.
        # Hard-fail instead of silently shipping it (audit P0 #3).
        if (ctx.config.backend is Backend.real and ligand.source != "rdkit"
                and not mock_fallback_allowed()):
            raise ValueError(
                f"ligand parsed as {ligand.source!r} (not 'rdkit') under "
                "backend=real: RDKit could not parse the ligand, so all "
                "downstream chemistry would be synthetic. Fix the ligand "
                "input / install RDKit, or set allow_mock_fallback=true."
            )
        (ctx.paths.inputs / "ligand.smi").write_text(
            f"{ligand.smiles}\t{ligand.id}\n"
        )

        # Additional cofactors/substrates (co-modelled in the s04 complex as
        # their own entities). Same RDKit-or-fail gate as the primary ligand so
        # a bad SMILES can't silently ship synthetic chemistry under real.
        extra_ligands = []
        for el in cfg.extra_ligands:
            elig = parse_ligand(el)
            if (ctx.config.backend is Backend.real and elig.source != "rdkit"
                    and not mock_fallback_allowed()):
                raise ValueError(
                    f"extra ligand {el.id!r} parsed as {elig.source!r} (not "
                    "'rdkit') under backend=real; fix the SMILES / install RDKit."
                )
            extra_ligands.append(elig)
        if extra_ligands:
            (ctx.paths.inputs / "extra_ligands.smi").write_text(
                "".join(f"{e.smiles}\t{e.id}\n" for e in extra_ligands)
            )

        validate_residue_tokens(cfg.catalytic_residues, seq, "catalytic",
                                 self.log)
        validate_residue_tokens(cfg.fixed_residues, seq, "fixed", self.log)
        validate_residue_tokens(cfg.known_binding_site, seq, "binding-site",
                                 self.log)
        catalytic = parse_residue_tokens(cfg.catalytic_residues)
        fixed = parse_residue_tokens(cfg.fixed_residues) or list(catalytic)
        known_site = parse_residue_tokens(cfg.known_binding_site)

        ctx.put("target_sequence", seq)
        ctx.put("ligand", ligand)
        ctx.put("extra_ligands", extra_ligands)
        # Role-based ligand manifest (GENERIC): resolve roles -> docking targets +
        # context chains. Additive — the positional ligand/extra_ligands above stay
        # for back-compat; multi-engine docking consumes dock_ligands/context_ligands.
        manifest = resolve_ligand_manifest(cfg.ligand, cfg.extra_ligands)
        ctx.put("ligand_manifest", manifest)
        ctx.put("dock_ligands", [m for m in manifest if m.dock])
        ctx.put("context_ligands", [m for m in manifest if m.keep_as_context])
        ctx.put("catalytic_positions", catalytic)
        ctx.put("fixed_positions", sorted(set(fixed) | set(catalytic)))
        ctx.put("known_binding_site", known_site)
        ctx.persist_meta("sequence_length", len(seq))
        ctx.persist_meta("ligand_smiles", ligand.smiles)
        ctx.persist_meta("ligand_n_heavy", ligand.n_heavy)
        # canonical atom-id list (atom-index lock reference, expert review #5)
        ctx.persist_meta("ligand_atom_ids", [a.id for a in ligand.atoms])
        if extra_ligands:
            ctx.persist_meta("extra_ligand_ids", [e.id for e in extra_ligands])
            self.log.info("extra ligands (co-modelled in s04): %s",
                          ", ".join(f"{e.id}={e.smiles}" for e in extra_ligands))
        ctx.persist_meta("catalytic_positions", catalytic)
        # pure explicitly-fixed positions (switch + structural; excludes catalytic)
        # so the s07 design-space report can render the protected core distinctly
        # from the catalytic core on a standalone (meta-only) reconstruction.
        ctx.persist_meta("fixed_positions", sorted(set(fixed)))

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
        self._write_report(ctx)

    def _write_report(self, ctx: RunContext) -> None:
        """s01 input/ligand provenance + chemistry report. Secondary — never
        fail the stage. Re-derives from config so it also works on --resume."""
        try:
            from datetime import datetime

            from evoliez.io._provenance import stamp
            from evoliez.io.input_report import (compute_input_stats,
                                                 write_input_report)
            cfg = ctx.config.input
            seq = ctx.get("target_sequence") or ""
            ligand = ctx.get("ligand") or parse_ligand(cfg.ligand)
            extras = []
            for el in cfg.extra_ligands:
                try:
                    extras.append((el.id, el.type, parse_ligand(el).smiles))
                except Exception:
                    extras.append((el.id, el.type, el.value))
            src = (f"FASTA file: {cfg.target_fasta}" if cfg.target_fasta
                   else "inline target_sequence (config)")
            stats = compute_input_stats(
                target_id=cfg.target_id, sequence=seq, sequence_source=src,
                ligand=(ligand.id, cfg.ligand.type, ligand.smiles),
                extra_ligands=extras,
                residues={"catalytic": list(cfg.catalytic_residues),
                          "fixed": list(cfg.fixed_residues),
                          "binding": list(cfg.known_binding_site)},
                organism=cfg.organism, ec_number=cfg.ec_number,
                target_ph=cfg.target_ph,
                ligand_atom_ids=[a.id for a in ligand.atoms],
                accession=cfg.accession, pdb_id=cfg.pdb_id,
                numbering_scheme=cfg.numbering_scheme,
            )
            write_input_report(
                ctx.paths.reports / "input_report.html", stats=stats,
                generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                provenance=stamp(ctx))
            self.log.info("input report: %s",
                          ctx.paths.reports / "input_report.html")
        except Exception as exc:  # report is secondary
            self.log.warning("input report failed: %s", exc)

    def load(self, ctx: RunContext) -> bool:
        fa = ctx.paths.inputs / "target.fasta"
        if not fa.exists():
            return False
        seq = _read_fasta(str(fa))
        ctx.put("target_sequence", seq)
        ctx.put("ligand", parse_ligand(ctx.config.input.ligand))
        # Extra cofactors/substrates co-modelled in s04 and docked as separate
        # context in s05 MUST be restored on resume too. run() puts these; if
        # load() omits them, a resumed s04/s05 silently regresses to
        # single-ligand (drops formate etc.). Re-parse from config — the same
        # source run() uses, and RDKit is present so the result matches.
        ctx.put("extra_ligands",
                [parse_ligand(el) for el in ctx.config.input.extra_ligands])
        # Role manifest must be restored on resume too (parity with run()), else a
        # resumed multi-engine docking sees no dock_ligands/context_ligands.
        manifest = resolve_ligand_manifest(ctx.config.input.ligand,
                                           ctx.config.input.extra_ligands)
        ctx.put("ligand_manifest", manifest)
        ctx.put("dock_ligands", [m for m in manifest if m.dock])
        ctx.put("context_ligands", [m for m in manifest if m.keep_as_context])
        # Reparse catalytic from CONFIG (not the persisted meta) so a config change
        # to catalytic_residues propagates on --resume — parity with fixed_residues
        # below and with run(). doctor validates the tokens; seq/RDKit are present.
        catalytic = parse_residue_tokens(ctx.config.input.catalytic_residues)
        ctx.put("catalytic_positions", catalytic)
        ctx.persist_meta("catalytic_positions", catalytic)
        ctx.put(
            "fixed_positions",
            sorted(
                set(parse_residue_tokens(ctx.config.input.fixed_residues))
                | set(catalytic)
            ),
        )
        ctx.put(
            "known_binding_site",
            parse_residue_tokens(ctx.config.input.known_binding_site),
        )
        self._write_report(ctx)
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
