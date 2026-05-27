"""Stage 01 - input & ligand preprocessing (spec section 6)."""

from __future__ import annotations

import re

from evoliez.adapters.md_preflight import run_md_preflight
from evoliez.context import RunContext
from evoliez.db.schema import Sequence
from evoliez.features.cofactors import (
    check_cofactor_matches,
    formula_of,
    resolve_ligand_spec,
)
from evoliez.features.ligand import parse_ligand
from evoliez.stages.base import PreflightBlocked, Stage

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
        # Resolve ligand.type='cofactor' to a curated SMILES (pH + redox
        # driven) BEFORE parsing, then run the formula guard. A declared
        # cofactor whose SMILES doesn't match the curated species is a
        # silent data-integrity bug (the original "GAFF can't parameterise
        # NADP" was actually NAD+ in NADP's clothing) - fail loudly here so
        # MD never sees the wrong molecule.
        eff_spec = resolve_ligand_spec(cfg)
        ligand = parse_ligand(eff_spec)
        # Use the SMILES (the source of truth) for the formula guard so
        # this works identically with or without RDKit; parse_ligand's
        # synthetic fallback samples elements stochastically and would
        # falsely trip the guard otherwise.
        guard = check_cofactor_matches(
            cfg.cofactor,
            formula_of(eff_spec.value) if eff_spec.type == "smiles"
            else formula_of(cfg.ligand.value),
            pH=cfg.target_ph, redox_state=cfg.cofactor_redox,
        )
        if not guard.ok:
            raise ValueError(
                "input.cofactor / input.ligand mismatch: " + guard.message
            )
        if guard.expected is not None:
            self.log.info("cofactor guard: %s", guard.message)
        (ctx.paths.inputs / "ligand.smi").write_text(
            f"{ligand.smiles}\t{ligand.id}\n"
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

        # P0 C — run MD parameterisation preflight here (not at s10) so:
        #   1. The report can label MD-skipped projects HONESTLY up-front
        #      instead of after stages 02-09 have already burned 25+ hours.
        #   2. The probe-cache disk sidecar is primed: every s10 candidate
        #      hits the cache instead of paying the probe per-candidate,
        #      and subprocess-isolated workers inherit it at startup.
        # Degrades to skipped_no_md_libs in the light mac venv.
        try:
            mdcfg = ctx.config.validation.md
            md_enabled = bool(getattr(mdcfg, "enabled", True))
            prefer_ff = getattr(mdcfg, "ligand_forcefield", None)
            # G: when MD itself runs in subprocess-isolated mode, the
            # preflight does too. A sqm hang in s01 is strictly worse
            # than the same hang at s10 (no per-candidate retry, the
            # rest of the pipeline never starts). 300 s default cap is
            # plenty for a Gasteiger probe and intentionally tighter
            # than `subprocess_timeout_seconds` so we fail fast.
            pre_isolated = bool(getattr(mdcfg, "subprocess_isolation", False))
            pre_timeout = int(getattr(mdcfg, "preflight_timeout_seconds", 300))
        except Exception:
            md_enabled = True
            prefer_ff = None
            pre_isolated = False
            pre_timeout = 300
        preflight = run_md_preflight(
            ligand.smiles, ctx.paths.md,
            prefer_ff=prefer_ff, md_enabled=md_enabled,
            subprocess_isolation=pre_isolated,
            timeout_seconds=pre_timeout,
        )
        ctx.persist_meta("md_preflight_status", preflight.status)
        if preflight.ff_used:
            ctx.persist_meta("md_preflight_ff", preflight.ff_used)
        if preflight.reason:
            ctx.persist_meta("md_preflight_reason", preflight.reason)
        self.log.info(
            "MD preflight: %s%s%s",
            preflight.status,
            f" (FF={preflight.ff_used})" if preflight.ff_used else "",
            f" — {preflight.reason}" if preflight.reason else "",
        )

        # K (post-expert-audit) — strict mode halts the pipeline at s01.
        # H added the s08b gate, but with strict_preflight=True s02-s07
        # would still run (homology / MSA / s04 Boltz / docking / ranking)
        # before s08b realised MD could never validate the candidates.
        # That's hours of waste for a project the preflight already
        # proved hopeless. Halt cleanly here instead — Pipeline catches
        # the PreflightBlocked exception and stops downstream stages
        # without surfacing a stack trace. Defence in depth: the s08b
        # gate is still in place for legacy / fallback configs that
        # disable this halt.
        try:
            strict = bool(getattr(ctx.config.validation.md,
                                  "strict_preflight", False))
        except Exception:
            strict = False
        _blocking = {"unsupported", "timeout_preflight", "failed_preflight"}
        if strict and preflight.status in _blocking:
            self.log.warning(
                "strict_preflight=True and md_preflight_status=%r — "
                "halting pipeline at s01 (no Boltz / no docking / no MD). "
                "Reason: %s",
                preflight.status, preflight.reason or "n/a",
            )
            ctx.persist_meta("preflight_blocked", preflight.status)
            ctx.persist_meta(
                "preflight_blocked_reason", preflight.reason or ""
            )
            raise PreflightBlocked(preflight.status, preflight.reason or "")

    def load(self, ctx: RunContext) -> bool:
        fa = ctx.paths.inputs / "target.fasta"
        if not fa.exists():
            return False
        seq = _read_fasta(str(fa))
        ctx.put("target_sequence", seq)
        ctx.put("ligand", parse_ligand(resolve_ligand_spec(ctx.config.input)))
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
