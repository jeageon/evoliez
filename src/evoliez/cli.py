"""`evoliez` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from evoliez import __version__
from evoliez.config import Backend, load_config
from evoliez.context import RunContext
from evoliez.logging_utils import setup_logging
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

app = typer.Typer(add_completion=False, help="EvoLigand-Enzyme Engineer")


@app.command()
def run(
    config: Path = typer.Option(..., "-c", "--config", help="YAML config path"),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="override global backend: mock | real"
    ),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    resume: bool = typer.Option(False, "--resume", help="skip completed stages"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print real tool commands without executing"
    ),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    to_stage: Optional[str] = typer.Option(None, "--to"),
    allow_small_disk: bool = typer.Option(
        False, "--allow-small-disk", help="permit runs on a near-full filesystem"
    ),
) -> None:
    """Run the full pipeline from a config file."""
    setup_logging()
    overrides = {}
    if backend:
        overrides["backend"] = backend
    if output_dir:
        overrides["project.output_dir"] = output_dir
    cfg = load_config(config, overrides)
    seed_everything(cfg.seed)

    ctx = RunContext(cfg, allow_small_disk=allow_small_disk or cfg.backend is Backend.mock)
    ctx.dry_run = dry_run
    ctx.setup()
    Pipeline().run(ctx, resume=resume, from_stage=from_stage, to_stage=to_stage)

    top = ctx.meta("top_candidate")
    typer.echo("")
    typer.echo(f"Done. Project: {ctx.root}")
    typer.echo(f"Report: {ctx.paths.reports / 'final_report.md'}")
    if top:
        typer.echo(f"Top candidate: {top['mutations']} "
                   f"(score {top['final_score']:.3f})")


@app.command()
def stages() -> None:
    """List pipeline stages in execution order."""
    for i, name in enumerate(Pipeline().stage_names(), 1):
        typer.echo(f"{i:2d}. {name}")


@app.command()
def show(config: Path = typer.Option(..., "-c", "--config")) -> None:
    """Validate a config and print the resolved settings."""
    cfg = load_config(config)
    typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2, default=str))


@app.command()
def version() -> None:
    typer.echo(__version__)


if __name__ == "__main__":
    app()
