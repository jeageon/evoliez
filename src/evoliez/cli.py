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
    stage_backend: Optional[list[str]] = typer.Option(
        None, "--stage-backend",
        help="per-stage backend, repeatable: --stage-backend s04_complex=real"
    ),
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
    if stage_backend:
        from evoliez.stages import ALL_STAGES

        valid = {s.name for s in ALL_STAGES}
        be = {}
        for item in stage_backend:
            stage, _, val = item.partition("=")
            stage = stage.strip()
            if stage not in valid:
                raise typer.BadParameter(
                    f"--stage-backend: unknown stage {stage!r} (a typo here "
                    f"would silently run mock and 'pass'). Valid: "
                    f"{', '.join(sorted(valid))}"
                )
            be[stage] = val.strip() or "real"
        overrides["backends"] = be
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


@app.command(name="train-gnn")
def train_gnn(
    config: Path = typer.Option(..., "-c", "--config"),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", help="graph dataset dir (default: <run>/datasets/graph_pt)"
    ),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
) -> None:
    """Train the server-grade EvoLigand-GNN on the exported graph dataset.

    Run the pipeline first so the dataset exists. torch is required (GPU
    server); on the laptop this prints a clear message and exits.
    """
    setup_logging()
    cfg = load_config(config)
    from evoliez.io.paths import ProjectPaths
    from evoliez.ml.train_gnn import train

    paths = ProjectPaths(Path(cfg.project.output_dir).resolve())
    ds = Path(dataset) if dataset else paths.graph_dataset
    ckpt = paths.root / cfg.gnn.checkpoint
    try:
        out = train(
            ds, ckpt, epochs=epochs or cfg.gnn.epochs,
            hidden=cfg.gnn.hidden_dim, layers=cfg.gnn.layers,
            lr=cfg.gnn.lr, amp=cfg.gnn.amp, seed=cfg.seed,
            coord_noise_min=cfg.gnn.coord_noise_min,
            coord_noise_alpha=cfg.gnn.coord_noise_alpha,
            equivariant=cfg.gnn.equivariant,
            rbf_n=cfg.gnn.rbf,
            graph_geom={
                "radius_lr": cfg.gnn.radius_lr,
                "radius_rr": cfg.gnn.radius_rr,
                "low_plddt_cutoff": cfg.gnn.low_plddt_cutoff,
                "drop_far_low_plddt": cfg.gnn.drop_far_low_plddt,
                "use_disorder": cfg.gnn.use_disorder,
            },
        )
        typer.echo(f"trained EvoLigand-GNN -> {out}")
    except RuntimeError as exc:
        typer.echo(f"skipped: {exc}")
        raise typer.Exit(code=0)


@app.command()
def bench(
    config: Path = typer.Option(..., "-c", "--config"),
    benchmark: Path = typer.Option(..., "--benchmark",
                                   help="CSV: mutation,label[,activity]"),
    k: int = typer.Option(20, "--topk"),
    ablation: bool = typer.Option(
        False, "--ablation",
        help="also re-run with each accuracy layer disabled (slower)"
    ),
    allow_no_overlap: bool = typer.Option(
        False, "--allow-no-overlap",
        help="do not fail when benchmark/candidate overlap is 0"
    ),
    external_format: str = typer.Option(
        "none", "--external-format",
        help="parse --benchmark as a public set: none|proteingym|flip"
    ),
    baselines: bool = typer.Option(
        False, "--baselines",
        help="also score random/conservation/MSA/interaction baselines"
    ),
) -> None:
    """Run the pipeline then score it against a known-mutation benchmark
    (recovery / deleterious avoidance / Spearman / AUROC / calibration),
    optionally with an ablation study - user §9 / expert review #4."""
    import json as _json

    setup_logging()
    cfg = load_config(config)
    seed_everything(cfg.seed)
    ctx = RunContext(
        cfg, allow_small_disk=cfg.backend is Backend.mock
    )
    ctx.setup()
    Pipeline().run(ctx)
    from evoliez.ml.benchmark import (
        compare_baselines,
        load_benchmark,
        load_external_benchmark,
        run_ablation,
        run_benchmark,
    )

    if external_format != "none":
        bench_rows = load_external_benchmark(benchmark, fmt=external_format)
    else:
        bench_rows = load_benchmark(benchmark)
    ranked = ctx.get("ranked_candidates", [])
    result = run_benchmark(
        ranked, bench_rows, k=k,
        catalytic_positions=ctx.get("catalytic_positions", []),
        known_site=ctx.get("known_binding_site", []),
        target_sequence=ctx.get("target_sequence", ""),
    )
    if baselines:
        result["baselines"] = compare_baselines(ranked, bench_rows, k=k)
    if ablation:
        result["ablation_study"] = run_ablation(
            str(config), bench_rows, k=k
        )
    (ctx.paths.reports / "benchmark.json").write_text(
        _json.dumps(result, indent=2)
    )
    typer.echo(_json.dumps(result, indent=2, default=str))
    for w in result.get("warnings", []):
        typer.echo(f"WARNING: {w}")
    if not result.get("valid", True) and not allow_no_overlap:
        typer.echo(
            "benchmark is NOT valid for this run (see warnings). "
            "Metrics above are not meaningful; fix the benchmark or pass "
            "--allow-no-overlap to proceed anyway."
        )
        raise typer.Exit(code=2)


@app.command()
def doctor(
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="also validate this run config"
    ),
) -> None:
    """Preflight: check tools / deps / GPU / disk before a real server run."""
    from evoliez.diagnostics import BLOCK, MISSING, OK, WARN, collect

    mark = {OK: "[OK]   ", WARN: "[WARN] ", MISSING: "[MISS] ",
            BLOCK: "[BLOCK]"}
    rep = collect(config)
    for c in rep.checks:
        typer.echo(f"{mark[c.status]} {c.name:<22} {c.detail}")
    n_warn = sum(1 for c in rep.checks if c.status in (WARN, MISSING))
    typer.echo(
        f"\n{len(rep.checks)} checks | {rep.n_block} blocking | {n_warn} warn/missing"
    )
    if rep.n_block:
        typer.echo("blocking issues must be fixed before a real run.")
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    typer.echo(__version__)


if __name__ == "__main__":
    app()
