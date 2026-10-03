"""`blindearth` command line.

    blindearth run spec.yaml --estimate     # plan + 50-point pilot; nothing else is spent
    blindearth run spec.yaml --estimate --no-pilot   # zero-spend estimate
    blindearth run spec.yaml                # execute, resumable
    blindearth compare <run> <run> --name n # saved runs, no API calls
    blindearth report <comparison> --html out.html --png grid.png
    blindearth serve                        # web UI
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Coroutine, Optional

import typer
from rich.console import Console
from rich.table import Table

from blindearth.types import RunStatus

if TYPE_CHECKING:
    from blindearth.runner.planner import Plan
    from blindearth.store.db import Store

app = typer.Typer(
    name="blindearth",
    help="Blind Earth eval runner: ask text-only models 'Land or Water?' per grid cell.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DbOpt = Annotated[Path, typer.Option("--db", help="SQLite results store.")]
RegistryOpt = Annotated[Path, typer.Option("--registry", "-r", help="Provider/model registry YAML.")]


# ----------------------------------------------------------------------------- helpers


def _store(db: Path) -> Store:
    from blindearth.store.db import Store

    return Store(db)


def _registry(path: Path) -> Any:
    from blindearth.providers.registry import load_registry

    if not Path(path).exists():
        console.print(f"[red]registry not found:[/red] {path}  (see README: registry setup)")
        raise typer.Exit(2)
    return load_registry(path)


def _fmt_usd(v: float | None) -> str:
    if v is None:
        return "?"
    return f"${v:,.4f}" if v < 1 else f"${v:,.2f}"


def _fmt_int(v: int | float | None) -> str:
    return "?" if v is None else f"{int(v):,}"


def _fmt_pct(v: Any) -> str:
    try:
        return f"{100 * float(v):.1f}%"
    except (TypeError, ValueError):
        return "-"


def print_plan(plan: Plan) -> None:
    spec = plan.eval_file.eval
    console.print(
        f"[bold]Eval spec[/bold] {plan.spec_id[:12]}  mask={spec.mask.id} ({plan.mask.source})  "
        f"grid={spec.grid.step_deg}° {spec.grid.placement}  points={len(plan.points):,}  "
        f"coords={spec.coord_format}  prompt={spec.prompt.id}"
    )
    t = Table(show_lines=False, header_style="bold")
    for col, kw in [
        ("Cell", {}),
        ("Provider", {}),
        ("Mode", {}),
        ("Points", {"justify": "right"}),
        ("Calls", {"justify": "right"}),
        ("Est in tok", {"justify": "right"}),
        ("Est out tok", {"justify": "right"}),
        ("Est think tok", {"justify": "right"}),
        ("Est USD", {"justify": "right"}),
        ("Status", {}),
    ]:
        t.add_column(col, **kw)
    n_cached = n_refused = 0
    for c in plan.cells:
        if c.refused:
            n_refused += 1
            status = f"[red]refused[/red]: {c.refused}"
        elif c.cached_run_id:
            n_cached += 1
            status = f"[green]cache hit[/green] {c.cached_run_id[:12]}"
        elif c.resume_run_id:
            status = f"[yellow]resume[/yellow] {c.resume_run_id[:12]}"
        else:
            status = "new"
        if c.active and c.n_points:
            status += f" [dim]({c.est_source})[/dim]"
        active = c.active
        t.add_row(
            c.label + (" [dim](batch)[/dim]" if c.use_batch and active else ""),
            c.provider.id,
            c.mode.value,
            _fmt_int(c.n_points) if active else "-",
            _fmt_int(c.n_calls) if active else "-",
            _fmt_int(c.est_usage.input_tokens) if active else "-",
            _fmt_int(c.est_usage.output_tokens) if active else "-",
            _fmt_int(c.est_usage.thinking_tokens) if active else "-",
            _fmt_usd(c.est_cost_usd) if active else "-",
            status,
        )
    console.print(t)
    active_cells = [c for c in plan.cells if c.active]
    console.print(
        f"[bold]Total[/bold] calls={sum(c.n_calls for c in active_cells):,}  "
        f"est cost={_fmt_usd(plan.total_cost_usd)}  cache hits={n_cached}  refusals={n_refused}  "
        f"budget={_fmt_usd(plan.budget_usd) if plan.budget_usd is not None else 'none'}  "
        f"pilot spend={_fmt_usd(plan.pilot_cost_usd) if plan.pilot_cost_usd is not None else '$0'}"
    )
    for c in plan.cells:
        for n in c.notes:
            console.print(f"  [dim]{c.label}: {n}[/dim]")
    for w in plan.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")


async def _with_progress(
    store: Store, factory: Callable[[Callable, asyncio.Event], Coroutine[Any, Any, Any]]
) -> Any:
    """Run an executor coroutine with a rich progress bar per run and Ctrl-C -> graceful pause."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop.set)
    except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows
        pass
    tasks: dict[str, Any] = {}
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        def on_progress(run_id: str, batch: list) -> None:
            rec = store.get_run(run_id)
            if run_id not in tasks:
                tasks[run_id] = progress.add_task(f"{rec.variant} [{run_id[:8]}]", total=rec.n_points_total)
            progress.update(tasks[run_id], completed=rec.n_points_done)

        try:
            return await factory(on_progress, stop)
        finally:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):  # pragma: no cover
                pass
            if stop.is_set():
                console.print("[yellow]stopped: unfinished runs are paused; `blindearth resume RUN_ID`[/yellow]")


def _print_runs(store: Store, run_ids: list[str]) -> None:
    t = Table(header_style="bold")
    for col in ("Run", "Variant", "Mode", "Status", "Done", "Cost", "Acc (area)", "Note"):
        t.add_column(col)
    for rid in run_ids:
        r = store.get_run(rid)
        m = store.load_metrics(r.id) or {}
        note = store.run_meta(r.id).get("note") or ""
        t.add_row(
            r.id[:12],
            r.variant + ("*" if r.forced_thinking else ""),
            r.extraction_mode.value,
            r.status.value,
            f"{r.n_points_done:,}/{r.n_points_total:,}",
            _fmt_usd(r.cost_usd),
            _fmt_pct(m.get("acc_area")),
            note,
        )
    console.print(t)


# ----------------------------------------------------------------------------- commands


@app.command()
def run(
    spec: Annotated[Path, typer.Argument(help="Eval spec YAML (eval, extraction, matrix).")],
    registry: RegistryOpt = Path("providers.yaml"),
    db: DbOpt = Path("blindearth.db"),
    estimate: Annotated[bool, typer.Option("--estimate", help="Plan and estimate only (pilot is the only spend).")] = False,
    budget: Annotated[Optional[float], typer.Option("--budget", help="Hard USD cap; runs pause when reached.")] = None,
    no_pilot: Annotated[bool, typer.Option("--no-pilot", help="Skip the pilot: zero-spend, heuristic estimate.")] = False,
    pilot_points: Annotated[int, typer.Option("--pilot-points", help="Pilot size per uncached cell.")] = 50,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Plan, estimate and (unless --estimate) execute an eval spec."""
    from blindearth.evalspec.load import EvalFileError, load_eval_file
    from blindearth.runner.executor import execute
    from blindearth.runner.planner import build_plan

    try:
        eval_file = load_eval_file(spec)
    except EvalFileError as e:
        where = f" at [bold]{e.key}[/bold]" if getattr(e, "key", None) else ""
        console.print(f"[red]invalid eval file[/red] {spec}{where}: {e}")
        raise typer.Exit(2) from e
    except FileNotFoundError as e:
        console.print(f"[red]eval file not found:[/red] {spec}")
        raise typer.Exit(2) from e
    reg = _registry(registry)
    store = _store(db)
    with console.status("planning" + ("" if no_pilot else " (running pilot)") + " ..."):
        plan = asyncio.run(build_plan(eval_file, reg, store, pilot_points=pilot_points, run_pilot=not no_pilot))
    if budget is not None:
        plan.budget_usd = budget
    print_plan(plan)
    if estimate:
        return
    if not any(c.active for c in plan.cells):
        console.print("nothing to run (all cells cached or refused)")
        _print_runs(store, [c.cached_run_id for c in plan.cells if c.cached_run_id])
        return
    if not yes and sys.stdin.isatty():
        typer.confirm("Run this plan?", abort=True)
    run_ids = asyncio.run(
        _with_progress(
            store,
            lambda cb, stop: execute(plan, store, reg, on_progress=cb, budget_usd=plan.budget_usd, stop_event=stop),
        )
    )
    _print_runs(store, run_ids)


@app.command()
def probe(
    model_ref: Annotated[str, typer.Argument(help="'provider/model_id' or 'model_id'.")],
    registry: RegistryOpt = Path("providers.yaml"),
    db: DbOpt = Path("blindearth.db"),
    calls: Annotated[int, typer.Option("--calls", help="Number of probe calls.")] = 20,
) -> None:
    """Probe what an endpoint actually supports (logprobs, effort, concurrency) and store it."""
    from blindearth.providers.probe import probe as run_probe
    from blindearth.providers.registry import build_adapter

    reg = _registry(registry)
    store = _store(db)
    model = reg.model(model_ref)
    provider = reg.provider_of(model)

    async def go() -> Any:
        adapter = build_adapter(provider, model)
        try:
            return await run_probe(adapter, n_calls=calls)
        finally:
            await adapter.aclose()

    with console.status(f"probing {model.ref} ({calls} calls) ..."):
        caps = asyncio.run(go())
    store.save_capabilities(provider.id, model.name, caps)
    t = Table(title=f"Capabilities: {model.ref} ({model.name})", header_style="bold")
    t.add_column("Field")
    t.add_column("Value")
    for k, v in caps.__dict__.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command("runs")
def runs_cmd(
    db: DbOpt = Path("blindearth.db"),
    status: Annotated[Optional[str], typer.Option("--status", help="queued|running|paused|complete|failed")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Filter by model ref or id.")] = None,
) -> None:
    """List stored runs."""
    store = _store(db)
    st = RunStatus(status) if status else None
    runs = store.list_runs(status=st, model_id=model)
    if not runs:
        console.print("no runs")
        return
    _print_runs(store, [r.id for r in runs])


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Run id or unique prefix.")],
    registry: RegistryOpt = Path("providers.yaml"),
    db: DbOpt = Path("blindearth.db"),
    budget: Annotated[Optional[float], typer.Option("--budget", help="Hard USD cap including earlier spend.")] = None,
) -> None:
    """Resume a paused, failed or crashed run (missing points, then failed points)."""
    from blindearth.runner.executor import resume as resume_run

    reg = _registry(registry)
    store = _store(db)
    rid = store.resolve_run_id(run_id)
    asyncio.run(
        _with_progress(
            store,
            lambda cb, stop: resume_run(rid, store, reg, on_progress=cb, stop_event=stop, budget_usd=budget),
        )
    )
    _print_runs(store, [rid])


@app.command()
def compare(
    runs: Annotated[list[str], typer.Argument(help="Run ids (or unique prefixes).")],
    name: Annotated[str, typer.Option("--name", help="Comparison name.")],
    db: DbOpt = Path("blindearth.db"),
) -> None:
    """Save a named comparison of stored runs. No API calls."""
    from blindearth.runner.planner import check_comparison

    store = _store(db)
    ids = [store.resolve_run_id(r) for r in runs]
    try:
        filters = check_comparison(store, ids)
    except ValueError as e:
        console.print(f"[red]refused:[/red] {e}")
        raise typer.Exit(1) from e
    cid = store.create_comparison(name, ids, filters=filters, ordering=None)
    console.print(f"comparison [bold]{name}[/bold] ({cid[:12]}) with {len(ids)} runs")
    for w in filters.get("warnings", []):
        console.print(f"[yellow]warning:[/yellow] {w}")


@app.command()
def score(
    runs: Annotated[list[str], typer.Argument(help="Run ids (or unique prefixes).")],
    db: DbOpt = Path("blindearth.db"),
    mask: Annotated[Optional[str], typer.Option("--mask", help="Built-in mask id or path to an uploaded mask; default: the run's mask.")] = None,
    threshold: Annotated[float, typer.Option("--threshold", help="P(Land) threshold.")] = 0.5,
    no_images: Annotated[bool, typer.Option("--no-images", help="Skip image-comparison metrics.")] = False,
) -> None:
    """(Re)score stored runs against a mask and threshold. No API calls."""
    import dataclasses

    from blindearth.evalspec.masks import load_mask
    from blindearth.scoring.score import score_run
    from blindearth.types import MaskSpec

    store = _store(db)
    t = Table(header_style="bold")
    for col in ("Run", "Variant", "Acc (area)", "Acc (unweighted)", "Skill", "F1", "Invalid"):
        t.add_column(col)
    for r in runs:
        rec = store.get_run(r)
        mask_obj = None
        if mask:
            spec, _, _ = store.get_eval_spec(rec.spec_id)
            if Path(mask).exists():
                ms = dataclasses.replace(spec.mask, id="upload", path=str(mask))
            else:
                ms = MaskSpec(id=mask, truth_rule=spec.mask.truth_rule)
            mask_obj = load_mask(ms)
        m = score_run(store, rec.id, mask=mask_obj, threshold=threshold, with_images=not no_images)
        t.add_row(
            rec.id[:12],
            rec.variant,
            _fmt_pct(m.get("acc_area")),
            _fmt_pct(m.get("acc_unweighted")),
            f"{m.get('skill', float('nan')):.3f}" if isinstance(m.get("skill"), (int, float)) else "-",
            f"{m.get('f1', float('nan')):.3f}" if isinstance(m.get("f1"), (int, float)) else "-",
            _fmt_pct(m.get("invalid_rate")),
        )
    console.print(t)


@app.command()
def report(
    comparison: Annotated[str, typer.Argument(help="Comparison name or id.")],
    html: Annotated[Optional[Path], typer.Option("--html", help="Self-contained HTML report path.")] = None,
    png: Annotated[Optional[Path], typer.Option("--png", help="Map grid PNG path.")] = None,
    db: DbOpt = Path("blindearth.db"),
    threshold: Annotated[float, typer.Option("--threshold")] = 0.5,
    mode: Annotated[str, typer.Option("--mode", help="PNG map mode: binary|probability|error")] = "binary",
) -> None:
    """Build an HTML report and/or a PNG map grid from a saved comparison."""
    if html is None and png is None:
        console.print("[red]give --html and/or --png[/red]")
        raise typer.Exit(2)
    store = _store(db)
    if html is not None:
        from blindearth.report.html import build_report

        out = build_report(store, comparison, html, threshold=threshold)
        console.print(f"wrote {out}")
    if png is not None:
        from blindearth.report.data import load_comparison
        from blindearth.report.mapgrid import render_map_grid

        comp, views, mask_obj = load_comparison(store, comparison, threshold=threshold)
        if not views:
            console.print("[red]comparison has no runs[/red]")
            raise typer.Exit(1)
        spec, _, _ = store.get_eval_spec(views[0].run.spec_id)
        data = render_map_grid(
            views,
            mode=mode,  # type: ignore[arg-type]
            step_deg=spec.grid.step_deg,
            placement=spec.grid.placement,
            mask=mask_obj,
            title=comp.get("name") if isinstance(comp, dict) else None,
        )
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(data)
        console.print(f"wrote {png}")


@app.command()
def export(
    run_id: Annotated[str, typer.Argument(help="Run id or unique prefix.")],
    format: Annotated[str, typer.Option("--format", "-f", help="parquet|csv (points) or json (metrics)")] = "parquet",
    out: Annotated[Path, typer.Option("--out", "-o", help="Output file or directory.")] = Path("."),
    db: DbOpt = Path("blindearth.db"),
) -> None:
    """Export a run's points (Parquet/CSV) or metrics (JSON)."""
    from blindearth.store.export import export_metrics, export_points

    store = _store(db)
    if format == "json":
        path = export_metrics(store, run_id, out)
    elif format in ("parquet", "csv"):
        path = export_points(store, run_id, out, format)  # type: ignore[arg-type]
    else:
        console.print(f"[red]unknown format {format}[/red]")
        raise typer.Exit(2)
    console.print(f"wrote {path}")


@app.command()
def serve(
    db: DbOpt = Path("blindearth.db"),
    registry: RegistryOpt = Path("providers.yaml"),
    port: Annotated[int, typer.Option("--port")] = 7860,
) -> None:
    """Start the web UI (requires the `ui` extra: gradio)."""
    try:
        from blindearth.ui.app import main
    except ImportError as e:
        console.print(f"[red]web UI unavailable:[/red] {e}. Install with: pip install 'blindearth[ui]'")
        raise typer.Exit(1) from e
    main(db_path=str(db), registry_path=str(registry), port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
