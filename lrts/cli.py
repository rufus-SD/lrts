from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
from sqlmodel import select

from lrts.db import init_db, async_session
from lrts.models import Prompt, PromptVersion, Dataset, DatasetItem, Run
from lrts.engines.runner import RunnerEngine
from lrts.engines.report import ReportGenerator
from lrts.engines.orchestrator import (
    cleanup_prompt,
    cleanup_dataset,
    run_test_spec,
    run_demo as orchestrate_demo,
)

def _version_callback(value: bool):
    if value:
        from lrts import __version__
        typer.echo(f"lrts {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="lrts",
    help="LRTS — LLM Regression Testing System. CI/CD for LLM behavior.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback,
        is_eager=True, help="Show version and exit.",
    ),
):
    pass

prompt_app = typer.Typer(help="Manage prompts and versions.")
dataset_app = typer.Typer(help="Manage test datasets.")
app.add_typer(prompt_app, name="prompt")
app.add_typer(dataset_app, name="dataset")


def _run(coro):
    return asyncio.run(coro)


async def _ensure_db():
    await init_db()


def _make_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    )


# ── Prompt commands ─────────────────────────────────────────────────────────


@prompt_app.command("add")
def prompt_add(
    name: str = typer.Option(..., "--name", "-n", help="Prompt name"),
    system_prompt: str = typer.Option(
        ..., "--system-prompt", "-s", help="System prompt text"
    ),
    version: int = typer.Option(1, "--version", "-v", help="Version number"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
    provider: str = typer.Option("openai", "--provider", "-p", help="Provider"),
    base_url: str = typer.Option(None, "--base-url", help="Custom base URL"),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Register a prompt version."""

    async def _do():
        await _ensure_db()
        async with async_session() as session:
            existing = (
                await session.execute(
                    select(Prompt).where(Prompt.name == name)
                )
            ).scalars().first()

            if existing:
                prompt = existing
            else:
                prompt = Prompt(name=name)
                session.add(prompt)
                await session.flush()

            pv = PromptVersion(
                prompt_id=prompt.id,
                version=version,
                system_prompt=system_prompt,
                model=model,
                provider=provider,
                base_url=base_url,
                model_config_json={"temperature": temperature},
            )
            session.add(pv)
            await session.commit()
            return prompt, pv

    prompt, pv = _run(_do())

    if output_json:
        console.print_json(
            json.dumps(
                {"prompt_id": prompt.id, "version_id": pv.id, "version": pv.version}
            )
        )
    else:
        console.print(
            f"[green]✓[/] Registered [bold]{name}[/] v{pv.version} "
            f"([dim]{pv.model} via {pv.provider}[/])"
        )


@prompt_app.command("list")
def prompt_list(
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """List all prompts and versions."""

    async def _do():
        await _ensure_db()
        async with async_session() as session:
            prompts = (
                await session.execute(select(Prompt))
            ).scalars().all()
            result = []
            for p in prompts:
                versions = (
                    await session.execute(
                        select(PromptVersion).where(PromptVersion.prompt_id == p.id)
                    )
                ).scalars().all()
                result.append((p, versions))
            return result

    data = _run(_do())

    if output_json:
        out = [
            {
                "name": p.name,
                "versions": [
                    {"version": v.version, "model": v.model, "provider": v.provider}
                    for v in vs
                ],
            }
            for p, vs in data
        ]
        console.print_json(json.dumps(out))
        return

    if not data:
        console.print("[dim]No prompts registered yet.[/]")
        return

    table = Table(title="Registered Prompts", box=box.ROUNDED)
    table.add_column("Name", style="bold cyan")
    table.add_column("Version", justify="right")
    table.add_column("Model")
    table.add_column("Provider")

    for p, versions in data:
        for v in versions:
            table.add_row(p.name, str(v.version), v.model, v.provider)

    console.print(table)


# ── Dataset commands ────────────────────────────────────────────────────────


@dataset_app.command("load")
def dataset_load(
    name: str = typer.Option(..., "--name", "-n", help="Dataset name"),
    file: Path = typer.Option(..., "--file", "-f", help="JSONL file path"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Load a dataset from a JSONL file. Each line: {"input": "...", "expected_output": "..."}"""

    if not file.exists():
        console.print(f"[red]File not found:[/] {file}")
        raise typer.Exit(1)

    async def _do():
        await _ensure_db()
        async with async_session() as session:
            ds = Dataset(name=name)
            session.add(ds)
            await session.flush()

            count = 0
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    di = DatasetItem(
                        dataset_id=ds.id,
                        input=item["input"],
                        expected_output=item.get("expected_output"),
                        metadata_json=item.get("metadata", {}),
                    )
                    session.add(di)
                    count += 1

            await session.commit()
            return ds, count

    ds, count = _run(_do())

    if output_json:
        console.print_json(json.dumps({"dataset_id": ds.id, "items": count}))
    else:
        console.print(
            f"[green]✓[/] Loaded [bold]{name}[/] with {count} items"
        )


@dataset_app.command("list")
def dataset_list(
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """List all datasets."""

    async def _do():
        await _ensure_db()
        async with async_session() as session:
            datasets = (
                await session.execute(select(Dataset))
            ).scalars().all()
            result = []
            for ds in datasets:
                items = (
                    await session.execute(
                        select(DatasetItem).where(DatasetItem.dataset_id == ds.id)
                    )
                ).scalars().all()
                result.append((ds, len(items)))
            return result

    data = _run(_do())

    if output_json:
        out = [{"name": ds.name, "items": c} for ds, c in data]
        console.print_json(json.dumps(out))
        return

    if not data:
        console.print("[dim]No datasets loaded yet.[/]")
        return

    table = Table(title="Datasets", box=box.ROUNDED)
    table.add_column("Name", style="bold cyan")
    table.add_column("Items", justify="right")

    for ds, count in data:
        table.add_row(ds.name, str(count))

    console.print(table)


# ── Run command ─────────────────────────────────────────────────────────────


@app.command("run")
def run_test(
    prompt_name: str = typer.Option(..., "--prompt", "-p", help="Prompt name"),
    version: int = typer.Option(..., "--version", "-v", help="Version to test"),
    baseline: int = typer.Option(
        None, "--baseline", "-b", help="Baseline version to compare against"
    ),
    dataset_name: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset name"
    ),
    evaluators: str = typer.Option(
        "exact,semantic", "--evaluators", "-e",
        help="Comma-separated: exact,semantic,judge",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable response caching"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Run a regression test comparing prompt versions."""

    async def _do():
        await _ensure_db()
        async with async_session() as session:
            prompt = (
                await session.execute(
                    select(Prompt).where(Prompt.name == prompt_name)
                )
            ).scalars().first()
            if not prompt:
                console.print(f"[red]Prompt '{prompt_name}' not found[/]")
                raise typer.Exit(1)

            pv = (
                await session.execute(
                    select(PromptVersion).where(
                        PromptVersion.prompt_id == prompt.id,
                        PromptVersion.version == version,
                    )
                )
            ).scalars().first()
            if not pv:
                console.print(
                    f"[red]Version {version} not found for '{prompt_name}'[/]"
                )
                raise typer.Exit(1)

            bl_pv = None
            if baseline is not None:
                bl_pv = (
                    await session.execute(
                        select(PromptVersion).where(
                            PromptVersion.prompt_id == prompt.id,
                            PromptVersion.version == baseline,
                        )
                    )
                ).scalars().first()
                if not bl_pv:
                    console.print(
                        f"[red]Baseline version {baseline} not found[/]"
                    )
                    raise typer.Exit(1)

            ds = (
                await session.execute(
                    select(Dataset).where(Dataset.name == dataset_name)
                )
            ).scalars().first()
            if not ds:
                console.print(f"[red]Dataset '{dataset_name}' not found[/]")
                raise typer.Exit(1)

            run = Run(
                prompt_version_id=pv.id,
                baseline_version_id=bl_pv.id if bl_pv else None,
                dataset_id=ds.id,
                evaluators=evaluators,
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)

            engine = RunnerEngine(use_cache=not no_cache)

            with _make_progress() as progress:
                task = progress.add_task("Running tests...", total=0)

                def on_progress(completed: int, total: int):
                    progress.update(task, total=total, completed=completed)

                await engine.execute(run, session, on_progress=on_progress)

            return await ReportGenerator().generate(run.id, session)

    report = _run(_do())

    if output_json:
        console.print_json(json.dumps(report.to_dict()))
        return

    _print_report(report)


# ── Report command ──────────────────────────────────────────────────────────


@app.command("report")
def show_report(
    run_id: str = typer.Argument(..., help="Run ID"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    html_output: Path = typer.Option(None, "--html", help="Write HTML report to file"),
):
    """Show the report for a completed run."""

    async def _do():
        await _ensure_db()
        async with async_session() as session:
            return await ReportGenerator().generate(run_id, session)

    report = _run(_do())

    if html_output:
        from lrts.engines.html_report import render_html
        html_output.write_text(render_html([report]))
        console.print(f"[green]✓[/] HTML report written to [bold]{html_output}[/]")
    elif output_json:
        console.print_json(json.dumps(report.to_dict()))
    else:
        _print_report(report)


# ── Report rendering ────────────────────────────────────────────────────────


def _drift_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    empty = width - filled
    if score < 0.15:
        color = "green"
    elif score < 0.3:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{'━' * filled}[/][dim]{'─' * empty}[/] [{color}]{score:.0%}[/]"


def _score_indicator(score: float | None) -> str:
    if score is None:
        return "[dim]—[/]"
    pct = int(score * 100)
    if score >= 0.85:
        return f"[green]{pct}%[/] [green]●[/]"
    elif score >= 0.6:
        return f"[yellow]{pct}%[/] [yellow]●[/]"
    else:
        return f"[red]{pct}%[/] [red]●[/]"


def _extract_judge_reason(detail: dict) -> str:
    evals = detail.get("evaluations", [])
    for ev in evals:
        if ev.get("evaluator") == "judge":
            return ev.get("detail", {}).get("reason", "")
    return ""


def _verdict_label(verdict: str) -> str:
    if verdict == "pass":
        return "[bold green]PASS[/]"
    elif verdict == "fail":
        return "[bold red]DRIFT[/]"
    elif verdict == "error":
        return "[bold yellow]ERROR[/]"
    return f"[dim]{verdict}[/]"


def _print_report(report):
    console.print()

    console.print(
        Panel(
            "[bold]L R T S[/]  [dim]—  LLM Regression Testing System[/]",
            box=box.HEAVY,
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()

    has_errors = any(i["verdict"] == "error" for i in report.items)

    drift_bar = _drift_bar(report.drift_score)
    if report.drift_score < 0.15:
        drift_verdict = "[bold green]LOW[/] — behavior is stable"
    elif report.drift_score < 0.3:
        drift_verdict = "[bold yellow]MODERATE[/] — some changes detected"
    else:
        drift_verdict = "[bold red]HIGH[/] — significant behavioral change"

    summary_lines = [
        f"  [bold]Drift[/]     {drift_bar}",
        f"  [bold]Verdict[/]   {drift_verdict}",
        "",
        f"  [bold]Tests[/]     {report.total} total   "
        f"[green]{report.passed} identical[/]   "
        f"[red]{report.failed} drifted[/]   "
        f"[yellow]{report.errors} errors[/]",
        f"  [bold]Run[/]       [dim]{report.run_id}[/]",
    ]
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="[bold]Regression Summary[/]",
            box=box.ROUNDED,
            padding=(1, 1),
        )
    )

    if has_errors:
        first_err = next(
            (i for i in report.items if i["verdict"] == "error"), None
        )
        if first_err and first_err.get("detail", {}).get("error"):
            console.print()
            console.print(
                Panel(
                    f"[red]{first_err['detail']['error']}[/]",
                    title="[bold red]Error Detail[/]",
                    box=box.ROUNDED,
                )
            )

    if not report.items:
        console.print()
        return

    console.print()

    has_judge = any(
        _extract_judge_reason(i.get("detail", {})) for i in report.items
    )

    if has_errors:
        table = Table(
            title="[bold]Test Results[/]",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("#", justify="right", width=3)
        table.add_column("Question", min_width=30)
        table.add_column("Error", min_width=40)
        table.add_column("Status", justify="center", width=8)

        for i, item in enumerate(report.items, 1):
            err_msg = item.get("detail", {}).get("error", "")
            table.add_row(
                str(i),
                item["input"] or "[dim]—[/]",
                f"[red]{err_msg}[/]" if err_msg else "[dim]—[/]",
                _verdict_label(item["verdict"]),
            )
    else:
        table = Table(
            title="[bold]Test Results[/]",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("#", justify="right", width=3)
        table.add_column("Question", min_width=25)
        table.add_column("Similarity", justify="center", width=12)
        table.add_column("Status", justify="center", width=8)
        table.add_column("What changed", min_width=30, max_width=50)

        for i, item in enumerate(report.items, 1):
            reason = _extract_judge_reason(item.get("detail", {}))
            if not reason:
                out_snip = (item.get("output_preview") or "")[:40]
                bl_snip = (item.get("baseline_preview") or "")[:40]
                if out_snip and bl_snip:
                    reason = f"[cyan]v2:[/] {out_snip}...\n[dim]v1:[/] {bl_snip}..."
                elif out_snip:
                    reason = f"[cyan]v2:[/] {out_snip}..."

            table.add_row(
                str(i),
                item["input"],
                _score_indicator(item["similarity"]),
                _verdict_label(item["verdict"]),
                reason,
            )

    console.print(table)

    console.print()
    console.print(
        Panel(
            "[green]●[/] [green]85%+[/] = Same behavior   "
            "[yellow]●[/] [yellow]60-84%[/] = Minor drift   "
            "[red]●[/] [red]<60%[/] = Significant change\n"
            "[bold]Similarity[/] = how closely v2 matches v1 "
            "(100% = identical, 0% = completely different)",
            box=box.ROUNDED,
            title="[dim]How to read this[/]",
            padding=(0, 1),
        )
    )
    console.print()


# ── Server command ──────────────────────────────────────────────────────────


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start the LRTS API server."""
    import uvicorn

    console.print(
        f"[bold green]LRTS[/] API server starting at "
        f"[link=http://{host}:{port}]http://{host}:{port}[/link]"
    )
    uvicorn.run("lrts.main:app", host=host, port=port, reload=True)


# ── Demo command ────────────────────────────────────────────────────────────


@app.command("demo")
def demo(
    provider: str = typer.Option(
        "openai", "--provider", "-p", help="Provider (openai, ollama, etc.)"
    ),
    model: str = typer.Option(
        "gpt-4o", "--model", "-m", help="Model to use"
    ),
    base_url: str = typer.Option(None, "--base-url", help="Custom base URL"),
    evaluators: str = typer.Option(
        None, "--evaluators", "-e",
        help="Evaluators to use (default: auto-detected based on provider)",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable response caching"),
):
    """Run a full demo with built-in example data. Zero config needed."""
    from lrts.examples.demo_data import DEMO_ITEMS

    if evaluators is None:
        if provider.lower() in ("openai",):
            evaluators = "exact,keyword,structure,semantic"
        else:
            evaluators = "exact,keyword,structure,judge"

    async def _do():
        await _ensure_db()
        async with async_session() as session:
            console.print()
            console.print(
                Panel(
                    "[bold]Customer Support Bot[/] — Regression Test\n"
                    "\n"
                    f"[dim]Comparing prompt v1 (basic) vs v2 (refined)[/]\n"
                    f"[dim]Model: {model} via {provider}[/]\n"
                    f"[dim]Evaluators: {evaluators}[/]",
                    title="[bold cyan]LRTS Demo[/]",
                    box=box.HEAVY,
                    padding=(1, 2),
                )
            )
            console.print()
            console.print(
                f"  [bold green]>[/] Setting up demo data "
                f"[dim]({len(DEMO_ITEMS)} test cases)[/]"
            )
            console.print("  [bold green]>[/] Running regression test [dim](v2 vs v1)[/]")
            console.print()

            with _make_progress() as progress:
                task = progress.add_task("Testing...", total=0)

                def on_progress(completed: int, total: int):
                    progress.update(task, total=total, completed=completed)

                return await orchestrate_demo(
                    session, provider, model, base_url, evaluators,
                    on_progress=on_progress,
                    use_cache=not no_cache,
                )

    report = _run(_do())
    _print_report(report)


# ── Init command ────────────────────────────────────────────────────────────

_INIT_CONFIG_TEMPLATE = """\
# LRTS — LLM Regression Testing System
# Configuration file for your project

provider: openai
model: gpt-4o
# base_url: http://localhost:11434/v1   # uncomment for Ollama/local models
threshold: 0.85

prompts:
  my-assistant:
    v1: prompts/v1.txt
    v2: prompts/v2.txt

datasets:
  test-set: datasets/test.jsonl

tests:
  - prompt: my-assistant
    version: 2
    baseline: 1
    dataset: test-set
    evaluators: [exact, keyword, structure, judge]
"""

_INIT_PROMPT_V1 = "You are a helpful assistant."
_INIT_PROMPT_V2 = "You are a helpful and concise assistant. Keep responses under 100 words."
_INIT_DATASET = '{"input": "What can you help me with?"}\n{"input": "Explain how you work."}\n'


@app.command("init")
def init_project(
    directory: Path = typer.Argument(
        ".", help="Directory to initialize (default: current)"
    ),
):
    """Scaffold an LRTS test suite in your project."""
    d = Path(directory).resolve()

    config_path = d / ".lrts.yml"
    if config_path.exists():
        console.print(f"[yellow].lrts.yml already exists in {d}[/]")
        raise typer.Exit(1)

    (d / "prompts").mkdir(exist_ok=True)
    (d / "datasets").mkdir(exist_ok=True)

    config_path.write_text(_INIT_CONFIG_TEMPLATE)
    (d / "prompts" / "v1.txt").write_text(_INIT_PROMPT_V1)
    (d / "prompts" / "v2.txt").write_text(_INIT_PROMPT_V2)
    (d / "datasets" / "test.jsonl").write_text(_INIT_DATASET)

    gitignore_path = d / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(".lrts/\n")
    else:
        gi = gitignore_path.read_text()
        if ".lrts/" not in gi:
            with open(gitignore_path, "a") as f:
                f.write("\n.lrts/\n")

    console.print()
    console.print("[bold green]Initialized LRTS test suite[/]\n")
    console.print(f"  [dim]{d}/[/]")
    console.print("  [cyan].lrts.yml[/]        config file")
    console.print("  [cyan]prompts/v1.txt[/]   baseline prompt")
    console.print("  [cyan]prompts/v2.txt[/]   new prompt version")
    console.print("  [cyan]datasets/test.jsonl[/]  test dataset")
    console.print()
    console.print("  [dim].lrts/ added to .gitignore (local test DB)[/]")
    console.print()
    console.print("  Edit the files, then run: [bold]lrts test[/]")
    console.print()


# ── Test command (CI/CD) ────────────────────────────────────────────────────


@app.command("test")
def test_command(
    config_path: Path = typer.Option(
        None, "--config", "-c", help="Path to .lrts.yml (auto-detected if omitted)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable response caching"),
    output_json: bool = typer.Option(False, "--json", help="JSON output for CI"),
    html_output: Path = typer.Option(None, "--html", help="Write HTML report to file"),
):
    """Run regression tests from .lrts.yml config. Exit code 1 on drift."""
    from lrts.config_file import find_config, load_config
    from lrts.db import use_local_db

    if config_path is None:
        config_path = find_config()
    if config_path is None or not config_path.exists():
        console.print(
            "[red]No .lrts.yml found.[/] Run [bold]lrts init[/] to create one."
        )
        raise typer.Exit(1)

    config_dir = config_path.parent
    cfg = load_config(config_path)

    use_local_db(config_dir)

    if not cfg.tests:
        console.print("[yellow]No tests defined in .lrts.yml[/]")
        raise typer.Exit(0)

    all_reports = []
    any_drift = False

    async def _do():
        nonlocal any_drift
        await _ensure_db()
        async with async_session() as session:
            console.print()
            console.print(
                Panel(
                    f"[bold]Running {len(cfg.tests)} test(s)[/]\n"
                    f"[dim]Config: {config_path}[/]\n"
                    f"[dim]Model: {cfg.model} via {cfg.provider}[/]\n"
                    f"[dim]Threshold: {cfg.threshold}[/]",
                    title="[bold cyan]LRTS Test[/]",
                    box=box.HEAVY,
                    padding=(1, 2),
                )
            )
            console.print()

            for ti, spec in enumerate(cfg.tests, 1):
                console.print(
                    f"  [bold]Test {ti}/{len(cfg.tests)}:[/] "
                    f"{spec.prompt} v{spec.version} vs v{spec.baseline} "
                    f"[dim]({spec.dataset})[/]"
                )

                with _make_progress() as progress:
                    task = progress.add_task("  Running...", total=0)

                    def on_progress(completed: int, total: int):
                        progress.update(task, total=total, completed=completed)

                    report = await run_test_spec(
                        session, cfg, spec, config_dir,
                        on_progress=on_progress,
                        use_cache=not no_cache,
                    )

                all_reports.append(report)

                if report.drift_score > (1.0 - cfg.threshold):
                    any_drift = True

    _run(_do())

    if html_output:
        from lrts.engines.html_report import render_html
        html_output.write_text(render_html(all_reports))
        console.print(f"[green]✓[/] HTML report written to [bold]{html_output}[/]")

    if output_json:
        out = [r.to_dict() for r in all_reports]
        console.print_json(json.dumps(out))
    else:
        for report in all_reports:
            _print_report(report)

    if any_drift:
        console.print("[bold red]DRIFT DETECTED[/] — threshold exceeded\n")
        raise typer.Exit(1)
    else:
        console.print("[bold green]ALL TESTS PASSED[/] — no significant drift\n")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
