"""Top-level Typer CLI for KVOptBench."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from kvoptbench.config import ConfigError, validate_config

app = typer.Typer(help="KVOptBench local/mock benchmark harness.")


@app.command("validate-config")
def validate_config_command(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML config."),
) -> None:
    """Validate an experiment config."""
    try:
        parsed = validate_config(config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print(f"[green]OK[/green] {config} -> {parsed.experiment_id}")


@app.command("endpoint-check")
def endpoint_check_command(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML config."),
) -> None:
    """Check whether a configured OpenAI-compatible endpoint is reachable."""
    import asyncio

    from kvoptbench.client.openai_compat import OpenAICompatClient

    try:
        parsed = validate_config(config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    health = asyncio.run(OpenAICompatClient(parsed).healthcheck())
    if not health.ok:
        print(f"[red]FAILED[/red] {health.url}: {health.error_message}")
        raise typer.Exit(code=1)
    models = ", ".join(health.model_ids) if health.model_ids else "models unavailable"
    print(f"[green]OK[/green] {health.url} ({models})")


@app.command("engine-command")
def engine_command_command(
    engine: str = typer.Option(..., "--engine", "-e"),
    strategy: str = typer.Option(..., "--strategy", "-s"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Render an engine server command preview without launching it."""
    from kvoptbench.engines.profiles import render_command_preview

    preview = render_command_preview(
        engine=engine,
        strategy=strategy,
        model_id=model_id,
        host=host,
        port=port,
    )
    print(f"[bold]{preview.engine}[/bold] / {preview.strategy}")
    print(preview.description)
    print(f"[cyan]{preview.command}[/cyan]")
    print(f"Endpoint: {preview.endpoint.base_url}")
    print(f"Notes: {preview.notes}")


@app.command("cache-plan")
def cache_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    experiment_prefix: str = typer.Option(..., "--experiment-prefix"),
    provider: str = typer.Option(..., "--provider"),
    engine: str = typer.Option(..., "--engine", "-e"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    base_url: str = typer.Option(..., "--base-url"),
    shared_workload_file: Path = typer.Option(..., "--shared-workload-file"),
    random_workload_file: Path = typer.Option(..., "--random-workload-file"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_output_tokens: int = typer.Option(256, "--max-output-tokens", min=1),
) -> None:
    """Write cache experiment YAML configs."""
    from kvoptbench.experiments.cache import write_cache_plan_configs

    written = write_cache_plan_configs(
        plan_dir=plan_dir,
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        shared_workload_file=shared_workload_file,
        random_workload_file=random_workload_file,
        output_dir=output_dir,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
    )
    print(f"[green]Wrote {len(written)} cache experiment configs[/green] to {plan_dir}")


@app.command("cache-run")
def cache_run_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
) -> None:
    """Run all YAML configs in a cache experiment plan directory."""
    from kvoptbench.experiments.cache import run_cache_plan

    outputs = run_cache_plan(plan_dir)
    print(f"[green]Ran {len(outputs)} cache experiment configs[/green]")
    for output in outputs:
        print(output)


@app.command("cache-compare")
def cache_compare_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare cache experiment JSONL results against random-prefix controls."""
    from kvoptbench.analysis.cache_compare import compare_cache_results

    compare_cache_results(input_path=input, output_path=output)
    print(f"[green]Wrote cache comparison[/green] {output}")


@app.command("generate-workload")
def generate_workload_command(
    profile: str = typer.Option(..., "--profile", "-p"),
    out: Path = typer.Option(..., "--out", "-o"),
    count: int = typer.Option(10, "--count", min=1),
    target_input_tokens: int = typer.Option(32768, "--target-input-tokens", min=1),
    target_output_tokens: int = typer.Option(256, "--target-output-tokens", min=1),
) -> None:
    """Generate workload JSONL."""
    from kvoptbench.workloads.generate import generate_to_file

    generated = generate_to_file(
        profile=profile,
        out=out,
        count=count,
        target_input_tokens=target_input_tokens,
        target_output_tokens=target_output_tokens,
    )
    print(f"[green]Wrote[/green] {generated} tasks to {out}")


@app.command("mock-server")
def mock_server_command(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
) -> None:
    """Start the local mock OpenAI-compatible server."""
    from kvoptbench.mock_server.main import run_server

    run_server(host=host, port=port)


@app.command("run")
def run_command(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML config."),
) -> None:
    """Run an experiment."""
    import asyncio

    from kvoptbench.runner.experiment import run_experiment

    output = asyncio.run(run_experiment(config))
    print(f"[green]Wrote results[/green] {output}")


@app.command("summarize")
def summarize_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Aggregate raw JSONL results into CSV."""
    from kvoptbench.analysis.summarize import summarize_results

    summarize_results(input_path=input, output_path=output)
    print(f"[green]Wrote summary[/green] {output}")


@app.command("report")
def report_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
    cache_input: Path | None = typer.Option(None, "--cache-input"),
) -> None:
    """Generate a markdown report from a summary CSV."""
    from kvoptbench.reports.generate import generate_report

    generate_report(input_path=input, output_path=output, cache_input_path=cache_input)
    print(f"[green]Wrote report[/green] {output}")


@app.command("strategy-select")
def strategy_select_command(
    input: Path = typer.Option(..., "--input", "-i"),
) -> None:
    """Run the placeholder strategy selector."""
    from kvoptbench.strategy.selector import select_strategy_from_summary

    print(select_strategy_from_summary(input))


if __name__ == "__main__":
    app()

