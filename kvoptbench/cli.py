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
) -> None:
    """Generate a markdown report from a summary CSV."""
    from kvoptbench.reports.generate import generate_report

    generate_report(input_path=input, output_path=output)
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

