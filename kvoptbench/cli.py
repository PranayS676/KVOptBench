"""Top-level Typer CLI for KVOptBench."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import typer
from rich import print

from kvoptbench.config import ConfigError, validate_config

app = typer.Typer(help="KVOptBench cache-aware LLM inference benchmark.")
dataset_app = typer.Typer(help="Prepare public dataset workloads.")
workflow_app = typer.Typer(help="Run end-to-end benchmark workflows.")
schema_app = typer.Typer(help="Export and check artifact contract schemas.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(workflow_app, name="workflow")
app.add_typer(schema_app, name="schema")


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


@app.command("init")
def init_command(
    output_dir: Path = typer.Option(
        Path(".kvoptbench-starter"),
        "--output-dir",
        help="Directory where starter configs, workloads, and manifests will be written.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite generated starter files."),
) -> None:
    """Scaffold a golden QASPER/mock starter benchmark pack."""
    from kvoptbench.init import scaffold_project

    try:
        result = scaffold_project(output_dir, force=force)
    except FileExistsError as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    print(f"[green]Wrote config[/green] {result.config_path}")
    print(f"[green]Wrote workload[/green] {result.workload_path}")
    print(f"[green]Wrote dataset manifest[/green] {result.dataset_manifest_path}")


@app.command("doctor")
def doctor_command(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML config."),
    skip_endpoint: bool = typer.Option(
        False,
        "--skip-endpoint",
        help="Validate local files and environment without probing the endpoint.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON.",
    ),
) -> None:
    """Run preflight checks for config, workload, telemetry, endpoint, and environment."""
    from kvoptbench.doctor import run_doctor

    report = run_doctor(config, check_endpoint=not skip_endpoint)
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        for check in report.checks:
            color = {
                "ok": "green",
                "warn": "yellow",
                "fail": "red",
                "skipped": "cyan",
            }[check.status]
            print(f"[{color}]{check.status.upper()}[/{color}] {check.name}: {check.message}")
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("validate-results")
def validate_results_command(
    input: Path = typer.Option(..., "--input", "-i", help="Result JSONL file or directory."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate request-level JSONL result rows against the stable contract."""
    from kvoptbench.contracts import validate_result_rows

    report = validate_result_rows(input)
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        _print_validation_report(report.model_dump(mode="json"))
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("validate-package")
def validate_package_command(
    path: Path = typer.Option(..., "--path", help="Result package directory."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate a result package manifest and package-relative artifact hashes."""
    from kvoptbench.contracts import validate_result_package

    report = validate_result_package(path)
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        _print_validation_report(report.model_dump(mode="json"))
    if not report.ok:
        raise typer.Exit(code=1)


@schema_app.command("export")
def schema_export_command(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for schema files."),
    check: bool = typer.Option(False, "--check", help="Fail if committed schemas are stale."),
) -> None:
    """Export JSON Schema files for stable KVOptBench artifacts."""
    from kvoptbench.contracts import check_schema_files, write_schema_files

    if check:
        mismatches = check_schema_files(output_dir)
        if mismatches:
            for mismatch in mismatches:
                print(f"[red]FAILED[/red] {mismatch}")
            raise typer.Exit(code=1)
        print("[green]Schema snapshots are current[/green]")
        return

    written = write_schema_files(output_dir)
    for name, path in written.items():
        print(f"[green]Wrote schema[/green] {name}: {path}")


@workflow_app.command("run")
def workflow_run_command(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML config."),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Directory for summary, report, and advisor outputs.",
    ),
    package_dir: Path | None = typer.Option(
        None,
        "--package-dir",
        help="Optional result package output directory.",
    ),
    dataset_manifest: list[Path] = typer.Option(
        [],
        "--dataset-manifest",
        help="Dataset manifest JSON file. Can be repeated.",
    ),
    run_name: str | None = typer.Option(None, "--run-name"),
    skip_run: bool = typer.Option(
        False,
        "--skip-run",
        help="Reuse the config output_file instead of running the endpoint.",
    ),
) -> None:
    """Run one config through run, summarize, report, advisor, and package steps."""
    from kvoptbench.workflow import run_config_workflow

    try:
        result = run_config_workflow(
            config,
            output_dir=output_dir,
            package_dir=package_dir,
            dataset_manifest_paths=dataset_manifest,
            run_name=run_name,
            skip_run=skip_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    print(f"[green]Wrote raw results[/green] {result.raw_results_path}")
    print(f"[green]Wrote summary[/green] {result.summary_path}")
    print(f"[green]Wrote report[/green] {result.report_path}")
    print(f"[green]Wrote strategy advisor JSON[/green] {result.strategy_json_path}")
    print(f"[green]Wrote strategy advisor markdown[/green] {result.strategy_markdown_path}")
    if result.package_manifest_path is not None:
        print(f"[green]Wrote result package[/green] {result.package_dir}")
        print(f"[green]Wrote package manifest[/green] {result.package_manifest_path}")


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


@dataset_app.command("prepare")
def dataset_prepare_command(
    source: str = typer.Option(..., "--source"),
    mode: str = typer.Option(..., "--mode"),
    out: Path = typer.Option(..., "--out", "-o"),
    manifest: Path = typer.Option(..., "--manifest"),
    source_path: Path | None = typer.Option(None, "--source-path"),
    split: str | None = typer.Option(None, "--split"),
    max_items: int | None = typer.Option(None, "--max-items", min=1),
    seed: int = typer.Option(7, "--seed"),
    target_input_tokens: int = typer.Option(32768, "--target-input-tokens", min=1),
    target_output_tokens: int = typer.Option(256, "--target-output-tokens", min=1),
    context_buckets: str | None = typer.Option(None, "--context-buckets"),
    book_ids: str | None = typer.Option(None, "--book-ids"),
    download: bool = typer.Option(False, "--download"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    dataset_revision: str | None = typer.Option(None, "--dataset-revision"),
    subset: str | None = typer.Option(None, "--subset"),
    force: bool = typer.Option(False, "--force"),
    tokenizer_id: str | None = typer.Option(None, "--tokenizer-id"),
    tokenizer_revision: str | None = typer.Option(None, "--tokenizer-revision"),
) -> None:
    """Prepare a public dataset workload JSONL file and manifest."""
    from kvoptbench.datasets.manifest import DatasetPrepareOptions
    from kvoptbench.datasets.registry import get_dataset_adapter

    try:
        adapter = get_dataset_adapter(source)
        result = adapter.prepare(
            DatasetPrepareOptions(
                source=source,
                mode=mode,
                out=out,
                manifest=manifest,
                source_path=source_path,
                split=split,
                max_items=max_items,
                seed=seed,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                context_buckets=_parse_context_buckets(context_buckets) or (),
                book_ids=_parse_csv_strings(book_ids),
                download=download,
                cache_dir=cache_dir,
                dataset_revision=dataset_revision,
                subset=_parse_csv_strings(subset),
                force=force,
                tokenizer_id=tokenizer_id,
                tokenizer_revision=tokenizer_revision,
            )
        )
    except ValueError as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    print(
        f"[green]Wrote {result.row_count} dataset workload rows[/green] "
        f"to {result.output_path}"
    )
    print(f"[green]Wrote manifest[/green] {result.manifest_path}")


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


@app.command("prefix-sweep-compare")
def prefix_sweep_compare_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare cache behavior across shared-prefix overlap ratios."""
    from kvoptbench.analysis.prefix_sweep import compare_prefix_sweep_results

    compare_prefix_sweep_results(input_path=input, output_path=output)
    print(f"[green]Wrote prefix sweep comparison[/green] {output}")


@app.command("prefill-decode-plan")
def prefill_decode_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    experiment_prefix: str = typer.Option(..., "--experiment-prefix"),
    provider: str = typer.Option(..., "--provider"),
    engine: str = typer.Option(..., "--engine", "-e"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    base_url: str = typer.Option(..., "--base-url"),
    workload_file: Path = typer.Option(..., "--workload-file"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_output_tokens: int = typer.Option(512, "--max-output-tokens", min=1),
) -> None:
    """Write prefill/decode experiment YAML configs."""
    from kvoptbench.experiments.prefill_decode import write_prefill_decode_plan_configs

    written = write_prefill_decode_plan_configs(
        plan_dir=plan_dir,
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        workload_file=workload_file,
        output_dir=output_dir,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
    )
    print(f"[green]Wrote {len(written)} prefill/decode experiment configs[/green] to {plan_dir}")


@app.command("prefill-decode-run")
def prefill_decode_run_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
) -> None:
    """Run all YAML configs in a prefill/decode experiment plan directory."""
    from kvoptbench.experiments.prefill_decode import run_prefill_decode_plan

    outputs = run_prefill_decode_plan(plan_dir)
    print(f"[green]Ran {len(outputs)} prefill/decode experiment configs[/green]")
    for output in outputs:
        print(output)


@app.command("prefill-decode-compare")
def prefill_decode_compare_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare prefill/decode timing metrics by input/output bucket."""
    from kvoptbench.analysis.prefill_decode import compare_prefill_decode_results

    compare_prefill_decode_results(input_path=input, output_path=output)
    print(f"[green]Wrote prefill/decode comparison[/green] {output}")


@app.command("long-context-plan")
def long_context_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    experiment_prefix: str = typer.Option(..., "--experiment-prefix"),
    provider: str = typer.Option(..., "--provider"),
    engine: str = typer.Option(..., "--engine", "-e"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    base_url: str = typer.Option(..., "--base-url"),
    workload_file: Path = typer.Option(..., "--workload-file"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_output_tokens: int = typer.Option(256, "--max-output-tokens", min=1),
) -> None:
    """Write long-context pressure experiment YAML configs."""
    from kvoptbench.experiments.long_context import write_long_context_plan_configs

    written = write_long_context_plan_configs(
        plan_dir=plan_dir,
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        workload_file=workload_file,
        output_dir=output_dir,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
    )
    print(f"[green]Wrote {len(written)} long-context experiment configs[/green] to {plan_dir}")


@app.command("long-context-run")
def long_context_run_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
) -> None:
    """Run all YAML configs in a long-context pressure plan directory."""
    from kvoptbench.experiments.long_context import run_long_context_plan

    outputs = run_long_context_plan(plan_dir)
    print(f"[green]Ran {len(outputs)} long-context experiment configs[/green]")
    for output in outputs:
        print(output)


@app.command("long-context-compare")
def long_context_compare_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare long-context timing and throughput metrics by context bucket."""
    from kvoptbench.analysis.long_context import compare_long_context_results

    compare_long_context_results(input_path=input, output_path=output)
    print(f"[green]Wrote long-context comparison[/green] {output}")


@app.command("kv-quant-plan")
def kv_quant_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    experiment_prefix: str = typer.Option(..., "--experiment-prefix"),
    provider: str = typer.Option(..., "--provider"),
    engine: str = typer.Option(..., "--engine", "-e"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    base_url: str = typer.Option(..., "--base-url"),
    workload_file: Path = typer.Option(..., "--workload-file"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    workload_profile: str = typer.Option("long_context_pressure", "--workload-profile"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_output_tokens: int = typer.Option(256, "--max-output-tokens", min=1),
) -> None:
    """Write KV cache quantization experiment YAML configs."""
    from kvoptbench.experiments.kv_quantization import write_kv_quantization_plan_configs

    written = write_kv_quantization_plan_configs(
        plan_dir=plan_dir,
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        workload_file=workload_file,
        output_dir=output_dir,
        workload_profile=workload_profile,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
    )
    print(f"[green]Wrote {len(written)} KV quantization experiment configs[/green] to {plan_dir}")


@app.command("kv-quant-run")
def kv_quant_run_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
) -> None:
    """Run all YAML configs in a KV cache quantization plan directory."""
    from kvoptbench.experiments.kv_quantization import run_kv_quantization_plan

    outputs = run_kv_quantization_plan(plan_dir)
    print(f"[green]Ran {len(outputs)} KV quantization experiment configs[/green]")
    for output in outputs:
        print(output)


@app.command("kv-quant-compare")
def kv_quant_compare_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare baseline and quantized KV cache results."""
    from kvoptbench.analysis.kv_quantization import compare_kv_quantization_results

    compare_kv_quantization_results(input_path=input, output_path=output)
    print(f"[green]Wrote KV quantization comparison[/green] {output}")


@app.command("kv-offload-plan")
def kv_offload_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    experiment_prefix: str = typer.Option(..., "--experiment-prefix"),
    provider: str = typer.Option(..., "--provider"),
    engine: str = typer.Option(..., "--engine", "-e"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    base_url: str = typer.Option(..., "--base-url"),
    workload_file: Path = typer.Option(..., "--workload-file"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    workload_profile: str = typer.Option("long_context_pressure", "--workload-profile"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_output_tokens: int = typer.Option(256, "--max-output-tokens", min=1),
) -> None:
    """Write KV offload experiment YAML configs."""
    from kvoptbench.experiments.kv_offload import write_kv_offload_plan_configs

    written = write_kv_offload_plan_configs(
        plan_dir=plan_dir,
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        workload_file=workload_file,
        output_dir=output_dir,
        workload_profile=workload_profile,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
    )
    print(f"[green]Wrote {len(written)} KV offload experiment configs[/green] to {plan_dir}")


@app.command("kv-offload-run")
def kv_offload_run_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
) -> None:
    """Run all YAML configs in a KV offload plan directory."""
    from kvoptbench.experiments.kv_offload import run_kv_offload_plan

    outputs = run_kv_offload_plan(plan_dir)
    print(f"[green]Ran {len(outputs)} KV offload experiment configs[/green]")
    for output in outputs:
        print(output)


@app.command("kv-offload-compare")
def kv_offload_compare_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare baseline and KV offload results."""
    from kvoptbench.analysis.kv_offload import compare_kv_offload_results

    compare_kv_offload_results(input_path=input, output_path=output)
    print(f"[green]Wrote KV offload comparison[/green] {output}")


@app.command("spec-decoding-plan")
def spec_decoding_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    experiment_prefix: str = typer.Option(..., "--experiment-prefix"),
    provider: str = typer.Option(..., "--provider"),
    engine: str = typer.Option(..., "--engine", "-e"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    base_url: str = typer.Option(..., "--base-url"),
    workload_file: Path = typer.Option(..., "--workload-file"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    workload_profile: str = typer.Option("decode_heavy", "--workload-profile"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_output_tokens: int = typer.Option(512, "--max-output-tokens", min=1),
) -> None:
    """Write speculative decoding experiment YAML configs."""
    from kvoptbench.experiments.speculative_decoding import (
        write_speculative_decoding_plan_configs,
    )

    written = write_speculative_decoding_plan_configs(
        plan_dir=plan_dir,
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        workload_file=workload_file,
        output_dir=output_dir,
        workload_profile=workload_profile,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
    )
    print(f"[green]Wrote {len(written)} speculative decoding experiment configs[/green] to {plan_dir}")


@app.command("spec-decoding-run")
def spec_decoding_run_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
) -> None:
    """Run all YAML configs in a speculative decoding plan directory."""
    from kvoptbench.experiments.speculative_decoding import run_speculative_decoding_plan

    outputs = run_speculative_decoding_plan(plan_dir)
    print(f"[green]Ran {len(outputs)} speculative decoding experiment configs[/green]")
    for output in outputs:
        print(output)


@app.command("spec-decoding-compare")
def spec_decoding_compare_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare baseline and speculative decoding results."""
    from kvoptbench.analysis.speculative_decoding import compare_speculative_decoding_results

    compare_speculative_decoding_results(input_path=input, output_path=output)
    print(f"[green]Wrote speculative decoding comparison[/green] {output}")


@app.command("disagg-plan")
def disagg_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    experiment_prefix: str = typer.Option(..., "--experiment-prefix"),
    provider: str = typer.Option(..., "--provider"),
    engine: str = typer.Option(..., "--engine", "-e"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    base_url: str = typer.Option(..., "--base-url"),
    workload_file: Path = typer.Option(..., "--workload-file"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    workload_profile: str = typer.Option("prefill_decode_grid", "--workload-profile"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_output_tokens: int = typer.Option(512, "--max-output-tokens", min=1),
) -> None:
    """Write prefill/decode disaggregation experiment YAML configs."""
    from kvoptbench.experiments.prefill_decode_disaggregation import (
        write_prefill_decode_disaggregation_plan_configs,
    )

    written = write_prefill_decode_disaggregation_plan_configs(
        plan_dir=plan_dir,
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        workload_file=workload_file,
        output_dir=output_dir,
        workload_profile=workload_profile,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
    )
    print(f"[green]Wrote {len(written)} disaggregation experiment configs[/green] to {plan_dir}")


@app.command("disagg-run")
def disagg_run_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
) -> None:
    """Run all YAML configs in a prefill/decode disaggregation plan directory."""
    from kvoptbench.experiments.prefill_decode_disaggregation import (
        run_prefill_decode_disaggregation_plan,
    )

    outputs = run_prefill_decode_disaggregation_plan(plan_dir)
    print(f"[green]Ran {len(outputs)} disaggregation experiment configs[/green]")
    for output in outputs:
        print(output)


@app.command("disagg-compare")
def disagg_compare_command(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare baseline and prefill/decode disaggregation results."""
    from kvoptbench.analysis.prefill_decode_disaggregation import (
        compare_prefill_decode_disaggregation_results,
    )

    compare_prefill_decode_disaggregation_results(input_path=input, output_path=output)
    print(f"[green]Wrote disaggregation comparison[/green] {output}")


@app.command("generate-workload")
def generate_workload_command(
    profile: str = typer.Option(..., "--profile", "-p"),
    out: Path = typer.Option(..., "--out", "-o"),
    count: int = typer.Option(10, "--count", min=1),
    target_input_tokens: int = typer.Option(32768, "--target-input-tokens", min=1),
    target_output_tokens: int = typer.Option(256, "--target-output-tokens", min=1),
    context_buckets: str | None = typer.Option(
        None,
        "--context-buckets",
        help="Comma-separated token buckets for long_context_pressure.",
    ),
) -> None:
    """Generate workload JSONL."""
    from kvoptbench.workloads.generate import generate_to_file

    generated = generate_to_file(
        profile=profile,
        out=out,
        count=count,
        target_input_tokens=target_input_tokens,
        target_output_tokens=target_output_tokens,
        context_buckets=_parse_context_buckets(context_buckets),
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


@app.command("import")
def import_command(
    tool: Literal["vllm-bench", "genai-perf", "aiperf"] = typer.Option(..., "--tool"),
    source: Path = typer.Option(..., "--source", "-s"),
    output: Path = typer.Option(..., "--output", "-o"),
    experiment_id: str = typer.Option(..., "--experiment-id"),
    workload: str = typer.Option(..., "--workload"),
    provider: str = typer.Option("local", "--provider"),
    engine: str = typer.Option("unknown", "--engine", "-e"),
    strategy: str = typer.Option("imported", "--strategy"),
    model_id: str | None = typer.Option(None, "--model-id", "-m"),
    run_id: str | None = typer.Option(None, "--run-id"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    granularity: Literal["auto", "request", "aggregate"] = typer.Option(
        "auto", "--granularity"
    ),
    manifest_output: Path | None = typer.Option(None, "--manifest-output"),
    fail_on_missing_required: bool = typer.Option(False, "--fail-on-missing-required"),
) -> None:
    """Import external benchmark artifacts without claiming KVOptBench ran them."""
    try:
        if tool == "vllm-bench":
            from kvoptbench.importers.vllm_bench import import_vllm_bench

            rows = import_vllm_bench(
                source,
                experiment_id=experiment_id,
                workload=workload,
                provider=provider,
                engine=engine if engine != "unknown" else "vllm",
                strategy=strategy,
                model_id=model_id,
                run_id=run_id,
                concurrency=concurrency,
            )
            missing_metrics = _collect_import_missing_metrics(rows)
            _write_jsonl(output, rows)
            manifest = _basic_import_manifest(
                tool=tool,
                source=source,
                granularity="request",
                row_count=len(rows),
                missing_metrics=missing_metrics,
            )
            written_kind = "request JSONL"
        else:
            from kvoptbench.importers.aiperf import import_aiperf
            from kvoptbench.importers.genai_perf import import_genai_perf

            importer = import_genai_perf if tool == "genai-perf" else import_aiperf
            result = importer(
                source,
                experiment_id=experiment_id,
                workload=workload,
                provider=provider,
                engine=engine,
                strategy=strategy,
                model_id=model_id,
                run_id=run_id,
                concurrency=concurrency,
                granularity=granularity,
            )
            missing_metrics = result.missing_metrics
            manifest = result.source_manifest
            if result.granularity == "request":
                _write_jsonl(output, result.request_rows)
                written_kind = "request JSONL"
            else:
                _write_csv(output, result.summary_rows)
                written_kind = "summary CSV"

        if manifest_output is not None:
            manifest_output.parent.mkdir(parents=True, exist_ok=True)
            manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if fail_on_missing_required and missing_metrics:
            raise ValueError("Missing imported metrics: " + ", ".join(missing_metrics))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]Wrote imported {written_kind}[/green] {output}")
    if manifest_output is not None:
        print(f"[green]Wrote import manifest[/green] {manifest_output}")


@app.command("strategy-plan")
def strategy_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    matrix_id: str = typer.Option(..., "--matrix-id"),
    provider: str = typer.Option(..., "--provider"),
    engine: str = typer.Option(..., "--engine", "-e"),
    model_id: str = typer.Option(..., "--model-id", "-m"),
    base_url: str = typer.Option(..., "--base-url"),
    workload_pack: Path = typer.Option(..., "--workload-pack"),
    strategy_family: list[str] = typer.Option(..., "--strategy-family"),
    strategy: list[str] = typer.Option(..., "--strategy"),
    concurrency: list[int] = typer.Option([1], "--concurrency", min=1),
    output_dir: Path = typer.Option(..., "--output-dir"),
    repeat_count: int = typer.Option(1, "--repeat-count", min=1),
    randomization_seed: int = typer.Option(0, "--randomization-seed"),
    run_label: str = typer.Option("exploratory", "--run-label"),
    max_output_tokens: int = typer.Option(256, "--max-output-tokens", min=1),
    endpoint_type: str | None = typer.Option(None, "--endpoint-type"),
) -> None:
    """Write a strategy matrix manifest and normal experiment YAML configs."""
    from kvoptbench.strategy.plan_run import write_strategy_plan

    try:
        result = write_strategy_plan(
            plan_dir=plan_dir,
            matrix_id=matrix_id,
            provider=provider,
            engine=engine,
            model_id=model_id,
            base_url=base_url,
            workload_pack=workload_pack,
            strategy_families=strategy_family,
            strategies=strategy,
            concurrencies=concurrency,
            output_dir=output_dir,
            repeat_count=repeat_count,
            randomization_seed=randomization_seed,
            run_label=run_label,
            max_output_tokens=max_output_tokens,
            endpoint_type=endpoint_type,
        )
    except ValueError as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    print(
        f"[green]Wrote {len(result.config_paths)} strategy configs[/green] "
        f"to {result.plan_dir}"
    )
    print(f"[green]Wrote matrix manifest[/green] {result.manifest_path}")


@app.command("strategy-run")
def strategy_run_command(
    matrix_manifest: Path = typer.Option(..., "--matrix-manifest"),
    output_run_manifest: Path | None = typer.Option(None, "--output-run-manifest"),
    repeat_count: int | None = typer.Option(None, "--repeat-count", min=1),
    randomization_seed: int | None = typer.Option(None, "--randomization-seed"),
    randomize: bool = typer.Option(False, "--randomize"),
    block_randomization: bool = typer.Option(False, "--block-randomization"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run or dry-run a strategy matrix through a deterministic schedule."""
    from kvoptbench.strategy.plan_run import run_strategy_plan

    try:
        result = run_strategy_plan(
            matrix_manifest=matrix_manifest,
            output_run_manifest=output_run_manifest,
            repeat_count=repeat_count,
            randomization_seed=randomization_seed,
            randomize=randomize,
            block_randomization=block_randomization,
            dry_run=dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    print(f"[green]Wrote run manifest[/green] {result.run_manifest_path}")
    if dry_run:
        print("[yellow]Dry run only; no experiment configs were executed.[/yellow]")
    else:
        print(f"[green]Ran {len(result.output_paths)} scheduled configs[/green]")


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
    prefix_sweep_input: Path | None = typer.Option(None, "--prefix-sweep-input"),
    prefill_decode_input: Path | None = typer.Option(None, "--prefill-decode-input"),
    long_context_input: Path | None = typer.Option(None, "--long-context-input"),
    kv_quant_input: Path | None = typer.Option(None, "--kv-quant-input"),
    kv_offload_input: Path | None = typer.Option(None, "--kv-offload-input"),
    spec_decoding_input: Path | None = typer.Option(None, "--spec-decoding-input"),
    disagg_input: Path | None = typer.Option(None, "--disagg-input"),
    strategy_input: Path | None = typer.Option(None, "--strategy-input"),
) -> None:
    """Generate a markdown report from a summary CSV."""
    from kvoptbench.reports.generate import generate_report

    generate_report(
        input_path=input,
        output_path=output,
        cache_input_path=cache_input,
        prefix_sweep_input_path=prefix_sweep_input,
        prefill_decode_input_path=prefill_decode_input,
        long_context_input_path=long_context_input,
        kv_quant_input_path=kv_quant_input,
        kv_offload_input_path=kv_offload_input,
        spec_decoding_input_path=spec_decoding_input,
        disagg_input_path=disagg_input,
        strategy_input_path=strategy_input,
    )
    print(f"[green]Wrote report[/green] {output}")


@app.command("result-package")
def result_package_command(
    summary: Path = typer.Option(..., "--summary", help="Summary CSV for the run."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to write package files."),
    raw_input: list[Path] = typer.Option(
        [],
        "--raw-input",
        help="Raw result JSONL file or directory. Can be repeated.",
    ),
    workload: list[Path] = typer.Option(
        [],
        "--workload",
        help="Workload JSONL file. Can be repeated.",
    ),
    dataset_manifest: list[Path] = typer.Option(
        [],
        "--dataset-manifest",
        help="Dataset manifest JSON file. Can be repeated.",
    ),
    report: list[Path] = typer.Option(
        [],
        "--report",
        help="Markdown report file. Can be repeated.",
    ),
    config: list[Path] = typer.Option(
        [],
        "--config",
        help="Experiment config file to snapshot with redaction. Can be repeated.",
    ),
    artifact: list[Path] = typer.Option(
        [],
        "--artifact",
        help="Extra artifact file to copy into the package. Can be repeated.",
    ),
    run_name: str | None = typer.Option(None, "--run-name"),
    sample_rows: int = typer.Option(3, "--sample-rows", min=0),
) -> None:
    """Build a reproducible result package from completed benchmark artifacts."""
    from kvoptbench.packaging.result_package import build_result_package

    try:
        package = build_result_package(
            output_dir=output_dir,
            summary_path=summary,
            raw_input_paths=raw_input,
            workload_paths=workload,
            dataset_manifest_paths=dataset_manifest,
            report_paths=report,
            config_paths=config,
            extra_artifact_paths=artifact,
            sample_rows=sample_rows,
            run_name=run_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    print(f"[green]Wrote result package[/green] {package.output_dir}")
    print(f"[green]Wrote manifest[/green] {package.manifest_path}")


@app.command("strategy-select")
def strategy_select_command(
    input: Path = typer.Option(..., "--input", "-i"),
) -> None:
    """Run the basic strategy advisor from a summary CSV."""
    from kvoptbench.strategy.selector import select_strategy_from_summary

    print(select_strategy_from_summary(input))


@app.command("strategy-recommend")
def strategy_recommend_command(
    summary: Path = typer.Option(..., "--summary", help="Required summary CSV."),
    advisor_config: Path | None = typer.Option(None, "--advisor-config"),
    cache_input: Path | None = typer.Option(None, "--cache-input"),
    prefix_sweep_input: Path | None = typer.Option(None, "--prefix-sweep-input"),
    prefill_decode_input: Path | None = typer.Option(None, "--prefill-decode-input"),
    long_context_input: Path | None = typer.Option(None, "--long-context-input"),
    kv_quant_input: Path | None = typer.Option(None, "--kv-quant-input"),
    kv_offload_input: Path | None = typer.Option(None, "--kv-offload-input"),
    spec_decoding_input: Path | None = typer.Option(None, "--spec-decoding-input"),
    disagg_input: Path | None = typer.Option(None, "--disagg-input"),
    json_output: Path | None = typer.Option(None, "--json-output"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output"),
) -> None:
    """Generate evidence-based strategy recommendations from comparison CSVs."""
    from kvoptbench.strategy.advisor import (
        build_strategy_advisor_report,
        write_strategy_advisor_outputs,
    )

    report = build_strategy_advisor_report(
        summary_path=summary,
        advisor_config_path=advisor_config,
        cache_input_path=cache_input,
        prefix_sweep_input_path=prefix_sweep_input,
        prefill_decode_input_path=prefill_decode_input,
        long_context_input_path=long_context_input,
        kv_quant_input_path=kv_quant_input,
        kv_offload_input_path=kv_offload_input,
        spec_decoding_input_path=spec_decoding_input,
        disagg_input_path=disagg_input,
    )
    written_json, written_markdown = write_strategy_advisor_outputs(
        report=report,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    print("[bold]Strategy Advisor[/bold]")
    print(f"Overall recommendation: [cyan]{report.overall_recommendation}[/cyan]")
    if written_json is not None:
        print(f"[green]Wrote JSON[/green] {written_json}")
    if written_markdown is not None:
        print(f"[green]Wrote markdown[/green] {written_markdown}")


def _parse_context_buckets(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _print_validation_report(report: dict[str, Any]) -> None:
    status = "OK" if report.get("ok") else "FAILED"
    color = "green" if report.get("ok") else "red"
    print(
        f"[{color}]{status}[/{color}] {report.get('artifact_type')} "
        f"checked_files={report.get('checked_files')} rows={report.get('row_count')}"
    )
    for error in report.get("errors", []):
        location = error.get("file", "<unknown>")
        if error.get("line") is not None:
            location = f"{location}:{error['line']}"
        print(f"[red]- {location}: {error.get('message')}[/red]")


def _parse_csv_strings(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return value


def _collect_import_missing_metrics(rows: list[dict[str, Any]]) -> list[str]:
    missing: set[str] = set()
    for row in rows:
        for metric in row.get("missing_metrics", []):
            missing.add(str(metric))
    return sorted(missing)


def _basic_import_manifest(
    *,
    tool: str,
    source: Path,
    granularity: str,
    row_count: int,
    missing_metrics: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "tool": tool,
        "source": {"file_name": source.name},
        "granularity": granularity,
        "row_count": row_count,
        "missing_metrics": missing_metrics,
    }


if __name__ == "__main__":
    app()

