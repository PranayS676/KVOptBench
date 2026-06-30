"""Top-level Typer CLI for KVOptBench."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from kvoptbench.config import ConfigError, validate_config

app = typer.Typer(help="KVOptBench cache-aware LLM inference benchmark.")
dataset_app = typer.Typer(help="Prepare public dataset workloads.")
app.add_typer(dataset_app, name="dataset")


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


def _parse_csv_strings(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(value.strip() for value in raw.split(",") if value.strip())


if __name__ == "__main__":
    app()

