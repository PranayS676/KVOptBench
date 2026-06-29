import json

from typer.testing import CliRunner

from kvoptbench.cli import app


def test_engine_command_cli_prints_preview_without_launching() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "engine-command",
            "--engine",
            "vllm",
            "--strategy",
            "cache_on",
            "--model-id",
            "example/model",
        ],
    )

    assert result.exit_code == 0
    assert "--enable-prefix-caching" in result.stdout
    assert "does not launch" in result.stdout


def test_cache_plan_cli_writes_yaml_configs(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "cache-plan",
            "--plan-dir",
            str(tmp_path / "plan"),
            "--experiment-prefix",
            "cache_exp",
            "--provider",
            "mock",
            "--engine",
            "sglang",
            "--model-id",
            "mock-frontier-model",
            "--base-url",
            "http://127.0.0.1:30000/v1",
            "--shared-workload-file",
            "workloads/generated/shared.jsonl",
            "--random-workload-file",
            "workloads/generated/random.jsonl",
            "--output-dir",
            "results/raw",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 8 cache experiment configs" in result.stdout
    assert len(list((tmp_path / "plan").glob("*.yaml"))) == 8


def test_cache_compare_cli_writes_cache_summary(tmp_path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "cache_summary.csv"
    rows = [
        _cache_row("shared_prefix_long_doc", "cold", 300.0, 10000, "shared_prefix"),
        _cache_row("shared_prefix_long_doc", "warm", 100.0, 10000, "shared_prefix"),
        _cache_row("random_prefix_control", "cold", 280.0, 0, "random_prefix"),
        _cache_row("random_prefix_control", "warm", 270.0, 0, "random_prefix"),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "cache-compare",
            "--input",
            str(raw_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote cache comparison" in result.stdout
    assert output.exists()


def test_prefix_sweep_compare_cli_writes_prefix_sweep_summary(tmp_path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "prefix_sweep.csv"
    rows = [
        _prefix_row(0.0, "cold", 100.0),
        _prefix_row(0.0, "warm", 100.0),
        _prefix_row(0.5, "cold", 260.0),
        _prefix_row(0.5, "warm", 160.0),
    ]
    (raw_dir / "prefix.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "prefix-sweep-compare",
            "--input",
            str(raw_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote prefix sweep comparison" in result.stdout
    assert output.exists()


def test_prefill_decode_plan_cli_writes_yaml_configs(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "prefill-decode-plan",
            "--plan-dir",
            str(tmp_path / "plan"),
            "--experiment-prefix",
            "prefill_decode",
            "--provider",
            "mock",
            "--engine",
            "vllm",
            "--model-id",
            "mock-frontier-model",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--workload-file",
            "workloads/generated/prefill_decode_grid.jsonl",
            "--output-dir",
            "results/raw",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 prefill/decode experiment configs" in result.stdout
    assert len(list((tmp_path / "plan").glob("*.yaml"))) == 1


def test_prefill_decode_compare_cli_writes_summary(tmp_path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "prefill_decode.csv"
    (raw_dir / "results.jsonl").write_text(
        json.dumps(_prefill_decode_row()) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "prefill-decode-compare",
            "--input",
            str(raw_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote prefill/decode comparison" in result.stdout
    assert output.exists()


def _cache_row(
    workload: str,
    cache_state: str,
    ttft_ms: float,
    shared_prefix_tokens: int,
    workload_profile: str,
) -> dict:
    return {
        "run_id": "run",
        "experiment_id": f"{workload}_{cache_state}",
        "provider": "mock",
        "engine": "vllm",
        "model_id": "mock-frontier-model",
        "strategy": "cache_on",
        "workload": workload,
        "task_id": f"{workload}_{cache_state}",
        "concurrency": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "target_input_tokens": 100,
        "target_output_tokens": 10,
        "shared_prefix_tokens": shared_prefix_tokens,
        "cache_state": cache_state,
        "ttft_ms": ttft_ms,
        "tpot_ms": 10.0,
        "itl_ms": 10.0,
        "e2e_latency_ms": ttft_ms + 100.0,
        "success": True,
        "missing_metrics": [],
        "metadata": {
            "config_metadata": {
                "cache_experiment": True,
                "workload_profile": workload_profile,
            }
        },
    }


def _prefix_row(ratio: float, cache_state: str, ttft_ms: float) -> dict:
    row = _cache_row(
        "partial_prefix_reuse",
        cache_state,
        ttft_ms,
        int(1000 * ratio),
        "shared_prefix",
    )
    row["metadata"]["workload_metadata"] = {"shared_prefix_ratio": ratio}
    return row


def _prefill_decode_row() -> dict:
    row = _cache_row(
        "prefill_decode_grid",
        "na",
        900.0,
        0,
        "prefill_decode_grid",
    )
    row["tpot_ms"] = 12.0
    row["itl_ms"] = 11.0
    row["output_tokens_per_second"] = 80.0
    row["metadata"]["config_metadata"] = {
        "prefill_decode_experiment": True,
        "workload_profile": "prefill_decode_grid",
    }
    row["metadata"]["workload_metadata"] = {
        "input_token_bucket": 32768,
        "output_token_bucket": 32,
        "expected_bottleneck": "prefill_bound",
    }
    return row
