import csv
import json

from typer.testing import CliRunner

from kvoptbench.cli import app


def test_cli_help_uses_project_level_positioning() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "cache-aware LLM inference benchmark" in result.stdout
    assert "local/mock benchmark harness" not in result.stdout


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


def test_telemetry_profile_cli_lists_and_exports_profiles() -> None:
    runner = CliRunner()

    list_result = runner.invoke(app, ["telemetry-profile", "list"])

    assert list_result.exit_code == 0
    assert "vllm_live" in list_result.stdout
    assert "gpu_only" in list_result.stdout

    show_result = runner.invoke(
        app,
        ["telemetry-profile", "show", "--profile", "vllm_live", "--json"],
    )

    assert show_result.exit_code == 0
    payload = json.loads(show_result.stdout)
    assert payload["name"] == "vllm_live"
    assert payload["telemetry"]["prometheus"][0]["name"] == "vllm"
    assert "engine_reported_cache_hit_rate" in payload["telemetry"]["prometheus"][0][
        "expected_metrics"
    ]


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


def test_strategy_plan_and_run_cli_write_manifests(tmp_path) -> None:
    runner = CliRunner()
    workload_pack = tmp_path / "workload_pack.yaml"
    workload_pack.write_text(
        "\n".join(
            [
                "workloads:",
                "  primary: workloads/generated/decode.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    plan_dir = tmp_path / "strategy_plan"
    run_manifest = tmp_path / "run_manifest.json"

    plan_result = runner.invoke(
        app,
        [
            "strategy-plan",
            "--plan-dir",
            str(plan_dir),
            "--matrix-id",
            "decode_matrix",
            "--provider",
            "local",
            "--engine",
            "vllm",
            "--model-id",
            "example/model",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--workload-pack",
            str(workload_pack),
            "--strategy-family",
            "decode-heavy",
            "--strategy",
            "baseline",
            "--concurrency",
            "1",
            "--output-dir",
            str(tmp_path / "raw"),
            "--repeat-count",
            "2",
            "--randomization-seed",
            "11",
        ],
    )

    assert plan_result.exit_code == 0
    assert "Wrote 1 strategy configs" in plan_result.stdout
    assert (plan_dir / "plan_manifest.json").exists()

    run_result = runner.invoke(
        app,
        [
            "strategy-run",
            "--matrix-manifest",
            str(plan_dir / "plan_manifest.json"),
            "--output-run-manifest",
            str(run_manifest),
            "--randomize",
            "--dry-run",
        ],
    )

    assert run_result.exit_code == 0
    assert "Wrote run manifest" in run_result.stdout
    assert "Dry run only" in run_result.stdout
    assert run_manifest.exists()


def test_import_cli_writes_request_jsonl_for_genai_perf(tmp_path) -> None:
    runner = CliRunner()
    source = tmp_path / "genai_perf.json"
    output = tmp_path / "imported.jsonl"
    manifest = tmp_path / "import_manifest.json"
    source.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "request_id": "request-1",
                        "model_name": "example/model",
                        "input_tokens": 256,
                        "output_tokens": 64,
                        "time_to_first_token_ms": 140,
                        "inter_token_latency_ms": 9.5,
                        "request_latency_ms": 740,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import",
            "--tool",
            "genai-perf",
            "--source",
            str(source),
            "--output",
            str(output),
            "--experiment-id",
            "genai-request",
            "--workload",
            "chat",
            "--manifest-output",
            str(manifest),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote imported request JSONL" in result.stdout
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["task_id"] == "request-1"
    assert row["metric_provenance"]["ttft_ms"]["source_type"] == "imported"
    assert json.loads(manifest.read_text(encoding="utf-8"))["source"]["file_name"] == source.name


def test_import_cli_fail_on_missing_required_ignores_optional_vllm_metrics(tmp_path) -> None:
    runner = CliRunner()
    source = tmp_path / "vllm.jsonl"
    output = tmp_path / "imported.jsonl"
    manifest = tmp_path / "manifest.json"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "model": "example/model",
                "num_input_tokens": 128,
                "num_output_tokens": 32,
                "ttft_ms": 120,
                "tpot_ms": 8,
                "latency_ms": 420,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import",
            "--tool",
            "vllm-bench",
            "--source",
            str(source),
            "--output",
            str(output),
            "--experiment-id",
            "vllm-import",
            "--workload",
            "sharegpt",
            "--manifest-output",
            str(manifest),
            "--fail-on-missing-required",
        ],
    )

    assert result.exit_code == 0
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["mapping_registry_version"] == "1"
    assert manifest_payload["source"]["sha256"]
    assert manifest_payload["required_metrics"]["request"] == [
        "input_tokens",
        "output_tokens",
    ]
    assert manifest_payload["missing_required_metrics"] == []
    assert "gpu_memory_peak_gb" in manifest_payload["missing_metrics"]


def test_import_cli_fail_on_missing_required_rejects_missing_tokens(tmp_path) -> None:
    runner = CliRunner()
    source = tmp_path / "vllm_missing.jsonl"
    output = tmp_path / "imported.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "model": "example/model",
                "num_input_tokens": 128,
                "ttft_ms": 120,
                "latency_ms": 420,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import",
            "--tool",
            "vllm-bench",
            "--source",
            str(source),
            "--output",
            str(output),
            "--experiment-id",
            "vllm-import",
            "--workload",
            "sharegpt",
            "--fail-on-missing-required",
        ],
    )

    assert result.exit_code == 1
    assert "Missing required imported metrics: output_tokens" in result.stdout


def test_import_mappings_cli_outputs_registry_metadata() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["import-mappings", "--tool", "genai-perf", "--granularity", "request", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mapping_registry_version"] == "1"
    assert payload["tool"] == "genai_perf"
    assert payload["required_metrics"] == ["input_tokens", "output_tokens"]
    assert any(mapping["normalized_field"] == "ttft_ms" for mapping in payload["mappings"])


def test_import_cli_writes_aggregate_csv_for_aiperf(tmp_path) -> None:
    runner = CliRunner()
    source = tmp_path / "aiperf.json"
    output = tmp_path / "summary.csv"
    source.write_text(
        json.dumps(
            {
                "request_count": 20,
                "success_rate_percent": 95,
                "time_to_first_token_mean_ms": 110,
                "inter_token_latency_mean_ms": 8,
                "request_latency_mean_ms": 510,
                "request_throughput": 4.2,
                "output_token_throughput": 190.5,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import",
            "--tool",
            "aiperf",
            "--source",
            str(source),
            "--output",
            str(output),
            "--experiment-id",
            "aiperf-aggregate",
            "--workload",
            "chat",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote imported summary CSV" in result.stdout
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["requests"] == "20"
    assert rows[0]["ttft_ms_mean"] == "110.0"
    assert "imported" in rows[0]["metric_provenance"]


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


def test_long_context_plan_cli_writes_yaml_configs(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "long-context-plan",
            "--plan-dir",
            str(tmp_path / "plan"),
            "--experiment-prefix",
            "long_context",
            "--provider",
            "mock",
            "--engine",
            "vllm",
            "--model-id",
            "mock-frontier-model",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--workload-file",
            "workloads/generated/long_context_pressure.jsonl",
            "--output-dir",
            "results/raw",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 long-context experiment configs" in result.stdout
    assert len(list((tmp_path / "plan").glob("*.yaml"))) == 1


def test_long_context_compare_cli_writes_summary(tmp_path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "long_context.csv"
    (raw_dir / "results.jsonl").write_text(
        json.dumps(_long_context_row()) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "long-context-compare",
            "--input",
            str(raw_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote long-context comparison" in result.stdout
    assert output.exists()


def test_kv_quant_plan_cli_writes_yaml_configs(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "kv-quant-plan",
            "--plan-dir",
            str(tmp_path / "plan"),
            "--experiment-prefix",
            "kv_quant",
            "--provider",
            "mock",
            "--engine",
            "vllm",
            "--model-id",
            "mock-frontier-model",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--workload-file",
            "workloads/generated/long_context_pressure.jsonl",
            "--output-dir",
            "results/raw",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 2 KV quantization experiment configs" in result.stdout
    assert len(list((tmp_path / "plan").glob("*.yaml"))) == 2


def test_kv_quant_compare_cli_writes_summary(tmp_path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "kv_quantization.csv"
    rows = [
        _kv_quant_row(strategy="baseline", ttft_ms=300.0, quality_score=1.0),
        _kv_quant_row(strategy="kv_fp8", ttft_ms=310.0, quality_score=0.99),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "kv-quant-compare",
            "--input",
            str(raw_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote KV quantization comparison" in result.stdout
    assert output.exists()


def test_kv_offload_plan_cli_writes_yaml_configs(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "kv-offload-plan",
            "--plan-dir",
            str(tmp_path / "plan"),
            "--experiment-prefix",
            "kv_offload",
            "--provider",
            "mock",
            "--engine",
            "vllm",
            "--model-id",
            "mock-frontier-model",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--workload-file",
            "workloads/generated/long_context_pressure.jsonl",
            "--output-dir",
            "results/raw",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 2 KV offload experiment configs" in result.stdout
    assert len(list((tmp_path / "plan").glob("*.yaml"))) == 2


def test_kv_offload_compare_cli_writes_summary(tmp_path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "kv_offload.csv"
    rows = [
        _kv_offload_row(strategy="baseline", ttft_ms=300.0, quality_score=1.0),
        _kv_offload_row(strategy="kv_offload", ttft_ms=310.0, quality_score=0.99),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "kv-offload-compare",
            "--input",
            str(raw_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote KV offload comparison" in result.stdout
    assert output.exists()


def test_spec_decoding_plan_cli_writes_yaml_configs(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "spec-decoding-plan",
            "--plan-dir",
            str(tmp_path / "plan"),
            "--experiment-prefix",
            "spec_decode",
            "--provider",
            "mock",
            "--engine",
            "vllm",
            "--model-id",
            "mock-frontier-model",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--workload-file",
            "workloads/generated/decode_heavy.jsonl",
            "--output-dir",
            "results/raw",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 2 speculative decoding experiment configs" in result.stdout
    assert len(list((tmp_path / "plan").glob("*.yaml"))) == 2


def test_spec_decoding_compare_cli_writes_summary(tmp_path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "speculative_decoding.csv"
    rows = [
        _spec_decoding_row(strategy="baseline", e2e_ms=900.0, output_tps=60.0),
        _spec_decoding_row(strategy="speculative_decoding", e2e_ms=720.0, output_tps=75.0),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "spec-decoding-compare",
            "--input",
            str(raw_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote speculative decoding comparison" in result.stdout
    assert output.exists()


def test_disagg_plan_cli_writes_yaml_configs(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "disagg-plan",
            "--plan-dir",
            str(tmp_path / "plan"),
            "--experiment-prefix",
            "disagg",
            "--provider",
            "mock",
            "--engine",
            "sglang",
            "--model-id",
            "mock-frontier-model",
            "--base-url",
            "http://127.0.0.1:30000/v1",
            "--workload-file",
            "workloads/generated/prefill_decode_grid.jsonl",
            "--output-dir",
            "results/raw",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 2 disaggregation experiment configs" in result.stdout
    assert len(list((tmp_path / "plan").glob("*.yaml"))) == 2


def test_disagg_compare_cli_writes_summary(tmp_path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "disaggregation.csv"
    rows = [
        _disagg_row(strategy="baseline", ttft_ms=900.0, tpot_ms=12.0),
        _disagg_row(strategy="prefill_decode_disaggregation", ttft_ms=700.0, tpot_ms=12.5),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "disagg-compare",
            "--input",
            str(raw_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote disaggregation comparison" in result.stdout
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


def _long_context_row() -> dict:
    row = _cache_row(
        "long_context_pressure",
        "na",
        220.0,
        0,
        "long_context_pressure",
    )
    row["input_tokens_per_second"] = 12000.0
    row["output_tokens_per_second"] = 80.0
    row["metadata"]["config_metadata"] = {
        "long_context_experiment": True,
        "workload_profile": "long_context_pressure",
    }
    row["metadata"]["workload_metadata"] = {
        "context_token_bucket": 32768,
        "pressure_level": "high",
        "expected_pressure": "prefill_latency_growth",
    }
    return row


def _kv_quant_row(*, strategy: str, ttft_ms: float, quality_score: float) -> dict:
    row = _long_context_row()
    row["experiment_id"] = f"kv_quant_vllm_{strategy}_long_context_pressure"
    row["strategy"] = strategy
    row["task_id"] = f"{strategy}_32768"
    row["ttft_ms"] = ttft_ms
    row["e2e_latency_ms"] = ttft_ms + 200.0
    row["quality_score"] = quality_score
    row["gpu_memory_peak_gb"] = None
    row["missing_metrics"] = ["gpu_memory_peak_gb"]
    row["metadata"]["config_metadata"] = {
        "kv_quantization_experiment": True,
        "workload_profile": "long_context_pressure",
    }
    return row


def _kv_offload_row(*, strategy: str, ttft_ms: float, quality_score: float) -> dict:
    row = _long_context_row()
    row["experiment_id"] = f"kv_offload_vllm_{strategy}_long_context_pressure"
    row["strategy"] = strategy
    row["task_id"] = f"{strategy}_32768"
    row["ttft_ms"] = ttft_ms
    row["e2e_latency_ms"] = ttft_ms + 200.0
    row["quality_score"] = quality_score
    row["gpu_memory_peak_gb"] = None
    row["missing_metrics"] = ["gpu_memory_peak_gb"]
    row["metadata"]["config_metadata"] = {
        "kv_offload_experiment": True,
        "workload_profile": "long_context_pressure",
    }
    return row


def _spec_decoding_row(*, strategy: str, e2e_ms: float, output_tps: float) -> dict:
    row = _cache_row(
        "decode_heavy",
        "na",
        120.0,
        0,
        "decode_heavy",
    )
    row["experiment_id"] = f"spec_decode_vllm_{strategy}_decode_heavy"
    row["strategy"] = strategy
    row["task_id"] = f"{strategy}_decode_256"
    row["target_output_tokens"] = 256
    row["output_tokens"] = 128
    row["tpot_ms"] = 8.0
    row["itl_ms"] = 8.0
    row["e2e_latency_ms"] = e2e_ms
    row["output_tokens_per_second"] = output_tps
    row["quality_score"] = 1.0
    row["metadata"]["config_metadata"] = {
        "speculative_decoding_experiment": True,
        "workload_profile": "decode_heavy",
    }
    row["metadata"]["workload_metadata"] = {
        "output_token_bucket": 256,
    }
    return row


def _disagg_row(*, strategy: str, ttft_ms: float, tpot_ms: float) -> dict:
    row = _prefill_decode_row()
    row["experiment_id"] = f"disagg_sglang_{strategy}_prefill_decode_grid"
    row["strategy"] = strategy
    row["task_id"] = f"{strategy}_32768_32"
    row["ttft_ms"] = ttft_ms
    row["tpot_ms"] = tpot_ms
    row["itl_ms"] = tpot_ms
    row["e2e_latency_ms"] = ttft_ms + 320.0
    row["output_tokens_per_second"] = 80.0
    row["quality_score"] = 1.0
    row["metadata"]["config_metadata"] = {
        "prefill_decode_disaggregation_experiment": True,
        "workload_profile": "prefill_decode_grid",
    }
    return row
