import csv
import json
from pathlib import Path

from kvoptbench.importers.aiperf import import_aiperf
from kvoptbench.importers.genai_perf import import_genai_perf
from kvoptbench.importers.metrics import metric_aliases
from kvoptbench.importers.vllm_bench import import_vllm_bench


def test_vllm_import_emits_imported_metric_provenance(tmp_path: Path) -> None:
    source = tmp_path / "vllm.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "model": "example/model",
                "num_input_tokens": 128,
                "num_output_tokens": 32,
                "ttft_ms": 120.5,
                "tpot_ms": 8.25,
                "latency_ms": 384.0,
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = import_vllm_bench(source, experiment_id="exp", workload="sharegpt")[0]

    ttft = row["metric_provenance"]["ttft_ms"]
    assert ttft["source_type"] == "imported"
    assert ttft["source_field"] == "ttft_ms"
    assert ttft["normalized_field"] == "ttft_ms"
    assert ttft["unit"] == "ms"
    assert row["metric_provenance"]["gpu_memory_peak_gb"]["available"] is False
    assert row["metadata"]["source_path"] == source.name
    assert row["metadata"]["source_file_name"] == source.name
    assert str(tmp_path) not in json.dumps(row)


def test_metric_registry_exposes_vllm_and_nvidia_aliases() -> None:
    assert "num_input_tokens" in metric_aliases("vllm_bench", "input_tokens", "request")
    assert "time_to_first_token_ms" in metric_aliases("genai_perf", "ttft_ms", "request")
    assert "time_to_first_output_token" in metric_aliases("aiperf", "ttft_ms", "request")


def test_genai_perf_csv_metric_table_imports_aggregate_summary(tmp_path: Path) -> None:
    source = tmp_path / "genai_perf.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "Value"])
        writer.writeheader()
        writer.writerow({"Metric": "Request Count", "Value": "12"})
        writer.writerow({"Metric": "TTFT Mean (ms)", "Value": "125"})
        writer.writerow({"Metric": "Inter Token Latency Mean (ms)", "Value": "7.5"})
        writer.writerow({"Metric": "Output Token Throughput", "Value": "310.2"})
        writer.writerow({"Metric": "Request Throughput", "Value": "4.5"})

    result = import_genai_perf(
        source,
        experiment_id="genai-aggregate",
        workload="sharegpt",
        engine="vllm",
        model_id="example/model",
    )

    assert result.granularity == "aggregate"
    assert result.request_rows == []
    row = result.summary_rows[0]
    assert row["requests"] == 12
    assert row["ttft_ms_mean"] == 125.0
    assert row["itl_ms_mean"] == 7.5
    assert row["output_tokens_per_second_mean"] == 310.2
    assert row["requests_per_second_mean"] == 4.5
    assert row["missing_metrics"] == []
    assert row["metric_provenance"]["ttft_ms_mean"]["source_type"] == "imported"
    assert row["metadata"]["import_granularity"] == "aggregate"
    assert result.source_manifest["source"]["file_name"] == source.name
    assert str(tmp_path) not in json.dumps(row)


def test_genai_perf_json_request_import_preserves_request_granularity(tmp_path: Path) -> None:
    source = tmp_path / "genai_perf.json"
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

    result = import_genai_perf(source, experiment_id="genai-request", workload="chat")

    assert result.granularity == "request"
    assert result.summary_rows == []
    row = result.request_rows[0]
    assert row["task_id"] == "request-1"
    assert row["input_tokens"] == 256
    assert row["output_tokens"] == 64
    assert row["ttft_ms"] == 140.0
    assert row["itl_ms"] == 9.5
    assert row["e2e_latency_ms"] == 740.0
    assert row["missing_metrics"] == []
    assert row["metadata"]["import_granularity"] == "request"


def test_aiperf_jsonl_request_import_maps_nested_error_field(tmp_path: Path) -> None:
    source = tmp_path / "aiperf.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "aiperf-1",
                "model": "example/model",
                "input_token_count": 128,
                "output_token_count": 16,
                "time_to_first_output_token": 90,
                "inter_token_latency": 6,
                "request_latency": 220,
                "error": {"type": "timeout"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_aiperf(source, experiment_id="aiperf-request", workload="chat")

    row = result.request_rows[0]
    assert result.granularity == "request"
    assert row["ttft_ms"] == 90.0
    assert row["itl_ms"] == 6.0
    assert row["e2e_latency_ms"] == 220.0
    assert row["success"] is False
    assert row["error_type"] == "timeout"
    assert row["metric_provenance"]["error_type"]["source_field"] == "error.type"
    assert row["metric_provenance"]["ttft_ms"]["source_field"] == "time_to_first_output_token"


def test_aiperf_json_aggregate_import_preserves_summary_granularity(tmp_path: Path) -> None:
    source = tmp_path / "aiperf-summary.json"
    source.write_text(
        json.dumps(
            {
                "request_count": 20,
                "error_count": 1,
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

    result = import_aiperf(
        source,
        experiment_id="aiperf-aggregate",
        workload="chat",
        engine="vllm",
        model_id="example/model",
    )

    assert result.granularity == "aggregate"
    assert result.request_rows == []
    row = result.summary_rows[0]
    assert row["requests"] == 20
    assert row["errors"] == 1
    assert row["success_rate"] == 0.95
    assert row["ttft_ms_mean"] == 110.0
    assert row["itl_ms_mean"] == 8.0
    assert row["e2e_latency_ms_mean"] == 510.0
    assert row["requests_per_second_mean"] == 4.2
    assert row["output_tokens_per_second_mean"] == 190.5
    assert row["missing_metrics"] == []
    assert row["metadata"]["import_granularity"] == "aggregate"


def test_missing_imported_metrics_remain_none_with_reasons(tmp_path: Path) -> None:
    source = tmp_path / "aiperf-missing.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "missing-tokens",
                "time_to_first_token_ms": 80,
                "inter_token_latency_ms": 5,
                "request_latency_ms": 210,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_aiperf(
        source,
        experiment_id="aiperf-missing",
        workload="chat",
        granularity="request",
    )

    row = result.request_rows[0]
    assert row["input_tokens"] is None
    assert row["output_tokens"] is None
    assert "input_tokens" in row["missing_metrics"]
    assert "output_tokens" in row["missing_metrics"]
    assert row["metric_provenance"]["input_tokens"]["available"] is False
    assert row["metric_provenance"]["input_tokens"]["source_type"] == "imported"
    assert row["metadata"]["missing_metric_reasons"]["input_tokens"].startswith("No AIPerf field")
