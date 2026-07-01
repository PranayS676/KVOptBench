from pathlib import Path

from kvoptbench.telemetry.nvidia_smi import normalize_gpu_metrics
from kvoptbench.telemetry.prometheus import parse_prometheus_samples


def test_prometheus_text_parser_normalizes_metric_records() -> None:
    text = """
# HELP vllm:request_success_total Number of successful requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="example/model",engine="vllm"} 7
vllm:time_to_first_token_seconds_sum{model_name="example/model"} 1.25
"""

    records = parse_prometheus_samples(text)

    assert [record.name for record in records] == [
        "vllm:request_success_total",
        "vllm:time_to_first_token_seconds_sum",
    ]
    assert records[0].value == 7.0
    assert records[0].labels == {"model_name": "example/model", "engine": "vllm"}
    assert records[0].metric_type == "counter"
    assert records[0].source_type == "prometheus_text"
    assert records[1].value == 1.25


def test_prometheus_parser_accepts_jsonish_samples_from_file(tmp_path: Path) -> None:
    sample_path = tmp_path / "prometheus.json"
    sample_path.write_text(
        """
{
  "data": {
    "result": [
      {
        "metric": {"__name__": "vllm:num_requests_running", "model_name": "m"},
        "value": [1710000000.0, "3"]
      }
    ]
  }
}
""",
        encoding="utf-8",
    )

    records = parse_prometheus_samples(sample_path)

    assert len(records) == 1
    assert records[0].name == "vllm:num_requests_running"
    assert records[0].value == 3.0
    assert records[0].timestamp == 1710000000.0
    assert records[0].labels == {"model_name": "m"}
    assert records[0].source_path == sample_path.name


def test_gpu_metric_normalization_from_csv_reports_memory_values() -> None:
    csv_text = """memory.used [MiB], memory.total [MiB]
10240 MiB, 24576 MiB
12288 MiB, 24576 MiB
"""

    snapshot = normalize_gpu_metrics(csv_text)

    assert snapshot.metrics["gpu_memory_used_gb"] == 12.0
    assert snapshot.metrics["gpu_memory_peak_gb"] == 12.0
    assert snapshot.missing_metrics == []
    assert snapshot.source_type == "nvidia_smi_csv"


def test_gpu_metric_normalization_preserves_missing_reasons() -> None:
    snapshot = normalize_gpu_metrics({"driver_version": "550.54"})

    assert snapshot.metrics == {
        "gpu_memory_used_gb": None,
        "gpu_memory_peak_gb": None,
    }
    assert [missing.metric for missing in snapshot.missing_metrics] == [
        "gpu_memory_used_gb",
        "gpu_memory_peak_gb",
    ]
    assert "not present" in snapshot.missing_metrics[0].reason.lower()
