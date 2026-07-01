from pathlib import Path

import httpx

from kvoptbench.runner.environment import capture_backend_environment
from kvoptbench.telemetry.nvidia_smi import GpuSampler, normalize_gpu_metrics
from kvoptbench.telemetry.prometheus import parse_prometheus_samples, scrape_prometheus_endpoint


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


def test_prometheus_scrape_uses_fake_http_and_reports_missing_expected_metric() -> None:
    client = _FakePrometheusClient(
        """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 3
vllm:prefix_cache_hit_rate{model_name="m"} 0.75
"""
    )

    result = scrape_prometheus_endpoint(
        "http://metrics.local/metrics",
        source_name="engine",
        timeout_seconds=0.5,
        expected_metrics=[
            "vllm:num_requests_running",
            "engine_reported_cache_hit_rate",
            "queue_time_ms",
        ],
        metric_aliases={"vllm:prefix_cache_hit_rate": "engine_reported_cache_hit_rate"},
        client=client,
    )

    assert client.calls == [("http://metrics.local/metrics", 0.5)]
    assert result.success is True
    assert [record.name for record in result.records] == [
        "vllm:num_requests_running",
        "engine_reported_cache_hit_rate",
    ]
    assert result.records[1].raw_name == "vllm:prefix_cache_hit_rate"
    assert [missing.metric for missing in result.missing_metrics] == ["queue_time_ms"]
    assert "not present" in result.missing_metrics[0].reason


def test_prometheus_scrape_timeout_keeps_expected_metrics_unavailable() -> None:
    result = scrape_prometheus_endpoint(
        "http://metrics.local/metrics",
        source_name="engine",
        expected_metrics=["engine_reported_cache_hit_rate"],
        client=_TimeoutPrometheusClient(),
    )

    assert result.success is False
    assert result.records == []
    assert result.error_type == "timeout"
    assert [missing.metric for missing in result.missing_metrics] == [
        "engine_reported_cache_hit_rate"
    ]
    assert "timed out" in result.missing_metrics[0].reason


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


def test_gpu_sampler_summarizes_fake_run_window_samples() -> None:
    samples = iter(
        [
            {"gpu_index": "0", "memory.used [MiB]": "1024 MiB"},
            {"gpu_index": "0", "memory.used [MiB]": "2048 MiB"},
        ]
    )
    sampler = GpuSampler(sample_provider=lambda: next(samples), sample_interval_seconds=1)

    sampler.start()
    assert sampler.sample_once().metrics["gpu_memory_used_gb"] == 1.0
    assert sampler.sample_once().metrics["gpu_memory_used_gb"] == 2.0
    snapshot = sampler.stop()

    assert snapshot.metrics == {
        "gpu_memory_used_gb": 2.0,
        "gpu_memory_peak_gb": 2.0,
    }
    assert snapshot.missing_metrics == []
    assert snapshot.samples == [
        {"gpu_index": "0", "memory.used [MiB]": "1024 MiB"},
        {"gpu_index": "0", "memory.used [MiB]": "2048 MiB"},
    ]
    assert snapshot.metadata["sample_interval_seconds"] == 1
    assert snapshot.metadata["sample_started_at"].endswith("Z")
    assert snapshot.metadata["sample_finished_at"].endswith("Z")


def test_gpu_sampler_preserves_unavailable_reasons_from_provider_failure() -> None:
    sampler = GpuSampler(sample_provider=lambda: None)

    sampler.start()
    snapshot = sampler.sample_once()
    final_snapshot = sampler.stop()

    assert snapshot.metrics == {
        "gpu_memory_used_gb": None,
        "gpu_memory_peak_gb": None,
    }
    assert [missing.metric for missing in snapshot.missing_metrics] == [
        "gpu_memory_used_gb",
        "gpu_memory_peak_gb",
    ]
    assert "returned no sample" in snapshot.missing_metrics[0].reason
    assert "No GPU telemetry samples" in final_snapshot.missing_metrics[-1].reason


def test_backend_environment_helper_sanitizes_and_reports_missing_fields() -> None:
    captured = capture_backend_environment(
        {
            "engine_version": " 0.6.4 ",
            "gpu_count": "2",
            "backend_launch_command": (
                "vllm serve example/model --api-key secret --hf-token=hf_secret"
            ),
        },
        expected_fields=[
            "engine_version",
            "model_revision",
            "gpu_count",
            "backend_launch_command",
        ],
    )

    assert captured["engine_version"] == "0.6.4"
    assert captured["gpu_count"] == 2
    assert captured["backend_launch_command"] == (
        "vllm serve example/model --api-key <redacted> --hf-token=<redacted>"
    )
    assert "secret" not in captured["backend_launch_command"]
    assert captured["missing_environment_fields"] == [
        {
            "field": "model_revision",
            "reason": "Model revision was not provided or exposed.",
        }
    ]


class _FakePrometheusClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, float]] = []

    def get(self, url: str, *, timeout: float) -> "_FakePrometheusResponse":
        self.calls.append((url, timeout))
        return _FakePrometheusResponse(self.text)


class _FakePrometheusResponse:
    status_code = 200

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _TimeoutPrometheusClient:
    def get(self, url: str, *, timeout: float) -> "_FakePrometheusResponse":
        request = httpx.Request("GET", url)
        raise httpx.TimeoutException("read timed out", request=request)
