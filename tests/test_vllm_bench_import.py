import csv
import json
from pathlib import Path

from kvoptbench.importers.vllm_bench import import_vllm_bench


def test_vllm_bench_import_jsonl_maps_token_and_latency_fields(tmp_path: Path) -> None:
    source = tmp_path / "vllm_bench.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "benchmark_name": "serve",
                "model": "example/model",
                "backend": "vllm",
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

    rows = import_vllm_bench(source, experiment_id="import-smoke", workload="sharegpt")

    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_id"] == "import-smoke"
    assert row["engine"] == "vllm"
    assert row["model_id"] == "example/model"
    assert row["task_id"] == "req-1"
    assert row["input_tokens"] == 128
    assert row["output_tokens"] == 32
    assert row["ttft_ms"] == 120.5
    assert row["tpot_ms"] == 8.25
    assert row["e2e_latency_ms"] == 384.0
    assert row["metadata"]["source_format"] == "jsonl"
    assert row["metadata"]["source_path"] == source.name


def test_vllm_bench_import_csv_preserves_missing_metrics(tmp_path: Path) -> None:
    source = tmp_path / "vllm_bench.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "request_id",
                "model_name",
                "input_len",
                "output_len",
                "time_to_first_token_ms",
                "success",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "request_id": "req-2",
                "model_name": "example/model",
                "input_len": "256",
                "output_len": "64",
                "time_to_first_token_ms": "140",
                "success": "true",
            }
        )

    rows = import_vllm_bench(
        source,
        experiment_id="csv-import",
        workload="synthetic",
        strategy="baseline",
    )

    row = rows[0]
    assert row["input_tokens"] == 256
    assert row["output_tokens"] == 64
    assert row["ttft_ms"] == 140.0
    assert row["tpot_ms"] is None
    assert row["e2e_latency_ms"] is None
    assert row["gpu_memory_used_gb"] is None
    assert row["gpu_memory_peak_gb"] is None
    assert row["missing_metrics"] == [
        "tpot_ms",
        "e2e_latency_ms",
        "gpu_memory_used_gb",
        "gpu_memory_peak_gb",
    ]
    assert row["metadata"]["missing_metric_reasons"]["e2e_latency_ms"].startswith(
        "No vLLM bench field"
    )
