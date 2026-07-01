import csv
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from kvoptbench.cli import app
from kvoptbench.datasets.hashing import sha256_file
from kvoptbench.packaging.result_package import build_result_package


def test_result_package_collects_artifacts_and_provenance(tmp_path: Path) -> None:
    inputs = _write_package_inputs(tmp_path)
    output_dir = tmp_path / "package"

    package = build_result_package(
        output_dir=output_dir,
        summary_path=inputs["summary"],
        raw_input_paths=[inputs["raw_dir"]],
        workload_paths=[inputs["workload"]],
        dataset_manifest_paths=[inputs["dataset_manifest"]],
        report_paths=[inputs["report"]],
        config_paths=[inputs["config"]],
        sample_rows=1,
        run_name="qasper-cache-smoke",
    )

    manifest_path = output_dir / "run_manifest.json"
    missing_metrics_path = output_dir / "missing_metrics.json"
    metric_provenance_path = output_dir / "metric_provenance.json"
    readme_path = output_dir / "README_result.md"
    raw_sample_path = output_dir / "samples" / "raw_results_sample.jsonl"
    workload_sample_path = output_dir / "samples" / "qasper_shared_sample.jsonl"
    redacted_config_path = output_dir / "configs" / "experiment.redacted.yaml"

    assert package.manifest_path == manifest_path
    assert manifest_path.exists()
    assert missing_metrics_path.exists()
    assert metric_provenance_path.exists()
    assert readme_path.exists()
    assert raw_sample_path.exists()
    assert workload_sample_path.exists()
    assert redacted_config_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_name"] == "qasper-cache-smoke"
    assert manifest["summary"]["providers"] == ["local"]
    assert manifest["summary"]["engines"] == ["vllm"]
    assert manifest["summary"]["workloads"] == ["qasper_shared_prefix"]
    assert manifest["artifact_count"] == len(manifest["artifacts"])
    assert manifest["workload_provenance"][0]["sha256"] == sha256_file(inputs["workload"])
    assert manifest["dataset_provenance"][0]["adapter_name"] == "qasper"
    assert manifest["dataset_provenance"][0]["dataset_source_url"].startswith(
        "https://huggingface.co/datasets/allenai/qasper"
    )

    manifest_text = json.dumps(manifest, sort_keys=True)
    assert str(tmp_path) not in manifest_text
    assert "cache_hit_rate" in missing_metrics_path.read_text(encoding="utf-8")
    assert "gpu_memory_peak_gb" in missing_metrics_path.read_text(encoding="utf-8")
    metric_provenance = json.loads(metric_provenance_path.read_text(encoding="utf-8"))
    assert metric_provenance["metrics"]["ttft_ms"]["source_types"] == ["client_observed"]
    assert (
        metric_provenance["metrics"]["gpu_memory_peak_gb"]["unavailable_reasons"]
        == ["GPU telemetry was not collected for this run."]
    )
    assert manifest["metric_provenance"]["ttft_ms"]["source_types"] == ["client_observed"]
    assert "Do not publish mock metrics as real endpoint results" in readme_path.read_text(
        encoding="utf-8"
    )
    assert "metric_provenance.json" in readme_path.read_text(encoding="utf-8")

    redacted_config = yaml.safe_load(redacted_config_path.read_text(encoding="utf-8"))
    assert redacted_config["base_url"] == "<redacted_url>"
    assert redacted_config["api_key_env"] == "<redacted>"
    assert redacted_config["tokenizer_id"] == "example/tokenizer"
    assert redacted_config["metadata"]["api_key"] == "<redacted>"


def test_result_package_cli_writes_manifest(tmp_path: Path) -> None:
    inputs = _write_package_inputs(tmp_path)
    output_dir = tmp_path / "cli-package"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "result-package",
            "--summary",
            str(inputs["summary"]),
            "--raw-input",
            str(inputs["raw_dir"]),
            "--workload",
            str(inputs["workload"]),
            "--dataset-manifest",
            str(inputs["dataset_manifest"]),
            "--report",
            str(inputs["report"]),
            "--config",
            str(inputs["config"]),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "cli-smoke",
            "--sample-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Wrote result package" in result.stdout
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_name"] == "cli-smoke"
    assert manifest["missing_metrics"][0]["metric"] == "cache_hit_rate"


def _write_package_inputs(tmp_path: Path) -> dict[str, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_rows = [
        {
            "run_id": "run-1",
            "experiment_id": "qasper_cache",
            "provider": "local",
            "engine": "vllm",
            "model_id": "example/model",
            "strategy": "cache_on",
            "workload": "qasper_shared_prefix",
            "task_id": "task-1",
            "concurrency": 1,
            "input_tokens": 1024,
            "output_tokens": 32,
            "target_input_tokens": 1024,
            "target_output_tokens": 32,
            "shared_prefix_tokens": 768,
            "cache_state": "warm",
            "ttft_ms": 120.0,
            "tpot_ms": 8.0,
            "e2e_latency_ms": 420.0,
            "success": True,
            "missing_metrics": ["cache_hit_rate", "gpu_memory_peak_gb"],
            "metric_provenance": {
                "ttft_ms": {
                    "source_type": "client_observed",
                    "measurement_method": "stream timing",
                    "unit": "ms",
                    "available": True,
                },
                "gpu_memory_peak_gb": {
                    "source_type": "gpu_reported",
                    "measurement_method": "GPU telemetry adapter",
                    "unit": "GB",
                    "available": False,
                    "missing_reason": "GPU telemetry was not collected for this run.",
                },
            },
        }
    ]
    raw_file = raw_dir / "qasper_cache.jsonl"
    raw_file.write_text("\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8")

    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "provider",
                "engine",
                "model_id",
                "strategy",
                "workload",
                "concurrency",
                "requests",
                "successes",
                "success_rate",
                "missing_metrics",
                "metric_source_types",
                "unavailable_metric_reasons",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "qasper_cache",
                "provider": "local",
                "engine": "vllm",
                "model_id": "example/model",
                "strategy": "cache_on",
                "workload": "qasper_shared_prefix",
                "concurrency": "1",
                "requests": "1",
                "successes": "1",
                "success_rate": "1.0",
                "missing_metrics": "cache_hit_rate;gpu_memory_peak_gb",
                "metric_source_types": "ttft_ms:client_observed;gpu_memory_peak_gb:gpu_reported",
                "unavailable_metric_reasons": (
                    "gpu_memory_peak_gb:GPU telemetry was not collected for this run."
                ),
            }
        )

    workload = tmp_path / "qasper_shared.jsonl"
    workload.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "workload": "qasper_shared_prefix",
                "category": "qa",
                "prompt": "Document prefix\nQuestion?",
                "expected_answer": "answer",
                "target_input_tokens": 1024,
                "target_output_tokens": 32,
                "shared_prefix_tokens": 768,
                "eval_type": "contains_expected",
                "metadata": {"dataset": "qasper"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_manifest = tmp_path / "qasper_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "adapter_name": "qasper",
                "adapter_version": "0.1.0",
                "dataset_name": "QASPER",
                "dataset_source_url": "https://huggingface.co/datasets/allenai/qasper",
                "source_url": "https://huggingface.co/datasets/allenai/qasper",
                "dataset_revision": "abc123",
                "license": "cc-by-4.0",
                "license_review_status": "public_dataset_reviewed",
                "redistribution_policy": "manifest_only",
                "mode": "shared_prefix",
                "row_count": 1,
                "workload_sha256": sha256_file(workload),
                "prompt_template_hash": "template-hash",
                "token_count_method": "char_approx_4",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = tmp_path / "report.md"
    report.write_text("# Report\n\nTTFT looked stable.\n", encoding="utf-8")

    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment_id: qasper_cache",
                "provider: local",
                "engine: vllm",
                "model_id: example/model",
                "strategy: cache_on",
                "base_url: http://private-endpoint.example/v1",
                "api_key_env: PRIVATE_TOKEN",
                "tokenizer_id: example/tokenizer",
                "metadata:",
                "  api_key: should-not-leak",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "raw_dir": raw_dir,
        "summary": summary,
        "workload": workload,
        "dataset_manifest": dataset_manifest,
        "report": report,
        "config": config,
    }
