import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from kvoptbench.cli import app
from kvoptbench.config import validate_config
from kvoptbench.doctor import run_doctor
from kvoptbench.init import scaffold_project
from kvoptbench.runner.experiment import load_workload
from kvoptbench.workflow import run_config_workflow


def test_init_scaffolds_golden_qasper_mock_pack(tmp_path: Path) -> None:
    starter = tmp_path / "starter"

    result = scaffold_project(starter)

    assert result.config_path == starter / "configs" / "golden_qasper_mock.yaml"
    assert result.workload_path == starter / "workloads" / "generated" / "golden_qasper_mock.jsonl"
    assert result.dataset_manifest_path == (
        starter / "workloads" / "generated" / "golden_qasper_mock_manifest.json"
    )
    assert result.config_path.exists()
    assert result.workload_path.exists()
    assert result.dataset_manifest_path.exists()

    config = validate_config(result.config_path)
    assert config.provider == "mock"
    assert config.engine == "mock"
    assert config.endpoint_type == "mock"
    assert config.strategy == "baseline"
    assert config.workload_file == result.workload_path
    assert config.output_file == starter / "results" / "raw" / "golden_qasper_mock.jsonl"
    assert config.metadata["dataset"] == "qasper"
    assert config.metadata["dataset_manifest"] == result.dataset_manifest_path.as_posix()

    rows = load_workload(result.workload_path)
    assert len(rows) == 4
    assert {row.metadata["dataset"] for row in rows} == {"qasper"}
    assert {row.workload for row in rows} == {"qasper_shared_prefix"}
    assert rows[0].prefix_group_id == rows[1].prefix_group_id
    assert rows[0].shared_prefix_tokens > 0

    manifest = json.loads(result.dataset_manifest_path.read_text(encoding="utf-8"))
    assert manifest["adapter_name"] == "golden_qasper_mock"
    assert manifest["dataset_name"] == "QASPER"
    assert manifest["mode"] == "shared_prefix"
    assert manifest["row_count"] == 4
    assert manifest["workload_sha256"]


def test_init_cli_and_doctor_validate_golden_pack(tmp_path: Path) -> None:
    runner = CliRunner()
    starter = tmp_path / "starter"

    init_result = runner.invoke(app, ["init", "--output-dir", str(starter)])

    assert init_result.exit_code == 0, init_result.stdout
    assert "golden_qasper_mock.yaml" in init_result.stdout

    config_path = starter / "configs" / "golden_qasper_mock.yaml"
    doctor_result = runner.invoke(
        app,
        ["doctor", "--config", str(config_path), "--skip-endpoint", "--json"],
    )

    assert doctor_result.exit_code == 0, doctor_result.stdout
    payload = json.loads(doctor_result.stdout)
    assert payload["ok"] is True
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["config"]["status"] == "ok"
    assert checks["workload"]["status"] == "ok"
    assert checks["dataset_manifest"]["status"] == "ok"
    assert checks["endpoint"]["status"] == "skipped"
    assert checks["environment"]["status"] == "ok"


def test_doctor_fails_when_workload_path_is_missing(tmp_path: Path) -> None:
    scaffold = scaffold_project(tmp_path / "starter")
    config_payload = yaml.safe_load(scaffold.config_path.read_text(encoding="utf-8"))
    config_payload["workload_file"] = (tmp_path / "missing.jsonl").as_posix()
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    report = run_doctor(bad_config, check_endpoint=False)

    assert report.ok is False
    checks = {check.name: check for check in report.checks}
    assert checks["config"].status == "ok"
    assert checks["workload"].status == "fail"
    assert "does not exist" in checks["workload"].message


def test_workflow_run_chains_summary_report_advisor_and_package(
    monkeypatch, tmp_path: Path
) -> None:
    scaffold = scaffold_project(tmp_path / "starter")
    output_dir = tmp_path / "workflow-output"
    package_dir = tmp_path / "package"

    async def fake_run_experiment(config_path: str | Path) -> Path:
        config = validate_config(config_path)
        config.output_file.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "run_id": "run-golden",
            "experiment_id": config.experiment_id,
            "provider": config.provider,
            "engine": config.engine,
            "model_id": config.model_id,
            "strategy": config.strategy,
            "workload": "qasper_shared_prefix",
            "task_id": "qasper-golden-1",
            "concurrency": config.concurrency,
            "input_tokens": 256,
            "output_tokens": 16,
            "target_input_tokens": 256,
            "target_output_tokens": 32,
            "shared_prefix_tokens": 128,
            "cache_state": "warm",
            "cache_hit_rate": 0.75,
            "cache_miss_penalty_ms": 25.0,
            "ttft_ms": 45.0,
            "tpot_ms": 6.0,
            "itl_ms": 6.0,
            "e2e_latency_ms": 141.0,
            "requests_per_second": 1.0,
            "success": True,
            "quality_score": 1.0,
            "quality_method": "contains_expected",
            "missing_metrics": ["gpu_memory_peak_gb"],
            "metric_provenance": {
                "ttft_ms": {
                    "source_type": "client_observed",
                    "measurement_method": "time_to_first_stream_chunk",
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
        config.output_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return config.output_file

    monkeypatch.setattr("kvoptbench.workflow.run_experiment", fake_run_experiment)

    result = run_config_workflow(
        scaffold.config_path,
        output_dir=output_dir,
        package_dir=package_dir,
        dataset_manifest_paths=[scaffold.dataset_manifest_path],
        run_name="golden-qasper-mock",
    )

    assert result.raw_results_path == (
        scaffold.config_path.parent.parent / "results" / "raw" / "golden_qasper_mock.jsonl"
    )
    assert result.summary_path == output_dir / "summary.csv"
    assert result.report_path == output_dir / "report.md"
    assert result.strategy_json_path == output_dir / "strategy_advisor.json"
    assert result.strategy_markdown_path == output_dir / "strategy_advisor.md"
    assert result.package_manifest_path == package_dir / "run_manifest.json"
    assert result.summary_path.exists()
    assert result.report_path.exists()
    assert result.strategy_json_path.exists()
    assert result.strategy_markdown_path.exists()
    assert result.package_manifest_path.exists()

    manifest = json.loads(result.package_manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_name"] == "golden-qasper-mock"
    assert manifest["dataset_provenance"][0]["adapter_name"] == "golden_qasper_mock"


def test_workflow_cli_supports_package_output(monkeypatch, tmp_path: Path) -> None:
    scaffold = scaffold_project(tmp_path / "starter")
    output_dir = tmp_path / "workflow-output"
    package_dir = tmp_path / "package"

    async def fake_run_experiment(config_path: str | Path) -> Path:
        config = validate_config(config_path)
        config.output_file.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "run_id": "run-cli",
            "experiment_id": config.experiment_id,
            "provider": config.provider,
            "engine": config.engine,
            "model_id": config.model_id,
            "strategy": config.strategy,
            "workload": "qasper_shared_prefix",
            "task_id": "qasper-golden-cli",
            "concurrency": config.concurrency,
            "input_tokens": 256,
            "output_tokens": 16,
            "target_input_tokens": 256,
            "target_output_tokens": 32,
            "shared_prefix_tokens": 128,
            "cache_state": "warm",
            "ttft_ms": 45.0,
            "tpot_ms": 6.0,
            "e2e_latency_ms": 141.0,
            "requests_per_second": 1.0,
            "success": True,
            "quality_score": 1.0,
            "quality_method": "contains_expected",
            "missing_metrics": [],
            "metric_provenance": {},
        }
        config.output_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return config.output_file

    monkeypatch.setattr("kvoptbench.workflow.run_experiment", fake_run_experiment)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "--config",
            str(scaffold.config_path),
            "--output-dir",
            str(output_dir),
            "--package-dir",
            str(package_dir),
            "--dataset-manifest",
            str(scaffold.dataset_manifest_path),
            "--run-name",
            "cli-golden",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Wrote summary" in result.stdout
    assert "Wrote result package" in result.stdout
    assert (package_dir / "run_manifest.json").exists()
