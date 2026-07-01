import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from kvoptbench.cli import app
from kvoptbench.contracts import (
    build_schema_bundle,
    validate_result_package,
    validate_result_rows,
    write_schema_files,
)
from kvoptbench.packaging.result_package import build_result_package
from kvoptbench.schemas import RequestResult


def test_schema_registry_exports_expected_contracts(tmp_path: Path) -> None:
    output_dir = tmp_path / "schemas"

    written = write_schema_files(output_dir)

    assert set(written) == {
        "request_result",
        "telemetry_run_summary",
        "strategy_advisor",
        "result_package_manifest",
        "dataset_manifest",
    }
    for name, path in written.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["x-kvoptbench-contract"] == name
        assert payload["x-kvoptbench-schema-version"] == "1"
        assert payload["title"]


def test_committed_schema_snapshots_match_current_models() -> None:
    expected_dir = Path("schemas") / "v1"
    bundle = build_schema_bundle()

    for contract in bundle.contracts:
        path = expected_dir / contract.file_name
        assert path.exists(), f"Missing committed schema snapshot: {path}"
        committed = json.loads(path.read_text(encoding="utf-8"))
        assert committed == contract.json_schema


def test_validate_result_rows_accepts_current_schema_and_rejects_future_schema(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.jsonl"
    bad_path = tmp_path / "bad.jsonl"
    valid_row = RequestResult(
        run_id="run-1",
        experiment_id="schema-test",
        provider="mock",
        engine="mock",
        model_id="mock-model",
        strategy="baseline",
        workload="qasper_shared_prefix",
        task_id="task-1",
        concurrency=1,
    ).model_dump(mode="json")
    valid_path.write_text(json.dumps(valid_row) + "\n", encoding="utf-8")
    bad_row = {**valid_row, "schema_version": "999"}
    bad_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    valid_report = validate_result_rows(valid_path)
    bad_report = validate_result_rows(bad_path)

    assert valid_report.ok is True
    assert valid_report.row_count == 1
    assert bad_report.ok is False
    assert "Unsupported schema_version" in bad_report.errors[0]["message"]


def test_validate_results_cli_reports_schema_errors(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "run_id": "run-1",
                "experiment_id": "schema-test",
                "provider": "mock",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["validate-results", "--input", str(bad_path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["artifact_type"] == "request_results"
    assert payload["errors"][0]["file"].endswith("bad.jsonl")


def test_result_package_validation_is_independent_of_original_paths(tmp_path: Path) -> None:
    package_dir = _write_minimal_package(tmp_path)

    report = validate_result_package(package_dir)

    assert report.ok is True
    assert report.artifact_type == "result_package"
    assert report.checked_files >= 1


def test_validate_package_cli_checks_manifest_and_artifact_hashes(tmp_path: Path) -> None:
    package_dir = _write_minimal_package(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["validate-package", "--path", str(package_dir), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["artifact_type"] == "result_package"


def test_schema_export_cli_can_check_committed_snapshots() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["schema", "export", "--output-dir", "schemas/v1", "--check"])

    assert result.exit_code == 0, result.stdout
    assert "Schema snapshots are current" in result.stdout


def test_schema_version_migration_policy_is_documented() -> None:
    policy = (Path("schemas") / "README.md").read_text(encoding="utf-8")

    assert "Current contract version: `1`" in policy
    assert "Breaking changes require a new `schemas/vN/` directory" in policy
    assert "Validators must reject unknown future `schema_version` values" in policy


def _write_minimal_package(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_file = raw_dir / "results.jsonl"
    row = RequestResult(
        run_id="run-1",
        experiment_id="schema-test",
        provider="mock",
        engine="mock",
        model_id="mock-model",
        strategy="baseline",
        workload="qasper_shared_prefix",
        task_id="task-1",
        concurrency=1,
        success=True,
    ).model_dump(mode="json")
    raw_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "schema-test",
                "provider": "mock",
                "engine": "mock",
                "model_id": "mock-model",
                "strategy": "baseline",
                "workload": "qasper_shared_prefix",
                "concurrency": "1",
                "requests": "1",
                "successes": "1",
                "success_rate": "1.0",
                "missing_metrics": "",
            }
        )

    package_dir = tmp_path / "package"
    build_result_package(
        output_dir=package_dir,
        summary_path=summary,
        raw_input_paths=[raw_dir],
        run_name="schema-test",
    )
    return package_dir
