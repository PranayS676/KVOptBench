from pathlib import Path


def test_ci_workflow_covers_quality_release_and_packaging_checks() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    required_fragments = [
        "workflow_dispatch:",
        "permissions:",
        "contents: read",
        "concurrency:",
        "cancel-in-progress: true",
        'python-version: ["3.11", "3.12", "3.13"]',
        "python -m ruff check .",
        "kvoptbench schema export --output-dir schemas/v1 --check",
        "kvoptbench release-check",
        "python -m kvoptbench version",
        "kvoptbench telemetry-profile list",
        "kvoptbench import-mappings --tool genai-perf --granularity request --json",
        "KVOPTBENCH_DATASET_DOWNLOAD: \"0\"",
        "python -m pytest -q",
        "python -m build",
        "python -m pip install dist/*.whl",
    ]

    for fragment in required_fragments:
        assert fragment in workflow
