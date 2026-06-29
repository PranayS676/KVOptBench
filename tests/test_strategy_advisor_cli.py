import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from kvoptbench.cli import app


def test_strategy_recommend_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    summary = tmp_path / "summary.csv"
    cache = tmp_path / "cache_summary.csv"
    json_output = tmp_path / "strategy.json"
    markdown_output = tmp_path / "strategy.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "workload": "shared_prefix_long_doc",
                "concurrency": 1,
                "requests": 4,
                "success_rate": 1.0,
                "quality_score_mean": 1.0,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "shared_cold_ttft_ms": 320.0,
                "shared_warm_ttft_ms": 110.0,
                "random_cache_miss_penalty_ms": 15.0,
                "control_adjusted_cache_gain_ms": 195.0,
                "shared_prefix_tokens": 12000,
                "interpretation": "credible_cache_reuse_signal",
            }
        ]
    ).to_csv(cache, index=False)

    result = runner.invoke(
        app,
        [
            "strategy-recommend",
            "--summary",
            str(summary),
            "--cache-input",
            str(cache),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
    )

    assert result.exit_code == 0
    assert "Strategy Advisor" in result.stdout
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["recommendations"][0]["strategy"] == "prefix_caching"
    assert payload["recommendations"][0]["decision"] == "recommend"
    assert "# Strategy Advisor" in markdown_output.read_text(encoding="utf-8")
