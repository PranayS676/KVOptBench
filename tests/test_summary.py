import json
from pathlib import Path

import pandas as pd

from kvoptbench.analysis.summarize import summarize_results
from kvoptbench.schemas import RequestResult


def test_summary_aggregation_creates_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "summary.csv"
    rows = [
        RequestResult(
            run_id="run-1",
            experiment_id="exp",
            provider="mock",
            engine="mock",
            model_id="model",
            strategy="baseline",
            workload="shared_prefix_long_doc",
            task_id=f"task-{idx}",
            concurrency=1,
            input_tokens=100,
            output_tokens=10,
            ttft_ms=100 + idx,
            tpot_ms=5,
            e2e_latency_ms=200 + idx,
            success=True,
            quality_score=1.0,
        ).model_dump()
        for idx in range(3)
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    summarize_results(input_path=raw_dir, output_path=output)

    summary = pd.read_csv(output)
    assert len(summary) == 1
    assert summary.loc[0, "requests"] == 3
    assert summary.loc[0, "success_rate"] == 1.0
    assert summary.loc[0, "ttft_ms_p95"] >= 100


def test_summary_aggregation_includes_reasoning_and_tool_call_columns(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "summary.csv"
    rows = [
        RequestResult(
            run_id="run-1",
            experiment_id="exp",
            provider="local",
            engine="openai_compatible",
            model_id="reasoning-model",
            strategy="baseline",
            workload="tool_calling",
            task_id=f"task-{idx}",
            concurrency=1,
            input_tokens=100,
            output_tokens=0,
            provider_completion_tokens=8,
            reasoning_content_present=True,
            reasoning_tokens=8,
            first_reasoning_token_ms=100 + idx,
            visible_answer_missing=True,
            tool_call_count=idx,
            ttft_ms=None,
            e2e_latency_ms=200 + idx,
            success=True,
            quality_score=0.0,
        ).model_dump()
        for idx in range(2)
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    summarize_results(input_path=raw_dir, output_path=output)

    summary = pd.read_csv(output)
    assert summary.loc[0, "reasoning_content_present_rate"] == 1.0
    assert summary.loc[0, "visible_answer_missing_rate"] == 1.0
    assert summary.loc[0, "reasoning_tokens_mean"] == 8.0
    assert summary.loc[0, "first_reasoning_token_ms_p50"] >= 100
    assert summary.loc[0, "tool_call_count_mean"] == 0.5

