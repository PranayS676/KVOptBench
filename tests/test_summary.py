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

