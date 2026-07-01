"""Import NVIDIA AIPerf artifacts into KVOptBench-compatible rows."""

from __future__ import annotations

from pathlib import Path

from kvoptbench.importers.external import ImportAdapterResult, ImportGranularity
from kvoptbench.importers.external import import_external_benchmark


def import_aiperf(
    source: str | Path,
    *,
    experiment_id: str,
    workload: str,
    provider: str = "local",
    engine: str = "unknown",
    strategy: str = "imported",
    model_id: str | None = None,
    run_id: str | None = None,
    concurrency: int = 1,
    granularity: ImportGranularity = "auto",
) -> ImportAdapterResult:
    """Read a local AIPerf CSV/JSON/JSONL artifact into normalized output."""
    return import_external_benchmark(
        source,
        external_tool="aiperf",
        experiment_id=experiment_id,
        workload=workload,
        provider=provider,
        engine=engine,
        strategy=strategy,
        model_id=model_id,
        run_id=run_id,
        concurrency=concurrency,
        granularity=granularity,
    )
