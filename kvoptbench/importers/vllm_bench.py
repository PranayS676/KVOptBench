"""Import vLLM bench artifacts into KVOptBench-like rows."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


TOKEN_INPUT_ALIASES = [
    "input_tokens",
    "num_input_tokens",
    "prompt_tokens",
    "input_len",
    "prompt_len",
]
TOKEN_OUTPUT_ALIASES = [
    "output_tokens",
    "num_output_tokens",
    "completion_tokens",
    "output_len",
    "generated_tokens",
]
TTFT_ALIASES = [
    "ttft_ms",
    "time_to_first_token_ms",
    "mean_ttft_ms",
    "median_ttft_ms",
]
TPOT_ALIASES = [
    "tpot_ms",
    "time_per_output_token_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
]
LATENCY_ALIASES = [
    "e2e_latency_ms",
    "latency_ms",
    "request_latency_ms",
    "end_to_end_latency_ms",
    "mean_latency_ms",
]
GPU_USED_ALIASES = ["gpu_memory_used_gb", "memory_used_gb", "gpu_memory_used"]
GPU_PEAK_ALIASES = ["gpu_memory_peak_gb", "peak_gpu_memory_gb", "gpu_memory_peak"]
MISSING_METRIC_ORDER = [
    "ttft_ms",
    "tpot_ms",
    "e2e_latency_ms",
    "gpu_memory_used_gb",
    "gpu_memory_peak_gb",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VllmBenchImportRow(BaseModel):
    """A KVOptBench-like imported row that can feed summary/report tooling later."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    experiment_id: str
    provider: str = "local"
    engine: str = "vllm"
    model_id: str
    strategy: str = "imported"
    workload: str
    task_id: str
    concurrency: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_latency_ms: float | None = None
    requests_per_second: float | None = None
    input_tokens_per_second: float | None = None
    output_tokens_per_second: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_peak_gb: float | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    created_at: str = Field(default_factory=_utc_now_iso)
    missing_metrics: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def import_vllm_bench(
    source: str | Path,
    *,
    experiment_id: str,
    workload: str,
    provider: str = "local",
    engine: str = "vllm",
    strategy: str = "imported",
    model_id: str | None = None,
    run_id: str | None = None,
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    """Read a local vLLM bench JSON/JSONL/CSV artifact into KVOptBench-like rows."""
    source_path = Path(source)
    raw_rows, source_format = _read_source_rows(source_path)
    imported: list[dict[str, Any]] = []

    for index, raw_row in enumerate(raw_rows):
        row = _normalize_row(
            raw_row,
            row_index=index,
            source_path=source_path,
            source_format=source_format,
            experiment_id=experiment_id,
            workload=workload,
            provider=provider,
            engine=engine,
            strategy=strategy,
            model_id=model_id,
            run_id=run_id,
            concurrency=concurrency,
        )
        imported.append(row.model_dump(mode="json"))

    return imported


def _read_source_rows(source_path: Path) -> tuple[list[dict[str, Any]], str]:
    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(source_path), "jsonl"
    if suffix == ".csv":
        return _read_csv(source_path), "csv"
    if suffix == ".json":
        return _read_json(source_path), "json"
    raise ValueError(f"Unsupported vLLM bench import format: {source_path.suffix}")


def _read_jsonl(source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _read_json(source_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ["requests", "results", "rows", "samples"]:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def _read_csv(source_path: Path) -> list[dict[str, Any]]:
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _normalize_row(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    source_path: Path,
    source_format: str,
    experiment_id: str,
    workload: str,
    provider: str,
    engine: str,
    strategy: str,
    model_id: str | None,
    run_id: str | None,
    concurrency: int,
) -> VllmBenchImportRow:
    missing_reasons: dict[str, str] = {}
    normalized = _normalize_keys(raw_row)
    resolved_model = model_id or _first_string(normalized, ["model_id", "model", "model_name"]) or "unknown"
    task_id = _first_string(normalized, ["task_id", "request_id", "id"]) or f"vllm-bench-{row_index + 1}"

    values = {
        "ttft_ms": _first_float(normalized, TTFT_ALIASES),
        "tpot_ms": _first_float(normalized, TPOT_ALIASES),
        "e2e_latency_ms": _first_float(normalized, LATENCY_ALIASES),
        "gpu_memory_used_gb": _first_float(normalized, GPU_USED_ALIASES),
        "gpu_memory_peak_gb": _first_float(normalized, GPU_PEAK_ALIASES),
    }

    missing_metrics: list[str] = []
    for metric in MISSING_METRIC_ORDER:
        if values[metric] is None:
            missing_metrics.append(metric)
            missing_reasons[metric] = (
                f"No vLLM bench field matched aliases: {', '.join(_aliases_for(metric))}."
            )

    error_message = _first_string(normalized, ["error_message", "error", "exception"])
    success = _first_bool(normalized, ["success", "succeeded", "completed"])
    if success is None:
        success = error_message is None

    return VllmBenchImportRow(
        run_id=run_id or "vllm-bench-import",
        experiment_id=experiment_id,
        provider=provider,
        engine=engine,
        model_id=resolved_model,
        strategy=strategy,
        workload=workload,
        task_id=task_id,
        concurrency=concurrency,
        input_tokens=_first_int(normalized, TOKEN_INPUT_ALIASES) or 0,
        output_tokens=_first_int(normalized, TOKEN_OUTPUT_ALIASES) or 0,
        ttft_ms=values["ttft_ms"],
        tpot_ms=values["tpot_ms"],
        e2e_latency_ms=values["e2e_latency_ms"],
        requests_per_second=_first_float(normalized, ["requests_per_second", "request_throughput"]),
        input_tokens_per_second=_first_float(
            normalized, ["input_tokens_per_second", "input_throughput"]
        ),
        output_tokens_per_second=_first_float(
            normalized, ["output_tokens_per_second", "output_throughput"]
        ),
        gpu_memory_used_gb=values["gpu_memory_used_gb"],
        gpu_memory_peak_gb=values["gpu_memory_peak_gb"],
        success=success,
        error_type=_first_string(normalized, ["error_type"]),
        error_message=error_message,
        missing_metrics=missing_metrics,
        metadata={
            "source": "vllm_bench",
            "source_path": source_path.name,
            "source_format": source_format,
            "source_row_index": row_index,
            "missing_metric_reasons": missing_reasons,
            "raw_fields": sorted(str(key) for key in raw_row),
        },
    )


def _normalize_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {_normalize_key(key): value for key, value in row.items()}


def _normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _first_string(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(_normalize_key(key))
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_int(row: dict[str, Any], keys: list[str]) -> int | None:
    value = _first_float(row, keys)
    return int(value) if value is not None else None


def _first_float(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(_normalize_key(key))
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_bool(row: dict[str, Any], keys: list[str]) -> bool | None:
    for key in keys:
        value = row.get(_normalize_key(key))
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aliases_for(metric: str) -> list[str]:
    return {
        "ttft_ms": TTFT_ALIASES,
        "tpot_ms": TPOT_ALIASES,
        "e2e_latency_ms": LATENCY_ALIASES,
        "gpu_memory_used_gb": GPU_USED_ALIASES,
        "gpu_memory_peak_gb": GPU_PEAK_ALIASES,
    }[metric]
