"""Shared pydantic schemas for config, workloads, and result rows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with a stable timezone marker."""
    return datetime.now(timezone.utc).isoformat()


class ExperimentConfig(BaseModel):
    """YAML-driven experiment configuration."""

    model_config = ConfigDict(extra="allow")

    experiment_id: str
    official_run: bool = False
    provider: str
    engine: str
    model_id: str
    strategy: str
    base_url: str
    endpoint_type: Literal["mock", "openai_compatible", "vllm", "sglang"] = "mock"
    healthcheck_path: str = "/v1/models"
    api_key_env: str | None = None
    workload_file: Path
    output_file: Path
    concurrency: int = Field(default=1, ge=1)
    request_rate: float | None = Field(default=None, gt=0)
    max_tasks: int | None = Field(default=None, ge=1)
    max_output_tokens: int = Field(default=256, ge=1)
    timeout_seconds: float = Field(default=120, gt=0)
    request_timeout_seconds: float | None = Field(default=None, gt=0)
    retries: int = Field(default=0, ge=0)
    retry_backoff_seconds: float = Field(default=0.25, ge=0)
    capture_response_headers: bool = False
    stream: bool = True
    endpoint_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id", "provider", "engine", "model_id", "strategy", "base_url")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


class WorkloadItem(BaseModel):
    """One generated benchmark task."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    workload: str
    category: str
    prompt: str
    expected_answer: str | None = None
    expected_schema: dict[str, Any] | None = None
    target_input_tokens: int = Field(default=0, ge=0)
    target_output_tokens: int = Field(default=0, ge=0)
    prefix_group_id: str | None = None
    shared_prefix_tokens: int = Field(default=0, ge=0)
    eval_type: str = "contains_expected"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "workload", "category", "prompt", "eval_type")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class QualityResult(BaseModel):
    """Structured evaluator output."""

    quality_score: float | None
    quality_method: str
    passed: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TimedResponse(BaseModel):
    """OpenAI-compatible client response with timing measurements."""

    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    token_count_method: str = "char_approx_4"
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: float | None = None
    e2e_latency_ms: float | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class EndpointHealth(BaseModel):
    """Health and metadata captured from an OpenAI-compatible endpoint."""

    ok: bool
    url: str
    status_code: int | None = None
    error_message: str | None = None
    model_ids: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineEndpoint(BaseModel):
    """OpenAI-compatible endpoint details implied by an engine command preview."""

    base_url: str
    healthcheck_path: str = "/v1/models"


class EngineStrategyProfile(BaseModel):
    """One engine strategy and its server-command preview metadata."""

    name: str
    description: str
    command_template: str
    notes: list[str] = Field(default_factory=list)
    placeholder: bool = False


class EngineProfile(BaseModel):
    """Config-driven engine profile used outside the generic runner."""

    engine: str
    display_name: str
    default_port: int
    strategies: dict[str, EngineStrategyProfile]
    notes: list[str] = Field(default_factory=list)


class EngineCommandPreview(BaseModel):
    """Rendered command preview. It is documentation, not process management."""

    engine: str
    strategy: str
    description: str
    command: str
    endpoint: EngineEndpoint
    launches_server: bool = False
    notes: str


class CacheExperimentCase(BaseModel):
    """One generated cache experiment config and its role in the ablation matrix."""

    strategy: str
    workload_profile: Literal["shared_prefix", "random_prefix"]
    cache_pass: Literal["cold", "warm"]
    is_control: bool = False
    config: ExperimentConfig


class RequestResult(BaseModel):
    """Required request-level JSONL result row."""

    run_id: str
    experiment_id: str
    official_run: bool = False
    provider: str
    gpu_type: str | None = None
    gpu_count: int | None = None
    engine: str
    engine_version: str | None = None
    model_id: str
    strategy: str
    workload: str
    task_id: str
    concurrency: int
    request_rate: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    target_input_tokens: int = 0
    target_output_tokens: int = 0
    shared_prefix_tokens: int = 0
    cache_state: Literal["cold", "warm", "na"] = "na"
    cache_hit_rate: float | None = None
    cache_hit_proxy: float | None = None
    cache_miss_penalty_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: float | None = None
    e2e_latency_ms: float | None = None
    requests_per_second: float | None = None
    input_tokens_per_second: float | None = None
    output_tokens_per_second: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_peak_gb: float | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    quality_score: float | None = None
    quality_method: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    token_count_method: str = "char_approx_4"
    missing_metrics: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MockMetrics(BaseModel):
    """Mock server metrics snapshot."""

    total_requests: int = 0
    streaming_requests: int = 0
    non_streaming_requests: int = 0
    error_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    warmed_prefixes: int = 0

